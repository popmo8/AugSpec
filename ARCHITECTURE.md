# aug_spec 系統架構與模組導覽（2026-07-11）

> 這份文件回答兩個問題：**整個系統怎麼運作**、**要加或改某個功能時該去哪裡動**。
> 讀者假設：懂 speculative decoding 與 MoE，但第一次接觸這個 repo。
> 環境、跑實驗、旋鈕全表請看 `PROJECT_GUIDE.md` 與 `configs/README.md`；本文件只講程式結構。
> 姊妹篇：`CODE_HEALTH_REVIEW.md`（本次架構檢討的發現）、`ARCHITECTURE_REVIEW_AND_PAPER_PLAN.md`（論文可重現性 roadmap）。

---

## 0. 一頁總覽

這個 repo 做的事：**用「合併後的 expert」當 speculative decoding 的 draft model**，
在 expert offloading（VRAM 不足以放下所有 expert）的場景下跑 Spec-Bench 評測。
target 與 draft 是**同一個模型物件**（共權重）——差別只在 MoE block 的 forward
在「draft 相位」走合併/替代路徑、在「verify 相位」走真實 routing。

```
┌─ 實驗定義 ─────────────────────────────────────────────────────────┐
│  configs/*.yaml ──→ cli.py（RunConfig.from_yaml → run_experiment）  │
└──────────────┬──────────────────────────────────────────────────────┘
               │ 組裝
┌─ 策略層（純 Python，可插拔）─────────────────────────────────────────┐
│  drafts/    draft 策略 registry（topm_count＝我方、specmoe＝baseline）│
│  clustering/ ClusterMethod registry（freq_slice / hybrid / …）        │
│  merging/   linear_merge（唯一合併入口，刻意非 registry）             │
│  kernels/   bmm.py（SwiGLU 批次 bmm 純 kernel）                      │
└──────────────┬──────────────────────────────────────────────────────┘
               │ Controller 接線（controller.py）
┌─ 模型接線層 ─────────────────────────────────────────────────────────┐
│  adapters/  模型家族 registry（qwen3 / mixtral / gptoss）             │
│             把 MoE block 的 forward 換成「相位分歧」版本              │
│  runtime/phase.py + specbench.py  HF assisted-decoding 的 monkey-patch│
└──────────────┬──────────────────────────────────────────────────────┘
               │ backend=offload 時
┌─ Offload 執行層 ─────────────────────────────────────────────────────┐
│  runtime/loader.py        載入 + VRAM 預算計算                        │
│  runtime/offload_merge.py OffloadMergeEngine（merge 時機/相位政策）   │
│  runtime/merged_cache.py  MergedCacheIndex（merged-slot 內容索引政策）│
│  moe_infinity/（vendored fork）                                       │
│    ├ Python: MoE 包裝、hook、Qwen3MoEBlock                            │
│    └ C++:   archer 引擎（fetch/cache/evict/pin/merge/bmm）            │
└──────────────────────────────────────────────────────────────────────┘
```

設計原則（已在多輪重構後定形，改動前先讀 `ARCHITECTURE_REVIEW_AND_PAPER_PLAN.md` §1）：

- **一個 YAML＝一個實驗**：加實驗加 YAML，不加 Python 檔。
- **三個 registry**（drafts / clustering / adapters）＋**一個非 registry**（merging 永遠線性）。
- **policy 在 Python、mechanism 在 C++**：C++ 只提供 slot/pin/fetch/merge 的機制 API，
  「何時 merge、pin 誰、驅逐誰」全部在 Python 決定。
- **YAML 為主、env 為 override**；純診斷開關（`AUG_PROFILE`、`AUG_DUMP_*`）刻意不進 YAML。

---

## 1. 端到端資料流：一個實驗的生命週期

入口：`python -m aug_spec.cli run --config configs/X.yaml`

### 1.1 啟動與組裝（`cli.py:run_experiment`）

1. `RunConfig.from_yaml` 解析 YAML（`cli.py:143`）。
2. `apply_offload_settings`（`adapters/base.py:46`）把 YAML 的 `merged_backend` / `early_pin`
   寫進 module 全域（env 已設定則 env 贏）。
3. **載入模型**：
   - `backend: hf` → `loader.load_model`（單純 HF 載入）。
   - `backend: offload` → 先算 VRAM 預算（見 §3.4），再 `loader.load_offload` →
     `moe_infinity.MoE(...)` 包裝 + 一份 CPU 權重副本 `cpu_source`（draft 合併的權重來源）。
4. 選 adapter（`adapters/adapter_for_config` 依 `config.model_type` 自動判斷）。
5. 建 draft（`drafts/get_draft`），若是 `ScoreBasedAvgDraft` 家族再注入 cluster method 與
   within-weight 設定。
6. 建 `Controller`（把 adapter × draft 接上模型；offload+merge 時再建 `OffloadMergeEngine`）。
7. `draft.prepare(...)`：一次性預計算（如 SpecMoE 的 pairwise L2 距離矩陣）。
8. `controller.install()`：把每個 MoE block 的 `forward` 換成 adapter 產生的相位分歧版本。
9. 進 `shared_model_phase_patch` context → `run_specbench(...)` 開跑。
10. 跑完寫 `summary.json`，`AUG_PROFILE=1` 時印 per-cycle profiling（`_dump_profile`），
    offload 模式最後 `os._exit(0)`（C++ thread pool 會卡住正常 shutdown 的 workaround）。

### 1.2 每題（question）的流程（`runtime/specbench.py:run_specbench`）

對每題 Spec-Bench 問題：

1. `on_question_start` → `controller.reset()`：清 draft 狀態、清 merged cache index、
   重新武裝 activation-capture（hybrid 用）。
2. `before_generate` → `moe._configure_hook(input_ids)`（offload 才有；
   moe_infinity 每次 generate 前要重建 expert-tracer sequence）。
3. `target_model.generate(..., assistant_model=同一個 model)` 進 HF assisted decoding，
   外面包著兩層 monkey-patch（見 §3.1）。

### 1.3 每個投機週期（cycle）內部

以預設 `run.prefill_warmup: true`（C-BOOT）為例：

```
第 0 輪（warmup round，只有每題第一次）:
  patched_get 回傳 0 個 candidate → HF 拿 target 對 prompt 做純 prefill
    └ 每層 MoE forward（verify 相位）: draft.capture() 記 routing 統計
      └ offload: dispatch_local 抓 expert → on_verify_layer → 逐層把 merged 建好
  update_candidate_strategy 尾端: on_prefill_warmup → controller.update_masks()
  （hf backend 在這裡才真正合併；offload 在 on_verify_layer 已建好，refresh 跳過）

之後每輪 cycle:
  ┌ draft 相位（in_draft_phase=True，由 phase patch 設定）
  │   第一次真 draft 前: 把 target 的 prompt KV 深拷貝給 assistant（KV-copy）
  │   assistant 連續 T 步 forward:
  │     merged 家族 → draft_cache[li]（"multi" dict）→ _route_multi_expert
  │       └ engine_bmm: C++ DispatchBmm ／ dispatch: 逐 expert ／ bmm: torch.bmm
  │     specmoe → draft_cache[li]（substitute table）→ 替代路由 → dispatch/bmm
  └ verify 相位（in_draft_phase=False）
      target 一次 forward 驗 T 個 token:
        每層: draft.capture(router_logits) 先記分 → dispatch_local 抓 expert 算真輸出
              → on_verify_layer: (需要時)累積 activation-sim、當層 merge（P3）
      HF 算 num_matches → patched_upd 收 CycleStats → on_cycle:
        controller.update_masks() → draft.refresh(...)   ← 重建 draft 狀態
        （merge_during_verify=true 時 refresh 只做 telemetry，merged 已在 verify 中建好）
```

### 1.4 輸出

`output/<dir>/per_question_summary.csv`、`overall_summary.csv`、`summary.json`、
（可選）`tokens.csv`、`expert_weights_history.json`。欄位意義見 `PROJECT_GUIDE.md`「讀結果」。

---

## 2. 模組職責與邊界

### 2.1 `cli.py`（~700 行）— 組裝根

- `RunConfig`：YAML schema 的唯一定義處。**加 YAML 欄位就改這裡**（dataclass 欄位
  + `from_yaml` 解析 + `configs/README.md` 文件）。
- `run_experiment`：組裝 + 跑 + 寫檔的一條龍。VRAM 預算計算（cache-mode 判定、
  slot carve、singleton pin budget）也在這裡（`cli.py:346-429`）。
- `_dump_profile`：AUG_PROFILE 的文字報表。**加 profiling 欄位**：C++ 加 counter →
  `dump_profile()` dict → 這裡加一行 `row(...)`。

### 2.2 `controller.py`（151 行）— adapter × draft 接線

唯一職責：把 `(adapter, draft)` 裝上模型。

- `install()/uninstall()`：依 `draft.cache_kind`（averaged / masked / substitute）
  選 adapter 的 forward factory，換掉每個 MoE block 的 `forward`。
- `draft_cache: Dict[layer_idx, Any]`：draft 相位用的每層狀態（merged dict、mask、
  或 substitute table）。**內容格式由 draft 決定，adapter 消費**。
- `update_masks()`（per-cycle）→ `draft.refresh`；`reset()`（per-question）→
  `draft.reset` + merge engine 的 `on_question_start`。
- offload 時把 `cpu_source` 的對應 block、`_merge_offload` 旗標掛到每個 block 上
  （attribute 注入，見 §3.5）。

### 2.3 `adapters/` — 模型家族層

一個 adapter 封裝「這個模型家族的 MoE 長什麼樣」：

| 檔案 | 內容 |
|---|---|
| `base.py` | `MoEAdapter` ABC + 共用的 `_route_multi_expert`（K 顆 merged 的 gate-remap 路由，含 engine_bmm / dispatch / bmm 三後端）+ module 全域 `_MERGED_BACKEND`/`_EARLY_PIN` |
| `qwen3.py` | 主力。block 定位、SwiGLU 合併（GPU 經 dispatcher `merge_experts_local` 或 CPU fallback）、offload 的 `_route_offload`（verify 用 `dispatch_local`）、`_dispatch_selected`（SpecMoE 用）、三種 forward factory |
| `mixtral.py` | 同構於 qwen3（hf-only 路徑較完整；offload 未接） |
| `gptoss.py` | fused-tensor 版本；`make_masked_forward` 未實作 |
| `__init__.py` | registry + `config.model_type` 自動對應 |

**加新模型家族**：抄 `qwen3.py` 實作 `MoEAdapter` 介面 → 在 `__init__.py`
的 `_REGISTRY` 與 `_MODEL_TYPE_MAP` 各加一行。若要跑 offload，還需要
moe_infinity 端有對應的 offload block（見 §2.8）。

### 2.4 `drafts/` — draft 策略層

`DraftStrategy` 的核心 contract（`drafts/base.py:32`）：

- `cache_kind`：決定 controller 裝哪種 forward——`averaged`（merged expert dict）、
  `masked`（bool mask）、`substitute`（替代表）。
- 類別層級旗標：`holds_merged_residency`（要不要從 VRAM 預算保留 merged 空間）、
  `needs_count_top_k` / `needs_num_experts`（CLI 自動補參數用）。**新 draft 自己宣告
  性質，cli.py 不需要 hardcode 名單**。
- 生命週期 hook：`prepare`（一次）→ 每題 `reset`/`prepopulate` → 每個 target forward
  `capture(layer_idx, router_logits)` → 每 cycle `refresh(adapter, blocks, draft_cache)`
  → draft forward 可 `lazy_build`。

家族樹：

```
DraftStrategy
├─ ScoreBasedAvgDraft（drafts/base.py，"averaged"；分數→分群→合併的共用機器）
│   ├─ CountDraft（count；PrunedCountDraft）
│   │   ├─ TopMCountDraft（topm_count ＝我方主方法，top-M 截斷）
│   │   └─ PrefillCountDraft（prefill 凍結）── PrefillTopMCountDraft
│   └─ SoftmaxDraft
├─ UniformDraft（1/n，lazy 一次）
├─ RandomMaskDraft（"masked" baseline）
└─ SpecMoeDraft（drafts/specmoe.py，"substitute" ＝ baseline；
                 kept-N mask + L2 最近替代表 + early_pin + kept_bmm）
```

`ScoreBasedAvgDraft` 是 merged 家族的心臟：`capture` 存每層分數向量（並依 cluster
method 需要累積 co-occurrence / activation-sim）；`refresh` → `_refresh_layer` →
K>1 時 `_cluster_and_build`（呼叫 `cluster_method.assign`，每群 `_build_one` →
`linear_merge`），cache-mode 時改走 `merged_cache.build_layer`。

**加新 draft**：在 `drafts/` 放一個 subclass、`drafts/__init__.py` `_REGISTRY` 加一行。
只要換「分數怎麼算」，繼承 `ScoreBasedAvgDraft` 覆寫 `_score_vector_from_logits`；
只要換「權重後處理」，覆寫 `_postprocess_weights`（`topm_count` 就只有這樣）。

`drafts/specmoe.py` 同時放了 SpecMoE 的 forward（`topk_substitute_forward`）、
engine-bmm 路徑（`specmoe_engine_bmm`）與 `pairwise_l2`——這是 draft 邏輯，
adapter 只是 lazy-import 它（避免 adapters↔drafts 循環 import）。

### 2.5 `clustering/` — 分群層

`ClusterMethod.assign(ctx, K) -> List[List[int]]`，輸入 `ClusterContext`
（active、weights、cooccur、pair_sim）。方法宣告自己需要哪些統計
（`needs_cooccur` / `needs_activation_sim` / `act_sim_prefill_only`），
draft 與 merge engine 據此才去累積——**不需要的統計零成本**。

| name | 檔案 | 訊號 | 備註 |
|---|---|---|---|
| `freq_slice` | freq_slice.py | 頻率排序切片 | 預設；大群 |
| `random` | random.py | 隨機 | 對照組 |
| `cooccur_pair` | cooccur.py | 共現 | greedy pair（≤2 人群）|
| `activation_similarity` | activation_sim.py | 輸出 cosine/L2 | 需 C++ 輸出捕捉 |
| `weight_similarity` | weight_sim.py | 權重相似度 | 靜態，prepare 一次可 cache |
| `hybrid` | hybrid.py | α×act-sim + (1−α)×co-occur | 論文的 Relation Map；λ＝`cluster.alpha` |

共用的 `greedy_pair`（最大值貪婪配對、群 ≤2）在 `clustering/base.py:53`。

**加新分群法**：subclass + registry 一行；若需要新統計訊號，要同時動
`ClusterContext`（base.py）、`ScoreBasedAvgDraft` 的累積邏輯（drafts/base.py）、
必要時 `OffloadMergeEngine` 的 capture 開關（見 §3.5 的旗標協定）。

### 2.6 `merging/` + `kernels/`

- `merging/linear.py:linear_merge`：**唯一**合併入口。有 merge engine 時走
  engine（→ GPU dispatcher merge），否則 adapter 的 CPU 合併。刻意不做 registry
  ——合併永遠線性，變的只有權重（來自 clustering / within_weight）。
- `kernels/bmm.py`：`stack_swiglu_weights`（K 顆權重堆疊、memoise 在 per-cycle
  cache dict 上）+ `bmm_swiglu`。純 tensor 函式，無 adapter/draft 知識。

### 2.7 `runtime/` — 評測與 offload 政策

| 檔案 | 職責 |
|---|---|
| `specbench.py`（768 行） | Spec-Bench driver：載題、`_locked_assist_patch`（鎖 T、收 CycleStats、C-BOOT 空首輪、KV-copy）、per-question 迴圈、CSV/聚合輸出、VRAM guard 取樣 |
| `phase.py` | `shared_model_phase_patch`（draft 相位旗標翻轉 + merge engine 的 on_draft_start/end + profiling 相位標記）、`specbench_callbacks` |
| `loader.py` | hf/offload 兩種載入、`compute_model_vram_bytes`/`compute_expert_geometry`/`compute_merged_bytes`（meta-device 算幾何，不載權重）、VRAM 量測工具 |
| `offload_merge.py` | `OffloadMergeEngine`：merge 的「時機政策」——P3 verify 中逐層 merge（`on_verify_layer`）、P1 相位互斥 flush、P4 side-stream overlap、activation-capture 開關、C1 cache 的 setup 與 reset |
| `merged_cache.py` | `MergedCacheIndex`：C1/C2 的「內容政策」——content addressing（`frozenset(members)→slot`）、adopt-first 兩段式分群、slot 竊取的 retention 規則、singleton 雙重身分 pin、telemetry |
| `scorers.py` | softmax→重要度向量的純函式（count / cooccur / softmax / hybrid scorer）|

### 2.8 `moe_infinity/`（vendored fork）

Python 端（`moe_infinity/moe_infinity/`）的載入流程（`entrypoints/big_modeling.py:MoE`
→ `runtime/model_offload.py:OffloadEngine`）：

1. `MoE(model_id, {offload_path, device_memory_ratio})` 依 `config.architectures`
   選定 HF 模型類別（Qwen3 → 原生 `Qwen3MoeForCausalLM`）。
2. `OffloadEngine.init(...)` 是一個 context manager，在裡面 monkey-patch 掉
   `from_pretrained`：所有參數建成 shape-(1,) 的 CPU placeholder、HF 的
   `Qwen3MoeSparseMoeBlock` 換成 `models/qwen.py:Qwen3MoEBlock`（offload 版）。
3. 第一次跑會做 checkpoint 轉換：每個 tensor 給一個整數 id、`archer_engine.offload`
   進 C++ store（host 常駐），對照表存 `offload_path/name_id_map.json`；之後直接重載。
4. 每個模組裝 forward pre/post hook：非 expert 權重經 `archer_engine.begin/end`
   即時 materialize/釋放；expert 則註冊進 `expert_dispatcher`，由 verify 時的
   `dispatch_local` 按需抓進 GPU cache。
5. 每個 MoE block 被注入 `expert_executor`、`expert_prefetcher`、`layer_id`、
   `expert_tensor_map` 等 handle——aug_spec 的 adapter/draft 讀的就是這些屬性。

`moe._configure_hook` 每次 generate 前必呼（specbench 的 `before_generate` 已接）。
注意：在 aug_spec 下，`Qwen3MoEBlock.forward` 本體**不會被執行**——controller 會把
它換成 adapter 的相位分歧 forward（只沿用注入的 handle 與 `gate`/`experts` 結構）。

C++ 端（`moe_infinity/core/`）：archer 引擎。對 aug_spec 最重要的是
`core/parallel/expert_dispatcher.cpp`（fetch/cache/evict/pin/merge/bmm/merged-slot
全在這）與 `core/python/py_archer_prefetch.cpp`（pybind 介面）。
詳細架構見 §5；**改 C++ 後要重編**：`cd moe_infinity && <venv> setup.py build_ext --inplace`
（照規範用 sbatch 跑，不要在 login node 編）。

aug_spec 對 C++ 的整合面**全部走 `expert_dispatcher` 的 pybind API**（Python 端以
`block.expert_executor.expert_dispatcher` 取得），目前用到：

| API | 用途 | 呼叫處 |
|---|---|---|
| `dispatch_local` / `wait_dispatch_local` | verify/specmoe 的 expert 執行 | `adapters/qwen3.py:158,188` |
| `dispatch_bmm` | 批次 bmm draft kernel（雙方法共用） | `adapters/base.py:143`、`drafts/specmoe.py:289` |
| `dispatch_merged_local` | K 顆 merged 逐顆跑（A/B 用） | `adapters/base.py:153` |
| `merge_experts_local` | GPU 上合併（legacy 路徑） | `adapters/qwen3.py:67` |
| `init_merged_slots` / `merge_experts_to_slot` / `get_merged_slot` / `set_merged_slot_pinned` | C1 merged-slot 機制 | `offload_merge.py:102`、`merged_cache.py` |
| `set_pinned` / `clear_pinned` | expert pin（specmoe kept-N、singleton 雙重身分） | `drafts/specmoe.py`、`merged_cache.py` |
| `get_resident_expert_weights` | 讀 resident 權重（specmoe kept bmm 堆疊） | `drafts/specmoe.py:228` |
| `flush_cache` | P1 相位互斥 flush | `offload_merge.py:257` |
| `set_capture_expert_out` / `get_captured_expert_outputs` | activation-sim 的輸出捕捉 | `offload_merge.py`、`drafts/base.py:308` |
| `set_profile_phase` / `dump_profile` | AUG_PROFILE | `phase.py`、`cli.py` |

---

## 3. 關鍵機制深入

### 3.1 相位切換：三層 monkey-patch 的疊法

target 與 draft 共用一個 model，靠 `controller.in_draft_phase` 讓 forward 分歧。
這個旗標由對 HF `AssistedCandidateGenerator` 的 patch 控制，共兩處、外加一個
forward wrapper，**安裝順序有語義**：`shared_model_phase_patch` 先裝
（cli.py:510 的 with），`_locked_assist_patch` 在 run_specbench 內後裝——
後裝者包在外層，所以 T-lock/C-BOOT 邏輯先於相位翻轉執行：

```
generate()
 └ locked_get（specbench.py:217）      ── 鎖 T、C-BOOT 空首輪、KV-copy 注入
    └ phase_get（phase.py:65）          ── in_draft_phase=True；merge engine on_draft_start
       └ 原始 get_candidates            ── 真正的 draft forwards
    （結束）in_draft_phase=False；on_draft_end
 └ target verify forward（stash_forward wrapper 順手存 target KV 供 KV-copy）
 └ locked_upd（specbench.py:284）       ── 收 CycleStats → on_verify → on_cycle → refresh
```

C-BOOT 空首輪刻意發生在 phase patch **外層**，所以 warmup prefill 不會翻相位旗標、
hybrid 的 prefill capture 不會被關掉——改這段順序前務必理解這點。

### 3.2 兩條 draft 執行路徑

**merged（topm_count 家族）**：`draft_cache[li]` 是 `{"kind":"multi", experts, weights,
indices}`。draft forward → `adapters/base.py:_route_multi_expert`：token 的 gate 質量
remap 到 K 群 → 選 top-k 群 → 三種 kernel 後端之一執行。K=1 時是單顆 dense dict，
直接 `_run_dense_expert`。

**substitute（specmoe）**：`draft_cache[li]` 是 `[num_experts]` 的替代表。
draft forward（`drafts/specmoe.py:topk_substitute_forward`）照常 top-k，
把 winner 經替代表 remap 後 dispatch；kept-N 全 resident 且 backend=engine_bmm 時
改走 `specmoe_engine_bmm`（與 merged 同一顆 C++ bmm kernel——公平比較的關鍵）。

### 3.3 C1 merged cache（cache mode）

啟動條件在 `cli.py:370`：offload + `merge_offload` + draft `holds_merged_residency`
+ K>1 + within_weight=uniform + 未設 `AUG_LEGACY_MERGE`。開了之後：

- **預算**：merged slot 從 usable VRAM 裡 carve 出來（torch 記憶體），archer pool 拿剩下的
  ——**兩本帳嚴格分離**（slot bytes 不進 archer 帳，歷史教訓見 PROJECT_GUIDE 結論 2026-07-10）。
- **build_layer**（`merged_cache.py:143`）取代 `_cluster_and_build`：
  adopt（content key 命中、成員全 active → 0 fetch 0 merge）→ greedy 分剩下的 →
  singleton 先 probe 再配 identity slot、pair 走 `merge_experts_to_slot`。
  slot 滿時 `_alloc_slot` 按 retention 規則竊取（本輪不偷、未保護先偷、singleton 先偷、
  低 co-occur 先偷、LRU tiebreak）。
- **singleton 雙重身分**：draft 一律從自有 slot buffer 讀（不持有 archer 管理的 tensor
  ——prefetcher 會原地搬走它）；原始 expert 另外 best-effort pin 住，讓 verify 免重抓。

### 3.4 VRAM 預算計算（`cli.py:346-429`）

`vram_budget_ratio`（論文 0.2）× 全模型 footprint（meta device 算出）＝ usable。
cache mode：先扣 verify floor（2×48×expert_bytes 永不可 pin），算出 K′ slot carve，
剩下給 archer pool，再導出 `device_memory_ratio` 給 moe_infinity。
legacy：merged 保留量 `compute_merged_bytes` 直接從 usable 扣。
`vram_guard` 只稽核（每 cycle 取樣 driver-level 用量、超標印警告），不強制。

### 3.5 Attribute 注入接線一覽（隱式耦合，改名前全域 grep）

模組之間不少接線是「把屬性掛在別人的物件上」，集中列出：

| 屬性 | 掛在誰身上 | 誰寫 | 誰讀 |
|---|---|---|---|
| `_aug_spec_orig_forward` | MoE block | controller.install | controller.uninstall |
| `_cpu_merge_source` | MoE block | controller.__init__ | adapter 合併、specmoe/weight_sim prepare |
| `_merge_device` | MoE block | controller.__init__ | adapter 合併 |
| `_merge_offload` | MoE block | controller.__init__ | `qwen3.build_weighted_avg` |
| `_merge_engine` | MoE block | engine.attach | `linear_merge`、`_refresh_layer`、`_route_offload` |
| `engine.controller` | merge engine | cli.py | `on_verify_layer`（拿 draft、draft_cache） |
| `expert_executor` / `expert_dispatcher` | offload block | moe_infinity | adapters、drafts、phase、cli（profiling） |
| `block.layer_id` / `top_k` / `num_experts` / `norm_topk_prob` | offload block | moe_infinity | adapters |

cluster method 與引擎之間的**旗標協定**：`needs_cooccur`、`needs_activation_sim`、
`act_sim_prefill_only`、`cooccur_scope`、`metric` ——由 `ScoreBasedAvgDraft`
與 `OffloadMergeEngine` 用 `getattr` 讀。加新旗標時記得兩邊都要接。

### 3.6 旋鈕與 env override

YAML 欄位全表在 `configs/README.md`。env 只有三類：
(1) YAML 旋鈕的 runtime override（`AUG_MERGED_BACKEND`、`AUG_EARLY_PIN`、
`AUG_CLUSTER_UNIFORM`）；(2) 過渡逃生口（`AUG_LEGACY_MERGE`）；
(3) 純診斷（`AUG_PROFILE`、`AUG_DUMP_ACTIVE_SET`、`AUG_DUMP_PAIRS`、
`AUG_DUMP_CYCLE_SIM`、`AUG_DUMP_CLUSTER_WEIGHTS`、`AUG_HANG_DEBUG`）。

---

## 4. 「我要改 X，去哪裡」對照表

| 想做的事 | 去哪裡 | 備註 |
|---|---|---|
| 加一個實驗 | `configs/` 加 YAML（抄 `q5_512_tm_on.yaml`）+ `scripts/` 抄一個 sbatch | 不用動 Python |
| 加 YAML 欄位 | `cli.py:RunConfig`（欄位+解析）→ 用的地方 → `configs/README.md` | |
| 加 draft 策略 | `drafts/` 新檔 + `drafts/__init__.py` registry | 分數變體只需覆寫 1–2 個 method（§2.4） |
| 加分群方法 | `clustering/` 新檔 + registry | 需要新訊號時見 §2.5 |
| 改合併權重邏輯 | `ScoreBasedAvgDraft._cluster_and_build` / `_postprocess_weights` | cache mode 下對應 `merged_cache.build_layer` 也要看 |
| 改 merge 執行（kernel 層） | `merging/linear.py` → `adapter.build_weighted_avg` → C++ `merge_experts_local`/`merge_experts_to_slot` | |
| 改 draft 的執行 kernel | `adapters/base.py:_route_multi_expert`（merged）、`drafts/specmoe.py`（substitute）、`kernels/bmm.py`、C++ `DispatchBmm` | |
| 加新模型家族 | `adapters/` 新檔 + registry；offload 還要 moe_infinity 端 block | §2.3 |
| 改 merged cache / pin / slot 政策 | `runtime/merged_cache.py`（政策）；機制在 C++ | **先讀 `merged_cache_plan.md`** |
| 改 merge 時機/相位行為 | `runtime/offload_merge.py` | P1/P3/P4 的家 |
| 改 VRAM 預算規則 | `cli.py` 預算區塊 + `loader.py` 的 compute_* | |
| 改 Spec-Bench 評測行為（T、warmup、KV-copy、輸出欄位） | `runtime/specbench.py` | monkey-patch 疊層見 §3.1 |
| 加 per-cycle 遙測 | draft 的 `capture`/`refresh` 或 `specbench_callbacks` 的 `on_cycle_extra` | |
| 加 profiling 計數器 | C++ counter → `dump_profile()` → `cli.py:_dump_profile` 加 row | 解析注意 label 全等比對（PROJECT_GUIDE） |
| 改 fetch/evict/pin 機制 | C++ `expert_dispatcher.cpp` | 改完重編；**先讀 `merged_cache_plan.md`、`deprecated_docs/` 的 remove_overload_plan** |
| 動 offload 載入/預算換算 | `runtime/loader.py:load_offload` + moe_infinity `MoE` | |

---

## 5. C++ 引擎（archer）內部架構

> 位置：`moe_infinity/core/`。改完要重編（sbatch 跑 `setup.py build_ext --inplace`）。
> 這是 upstream MoE-Infinity 的重改版；aug_spec 加的機制（merged slot、pin-aware
> evict、DispatchBmm、profiling、輸出捕捉）全部集中在 `ExpertDispatcher`。

### 5.1 兩個並存的子系統

C++ 端其實是**兩套半獨立系統**共用同一批 `Node`（tensor 的 placement 記錄）：

1. **archer prefetch/streaming**（`ArcherPrefetchHandle` + `ArcherTaskPool` +
   `ArcherTopologyHandle`）——服務**非 expert 權重**：模型 forward 的 pre/post hook
   走 `begin()/end()`（AcquireTensor/ReleaseTensor），task pool 有自己的搬運 thread
   與 LRU 驅逐（`RemoveCachedDenseNode/SparseNode`，`prefetch/task_scheduler.cpp`）。
2. **`ExpertDispatcher`**（`parallel/expert_dispatcher.cpp`，aug_spec 的主戰場）——
   擁有 **sparse expert 的 GPU cache** 與自己的帳本（`cache_sizes_`），做 verify 的
   expert fetch+forward，以及所有 aug_spec 擴充。

兩套系統的驅逐決策彼此看不見（task pool 搬走 Node 不會通知 dispatcher 的帳本），
所以 `FindExpertEvict` 內建「殭屍 key 收屍」邏輯（發現 Node 已不在 GPU 就還帳）
——這是結構性耦合的補丁，動任何一邊的驅逐邏輯前要先理解（PROJECT_GUIDE
2026-07-10 結論的根源）。

### 5.2 執行緒與佇列

- 每 GPU 一條 **fetch thread**（`GPUFetchFunc`）＋ N 條 **exec thread**
  （`GPUExecFunc`，各自有 non-blocking CUDA stream）。
- `Enqueue`：cache 命中（已 resident）直接進 `exec_queue_`（省掉 fetch）；
  未命中進 `input_queue_` 由 fetch thread 處理（必要時先 `FindExpertEvict` 騰位、
  再 `Node::SetDevice` 做 host→GPU 搬運）。
- `OutputFunc` 把每顆 expert 的輸出加權 `index_add_` 進 `final_hidden_states_`，
  `pending_` 歸零時喚醒 Python 端阻塞中的 `wait_expert`（=`WaitHiddenStates`）。

### 5.3 三條資料流

- **verify fetch/forward**：Python 算好 routing → `set_inputs` → 逐 expert
  `enqueue_expert` → `notify_fetch_start` → fetch/exec threads → `wait_expert`。
  實際 GEMM 走 `MoEMLP::forward`（`parallel/expert_module.cpp`，CUTLASS fused kernel）。
- **merged 路徑**：合併的來源讀取在 `MergeAccumulate`（resident 就地讀 0 PCIe、
  冷的暫時搬上來用完即丟，**不動 cache**）。持久化 merged slot（C0/C1）
  是 torch-allocator 的 GPU buffer，**刻意不進 archer 帳本**；
  `MergeExpertsToSlot`/`GetMergedSlot`/`SetMergedSlotPinned` 是機制 API，
  政策全在 Python 的 `MergedCacheIndex`。`DispatchBmm` 收 Python 預先堆疊好的
  `[E,D,I]` 權重直接跑 3 個 `torch::bmm`——merged 與 SpecMoE 兩種 draft 共用這顆。
- **驅逐與 pin**：`FindExpertEvict` 是唯一驅逐路徑（overload 路徑 2026-07-08 已刪）
  ——跳過 pinned、收殭屍 key、選 `incache_visit_count` 最小者，**回傳時 victim
  仍上鎖**（呼叫端 SetDevice 完才解鎖，防 select/evict race）。`SetPinned` 整層替換
  pin 集合；merged slot 的 pin 獨立（`MergedSlot::pinned`）。全 pinned 餓死時有
  分級守門：2ms timed retry（防 lost wakeup）→ 10s 印 forensic → 60s／全 pinned
  且 pipeline 排空 10s → FATAL。`FlushCache` 是 EvictLayer 的後繼（P1 相位互斥用）。

### 5.4 不可誤刪的 invariants（重構前必讀）

1. **merged slot 永遠不碰 `cache_sizes_` 帳本**——兩本帳合一會造成幽靈扣款活鎖
   （2026-07-10 踩過，header 內有大段註解）。
2. **`FindExpertEvict` 回傳 locked victim**，呼叫端搬完才解鎖——改成回傳後再鎖
   會重新引入 race。
3. **fetch thread 用 `wait_for(2ms)` timed retry 而非純 condition wait**——
   防 lost wakeup，不要「優化」回去。
4. `Enqueue` 的 spin-lock（~10s 上限）是對 fetch/evict 中途 `SetDevice` 的防護。

pybind 介面全表見 §2.8；`prefetch_handle` 那側 aug_spec 只間接使用（經 moe_infinity
Python 的 hook），直接呼叫的都在 `expert_dispatcher` 上。

---

## 6. 測試與驗證

- `tests/unit/`：pytest、純 CPU、login node 可跑（`.venv/bin/python -m pytest tests/unit -q`）。
  目前覆蓋 hybrid clustering 與 merged cache 政策。**改 clustering/drafts/merging/
  merged_cache 前後都要跑**。
- `tests/offload/`：歷史手動 probe（m1–m9、c0–c2），不是自動化測試，僅供考古。
- GPU 行為驗證一律 sbatch smoke config（`configs/smoke_*.yaml`、`c1_smoke.yaml`）。
- offload 推論**非確定性**：不能 bit-exact 比對；pp 級差異要 ≥3 重複（PROJECT_GUIDE 結論）。

## 7. 相關文件索引

| 文件 | 讀的時機 |
|---|---|
| `PROJECT_GUIDE.md` | 每個新 session 先讀：環境、跑法、關鍵結論 |
| `configs/README.md` | 寫/改 YAML |
| `merged_cache_plan.md` | 動 merged cache / pin / dispatcher |
| `hybrid_cluster_plan.md` | 動 hybrid 分群 |
| `verify_merge_plan.md` | P1–P4 merge-during-verify 的設計依據 |
| `refactor_and_cooccur_plan.md` | A1–A5 重構歷史 + B 系列 backlog |
| `reconstruct_fetch_plan.md` | 代數重建（未來工作） |
| `ARCHITECTURE_REVIEW_AND_PAPER_PLAN.md` | 論文/repro/打包工作 |
| `CODE_HEALTH_REVIEW.md` | 本次程式碼檢討的發現與建議 |
