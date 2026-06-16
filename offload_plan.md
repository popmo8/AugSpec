# `offload_plan.md` — moe_infinity backend 漸進式落地計畫

[next_step.md](next_step.md) 回答「**架構長什麼樣**」（backend 放 loader、
adapter 只分流 routing、merge 來源用 CPU copy …）。本文件回答
「**按什麼順序做才好 debug**」—— 把原本 next_step.md 的 Phase 1–3
（一次接通 config → loader → adapter → E2E）重切成更小的階梯，
每階只動一個檔案、有獨立驗收標準、紅了就停在當階修。

---

## 0. 三條鐵律

1. **先 script、後整合。** 所有跟 moe_infinity 的互動，先在
   `tests/offload/` 下的獨立 scratch script 證明可行，才允許搬進
   `src/aug_spec/`。script 炸了只要 debug 50 行；整合後炸了要 debug
   整條 spec-bench 管線。
2. **每個 milestone 一個 commit，且既有 GPU 路徑必須保持綠。**
   每階收尾都重跑 `aug_spec run --config configs/_smoke.yaml`，
   數字（MAT / AccRate）必須與改動前一致 —— 把改動前的
   `output/_smoke/summary.json` 留一份當 golden。
3. **E2E 之前不碰 config schema。** `RunConfig` 加 `backend` key
   是 M6 的事；M1–M5 全部用 script 直接呼叫，避免「config 解析 +
   loader + adapter + 管線」四種錯誤攪在一起。

Smoke 規格統一：`questions_per_cat=1`、`max_new_tokens=32`、
`seed=0`，幾分鐘內可重跑。

---

## 1. 終點圖像與 0.2x 預算規則

（2026-06-12 討論收斂。架構細節仍見 next_step.md，但**預算帳以本節為準** ——
next_step.md §2.7「merged expert 在 `device_memory_ratio` 預算之外、暫存不設限」
的前提已被本節的嚴格帳取代。）

**符號**（draft = 把每層 E 顆 expert 壓成 K 顆 merged expert）：
- **E** = 每層原始 expert 數（Mixtral 8、Qwen3 128）；**L** = 層數（Mixtral 32、
  Qwen3 48）；**expert_size**（Mixtral 352 MB、Qwen3 9.44 MB）。
- **r = 壓縮倍數**；**K = E / r** = 每層保留的 merged expert 數（= draft residency）。
- **本文主線選 r = 8**（保留 1/8 expert）→ **Mixtral K = 1、Qwen3 K = 16**。residency
  VRAM = K·expert_size·L（Mixtral 11.3 GB、Qwen3 7.25 GB）。K=1 是論文標題「單顆 merged
  expert」的極限；Qwen3 routing 太細、K=1 崩 → 放寬到 K=16。

### 1.1 終點圖像（M7 完成時的狀態）

| 環節 | 行為 |
|---|---|
| GPU 常駐 | 非 expert 層（attention / gate / embed）+ 每層 K 顆 merged expert + KV cache |
| target verify | on-demand：需要的 expert 不在 GPU 就 H2D fetch 進工作區、用完即被覆蓋 |
| 進 draft 前 | frequency-based merge 得到 K 顆 draft expert，寫進常駐 slot |
| draft phase | 走 dense 的 K 顆 merged expert、不呼叫 `dispatch_local` —— **零 expert 搬運、零 cache 擾動** |

### 1.2 0.2x 預算規則

定義 **x = 整顆模型 bf16 所需 VRAM**。所有被比較的系統（baseline / SpecMoE /
Ours）的常駐權重 + **兩邊同大小**的串流工作區一律 ≤ **0.2x**（0.15x 太緊；放寬到
0.2x 留一點緩衝、工作區也塞得下）。這層帳屬工程細節，論文不展開。

| 模型 | x | 0.2x | 扣非 expert | Ours residency = K·expert_size·L | 緩衝 |
|---|---|---|---|---|---|
| Mixtral-8x7B | 93.4 GB | 18.7 GB | 15.5 GB | K=1（11.3 GB）| ~3 GB |
| Qwen3-30B-A3B | 61 GB | 12.2 GB | 9.2 GB | K=16（7.25 GB）✓ | ~2 GB |

（Qwen3 experts 58 GB = 48 層 × 128 顆 × 9.44 MB、非 expert ≈ 3 GB。原本 K=16 在
0.15x 超標 1.1 GB 的待決，放寬到 0.2x 後消失。）

| 系統 | 0.2x 的用法 |
|---|---|
| MoE-OnDemand | 非 expert + archer cache |
| SpecMoE | 非 expert + 每層 N 顆 pinned expert（同預算 Mixtral 仍 **N=1**；其原生 N=4 要 0.48x，此帳下不合法）|
| Ours | 非 expert + 每層 **K 顆** merged expert（自己的 torch tensor，在 archer 之外）|

**串流工作區**：1–2 expert size（0.35–0.7 GB），兩邊同大小、明文入帳 —— 工作區一旦
變大，未被覆蓋的 expert 再次命中就是 cache，等於偷渡 VRAM 優勢。verify 的 on-demand
fetch 與 merge 暫存都只用這塊 = archer cache 調到最小（archer 的 miss→fetch→evict
就是工作區行為；pool 大小 init 設定一次、不能 runtime 縮放）。

**換算與稽核**：`device_memory_ratio` 分母是 GPU 實體容量（80 GB）不是 x，每個
config 註解寫出換算式。archer 不走 torch allocator（C++ `DeviceMemoryPool`，
[memory_pool.cpp:150](moe_infinity/core/memory/memory_pool.cpp#L150)）→ 預算用 NVML
driver 層峰值稽核（M9），驗收 peak ≤ 0.2x + 工作區。KV cache / activations 排除在外
（兩邊同 target 同 T，入帳只稀釋訊號）。

### 1.3 merge 實作與成本（Qwen3 主線 = CPU merge）

**精度基準**：offload merge 必須與 hf backend 的
[`build_weighted_avg`](src/aug_spec/adapters/mixtral.py#L40)（fp32 累加、最後捨入
一次到 bf16）**逐位元相同**，否則 M7「±0.1 對 hf backend」驗收會混進無法歸因的差異。

**裁決 = CPU merge**（host fp32 合完、只把結果搬 GPU）。成本以 K、r 表示：CPU merge
每 cycle 上傳 **K 顆/層** = K·expert_size·L；GPU merge 要拉全部 **E 顆來源** =
E·expert_size·L → **CPU/GPU PCIe 比 = K/E = 1/r**（r=8 → CPU 只搬 1/8）。贏的原理：
merge 是 memory-bound、瓶頸在 PCIe，CPU 在 DDR 內讀來源、只讓結果過 PCIe。M4（job
234890）實測 merge primitive（Qwen3、16 來源→1 顆）逐位元對齊 reference、13.6 ms、
上傳 9.4 MB，CPU 完勝。**⚠ production K=16 的 per-cycle 數字尚未實測**（上傳
K·expert_size·L = 7.25 GB、full re-merge 要讀全 E 顆）—— M7 跑的是 count（K=1），
轉 K=16 需重測 acceptance + timing。**⚠ Mixtral 不外推**（expert 352 MB、CPU 合受
記憶體頻寬限制）→ 最優留到亂碼 side-quest 修好後另測（§1 待決）。

**latency：merge 計算可藏在 verify 底下，但上傳量隨 K 變重**。CPU merge 的重活（讀
E 顆來源、加權求和）在 CPU + DDR，與 GPU verify 的計算 + PCIe fetch **資源正交** →
結構上可重疊（逐層 pipeline，counts 變化慢、甚至能用上個 cycle 的）。critical path
只剩 **K 顆上傳**：K=1 時 0.45 GB/cycle 可忽略，**K=16 時 7.25 GB/cycle（仍是 GPU-merge
的 1/r，但已非可忽略、與 verify fetch 同搶 PCIe）**。壓 per-cycle 成本主要靠 **refresh
節流**（每 k cycle 才 merge、成本 ÷k；counts 變化慢）；full re-merge 是否塞得進 verify
牆鐘由 M8 量。（舊「重疊救不了 topm」是 GPU-merge 前提：merge 本身即 PCIe 流量、與
verify fetch 搶同條；CPU merge 把 compute 移走、只剩 1/r 上傳，前提失效。）

### 1.4 verify 階段的記憶體回收（phase-exclusive memory）

**觀察**：merged expert 在 verify 階段是死的（verify 走真 expert routing、不碰
merged）→ draft 一結束即可整批 flush，釋出整包 draft residency = K·expert_size·L
（Qwen3 K=16 → **7.25 GB**）的**預算內**空間，整個 verify 都能用。peak 不變、不違反
0.2x —— 這塊大到足以在 verify 期間當大型 expert cache / prefetch buffer。

**對 SpecMoE 的結構性不對稱（賣點）**：我們常駐物在 draft / verify 的「有用占用者」
會翻面，而兩階段時間互斥 → 同一塊預算服務兩用途；SpecMoE 的 pinned expert 在 verify
當 cache、兩階段都有用，無法回收。

| | draft 佔用 | verify 那塊 slot |
|---|---|---|
| Ours | merged expert（有用）| 死的 → **可回收** |
| SpecMoE | pinned experts | 當 cache（有用）→ 不可回收 |

**拿回收空間做 prefetch buffer（時間收益待量、不預設）**：本 fork 把 moe_infinity 的
tracer 預測式 prefetch 註解掉了（[mixtral.py:71-86](moe_infinity/moe_infinity/models/mixtral.py#L71)）、現為純 on-demand。用釋出空間當 buffer 重啟
預測式 prefetch、填 verify 的串行氣泡。但收益被兩件事 gate：
- verify 若是 PCIe-bandwidth-bound（§1.3：系統瓶頸在 PCIe），預抓只重排、搬不掉
  同量位元組 → 省的僅限氣泡（≈ 每層 attention+gate compute，已被背景 thread 遮一部分），
  有閒置 PCIe 才有肉。**目前無 verify PCIe 利用率數據 → 由 M9 profile 後才能定主打。**
- 下一層 routing 在當層算完前未知 → 無法預抓「正確那幾顆」，只能預測（tracer，或拿
  draft 階段 merged-expert hidden states 跑 gate 當預測訊號）。抓全部 E=128 顆（實際
  只 route 到一小撮）= 流量爆數倍（7.25 GB 空間塞得下，純 bandwidth 浪費），排除。

**結論**：記憶體不對稱是乾淨賣點（免費、預算內、可寫進論文）；prefetch 的時間收益
不預設，待 M9 量 verify PCIe 利用率再決定要不要做、做哪種預測。

---

## 2. 已經確認的事實（不用再驗）

寫計畫前已直接讀過 vendored 原始碼確認：

| 事實 | 證據 |
|---|---|
| offload 後 block 變成 `SyncMixtralSparseMoeBlock`，`forward(hidden_states) → (final, router_logits)`，**簽名與 HF 版相同** | [mixtral.py:40-121](moe_infinity/moe_infinity/models/mixtral.py#L40) |
| verify 走 `expert_executor.dispatch_local(layer_id, hs, router_mask, routing_weights_mask)` + `wait_dispatch_local()` | [mixtral.py:87-90](moe_infinity/moe_infinity/models/mixtral.py#L87) |
| Sync forward 內部 **hardcode top-2**（`logical_or(mask[:,:,0], mask[:,:,1])`）→ 我們自己的 `_route_offload` 必須泛化到任意 top_k（Qwen3 = 8） | [mixtral.py:61-64](moe_infinity/moe_infinity/models/mixtral.py#L61) |
| offload 後 `experts[e].w1.weight` 是 shape-`(1,)` 零 placeholder，**直接讀必得垃圾** | [model_offload.py:213](moe_infinity/moe_infinity/runtime/model_offload.py#L213) |
| **連 gate / dense 權重靜止時也是 `(1,)` placeholder（在 CPU）**：moe_infinity 在**每個 module** 上遞迴掛 forward pre/post hook，前向瞬間 `archer_engine.begin/end` 物化/釋放該 module 自己的參數。→ (i) 我們替換的 `block.forward` 內呼叫 `block.gate(hs)` 走 module `__call__`，hook 照常觸發，**安全**；(ii) 直呼 `block.forward(hs)` 會跳過 hook，測試一律用 `block(hs)`；(iii) next_step.md「非 expert 常駐 GPU」的記憶體模型**不準**，VRAM 帳一律以 NVML 實測為準（M1 實測：ratio=0.15 時 generate 中 GPU used 26 GB） | [model_offload.py:962-1050](moe_infinity/moe_infinity/runtime/model_offload.py#L962)；M1 job 234581 |
| `cpu_source`（`device_map="cpu"`）經 safetensors **mmap** 載入，RSS 僅 +1.5 GB（file-backed、觸頁才進 RAM、可被回收）→ next_step.md 估的「host RAM +94 GB」**大幅高估**，merge 觸到哪層才付哪層 | M1 job 234581 Q2 |
| **`model.device` 回報 cpu**（由 placeholder 參數推得），而 HF 的 AssistedCandidateGenerator 用它搬 input_ids（`input_ids.to(assistant_model.device)`）→ assisted generation 直接 device mismatch + device-side assert 毒化 CUDA context。**Workaround：把 `type(model).device` 釘成 cuda:0 的 property** —— M5 `load_offload` 必須帶這個 patch。另：device-side assert 毒化後同進程所有 CUDA 呼叫亂報錯，探針類測試一律每檢查一個進程 | [candidate_generator.py:211](.venv/lib/python3.10/site-packages/transformers/generation/candidate_generator.py#L211)；M2 job 234621 |
| **`dispatch_local` 只在「完整逐層 forward 序列」內數值可靠，不能單層脫序測**。引擎用背景 thread 把 expert 權重搬上 GPU，`wait_dispatch_local` 不等搬完；正常 forward 的層間計算（attention 等）掩蓋了搬運延遲，**單獨抓一層、random 輸入、反覆 dispatch 則踩搬運競態 → 又錯又跳（vs 真實權重算的 ground truth 差 ~2.0、連續兩次差 ~2.3）**。`torch.cuda.synchronize` 修不了（archer 自己的 thread）。序列內證據：完整 forward 同層輸出兩次差 0.05、token 級完全確定。**後果：offload 的單元測試一律用「完整 forward / 端到端」，禁用單層脫序比對** | M6a debug jobs 234923/234931/234939/234950 |
| 每次 generate 前必須呼叫 `moe._configure_hook(input_ids)`（建 expert-tracer seq entry）—— specbench 直接呼叫 `model.generate(...)`，**不會經過 `MoE.generate`**，要自己接線 | [big_modeling.py:152-174](moe_infinity/moe_infinity/entrypoints/big_modeling.py#L152) |
| `random_mask` draft 只動 gate logits、**完全不讀 expert 權重** → 是 offload 上最安全的第一個 E2E draft | [random_mask.py](src/aug_spec/drafts/random_mask.py) |
| moe_infinity 支援 mixtral / qwen3 / deepseek / switch / nllb / arctic / grok；**GPT-OSS 不支援** | [constants.py:19-29](moe_infinity/moe_infinity/common/constants.py#L19) |
| **Mixtral 在本 fork 上輸出亂碼**（M0 job 234500 重現：無 crash、peak 88.3 GB、64-token 全亂碼；與使用者既往經驗一致。stderr 有先前追查留下的 `[DBG-MoEMLP]` 權重映射 debug 輸出 —— 嫌疑在 expert GEMM/權重順序）→ **offload 主線 = Qwen3-30B-A3B**，Mixtral 修復為獨立 side-quest | `/work/morrisliu07/job_log/m0_offload_234500.log` |
| Qwen3 offload 後 block = `Qwen3MoEBlock`，掛在 `layer.mlp`（aug_spec 的 `iter_moe` 靠 gate+experts 屬性辨識，**兩家通吃**）；`forward(hidden_states) → (final, router_logits)` 與 Mixtral 版同簽名同回傳；routing 走 fused kernel `self.lib.topk_softmax(router_logits)`（**天生任意 top_k**，無 top-2 hardcode 問題；`norm_topk_prob` 的 renorm 是否在 kernel 內處理 → M3 驗）；dispatch 介面相同 | [qwen.py:10-121](moe_infinity/moe_infinity/models/qwen.py#L10) |

---

## 3. Milestone 階梯

> 標 🔴 的是高風險未知，刻意排在最前面 —— 如果會死，要死在
> 50 行的 script 裡，不是死在整合完的管線裡。

### 第一段：純 script，0 行 `src/aug_spec/` 改動（M0–M4）

#### M0 — 原生 example 跑通（環境驗證）✅ 綠（Qwen3，job 234546）

**第一輪（Mixtral）紅了**：job 234500 無 crash、peak 88.3 GB，但
64-token 輸出全是亂碼 —— 重現了使用者既往經驗「moe_infinity 跑
Mixtral 會這樣、跑 Qwen3 不會」（詳見 §2 新增事實）。
**處置：offload 主線全面改用 Qwen3-30B-A3B**，Mixtral 修復列為
獨立 side-quest（不擋 M1–M10）。

**第二輪（Qwen3）綠**：job 234546 —— 續寫合理（正確接續 MoE 解釋）、
無 crash、peak VRAM **35.5 GB**（遠低於全量 61 GB：ratio=0.75 下
64 token 只觸發部分 expert 進 cache，offload 行為已有直接證據）。

- **做**：`sbatch scripts/example.sh` —— 跑 vendored 的
  `examples/mixtral_example.py`（檔名誤導，內容是通用 example）配
  `Qwen/Qwen3-30B-A3B`、`device_memory_ratio=0.75`；含背景 VRAM
  取樣器，結尾回報峰值。
- **動的檔案**：只有 `scripts/example.sh`（未進 git；最初版備份在
  `scripts/example.sh.bak`）。
- **驗收**：log 結尾 (1) 64-token 續寫合理（Mixtral 的失敗模式正是
  這裡亂碼）、(2) `[m0] PASS` 出現。peak VRAM 僅記錄 —— Qwen3 全量
  ~61 GB，ratio=0.75 下 expert 可能整包進 cache，offload 的硬證據
  改由 M1（ratio=0.15）提供。
- **若紅**：是環境問題（CUDA / C++ engine / .so），跟 aug_spec 無關，
  在這裡修完再前進。

#### M1 — 結構探勘 script（read-only）✅ 全綠（job 234584）

第一輪 job 234581 4/6：Q3 紅 = 發現 gate/dense 也是 placeholder
（升格為 §2 事實）、FW 紅 = script bug（改用 config 取維度 +
`block(hs)` 走 hook）。修正後第二輪 6/6 全過 —— next_step.md §5
四個 pre-flight 問題全部 yes，Phase 0 收工。報告：
`tests/offload/m1_probe.out`。

- **做**：`sbatch tests/offload/m1_probe.sh`（快跑加 `--skip-cpu-copy`）——
  `m1_probe.py` 載入 `MoE(...)`（預設 **Qwen3-30B-A3B**、
  `device_memory_ratio=0.15`，順便答 next_step.md §5 Q4 的單卡容量；
  block 搜尋兩家通吃，Mixtral 可用 `--model ... --skip-generate`
  做純結構探勘）後跑六項檢查：
  Q1 block 路徑/type/forward 簽名、Q3 placeholder 證據、
  EX executor/lib/engine 介面盤點（含 `lib.topk_softmax`）、
  Q4 短 generate + VRAM/RSS、
  FW 單獨呼叫一個 MoE block forward（M3 前提）、
  Q2 CPU copy 同進程共存 + host RAM 增量。
- **動的檔案**：只新增 `tests/offload/m1_probe.{py,sh}`。
- **驗收**：六項各自 PASS/FAIL + 證據輸出，結尾 `RESULT: ALL PASS`；
  報告自動存 `tests/offload/m1_probe.out` 備查。單項紅了會繼續跑完
  其他項（例如 Q4 在 0.15 下 OOM 是「答案」不是「故障」）。

#### M2 — assisted generation 探針 ✅ 全綠（job 234630，三輪迭代）

spec-bench 的本質是
`model.generate(assistant_model=model)`（同一物件、shared weights）。
這在 offloaded model 上**從來沒被驗證過** —— HF 的
AssistedCandidateGenerator 會做 KV-cache crop / re-forward，跟
archer engine 的狀態管理可能互咬。這一步如果過不了，後面全部不用做，
所以排在所有整合工作之前。

- **做**：`tests/offload/m2_assisted.py` ——
  手動 `moe._configure_hook(input_ids)` 後跑
  `moe.model.generate(input_ids, assistant_model=moe.model,
  max_new_tokens=32)`。同場測兩件事：
  1. 短 prompt（~30 tokens）能不能跑完；
  2. **長 prompt（≥1024 tokens）**能不能跑 ——
     `interface_example.py` 把 input+output 上限壓在 128
     （`kernel_max_tokens`），不確定是 example 的保守設定還是
     kernel 真實限制；Spec-Bench prompt 動輒上千 token，這必須提前驗。
- **動的檔案**：只新增 script。
- **驗收**：兩種長度都產出 token、連跑 3 個 prompt 不 crash
  （驗證 `_configure_hook` 重複呼叫沒問題）。
- **若紅**：在 script 層 debug（嫌疑：DynamicCache crop、
  `_configure_hook` 的 seq_id 狀態、kernel token 上限）。
  必要時這裡就是「要不要改 C++」的決策點，提早知道比 M7 才知道好。

**第一輪（job 234621）紅，根因已定位**：`model.device` 由 placeholder
推得 cpu → HF assisted 路徑把 input_ids 搬去 CPU（§2 新事實）。
plain greedy 不受影響（不查 `model.device`）。A2/A3 的錯誤是 A1
device-side assert 毒化 context 的連帶傷亡。

**第二輪（job 234626）**：device patch 生效 —— **A1 ✅（assisted ==
plain 全等，最大未知排除）、A3 ✅**。A2 在乾淨進程裡仍紅 → token
長度上限是真的，元兇兩層：
  1. [model_offload.py:381](moe_infinity/moe_infinity/runtime/model_offload.py#L381)
     用寫死的 **1024** 初始化 MoELayer 的 routing buffer
     （[moe.h:57-89](moe_infinity/core/model/moe.h#L57) 預配固定 buffer，
     超過即越界）→ **Python 一行修掉（1024→4096），不用 rebuild**。
  2. [expert_module.cpp:12](moe_infinity/core/parallel/expert_module.cpp#L12)
     `kMaxTokens = 128` —— expert GEMM workspace 的 C++ 硬上限，
     `dispatch_local` 單顆 expert 單次最多 128 token。1200-token
     prefill 平均每顆 75 token（1200×8/128），但 routing 偏斜可能
     超標 → **第三輪實測才知道會不會撞**；撞了的話兩個選項：
     (a) Python 端把 prefill 的 dispatch 按 token 切 ≤128 的塊
     （多次 dispatch、concat，逐 token 正確性不變）；(b) 改 C++
     常數 + rebuild（要先評估 per-expert workspace 的 VRAM 放大）。

**第三輪（job 234630）✅ A1+A2+A3 全綠**：1024→4096 修正後，
1200-token prefill 的 plain 與 assisted 都跑通且逐 token 全等
（plain 10.0s / assisted 10.6s @32 tok）。`kMaxTokens=128` 的
dispatch 上限在 1200-token prefill 下**沒有**被撞到（expert
dispatcher 想必有自行分批，或偏斜未超標）—— M3/M6b 用更長的
Spec-Bench 真實 prompt 時持續留意，撞到再啟用 token 分塊方案。
**整個計畫最大的未知正式排除**；spec decoding 的機制
（同一模型自任 draft + KV crop + per-question hook）在 offload
模型上成立。M5 `load_offload` 的必帶清單：device property patch
（§2）+ vendored `model_offload.py` 的 4096 routing buffer（已進
vendored 源碼）。

#### M3 — `_route_offload` 原型（搬進 adapter 前先單測，Qwen3）✅ 全綠（job 234884）

R1–R4 全過。`route_offload_torch`（[m3_route.py](tests/offload/m3_route.py)）
即 M6a 要搬進 [qwen3.py](src/aug_spec/adapters/qwen3.py) 的版本：自組
top-k softmax + `norm_topk_prob` renorm 與 C++ kernel **逐位元相同**
（R2 權重差 0.0），輸出差 5.96e-08（R3）。R1 與原生 forward 差 7.5e-3
= 原生 `.to(bf16)` 截斷 vs 我們保 fp32，**我們更精確、非錯**。

搬進 adapter 時的兩個收尾（M3 已知、M6a 補上）：
1. **補 `.to(hs_flat.dtype)`** —— `wait_dispatch_local()` 回傳 fp32，
   原生 forward 最後會轉回 bf16，搬進去要照做。
2. **masked draft 的效率優化（非正確性，可延到 M8）**：R4 證明只留
   一顆 expert 時輸出正確,但自組 mask 仍標 top-k 個 index（7 顆零權重
   照樣被 dispatch 搬）。要真正「只搬一顆」,`router_mask` 改從
   `weights_mask != 0` 推（masked 自然剩一顆,verify 自然 8 顆）。
   對 M6b 正確性無影響（輸出相同→acceptance 相同）,只影響 throughput。

- **做**：`tests/offload/m3_route.py` —— 寫一個 standalone 函式
  `route_offload(block, hs_flat, gate_logits)`：
  組 per-token `router_mask` / `routing_weights_mask` →
  `dispatch_local` + `wait_dispatch_local`。mask 的組法兩條路都試：
  (i) 直接用 `block.lib.topk_softmax(gate_logits)`（fused kernel，
  天生任意 top_k）；(ii) 自己用 torch 組（對照組，順便驗 kernel 是否
  處理了 `norm_topk_prob` 的 renorm）。
  對同一個 hidden_states 輸入，比對與原生 `Qwen3MoEBlock.forward`
  的輸出。
- **動的檔案**：只新增 script。
- **驗收**：(i) 與原生 forward 輸出**逐位元相同**（同一條 dispatch
  路徑，理應全等）；(ii) 與 (i) allclose（差異就是 renorm 的線索）；
  另外用 masked gate（模擬 `random_mask` draft 的 -inf mask）跑一次
  確認不炸。
- 這一步的產出物（那個函式）就是 M6a 要原封搬進
  [qwen3.py](src/aug_spec/adapters/qwen3.py) 的程式碼。Mixtral 版
  （需另處理 top-2 hardcode）等亂碼 side-quest 修好後比照辦理。

#### M4 — CPU-source merge 原型：三變體計時 + 精度比對 ✅ 全綠（job 234890）

三變體（(a)/(c)/(d)）全逐位元對齊 reference，**Qwen3 主線
裁決 = (d) CPU merge**（四項指標全勝，數字見 §1.3）。對 M7 是大簡化：
(d) 本質就是「現有 `build_weighted_avg(cpu_block)` + `.to(cuda)`」，
M4 已證逐位元正確，**幾乎不用寫新 merge 程式碼**。一個小優化記下：
現有 `build_weighted_avg` 迭代全 128 顆 expert（跳過零權重），offload
版應只迭代 nonzero（M7 接入時改）。

- **做**：`tests/offload/m4_merge.py` ——
  `cpu_source = AutoModelForCausalLM.from_pretrained(..., device_map="cpu")`
  與 `MoE(...)` 同進程共存；實作**兩個合規變體**：
  (c) GPU 分塊 fp32（chunk 外層、expert 內層，fp32 chunk buffer）、
  (d) CPU merge（host fp32 合完只傳結果）。對任一層、任一組 weights，
  兩個變體都比對 **hf backend** 的 `adapter.build_weighted_avg`
  （另開一個純 CPU 的 HF model 跑現有函式即可，不需要 GPU 版）。
- **驗收**：
  1. (c)、(d) 與 hf backend 結果**逐位元相同**（`torch.equal`；
     退一步至少 `allclose(rtol=1e-3)`，但理論上應全等 —— 不全等就是
     實作順序寫錯了）；
  2. (c) 的 GPU 瞬時暫存 ≤ 工作區（幾十 MB 級，nvidia-smi 佐證）；
  3. 兩變體各記「merge 全部 MoE 層」的 wall time 與 PCIe 傳輸量 ——
     **這個數字直接決定 M7 之後用哪個變體**，也是論文 draft-side
     成本表的素材。（Qwen3 注意：expert 小（9.4 MB）但每層 merge
     K=16 顆、來源 pool M=32，傳輸模式與 Mixtral 很不同，兩變體的
     勝負可能反過來。）

### 第二段：進 `src/aug_spec/`，一次一個檔案（M5–M7）

#### M5 — `loader.load_offload()`（不接 CLI）✅ 全綠（job 234913）

L1–L4 全過：回傳 (model, tokenizer, moe, cpu_source) 形狀對、device
patch 生效（model.device==cuda:0）、`model.generate`（手動
`_configure_hook` 後）跑通、cpu_source 是真實 bf16 權重。hf 路徑
未動（本地驗：`load_model` 簽名不變、`moe_infinity` 非 eager import），
完整 golden 比對延到 M6b（動 cli.py backend 分流時）。
**M6 注意**：loader 舊的 `get_model_device()` 讀
`next(model.parameters()).device`，offload 下回 cpu（參數是
placeholder）≠ 我們 patch 的 `model.device==cuda:0`；接 controller
時別誤用它。

- **做**：在 [loader.py](src/aug_spec/runtime/loader.py) 加
  `load_offload(model_id, offload_path, device_memory_ratio, ...)
  → (model, tokenizer, moe, cpu_source)`。**cli.py 完全不動。**
- **驗收**：
  1. `python -c "from aug_spec.runtime.loader import load_offload"`；
  2. golden 比對：`configs/_smoke.yaml` 重跑數字不變；
  3. `tests/offload/m5_loader.py` 用 `load_offload` 載入後跑一次
     forward。

#### M6a — adapter 加 offload 分流（只 Qwen3）⚠️ code 完成；單層比對測法作廢，驗證移交 M6b

`_route_offload` + `_standard_routing` 分流已寫進
[qwen3.py](src/aug_spec/adapters/qwen3.py)，本地驗過（import、hf block
不被分流）。GPU 單層比對（m6a_route）A1/A2 爆 0.66/1.42 —— 四輪 debug
（§2 新事實）證明**這是測法問題不是 code 問題**：`dispatch_local` 脫離
完整 forward 序列時踩搬運競態，單層比對本質不可靠。`_route_offload` 餵
dispatch 的 mask 與原生 forward 數值相同（M3 R2），序列內 dispatch 可靠
（F3 0.05、F2 token 確定），故邏輯上序列內等價原生。**真正的驗證 = M6b
端到端 acceptance 比對**。m6a_route.py / m6a_debug.py 保留作脫序競態的紀錄。

- **做**：把 M3 的函式搬進
  [qwen3.py](src/aug_spec/adapters/qwen3.py)，
  `_standard_routing` 開頭加
  `isinstance(block, Qwen3MoEBlock)` 分流
  （import 包在 try/except，hf-only 環境不需裝 moe_infinity 也能跑）。
  Mixtral adapter **這一階不動**（亂碼 side-quest 修好後比照）。
- **驗收**：M3 的比對測試改成呼叫 adapter 方法重跑，仍全等；
  `_smoke.yaml` golden 不變。

#### M6b — 第一條 E2E：`backend: offload` + `random_mask` ✅ 管線打通（job 236221）

hf 對照(`_smoke_random`)與 offload(`_smoke_offload_random`)的
**MAT 1.005 / AccRate 0.0017 完全相同**，offload TPS 1.23 < hf 1.78
（真的在搬權重）。整條新管線（config backend → load_offload → adapter
dispatch 分流 → phase patch → specbench → `_configure_hook`）接對，且
最終實證了「`_route_offload` 在真實序列裡 ≡ 原生」(M6a debug 的結論)。
改動：specbench 加 `before_generate`、loader `get_model_device` 偵測
placeholder fallback、cli 加 backend/offload 分流 + offload force-exit
（避 moe_infinity C++ thread pool 的 shutdown hang）。
**限制**：AccRate 趨近 0（random_mask 在 128-expert 幾乎必被拒，預期；
Mixtral 8-expert 才有 0.22）→ 兩 backend「都爛到底」，一致性驗證力有限。
**強數值一致性驗證移交 M7**（merge draft，acceptance ~0.4，hf vs offload
比才有說服力）。

第一條端到端刻意選 `random_mask`：masked 路徑**零權重讀取**、
不需要 cpu_source、不需要 merge —— 它只驗「config → loader →
adapter 分流 → phase patch → specbench」這條管線本身。

- **做**（這一階動三個檔案，是全計畫最大的一步，但每塊都已被
  M2/M3/M5 單獨驗過）：
  1. [cli.py](src/aug_spec/cli.py)：`RunConfig` 加
     `model.backend`（預設 `hf`）+ `model.offload.*`；
     `run_experiment` 依 backend 選 loader；
  2. `_configure_hook` 接線：offload 時把
     `lambda q: (moe._configure_hook(ids), controller.reset())`
     塞進 `on_question_start`（或加在
     [phase.py](src/aug_spec/runtime/phase.py) 的 callbacks 工廠，
     看哪邊乾淨）；
  3. 新增 `configs/_smoke_offload_random.yaml`
     （qwen3 + random_mask + offload）。
- **驗收**：
  1. E2E 跑完不 crash；
  2. MAT / AccRate 與 **hf backend 的 random_mask smoke**
     （新增 `configs/_smoke_random.yaml` 當對照組）在 ±0.1 內 ——
     random_mask 的 acceptance 本來就爛（~0.2），重點是兩個 backend
     **一樣爛**；
  3. `_smoke.yaml` golden 不變。

#### M7 — merge 類 draft 上 offload（論文主菜）✅ 管線達標（job 237171），強一致性待大樣本

merge draft（count）在 offload 跑通、無 crash。MAT hf 1.33 vs offload 1.41
（差 0.08 < 0.1，**±0.1 達標**）；AccRate 0.1145 vs 0.142。
實作：build_weighted_avg 在 offload 從 `_cpu_merge_source`（controller
attach 的 cpu_block）CPU 合、搬 GPU（M4 (d)，逐位元正確），hf 路徑不變。
**用 CPU merge 後 count 也 offload-safe**（PCIe 恆 1 顆，推翻 next_step.md
§2.3「offload 限 topm」的 GPU-merge 前提）。
**說服力限制**：smoke 樣本小 + count 冷啟動 → acceptance 0.11 << 大樣本
真實 0.43,0.11 vs 0.14 的差無法判定是噪音還是偏差。merge 兩 backend
逐位元相同(M4),差異只來自 offload verify 的 cache 噪音(0.05/層,§2)。
TPS offload 0.99 vs hf 3.44(搬權重,M8 細究)。
**強一致性 + 論文數據移交大樣本對照**：questions_per_cat 多、
max_new_tokens 大的 hf count vs offload count(同時產出 PROGRESS.md 的
offload acceptance 表)。

- **做**：
  1. 把 M4 **勝出的那個變體**搬進 adapter（輸的留在 m4_merge.py 當紀錄）；
  2. offload backend 時讓 merge 來源指向 `cpu_source` 的對應 block
     （controller 或 adapter 持有 `layer_idx → cpu_block` 映射，
     實作時選侵入最小的位置）；
  3. 只開放 `topm_count` / `prefill_count` / `prefill_topm_count`；
     `count`/`uniform`/`softmax` 在 offload backend 直接
     raise，錯誤訊息指到 next_step.md §2.3；
  4. 新增 `configs/_smoke_offload_topm.yaml`，config 註解寫出 §1.2 的
     0.2x → `device_memory_ratio` 換算式。
- **驗收**：
  1. MAT 與 hf backend 同 draft 的 smoke 在 ±0.1 內（merge 數值經 M4
     保證逐位元相同，差異只可能來自 verify 路徑 —— 超標就回 M3 查
     dispatch）；
  2. **archer cache 縮到工作區大小（1–2 expert size）仍可正常運作**
     —— archer cache 最小可運作尺寸的待實證。太小若死鎖/抖動，量出
     最小可運作值並回填 §1.2 的工作區定義。
- **到這裡，「用 moe_infinity 當 inference engine 跑我們的
  merge-based spec decoding」就完成了。** M8 之後是論文的
  throughput 軸，獨立可排程。

### 第三段：throughput 軸（M8–M10，對應 next_step.md Phase 4–8）

| # | 內容 | 驗收 |
|---|---|---|
| M8 | `drafts/none.py` + `aug_spec bench` 子命令（純 generate 迴圈） | `bench` 在 hf 與 offload 都跑得動、報 tokens/sec |
| M9a | **三資源時間線 profiling**（GPU inference / CPU merge / PCIe 的 busy/idle/overlap）+ 0.2x 預算稽核（NVML peak）；評估能否做成 §1.3/§1.4 的量化分析 | 三資源 busy/idle/overlap 有數字；§1.3「CPU merge 藏在 verify 底下」、§1.4「verify PCIe 閒置率」可定量；peak ≤ 0.2x（詳見下方 M9a） |
| M9b | 據 M9a 把 **CPU/GPU 平行化**（背景 thread 預合下一 cycle merge）+ **expert prefetch**（回收的 7.25 GB 當 buffer）寫進程式碼 | overlap 後 end-to-end throughput 提升、prefetch 填 verify 氣泡（詳見下方 M9b） |
| M10 | `cache_policy`（ondemand/caching）、specmoe 的 `replace_cache_candidates` hook（§1.2 帳下 SpecMoE = N=1）、`configs/baselines/` 等量產 configs；topm 的 **refresh 節流 ablation**（每 k cycle 才 merge，§1.3） | 四方對比表（OnDemand / Caching / SpecMoE / Ours）跑完，數字進 PROGRESS.md |

#### M9a — 三資源時間線 profiling + 量化評估

§1.3 主張「CPU merge 與 GPU verify 資源正交、可重疊」、§1.4 主張「verify 階段
7.25 GB 可回收當 prefetch buffer」——兩者的**時間收益都標『待量』**。M9a 把這兩個
口頭主張變成數字,並評估能否寫成 §1.3/§1.4 那樣的量化分析（roofline-style,可進論文）。

**三個要 profile 的資源**（一個 spec-decoding cycle 內）:

| 資源 | 在做什麼 | 量法 |
|---|---|---|
| **GPU inference** | draft 的 dense merged-expert forward + verify 的 `dispatch_local` | `torch.cuda.Event` 夾 draft / verify 段 |
| **CPU expert merging** | `build_weighted_avg(cpu_source)`（refresh 觸發） | `time.perf_counter` 夾 refresh |
| **PCIe** | merged expert H2D 上傳 + verify 的 archer expert fetch（+ 未來 prefetch） | NVML `pcie.tx/rx` 取樣 + archer fetch 計時 |

**做**:給每個運算打 start/end 時間戳,在 ~10 個 cycle 內收集,算出每資源的 **busy /
idle 時間**與**兩兩 overlap**。關鍵兩個比值:
1. **CPU merge busy / GPU verify busy** —— < 1 表示 CPU merge 能完全藏在 verify 底下
   （§1.3 主張成立的量化條件）。
2. **verify 期間 PCIe idle 比例** —— 有閒置才值得 prefetch（§1.4 的「有閒置 PCIe 才有肉」
   的直接答案）。

**評估產出**:一張像 §1.3/§1.4 的時間線分解表 + 結論「(a) CPU merge 可藏多少 %、
(b) verify PCIe 閒置率多少 → M9b 該做哪些重疊、prefetch 值不值得」。順帶吃掉原 M9 的
0.2x 預算稽核（NVML driver peak 進 summary,所有系統 ≤ 0.2x + 工作區）。

**風險**:archer 的 fetch 在 C++ thread,Python 端時間戳抓不到引擎內部 → PCIe busy 可能
只能靠 NVML 取樣間接估（取樣頻率 vs cycle 長度要對得上）。先確認取樣解析度夠不夠。

#### M9b — CPU/GPU 平行化 + expert prefetch（據 M9a 數字才做）

**前提**:M9a 顯示有閒置可榨（CPU merge 藏得進 verify、或 verify PCIe 有閒置）。沒肉就
不做（§1.4 已言明 prefetch 收益不預設）。

1. **CPU/GPU 平行化（把 §1.3「結構上可重疊」變現實）**:起一個 CPU worker thread,
   verify 一結束就**不阻塞地**用當前 counts 預合下一 cycle 的 merged expert（甚至逐層
   pipeline:第 ℓ 層 verify 跑時合第 ℓ 層的下一版）；draft phase 開始前 join + H2D。把
   §1.3 的 651 ms merge 從 critical path 移到 GPU 計算的影子裡。
   - **正確性關鍵**:merge 結果必須在下個 draft 用到前 ready;counts 變化慢,容許用
     上個 cycle 的（§1.3 已備此論點）。
2. **expert prefetch（用 §1.4 回收的 7.25 GB buffer）**:draft 一結束 flush merged
   residency,把那塊當 prefetch buffer,在 verify 時預抓下一層 expert。預測訊號:
   moe_infinity 的 tracer（本 fork 已註解,要重啟）或拿 draft 階段 merged-expert hidden
   states 跑 gate 當預測。填 verify 的串行氣泡（每層 attention+gate compute）。

**驗收**:(1) 平行化後 end-to-end tokens/sec 對比 M9a 的同步版有提升;(2) prefetch 開/關
的 verify 牆鐘差（填掉多少氣泡）。兩者都對比 SpecMoE,坐實「三資源解耦」是我們相對
SpecMoE 的結構性 throughput 優勢（SpecMoE 的 draft 在 GPU、無 CPU merge 可重疊、pinned
expert 無法回收當 prefetch buffer——§1.4 的不對稱表）。

**風險**:thread 同步的正確性（merge race、用錯版本的 counts）;prefetch 預測準確度低時
純浪費 PCIe（§1.4 已排除「抓全部 E=128」）。M9a 沒肉就整個 M9b 不做。

---

## 4. Debug 工具箱（卡住時先查這張表）

| 症狀 | 第一嫌疑 |
|---|---|
| 輸出全是同一個 token / 亂碼 | 讀到 placeholder 權重 —— 檢查是不是有程式碼在 offload model 上讀 `experts[e].w*.weight` |
| 第二個 question 開始 crash 或 hang | `_configure_hook` 沒有每次 generate 前重呼叫（seq_id 過期） |
| Mixtral 輸出亂碼 | **已知問題**（M0 job 234500，§2）—— 不要在 offload 主線用 Mixtral；修復是獨立 side-quest（嫌疑：expert GEMM 權重映射） |
| assisted generation 一掛就 device mismatch（cpu vs cuda） | `model.device` 從 placeholder 推得 cpu（§2）—— 確認 `load_offload` 的 device property patch 有套上 |
| 一個錯之後同進程所有 CUDA 呼叫全亂報 | device-side assert 毒化 context —— 只有**第一個**錯誤可信，重跑時每檢查一個進程 + `CUDA_LAUNCH_BLOCKING=1` |
| Qwen3 對、Mixtral 錯（或反過來） | 除上述已知問題外：`router_mask` 組法差異（Mixtral Sync forward hardcode top-2；Qwen3 走 `lib.topk_softmax`） |
| 長 prompt 掛、短 prompt 好 | kernel token 上限（M2 第 2 項就是在防這個） |
| offload MAT 明顯低於 hf 同 draft | verify 路徑數值不等 —— 退回 M3 的單 block 比對重查 |
| host RAM OOM | `MoE(...)` 的 offload 工作區 + `cpu_source` 疊加超過節點上限 —— M1 有量測，回去對數字 |

另外兩個通用原則：

- **二分回退**：每個 milestone 一個 commit，出問題先
  `git stash` / checkout 回上一階確認那階還綠，再前進找 diff。
- **對照組先行**：任何 offload 數字出來前，先有同 draft、同 smoke
  規格的 hf backend 數字放旁邊 —— 沒有對照組的「數字怪怪的」
  無法 debug。
