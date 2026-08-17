# GPT-OSS 20B acceptance 落地計劃（gptoss_acceptance_plan.md）

> **定位**：`baseline_tables_plan.md` WS-G（G0–G4）的執行計劃——把 Table 1 的
> GPT-OSS 六列跑出來。**只跑 hf-backend acceptance，不跑 offload/TPS**（S1 定案）。
> 正式數字寫回 `paper/tab_main_baselines.tex`；本檔 §6 狀態表與
> `baseline_tables_plan.md` §7 同步更新。

---

## 1. 現況盤點（2026-07-16 已驗證）

**環境已就緒：**
- venv transformers **4.57.6**（≥4.55，支援 `gpt_oss` model_type）。
- `openai/gpt-oss-20b` 已在 HF cache（`models--openai--gpt-oss-20b`）。
- adapter 由 `adapter_for_config(model.config)` 自動偵測，config 不用寫 family。
- specbench 的 prompt：gpt-oss tokenizer 有 chat_template（harmony）→ 走
  `apply_chat_template`（Qwen3-Base 是 vicuna fallback）；humaneval 照 A4 走 raw completion。

**gptoss adapter（`adapters/gptoss.py`）已實作：**
- target forward（softmax-after-topk + `capture_softmax`）、`build_weighted_avg`
  （fused 四張量 fp32 累加）、`_run_dense_expert`（clamped SwiGLU）、multi 分支
  （`_route_multi_expert`；`_swiglu_stack` 回 None → 泛用 per-cluster 迴圈，正確稍慢）。
- freq / co-occur 統計走 `capture_softmax` 累積（`drafts/base.py`；hybrid 的
  `cooccur_scope: decode` 由 `_prefill_seen` 機制處理），hf backend 有效。
- → **random_merge 不用寫新 code 就能跑**（Step 0a 直接 smoke）。

**缺口（本計劃要補的全部程式碼）：**
- **hybrid 的 act-sim 在 hf backend 無捕捉路徑**：`accumulate_activation_sim`
  只從 C++ offload engine 的 `dispatcher.get_captured_expert_outputs()` 拉資料
  （`offload_merge.on_verify_layer` 呼叫）。act-sim 缺席時 hybrid **靜默退化**
  （`pair_sim=None` → 全 pair 均攤 -α 偏移，排序 = 純 cooccur）→ 擋 Ours（K1）。
- `make_masked_forward` = NotImplementedError（`gptoss.py:184`）→ 擋 Random-prune、Enumerate。
- `make_substitute_forward` / `expert_flat_weights` 未實作 → 擋 SpecMoE。
- `scripts/collect_calibration.py` 對 gptoss 直接 raise（fused 張量沒有 raw expert
  module 可堆疊）→ 擋 HC-SMoE / Enumerate 的 calibration。
- `scripts/search_naee.py` 重路由是 softmax-before-topk（qwen3/mixtral）語義，
  gptoss 要 softmax-after-topk 分支。

## 2. 決策記錄

| # | 決定 | 理由 |
|---|---|---|
| K1 | **Ours = 目前最優設定 hybrid a75（2026-07-16 用戶拍板；曾議 freq_slice 已否決）**：`cluster: {name: hybrid, alpha: 0.75, metric: l2, norm: rank, cooccur_norm: raw, cooccur_scope: decode, within_weight: uniform}`，沿 `t1_qwen3_ours` | 兩模型 Ours 同方法定義，Table 1 才可比。代價 = 新增 **Step 0.5：hf-backend act-sim 捕捉**（cooccur 分量 hf 已有效，只缺 act-sim） |
| K2 | draft 尺寸：`topm_count{M:8, K:4, draft_top_k:4}`、`random_merge{K:4}`、`random_mask{num_keep:4}`、`specmoe{N:4, route_top_k:4}`、HC-SMoE K=4 | 32 experts、native top-4；M=2K 慣例比照 Qwen3（§4 定案）。M=2K → \|group\|=2 pair regime，`within_weight: uniform` 與論文 Eq（α=1/\|C\|）一致 |
| K3 | `reasoning_effort` 不注入（用模型預設 medium） | 少一個自由變數；記進 provenance |
| K4 | 載入要求 **bf16（≈42GB）**：transformers 對 MXFP4 checkpoint 在無 triton kernels 時自動 dequant 成 bf16 | Step 0a 看 log 確認；若環境 triton 可用而載成量化版，在 loader 加 opt-in `Mxfp4Config(dequantize=True)`（僅此一處小改） |
| K5 | 跑數協議 = t1：`T:5, questions_per_cat:15, max_new_tokens:512, humaneval:true, mt_bench_pooled:true` | 與 Qwen3 t1 六列同協議 |

## 3. 步驟

### Step 0a｜randmerge smoke（不寫新 code，環境風險早爆，最先送）
- config：`g0_gptoss_randmerge_smoke.yaml`（t1 協議欄位但 qpc=1、mnt=128）。
- 驗收：載入為 bf16（log 無量化路徑 / 有 dequant 訊息）、harmony chat template 正常、
  C-BOOT（prefill_warmup + KV-copy）不炸、AccR > 0 且量級合理、humaneval raw 分支正常。

### Step 0.5｜hf-backend act-sim 捕捉（新工作項，解鎖 hybrid-on-hf）
- `drafts/base.py`：把 `accumulate_activation_sim` 的累積核心抽成
  「吃 captured list」的內部函式（offload 路徑行為不變，抽取式重構）；
  新增 hf 入口 + 「本題本層 prefill 是否已捕捉」的判定 helper
  （語義同 `act_sim_prefill_only`：只收 prefill，之後整題凍結）。
- `adapters/gptoss.py` target 分支：當 cluster method `needs_activation_sim`
  且本層 prefill 未捕捉 → 用 fused 權重對 routed tokens 重算 per-expert 輸出
  （top-4 winners，成本 ≈ 一次額外 MoE prefill forward/題，僅 prefill 相位）→ 餵累積器。
  首次成功累積印一行 INFO（smoke 驗收用，防靜默退化假綠）。
- **不動 qwen3/mixtral adapter**（Qwen3 Ours 走 offload，不需要）。
- 單元測試：合成 captured list 的累積數值 vs 手算；prefill-only 凍結；
  offload 抽取式重構前後 `tests/unit` 全綠。

### Step 0b｜Ours（hybrid a75）smoke（吃 Step 0.5）
- config：`g0_gptoss_ours_smoke.yaml`（hybrid a75 + uniform，qpc=1、mnt=128）。
- 驗收：act-sim INFO 有出現（確認非退化）、分群產生 pair、AccR > 0 量級合理。

### Step 1｜G1 `gptoss.make_masked_forward`（解鎖 random_mask / static_mask）
- draft 相位：router logits → 非 kept `masked_fill(-inf)` → topk → **softmax-after-topk**
  → scatter → `mlp.experts(...)`；target 相位照 averaged forward 的 target 分支
  （含 `capture_softmax`）。照 `qwen3.py:278` 的形狀，語義照 gptoss target 分支寫。
- 單元測試：mask 全開 = 原始路由；mask 只留 k 顆時輸出只含 kept 貢獻。

### Step 2｜G2 gptoss SpecMoE（解鎖 specmoe）
- `expert_flat_weights`：fused 3D 張量切 per-expert（gate_up/down + 兩組 bias 攤平
  concat，fp32）。
- `make_substitute_forward`：full softmax 給 capture、topk winner 經 substitute
  table remap、重建 router_scores 走 `mlp.experts`；offload 分支一律不進。
- 單元測試：identity table = 原始路由；flat weights 長度/數值抽查。

### Step 3｜G3 calibration gptoss 路徑 + 送 B1 job
- `collect_calibration.py` 加 fused bmm 全 expert 輸出路徑（`gate_up_proj`+bias →
  clamp/GLU → `down_proj`+bias，照 `_run_dense_expert` 向量化成 [n,T,D]）。
- 送 job：C4 32×2048、逐層落盤 → `output/calibration/gpt-oss-20b/`。

### Step 4｜G4 + C1/D1 產物（吃 Step 3 產物）
- `search_naee.py` 加 softmax-after-topk 重路由分支（`norm_topk_prob` 語義不適用
  gptoss，別沿用）；精確枚舉 C(32,4)=35,960/層 → `output/naee/gpt-oss-20b_r4.json`。
- `build_hc_smoe.py` 跑 gptoss（32→4 群，average-linkage）→ `output/hc_smoe/gpt-oss-20b_K4.json`。

### Step 5｜送六列 acceptance run（t1 協議，config 命名 `t1_gptoss_<method>.yaml`）

| Table 1 列 | draft | 依賴 |
|---|---|---|
| Random (prune) | `random_mask{num_keep:4, seed:0}` | Step 1 |
| Enumerate | `static_mask`（NAEE json） | Step 1+3+4 |
| Random (merge) | `random_merge{K:4, draft_top_k:4, seed:0}` | Step 0a 即驗 |
| HC-SMoE | `static_merge`（HC json） | Step 3+4 |
| SpecMoE | `specmoe{N:4, route_top_k:4}` | Step 2 |
| Ours | `topm_count{M:8, K:4, draft_top_k:4}` + hybrid a75（K1） | Step 0.5+0b |

- n=1 起跳；與 Ours 差 <3pp 的對手補到 n=3（照 baseline_tables_plan §4 規則）。

### Step 6｜收尾
- 數字 + provenance（config、job id、seed、日期、hf backend、K1/K3 註記）
  → `tab_main_baselines.tex`。
- `baseline_tables_plan.md` §7 與本檔 §6 打勾；`PROJECT_GUIDE.md` 同步；
  `tests/unit` 全綠。

## 4. Guardrails（沿 baseline_tables_plan §2）

- 不動 C++；不動 qwen3/mixtral 既有執行路徑——改動 = gptoss adapter 內**新增**方法、
  `drafts/base.py` 的**抽取式重構 + 新增** hf 累積入口、兩個離線 script **新增**分支。
- 改 adapter/drafts 前後跑 `tests/unit`；一律 sbatch（watchdog），不在 login node 跑。

## 5. 風險

1. **MXFP4 載入型態**（K4）：Step 0a 最先驗；不對就加 dequantize opt-in。
2. **hybrid 靜默退化**：act-sim 缺席時排序退化為純 cooccur 且無報錯——Step 0b 驗收
   必須看到 act-sim INFO，正式 run 前不可跳過。
3. **harmony template 下的 acceptance 語境**：gpt-oss 生成含 analysis channel，
   mnt=512 內 reasoning 佔比高——六列同協議內部公平，跨模型解讀時註記即可。
4. **GPU 排隊**：Step 0a 不依賴新 code，最先送；其餘 job 寫完即送。
5. **枚舉量 35,960/層 × 24 層**：Qwen3 10⁵ 抽樣已驗 31 分鐘，gptoss 量級更小，預期 <1 小時。

## 6. 狀態追蹤

| 項目 | 狀態 | 備註（config/job id） |
|---|---|---|
| Step 0a randmerge smoke | ✅ 2026-07-16 | job 264776：**bf16 dequant 確認**（log 明寫 fallback、VRAM 43.7GB）、7 subtask 含 humaneval 全跑、lazy_build 無 C-BOOT fail。AccR 0.0159（弱 baseline 預期內，見 §6 註記） |
| Step 0.5 hf act-sim 捕捉 + 測試 | ✅ 2026-07-16 | `drafts/base.py` 抽取 `_accumulate_act_sim` + `wants_prefill_act_sim`/`accumulate_prefill_act_sim`；`gptoss.py` `_fired_expert_outputs` + target prefill 接線；`tests/unit/test_actsim_hf.py` 5 測試，全套 82 綠。GPU 實證看 Step 0b 的 `[act_sim]` INFO |
| Step 0b Ours（hybrid a75）smoke | ✅ 2026-07-16 | job 264778：**`[act_sim] hf prefill capture engaged (layer 0, 31 fired experts, n=32)` 出現 = hf act-sim 路徑生效、非退化**。AccR 0.0468（randmerge 的 3 倍、humaneval 0.118 最高 → 排序有真訊號）。⚠️ 絕對值遠低於 Qwen3 Ours ~0.5——單層 prefill 31/32 顆 expert fired，gpt-oss routing 遠比 Qwen3 平坦，merge 類 draft 天生難；「系統性問題 vs 模型難度」由 specmoe smoke 分辨（真 expert draft，若也 ~0.05 = 系統性） |
| Step 1 G1 masked forward + 測試 | ✅ 2026-07-16 | `gptoss.make_masked_forward`（softmax-after-topk on masked logits；target 不 capture，同 qwen3）；`test_gptoss_forwards.py` |
| Step 2 G2 substitute forward + flat weights + 測試 | ✅ 2026-07-16 | `drafts/specmoe.py::gptoss_substitute_forward`（remap 碰撞用 scatter_add_；GPU dense path 只吃 dense routing_weights，已核對 HF 原始碼）+ `gptoss.expert_flat_weights`（含 bias）；SpecMoeDraft prepare→refresh roundtrip 測試 |
| Step 3 G3 calibration 路徑 + B1 job | ✅ 2026-07-16 | code：`expert_output_fn`/`gptoss_expert_outputs`（fused clamped-GLU，⚠ gptoss block 的 output[1] 是 post-scatter scores → hook 重算 full logits）；job 264785 ✅ → `output/calibration/gpt-oss-20b/`（24 層 ×4096 tokens，1.1GB） |
| Step 4 G4 NAEE 語義 + C1/D1 產物 | ✅ 2026-07-16 | `naee_losses` 加 `softmax_after_topk`（= norm=True 舊分支，等價測試鎖定）；D1 job 265017 ✅ `output/naee/gpt-oss-20b_r4.json`（exact 35,960/層、252s，best/mean ≈ 0.4 有鑑別力）；C1 job 265022 ✅ `output/hc_smoe/gpt-oss-20b_K4.json`（**又是巨群退化**，後層 [29,1,1,1]，同 Qwen3 已知行為、預期弱） |
| Step 5 六列 t1 run | 🔶 M8K4/M4K4 ✅ | 第一波 jobs 265024/265025/265029–265032 全部 scancel（2026-07-16，讓 GPU）。**正式數字（2026-07-17，t1 協議 105 題）：M8K4 Ours（合併，hybrid a75）= job 265219，overall AccR 0.0694 / MAT 1.35（act_sim 有 engage，非退化）；M4K4（不合併，同預算）= job 265220，overall AccR 0.3355 / MAT 2.67**——完整協議下確認合併代價 ≈4.8×（差距遠超 3pp 噪音門檻，n=1 可下結論）。其餘四列（randmerge/randmask/specmoe/enum/hcsmoe）configs 就緒待指示：`for c in randmerge randmask specmoe enum hcsmoe; do sbatch --job-name=t1_gptoss_$c --time=10:00:00 scripts/run_watchdog.sh configs/t1_gptoss_$c.yaml; done` |
| **診斷：merge 類 AccR 異常低 → 結論：非 bug，線性合併在 gpt-oss 上本質失效** | ✅ 2026-07-16 | smoke qpc=1 全比較：**singleton（M=8,K=8 恆等拷貝，同一條 multi 路徑）AccR 0.4033 / MAT 2.98 = 全場最高** > specmoe 0.228 > randmask 0.189 >> ours(M=8→K=4 pair merge) 0.047 > randmerge 0.016。判定：**gptoss multi/merged 路徑正確**（恆等拷貝 0.40 證明 plumbing 無損）；把 top-8 兩兩合併成 4 顆就從 0.40 崩到 0.047——**pair merging 對 gpt-oss expert 毀滅性 lossy**。機理推測：32 顆粗粒度 expert + 平坦 routing（單層 prefill 31/32 fired）⇒ expert 間冗餘遠低於 Qwen3 的 128 顆細粒度,無「相似 pair」可合;與 Qwen3 上 merge>prune 的結果不矛盾,是 negative transfer。**論文層面選項（待用戶拍板）**：(A) 誠實照跑六列,Ours 在 GPT-OSS 輸 SpecMoE,加 discussion（fine-grained MoE 才有 merge 空間——可反向支撐「為何 Qwen3/128-expert 是對的靶」）;(B) GPT-OSS 的 Ours 換操作點（如 M=4,K=4 無合併=dynamic prune,但破壞 method 軸語義）;(C) GPT-OSS 降級 footnote。singleton 0.40 可當 mechanism 佐證引用。**補測 M=4,K=4（job 265169，2026-07-16，用戶指示）：AccR 0.3082 / MAT 2.52**——同預算（4 顆駐留）下「count 動態 top-4、不合併」贏 SpecMoE(0.228) +8pp、贏 randmask(0.189) +12pp ⇒ Ours 的**動態 count 選集在 gpt-oss 上有效且同預算最強**,失效的只有合併那一步(0.308→0.047)。選項 B 的具體形態即 M4K4（帳面是 dynamic prune;若論文把 Ours 定義為 relation-guided cluster-and-merge 的自適應退化——expert 無冗餘時 relation map 指向不合併——則可辯護為 Ours 的退化操作點,敘事由用戶拍板） |
| Step 6 tex + provenance + 文件同步 | ⬜ | |

（unit suite 91 綠，2026-07-16；jobs 終態由 filesystem monitor 盯，不打 SLURM query）
