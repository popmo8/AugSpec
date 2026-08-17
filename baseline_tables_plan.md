# 論文主表補齊計劃（baseline_tables_plan.md）

> **Deadline：投稿約 2026-07-25。** 目標：填完 `paper/tab_main_baselines.tex` 的
> Table 1（acceptance 軸）與 Table 2（throughput 軸）。正式數字一律寫回該檔的表格
> 與 provenance 註解（唯一紀錄處）。進度照 §7 狀態表更新。

---

## 0. Policy 軸語義（Table 1 的 {random, static, dynamic}，實作前先讀）

| policy | 意義 | 對應列 |
|---|---|---|
| **random** | prune/merge 集合**隨機決定、整個 run 固定**（seed 記進 provenance） | Random ×2 |
| **static** | 集合由 calibration 離線決定、整個 run 固定 | Enumerate（NAEE）、HC-SMoE、MC-SMoE |
| **dynamic** | 集合每個 verify cycle 依 target routing 統計更新——**只有這兩個是動態** | SpecMoE、Ours |

## 1. 範圍（已定案，不重議）

| # | 決定 |
|---|---|
| S1 | GPT-OSS：跑 acceptance（hf backend）填滿 Table 1 六列；**不跑 offload/TPS**（引擎不支援該家族），Table 2 不含 GPT-OSS |
| S2 | Enumerate：Qwen3 用**抽樣版**（10⁵ 子集，表格標註 sampled）；Mixtral（C(8,1)=8）與 GPT-OSS（C(32,4)=35,960/層）精確枚舉 |
| S3 | Mixtral 只跑 hf-backend acceptance（bf16 ≈93GB 單卡 H200 可載）；不移植 offload 路徑 |
| S4 | 新 baseline 全部只需 acceptance（hf）；**唯一要新跑的 TPS = MoE-Caching**（Qwen3、offload、非投機、batch 1） |
| S5 | **Coding 欄 = HumanEval**（164 題 seed 抽 80，與其他 subtask 的 80 對齊），兩表所有列都要填；整合規格見 A4 |

tex 結構已改完（Table 1 無 TPS 欄、Table 2 Qwen3-only 且無 Model 欄、batch-64 敘述已移除）；
batch size = 1 要寫進 experimental setup 內文。

## 2. Guardrails（投稿前絕對不做）

- 不動 C++；不動既有方法的執行路徑（topm_count / specmoe / merged_cache / offload_merge）
  ——已填數字不能被影響。改動只能是**新增** draft、**新增** CLI 分支、**新增** script。
- 不做 `CODE_HEALTH_REVIEW.md` 的重構；改 drafts/clustering 前後跑 `tests/unit`。
- SLURM 輪詢 ≥30s；sbatch 一律走 `run_watchdog.sh`，不再複製 script。

## 3. 工作分解

### WS-A：基礎設施小改

**A1｜Random-prune**：`drafts/random_mask.py` 加 `num_keep: int = 1`——
mask 在 `prepare` 抽一次（**一次/run；注意不能在 `prepopulate` 抽，那是每題都被
`controller.reset()` 呼叫的**），`prepopulate` 每題把同一組 mask 塞回 draft_cache，
`refresh` 改 no-op。既有「每 cycle 重抽單顆」行為留 `per_cycle: true` 旗標保底（預設 false）。
Qwen3 `num_keep:16`、GPT-OSS `num_keep:4`、Mixtral 沿用舊值不重跑。
masked forward（qwen3/mixtral 已有，gptoss 見 G1）把非 kept logits 填 -inf → 在 kept 內 top-k。
單元測試：恰 num_keep 顆、seed 可重現、refresh 後不變。

**A2｜Random-merge**：新 `drafts/random_merge.py`（registry `random_merge`）——
`prepare` 時把**全部 n 顆** expert 隨機均分成 K 群（seed 固定）、群內 uniform 合併
（`_build_one`），組成 "multi" cache（`indices` 覆蓋全 n 顆，`_route_multi_expert`
gate-remap 路由），**整個 run 固定**；`prepopulate` 每題塞回同一份 cache，
`refresh`/`capture` no-op。
args：`{K, draft_top_k, seed}`；`holds_merged_residency = True`。
（注意：既有 `q5_512_tm_rand_uniform.yaml` 是 count 選 top-M + 隨機分群，屬 dynamic 選集，語義不符，不要拿來充數。）

**A3｜非投機模式（MoE-Caching 用）**：
- `cli.py` 支援 `draft.name: none`——不建 Controller/不裝 forward/不進 phase patch；
  offload 載入照舊（`moe._configure_hook` 仍接）；budget 走 `wants_merged=False` 分支（pool 拿滿 usable）。
- `run_specbench` 接受 `assistant_model=None` → 純 target generate、跳過 assist patch；TPS 照常，MAT/AccR 填 N/A。
- MoE-Caching 實作 = archer 引擎的 LFU expert cache（`FindExpertEvict` 按 `incache_visit_count`）
  吃同一個 `vram_budget_ratio: 0.2`。論文寫法：frequency-based (LFU) expert cache under the same total GPU budget。

**A4｜HumanEval 整合（Coding 欄）**
- 資料：`openai_humaneval`（164 題），下載一次快取 `data/humaneval/`；由 loader 的
  seeded shuffle 抽 80（`questions_per_cat: 80` 機制沿用）。
- `specbench.py`：HumanEval 映成 `{question_id: task_id, category: "humaneval",
  turns: [prompt]}` 併入題庫；`SPEC_BENCH_SUBTASKS` 加 `humaneval`（排 overall 前）。
- **類別名必須用 `humaneval`，不能用 `coding`**——mt_bench 有同名子類（10 題），
  `_category_matches` 會讓兩者互相污染聚合（已驗證本地資料：480 題、六 subtask 各 80、
  mt_bench = 8 子類 × 10，其一即 `coding`）。表頭 Coding 欄 = `humaneval` subtask。
- **prompt 用 raw completion，不套 chat/vicuna template**（HumanEval 是 code-completion
  前綴，Base model 直接續寫）——在 `_format_chat_prompt` 加 per-category 分支，
  既有六類行為不變。mnt 與其他 subtask 相同；只量 acceptance/TPS，不評 pass@1。
- Overall 欄：runner 的 `overall` 是 cycle-pooled，表格的 Overall 定義為**七 subtask
  等權平均**（tex provenance 已載明）→ 填表時由 per-subtask 值離線計算，不改 runner。
- 已填列的補測：A4 落地後，新 run 自動含 humaneval；**已沿用舊值的列**
  （Qwen3 SpecMoE/Ours、Mixtral Random×2/SpecMoE/Ours）用
  `skip_categories: [mt_bench, translation, summarization, qa, math_reasoning, rag]`
  跑 humaneval-only 補齊該格（若 §4 設定對齊判定需全面重跑，則隨重跑涵蓋）。

### WS-B：calibration 收集（HC-SMoE / Enumerate 共用）

**B1｜`scripts/collect_calibration.py`**：C4、32 條 × 2048 tokens。hf 載 target，對每個
MoE block 掛 pre-forward hook 存（a）block 輸入 hidden states（每層抽樣 ≤4k tokens）、
（b）router logits。離線把 n 顆 expert 權重堆疊、`bmm_swiglu` 一次算 `[n, T, D]`
全 expert 輸出（Qwen3/Mixtral；gptoss 走 G3 的 fused 路徑），導出：
HC-SMoE 的 `o_j = mean_t E_j(x_t)` + frequency；Enumerate 的全 expert 輸出快取。逐層落盤。

### WS-C：HC-SMoE（merge–static）

**C1｜`scripts/build_hc_smoe.py`**：讀 `o_j`，每層 average-linkage 階層分群（Euclidean，
確定性）到 K 群（Qwen3 128→16、GPT-OSS 32→4、Mixtral 8→1 退化單群照跑）。
輸出 `output/hc_smoe/<model>.json`：每層 `{groups, freq}`。

**C2｜`drafts/static_merge.py`**（registry `static_merge`）：args `{spec_path, draft_top_k}`。
`prepare` 讀 json、每層一次建 K 顆 merged（**frequency-weighted**，HC-SMoE 原文的 merging），
`prepopulate` 塞 draft_cache；`refresh`/`capture` no-op，全程凍結。
**必須實作 `lazy_build`（回傳凍結 cache）**——compile warmup 的 generate 發生在第一次
`controller.reset()`/prepopulate 之前，沒有 lazy_build 會被 C-BOOT fail-fast 殺掉
（random_merge 在 job 259444 踩過，同型坑）。
`indices` 覆蓋全 n 顆 → 與 Ours 共用 `_route_multi_expert`（同 kernel 同路由，只差分群與凍結）。
單元測試：群覆蓋全部、不重疊、群數 = K、freq 歸一。

### WS-C′：MC-SMoE（merge–static；2026-07-16 新增，tex 同日加列）

M-SMoE（Li et al. ICLR 2024）的 **merging 階段**（low-rank 壓縮與 KD 微調在
expert-count 預算的 zero-shot 設定下無對應物，不實作）。吃 B1 既有產物
（freq + router_logits），**不需要新 calibration**。

**C3｜`scripts/build_mc_smoe.py`** + `clustering/mc_smoe.py`：dominant expert =
adaptive layer-wise ratio（層內 max 正規化 freq、全域 top L·K，每層 ≥1 保證；
**每層群數不固定，K 是平均**）；非 dominant 依 router-logits cosine（M-SMoE Eq.1，
用 B1 的 `layer_<li>.pt` router_logits）靠攏最相似 dominant。輸出
`output/mc_smoe/<tag>_K<K>.json`：每層 `{groups, dominant, freq}`
（= HC-SMoE spec + 每群 dominant）。純 CPU 秒級。

**C4｜`drafts/mc_smoe.py`**（registry `mc_smoe`）：繼承 `StaticMergeDraft`
（static_merge 抽出 `_merge_group` hook，行為不變、tests 綠），唯一差異 =
合併前把每個群員 **permutation-align 到該群 dominant**（Hungarian weight
matching，SwiGLU 三矩陣 Frobenius 內積 cost；`prepare()` 一次性 ~5.4k 次
768×768 LSA，幾分鐘）。SwiGLU 家族限定（qwen3/mixtral）；gptoss fused 版
raise NotImplementedError。args 同 static_merge：`{spec_path, draft_top_k}`。

### WS-D：Enumerate（prune–static，NAEE）

**D1｜`scripts/search_naee.py`**：讀 B1 快取。Mixtral 精確枚舉；Qwen3 抽樣 10⁵ 個
16-子集（seed 固定）；GPT-OSS 精確枚舉 35,960/層。候選評分不重跑模型：kept logits 上
重做路由（各家族語義，見 G4）+ gather 快取輸出加權和，對原始層輸出算 Frobenius
（NAEE Eq.3），GPU 逐層做。輸出 `output/naee/<model>.json`。

**D2｜`drafts/static_mask.py`**（registry `static_mask`）：args `{spec_path}`。
`prepopulate` 把每層 kept-set 轉 bool mask 塞 draft_cache；`refresh` no-op；
`cache_kind="masked"` 走現有 masked forward。約 40 行。

### WS-G：GPT-OSS hf-backend 支援

只跑 acceptance（hf），gpt-oss-20b bf16 ≈42GB。
**路由語義：gpt-oss 是 softmax-after-topk**（先 topk 再對 topk 值 softmax），與 qwen3 相反——G1/G2/G4 照 gptoss adapter 現有 target 分支寫，別抄 qwen3。

- **G0｜smoke（最先做）**：確認 transformers 支援 `gpt_oss`（pyproject ≥4.55）+
  現有「Ours」路徑直接跑 1–2 題（averaged forward 的 multi 分支與 fused
  `build_weighted_avg` 已實作；`_swiglu_stack` 回 None → per-cluster 泛用迴圈，正確稍慢）。
  `random_merge` 順帶 smoke。
- **G1｜`gptoss.make_masked_forward`**（解鎖 Random-prune、Enumerate）：router logits →
  非 kept 填 -inf → topk → softmax-after-topk → scatter → `mlp.experts(...)`；target 相位照常 `capture_softmax`。
- **G2｜gptoss 的 SpecMoE**：`expert_flat_weights`（fused 3D tensor 切 per-expert + bias 攤平）
  ＋ `make_substitute_forward`（full softmax 給 capture、topk winner 經 substitute table remap、
  重建 router_scores 走 `mlp.experts`；offload 分支一律不進）。args `{N:4, route_top_k:4}`。
- **G3｜B1 的 gptoss 路徑**：全 expert 輸出用 fused bmm（`gate_up_proj` + bias → clamp/GLU →
  `down_proj` + bias，照 `_run_dense_expert` 向量化）。
- **G4｜D1 的 gptoss 路徑**：kept-子集重路由用 softmax-after-topk。

### WS-F：收尾

數字 + provenance（config 名、job id、seed、日期、backend）寫進 `tab_main_baselines.tex`；
更新 `PROJECT_GUIDE.md`（新 draft 名、新 script）；本文件 §7 打勾；`pytest tests/unit` 全綠。

## 4. 跑數矩陣與設定對齊

**先做設定對齊**：讀 `tab_main_baselines.tex` provenance 確認既有數字的確切設定
（caption 是 **T=5**、full 80 q/subtask、mnt 固定、MT-Bench first-turn-only）。所有新列
沿用完全相同設定；若發現既有數字設定不一致（如 T=3 舊 run），**先回報再決定**。

| 表格列 | 模型 | draft | 備註 |
|---|---|---|---|
| Random (prune) | Qwen3 / GPT-OSS | `random_mask{num_keep:16 / 4}` | A1；固定集合 |
| Enumerate | Qwen3 / Mixtral / GPT-OSS | `static_mask` | D1+D2；Qwen3 sampled |
| Random (merge) | Qwen3 / GPT-OSS | `random_merge{K:16 / 4}` | A2；固定分割 |
| HC-SMoE | Qwen3 / Mixtral / GPT-OSS | `static_merge{K:16 / 1 / 4}` | C1+C2 |
| MC-SMoE | Qwen3（Mixtral/GPT-OSS 待議） | `mc_smoe{spec_path}` | C3+C4；gptoss 需另解 fused 對齊 |
| SpecMoE | GPT-OSS | `specmoe{N:4, route_top_k:4}` | G2；Qwen3/Mixtral 已有值 |
| Ours | GPT-OSS | `topm_count{M:8, K:4, draft_top_k:4}` | G0；M=2K 慣例比照 Qwen3；Qwen3/Mixtral 已有值 |
| MoE-Caching（Table 2） | Qwen3 | `draft: none`（offload） | A3；n=3 |
| humaneval-only 補跑（Table 1，hf） | Qwen3 SpecMoE/Ours、Mixtral Random×2/SpecMoE/Ours | 各自原 draft + skip 六類 | A4；補沿用舊值列的 Coding 格 |
| humaneval-only 補跑（Table 2，offload） | Qwen3 SpecMoE/Ours | 各自原 offload config + skip 六類 | 補 Table 2 的 Coding TPS 格；MoE-Caching 的新 run 已自動涵蓋 |

全部 acceptance 走 hf backend（A4 落地後自動含 humaneval）；Mixtral 其餘列的六類沿用舊值（時間剩餘才考慮重跑）。
**重複次數**：hf acceptance n=1 起跳，與 Ours 差距 <3pp 的對手補到 n=3；MoE-Caching TPS n=3。
**新 config** 命名 `t1_<model>_<method>.yaml` / `t2_qwen3_moecaching.yaml`，檔頭註明對應哪列。

## 5. 時程（倒推自 ~07-25）

| 天 | 內容 |
|---|---|
| D1–2 | A1、A2、A3、A4、C2、D2 程式 + 測試；**G0 smoke（最先，環境風險早爆）**；B1 寫好並**送出 calibration job** |
| D3–4 | C1 分群 + D1 搜尋（吃 B1 產物）；G1/G2/G3/G4；送 Qwen3 random ×2 acceptance（不依賴 B1） |
| D5–7 | 送 HC-SMoE / Enumerate（三模型）+ GPT-OSS 其餘列 + humaneval-only 補跑 ×6 + MoE-Caching TPS；Mixtral hf smoke |
| D8–10 | 收數、貼表、close-call 補 n=3 |
| D11–12 | tex 定稿 + provenance + PROJECT_GUIDE |
| D13–14 | 緩衝 |

原則：能排隊的 job 越早送越好；緩衝天不排新工作。

## 6. 風險

1. **GPU 排隊**（最大）：D1 就送不依賴新程式的 job。
2. **Mixtral 93GB 單卡**：先 5 分鐘 smoke；塞不下退 `device_map=auto`（慢但 acceptance 有效），記 provenance。
3. **GPT-OSS 環境**（transformers `gpt_oss` 支援、MXFP4→bf16 載入）：G0 排最先。
4. **既有數字設定不一致**（T=5 vs 舊 run）：§4 對齊最先做，發現就回報。
5. **C4 下載**：login node 可外網；快取進 `/work`。
6. **calibration 記憶體**：每層 ≤4k tokens、逐層落盤。

## 7. 狀態追蹤

| 項目 | 狀態 | 備註（config/job id） |
|---|---|---|
| 設定對齊（provenance 確認 T/mnt/qpc） | ⬜ | |
| A1 random_mask num_keep（固定集合）+ 測試 | ✅ 2026-07-11 | GPU smoke ✅ job 259444（humaneval 列正常） |
| A2 random_merge（固定分割）+ 測試 | ✅ 2026-07-11 | job 259444 warmup 撞 C-BOOT fail-fast → 補 `lazy_build` 修復（凍結 draft 通用教訓，C2 同款）；resmoke job 259767 |
| A3 draft:none 非投機模式 | ✅ 2026-07-11 | GPU smoke ✅ job 259444：pool 拿滿 usable、MAT/AccR=0、TPS 有效 |
| A4 HumanEval 整合（loader / `humaneval` subtask / raw-prompt 分支） | ✅ 2026-07-11 | GPU smoke ✅（三 run 的 humaneval 列皆正常聚合）；資料已預下載 `data/humaneval/` |
| B1 calibration 收集（Qwen3 / Mixtral / GPT-OSS） | 🔶 Qwen3 ✅ | job 259832（259816 因 phase-2 漏 no_grad OOM，已修）；產物 `output/calibration/Qwen3-30B-A3B-Base/`（1.7GB）；Mixtral/GPT-OSS 待送 |
| C1 HC-SMoE 分群產物 | 🔶 Qwen3 ✅ | `output/hc_smoe/Qwen3-30B-A3B-Base_K16.json`。**觀察：128→16 下 average-linkage 退化為「~113 顆巨群 + 15 singleton」/層（資料健康已驗，是方法在 87.5% 壓縮比的真實行為；HC-SMoE 原文最多測 75%）→ 其 acceptance 預期偏弱，論文解讀時要記得這點** |
| C2 static_merge draft + 測試 | ✅ 2026-07-11 | 含 lazy_build（259444 教訓）、freq 加權數值驗證、C1→C2 roundtrip；**GPU smoke ✅ job 259849**（q1 AccR 0.098，巨群預期內） |
| C3 MC-SMoE 分群產物（build_mc_smoe.py + clustering/mc_smoe.py） | 🔶 Qwen3 ✅ 2026-07-16 | `output/mc_smoe/Qwen3-30B-A3B-Base_K16.json`（吃既有 B1 calibration）；adaptive：每層 dominant min 2 / max 30、總 768（=48×16）；群型態比 HC-SMoE 健康（最大群 ~30–80，非 113 巨群） |
| C4 mc_smoe draft（permutation-aligned merge）+ 測試 | ✅ 2026-07-16 | 繼承 static_merge（`_merge_group` hook 抽出，行為不變）；Hungarian 對齊數值驗證（permuted-clone 恆等）+ registry + fail-fast；tests/unit 108 綠；**GPU smoke ✅ job 265209**（q1/mnt64 全 14 題 rc=0，overall AccR 0.0396 vs HC-SMoE 同協議 smoke 0.098——凝聚式分群偏弱符合 partition A/B 既有結論，正式值看 t1） |
| Qwen3 acceptance：MC-SMoE（t1 協議） | ✅ 2026-07-17 | job 265221（`t1_qwen3_mcsmoe.yaml`）：**mean7 Overall 0.1255**（cells 已填 tex；> Random-merge 0.1026、< HC-SMoE 0.2939——凝聚式分群在 87.5% 壓縮劣勢，與 partition A/B 結論一致） |
| D1 NAEE 搜尋產物（Q3 sampled / Mix+GPT exact） | 🔶 Qwen3 ✅ | job 259883（10⁵ 抽樣、31 分）→ `output/naee/Qwen3-30B-A3B-Base_r16.json`；mean/best loss 比中位數 1.21（搜尋有鑑別力）；Mixtral/GPT-OSS 待 B1 後跑 |
| D2 static_mask draft + 測試 | ✅ 2026-07-12 | 凍結 mask、驗證 fail-fast；走現有 masked forward（259444 smoke 已驗該路徑）；tests/unit 62 綠 |
| G0 GPT-OSS smoke（Ours / random_merge） | 🔶 已送 | jobs 264776（randmerge）/ 264778（ours=hybrid a75）。**執行計劃移至 `gptoss_acceptance_plan.md`（2026-07-16）**：Ours 用戶拍板 hybrid a75 → 新增 hf-backend act-sim 捕捉（原本只有 offload engine 路徑，缺席時 hybrid 靜默退化純 cooccur） |
| G1 gptoss masked forward | ✅ 2026-07-16 | softmax-after-topk on masked logits；tests 91 綠 |
| G2 gptoss substitute forward + flat weights | ✅ 2026-07-16 | `gptoss_substitute_forward`（remap 碰撞 scatter_add_，GPU dense path 語義已核對）+ flat weights 含 bias |
| G3/G4 離線 script 的 gptoss 路徑 | ✅ 2026-07-16 | hook 重算 full logits（gptoss output[1] 是 post-scatter scores）+ fused clamped-GLU 輸出 kernel + NAEE softmax_after_topk；calibration job 264785 已送 |
| Qwen3 acceptance ×4（rand-p / rand-m / enum / hc） | 🔶 跑數中 | jobs 259977–259980；**t1 協議：T=5、qpc=15、mnt=512、humaneval、`mt_bench_pooled: true`（mt_bench 整包 15 題）** |
| Qwen3 acceptance：Ours + SpecMoE 同協議重跑 | 🔶 跑數中 | jobs 259987（**Ours = topm/hybrid a75**，cache mode+C3，沿 f15_hybrid_a75）/ 259988（**SpecMoE ep1**，沿 f15_specmoe_ep1），run 段換成 t1 協議（含 humaneval/pooled，不 skip mt_bench）→ **Table 1 Qwen3 全六列將是同一協議，q5 舊值屆時整批汰換** |
| Mixtral acceptance（全九列，mx1 系列） | 🔶 已送 2026-07-20 | **執行計劃移至 `mixtral_acceptance_plan.md`**：m1 協議（T=3/qpc=all/mnt=512/hf）、九列零新實作；jobs 268210–268224 dependency chain（smoke→full、calib→NAEE/HC/MC→後三列、DV search→DV） |
| ~~GPT-OSS acceptance ×6~~ | ❌ 作廢 2026-07-21 | 用戶定案 GPT-OSS 行不通(merge 在粗粒度 expert 上崩潰,見 tex Table 3);第三模型改 **DeepSeek-MoE-16B**,候選評估(base vs chat)見 `deepseek_plan.md` |
| humaneval-only 補跑：Table 1 ×6（hf）+ Table 2 ×2（offload） | ⬜ | |
| MoE-Caching TPS n=3 | 🔶 r1 ✅ | job 259777（qpc5/mnt512/humaneval）：**overall TPS 7.6991**（vs Ours 3.7172 / SpecMoE 3.4382）。vram 稽核 peak 18.74GB「OVER」屬 audit 常態（歷史 spec run 同線 24.8–31.5GB——MoE-Caching 總用量反而更低，數字有效）。**⚠ batch-1 下 MoE-Caching 快 2×，Table 2 敘事待用戶拍板**；r2/r3 待該決策後再跑 |
| close-call 補 n=3 | ⬜ | |
| tex 填數 + provenance | ⬜ | |
| PROJECT_GUIDE 同步 | ⬜ | |
