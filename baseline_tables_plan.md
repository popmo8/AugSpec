# 論文主表補齊計劃（Table 1 / Table 2 baselines，2026-07-11 定稿）

> **Deadline：投稿約 2026-07-25**（兩週）。本文件是唯一的執行藍圖；照 §7 狀態表逐項更新。
> 目標：把 `paper/tab_main_baselines.tex` 的 Table 1（acceptance 軸）與 Table 2（system 軸）填完。
> 正式數字一律寫回 `tab_main_baselines.tex` 的表格與 provenance 註解（PROJECT_GUIDE 規定的唯一紀錄處）。
> 相關背景：系統架構見 `ARCHITECTURE.md`；本計劃刻意**不做** `CODE_HEALTH_REVIEW.md` 裡的任何重構。

---

## 1. 已定案的範圍決定（2026-07-11，與用戶確認）

| # | 決定 | 理由 |
|---|---|---|
| S1 | **GPT-OSS 整個從兩張表刪除**（連 caption 的「32→4 for GPT-OSS-20B」一併刪） | offload 引擎完全不支援（不在 `MODEL_MAPPING_NAMES`、融合 3D expert tensor 與 archer 的 per-expert tensor 模型不相容、C++ `MoEMLP` 只實作 Mixtral/DeepSeek 佈局）；兩週內補不完 |
| S2 | **Table 2 caption 的 batch size 64 改成 batch size 1** | HF assisted decoding 只支援 batch=1；表內既有數字（SpecMoE 3.4382 / Ours 3.7172）本來就是 batch-1，provenance 早有「batch-64 重測警告」TODO。這是文字修正，不是實驗修正 |
| S3 | **Enumerate 用抽樣版**（à la HC-SMoE 論文的 O-prune(10⁵)），表格標註 sampled | C(128,16)≈10²⁰ 精確枚舉不可行；NAEE 原文自認只適用 ≤8 experts。Mixtral 8→1 為 C(8,1)=8 可精確枚舉 |
| S4 | **Table 1 的 TPS 欄維持 Qwen3-only**；不移植 Mixtral offload 路徑 | Mixtral adapter 無 offload 分支（`_route_offload` 等只在 qwen3.py）；acceptance 用 hf backend 即可，TPS 欄 Mixtral 本來就是「–」 |
| S5 | Mixtral 的 acceptance 走 **hf backend**（bf16 ≈93GB，單卡 H200 141GB 放得下） | acceptance 與 offload 無關；舊 Mixtral 數字亦源自 hf 路徑 |
| S6 | 新 baseline 全部只需 acceptance（hf backend）；**唯一要新跑的 TPS = MoE-Caching（Qwen3、offload、非投機）** | Table 1 TPS 只有 specmoe/ours 有值（已有）；Table 2 = MoE-Caching/SpecMoE/Ours |

**待用戶確認的兩個小決定**（執行到 A4 前要拍板）：
- D1：Table 2 的 Mixtral 列——建議**刪除**（S4 之下量不到 Mixtral TPS）；替代方案是留「–」。
- D2：兩表的 Coding (HumanEval) 欄——runner 只支援 Spec-Bench，建議**刪欄**；替代方案是全留「–」。

## 2. Guardrails（兩週內絕對不做）

- 不動 C++（`moe_infinity/core/`）——所有新 baseline 都是純 Python。
- 不動既有方法的執行路徑（topm_count / specmoe / merged_cache / offload_merge）——已填的數字不能被影響。改動只能是**新增** draft、**新增** CLI 分支、**新增** script。
- 不做 CODE_HEALTH_REVIEW 的重構項目；不補 P0–P4（deadline 後再說）。
- 改 clustering/drafts 相關核心前後都跑 `tests/unit`（PROJECT_GUIDE 規定）。
- SLURM 輪詢間隔 ≥30s（CLAUDE.md 規則 3；一律用 `run_watchdog.sh`，不要再內聯 watchdog）。

## 3. 工作分解

### WS-A：基礎設施小改（共用，先做）

**A1｜Random-prune draft：`random_mask` 擴成 N 顆**
- `drafts/random_mask.py`：加 `num_keep: int = 1` 參數，mask 從 one-hot 改成 num_keep 顆 true；每 cycle 重抽（沿用現有 refresh 節奏）。`num_keep=1` 行為不變（Mixtral 沿用）。
- Qwen3 config：`{num_experts: 128(自動補), num_keep: 16, seed: 0}`。
- masked forward（qwen3/mixtral 都有）會把非 kept logits 設 -inf → top-8 在 16 顆內路由，語義正確。
- 單元測試：mask 恰 num_keep 顆、seed 固定可重現。

**A2｜Random-merge draft：全隨機分割（無統計）**
- Table 1 的 policy 軸 {random, static, dynamic} 中，random 應指**完全不用 routing 統計**：把全部 n 顆 expert 隨機均分成 K 群、群內 uniform 合併、每題重抽一次。
- 現有 `q5_512_tm_rand_uniform.yaml`（topm_count + cluster random）仍用 count 選 top-M，是「dynamic 選擇+random 分群」，語義不符，不要拿來充數。
- 新 draft `random_merge`（新檔 `drafts/random_merge.py`，仿 UniformDraft）：`prepopulate` 時隨機分割 → 每群 `_build_one`（uniform 權重）→ 組 "multi" cache dict（`indices` 覆蓋全部 n 顆，gate remap 由現有 `_route_multi_expert` 處理）；`refresh` no-op。args：`{K: 16, draft_top_k: 8, seed}`。`holds_merged_residency = True`。
- Mixtral 的 Random-merge 舊值 0.1532 已在表上，可不重跑（見 §4 設定對齊）。

**A3｜非投機模式（MoE-Caching 用）**
- `cli.py`：支援 `draft.name: none`——不建 Controller、不裝 forward、不進 phase patch；offload 載入照舊（`moe._configure_hook` 仍要接）。budget 走現有 `wants_merged=False` 分支（merged reserve=0，archer pool 拿滿 usable）。
- `runtime/specbench.py`：`run_specbench` 接受 `assistant_model=None` → `generate` 不帶 assistant、跳過 `_locked_assist_patch`；per-question TPS 照常（new_tokens/wall），MAT/AccR 欄填 N/A。
- MoE-Caching 的「cache」實作＝archer 引擎本身的 LFU expert cache（`FindExpertEvict` 按 `incache_visit_count`），預算 = 同一個 `vram_budget_ratio: 0.2`——即「同總預算、全部給 target-side cache、無 draft」，正是 Table 2 caption 的定義。論文寫法：MoE-Caching implemented as the engine's frequency-based (LFU) expert cache under the same total GPU budget。
- （可選強化，時間允許才做）用 profiled hotness `set_pinned` 釘 top-10% 熱門 expert，更貼近 SpecMoE 原文——非必要，LFU 動態快取本身已是同等或更強的 caching baseline。

**A4｜tex 修改**
- 刪 GPT-OSS 區塊（兩表 + caption 文字）；Table 2 caption batch 64→1；Enumerate 標註 "(sampled)"＋footnote 說明 10⁵ 抽樣；依 D1/D2 處理 Mixtral 列與 Coding 欄；provenance 註解同步（每列標來源 config + job id）。

### WS-B：離線統計收集（HC-SMoE 與 Enumerate 共用）

**B1｜calibration 收集 script（新 `scripts/collect_calibration.py`）**
- Calibration set：C4，32 條 × 2048 tokens（兩篇 baseline 論文的共同慣例）。
- hf backend 載入 target（Qwen3 61GB / Mixtral 93GB，單卡可跑），對每個 MoE block 掛 pre-forward hook 存：(a) 進 block 的 hidden states（每層抽樣 ~2–4k tokens 即可）、(b) router logits（供 frequency 與 Enumerate 的 routing 重算）。存成每層一個 .pt。
- 離線每層算全 expert 輸出：把 n 顆 expert 權重用 `kernels/bmm.py:stack_swiglu_weights` 堆疊、`bmm_swiglu` 一次算出 `[n, T, D]`（Qwen3：128×3×2048×768 權重 ≈2.4GB，單卡輕鬆）。由此導出：
  - HC-SMoE 用：`o_j = mean_t E_j(x_t)`（HC-SMoE Eq.4 的逐 token 全 expert 期望）＋ frequency（top-k 命中計數）。
  - Enumerate 用：cache 住 `[n, T, D]` 全 expert 輸出 + router logits + 原始 layer 輸出（逐層處理控記憶體）。

### WS-C：HC-SMoE baseline（merge–static）

**C1｜離線分群 script（新 `scripts/build_hc_smoe.py`）**
- 讀 B1 的 `o_j`，每層做 **average-linkage 階層分群**（Euclidean；scipy `linkage/fcluster` 或 ~40 行純 Python，確定性演算法）到 K 群（Qwen3 128→16；Mixtral 8→1，退化為單群、仍照演算法跑）。
- 輸出 `output/hc_smoe/<model>.json`：每層 `{groups: [[ids]...], freq: [n]}`。

**C2｜靜態 merged draft（新 `drafts/static_merge.py`，registry key `static_merge`）**
- args：`{spec_path: <json>, draft_top_k: 8}`。`prepare()` 讀 json、對每層一次性建 K 顆 merged（`_build_one`，**frequency-weighted**，權重來自 calibration freq——HC-SMoE 原文的 merging 策略），存成 "multi" cache；`prepopulate` 塞進 draft_cache；`refresh`/`capture` 全 no-op（全程凍結，跨題不變）。
- `indices` 覆蓋全部 n 顆 → gate remap 與我方共用同一條 `_route_multi_expert`，比較公平（同 kernel、同路由機制，只差分群與凍結）。
- `holds_merged_residency = True`；hf backend 跑 acceptance 就好，不需 offload。
- 單元測試：群覆蓋全部 expert、不重疊、群數 = K；freq 權重歸一。

### WS-D：Enumerate baseline（prune–static，NAEE sampled）

**D1｜離線搜尋 script（新 `scripts/search_naee.py`）**
- 讀 B1 快取。每層：
  - Mixtral（8→1）：**精確枚舉** 8 個候選。
  - Qwen3（128→16）：抽樣 S=10⁵ 個 16-子集（seed 固定）。
- 候選評分不用重跑模型：對每個 token，kept 子集的輸出 = 在 kept logits 上重做 top-k softmax + 從快取的 `[n,T,D]` gather 加權和；與原始層輸出的 Frobenius 距離即 reconstruction loss（NAEE Eq.3）。逐層在 GPU 上做，10⁵ 候選 × 48 層估數小時等級。
- 輸出 `output/naee/<model>.json`：每層 kept ids。

**D2｜靜態 mask draft（新 `drafts/static_mask.py`，registry key `static_mask`）**
- args：`{spec_path: <json>}`。`prepopulate` 把每層 kept-set 轉成 bool mask 塞 draft_cache；`refresh` no-op。`cache_kind="masked"` → 走現有 masked forward（-inf 非 kept → top-8 在 16 顆內）。約 40 行。

### WS-E：跑數（見 §4 矩陣）

### WS-F：收尾
- 全部數字寫進 `tab_main_baselines.tex` + provenance（config 名 + job id + 日期 + backend）。
- 更新 `PROJECT_GUIDE.md`（新 draft 名、新 script、GPT-OSS 出表的決定）；本文件狀態表打勾。
- `pytest tests/unit` 全綠（新增測試含在內）。

## 4. 跑數矩陣與設定對齊

**先做設定對齊（跑任何新數前）**：開 `paper/tab_main_baselines.tex` 的 provenance 註解，確認既有 Qwen3/Mixtral 數字的確切設定（**T=5**（caption）、mnt、questions_per_cat、seed、skip_categories、MT-Bench first-turn-only）。**所有新列必須沿用完全相同設定**，否則不可同表比較（PROJECT_GUIDE：mnt 不同不可橫比）。若既有數字實為 T=3 等不一致，先回報再決定（重跑全表 vs 改 caption）。

| 表格列 | 模型 | draft/mode | backend | 新跑? | 備註 |
|---|---|---|---|---|---|
| Random (prune) | Qwen3 | `random_mask{num_keep:16}` | hf | ✅ | A1 |
| Enumerate | Qwen3 | `static_mask` | hf | ✅ | D1+D2，sampled |
| SpecMoE | Qwen3 | 既有 | — | ❌ 已有 | 0.3801/3.4382 |
| Random (merge) | Qwen3 | `random_merge{K:16}` | hf | ✅ | A2 |
| HC-SMoE | Qwen3 | `static_merge` | hf | ✅ | C1+C2 |
| Ours | Qwen3 | 既有 | — | ❌ 已有 | 0.5829/3.7172 |
| Enumerate | Mixtral | `static_mask` | hf | ✅ | 精確枚舉 |
| HC-SMoE | Mixtral | `static_merge{K:1}` | hf | ✅ | 128GB 級載入，先 smoke 確認塞得下 |
| 其餘 Mixtral 列 | Mixtral | — | — | ❌ 舊值沿用 | 若審稿風險在意一致性，時間剩餘才重跑 |
| MoE-Caching | Qwen3 | `draft: none` | **offload** | ✅ | A3；Table 2 |

**重複次數**：hf-backend acceptance 跑法穩定，n=1 起跳；**與 Ours 差距 <3pp 的對手（預期是 HC-SMoE）加跑到 n=3 取平均**。MoE-Caching TPS 在 offload 上（非確定性），n=3。

**新 config**：每列一個 YAML（命名 `t1_<model>_<method>.yaml` / `t2_qwen3_moecaching.yaml`），開頭註解寫「對應 Table 幾哪一列」；sbatch 一律 `scripts/run_watchdog.sh <configs...>`，不要新增複製貼上的 script。

## 5. 時程（倒推自 ~07-25）

| 天 | 內容 |
|---|---|
| D1–2（~07/12-13） | A1、A2、A3、D2、C2 的程式 + 單元測試；A4 的 tex 先改 caption/刪 GPT-OSS（不等數字）；B1 收集 script 寫好、**送出 calibration 收集 job** |
| D3–4 | C1 分群 + D1 搜尋（離線，吃 B1 產物）；同時送 Qwen3 random-prune / random-merge 的 acceptance job（不依賴 B1） |
| D5–7 | 送 HC-SMoE / Enumerate（Qwen3、Mixtral）acceptance jobs + MoE-Caching TPS jobs；Mixtral hf smoke |
| D8–10 | 收數、貼表、close-call 補 n=3、MoE-Caching n=3 |
| D11–12 | tex 定稿 + provenance + PROJECT_GUIDE 更新 |
| D13–14 | **緩衝**（GPU 排隊、重跑、審稿人視角自查） |

原則：**能排隊的 job 越早送越好**，程式開發與排隊並行；緩衝天不排新工作。

## 6. 風險與對策

1. **GPU 排隊**（最大風險）：D1 就把不依賴新程式的 job 送出；acceptance 跑 hf 較快，優先塞。
2. **Mixtral 93GB 是否塞得下該卡**：先跑 5 分鐘 smoke（load + 1 題）確認；塞不下退 `device_map=auto`（慢但 acceptance 有效）並記錄在 provenance。
3. **既有表格數字的設定不一致**（T=5 vs 舊 run 的 T）：§4 對齊步驟最先做；發現不一致立即回報用戶決策，不要悶著頭跑。
4. **HC-SMoE Mixtral 8→1 退化情形**（單群 = 全 8 顆 freq 加權合併）：演算法上合法，但跟 Random-merge 8→1 只差權重；結果若難看不是 bug。
5. **C4 下載**：login node 可外網（HF 模型都這樣抓的）；先確認 `datasets`/直抓 raw 檔皆可，快取進 `/work`。
6. **calibration capture 記憶體**：每層抽樣 token 數上限 4k、逐層落盤，不整模型駐留。

## 7. 狀態追蹤

| 項目 | 狀態 | 備註（config/job id） |
|---|---|---|
| 設定對齊（讀 provenance 確認 T/mnt/qpc） | ⬜ | |
| A1 random_mask num_keep + 測試 | ⬜ | |
| A2 random_merge draft + 測試 | ⬜ | |
| A3 draft:none 非投機模式 | ⬜ | |
| A4 tex：刪 GPT-OSS / batch 1 / sampled 標註 | ⬜ | |
| D1/D2 決策（Table 2 Mixtral 列、Coding 欄） | ⬜ | 問用戶 |
| B1 calibration 收集（Qwen3 / Mixtral） | ⬜ | |
| C1 HC-SMoE 分群產物 | ⬜ | |
| C2 static_merge draft + 測試 | ⬜ | |
| D1 NAEE 搜尋產物（Qwen3 sampled / Mixtral exact） | ⬜ | |
| D2 static_mask draft + 測試 | ⬜ | |
| Qwen3 acceptance ×4（rand-p / rand-m / enum / hc） | ⬜ | |
| Mixtral acceptance ×2（enum / hc） | ⬜ | |
| MoE-Caching TPS n=3 | ⬜ | |
| close-call n=3 補跑 | ⬜ | |
| tex 填數 + provenance | ⬜ | |
| PROJECT_GUIDE 同步 | ⬜ | |
