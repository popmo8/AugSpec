# Mixtral-8x7B Table 1 acceptance 執行計劃（mx1 系列）

> 2026-07-20 建立。baseline_tables_plan.md S3/S4 的 Mixtral 落地：**九列全部只跑
> hf-backend acceptance（不移植 offload、無 TPS）**。接手 Mixtral 相關工作先讀這份；
> 進度、重送、填表都照本文件 SOP，狀態表要跟現況同步。

## 1. 協議與方法定義

- **協議 = m1 系列（Qwen3 的 Table-1 新協議）**：`T=3`、`questions_per_cat: all`
  （六類 Spec-Bench 各 80 + HumanEval 80 = **560 題/列**）、`mnt=512`、`B=1`、
  `mt_bench_pooled: true`、hf backend（`device_map: auto`，單卡 H200 裝得下，
  舊 run 實測 peak ~98.5GB）。
- 模型：`mistralai/Mixtral-8x7B-v0.1`（base，權重已在 HF cache）。
- 預算：12.5% expert memory = **8→1**（每層 draft 側 1 顆 expert）。
- config 一律 `configs/mx1_<method>.yaml`（smoke 版 `mx1_smoke_<method>.yaml`，
  qpc=1/mnt=64）；輸出 `output/mx1_<method>/`。

| Table 1 列 | draft | 關鍵 args | 8→1 的退化/備註 |
|---|---|---|---|
| Prune/Random | `random_mask` | `num_keep:1, seed:0` | — |
| Prune/Speed | `speed` | `num_layers:4` | 4/32 層 = 12.5%（深度軸） |
| Prune/Draft&Verify | `draft_verify` | （auto） | 12.5% → keep 4/32；讀 `output/draft_verify/Mixtral-8x7B-v0.1_L4.json` |
| Prune/Enumerate | `static_mask` | spec=`output/naee/Mixtral-8x7B-v0.1_r1.json` | C(8,1)=8 → **exact 枚舉**（非 sampled） |
| Prune/SpecMoE | `specmoe` | `N:1, route_top_k:2` | substitute 對全 8 顆 top-2 再映射到 kept-1；pin/early_pin 在 hf inert |
| Merge/Random | `random_merge` | `K:1, seed:0` | K=1 → 全 8 顆 uniform 合併成 1（分割退化，= 舊 AAAI27 草稿的 "Average"） |
| Merge/MC-SMoE | `mc_smoe` | spec=`output/mc_smoe/Mixtral-8x7B-v0.1_K1.json` | 平均 K=1 的 adaptive 分群 + permutation-aligned merge |
| Merge/HC-SMoE | `static_merge` | spec=`output/hc_smoe/Mixtral-8x7B-v0.1_K1.json` | K=1 單群、freq 加權合併全 8 顆 |
| Merge/**Ours** | `topm_count` | `M:2, K:1` | M=2K 慣例；**K=1 走 `drafts/base.py` 的 single-dense 路徑，cluster method 整個被繞過**（config 不放 cluster 區塊；merge 權重 = count 頻率，即論文的 count-weighted linear merge） |

**可行性盤點結論（2026-07-20，逐項驗過 code）：九列零新實作。** mixtral adapter 已有
masked/averaged/substitute forward；`_route_multi_expert` 對 top_k 有 `min(top_k,K)`
clamp；speed/DV 只要求 `.model.layers` 結構（DV 註解明寫 mixtral 的 (zeros,None)
解包相容）；mc_smoe 的 `_SWIGLU_KEYS` 含 mixtral（w1,w3,w2）；transformers 4.57.6 的
MixtralSparseMoeBlock 屬性（gate/experts/top_k/num_experts）與 adapter 相符；
decoder layer 回傳 tensor（speed 的 identity-skip 相容）。

## 2. Job DAG（2026-07-20 全部送出，SLURM dependency 自動串）

```
268210 mx1_smk_a (smoke: randmask+specmoe+randmerge+ours+speed)
  └─afterok→ 268213 mx1_randmask / 268214 mx1_specmoe / 268215 mx1_randmerge
             268216 mx1_ours / 268217 mx1_speed          （full，--time=24h）
268211 mx1_calib (B1 calibration)
  └─afterok→ 268218 mx1_artifacts (NAEE exact r1 + HC K1 + MC K1)
      └─afterok→ 268219 mx1_smk_b (smoke: enum+hcsmoe+mcsmoe)
          └─afterok→ 268220 mx1_enum / 268221 mx1_hcsmoe / 268222 mx1_mcsmoe（full）
268212 mx1_search_dv (DV BO search --num-keep 4)
  └─afterok→ 268223 mx1_smk_dv (smoke) └─afterok→ 268224 mx1_dv（full）
```

- smoke 失敗 → 下游 full 變 DependencyNeverSatisfied 自動不跑（安全）。
- 預估 full run 時長：Qwen3 m1 hf 每列 9–13h；Mixtral active params 較大但
  expert 模組數少（256 vs 6144），抓 24h 上限（partition 上限 2 天）。

## 3. 進度狀態表（接手 session 更新這裡）

| 項目 | 狀態 | 備註 |
|---|---|---|
| smoke ×5（rand-p/specmoe/rand-m/ours/speed） | 🔶 已送 | job 268210 |
| B1 calibration（Mixtral） | 🔶 已送 | job 268211 → `output/calibration/Mixtral-8x7B-v0.1/` |
| artifacts（NAEE r1 / HC K1 / MC K1） | 🔶 已送 | job 268218（吃 268211） |
| DV BO search（L4） | 🔶 已送 | job 268212 → `output/draft_verify/Mixtral-8x7B-v0.1_L4.json` |
| smoke ×3（enum/hc/mc）+ smoke dv | 🔶 已送 | jobs 268219 / 268223 |
| full ×9 | ✅ 全完成（2026-07-21） | jobs 268213–217、268220–222、268224 全 COMPLETED |
| 讀數 + 填 tex | ✅ 9/9（2026-07-21） | 全九列 mean7 已填 tab_main_baselines.tex Mixtral block（腳本逐格核對 = CSV）；Ours 每欄皆最佳、整列 bold 完成；provenance 更新（含 8→1 三個 finding）；檔頂 STALE 警告改為只剩 GPT-OSS pending |

### 最終數據（mean7 overall，2026-07-21）

| 家族 | 方法 | Overall |
|---|---|---|
| Prune | Random | 0.7273 |
| Prune | Speed | 0.0000 |
| Prune | Draft&Verify | 0.0427 |
| Prune | Enumerate | 0.7008 |
| Prune | SpecMoE | 0.7839 |
| Merge | Random | 0.8019 |
| Merge | MC-SMoE | 0.8022 |
| Merge | HC-SMoE | 0.8016 |
| Merge | **Ours** | **0.8327** |

三個 8→1 finding（已寫進 tex provenance 供行文用）：
1. K=1 下所有 merge baseline 退化為單群單 merged expert，Random/MC/HC 幾乎相同（~0.80，只差合併權重）；Ours 靠 top-M=2 只合最熱 2 顆拉開差距（0.833）。
2. Enumerate 0.701 < Random-prune 0.727，**即使精確枚舉**——與 Qwen3「smart static prune 輸給 random」同結論，且非抽樣假象。
3. 深度軸 prune 在 Mixtral 同樣失效：Speed ~0、Draft&Verify 0.043（與 Qwen3 一致）。

## 4. 查進度與重送 SOP

- **查狀態：`bash scripts/mx1_status.sh`**（單次 squeue + 檔案盤點 + 已完成列的
  mean7；符合 SLURM 查詢規範，**不要包進迴圈輪詢**）。
- 看單一 run 的即時輸出：`/work/morrisliu07/job_log/wd_<jobid>_<config名>.out`
  （watchdog 的 RUNLOG）；sbatch stdout 在 `job_log/mx1_*_<jobid>.log`。
- **重送單一 full run**（smoke 已綠就不用再串依賴）：
  `sbatch --job-name=mx1_<m> --time=24:00:00 scripts/run_watchdog.sh configs/mx1_<m>.yaml`
- 重送整條 artifacts 鏈（calibration 壞掉時）：依 §2 順序手動補 `--dependency=afterok:<id>`。
- 超過 24h 被砍：`per_question_summary.csv` 有部分結果可先看；重送前把
  `output/mx1_<m>/` 改名保留（runner 會覆寫），或直接重送覆寫。
- watchdog stall（rc=124，20 分無輸出）：通常是 hf 卡住或 OOM，看 RUNLOG 尾巴
  與 `job_err/`；hf backend 無 offload 死結問題，重送一次不行就要查。

## 5. 讀數與填表

- 每列取 `output/mx1_<m>/overall_summary.csv` 的 per-subtask `acceptance_rate`
  （七格：mt_bench/translation/summarization/qa/math_reasoning/rag/humaneval），
  **Overall = mean7**（與 tex Table 1 慣例一致，不用 runner 的 pooled overall）。
- ✅ 協議已統一（2026-07-21 確認）：tex 的 Qwen3 塊已換成 m1（T=3、qpc=all）新值、
  caption 已改 `T{=}3` 並標注「acceptance is backend-independent」。**Mixtral mx1
  值與 Qwen3 同協議，直接填入同一張表無混協議問題**（先前擔心的 t1/m1 混用已解除）。
  Mixtral 舊 AAAI27 placeholder 已整批汰換。
- 單跑數字：pp 級比較需重複（offload 非確定性不適用於 hf，但 sampling/題序
  效應仍在，<~3pp 差距不要下結論）。

## 6. 決策記錄

- **hf-only**（用戶定案，baseline_tables_plan S3）：不移植 offload，Table 2 維持
  Qwen3-only；Mixtral 無 TPS。
- **協議跟 m1 不跟 t1**（用戶 2026-07-20 指示「method config 參考 m1 系列」）。
- Ours 不放 cluster 區塊：K=1 在 `drafts/base.py` 直接走 single-dense 路徑
  （`if self.K > 1` 才進 `_cluster_and_build`），hybrid/act-sim 對 Mixtral 無作用；
  mixtral adapter 也沒有 hf act-sim 捕捉——若未來要跑 K>1 的 Mixtral hybrid，
  需比照 gptoss_acceptance_plan 補捕捉路徑。
- specmoe 帶 `pin:true`/`early_pin:1` 純粹鏡射 m1_specmoe（hf 上 inert），
  避免與 Qwen3 列的 config 出現無意義 diff。
