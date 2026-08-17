# Batch-scaling 實驗計劃（batch_scale_plan.md，2026-07-21）

> **目標**：把 Table 2 的三方法（MoE-On-Demand / SpecMoE / Ours）推到 **B=128 / 256 / 512**，
> 看 batch 放大後投機是否更強。每個 (方法, B) 一個 sbatch = **9 個 job**。
> **前置（用戶指示）**：先跑 **Table 2 現行協議的 T=3 版**（SpecMoE、Ours 兩列；
> MoE-On-Demand 非投機無 T，現值沿用）。

## 0. Base configs（用戶已確認 = Table 2 現行數據來源）

| 列 | config | 關鍵設定 |
|---|---|---|
| MoE-On-Demand | `ondemand_ser_q64b64.yaml` | `moe_ondemand` + `serial_dispatch: true`（無 cache＋無 overlap）|
| SpecMoE | `specmoe_nobmm_q64b64.yaml` | N=16、pin、ep1、`merged_backend: dispatch`（bmm-off）、T=5 |
| Ours | `ours_q64b64.yaml` | topm M32/K16 + hybrid a75、cache mode + C3、T=5 |

共同：mnt=512、`batch_by_category: true`（親和批次）、qpc=64。

## 1. 前置：Table 2 的 T=3 版（先跑）

- configs：`specmoe_nobmm_q64b64_t3.yaml`、`ours_q64b64_t3.yaml`（= base 只改 `T: 5→3`）。
- 動機：Table 1 已改 m1/T=3 協議，Table 2 的 spec 列與其對齊；且 m1 顯示 T=3 對 Ours 的
  acceptance 大幅有利（短提案全中率高），TPS 影響待測（cycle 短 → verify 頻率高，
  fetch/compute 比例改變，方向不明——這正是要測的）。
- 若 T=3 TPS ≥ T=5：Table 2 全面改用 T=3（**記得同步改 fig:throughput_bars 的
  pgfplots 座標數據**——2026-07-21 起表格下方有長條圖，數據重複兩處）。

## 2. B-scaling 的 config 生成規則（9 個）

每個 base 各生三版（命名 `<base>_b{128,256,512}.yaml`，label/dir 同名）：
```yaml
questions_per_cat: all      # 每 subtask 全部 80 題
batch_size: 128 | 256 | 512
batch_fill_repeat: true     # ★ 新開關（見 §3）
```
T 版本：**跟隨 §1 的結論**（若 T=3 版更好，B-scaling 直接用 T=3 base；否則 T=5）。

## 3. 新開關 `run.batch_fill_repeat`（需實作，預設 false）

- 語意：`batch_by_category` 下，若某 subtask 的題數 < batch_size，**循環重複**該組題目
  填滿一個 batch：`(group * ceil(B/len(group)))[:B]`。每 subtask = 恰一個 B-wide 純類別 batch。
- 位置：`batch_spec.py` 的分組迭代器（batch_by_category 分支）+ `cli.py` RunConfig 欄位。
- 指標語意：重複題的每個 replica 都算一條序列（throughput 導向；AccR 池化含 replica）。
- 與 AUG_BATCH_REPLICATE 的差別：那是「單題×B」診斷；這是「整組循環填滿」正式功能。

## 4. Watchdog 修正（需實作；現況在大 B 會誤殺）

兩段無輸出空窗：
1. **prefill 迴圈**（B 個序列逐一 prefill，完全不印）——serial B=512 估數小時 → 必死。
2. **cycle 心跳每 25 cycles**——serial 大 B 一個 cycle 分鐘級 → 間隔 >20 分 → STALL 誤殺。

修法（`batch_spec.py` 純輸出，零行為改變）：
- prefill 每 16 題印 `[prefill i/B]`；
- 心跳 25 → **5 cycles**；
- serial 系 job 另設 `STALL_SECONDS=3600` 保險。

## 5. 已知風險（先寫下）

- **B=512 可能 OOM**：KV ≈ 96KB/token；512 條 ×（長 prompt ~1500 + 512 生成 + 髒洞）
  ≈ 70–100GB > H100 80GB。長 prompt 類別（summarization/rag）最危險。
  若 OOM：記錄為「最大可行 batch」資料點，不視為失敗。B=128 (~25GB) 安全、B=256 (~50GB) 緊。
- **serial 版時間**：B 平方級的序列化成本，B=512 估 >20h。
  time 配置：ondemand_ser 128/256/512 = **16h/24h/32h**；specmoe/ours 全 16h。
- **wedge**（offload 隨機 race）：大 B 長跑曝險高；batch loop 有增量 CSV，死了損失有限;
  失敗一律記錄後重送。

## 6. 執行順序與狀態

| # | 項目 | 狀態 |
|---|---|---|
| S1 | T=3 版 Table 2 | ✅ 2026-07-21（270253/254）：T=3 拉高絕對 TPS（Ours 27.68、SpecMoE 27.41）但**把方法區隔壓到噪音內**（TPS 差 +1%、AccR 差 +6.8pp vs T=5 的 +31%/+17.7pp）。**用戶定案：Table 2 維持 T=5**（區隔即故事）；T=3 數字留 output/*_q64b64_t3 當 ablation。tex 曾短暫改 T=3 後已還原（含決策註記）|
| S2 | `batch_fill_repeat` + watchdog 修正 | ✅ 心跳 25→5、prefill 每16題印進度、fill-repeat smoke 過（28 題各×2）、126 tests 綠 |
| S3 | 生 configs | ✅ **T=5 base**（用戶定案）：`{specmoe,ours}_t5_b{128,256,512}` + `ondemand_ser_b{128,256,512}`（無 T）|
| S4 | 送 jobs | 🔵 2026-07-21（重排序：**B=128 全部先跑**，B=256 掛 afterany dependency）：B=128 = ondemand_ser 271365 / specmoe_t5 271366 / ours_t5 271367；B=256（dep 對應 B=128）= 271368 / 271369 / 271370。歷史：B=512 三個依用戶指示取消（271337/350/351）；T=3 base 誤送 6 個已撤；第一輪 271331/334/346/347/363/364 依用戶指示全撤重排 |
| S5 | 彙整 → Table 2 加列或 fig:batch_sweep;同步 fig:throughput_bars | 🔵 B=128 已入 Table 2b(2026-07-22,含 ondemand humaneval 補跑 271603);B=256 OnDemand 52.72/Ours 55.58 ✅、specmoe 271778 重送中(跨 batch state KV 洩漏修法後) |
| S6 | 小 B 端點(2026-07-22 用戶加開):B=1/4/16 × 三方法,qpc=16/mnt=512/T=5 base → 9 jobs 271783-791(命名 {tag}_q16b{B}) | 🔵 B=1 全完成;B=4 specmoe ✅、ondemand 271786 wedge於math batch(部分數據已入表)→重送 272478、ours 271788 跑中;B=16 三個在跑/排隊 |
| S7 | 統一 scaling 表:Table 2+2b 合併成單一 tab:throughput(B=1..256 六 block × 三方法) | ✅ 2026-07-22 全表完成:17/18 列有數據,唯一缺格 = SpecMoE B=256(結構性 INFEASIBLE,見 S8)。關鍵結論:**crossover 在 B≈8-16**(B≤4 naive 贏、B=16 起 Ours 領先,峰值 +14-18% @ B=16-64,B=256 收斂到 +5.4%);B=4/16 od 數據分別來自 273061(_r2 全量)與 271789+273059(_relay 接力合併) |

## S9. RAG@B=16 input/output 掃描（2026-07-25 用戶指示）

- 目的：挑雙邊領先最大的格子（RAG@B=16：+45% vs on-demand、+87% vs SpecMoE），
  以 input/output token 為變數看三方法表現。
- 網格：input cap × max_new_tokens ∈ {32,128,512,1024}²（16 點/方法）。
- 新開關：`run.prompt_token_cap`（>0 = prompt 截為前 N tokens；cli.py + batch_spec.py，126 tests 綠）。
- configs：`ragsweep_{od,sm,ou}_i{in}_o{out}.yaml`（48 個）；每方法一個 sbatch 依序跑 16 個
  config（watchdog 多 config 模式），小 out 先跑。
- jobs（2026-07-25）：od=276710（STALL3600）、sm=276712、ou=276713（STALL5400），皆 16h。
- ✅ 2026-07-25 全 48 格零失敗跑完。圖 = fig:ragsweep（figure*，四聯 panel per input cap，
  x=output log2，三方法線；已入 tab_main_baselines.tex，含 provenance/outlier 註解）。
- 關鍵結論：(1) Ours 的 AccR 隨 output 長度上升（in=32: 0.06→0.54），TPS 隨之爬升，
  o=1024 時比 on-demand 快最多 +105%、比 SpecMoE 快 2.4-3.5×；(2) 交叉點在 o=128~512，
  o=32 時 naive 最快（投機付不起 verify 開銷）；(3) input 變長會侵蝕 serial anchor
  （attention 無 overlap 可藏）但幾乎不動 Ours。
- Outlier：SpecMoE i512/o1024 = 15.70（AccR 0.357，鄰格 ~0.1）——單 batch 異常
  （截斷 context 的退化重複生成特別好接受），照實保留並註記。
