# DeepSeek-MoE-16B 導入計劃（取代 GPT-OSS 的 Table 1 第三模型）

> 2026-07-21 建立。**GPT-OSS 已確定不用**（用戶定案「行不通」——merge 在其粗粒度
> expert 上崩潰，見 tab_main_baselines.tex Table 3 消融），Table 1 第三 block 改用
> DeepSeek-MoE-16B。目前階段 = **候選評估**：base vs chat 各跑 Ours/SpecMoE，
> 由 acceptance 決定採用哪個，之後再跑該模型的完整九列（比照 mx1）。

## 1. 模型與 adapter

- 候選：`deepseek-ai/deepseek-moe-16b-base` / `deepseek-ai/deepseek-moe-16b-chat`
  （**trust_remote_code 家族**，model_type=`deepseek`；cli 預設 trust_remote_code=true）。
- 結構：28 層（**layer 0 是 dense MLP**，`first_k_dense_replace=1`）、64 顆
  fine-grained routed experts（top-6、`norm_topk_prob=false`）+ **2 顆 shared
  experts**（合成一個加寬 DeepseekMLP，兩相位都跑、在 draft budget 之外）。
- **`adapters/deepseek.py`（2026-07-21 新增）**，關鍵差異全記在檔頭 docstring：
  gate 是 MoEGate 模組（回傳 topk 非 logits → adapter 以 fp32 `F.linear` 重算
  logits 對齊原味 routing）、MoE block 回傳**單 tensor**（非 qwen3/mixtral 的
  tuple）、shared experts 輸出在兩相位都要加回、**hybrid 的 hf act-sim 捕捉
  行內做在 verify expert loop**（raw expert 輸出本來就算好，捕捉零成本；
  格式同 gptoss `_fired_expert_outputs` 路徑）。hf-only，無 offload 分支。
- 預算：12.5% of 64 = **8**；Ours = `topm_count {M:16, K:8, draft_top_k:6}` +
  hybrid a75（同 m1 method 定義）；SpecMoE = `{N:8, route_top_k:2→6}`
  （native top-6）。
- ⚠️ 風險備忘：deepseek-moe 的 remote code 是 transformers ~4.36 時代寫的，
  在 venv 的 4.57.6 下 cache API 相容性未驗——smoke 若在 `past_key_values` /
  `get_usable_length` 類的呼叫掛掉，解法是 vendor `modeling_deepseek.py`
  進 repo 修 cache 介面後改走本地 class。

## 2. dsm 候選評估（2026-07-21 送出）

協議：T=3、qpc=15、mnt=512、B=1、humaneval、mt_bench_pooled（105 題/run）、hf。
chat 版由 specbench 自動套 chat template（tokenizer 有 template 就用；base 走
Vicuna-style fallback）。

| job | config | 內容 |
|---|---|---|
| 270033 | `scripts/run_dsm_prefetch.sh` | 兩模型權重 sbatch 下載（**下載也不准在 login node**）+ config 欄位 sanity print |
| 270035 | `dsm_smoke_base_ours` → `dsm_base_ours` | afterok:270033，12h |
| 270036 | `dsm_smoke_base_specmoe` → `dsm_base_specmoe` | 同上 |
| 270037 | `dsm_smoke_chat_ours` → `dsm_chat_ours` | 同上 |
| 270038 | `dsm_smoke_chat_specmoe` → `dsm_chat_specmoe` | 同上 |

每個 job = smoke（qpc=1/mnt=64）+ full 同 job 序跑（watchdog 多 config 模式）。
查進度：`squeue` 單次查 + 看 `output/dsm_*/overall_summary.csv`；
`job_log/wd_<jobid>_dsm_*.out` 是即時輸出。
Ours run 的 log 要確認出現 `[act_sim] hf prefill capture engaged` 行——
沒有就是 hybrid 靜默退化成純 cooccur（gptoss 計劃踩過的坑）。

## 3. 狀態與後續

| 項目 | 狀態 | 備註 |
|---|---|---|
| adapter + 註冊 + unit tests | ✅ 2026-07-21 | 126 tests 綠；`get_adapter("deepseek_moe")`/model_type map 驗過 |
| 權重 prefetch | 🔶 已送 | job 270033（base 已部分快取 31G，會續傳） |
| dsm 候選評估 ×4 | ✅ 全完成（2026-07-21） | jobs 270035–038；smoke 全綠、act-sim capture 確認 engage、4.57 下 remote code 無相容性問題 |
| 讀數 + 選模型 | ✅ **用戶拍板選 base**（2026-07-21） | 結果見下表 |
| md1 九列準備（code+configs） | ✅ 2026-07-21 | 見 §4；unit tests 126 綠 |
| md1 九列送出 | 🔶 已送 2026-07-21（用戶解除 HOLD） | 15 jobs：A=273065(smoke5) → 273068–072(full: randmask/specmoe/randmerge/ours/speed)；B=273066(calib) → 273073(artifacts) → 273074(smoke3) → 273075–077(full: enum/hcsmoe/mcsmoe)；C=273067(DV search L3) → 273078(dv smoke) → 273079(dv full)。查進度 `bash scripts/md1_status.sh` |
| md1 讀數 + 填 tex + bold | ✅ 2026-07-24 全部完成 | 九列全填（腳本逐格驗證 = CSV）+ bold + provenance。mean7：Random-p 0.2305 / Speed 0.0009 / DV 0.0000 / Enum 0.2317 / SpecMoE 0.5945 / Random-m 0.1499 / MC 0.6218 / **HC 0.7022** / **Ours 0.6936** |

### ⚠️ 關鍵 finding（2026-07-24，行文必看）

**HC-SMoE（0.7022）在 DeepSeek 整體小勝 Ours（0.6936，差 0.86pp）**——三塊中唯一
Ours 非全勝的 block。bold 分裂：Ours 拿 MT/Summ/Coding、HC 拿 Trans/QA/Math/RAG/
Overall（QA 只差 0.0007）。單跑、差距低於專案的 pp 級噪聲門檻 →
**投稿前建議 hcsmoe+ours 各補 ≥3 重複**；行文方向：DeepSeek 的 64 顆中度特化
expert 讓「靜態 freq 加權全域合併」已近最優，Ours 在 Mixtral（粗粒度）與
Qwen3（細粒度）全勝、在此與最強 static baseline 同水準且**不需 calibration 語料**。
其他發現與他塊同型：深度軸 prune 失效（Speed≈0、DV=0）、Enum(0.2317)≈Random(0.2305)、
MC<HC。

### 候選評估結果（mean7 acceptance，T=3/qpc=15/mnt=512，2026-07-21）

| 模型 | Ours | SpecMoE | Ours−SpecMoE | 總 decode cycles |
|---|---|---|---|---|
| **base** | **0.6970**（MAT 3.08） | 0.5972 | **+10.0pp** | ~17–19k |
| chat | 0.3737（MAT 2.07） | 0.3124 | +6.1pp | ~9–10k（答案短、早停） |

- 兩個變體上 **Ours 都全七欄壓制 SpecMoE**（per-subtask clean sweep）——核心主張不受選擇影響。
- base 絕對值約 chat 的 2×。chat 偏低有協議性成因：套 chat template 後生成明顯偏短
  （cycles 約一半，早出 EOS），而 acceptance 隨 decode 長度上升（count 累積、draft
  狀態需要 warm-up），短答案吃虧；且 chat 對齊後的路由分布也可能較分散。
- 傾向 **base** 的理由：與 Table 1 另兩塊一致（Qwen3-Base、Mixtral base）、decode
  訊號量足、絕對值健康。
| Table 1 GPT-OSS block 替換 | ⬜ | md1 九列跑完後把 tex 的 GPT-OSS 列換成 DeepSeek block（budget 64→8）；Table 3（gptoss merge 消融）去留由用戶決定 |

## 5. 查進度與重送 SOP（md1）

- **查狀態：`bash scripts/md1_status.sh`**（單次 squeue + artifacts 盤點 + 已完成列
  mean7；**不要包迴圈輪詢**）。即時輸出在 `job_log/wd_<jobid>_md1_*.out`。
- **Ours run 必檢**：log 要有 `[act_sim] hf prefill capture engaged`（dsm 已驗，
  但重跑仍要確認）。
- **重送單一 full run**（smoke 已綠就不用再串依賴）：
  `sbatch --job-name=md1_<m> --time=24:00:00 scripts/run_watchdog.sh configs/md1_<m>.yaml`
- **重送 artifacts 鏈**：calib 壞 → 重送 `run_collect_calib.sh deepseek-ai/deepseek-moe-16b-base`，
  再 `sbatch --dependency=afterok:<新id> scripts/run_md1_artifacts.sh`，後續照 §3 補掛。
- **DV search 壞** → `sbatch scripts/run_search_dv.sh deepseek-ai/deepseek-moe-16b-base --num-keep 3`。
- smoke 失敗 → 下游 full 自動 DependencyNeverSatisfied（不浪費 GPU）；先看
  `job_err/` 與 RUNLOG 尾巴再修再重送。
- 整條 chain 重來：`bash scripts/submit_md1_chain.sh`（會送全新 15 jobs，注意別跟
  殘存 pending 重複）。

## 4. md1 九列（configs 與 code 細節）

- **Configs**：`configs/md1_<row>.yaml` ×9 + `md1_smoke_<row>.yaml` ×9（RunConfig
  parse 全過）。協議 = m1（T=3、qpc=all 560 題、mnt=512、B=1、hf）。方法定義：
  randmask `num_keep:8`、specmoe `N:8/route_top_k:6`、enum spec
  `naee/..._r8.json`（**sampled 1e5**，C(64,8)≈4.4e9 不可精確枚舉，表格要標
  sampled）、speed `num_layers:4`（前 4 層 = dense L0 + 3 MoE = 3/27，對齊 DV）、
  dv auto（`draft_verify/..._L3.json`，round(0.125×27)=3）、randmerge `K:8`、
  hcsmoe/mcsmoe spec `..._K8.json`、ours `M16/K8` + hybrid a75。
- **腳本**：`scripts/run_md1_artifacts.sh`（NAEE sampled + HC K8 + MC K8，吃 B1）、
  `scripts/md1_status.sh`（單次查詢）、**`scripts/submit_md1_chain.sh`（一鍵送出
  15 job chain：smoke5→full5、calib→artifacts→smoke3→full3、DV search→smoke→full）**。
- **為 md1 落地的 code 修改（2026-07-21，unit tests 126 綠）**：
  1. skip 回傳約定 adapter 化：`MoEAdapter.mlp_skip_output`（預設 `(zeros,None)`）
     與 `decoder_skip_output`（預設裸 tensor）新 hook；deepseek override 成
     單 tensor zeros / 4.36 tuple（use_cache 時帶 cache pass-through）。
     `draft_verify`、`speed`、`runtime/dv_search` 改走 hook（對既有家族零行為差）。
  2. `mc_smoe._SWIGLU_KEYS` 加 `deepseek_moe`。
  3. `collect_calibration.py`：deepseek 分支——full logits 用 fp32 gate 重算
     （MoEGate 不回 logits）、**儲存的 output 扣掉 shared experts**（NAEE 重建
     的是 routed-only，參考值要一致）、stack keys 併入 qwen3 分支。
  4. `search_naee.py`：`norm_topk_prob` 改同時查 `block.gate`（deepseek 的
     flag 在 gate 上且為 False，舊碼會誤用預設 True）。
