# 專案運作快速上手（新 session：先讀這份，不用重新摸索）

> 這份是 aug_spec 專案的環境 + 跑實驗 know-how。CLAUDE.md 只放硬性規則,操作細節都在這。

## 這個專案在做什麼
- 主題:**merged-expert speculative decoding for offloaded MoE inference**(論文題目 *Speculative MoE: Memory-Bounded Expert Merging*)。
- 核心主張:把每層 MoE 的 top-M experts 用 **count-weighted 線性合併**成 K 顆「merged expert」當 draft(全 resident、deterministic),acceptance 贏過 baseline **SpecMoE** 且更省 VRAM。
- Target model:`Qwen/Qwen3-30B-A3B-Base`(128 experts, top-8);另支援 Mixtral-8x7B、gpt-oss。
- 我方 draft = `topm_count`(K=16);baseline draft = `specmoe`。

## 專案位置與重要檔案
- **專案根目錄:`/work/morrisliu07/aug_spec`**(舊碼在 `thesis_experiment/`,不要動)。
- Python 套件:`src/aug_spec/`
  - `cli.py` — 入口(`python -m aug_spec.cli run --config ...`);`RunConfig.from_yaml` 解析 config;`_dump_profile` 印 profiling。
  - `controller.py` — 把 adapter × draft 接到 model;install/uninstall forward。
  - `adapters/` — 模型家族(qwen3 / mixtral / gptoss);`base.py` 有共用 forward + bmm helper + specmoe substitute forward。
  - `drafts/` — draft 策略(registry);`topm_count.py`(我方)、`specmoe.py`(baseline)、`base.py`(`ScoreBasedAvgDraft` = merge/cluster 核心,`_cluster_and_build` 在這)。
  - `runtime/` — `loader.py`(load_offload / VRAM 預算)、`specbench.py`(跑 Spec-Bench)、`phase.py`(draft/verify 階段切換)、`offload_merge.py`(merge-during-verify 引擎)、`scorers.py`。
- **C++ 引擎:`moe_infinity/`**(vendored,expert offloading)。核心 `core/parallel/expert_dispatcher.cpp`(fetch/cache/evict/merge)。
  - 改 C++ 後要重編:`cd moe_infinity && <venv> setup.py build_ext --inplace`(改 Python 不用)。
- 計劃文件:`verify_merge_plan.md`(P1–P4 merge-during-verify)、`refactor_and_cooccur_plan.md`(重構 A1–A5 + co-occurrence/cache B1–B3)。

## 怎麼跑實驗(一律用 sbatch,不要在 login node 跑 GPU)
- venv:`/work/morrisliu07/aug_spec/.venv/bin/python`。
- sbatch header 慣例:`--partition=normal2 --account=MST114471 --gpus-per-node=1 --cpus-per-task=8`;log → `/work/morrisliu07/job_log/<name>_%j.log`,err → `/work/morrisliu07/job_err/`。
- module:`ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0`;env:`HF_HOME=/work/morrisliu07/.cache/huggingface`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 一個實驗 = 一個 YAML:`configs/*.yaml`。跑法:`<venv> -m aug_spec.cli run --config configs/X.yaml`。
- offload config 關鍵欄位:`model.backend: offload`、`offload.path: .../moe_infinity/offload_output/Qwen3-30B-A3B-Base`、`offload.vram_budget_ratio: 0.2`(論文預算)、`merge_offload: true`、`merge_during_verify: true`;draft 用 `topm_count` args `{M:32, K:16, draft_top_k:8}`。
- 既有 script 範例可參考:`scripts/run_q5_512_*.sh`(qpc=5、mnt=512),尾段會跑完印 compare 表。

## 讀結果
- 輸出在 `aug_spec/output/<dir>/`:`per_question_summary.csv`、`overall_summary.csv`、`summary.json`。
- 主要指標(per_question_summary.csv 逐題,取平均):`mean_accept_length`(= **MAT**)、`acceptance_rate`(AccR)、`tokens_per_second`(**TPS**)。`overall_summary.csv` 的 `total_cycles` = cycle 數。
- **mnt(max_new_tokens)會影響 MAT/TPS**:acceptance 隨生成長度上升,所以跨 run 比較要固定 mnt(128 與 512 不可橫比)。

## Profiling(找瓶頸用)
- 設 `AUG_PROFILE=1`,結尾會印 per-cycle breakdown:`verify_fetch / draft_fetch / overload_wait / expert_forward / draft_dispatch(bmm) / merge(P3) / evict`。
- ⚠️ 解析 profiling 文字表時:row label 是 `draft_dispatch` 不是 `dispatch`,用「整行第一個 token 完全相等」比對,別用 `\s+dispatch\b`(會配不到 → 誤報 0,踩過)。
- 這些 ms/cyc 是重疊的非加總值(fetch 在獨立 thread 跟 compute overlap),**不能當成相加 = cycle 時間**。

## 實驗用環境變數開關(目前用 env,計劃 A4 會收進 YAML)
- `AUG_NO_OVERLOAD=1` — **重要,topm/specmoe 都應該開**。關掉 moe_infinity「cache 滿時帳外偷塞一個 slot、用完即丟、無視 pin、且驅逐有 race」的 overload 路徑,改走正規 FindExpertEvict。實測:消除 overload_wait、讓 pin 生效(specmoe bmm 才會 engage)、且修掉一個會壓低 MAT 的 verify race。topm +24% TPS、specmoe +33% TPS。
- `AUG_MERGED_BACKEND` — `engine_bmm`(預設,C++ DispatchBmm)/ `dispatch` / `bmm`。
- `AUG_EARLY_PIN` — specmoe 用:0/1/2,verify 時提早 pin 下個 draft 的 kept-N。
- `AUG_CLUSTER_UNIFORM=1` — 實驗開關:K-cluster 的群內合併權重改 uniform(`1/|group|`),slicing 與 cross-cluster mass 仍 frequency。

## 已知關鍵結論(別重新踩)
- 公平對比要兩邊同 engine、同 mnt、同 vram budget、都開 no_overload。
- topm 的 draft 幾乎全程走 bmm(每題第一個 cycle 因 merged 尚未建會 fallback);specmoe 要 kept-N 全 resident(配 no_overload + pin)bmm 才 engage。
- SVD merge 已整包刪除(2026-06-28);**未來 merge 一律線性、只是權重不同,不做 configurable merge strategy**。
- verify-time merge 的 P1–P3 已實作(`merge_during_verify`),P4(overlap)尚未做。
