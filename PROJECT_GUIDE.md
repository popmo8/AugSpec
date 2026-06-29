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
  - `adapters/` — 模型家族(qwen3 / mixtral / gptoss);`base.py` = `MoEAdapter` + 共用 forward。bmm helper 與 SpecMoE substitute forward 已於 A5 移出(見下)。`base.py` 仍持有 `_MERGED_BACKEND` / `_EARLY_PIN` 全域 + A4 的 `apply_offload_settings`。
  - `drafts/` — draft 策略(registry);`topm_count.py`(我方)、`specmoe.py`(baseline draft + A5 搬入的 SpecMoE forward `topk_substitute_forward` / `specmoe_engine_bmm` / `pairwise_l2`)、`base.py`(`ScoreBasedAvgDraft` = merge/cluster 核心;`_cluster_and_build` 在這,改呼叫 `self.cluster_method.assign(ctx, K)`)。
  - `clustering/` — **(A3 新增)** ClusterMethod registry;目前只有 `freq_slice`(原 `_assign_clusters`)。YAML `cluster.name` 選。
  - `merging/` — **(A2 新增)** `linear.py` 單一線性 merge 入口(`_build_one` 委派);非 registry(merge 永遠線性)。
  - `kernels/` — **(A5 新增)** `bmm.py`:SwiGLU 批次 bmm kernel(`stack_swiglu_weights` / `bmm_swiglu`,從 adapters 拆出)。
  - `runtime/` — `loader.py`(load_offload / VRAM 預算;A4 起接受 `no_overload` 並在 `MoE()` 前設環境變數)、`specbench.py`(跑 Spec-Bench)、`phase.py`(draft/verify 階段切換)、`offload_merge.py`(merge-during-verify 引擎)、`scorers.py`。
- **C++ 引擎:`moe_infinity/`**(vendored,expert offloading)。核心 `core/parallel/expert_dispatcher.cpp`(fetch/cache/evict/merge)。
  - 改 C++ 後要重編:`cd moe_infinity && <venv> setup.py build_ext --inplace`(改 Python 不用)。
- 計劃文件:`verify_merge_plan.md`(P1–P4 merge-during-verify)、`refactor_and_cooccur_plan.md`(重構 **A1–A5 已全部完成 2026-06-29** + co-occurrence/cache B1–B3 待做;見 0.5 前置實驗結論)。

## 怎麼跑實驗(一律用 sbatch,不要在 login node 跑 GPU)
- venv:`/work/morrisliu07/aug_spec/.venv/bin/python`。
- sbatch header 慣例:`--partition=normal2 --account=MST114471 --gpus-per-node=1 --cpus-per-task=8`;log → `/work/morrisliu07/job_log/<name>_%j.log`,err → `/work/morrisliu07/job_err/`。
- module:`ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0`;env:`HF_HOME=/work/morrisliu07/.cache/huggingface`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 一個實驗 = 一個 YAML:`configs/*.yaml`。跑法:`<venv> -m aug_spec.cli run --config configs/X.yaml`。
- offload config 關鍵欄位:`model.backend: offload`、`offload.path: .../moe_infinity/offload_output/Qwen3-30B-A3B-Base`、`offload.vram_budget_ratio: 0.2`(論文預算)、`merge_offload: true`、`merge_during_verify: true`、**`offload.no_overload: true`(A4 起在 YAML,取代 `AUG_NO_OVERLOAD=1`)**;draft 用 `topm_count` args `{M:32, K:16, draft_top_k:8}`;**`cluster: {name: freq_slice, within_weight: freq}`(A4 新區塊)**。`configs/q5_512_tm_on.yaml` 是自包含的參考範例;欄位全表見 `configs/README.md`。
- 既有 script 範例可參考:`scripts/run_q5_512_*.sh`(qpc=5、mnt=512),尾段會跑完印 compare 表。**注意:多數舊 script 仍 `export AUG_NO_OVERLOAD=1`(冗餘但無害);新 config 已把它放進 YAML。**

## 讀結果
- 輸出在 `aug_spec/output/<dir>/`:`per_question_summary.csv`、`overall_summary.csv`、`summary.json`。
- 主要指標(per_question_summary.csv 逐題,取平均):`mean_accept_length`(= **MAT**)、`acceptance_rate`(AccR)、`tokens_per_second`(**TPS**)。`overall_summary.csv` 的 `total_cycles` = cycle 數。
- **mnt(max_new_tokens)會影響 MAT/TPS**:acceptance 隨生成長度上升,所以跨 run 比較要固定 mnt(128 與 512 不可橫比)。

## Profiling(找瓶頸用)
- 設 `AUG_PROFILE=1`,結尾會印 per-cycle breakdown:`verify_fetch / draft_fetch / overload_wait / expert_forward / draft_dispatch(bmm) / merge(P3) / evict`。
- ⚠️ 解析 profiling 文字表時:row label 是 `draft_dispatch` 不是 `dispatch`,用「整行第一個 token 完全相等」比對,別用 `\s+dispatch\b`(會配不到 → 誤報 0,踩過)。
- 這些 ms/cyc 是重疊的非加總值(fetch 在獨立 thread 跟 compute overlap),**不能當成相加 = cycle 時間**。

## 旋鈕:YAML 為主,env 為 override(A4 已收編,2026-06-29)
這四個原本是 import/runtime 期讀的 env,A4 後都有 YAML 欄位;**env 仍可 override(env 設了就贏 YAML)**。全表見 `configs/README.md`。
- `offload.no_overload`(env `AUG_NO_OVERLOAD`)— **重要,topm/specmoe 都應該開**。關掉 moe_infinity「cache 滿時帳外偷塞一個 slot、用完即丟、無視 pin、且驅逐有 race」的 overload 路徑,改走正規 FindExpertEvict。實測:消除 overload_wait、讓 pin 生效(specmoe bmm 才會 engage)、且修掉一個會壓低 MAT 的 verify race。topm +24% TPS、specmoe +33% TPS。
- `offload.merged_backend`(env `AUG_MERGED_BACKEND`)— `engine_bmm`(預設,C++ DispatchBmm)/ `dispatch` / `bmm`。
- `draft.early_pin`(env `AUG_EARLY_PIN`)— specmoe 用:0/1/2,verify 時提早 pin 下個 draft 的 kept-N。
- `cluster.within_weight: freq|uniform`(env `AUG_CLUSTER_UNIFORM`)— K-cluster 的群內合併權重;`uniform` = `1/|group|`,slicing 與 cross-cluster mass 仍 frequency。
- **純診斷 env(刻意不進 YAML)**:`AUG_PROFILE`、`AUG_DUMP_CLUSTER_WEIGHTS=<path>`、`AUG_DUMP_ACTIVE_SET=<path>`。
- **已移除**:`AUG_CLUSTER_LABELS`(A3 拿掉的一次性 partition A/B harness)。

## 已知關鍵結論(別重新踩)
- 公平對比要兩邊同 engine、同 mnt、同 vram budget、都開 no_overload。
- topm 的 draft 幾乎全程走 bmm(每題第一個 cycle 因 merged 尚未建會 fallback);specmoe 要 kept-N 全 resident(配 no_overload + pin)bmm 才 engage。
- SVD merge 已整包刪除(2026-06-28);**未來 merge 一律線性、只是權重不同,不做 configurable merge strategy**。
- verify-time merge 的 P1–P3 已實作(`merge_during_verify`),P4(overlap)尚未做。
- **offload 推論是 run-to-run 非確定性的(2026-06-29)**:同 code、同 config 重跑,per-question AccR 平均差 ~0.15(最大 0.68),aggregate AccR 65 題 SD ~0.018。bf16/GPU 微小浮點差 → 早期 accept 翻轉 → 軌跡發散。**含意:不能 bit-exact 比對;小效果(<~3pp)要多跑幾次取平均,大效果才能單跑下結論。**
- **群內 uniform 加權會傷 acceptance(2026-06-29)**:`cluster.within_weight: uniform` vs `freq`,q5 上 acceptance −6.6pp(−11.3%)、cycle +18.7%、TPS −13%。→ **群內預設一律 freq;uniform 只做消融。**
- **co-occurrence 當「併在一起」(must-link)分群是錯方向(2026-06-29)**:partition A/B(只換分群、群內 freq),random −8.6pp、static-freq −17pp、cooccur(平衡)−22pp、cooccur(凝聚)−39pp,**兩個共現變體都輸給隨機**。→ 若要做共現分群,方向是 **cannot-link / 圖切割(把高共現切開)**,且相似度用 cosine 非 lift。這是 B2 的前置結論。
- **重構 A1–A5 全部完成(2026-06-29)**,皆驗證為行為等價(結構等價 + import smoke;bit-exact 因上述非確定性不可用)。Mixtral / GPT-OSS / smoke configs 已退役,專案以 Qwen3 為主。
