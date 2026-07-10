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
  - `clustering/` — **(A3 新增)** ClusterMethod registry,YAML `cluster.name` 選:`freq_slice`(預設,原 `_assign_clusters`)、`random`、`cooccur_pair`、`activation_similarity`、`weight_similarity`、**`hybrid`(2026-07-04 新增:α×prefill-only act-sim(L2) + (1−α)×decode-only cooccur,norm=rank;q15 sweep 用 `cooccur_norm: raw`,code 預設 cosine;見 `hybrid_cluster_plan.md`)**。旋鈕全表見 `configs/README.md` cluster 段。
  - `merging/` — **(A2 新增)** `linear.py` 單一線性 merge 入口(`_build_one` 委派);非 registry(merge 永遠線性)。
  - `kernels/` — **(A5 新增)** `bmm.py`:SwiGLU 批次 bmm kernel(`stack_swiglu_weights` / `bmm_swiglu`,從 adapters 拆出)。
  - `runtime/` — `loader.py`(load_offload / VRAM 預算;A4 起接受 `no_overload` 並在 `MoE()` 前設環境變數)、`specbench.py`(跑 Spec-Bench)、`phase.py`(draft/verify 階段切換)、`offload_merge.py`(merge-during-verify 引擎)、`scorers.py`。
- 論文素材:`paper/method_relation_merge.tex`(2026-07-04,MergeSpec method 的 Expert Relation Map + Relation-Guided Cluster-and-Merge 兩小節 + Algorithm 1;AAAI 格式,變數沿用草稿,混合係數用 **λ**(= code 的 `cluster.alpha`))、`paper/method_reconstruct.tex`(**2026-07-06 大改**:改為「Merge-Aware System Optimization」小節,檔名保留以免斷 \input——內容為 (a) Merged-expert memory hierarchy:GPU K-slots merged cache + CPU 存全部原始 expert weights,retention 為文字規則(先看 pair 兩員是否都在 𝓜_ℓ、再看 co-occur rank;**該小節改寫後無任何公式/label,舊的 eq:retention/eq:overlap 已不存在,引用時用文字**);(b) Fetch-Overlapped Merge Pipelining:verify 逐層 expert 排程,要 merge 且不在 cache 的先 fetch→inference→merge,藏進 transfer-bound 的 fetch 流。**Reconstructive Fetching 已整段刪除**(機制成立但不支撐主論點;留在 reconstruct_fetch_plan.md 當未來工作))、**`paper/tab_main_baselines.tex`(2026-07-08 新增:論文兩張主表合一檔,= 正式 performance 數字的唯一紀錄處**——Table 1 method 軸(acceptance+TPS,{prune,merge}×{random,static,dynamic},baseline 定案:Enumerate/NAEE、HC-SMoE、SpecMoE)+ Table 2 system 軸(MoE-Caching/SpecMoE/Ours throughput);正式數字更新一律改此檔的表格與 provenance 註解,不要散落他處;檔內 provenance 有已知 TODO(bold/cite/batch-64 重測警告)。舊的 `tab_main_acc_tps.tex`/`tab_hybrid_ablation.tex` 是 q15 消融素材,非正式主表)。
- 單元測試:`tests/unit/`(pytest,純 CPU 秒級,login node 可跑:`.venv/bin/python -m pytest tests/unit -q`)。改 clustering/drafts/merging 核心邏輯前後都要跑。`tests/offload/` 是歷史手動 probe,不是測試。
- **C++ 引擎:`moe_infinity/`**(vendored,expert offloading)。核心 `core/parallel/expert_dispatcher.cpp`(fetch/cache/evict/merge)。
  - 改 C++ 後要重編:`cd moe_infinity && <venv> setup.py build_ext --inplace`(改 Python 不用)。
- 計劃文件:`verify_merge_plan.md`(P1–P4 merge-during-verify)、`refactor_and_cooccur_plan.md`(重構 **A1–A5 已全部完成 2026-06-29** + co-occurrence/cache B1–B3 待做;見 0.5 前置實驗結論)、**`ARCHITECTURE_REVIEW_AND_PAPER_PLAN.md`(2026-07-03 新增:全 repo 架構 review + 論文可重現性 P0–P4 roadmap;做 paper/repro/測試/打包相關工作前先讀它,並照它的狀態追蹤表更新進度)**、**`hybrid_cluster_plan.md`(2026-07-03 新增:`cluster.name: hybrid` 的完整實作規格——α 凸組合 prefill-only act-sim(L2) map 與 decode-only cosine-normalized co-occur map,greedy_pair 分群;含相位開關接線、q15 α sweep 實驗計劃)**、**`merged_cache_plan.md`(2026-07-09 新增,取代已作廢的 `memory_hierarchy_plan.md`;同日定案 A′ 架構:所有 expert 權重收斂到單一 C++ pool——merged slot 用合成 id 進 archer(驅逐 = discard 可重建)、singleton 直接 pin 原始 expert 享雙重身分(draft + verify 共用免重抓)、auto 容量管理、零新旋鈕(merged 盡量保留、verify 壓力來 on-demand discard,唯一硬限制是內部 verify floor)、content-addressable + retention(論文文字規則:pair⊆𝓜_ℓ 先保、餘按 co-occur rank)、fetch-overlapped merge pipelining;C++ 只加機制、policy 全留 Python;= 論文 Merge-Aware System Optimization 小節的實作計劃;動 merged cache / pin / dispatcher 相關工作前先讀它)**、**`reconstruct_fetch_plan.md`(2026-07-06 新增:用駐留的 merged C=0.5A+0.5B 做 `B=2C−A` 代數重建、取代 verify fetch——draft cache 反向加速 verify;含 bf16 lossy vs fp32 exact 的數值 trade-off(觸碰 target-exact 主張,V2 是生死關)、sticky clustering β、C++ 重建分支規格)**。

## 怎麼跑實驗(一律用 sbatch,不要在 login node 跑 GPU)
- venv:`/work/morrisliu07/aug_spec/.venv/bin/python`。
- sbatch header 慣例:`--partition=normal2 --account=MST114471 --gpus-per-node=1 --cpus-per-task=8`;log → `/work/morrisliu07/job_log/<name>_%j.log`,err → `/work/morrisliu07/job_err/`。
- module:`ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0`;env:`HF_HOME=/work/morrisliu07/.cache/huggingface`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 一個實驗 = 一個 YAML:`configs/*.yaml`。跑法:`<venv> -m aug_spec.cli run --config configs/X.yaml`。
- offload config 關鍵欄位:`model.backend: offload`、`offload.path: .../moe_infinity/offload_output/Qwen3-30B-A3B-Base`、`offload.vram_budget_ratio: 0.2`(論文預算)、`merge_offload: true`、`merge_during_verify: true`;draft 用 `topm_count` args `{M:32, K:16, draft_top_k:8}`;**`cluster: {name: freq_slice, within_weight: freq}`(A4 新區塊)**。`configs/q5_512_tm_on.yaml` 是自包含的參考範例;欄位全表見 `configs/README.md`。
- 既有 script 範例可參考:`scripts/run_q5_512_*.sh`(qpc=5、mnt=512),尾段會跑完印 compare 表。**注意:舊 script 的 `export AUG_NO_OVERLOAD=1` 與舊 YAML 的 `offload.no_overload` 都已 inert(2026-07-08 起 overload 路徑整個刪除、該行為變成唯一預設;YAML 出現該欄位只印 deprecation 提示,見 `remove_overload_plan.md`)。**

## 讀結果
- 輸出在 `aug_spec/output/<dir>/`:`per_question_summary.csv`、`overall_summary.csv`、`summary.json`。
- 主要指標(per_question_summary.csv 逐題,取平均):`mean_accept_length`(= **MAT**)、`acceptance_rate`(AccR)、`tokens_per_second`(**TPS**)。`overall_summary.csv` 的 `total_cycles` = cycle 數。
- **mnt(max_new_tokens)會影響 MAT/TPS**:acceptance 隨生成長度上升,所以跨 run 比較要固定 mnt(128 與 512 不可橫比)。

## Profiling(找瓶頸用)
- 設 `AUG_PROFILE=1`,結尾會印 per-cycle breakdown:`verify_fetch / draft_fetch / expert_forward / draft_dispatch(bmm) / merge(P3) / evict`(`overload_wait` row 已隨 overload 路徑刪除,2026-07-08)。
- ⚠️ 解析 profiling 文字表時:row label 是 `draft_dispatch` 不是 `dispatch`,用「整行第一個 token 完全相等」比對,別用 `\s+dispatch\b`(會配不到 → 誤報 0,踩過)。
- 這些 ms/cyc 是重疊的非加總值(fetch 在獨立 thread 跟 compute overlap),**不能當成相加 = cycle 時間**。

## 旋鈕:YAML 為主,env 為 override(A4 已收編,2026-06-29)
這些原本是 import/runtime 期讀的 env,A4 後都有 YAML 欄位;**env 仍可 override(env 設了就贏 YAML)**。全表見 `configs/README.md`。
- ~~`offload.no_overload`(env `AUG_NO_OVERLOAD`)~~ — **已刪除旋鈕、行為內建(2026-07-08,`remove_overload_plan.md`)**:moe_infinity「cache 滿時帳外偷塞一個 slot、用完即丟、無視 pin、且驅逐有 race」的 overload 路徑整段移除,pin-aware 的 FindExpertEvict(滿了必 evict、pinned 不可侵犯)成為唯一路徑。歷史實測(當年開 no_overload 的效果):消除 overload_wait、讓 pin 生效(specmoe bmm 才會 engage)、修掉壓低 MAT 的 verify race,topm +24% TPS、specmoe +33% TPS。舊 YAML 的該欄位與 env 均 inert(只印 deprecation 提示)。另有 pinned-starvation guard(2026-07-09):cache 全 pinned 時 fetch 立即 WARN;全 pinned + exec pipeline 排空持續 10s 判定為死結 → FATAL abort(細節見 remove_overload_plan.md §5)。
- `offload.merged_backend`(env `AUG_MERGED_BACKEND`)— `engine_bmm`(預設,C++ DispatchBmm)/ `dispatch` / `bmm`。
- `draft.early_pin`(env `AUG_EARLY_PIN`)— specmoe 用:0/1/2,verify 時提早 pin 下個 draft 的 kept-N。**實測(2026-07-08 §6-0 probe)**:ep1/2 把 draft_fetch 4.5TB→~0.95TB、kept 駐留 75%→99%(bmm 全程 engage)、TPS +22~26%,AccR 不受影響;**specmoe 對照一律開(建議 `early_pin: 2`)**。
- `cluster.within_weight: freq|uniform`(env `AUG_CLUSTER_UNIFORM`)— K-cluster 的群內合併權重;`uniform` = `1/|group|`,slicing 與 cross-cluster mass 仍 frequency。
- **`run.prefill_warmup`(2026-07-09 新增,C-BOOT,無 env)— 預設 true**:每題首輪 candidate 回 0 → target 純 prefill 先建好 draft 狀態(merged/mask/table)**並把 target 的 prompt KV 複製給 assistant(KV-copy,必要配套——否則 draft 會用 merged 重 encode 整個 prompt,context 劣化 → AccR 崩到 0.01,踩過)**,第一次真 draft 直接投機、不走 standard-routing fallback(走到即 raise);`false` = 論文 ablation(舊 draft-先行 + 首 cycle fallback,~47GB/題 draft fetch)。對所有方法一體生效;與 `run.warmup`(compile 暖身)無關。見 `merged_cache_plan.md` §2.4。
- **純診斷 env(刻意不進 YAML)**:`AUG_PROFILE`、`AUG_DUMP_CLUSTER_WEIGHTS=<path>`、`AUG_DUMP_ACTIVE_SET=<path>`。
- **已移除**:`AUG_CLUSTER_LABELS`(A3 拿掉的一次性 partition A/B harness)。

## 已知關鍵結論(別重新踩)
- 公平對比要兩邊同 engine、同 mnt、同 vram budget(no_overload 行為 2026-07-08 起已內建,不再是需要開的條件)。
- topm 的 draft 幾乎全程走 bmm(每題第一個 cycle 因 merged 尚未建會 fallback);specmoe 要 kept-N 全 resident(配 no_overload + pin)bmm 才 engage。
- SVD merge 已整包刪除(2026-06-28);**未來 merge 一律線性、只是權重不同,不做 configurable merge strategy**。
- verify-time merge 的 P1–P3 已實作(`merge_during_verify`),P4(overlap)尚未做。
- **offload 推論是 run-to-run 非確定性的(2026-06-29)**:同 code、同 config 重跑,per-question AccR 平均差 ~0.15(最大 0.68),aggregate AccR 65 題 SD ~0.018。bf16/GPU 微小浮點差 → 早期 accept 翻轉 → 軌跡發散。**含意:不能 bit-exact 比對;小效果(<~3pp)要多跑幾次取平均,大效果才能單跑下結論。**
- **群內 uniform 加權會傷 acceptance——僅限 freq_slice 大群 regime(2026-06-29;2026-07-09 加範圍註記)**:`cluster.within_weight: uniform` vs `freq`,q5 + `freq_slice`(K=16 大群)上 acceptance −6.6pp、TPS −13%。**但 pair 化方法(hybrid/greedy_pair,|group|≤2)實務一律 uniform**——q15 campaign 48 份 clustering config 全部 `within_weight: uniform`,與論文 Eq.(α=1/|C|)一致,也是 merged-cache content-addressing 的前提(見 `merged_cache_plan.md`)。code 預設值仍是 freq,寫新 pair config 記得顯式設 uniform。
- **co-occurrence 當「併在一起」(must-link)分群是錯方向(2026-06-29)**:partition A/B(只換分群、群內 freq),random −8.6pp、static-freq −17pp、cooccur(平衡)−22pp、cooccur(凝聚)−39pp,**兩個共現變體都輸給隨機**。→ 若要做共現分群,方向是 **cannot-link / 圖切割(把高共現切開)**,且相似度用 cosine 非 lift。這是 B2 的前置結論。
- **重構 A1–A5 全部完成(2026-06-29)**,皆驗證為行為等價(結構等價 + import smoke;bit-exact 因上述非確定性不可用)。Mixtral / GPT-OSS / smoke configs 已退役,專案以 Qwen3 為主。
- **draft 的 context KV 品質是 first-order(2026-07-10,C-BOOT 除錯的設計教訓)**:「draft 權重不變 ⇒ acceptance 不變」只對單步成立。HF assisted decoding 的 assistant 有獨立 KV,若讓它在 draft 相位(merged/substitute routing)重 encode 整個 prompt,context 劣化會讓 AccR 從 ~0.5 崩到 ~0.01(兩法兩 backend 同崩,jobs 258368/258378)。歷史上 cycle-1 fallback 一直默默提供「真 routing 的 prompt KV」。共權重單模型的正解 = **KV-copy**(target prefill 的 KV 複製給 assistant;prompt 免重算、context 最優),per-cycle accepted-token 的 merged re-encode 是二階效應可維持。改動投機解碼迴圈時務必想清楚 draft 的 KV 從哪來。
- **SpecMoE 最優跑法 = pin + no_overload + early_pin(2026-07-08,§6-0 ep probe job 254248,詳見 deprecated_docs/memory_hierarchy_plan.md §5 末;no_overload 現已內建,remove_overload_plan 亦已移入 deprecated_docs)**:ep0 下 kept 駐留只有 75%、draft_fetch 4.5TB/run,TPS 被低估;ep1/ep2 駐留 99%、draft_fetch ~0.95TB,q15 TPS 3.26→3.97/4.11。AccR 不受 ep 影響(ep 只改 residency/時序,substitute table 相同)。ep1 vs ep2 差距在 noise 內,單跑不可分;對照一律用 `early_pin: 2`。**注意:`q15_specmoe_r1–r3` 三重複是 ep0 跑的 → AccR 可沿用、TPS 低估;主表 Table 2 的 specmoe TPS(3.4382,`q5_512_sm_on`)是 ep1(script 用 env 開的,YAML 本身沒寫)。**
