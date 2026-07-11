# aug_spec 程式碼架構檢討（2026-07-11）

> 範圍：全 repo——`src/aug_spec/`（逐行讀）、vendored `moe_infinity/`（Python + C++
> 核心）、`configs/`、`scripts/`、`tests/`、全部文件。
> 目的：回答「哪裡是不合理的實作、哪裡冗贅、哪裡需要模組化」。
> 與 `ARCHITECTURE_REVIEW_AND_PAPER_PLAN.md`（2026-07-03，聚焦論文可重現性）互補：
> 該文件的問題清單本文不重複展開，只標註引用；本文聚焦**程式結構本身**。
> 系統怎麼運作見姊妹篇 `ARCHITECTURE.md`。
>
> 標記：🔴 建議優先處理 ／ 🟡 值得做、可排程 ／ 🟢 記錄在案、可不做。
> C++ 端的行號會隨未 commit 的變更漂移，定位以函式名為準。

---

## 1. 結論摘要

整體判斷：**核心 Python 套件（`src/aug_spec/`）的架構是健康的**——registry 化的
drafts/clustering/adapters、「政策在 Python、機制在 C++」的分層、YAML-first 的旋鈕
設計，都是對的且經過多輪重構驗證。主要的債不在核心設計，而在：

1. 🔴 **2026-07-03 的 review（P0–P4）至今一項都沒執行**，且它警告的「git 髒 +
   結果無法對應版本」問題此刻正在發生（未 commit 的 C++ 功能碼 + Python 改動）。
2. 🔴 **specbench 的逐題 except 會把 fail-fast 例外吞成 WARN**——設計上要求
   fail-fast 的斷言（C-BOOT）實際上不會讓 run 停下來，只會燒 walltime。
3. 🟡 **重複碼集中在三處**：adapters 的 forward 樣板（qwen3/mixtral 幾乎逐行相同
   ×3 份 hf expert-loop）、87 個 sbatch script 的複製貼上（含 21 份內聯 watchdog）、
   148 個 YAML 無任何共用機制。
4. 🟡 **vendored moe_infinity 約六成是死碼**（非 Qwen 模型家族、整個 Python 預測
   堆疊、distributed/、openai entrypoints），C++ 端也累積了明確可刪的殘骸
   （死佇列、被 CUTLASS 路徑取代的 per-expert module、295 行註解掉的舊實作）。
5. 🟡 **單元測試防線與計劃不符**：現有兩個測試檔品質好，但 2026-07-03 review
   點名的四個核心防禦測試（config 解析、linear_merge 數值、cluster 性質、top-M
   截斷）一個都不存在。

---

## 2. 首要事項：舊 review 的欠帳（先於本文所有建議）

`ARCHITECTURE_REVIEW_AND_PAPER_PLAN.md` §6 的狀態表全部還是 ⬜，逐項核實結果：

- `summary.json` 無 provenance（P0.2 未做）；無 `requirements-lock.txt`（P0.3）；
  無 `configs/paper/`、`model.revision`、Spec-Bench pin、`reproduce/`（P2）；
  無 `AUGSPEC_CHANGES.md`、`LICENSE`（P3）。
- **現在 git 又是髒的**：`expert_dispatcher.cpp/.h`、`py_archer_prefetch.cpp`、
  `cli.py`、`merged_cache.py`、`drafts/base.py` 有未 commit 修改，
  `c3_pipeline_plan.md`、`run_d0.sh` untracked——目前的 `.so` 與最近的 C2.1
  結果又對應不到任何 commit，正是該 review §2.1 描述的事故重演。

**建議**：本文所有 🟡 級重構都排在 P0/P1 之後。先 commit 現況（語義分拆）、
補 provenance、把 P1 四個防禦測試補齊，才有安全網做其他事。

---

## 3. `src/aug_spec/` Python 套件

### 3.1 值得保持的設計（改動前先確認不破壞這些）

- draft 用 class attribute 自我宣告性質（`holds_merged_residency` 等），CLI 零名單
  （`drafts/base.py:49`）；cluster method 用旗標宣告需要的統計，不用的零成本
  （`clustering/base.py:35-44`）。
- `merging/` 刻意非 registry（merge 永遠線性）——已是定案，別翻案。
- 政策／機制分離：`merged_cache.py`（政策）vs C++ slot API（機制）切得很乾淨，
  telemetry 也齊全。
- 診斷 env 與實驗定義（YAML）分離；`_maybe_dump_*` 全部 no-op-when-unset。

### 3.2 不合理或有風險的實作

**🔴 R1｜逐題 except 吞掉 fail-fast 例外**（`runtime/specbench.py:708-710`）
`run_specbench` 對每題 `except Exception` 印 WARN 後繼續。後果有二：
(a) adapter 的 C-BOOT 斷言（`adapters/qwen3.py:243-248` 的 `RuntimeError`）和
`merged_cache` 的 fail-fast（`merged_cache.py:132,312,327`）被設計成「立刻炸」，
實際上卻被吞成 WARN——一旦進入這種狀態，後續每題都會同樣失敗，65 題 WARN
燒完整個 walltime 才結束；(b) 掉題**不進**任何統計（per_question 不含失敗題），
某方法若恰好在難題上 crash，aggregate 反而變好看，且只有翻 stdout 才會發現。
**建議**（小改、高價值）：失敗題計數寫進 summary.json（如 `n_failed_questions`）；
連續失敗 ≥3 題就 abort；或讓特定例外型別（RuntimeError from draft path）直接 re-raise。

**🟡 R2｜monkey-patch 疊層是最脆弱的依賴面**（`specbench.py:166-369`、`phase.py:47-90`）
兩處 patch 同一個 `AssistedCandidateGenerator.get_candidates`（class-level），疊層
順序有語義（見 ARCHITECTURE.md §3.1），再加 `target_model.forward` wrapper 與
`loader._patch_offload_device` 改**類別** property（`loader.py:149-164`）。這是對
transformers 私有介面（`candidate_generator` 模組、`assistant_kwargs` dict 內容）
的深度耦合——transformers 小版本改動就可能靜默改變行為。
**建議**：不重寫（老 review 也這麼判斷），但 (a) 在 pyproject 把 transformers 釘死
到已驗證版本（與 P0.3 一起做）；(b) 在 `_locked_assist_patch` 開頭加一個 cheap
sanity check（patch 的 method 存在且簽名符合預期，不符就大聲炸）。

**🟡 R3｜module 全域旋鈕 + attribute 注入接線**（`adapters/base.py:38-57`、
`controller.py:63-75`、`offload_merge.py:89-99`）
`_MERGED_BACKEND`/`_EARLY_PIN` 是 import 期讀 env 的可變全域，`apply_offload_settings`
再從 YAML mutate；模組間接線大量靠往別人物件上掛底線屬性（`_merge_engine`、
`_cpu_merge_source`…，全表見 ARCHITECTURE.md §3.5）。可運作、也有歷史理由
（避免把參數穿針引線過六層呼叫），但資料流全靠 `getattr` 追。
**建議**：維持現狀（風險>效益，老 review 同判），但把 ARCHITECTURE.md §3.5 的
接線表當成 contract 維護——加新注入屬性時同步更新該表。

**🟡 R4｜`cpu_source`：整份 30B 模型常駐 host RAM**（`loader.py:216-221`）
offload 跑法會再載一份完整 bf16 模型（~60GB host RAM）當 draft 合併權重來源，
但 archer 的 store 本來就在 host 持有全部 expert 權重（雙份）。目前 `cpu_source`
的實際用途只剩：SpecMoE `prepare` 的 pairwise L2（一次性）、`weight_similarity`
的表（一次性、可 cache 到磁碟）、CPU merge fallback（merge_offload 下幾乎不走）。
**建議**：非緊急（host RAM 充裕）。若要省：prepare 類用途改成直接讀 safetensors
逐層載入即棄，或 prepare 完 `free_model(cpu_source)`（需先確認 merge fallback
確實不再需要它）。

**🟢 R5｜`DraftStrategy.capture` 的參數語義不一致**（`drafts/base.py:66` vs
`drafts/specmoe.py:117`）
ABC 說 capture 收 `router_logits`；averaged forwards 傳 logits，但 SpecMoE 的
substitute forward 傳的是 softmax（`specmoe.py:318`）。配對由 `cache_kind` 保證
不會錯接，但簽名撒謊。改個參數名／docstring 即可。

**🟢 R6｜其他小項**
- `_NamespaceFromDict.__getattr__` 依序掃四個 section，同名 key 被 model 段遮蔽
  （`cli.py:620-634`）——加註解警告即可（老 review P4 同項）。
- `CycleStats` 每 cycle 無條件建 4 個 top-8 tensor + list（`specbench.py:284-325`），
  即使 `emit_tokens_csv=false` 也照做——單 batch 開銷極小，記錄在案即可。
- `phase.py:30` 的 `_AUG_PROFILE` 是 import 期讀 env——與 AUG_PROFILE 為純診斷
  env 的定位一致，但注意它因此不能事後開關。
- `linear_merge` → `engine.build` 是純轉手 shell（`offload_merge.py:141-149`
  docstring 自承）——它是為 B3 cache 預留的接縫，而 B3 已被 C 系列取代。
  可考慮在下次動 merging/ 時把這一跳收掉（低優先）。

### 3.3 冗贅

**🟡 D1｜adapters 的三重複製**（最大宗的 Python 重複碼）
- `make_averaged_forward`：`qwen3.py:224-269` 與 `mixtral.py` 版本逐行同構
  （只差 attribute 名，已由 `_run_dense_expert`/`_swiglu_stack` 抽象掉）。
- `make_masked_forward`：兩家完全相同。
- hf expert-loop（`_standard_routing` 的 hf 分支）存在**三份**：
  `qwen3.py:198-222`、`mixtral.py:76-99`、`drafts/specmoe.py:364-378`（fallback）。
- `make_substitute_forward` 的 lazy-import 兩家重複（`qwen3.py:297-301`、mixtral 同）。
- `_route_offload` 與 `_dispatch_selected` 的 router_mask/weights_mask 建構重複
  （`qwen3.py:147-157` vs `179-186`）。
**建議**：把 averaged/masked forward 樣板與 hf expert-loop 上提到 `MoEAdapter`
base（家族差異已經全部藏在現有的小 hook 後面），`make_substitute_forward` 直接
在 base 給預設實作。約 −150 行、不改行為；**排在論文後**（P4 性質），改完跑
pytest + smoke。

**🟡 D2｜半數 draft registry 是殭屍**
148 個 config 的使用統計：`topm_count` 111、`specmoe` 31、`prefill_topm_count` 2、
`count` 2、`pruned_count` 1、`prefill_count` 1；**`uniform`、`softmax`、
`random_mask` 零使用**。mixtral configs 剩 5 個歷史檔、gptoss 0 個，且
moe_infinity **根本不支援 gpt-oss**（`constants.py` 無此家族——gptoss 只可能走
hf backend），`gptoss.make_masked_forward` 也未實作。
**建議**：都不必刪（支撐「方法通用」敘事 + 對照組價值），但在
`drafts/__init__.py`／`adapters/__init__.py` docstring 標注「active / legacy /
untested」狀態（老 review P4 已有此項，擴充到 drafts）。

**🟢 D3｜`drafts/base.py` 的診斷 dump 稀釋核心**（`drafts/base.py:499-627`）
4 個 `_maybe_dump_*` 約 130 行 + reset() 裡對應的 counter 初始化。老 review 建議
抽到 `drafts/diagnostics.py`，維持該建議（純搬移，P4 執行）。

**🟢 D4｜`merged_cache.build_layer` 與 `_cluster_and_build` 的收尾重複**
兩處各自做「按 mass 排序、組 multi dict」（`merged_cache.py:347-354` vs
`drafts/base.py:484-491`）。共用一個 `assemble_multi_cache(experts, masses, indices)`
即可，順帶保證兩條路徑的 cache 格式永遠一致。

### 3.4 模組化建議

**🟡 M1｜`cli.py:run_experiment`（~300 行）拆出預算計算**
`cli.py:346-429` 的 VRAM 預算/cache-mode 判定是一整塊純函數邏輯（輸入 config +
幾何，輸出 ratio/k_prime/floor/pin budget），卻只能連著整個 CLI 跑才測得到。
抽成 `runtime/budget.py` 的 pure function 後可直接 unit test（cache-mode 的
k_prime/floor 數學正是最該有測試的地方）。這是 P1 測試的前置最佳解。

**🟡 M2｜`ScoreBasedAvgDraft` 職責過多**（`drafts/base.py:84`，430 行）
一個 class 同時管：分數 capture、co-occurrence 累積、activation-sim 累積、
pair_sim 表、history 記錄、diagnostics dump、分群呼叫、merge 組裝。每加一種
統計訊號（B 系列、hybrid）它就長一節。
**建議**：下次要再加訊號時（而不是現在），把「per-question 統計累積器」
（cooccur/act_sim/pair_sim）抽成獨立物件掛在 draft 上，cluster method 的旗標
協定改成向累積器註冊。現在動它風險大於收益。

**🟢 M3｜specbench.py 的 patch 與 driver 可分檔**
`_locked_assist_patch`（200 行，純 HF-integration）與評測 driver（載題/聚合/CSV）
是兩種變更頻率不同的東西，可分成 `specbench.py` + `hf_patch.py`。純搬移，
排論文後。

---

## 4. vendored `moe_infinity/`（Python 端）

對本專案而言 **reachable 集合很小**：`big_modeling.py`（MoE、`_configure_hook`）、
`model_offload.py`（OffloadEngine + hooks）、`models/qwen.py`、
`distributed/expert_executor.py` 的 `dispatch_local/wait_dispatch_local`、
`expert_tracer.create_entry`、utils/。其餘大多是死碼：

- 🟡 **整個 Python 預測/快取堆疊在 Qwen 路徑上是死的**：`memory/expert_cache.py`
  （且內含呼叫簽名錯誤，跑到必炸——證明從未執行過）、`expert_predictor.py`、
  `expert_priority_score.py`；`ExpertPrefetcher` 被建構、`set_archer_engine` 後
  **從未被呼叫**（唯一 call site 在 mixtral.py 且被註解）。
- 🟡 **`distributed/` 三件套**：`DistributedExpertPrefetcher`（與 memory/ 版
  copy-paste 重複、無人實例化）、`DistributedExpertExecutor.dispatch`（RPC 路徑，
  無人呼叫）、`DeviceMapManager`（建構點被註解）。
- 🟢 非 Qwen 模型家族 wrappers + 四個 `modeling_*` vendored 子包、
  `entrypoints/openai/`、`runtime/state_dict.py`（依賴不存在的 `sllm_store`、
  呼叫簽名也錯）、`runtime/compile.py`、`kernel/router.py`、`_engine.so`。
- 🟢 小 bug 級：`OffloadEngine.__enter__` 的 GPTQ patch 區塊**貼了兩次**
  （`model_offload.py:258-271`）；`MoE.generate` 有 return 後的 unreachable 行
  （`big_modeling.py:196`）；`clean_up()` 引用錯的模組路徑（沒人呼叫所以沒炸）；
  `qwen.py:7` unused import + 檔尾 30 行死字串。
- 🟡 **`ExpertTracer` singleton 在 `cuda:0` 上常駐 ~25MB**（`expert_tracer.py:33-35`，
  capacity 1000 × 48 層 × 128 experts 的 zeros）——在刻意壓到 0.2× 預算的實驗裡，
  這是純浪費的 VRAM。`_configure_hook` 只需要 `create_entry` 的 side effect。

**建議**：論文前**不要動**（vendored 碼改了就要重驗）。正確的順序是：
(1) 先寫 `AUGSPEC_CHANGES.md`（P3.1，把 fork 改了什麼、哪些子系統 reachable
記下來——本文 §4/§5 可直接作為底稿）；(2) 論文後若要瘦身，上面 🟡 項是安全
起手（有「從未執行過」的證據），每刪一類跑一次 smoke。`ExpertTracer` 的 25MB
若要處理，改成 lazy 建表或 capacity=1 即可，屬低風險小改。

---

## 5. C++ 引擎（`moe_infinity/core/`）

### 5.1 結構性問題

- 🟡 **兩套驅逐系統共存**（`ExpertDispatcher::FindExpertEvict` vs
  `ArcherTaskPool::RemoveCachedSparseNode/DenseNode`）互相看不見，dispatcher 靠
  「殭屍 key 收屍」補帳（歷史 bug 2.2GB 洩漏的根源，PROJECT_GUIDE 2026-07-10
  結論）。現狀已修穩，但這是**結構債**：任何動到 Node placement 的改動都要同時
  想兩套帳。建議至少在 `AUGSPEC_CHANGES.md` 裡把這個耦合寫成一節；統一驅逐
  屬大手術，論文後再議。
- 🟡 **per-expert `ExpertNode::module` 路徑整條 vestigial**：ctor 為每個
  (layer,expert) 建 `Expert<T>` module、`Enqueue`/fetch 呼叫
  `SetTensorsFromBlob`，但 **forward 從不在這些 module 上執行**（實際計算走
  `modules_[gpu]` 的 `MoEMLP`）。連帶 `SetModuleFromBlob` 全家、`jit_module`、
  `register_expert` 的 `jit_path` 參數都是死的。刪掉可簡化 Enqueue 熱路徑。
- 🟢 **明確死碼清單**（皆已抽查或有清楚證據）：`Wait()`/`WaitExpert()`/
  `CallResult`/`output_queue_`（只被 swap 從未被 push；pybind 綁的是
  `WaitHiddenStates`）、`start_`/`Start()`、`cache_capacity_`、`MUTEX_TYPE`、
  `SET_TENSORS_AND_MODULE_FROM_BLOB` 宏、`ExecArgs.hit/out_dtype/hidden_states`、
  `MoEMLP` 的 CUDAGraph 成員（capture/replay 全註解）、`expert_module.cpp:21-316`
  的 295 行註解舊實作、topology 的 `GetLFUNodes`/`GetLastActivateStage`/
  `GetDenseNodes(node,k)`/`GetSparseNodes(node,k)`、`task_scheduler` 的
  `RemoveCachedNode`（全註解）。

### 5.2 品質雜項

- 🟢 魔數：`kMaxTokens=2048` 兼作 fatal 上限（超過直接 abort run）；`MoEMLP`
  ctor 的 `i<8`/`i<4` buffer 數；三種「可用記憶體」常數並存（`DEVICE_CACHE_LIMIT`
  0.7、`memory_pool.h` 0.8、runtime ratio）。
- 🟢 `InitMergedSlots` 硬編 `cuda:0`、`GetMergedSlot` 忽略 `gpu_id`——單 GPU
  假設成立所以無害，但 API 簽名撒謊；哪天上多 GPU 這是第一個雷。
- 🟢 三個重疊的 prefetch 入口（`prefetch_tensors` 是 no-op、`enqueue_prefetch`
  忽略 `gpu_id`、`fetch_tensors`）；typos（`MCIROSECONDS_SINCE_EPOCH`、
  `ARCHER_IHDEX_NAME`——後者已寫進磁碟索引檔名，**不要改**，改了舊 offload
  export 就讀不到了）。
- 🟢 `Node::is_overflow` 是 overload 路徑的概念殘留，但 prefetch 子系統仍在讀寫
  ——刪 overload 時沒清到的邊角，動之前先確認 task pool 語義。

### 5.3 不可誤刪的 invariants

（詳見 ARCHITECTURE.md §5.4；此處僅列名：merged slot 不進 archer 帳本、
FindExpertEvict 回傳 locked victim、fetch thread 的 2ms timed retry、
Enqueue 的 bounded spin。）**任何 C++ 清理 PR 都必須逐條確認未觸碰。**

---

## 6. scripts / configs / tests / docs

### 6.1 scripts（87 個 .sh）

- 🟡 **全量複製貼上**：87/87 硬編 `--account=MST114471` 與
  `REPO_ROOT=/work/morrisliu07/aug_spec`，45 個硬編個人 email。老 review §2.4
  已列，解法就是 P2.4 的 `reproduce/env.sh + submit.sh`——新 script 從現在起
  應該只用 `run.sh`/`run_watchdog.sh` 傳 config，不再複製。
- 🟡 **watchdog 迴圈被內聯了 ~21 份**（`run_q15_hybrid_*` 等各自貼了 15 行
  mtime 檢查 + SIGKILL），而通用的 `run_watchdog.sh` 就在旁邊。
- 🟢 **12 個 script 還在 `export AUG_NO_OVERLOAD=1`**（已 inert）——無害但誤導
  讀者以為是有效旋鈕。順手清即可。
- 🟢 `run_q5_512_sm_on.sh` 用 env（非 YAML）開 `AUG_EARLY_PIN`——Table 2 的
  specmoe TPS 依賴一個 config 檔上看不到的設定（PROJECT_GUIDE 已標注此陷阱；
  P2 收 paper configs 時務必把它寫回 YAML）。

### 6.2 configs（148 個 YAML）

- 🟡 **零共用機制**：無 anchors、無 include。`offload.path`、
  `vram_budget_ratio: 0.2`、draft args 區塊在幾十個檔案裡物理重複——改 offload
  目錄要動 ~130 個檔。**選項 A**（保守，建議）：接受現狀，靠 `configs/paper/`
  收斂論文用集合（P2.1）；**選項 B**：給 `RunConfig.from_yaml` 加 `extends:`
  單層繼承（~20 行 + 測試），論文後再做。
- 🟢 78 個 config 帶著 inert 的 `no_overload: true`；18 個 config 無任何 script
  引用（orphan）；15 個 script 引用**已刪除**的 config（mixtral/gptoss/SVD 殘留），
  包括 `README.md` 自己的範例 `configs/mixtral_count.yaml`。
- 🟢 `run.skip_categories` 被 66 個 config 使用但 `configs/README.md` 沒寫——
  唯一的 schema 文件缺口，一行補上。

### 6.3 tests

- 🔴 現有 `tests/unit/` 只有 hybrid 與 merged_cache 兩個（品質好的）feature 測試；
  **P1 點名的四個核心防禦測試全缺**：`RunConfig.from_yaml` 解析/錯誤路徑、
  `linear_merge` 數值、各 ClusterMethod 的分割性質（覆蓋/不重疊/群數）、
  `topm_count` 截斷 + `_cluster_and_build` 的 mass 正規化。這些純 CPU、秒級，
  是唯一防線，照 P1 規格補。
- 🟢 `tests/offload/` 維持老 review 判斷（搬 `tests/probes/` + 移除 .out），未執行。

### 6.4 docs

- 🟢 `README.md` 三處 stale：範例用已刪除的 `configs/mixtral_count.yaml`、
  以退役的 Mixtral/GPT-OSS 為主角、連到已搬走的 `PROGRESS.md`。十分鐘可修。
- 🟢 `verify_merge_plan.md` 仍以 EvictLayer/merged-reserve 描述機制（均已被
  C-DEL/兩本帳取代）——文件開頭加一行「部分機制已被 merged_cache_plan 取代」
  的 banner 即可，不必重寫。

---

## 7. 建議行動排序

| 順位 | 事項 | 依據 | 工作量 | 風險 |
|---|---|---|---|---|
| 1 | Commit 現況 + provenance + lock（P0） | §2 | 半天 | 低 |
| 2 | 補 P1 四個防禦測試（可先做 §3.4 M1 的 budget 抽離讓它可測） | §6.3, §3.4 | 1 天 | 低 |
| 3 | R1：specbench 失敗題計數 + 連續失敗 abort | §3.2 | 1 小時 | 低 |
| 4 | 文件同步：README 修 stale、configs/README 補 skip_categories、verify_merge_plan 加 banner | §6.4, §6.2 | 1 小時 | 無 |
| 5 | `AUGSPEC_CHANGES.md`（P3.1；本文 §4/§5 當底稿） | §4, §5 | 半天 | 無 |
| 6 | scripts 停止複製：新實驗一律走 run_watchdog.sh；paper configs 把 env 旋鈕寫回 YAML | §6.1 | 漸進 | 低 |
| — | 以下論文後再做 | | | |
| 7 | D1 adapters 樣板上提、D3 diagnostics 抽離、D4 assemble 共用 | §3.3 | 1 天 | 中（有測試後） |
| 8 | vendored Python 死碼瘦身（§4 的 🟡 項）＋ ExpertTracer lazy | §4 | 1 天 | 中 |
| 9 | C++ 死碼清理（§5.1 🟢 清單；invariants 逐條核對） | §5 | 1 天 | 中 |
| 10 | 選配：config `extends:`、統一 C++ 驅逐系統 | §6.2, §5.1 | 大 | 高 |

**不要做**（延續老 review §5 的守則）：論文前不重寫 cli/specbench、不動 C++
機制碼（C3 計劃內的除外）、不刪 mixtral/gptoss adapter、不把 merge 改 registry、
不嘗試 bit-exact 驗證重構等價性（offload 非確定性）。
