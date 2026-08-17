# C3 Pipeline Plan — 同層 fetch ∥ forward ∥ merge(2026-07-11)

取代 `merged_cache_plan.md` §4.2 的 C3-0 / C3-v1 / v2 分階段路線。舊 P4(側流
side-stream merge)為 post-dispatch 一次性 merge 的舊架構而設計,與現行地基
(slot carve、pair-adopt-first、fail-fast、pin-aware evict)不合,且有 merge
輸入存活性 race(見 §3.1),**整段作廢重寫,直接到位**。

## §0 目標與非目標

**目標**:一個 verify 層之內,三件事同時進行——
1. **fetch**:缺席的 routed expert 由 fetch 線程持續上載(現況已有);
2. **forward**:已駐留的 expert 由 exec 線程持續計算(現況已有);
3. **merge**:下一輪 draft 的 merged expert,其成員一到齊就立刻 merge 進
   slot,不等整層 dispatch 結束、不佔 verify 關鍵路徑(**本計劃新增**)。

量化目標:merge(P3) ≈ 164-177s(~7% wall,job 258696/258941/258998)整段移出
關鍵路徑;draft 起跑前的 drain 等待 ≈ 0(有 telemetry 驗證)。

**非目標**(明確不做,除非數據要求):
- fetch 隊優先級/重排(舊 v1)。理由見 §2.5:unrouted 成員改由 merge 線程
  H2D 直讀,不需要 prefetch 類;`merge_cold_bytes` telemetry 若顯示量大再補。
- 跨題 warm start(2026-07-10 已定案不做)。
- 非 cache-mode(legacy merge)的 overlap:舊 P4 分支刪除,legacy 一律 P2。

## §1 現況的兩個序列化點(為什麼 P2/舊 P4 都不夠)

現況(P2):`dispatch_local` → `wait_dispatch_local()`(擋住)→
`on_verify_layer` → `build_layer`(分群+同步 merge+sync)→ 下一層。

1. **分群被不必要地延後**:partition 只需要 score(`capture` 在 dispatch
   **之前**發生,qwen3.py 已確認)+ CPU 端的 cooccur/pair_sim 表,完全不需要
   expert 輸出。它可以在 dispatch 開始前就算完。
2. **merge 佔關鍵路徑**:P2 同步 merge + sync;舊 P4 丟側流雖離開主線程,但
   kernel 延後執行時讀的是 archer pool 的 from_blob view——demand evict
   (單跑 12 萬次)與 pin-blind prefetcher(258555/258658 教訓)可在 enqueue
   與執行之間搬走該記憶體 → **靜默錯權重**。這是舊 P4 作廢的直接原因。

## §2 目標架構

```
verify layer ℓ(單一 forward 內的時間軸)
────────────────────────────────────────────────────────────────
capture(score_ℓ)                      ← 已有,dispatch 前
mc.plan_layer(ℓ)          [Python]    ← 前移:分群+adopt/probe+steal+pin
  ├─ adopt/probe 命中組: 直接發 slot handle(零工作)
  └─ miss 組:           SubmitMergeJobs(ℓ, [{slot, members, weights}])
dispatch_local(ℓ)         [C++]       ← fetch ∥ forward 照舊
  ├─ fetch 線程: 缺席 routed expert 上載;每個 expert 落地時
  │              → 觸發 watching 的 MergeJob(§2.2)
  └─ exec 線程:  駐留/到貨 expert forward
merge 線程                 [C++,新]   ← job 成員到齊即執行:
  snapshot(mutex 內,D2D clone + 等完成)→ 放鎖 → fp32 累加 → slot copy_
wait_dispatch_local()                 ← verify 只等 forward,不等 merge
────────────────────────────────────────────────────────────────
…layers ℓ+1..47 期間,ℓ 的殘餘 merge 繼續在 merge 線程消化…
on_draft_start → WaitMergesDone()     ← 唯一 drain 點,之後 draft 讀 slot
```

### 2.1 Plan 前移(Python,merged_cache.py)

`build_layer` 拆名為 `plan_layer`,邏輯**不變**(pair-adopt-first、singleton
probe、retention steal、progressive pin、_emit device 檢查、telemetry),唯一
差異:miss 組不再同步呼叫 `merge_experts_to_slot`,改為收集 job 清單,結尾一次
`disp.submit_merge_jobs(li, jobs)`。emit 給 draft_cache 的 tensor 是 slot
handle——內容稍後到,draft 在 drain 之後才讀(§3.2)。

呼叫點:新增 engine hook `on_verify_layer_plan(li, block)`,插在三個 adapter
(qwen3 / mixtral / gptoss)的 `dispatch_local` **之前**(capture 之後)。
`on_verify_layer`(post-dispatch)保留兩個殘留職責:act-sim 累加(需要輸出的
cluster 方法)、非 cache-mode 的 P2 merge。

分群輸入時效:score 為本 cycle 當層(capture 剛寫入);cooccur 為截至上一
forward 的累加表(本層本次的貢獻在 dispatch 後才進表,晚一步——與現行
build_layer 讀到的相同,無變化);needs_activation_sim 的方法 pair_sim 舊一個
cycle(hybrid decode 是 capture-free,不受影響;僅逐 cycle act-sim 的實驗變體
受一格延遲,可接受,記入文件)。

### 2.2 C++ MergeJob pipeline(expert_dispatcher)

新增狀態:`pending_jobs_[layer]`(job = slot_idx + expert_ids + weights +
missing_cnt)、expert→jobs 的 watcher map、merge job queue、**merge 線程**
(dedicated,自帶 CUDA stream)。

- `SubmitMergeJobs(layer, jobs)`:對每個 job,在 `cache_mutex_` 下檢查成員:
  駐留 → 不設門;**缺席且在 fetch 隊/在途** → 註冊 watcher(與 fetch 線程標記
  駐留同一把鎖,無 lost-wakeup);缺席且無人要 fetch → 不設門(exec 時 H2D 直
  讀,§2.5)。missing_cnt==0 的 job 直接入 merge queue。
- fetch 線程在 expert 落地點(SetDevice(device) 完成後)回呼:遞減 watcher,
  歸零即把 job 推入 merge queue。
- merge 線程執行 job:
  1. 取 `cache_mutex_`,把每個成員 **clone 到 torch 自有 staging**(駐留→D2D;
     host→H2D),在 merge stream 上 **等 clone 完成**(event sync)才放鎖——
     放鎖之後成員被 evict/搬移都無所謂,輸入已是 torch 持有、refcount 保命。
  2. 鎖外:fp32 累加(現 MergeAccumulate 的主體,讀 staging)+ `slot.copy_`,
     全部在 merge stream 上非同步執行。
  3. CPU 記帳:jobs_done++,condvar 通知。
- `WaitMergesDone(timeout_s)`:等 pending==0 + `merge_stream` sync。60 秒未
  完成 → **fatal + forensic dump**(卡住的 job:層/slot/缺哪些成員/駐留狀態)
  ——與 starvation guard 同風格,不留降級路徑。

保留 `MergeExpertsToSlot`(P2 ablation 路徑用);`MergeAccumulate` 抽出
staging-讀取版共用。

### 2.3 Slot 預配(前置修改,D0)

`InitMergedSlots` 改為**開機即配滿** S×L 個 buffer(現況 lazy:首次 merge 才
`std::move` 進去)。理由:plan 前移後,emit 需要在內容寫入**之前**拿到有效
handle。碳掉 `slot.cached` 的 lazy 分支 → `copy_` 恆定路徑。VRAM 成本 = slot
carve 本身,cli 已劃帳,無新增。

### 2.4 Drain 語意

- `on_draft_start` → `WaitMergesDone()`(取代 `_drain_pending` 的側流 sync)。
- `on_question_start` / `reset()`:先 `WaitMergesDone()` 再清 index/pin(不
  留孤兒 job)。
- 期望值:drain 等待 ≈ 0(48 層的 merge 分散在整段 verify 消化)。上報
  `drain_wait_us`,它就是 C3 的 KPI。

### 2.5 Unrouted 成員:H2D 直讀,不做 prefetch

wanted 但本 cycle 未被 route 的成員,沒人會 fetch。兩案比較後選簡單案:merge
線程 exec 時從 host 直接 H2D 讀進 staging(archer 有 host backing)。PCIe 位元
組數與 prefetch 案相同、不動 fetch 隊、job 永不因 unrouted 成員卡住。上報
`merge_cold_bytes`;若實測量大(PCIe 爭用可觀),再考慮 fetch 隊優先級類
(routed > prefetch)作後續項。

### 2.6 旗標語意與舊碼處置

- cache mode + `merge_overlap: true`(D3 後預設)→ 本 pipeline。
- cache mode + `merge_overlap: false` → 現行 P2(post-dispatch 同步 merge),
  留作論文 overlap ablation 與安全網。
- legacy mode:`merge_overlap` 忽略 + deprecation print(比照 no_overload)。
- 舊 P4 側流分支、`_merge_stream`、`_drain_pending` 刪除(不留死碼)。
- P2 分支裡殘留的 per-layer `torch.cuda.synchronize()` 一併移除(原 C3-0 項)。

## §3 正確性論證(逐一 race / 序點)

1. **merge 輸入存活性**:snapshot-under-mutex + clone 完成才放鎖。放鎖後輸入
   為 torch 自有記憶體,evict/prefetcher 搬原件不影響。同時縮短 mutex 持有
   (重活在鎖外),fetch 線程更順。
2. **slot 寫 vs draft 讀**:draft 只在 `WaitMergesDone`(jobs==0 + stream
   sync)之後讀 slot。
3. **同 build steal 撞在途 job**:steal 只挑非本 build 條目(C2 stamp),本
   build 剛 submit 的 job 的 slot 不可能被同 build 偷;跨 build 的 job 在上輪
   draft start 已 drain。全部 slot 皆本 build 時 fail-fast(C2.1 既有)。
4. **adopt 命中的 slot 不被寫**:adopt/probe 只發 handle,不產生 job;寫 slot
   的只有本 build 為 miss 組新配/偷來的 slot。
5. **題界**:reset 前先 drain(§2.4),index/pin 清空時無在途 job。
6. **starvation guard 不變式**:merge 不 pin、不碰 archer ledger(讀走 staging
   copy),guard 的 all-pinned/drained 判定不受影響。

## §4 改動清單(檔案級)

| 檔案 | 改動 |
|---|---|
| `moe_infinity/core/parallel/expert_dispatcher.h/.cpp` | D0 slot 預配;D1 MergeJob 結構、SubmitMergeJobs、fetch 落地回呼、merge 線程+stream、WaitMergesDone、prof 計數(jobs / gated_wait_us / snapshot_us / merge_cold_bytes / drain_wait_us) |
| `moe_infinity/core/python/py_archer_prefetch.cpp` | 綁 `submit_merge_jobs` / `wait_merges_done` |
| `src/aug_spec/runtime/merged_cache.py` | `build_layer` → `plan_layer`(miss 組改收集 job;其餘不動) |
| `src/aug_spec/runtime/offload_merge.py` | 新 hook `on_verify_layer_plan`;on_draft_start / on_question_start 接 WaitMergesDone;刪舊 P4 分支與 `_merge_stream` |
| `src/aug_spec/adapters/qwen3.py`(+mixtral/gptoss) | `dispatch_local` 前插 plan hook |
| `src/aug_spec/drafts/base.py` | `_refresh_layer` cache-mode 路由改走 plan 時點 |
| `src/aug_spec/cli.py` | D3:cache mode 預設 merge_overlap=true;profile 行加 drain_wait / merge_cold |
| `tests/unit/test_merged_cache.py` | FakeDisp 增 submit/wait;plan_layer 測試沿用+補 job 清單斷言 |
| `tests/offload/c3_jobs.py`(新) | C++ probe:駐留/在途/unrouted 三型成員 job,slot 內容與同步參考 merge 逐位元比對 |

## §5 分步實作與驗證(每步 sbatch,不在 login node 編譯)

| 步驟 | 內容 | 驗證 |
|---|---|---|
| D0 | slot 開機預配,刪 lazy 分支 + `cached`/`DiscardMergedSlot` 刪除 | 🔄 已實作(2026-07-11,與 D1-D3 併批);c0_slots.py 改版(pre-write handle 檢查) |
| D1 | C++ MergeJob pipeline 全套(submit / 雙 arrival hook:GPUFetchFunc 落地 + Enqueue resident(防 prefetcher 提前落地的 lost-wakeup)/ merge 線程+專屬 stream / node-mutex snapshot / WaitMergesDone 60s forensic-fatal / mg_* + drain prof) | 🔄 已實作;probe `c3_jobs.py`(冷成員/identity/重寫逐位元、drain no-op、OOB) |
| D2 | plan 前移:`build_layer(routed=)` 雙模式、`on_verify_layer_plan`(qwen3 dispatch 前)、drain 接 WaitMergesDone(draft start + 題界) | 🔄 已實作;33 unit tests(FakeDisp 改 D0 契約 + pipeline 兩測試) |
| D3 | `merge_overlap` 預設 true、舊 P4 側流刪除、P2 per-layer sync 移除、legacy+overlap deprecation | 🔄 已實作 |
| 驗證 | rebuild → binding → D0/D1 probe → c3_smoke + c3_q5_512(獨立輸出資料夾) | ✅ 2026-07-11 job 259252:probe 全過(逐位元);**KPI 達標:drain 1997 次共 0.009s ≈ 0**——merge 完全被 verify 遮蔽;jobs 539k(98% gated = 到齊即發機制實際運作);cold H2D 1336 GB(§2.5 telemetry,未見 TPS 傷害);verify fetch 16.9TB(歷次最低);AccR 0.6657 / TPS 4.04(vs B 3.67,方向 ↑,幅度照例交 q15 三重複);60s fatal 全程未觸發。**q15 規模再確認(2026-07-12 f15 批)**:drain 6919 次共 0.032s ≈ 0,七跑零 fatal;cold H2D 於 q15 放大到 5.4TB(fetch 的 ~9%)→ prefetch 優先級類升為有數據支撐的後續項。**後續優化候選**(非阻塞):merge 線程每 job 全 stream sync(merge 桶 604s,全在線外)可改 per-snapshot event;cold 1336GB 若要壓可加 fetch 優先級類 |
| 最終 | q15 **三重複**(與 C1/C2/C2.1 一併裁決幅度) | AccR 噪音內、TPS ≥ 基線、draft_fetch≈0 |

**回退**:git 為回滾機制,每步一 commit;`merge_overlap: false` = P2 全程可用。

## §6 驗收判準(彙總)

1. `drain_wait_us` ≈ 0(KPI:merge 完全被 verify 遮蔽);
2. q5 上 TPS 相對 C2.1 基線方向為正,幅度結論交給 q15 三重複;
3. AccR 與基線同量級(逐位元正確性由 D1 probe 把關,不靠 AccR 猜);
4. `merge_cold_bytes` 上報(決定是否需要 prefetch 優先級類的後續項);
5. 全程無 RuntimeError / fatal / device 混雜。
