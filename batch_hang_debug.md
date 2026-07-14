# Batch-loop offload hang — 診斷與修復計劃（batch_hang_debug.md，2026-07-13）

> **前提：沒有停損選項，batch 實驗一定要修好。**
> **鐵律（用戶指定）**：
> 1. 修復盡量只在 `src/aug_spec/runtime/batch_spec.py`（新隔離模組），不影響 `run.batch_loop: false` 的既有路徑。
> 2. 需要動 C++ 時，**先加 debug print 定位**，不要一開始就改行為。
> 3. **任何改 C++「行為」的改動，一定要先經用戶同意。**
> 4. 裝套件用專案現行的 venv（uv 管，無 pip；py-spy 裝不起來 → 用系統 gdb）。
>
> 母計劃見 `batch_spec_plan.md`（V0–V4 驗收、sweep 矩陣）；本檔專責這個 hang 的除錯。

---

## 1. 症狀與已確定事實

`run.batch_loop: true` + **offload backend** 下，單題跑到 **~cycle 100–125** 後**無聲卡死**（watchdog STALL rc=124，無 C++ FATAL → pinned-starvation guard 沒觸發）。hf backend 的 V1（B=1）完全正常、SpecMoE 的 offload batch 也完全正常。

三個配置、三個不同卡點，但**都在 ~cycle 125**：

| # | 配置 | 卡在哪（Python faulthandler 主執行緒） |
|---|---|---|
| 1 | cache mode + C3 pipeline（`merge_overlap: true`）| verify：`merged_cache.py:357` `disp.merge_experts_to_slot`（C++ 同步 merge） |
| 2 | cache mode 無 pipeline（`merge_overlap: false`）| verify：`expert_executor.py:62` `wait_dispatch_local`（archer dispatch wait） |
| 3 | **無 cache mode**（`AUG_LEGACY_MERGE=1`，reserve merged 7.25GB）| **draft**：`adapters/base.py:130` `_route_multi_expert` 的 `weight.scatter_`（純 tensor CUDA kernel） |

## 2. 關鍵推論（已排除的假設）

- **不是 C3 pipeline**：配置 2 關掉 pipeline 仍卡。
- **不是 cache mode**：配置 3 關掉 cache mode（走 legacy per-cycle merge）仍卡。
- **不是記憶體爆掉（至少不是明顯 OOM）**：B=1、帶洞 KV ~300–600MB，141GB H200 綽綽有餘。
- **共同因子 = archer 引擎的 CUDA 操作 × batch loop 的緊湊 forward 節奏**。即使 legacy merge，draft 的 `_route_multi_expert` 走 `engine_bmm` → 一樣呼叫 archer `dispatch_bmm`。**每個 draft、每個 verify 都碰 archer 的 CUDA 操作。**
- 配置 3 卡在**純 scatter_**（不是 archer wait）= **device/CUDA context 已 wedged**，不是在等 fetch——某個先前的 async archer 操作把 CUDA context 卡死，後續任何 kernel 都動不了。
- 「不管哪個配置都在 ~cycle 125 卡」的一致性 → 某個東西**隨 cycle 累積**（deterministic-ish），到臨界點觸發。最可能是 batch loop 與 legacy 的差異：**forward 節奏更緊**（draft→verify→下一 draft 之間沒有 HF generate 的 Python 開銷與隱含 sync），archer 的 fetch/exec/merge 線程在上一個 forward 還沒靜下來就被下一個打斷。

**主假設**：batch loop 沒有在 forward 之間讓 archer 引擎的 CUDA stream/線程「靜下來」（legacy 靠 HF generate 的開銷隱含做到），累積到某 cycle 出現 lost-wakeup 或 CUDA stream 死結。

**2026-07-14 決定性證據：hang 是 STOCHASTIC（race）。** 同一個 cache-mode+C3 config：
job 261411 於 ~cycle 125 卡死、job 261535 **跑完整 35 題**（AccR 0.5087、TPS 3.09）。
確定性 bug 會每次都卡 → 這是 race，機率隨 cycle 累積。gdb 抓 hang 因此要碰運氣重跑，
不划算；證據已足夠指向修法 A（forward 間 CUDA quiesce）。

## 3. 診斷階梯（Phase 1）

| 測試 | 狀態 | 結論 |
|---|---|---|
| 換 config（cache on/off、pipeline on/off） | ✅ 已到頭 | 排除 cache/pipeline，確認是 archer×batch 根本交互 |
| **T2：gdb 抓全線程 native stack**（`scripts/run_dbg_gdb.sh`，job 261535） | 🔶 跑數中 | **決定性**：Python faulthandler 看不到 archer C++ 線程；gdb `thread apply all bt` 直接看 fetch/exec/merge 線程卡在 `pthread_cond_wait`／`cudaMalloc`／mutex 哪一種 |
| T3：per-cycle `mem_get_info` log | ⬜ 備用 | 確認/排除記憶體攀升 |
| T4：forward 邊界加 `torch.cuda.synchronize()` + engine quiesce | ⬜ 備用（也可能直接是修法 A） | 若不卡 = race（主假設成立） |

**T2 判讀對照**：
- 卡 `pthread_cond_wait`（archer fetch 線程）→ lost-wakeup，主執行緒沒發 notify / 沒 sync → **修法 A**。
- 卡 `cudaMalloc`/`cudaFree` → pool/記憶體分配阻塞 → **修法 B**。
- 卡 mutex（兩線程互鎖）→ 看是哪兩個；可能要 C++ 加 print 定位（先 print、改行為前問用戶）。

## 4. 修復策略（Phase 2；按 T2 結果選，全在 batch_spec.py）

**修法 A（最可能，race / lost-wakeup）— forward 之間讓 archer 靜下來**
- batch loop 每個 forward 後（尤其 verify→下一 draft）加 `torch.cuda.synchronize()`；cache mode 下不論 `merge_overlap` 都呼叫既有 drain（`engine._drain_pending()` / `wait_merges_done`），確保 merge/pin 的 CUDA kernel 落定後才進下一個 archer fetch。
- 純 batch_spec.py，legacy 零影響。代價：每 cycle 多一次 sync（batch 下攤提）。

**修法 B（記憶體壓力）— 留 pool headroom + 週期性壓平 KV 洞**
- batch loop：KV 洞超過閾值就 physical crop 壓平（移除洞、重建 mask），bound 住 KV 物理長度。
- `cli.py` budget 區塊給 batch 模式留 KV headroom（只在 `batch_loop` 分支，不影響既有預算）。

**修法 C（archer 引擎狀態 / 時序）— 對齊 hook 時序**
- batch loop 的 `on_draft_start`/`update_masks`/`on_verify_layer` 呼叫順序與 legacy `phase.py` 逐條對齊（見 batch_spec_plan §4.6）。

三者不互斥、可疊加。**若 T2 顯示是 archer 引擎自身的 C++ bug（非 batch loop 可繞過）**：先在 archer 加 debug print 定位確切死結點 → 把診斷結果與提議的改法列給用戶 → **經同意後**才改 C++ 行為。

## 5. 驗證（Phase 3）

修好後照母計劃 V 階梯：V2 重驗（Ours+SpecMoE B=1 無 hang、AccR 噪音內）→ V3（B=4 不變量）→ V4（B=64 pilot，KV 記憶體在此才真大，修法 B 若選在此驗）→ sweep。

## 6. 相關檔案

- 診斷 script：`scripts/run_dbg_hang.sh`（faulthandler，Python-only）、`scripts/run_dbg_gdb.sh`（gdb native，T2）。
- 診斷 config：`configs/dbg_ours_batch1.yaml`（cache+pipeline，主目標）、`dbg_ours_nopipe.yaml`、`dbg_ours_legmerge.yaml`。
- 卡點原始碼：`merged_cache.py:357`、`adapters/qwen3.py:167`（`_route_offload`）、`adapters/base.py:130`（`_route_multi_expert`）、`moe_infinity/moe_infinity/distributed/expert_executor.py:62`（`wait_dispatch_local`）。
- 心跳：`batch_spec.py` 每 25 cycle 印 `[cycle N] B=… committed≈…`（已加，永久保留）。

## 7. 狀態

| 項目 | 狀態 |
|---|---|
| Phase 1 換-config 診斷 | ✅ 排除 cache/pipeline |
| Phase 1 T2 gdb native stack（261535） | ✅（意外）該次沒 hang → 證實 STOCHASTIC race，不再追 gdb |
| **Phase 2 修法 A**：`_forward` 後 `torch.cuda.synchronize()`（batch_spec.py，env `AUG_BATCH_NOSYNC` 可關以重現）| 🔶 驗證中 job 261612（cache+pipeline B=1 重跑 3×；r1 已跑到 batch 26/35 無 stall，看起來有效）|
| 增量寫檔（per_question_summary 每 batch flush + 補 overall_summary） | ✅ batch_spec.py：中途死保住已完成題、可即時看進度 |
| `run.debug_invariants` config 開關（V3 用） | ✅ cli.py + batch_spec.py |
| Phase 3 V3(B=4,invariants)+V4(B=64) ours/specmoe | 🔶 跑數中 jobs 261658/261659（qpc15/mnt512，AccR 可對比 B=1、overall TPS 看 crossover 雛形）|
| Phase 3 全 sweep 產 fig | ⬜ 待 V3/V4 綠 |
