# moe_ondemand 實作計劃（2026-07-15）

> **目標**：新增第三個非投機 baseline `draft.name: moe_ondemand`——**沒有任何 cache**：
> 每個被 route 到的 expert 一律 on-demand fetch（host→GPU）、算完即被驅逐、
> **不跨層、不跨 step 留存任何 expert**。支援 B=1（legacy specbench）與 B>1（batch_spec）。
>
> **定位**：完成非投機三點 ablation 階梯，量化「cache 的價值」：
> `moe_caching`（0.2× 全預算動態 cache）⊃ `moe_precache`（僅靜態釘 10%）⊃ **`moe_ondemand`（零 cache 下界）**。

---

## 1. 語意規格

| 項目 | 規格 |
|---|---|
| 投機 | 無（同 moe_caching：無 draft/controller/forward swap，純 target generate）|
| pin | **0 個 expert**（沒有 pinned set）|
| cache | **無**。非 route 中的 expert 不得駐留；route 中的 expert 用完即可被驅逐 |
| fetch | 每次 route 到＝一次 H2D fetch（帳目上 fetch 數 ≈ dispatch 總數）|
| batch | B=1 與 B>1 都支援 |

## 2. 關鍵設計事實（為什麼不用動 C++、也不用逐層 flush）

1. **archer 引擎本來就是 streaming 的**（`expert_dispatcher.cpp Enqueue`）：resident→exec、
   不在→fetch queue；pool 滿了 `FindExpertEvict` 驅逐非 pin 的。**「無 cache」= 把 pool
   縮到只剩 in-flight 管線緩衝**，這正是 moe_precache 修正版已驗證的機制（smoke 263076/263077 綠）。
2. **expert 權重是 per-layer 的**：第 i 層的 expert 只可能在「下一個 forward pass 的第 i 層」
   （48 層的 turnover 之後）被重用。只要緩衝容量 « 一個 forward pass 的 fetch 總量
   （B=1 每 pass ≈ 384、B=64 ≈ 6144），舊 expert 必在輪回前被擠掉 → **架構上保證零重用**。
   修正後緩衝 = 144 格（見 §9 prefill floor）：B=1 餘裕 384/144 ≈ 2.7×、B=64 ≈ 43×，不變量仍成立。
   **（2026-07-15 實測補充）pool 總量有硬下限**：一次 dispatch 會把該層全部 routed expert
   （prefill 下可達 E=128）同時 commit 在飛，pool 裝不下一整個 dispatch 就會
   `evict starvation` FATAL（`expert_dispatcher.cpp:1055`）。pin 可計入這個底線——
   這就是 precache（624 pins）headroom 只要 24 也沒事、ondemand（0 pin）不行的原因。
3. 所以「算完就 evict」用 **lazy eviction（pool 滿才逐）在效果上完全等價**，
   不需要逐層 `flush_cache`（那會殺掉 fetch/compute overlap、又要每層掛 hook，徒增變因）。
4. **零 pin → pinned-starvation guard 永不觸發**（它只在 cache 全 pinned 時作動）。
5. runtime 行為與 moe_caching **唯一**差別 = pool 預算大小。因此實作量極小、風險集中在
   「pool 太小會不會出 init/邊界問題」（見 §6 風險）。

## 3. 變更清單（全部 Python，不動 C++、不動既有 baseline 行為）

| 檔案 | 變更 |
|---|---|
| `src/aug_spec/runtime/loader.py` | `compute_precache_pool_bytes` 允許 `pin_fraction=0`（此時 `n_pin=0`、pool=純 streaming 緩衝：`min(E, B×top_k) + slack×top_k` 個 expert）。公式與 moe_precache 完全共用 → 兩 baseline 只差 pinned set，對照乾淨 |
| `src/aug_spec/cli.py` | ① `NON_SPEC_DRAFTS += "moe_ondemand"`；② 預算分支：`is_ondemand` 走同一 helper（pin_fraction 固定 0），budget 印出 `pin 0/128, pool=純緩衝`；③ **不建 PrecacheManager、不掛任何 hook**（adapter 行照 moe_caching 印 non-speculative）；④ `_dump_precache_profile` 一般化為 `_dump_offload_fetch_profile`：dispatcher 改由 `model.modules()` 找 `expert_executor` 取得（precache/ondemand/caching 三者通用，AUG_PROFILE=1 才印） |
| `src/aug_spec/runtime/batch_spec.py` | **零變更**（controller=None 的非投機路徑已被 moe_caching B=4/64 驗證；ondemand 無 pin/reset 需求）|
| `configs/` | `ondemand_smoke.yaml`（B=1, qpc1, mnt32）、`ondemand_smoke_b4.yaml`、`ondemand_b1/b4/b64.yaml`（t1 協議：qpc15/mnt512/humaneval/mt_bench_pooled）|

pool（Qwen3-30B, expert≈9.4MB）：**所有 B 一律 144 experts ≈ 1.36GB**（prefill floor
`E+slack` 主導；修正前 B=1 只給 24 → V1 實測 evict starvation，見 §9）。
（memory 軸的賣點：零 cache 下界只要 ~1.4GB expert VRAM。）

## 4. 驗證階梯（每步過了才走下一步）

| # | 測試 | 判準 |
|---|---|---|
| V1 | smoke B=1 + B=4（qpc1/mnt32, 2 sbatch）| rc=0、無 deadlock、budget 行印 `pin 0`、正常產 token |
| V2 | **帳目證明**（AUG_PROFILE, B=1, mnt128, 1 sbatch）| `pinned_hit_n = 0`；`evictions ≈ fetches`（1:1 turnover）；`fetch 數 ≈ dispatch 總數`（≈ tokens×48×8 + prefill）→ 證明零 cache |
| V3 | full t1 三點（B=1/4/64, 3 sbatch）| 得正式數字，入三方對照表 |
| V4（選）| 同一 profile dump 跑一次 moe_caching | 首次量到 moe_caching 的實際 hit rate，補完三方機制表 |

## 5. 預期結果與解讀（先寫下 prediction，避免事後合理化）

- **fetch 流量**：B=1 每 token ≈ 48×8×9.4MB ≈ **3.6GB**（vs precache 修正版 ~2.35GB、+53%）。
- 若 `ondemand ≈ precache ≈ caching`（TPS 都在 ~7 左右）→ 證明 **B=1 時 fetch 完全被 overlap
  隱藏，cache 在此 regime 沒有價值**——這回答你「覺得很奇怪」的疑問的最後一塊。
- 若 `ondemand < precache < caching` 拉出階梯 → 階梯差就是「pin 10%」與「full cache」各自的
  真實價值，直接可畫進論文。
- B=64 預期 fetch-bound 更重，ondemand 應明顯低於另外兩者（每 step 全 6144 expert 重抓）。

## 6. 風險與緩解

| 風險 | 緩解 |
|---|---|
| R1：pool 太小觸發 archer 邊界問題 | **已發生並修好（2026-07-15，見 §9）**：B=1/B=4 smoke 均 `evict starvation >60s` FATAL——一次 prefill dispatch 全層在飛、24 格裝不下。修法＝pool 底線 `E+slack`（pins 計入）→ ondemand headroom 144；精確 forensics：`cached 0 pinned 0 stale 0 locked 0`（可驅逐數 0，全部 in-flight）|
| R2：B=64 純 fetch 太慢跑不完 t1 | V3 的 sbatch 給 8h；不夠就砍 qpc 或標註 partial |
| R3：與 moe_precache 共用 helper 改壞 precache | `pin_fraction>0` 路徑逐字不動；改完重跑 precache 預算 print 對照（純算術驗證，不用 GPU）|

## 7. 不做的事

- 不動 C++（現有 binding 已足夠；符合鐵律）。
- 不動 moe_caching / moe_precache / 投機路徑的任何行為。
- 不做逐層 flush_cache 變體（§2.3 的理由；若未來要「嚴格 eager evict」再議）。

## 8. 收尾

- 實作完成後全改動 review 一遍（CLAUDE.md 規則 2）。
- 更新 `PROJECT_GUIDE.md` 非投機 baseline 段（規則 4）：三個 baseline 的語意一覽。
- 更新 `baseline_tables_plan.md` §7 狀態表（若三點數字入表）。

---

## 9. 進度（2026-07-15）

| 項目 | 狀態 |
|---|---|
| 實作（loader pin_fraction=0、cli 註冊、profile dump 一般化、5 configs）| ✅ 完成 + review（AST/import/預算 sanity 過；precache 13 pins 不受影響）|
| V1 smoke 第一輪（263243 B=1 / 263244 B=4）| ❌ **雙雙 FATAL：`evict starvation >60s`**（layer 1 prefill）。forensics：`cached 0 pinned 0 stale 0 locked 0, need 9.4MB`——pool 24/48 格全被當前 dispatch 的 in-flight expert 佔滿、可驅逐數 = 0 |
| 根因 | 一次 dispatch 把該層**全部** routed expert 同時 enqueue（prefill 下最多 E=128 個在飛）；pool 必須裝得下一整個 dispatch。precache 的 624 pins 恰好撐起這個底線所以沒踩到；ondemand 無 pin → 24 格不夠 |
| 修法（已改 `loader.py`，**未驗證**）| pool 底線 = `E + slack`（144 格），pins 計入底線 → **ondemand headroom：全 B 一律 144 ≈ 1.36GB；precache 完全不變**。零 cache 不變量重驗：B=1 每 step fetch 384 > 144（2.7×）、B=64 每 step 6144（43×）→ 跨步重用仍不可能 |
| 待辦 | 見 §10 |

## 10. 縮 pool 路線失敗 → 改走 Path A（flush，2026-07-15）

**N2 re-smoke（144 格）仍 starve**，只是死點從 layer 1 推到 layer 27。讀 C++ 定案根因：

- Forensic 矛盾：`cached_experts_`（可驅逐集合）**空的**，但 `cache_sizes` 顯示 144 格全滿。
  → 144 個 expert 佔著記憶體卻不在可驅逐登記表 = **卡在 in-flight / prefetcher 搬進來的
  zombie 狀態**（`expert_dispatcher.cpp` 有 "prefetcher relocates nodes without going through
  cached_experts_" 的 zombie-reap 註解，1126 行附近）。
- **archer 的 prefetcher 會預抓未來層的 expert 填滿 pool**，這些 resident 但未登記為可驅逐 →
  放大 pool 只是延後填滿。引擎的工作池下限 ∈ **(144, 648]**（precache 靠 624 pins 湊到 648 total
  才跑得動；pins 計入下限）。
- **`FlushCache`（739 行）只 SetDevice(host)`cached_experts_` 裡的**，對 in-flight/zombie 無效；
  且 flush 發生在 forward 之間，**救不了 prefill 那個 forward 內的 starvation**。

**結論**：縮 pool 逼零 cache 這條路，跟引擎的 prefetch 設計本質衝突、跑不動。**改走 Path A（用戶已選）**。

### Path A 設計：cache-disabled（= caching 的 pool + 每步 flush）

> `moe_ondemand` = **`moe_caching` 完全相同的 pool（vram_budget_ratio 0.2）** + **每個 forward
> 完成後 `flush_cache(0)`**。同 VRAM、同運行條件（保證跑得動），差別只在「每步把 cache 清光 →
> 下一步全部重抓」。這乾淨地隔離出**「cache 到底省多少 TPS」**——同記憶體、cache on vs off。

- **為何同 VRAM 不縮**：引擎跑不動 tiny pool（上面）。硬縮 → starve。所以 memory 軸不再是賣點，
  但**實驗目的（cache 的 TPS 價值）反而更乾淨**：唯一變因就是 cache。
- **零 cache 怎麼保證**：一個 decode step = 一次 forward，層內每個 expert 只 route 一次（無層內重用）；
  step 之間 flush 清光 → **無跨步重用 → hit rate ≈ 0**（殘留只可能來自 prefetcher zombie，用
  AUG_PROFILE 的 `CACHE hit rate` 實測驗證；若 >0 再考慮 per-layer flush）。
- **flush 注入點**：在 `model` 掛一個 forward post-hook → 每次 `model.forward`（= prefill + 每個
  decode step，B=1 與 B>1 都走 model.forward）之後 `disp.flush_cache(0)`。**一個 hook 同時涵蓋
  兩條路徑**，batch_spec 不用改。實作在 `runtime/precache.py` 的 `OndemandFlusher`。

### 變更（Path A）

| 檔案 | 變更 |
|---|---|
| `loader.py` | `pin_fraction=0` 支援保留（無害）；**ondemand 不再用這個 tiny-pool 預算** |
| `cli.py` | is_ondemand **移出** precache 預算分支 → 走 `vram_budget_ratio` 分支（= caching 同 pool）；建 `OndemandFlusher` + arm/disarm |
| `runtime/precache.py` | 新增 `OndemandFlusher`（model post-forward hook → flush_cache）|
| configs | 5 個 ondemand configs 加 `vram_budget_ratio: 0.2` |

## 11. 結果（Path A 全綠 + bottleneck 定案，2026-07-16）

- **P2 smoke**：零 cache 帳目鐵證——fetches == forwards（203,869==203,869 / 166,820==166,820）、
  `CACHE hit rate 0.0000`、pinned 0、無 starvation。（附帶證明 archer 預測式 prefetch 從未在
  enqueue 前落地——否則會以非-pin 命中出現。）
- **P4 full t1**：ondemand B=1 **6.17** / B=4 **9.33** / B=64 **36.06**（flush 數與步數吻合）。
  vs caching 7.70/9.41/34.17 → cache 價值 = B=1 +25%、B=4 ~0、**B=64 −5%（零 cache 反而贏）**。
- **E1-E3 profiling（jobs 263362-364, qpc1/mnt128）**：
  - E2 **hf 全駐 GPU 天花板 = 9.24 tok/s**（B=1，無 offload 引擎）→ offload+全量 refetch 只佔
    wall ≤33%；瓶頸 = eager 逐-expert forward + 48 層序列本身。
  - E1 ondemand：fetch 執行緒總時 60.1s、有效頻寬 ~50GB/s（Gen5 滿速、只是總量小）、exec 35.1s。
  - E3 Ours：verify_fetch 43.1s / exec 33.0s / **merge 執行緒 70.3s** / mg gate-wait 352s（overlap 等待）
    / fetched 2.1TB；TPS 3.04（AccR 0.475 qpc1 樣本，時間歸因有效；cycles 計數器每題重置，只用 totals）。
- **定案結論**：draft 免 fetch 省的是**非瓶頸資源**；每 committed token 付 (T+1)/MAT ≈ 1.7-1.8×
  全深度 forward（**正是瓶頸**）→ 本機任何 vram budget 下 B=1 TPS 軸投機贏不了 plain
  （ondemand 6.2 是 budget 無關下界）。細節已寫入 PROJECT_GUIDE 關鍵結論。

## 12. 殘留待辦

| # | 動作 | 狀態 |
|---|---|---|
| P5 | precache 修正版 full 重跑（或改 flush-except-pins 設計，需改 C++ FlushCache）| 待用戶決定 |
| P6 | baseline_tables_plan §7 若把 ondemand 三點入表 | 待定 |
