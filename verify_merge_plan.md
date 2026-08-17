# Verify-time merge（merge↔PCIe 同步）優化計劃

> 把 offload_plan.md §1.3 的「merge-during-verify、experts resident、overlap
> PCIe」理想，落到 **verify 階段的 fetch 排序 + 逐層就地 merge**。全部實作在
> `OffloadMergeEngine`（`runtime/offload_merge.py`），由 `merge_offload` gate 進場、
> 各階段自己的 flag 做 on/off ablation。
>
> **順序經 P3 初測修正**（job 240400）：mid-forward 的逐層 merge 在低預算下會撞
> moe_infinity 的 **overload-eviction 競態**（expert 用完即被搬回 host、非 resident，
> merge 讀到半 evict 的 node → dispatcher FATAL）。因此核心 merge **依賴**先有空間/
> eviction 控制——順序改成 **flush → reorder+evict → 核心 merge → overlap**。

---

## 0. 狀態總覽

| 階段 | 內容 | 狀態 |
|---|---|---|
| **P0** | VRAM 預算語意（`vram_budget_ratio`）+ NVML audit + guard | **✅ DONE**（job 240383 驗證） |
| **P1** | phase-exclusive flush（archer@draft-start + merged@draft-end） | **✅ DONE**（job 240427；機制安全；peak 由 merged-reserve 解決） |
| **P2+P3** | per-layer merge during verify + 逐層 evict（統一機制） | **✅ DONE**（job 240453；無 crash + TPS +14%） |
| **merged-reserve** | 從預算預扣 merged → archer pool 縮小 → 技術記憶體塞進 0.2x | **✅ DONE**（job 240585；peak 24→16.84GB；技術=12.2GB=0.2x） |
| **P4** | overlap：merge ‖ fetch | ⬜ TODO |

**已就緒的基礎**（offload_plan.md M9b）：`OffloadMergeEngine`（`attach`/`build`/
`on_verify_layer`/`on_draft_start`/`on_draft_end`）、`merge_offload` flag（byte-identical
重構）、C++ `merge_experts_local` / `get_resident_expert_weights` / `flush_cache` /
`evict_layer`、dispatcher deadlock 修正、kMaxTokens 128→2048。

**現況總結**：offloading 技術三件大事都成立——(1) **TPS**：P2+P3 的 0-refetch merge
+14% @ b=0.2；(2) **無 crash**：EvictLayer 解開 overload 競態；(3) **真的塞進 0.2x**：
merged-reserve 讓技術記憶體 = 12.2GB（archer pool 4.97 + merged 7.25）。
**下一步**：(a) specmoe pinning（[specmoe_pin_plan.md](specmoe_pin_plan.md)，讓 baseline
忠實）→ 核心對比 topm vs specmoe @ b=0.2；或 (b) P4 overlap 再榨 TPS。

---

## 1. P0 — VRAM 預算語意 + 帳/audit/guard ✅ DONE

### 1.1 預算語意（`vram_budget_ratio` 取代 `device_memory_ratio`）
現行 `device_memory_ratio` = archer pool ÷ GPU 實體，GPU-dependent 又不直觀。改用：
> 「可用 VRAM = `vram_budget_ratio` × 整個模型佔的 VRAM」（GPU 無關、對應論文 0.2x）

- `model_vram` = 全模型 bf16 footprint（meta-model 算 param 總量，**實測 61.1 GB** = 論文 x）。
- `device_memory_ratio = vram_budget_ratio × model_vram ÷ GPU_total`（loader 推導、log 印出）。
- 保留 `device_memory_ratio` 為 escape hatch。

**換算（H200 GPU_total=150 GB, x=61.1 GB）**：

| `vram_budget_ratio` (b) | usable VRAM | 等效 device ratio | sparse cache | 能 resident |
|---|---|---|---|---|
| **0.2**（論文預算 0.2x） | 12.2 GB | 0.081 | ~9.2 GB | **~974 顆** /6144 → 重度 evict |
| 0.35 | 21.4 GB | 0.143 | ~18 GB | ~1900 顆 |
| **1.84**（= 現在 device 0.75，cmp/base 全用） | 112 GB | 0.75 | ~109 GB | **6144 全部** → 不 evict |

**範圍**：backend 層級，**所有 offload draft 一律生效，含 specmoe**——公平對比要共用同一
budget。merge 優化（P3+）才是 topm-only。

### 1.2 帳/audit/guard
**兩套記憶體**：archer C++ pool（expert cache，有 eviction 自約束）＋ torch allocator
（merged experts + 暫存 + activations + KV，**不在 archer 帳內**）。**沒有單一旋鈕硬 cap
總和**——總和守在預算內靠結構性 bound（archer cap + flush）＋ 監控驗證。

- **audit**：背景每 verify cycle 取樣 NVML（`mem_get_info`，**含 archer pool**——torch
  `max_memory_allocated` 看不到），記 peak、結尾印 `peak vs limit`。
- **guard**（flag `vram_guard`，預設 true）：`used > usable × 1.05` 就 log WARNING，
  **只 warn 不 enforce**。完成系統下不該響；響了=漏帳/leak（除錯網）。
- **預算範圍裁決 = (A) 涵蓋全部**（raw NVML 總和，含 CUDA context + activations）。保守、
  簡單；context（~6 GB）一致存在於所有系統。

**P0 驗證結果（b=0.2）**：`model_vram=61.1 GB`（精準命中 x）、`usable=12.21 GB`（正好
0.2x）、`device_memory_ratio=0.0814`；guard 正確 WARN（used 25.28 GB > 12.21 GB）；
**torch peak 6.98 GB vs NVML 25.48 GB → torch 少算 3.6×、NVML 必要**；run 仍 PASS。
**實證 §1.2 漏帳是真的**：b=0.2 宣稱 12.2 GB 實際 25.5 GB（archer 12.2 + merged 7.25 +
context ~6）→ 不 flush 爆預算 2×。

**實作**：`loader.compute_model_vram_bytes` / `loader.get_gpu_used_bytes` /
`cli`（推導 + 傳 limit）/ `specbench`（取樣 + guard）。flag：`vram_budget_ratio`,
`vram_guard`（offload config 下）。

---

## 2. 背景：cache 容量與價值區間

單層 128 顆、verify ~100 顆 active，全部 ≈ 944 MB，任何 b 下都塞得進「單層工作集」→
**eviction 是跨層的（全域 cache 填滿），不是單層內的**。但在低 b（cache 滿）下，
moe_infinity 走 **overload 路徑**：batch>1 的 verify 把 expert **exec 完即搬回 host**
（`expert_dispatcher.cpp` OutputFunc 的 evict 分支）→ expert **用完就不 resident**。

**價值區間**：
- **b ≥ 1（VRAM ≥ 模型）→ 全 resident、不 evict、不 overload**：merge 本來就 0 re-fetch，
  本優化是 **no-op**。
- **0.2x 論文預算（b ≈ 0.2–0.35）→ 戰場**：cache 滿、跨層狂 evict + overload 用完即搬走。
  現行「verify 全跑完才 refresh merge」會大量 re-fetch（merge 時 expert 早被擠掉）。
- **⚠ 驗證必須在低 b（≈0.2–0.35）；b≥1 量不到差別。**

---

## 3. 機制拆解 + 依賴關係

| 元件 | 買到什麼 | 對應階段 |
|---|---|---|
| per-layer merge during verify | 0 re-fetch（merge 時 expert 仍 resident） | **P3 核心** |
| flush merged at draft end | 騰出 7.25 GB → cache 不那麼滿 → **少觸發 overload** + 預算合規（§1.2） | **P1** |
| reorder（non-merge 先、merge 後）+ evict | merge-expert 最後 fetch、resident 最久；積極 evict 壓 footprint | **P2** |
| overlap merge‖fetch | 隱藏 merge 計算 | **P4** |

**關鍵依賴（P3 初測學到的）**：核心 merge（P3）要 expert **verify 後仍 resident**。但低 b
下 overload 路徑**用完即 evict** → expert 非 resident → mid-forward merge 既無好處、又撞
正在進行的 eviction（競態 FATAL）。**所以 P3 依賴先用 P1（flush 騰空間）+ P2（控制 evict、
讓 top-M 留住）把 overload 壓掉、把 top-M 釘住。** 這就是新順序的理由。

兩個仍成立的洞察：(1) **零預測**——`_route_offload` 算完 gate 就有這層 count、當下知道
top-M；(2) merge 與 `draft.refresh` 同源 count → 換時機**不該改 acceptance**（bit-exact）。

---

## 4. Flag 設計（for ablation）

全在 `model.offload` 下：

```yaml
model:
  backend: offload
  offload:
    vram_budget_ratio: 0.2        # P0：可用 VRAM = 0.2 × 模型大小（自動預扣 merged）
    vram_guard: true              # P0：over-budget 警告（預設 true）
    merge_offload: true           # 進場：用 OffloadMergeEngine
    flush_on_draft_end: true      # P1：phase-exclusive flush
    merge_during_verify: true     # P2+P3：逐層 verify merge + 逐層 evict
    merge_overlap: true           # P4：merge 在 side stream 重疊 fetch（未做）
```

| flag | 預設 | 角色 |
|---|---|---|
| `vram_budget_ratio` | — | P0 預算（未設則用 `device_memory_ratio` escape hatch）；merge draft 自動預扣 merged |
| `vram_guard` | true | P0 guard |
| `merge_offload` | false | engine 進場（GPU resident merge） |
| `flush_on_draft_end` | false | **P1** ablation（archer@draft-start + merged@draft-end flush） |
| `merge_during_verify` | false | **P2+P3** ablation（=true：逐層 merge + EvictLayer；false：after-verify refresh） |
| `merge_overlap` | false | **P4** ablation（未實作） |

**ablation 量法**：固定 `merge_offload=true` + 同 b，逐一 toggle 各階段 flag，比 TPS /
per-cycle 牆鐘 / re-fetch（archer `get_hit_rate`）/ NVML peak → 得每個優化的純幅度。

---

## 5. 分階段實作（新順序）

### P0 — VRAM 預算語意 + audit + guard ✅ DONE（見 §1）

### P1 — phase-exclusive flush ✅ DONE（job 240427）
> bidirectional flush，§1.4 phase-exclusive memory。

- **實作**：`OffloadMergeEngine.on_draft_start()` 在 draft 開始 flush archer cache
  （C++ `FlushCache`：evict 全部 resident expert 回 host、reset cache_sizes；draft 用
  merged dense 不碰 archer → 閒置可丟，host 副本還在故無 D2H）；`on_draft_end()` flush
  merged（clear `draft_cache` + `empty_cache`）。flag `flush_on_draft_end`。

**驗證結果（b=0.2，merge_during_verify=false 隔離 flush）**：
- **✅ 安全、無 crash**：FLUSH ON 完整跑完 13 題。確認 flush 在**靜止窗口**（draft 邊界、
  非 mid-forward）安全——不像 P3。回答了「會不會撞同一個競態」= 不會。
- **❌ peak 幾乎沒降**（OFF 23.99 → ON 23.92 GB）。**根因**：peak 發生在 **merge-build
  時刻（refresh）**——resident-merge 本質是「從 archer-resident expert 建 merged」→ 建的
  當下 archer 必須滿 + merged 同時存在 → 必然並存。flush archer 在 draft-start（build 之後）
  → 擋不掉這個 peak。**flush 機制本身對；降 peak 要靠 P2 的積極 evict + P3 的逐層增量 build
  （避免「archer 全滿 + 全部 merged」同時並存）。**
- **❌ 預算物理不可行（K=16）**：archer(12.2 滿) + merged(7.25, K=16) + context(6) = 25 GB
  ≫ 12.2 GB。即使完美 flush 也塞不下。K=1（論文單顆 merged，0.45 GB）才塞得下。
- acceptance OFF 3.36 vs ON 3.58（offload 低預算 dispatch 非確定性，§2；微差忽略）。

**結論**：flush 機制完成、安全。**peak 降幅延到 P2**（積極 evict 讓 archer 在 build 時不滿
→ 不再全並存）。flush 是 §1.4 必要的一塊，但單獨不夠——要配 P2/P3 的 evict + 逐層 build。

### P2+P3 — per-layer merge during verify + 逐層 evict　✅ DONE（job 240453）

> 原計劃 P2（reorder + evict）與 P3（per-layer merge）耦合，合併實作。最終沒做 reorder
> （EvictLayer 讓 cache 不滿就解決了 residency），核心 = 逐層 merge + 逐層 evict。

**統一機制**（P2 與 P3 耦合，合併實作）：`on_verify_layer` = 逐層 merge（resident，P3）
→ `torch.cuda.synchronize()`（等 merge 的 async add_ 讀完）→ **`EvictLayer`（新 C++）evict
該層全部 expert**。每層 merge 完即 evict → sparse cache 永不填滿 → **overload evict-after-use
不觸發 → merge 讀 resident、無並行 eviction（P3 競態解除）**。flag `merge_during_verify`。

**改動**：C++ `ExpertDispatcher::EvictLayer` + bind `evict_layer`；`on_verify_layer` 加
sync + evict（sync 必要——merge 的 add_ 是 async，不 sync 就 free 掉正在被讀的 source）。

**驗證結果（b=0.2，merge_during_verify off vs on）**：
- **✅ ON 不再 crash**：完整跑完 13 題。**P2 成功解開 P3 的 overload 競態**——最關鍵的成果。
- **✅ TPS +14%**（OFF 1.96 → ON 2.23）：during-verify 的 0-refetch merge 真的更快。
- MAT/AccRate 基本相同（3.55→3.52，非確定性噪音內）。
- **❌ NVML peak 沒降**（24.03 → 24.04 GB）。**根因**：archer 用自己的 `DeviceCachingAllocator`
  （非 torch），`free` 把記憶體還回 cache 池、**不 cudaFree**（只有 `free_cached()` 真釋放，
  平常不呼叫）。所以 evict 只降邏輯佔用、實體 NVML 不變。**和 P1 flush 同一個道理。**

**結論**：throughput 機制成功（無 crash + TPS↑，主軸達成）。peak 當下沒降的根因 = archer
的 `DeviceCachingAllocator` 保留（evict 還回 cache 池、不 cudaFree）——但這由下面的
**merged-reserve 從另一個方向解掉**（直接縮 archer pool cap，而非靠 evict 釋放）。

### merged-reserve — 把 merged 塞進 0.2x　✅ DONE（job 240585）

**機制**：cli 算 budget 時**預扣 merged 的固定大小**（`K×L×expert`）→ 縮小
`device_memory_ratio` → **archer pool 被物理縮死**，留出的空間給 merged。merged 在 torch
（大小固定 = K×L×expert），archer pool + merged = 0.2x。

```
usable 12.2GB − merged 7.25GB(reserve) → archer pool 4.97GB (ratio 0.033, sparse 1.97GB)
技術記憶體 = archer 4.97 + merged 7.25 = 12.2GB = 0.2x ✓
```

**為何這解決而 evict 沒有**：evict 只降「邏輯佔用」、不降實體 NVML（allocator 保留 high-water）；
**reserve 直接把 pool cap 縮下來** → allocator 最多只 malloc 到 4.97GB → 實體降。兩者**互補**：
EvictLayer（P2）讓 cache 邏輯只用 ~1 層 → 所以縮到 sparse 1.97GB 還跑得動；reserve 把實體
cap 縮死 → NVML 真的降。

**改動（純 Python）**：`loader.compute_merged_bytes`（config 算 K×L×expert）；`cli` 預扣
+ `_MERGE_DRAFTS` 判定（specmoe/random_mask 不扣，拿滿 cache）。

**驗證（b=0.2, K=16, merge_during_verify=true）**：
- **✅ peak 24 → 16.84GB**（archer 12→4.97，省 ~7GB）。
- **✅ 跑得動**：sparse 1.97GB + EvictLayer（cache ~0.5GB）→ 13 題完整、無 OOM、無 crash。
- **✅ 技術記憶體 = 12.2GB = 0.2x**——誠實的 scarce-VRAM 模擬達成（peak 16.84 = 技術 12.2 +
  context 4.6；context 是固定 overhead、排除）。
- merged 在 torch 是**軟上限**（transient 理論可多一點）；實測沒爆，**暫不需升級 pool buffer
  硬上限**。

### P4 — overlap：merge ‖ fetch
- merge 改用 `get_resident_expert_weights` 讀 resident tensor、在 **side CUDA stream** 累加，
  不同步直到下個 draft 需要 → 藏進後續層的 fetch。
- **flag**：`merge_overlap`。
- **驗收**：per-cycle 牆鐘再降；merge 不出現在 critical path（profile 佐證）。

---

## 6. 風險 / 決策點

1. **overload-eviction 競態（P3 的根本阻擋，已實測）**：低 b 下 verify(batch>1) 用完即 evict
   → mid-forward merge 撞並行 eviction。必須先 P1（降 cache 壓力）+ P2（pin top-M）。這是
   §2「archer 在完整 forward 序列中途不能被插入操作」的再現。
2. **side-stream residency 競態（P4）**：side-stream merge 未完時 expert 被 evict → 持 C++
   lock（序列化）或「pin until merged」。先 P3 同步版建正確性基準，再上 overlap。
3. **flush 的 C++ headroom（P1）**：`del` torch tensor 不還 archer pool；要 C++ method 真正
   把 bytes 還回 sparse cache 額度。
4. **拆批 dispatch 等價性（P2）**：兩次 dispatch 輸出相加須 = 原本一次 → bit-exact 驗。
5. **驗證環境**：必須 `vram_budget_ratio` ≈ 0.2–0.35；b≥1 量不到（無 eviction/overload）。

---

## 7. 接線位置

| 位置 | 現狀 | 哪個階段長肉 |
|---|---|---|
| `OffloadMergeEngine.on_draft_end` | `pass` | **P1**：flush merged + 還 archer headroom |
| `dispatch_local`（expert_executor.py） | arange 順序 enqueue | **P2**：non-merge 先、merge 後 |
| C++ `ExpertDispatcher` | 有 merge/resident 讀；deadlock 已修 | **P1** evict-headroom method、**P2** evict 控制/pin |
| `OffloadMergeEngine.on_verify_layer` | **P3 已實作**（逐層 resident merge） | P1/P2 就位後解除阻擋、過測 |
| `drafts/base.py` `_refresh_layer` | **P3 已抽出**；`refresh` 已支援 during_verify skip | — |
| `qwen3._route_offload` 尾 | **已呼叫 on_verify_layer** | P3 生效點 |
| `OffloadMergeEngine.build` | 委派 `build_weighted_avg`（cpp resident merge） | **P4** 改 side-stream |

新增各階段 flag 的共用檔（與 `merge_offload` 同模式）：`cli.py`（RunConfig + offload_cfg
讀取）、`controller.py`（傳進 engine）。engine 以各自的 `self.<flag>` 內部切換，**其他
method 不受影響**。
