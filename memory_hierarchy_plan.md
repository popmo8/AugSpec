# Merged-Expert Memory Hierarchy 計劃（2026-07-04 起草）

> 系統優化:把記憶體切成 **disk / CPU / GPU** 三層。GPU 維持 `vram_budget_ratio: 0.2`;
> CPU 改為「放不下全部 expert」(新預算 c);disk 存完整 checkpoint。
> specmoe 把 CPU 全部拿來當 raw expert cache;topm/hybrid 把一部分 CPU 撥給
> **merged-expert cache**(同一對 pair 下次不用重 merge/重抓 source)。
> 這對應論文 intro 已承諾的 hierarchy-aware pipeline(GPU draft image / CPU merge
> cache / backing store),即 method 第三小節的系統。
> 實作前先讀 `ARCHITECTURE_REVIEW_AND_PAPER_PLAN.md` 守則與本文件 §6 風險。

## 0. 實測基礎（q15 full runs, 75q, mnt=512, T=5, AUG_PROFILE）

| 量 | specmoe | topm 家族(freqslice/hybrid_a100) |
|---|---|---|
| verify fetch 總量 | 49.8 TB / 999 s | 70.8–75.2 TB / 1428–1501 s |
| **draft-side fetch 總量** | **14.5 TB / 292 s（= verify 的 22.5%）** | 3.5 TB / 70 s（幾乎只有首 cycle fallback） |
| host→GPU 有效頻寬 | ~50 GB/s | ~50 GB/s |
| merge(P3) 總時間 | — | 354–371 s |
| kept 駐留率 | 75%（bmm 需 100%） | — |

- Qwen3-30B-A3B expert 尺寸:3×2048×768×2B = **9.44 MB/expert**;128/層 = 1.21 GB/層;
  48 層全部 = **58 GB**（= 新 CPU 預算的分母 Mem_exp）。
- ⚠️ profile 的 `/cyc` 欄被 update_count 的 per-question reset 影響（分母只算最後一題），
  **只用 total 欄**。
- **draft_fetch 的由來（2026-07-04 釐清,別誤讀）**:`draft_fetch` = dispatcher 在
  `profile_phase==1`(draft forward 期間)發出的 fetch(expert_dispatcher.cpp:732)。
  - topm 的 3.5TB:每題第一個 draft cycle merged 未建 → fallback 走 standard routing
    跑真 experts + warmup;其餘 cycle 為 0(merged 住在 reserve,verify evict 不到,
    重建 = GPU merge 不搬權重)。→ merged cache 的跨題 warm start 正好消掉這塊。
  - specmoe 的 14.5TB(pin:true + no_overload 下仍存在)——**2026-07-04 讀碼修正:
    verify 不會踢 pinned**(`FindExpertEvict` 首行 skip pinned,expert_dispatcher.cpp:541;
    no_overload 已關掉無視 pin 的舊路徑)。真正來源是 **kept churn**:每 cycle
    `SetPinned` 整組替換,新進榜 ~8 顆/層(51% churn,`kept_changed 397/cyc`)只被
    標記不被搬運,draft 用到才 on-demand fetch;其中約半數剛好還在 cache(近期變熱門
    → verify 抓過),淨結果 = 駐留 75%。**churn 是結構性的、reserve/pin 都消不掉**:
    specmoe 的 draft 適應 = 權重搬移;topm 的 draft 適應 = 駐留權重的 GPU 重組。
  - dispatcher 邊界行為(讀碼確認):無 unpinned 可踢時 fetch thread 走 2ms timed-wait
    重試迴圈(不偷 pinned);**若 pinned 佔滿 pool 會無限等待(hang)**,今天
    7.25GB<12.2GB 不會發生,但加大 N / 做 reserve 前必須加 pinned-bytes 上限 guard。
  - 對稱事實:specmoe 的 verify 滾動窗口 = 12.2−7.25 ≈ **4.95GB** ≈ topm 的
    archer pool(12.2−7.25 merged reserve ≈ 4.97GB)——兩法 verify 窗口天然對稱。
- moe_infinity 既有機制:`Node.initial_host = DISK_DEVICE`、`HostMemoryPool` 容量 =
  系統 RAM × `HOST_MEMORY_RATIO`(0.8, 編譯期 macro)、`NodeBody` 已有
  `cpu_hit_cnt/cpu_miss_cnt/gpu_*` 計數器、aio/ 目錄有 disk IO 路徑。
  → disk tier 骨架存在,主要工作是「上限 runtime 化 + 驗證 demand-load 路徑 + 新增 merged cache」。

## 1. CPU 預算 c 怎麼設

定義 `c = B_cpu / Mem_exp`(仿 vram_budget_ratio 的 GPU 無關語意,Mem_exp=58GB)。

三個 regime:
- **c ≳ 1.0**:全放得下,disk 永遠不在 critical path → 優化無感,退化成現況。
- **c ≲ 0.2**(~12GB):per-cycle hot set(~13–16 GB/cycle 的 verify 流量對應的
  unique experts)都放不下 → 兩邊 verify 一起 thrash,common-mode 淹沒差異,
  而且 merge source 也常 miss → 我們也痛。
- **有意義的帶:c ∈ [0.3, 0.7]**(17–40 GB):per-cycle hot set 放得下、
  question 級 working set(粗估 27–50 GB,隨 task 而異)放不下 →
  compulsory miss 出現在 routing 轉移處,**draft-side 行為成為方法間的差異項**。

**建議:主設定 c = 0.5(29 GB),sweep {0.35, 0.5, 0.7}。**

公平性規則(論文用):兩法同 c、同 disk、同 vram ratio;specmoe 把整個 B_cpu
當 raw expert cache(它的最優用法);我們 `B_raw + B_merged = B_cpu`。

**Strong-baseline specmoe(2026-07-04 修正版,hierarchy 實驗用)**:讀碼確認
pin + no_overload **已經等效於硬保留區**(FindExpertEvict skip pinned、驅逐改
timed-wait,pinned 不可侵犯)——原提案 (A) kept reserve 不會帶來額外改善,
75% 非駐留的根因是 **churn 不是驅逐**。所以 strong baseline = 現行 pin +
**(B) `draft.early_pin`(已實作)**:verify 進行中預抓下一輪新 kept,把 churn
fetch 與 verify 重疊、移出 draft critical path(bytes 不變、latency 藏掉)。
B 之後仍剩不可消除的 churn 位元組流(~8 顆/層/cycle 真權重必須搬),hierarchy
下它會打 disk——這才是乾淨的對比點。凍結 kept(prefill 後不變)可消 churn
但賠 acceptance,只做消融不當 baseline。
**現行 `load_cpu_source=True` 的整份 CPU model 副本必須關掉**(≈58GB,直接爆預算;
merge 走 GPU resident 路徑,cpu_source 只是 fallback)——新 regime 下
`load_cpu_source=False` 並在 loader 斷言。

## 2. 為什麼這樣切對 specmoe 有優勢

差異項 = **draft-side fetch 流**。實測 specmoe 的 draft 流 = 14.5 TB/run
(kept-N 隨 mask 變動重 pin;kept 駐留率只有 75%);topm 家族只有 3.5 TB
(首 cycle fallback)。CPU 受限後,這條流的 m(c) 比例會打到 disk
(TWCC /work GPFS 有效吞吐約 1–3 GB/s vs H2D 50 GB/s,**每 byte 慢 15–50×**)。

merged cache 給 topm/hybrid 兩個新能力:
1. **跨題 warm start**:within-weight=uniform ⇒ merged tensor 只由
   (layer, {i,j}) 決定、與題目無關 ⇒ **cache 全 run 有效**。今天 topm 每題重現的
   draft_fetch(3.5TB,~47GB/題)是候選目標——但其確切組成尚未診斷(§3.5-3),
   能吃掉多少以診斷結果為準。
   (這也是「群內必須 uniform」的系統級理由:freq 權重逐 cycle 變 → key 含權重
   → 永遠 miss。)
2. **非駐留 source 救援**:group 成員這輪 verify 沒被 route 到(不在 GPU)時,
   從 host 抓 1 顆 merged(9.44MB, PCIe)取代抓 1–2 顆 source(可能從 disk)。

## 3. 系統設計(2026-07-06 細化為實作規格;實作者按 3.0→3.4 讀)

### 3.0 成本邏輯與設計原則(先讀,決定所有優先序)

單位成本(實測/推導,bf16 expert = 9.44MB):

| 操作 | 成本 | 備註 |
|---|---|---|
| GPU merge 1 個 pair(sources 駐留) | ~7 µs | 讀 2×9.44MB + 寫 9.44MB @ ~4TB/s |
| merged cache hit(H2D 9.44MB) | ~190 µs | @ 50GB/s(實測 H2D 有效頻寬) |
| 抓 1 顆 source(host hit) | ~190 µs | 同上;缺 2 顆 = ×2 |
| 抓 1 顆 source(host miss → disk) | ~3–10 ms | @ 1–3GB/s(待 §6-2 校準) |

⇒ **成本階梯(2026-07-06 加入 GPU L1 後)**:
**L1 hit(~0,指標重用)> GPU merge(~7µs)> L2 hit(~190µs)> raw fetch(190µs–10ms)**。

**三條鐵律**:
1. 重用永遠優先於重算,重算永遠優先於搬運:先查 GPU L1(§3.1),miss 且 sources
   駐留才 GPU merge,sources 不駐留才查 CPU L2(§3.2),再 miss 才 raw fetch。
2. L2 的價值只在「source 的 host/disk miss」出現時兌現——CPU 不受限時幾乎無感,
   **收益全部來自避開 disk**;L1 的價值(省 merge 重算 + 跨題重用)則**任何 regime
   都存在**。實驗解讀時分清楚兩者。
3. 一切 D2H(write-back)走 side stream + event,絕不擋 verify 的 H2D
   (PCIe 全雙工,反向不搶頻寬)。

### 3.1 GPU L1:merged reserve 本身就是 cache(delta rebuild;**優先實作**)

動機(2026-07-06,回應「CPU cache 沒什麼時間被用到」):不加任何新 GPU 記憶體
——現有 merged reserve(K=16/層,7.25GB)每 cycle 整層重建,而 uniform 權重下
merged 內容只由 member set 決定 ⇒ **pair 重現時重建 = 重算相同內容**。把 slots
變成內容定址、只重建變動的 group:

- **為什麼不做「額外的 GPU cache」**:每多 1 pair/層 = 453MB 總量,只能從 verify
  的 4.95GB 窗口挖(−9%/pair);且 GPU cache 屬 draft-path 記憶體,會被挑戰
  Eq. (3) 的 ρ 預算。delta-rebuild 的 ρ 記帳不變(仍 K slots)。
  額外 slot 數 Q>K 只做小 ablation(如 Q=20,+1.8GB)驗證邊際價值。
- **實作(~30 行純 Python)**:`_cluster_and_build` 在 merge 各 group 前,由上一版
  multi dict 建 `{frozenset(indices[k]): experts[k]}` 映射(member ids 已存在
  `indices`);hit 直接引用舊 tensor dict,miss 才走 `_build_one`(→ §3.0 階梯
  繼續往下)。masses/weights 每 cycle 重算(免費,只是純量)。
- **跨題不清**:`controller.reset()` 改為 merge 家族保留上一題最後的 multi dict
  (內容定址、題目無關,永不 stale)——新題首 cycle 重現的 pair 直接 hit。
  這是 §3.5-3 謎題(每題重現的 3.5TB draft_fetch)的候選解,診斷後驗證。
- **⚠ engine_bmm 審計(實作前必查)**:C++ DispatchBmm 的 merged 駐留是否支援
  「部分更新 + 跨題保留」(讀 merge_experts_local / flush 邏輯)。若 C++ 側每
  cycle 全量重灌:先在 Python 層 skip 重算、仍全量上傳(省 merge 不省上傳),
  C++ 部分更新做為 v2。
- **上限誠實計**:省 merge(P3) 354–371 s/run(wall ~5–6%)× skip 率 + 部分
  per-layer sync/evict——單獨是個位數 % TPS。**它的主要價值是把 pair 重現率
  h(λ) 變成一級公民指標**(L1 hit rate = h),而 h 正是 co-occur/hybrid 低 λ
  在最佳化的量(§4)——co-occur 的機制效用因此不依賴 CPU 受限前提,今天就可量。
- **metrics**:`l1_hits / l1_misses / merges_skipped / merge_us_saved` 進
  profile 與 summary.json。
- **下游應用**:L1 的內容定址 slots 是 `reconstruct_fetch_plan.md`(2026-07-06)
  的前置——駐留的 C=0.5A+0.5B 可在 verify 用 `B=2C−A` 重建缺席成員、
  取代 fetch(把差異化延伸到 verify 側)。實作 L1 時保持 slot↔member-set
  映射可查詢,重建計劃會用到。

### 3.2 CPU L2:merged cache(Python-first,不動 C++)

**放哪/歸屬**:新類 `MergedExpertCache`,由 `OffloadMergeEngine.__init__` 建立
(`engine.merged_cache`),blocks 經 `block._merge_engine` 可達(既有 back-ref
模式)。單執行緒使用(所有讀寫都發生在 on_verify_layer / refresh 的 Python 路徑),
**不需要鎖**;唯一的非同步是 D2H write-back(side stream + CUDA event)。

**資料結構**:
```python
class MergedExpertCache:
    # key = (layer_idx, frozenset(member_ids))   # 只在 within_weight=uniform 有效
    # val = CacheEntry(tensors: Dict[str, Tensor(pinned CPU)], bytes: int,
    #                  last_use: int, use_count: int, d2h_event: cuda.Event|None)
    entries: "OrderedDict[key, CacheEntry]"       # 全域 LRU 序
    per_layer_bytes: Dict[int, int]; total_bytes: int
    cap_bytes: int                                # merged_cache_ratio × B_cpu
    layer_quota: int                              # cap_bytes / num_moe_layers
```
- **enabled 條件**(建構時決定,任一不滿足 ⇒ 整個 cache no-op):
  `within_weight == "uniform"`(uniform 才內容定址;freq 權重逐 cycle 變,
  key 要含權重 → 永遠 miss,直接禁用並 log 一行警告)、`merge_offload=true`、
  `merged_cache_ratio > 0`。
- pinned CPU tensor 用 `t.detach().to("cpu", non_blocking=True)` 進預先
  `torch.empty(..., pin_memory=True)` 的 buffer;**bytes 記帳以 pinned buffer
  實際大小為準**,插入前先檢查配額、必要時先 evict。

**接線點(唯一要動的既有函式)**:`drafts/base.py::ScoreBasedAvgDraft._build_one`
——所有 merge 都經過它(K=1 與 K>1 的每個 cluster 都是)。簽名加 `layer_idx: int = -1`
(由 `_refresh_layer`/`_cluster_and_build` 傳入;`-1` 表未知 ⇒ 不進 cache)。
```python
def _build_one(self, adapter, block, weights, layer_idx=-1):
    member_ids = [i for i, w in enumerate(weights) if w > 0.0]
    engine = getattr(block, "_merge_engine", None)
    cache = getattr(engine, "merged_cache", None)
    key = ((layer_idx, frozenset(member_ids))
           if cache is not None and cache.enabled and layer_idx >= 0 else None)
    # 鐵律 1:sources 皆駐留 → GPU merge(現況路徑),完成後 write-back
    if key is not None and not self._sources_resident(block, member_ids):
        hit = cache.get_to_gpu(key)        # H2D + sync;None = miss
        if hit is not None:
            return hit
    out = linear_merge(adapter, block, member_ids, weights)
    if key is not None:
        cache.put_async(key, out)          # D2H(side stream)+ 記帳 + evict
    return out
```
- `_sources_resident(block, ids)`:問 dispatcher 這些 (layer, expert) 是否
  GPU 駐留。**C++ 已有查詢介面與否要先確認**(讀 `expert_dispatcher.h`;
  若無,加一個 `GetResidentMask(layer)` 的 pybind,~15 行,是本節唯一可能的
  C++ 增項)。查不到就保守回 True(= 永遠走 GPU merge,行為退化成現況,安全)。
- **engine_bmm 相容性(實作前必查)**:`merged_backend=engine_bmm` 時 merged
  權重會被送進 C++ DispatchBmm 的駐留區。cache-hit 路徑回傳的 GPU tensor dict
  必須走**與 `linear_merge` 產物完全相同的下游消費路徑**(讀
  `adapters/base.py::build_weighted_avg` 的 offload 分支 + `merge_experts_local`
  call site 確認 dict 格式與上傳點),否則 bmm 讀不到。驗收:cache-hit run 的
  `draft_dispatch` profile row 仍正常、輸出與 no-cache run 的 AccR 同量級。

**admission(擋污染)**:預設「二次機會」——第一次見到某 key 只記入
`seen: Dict[key, int]`(輕量計數,不佔 cache),第二次出現才真正收進 cache。
`seen` 每題 `reset()` 清空,cache 本體**跨題不清**(這就是 warm start)。
可選旋鈕 `admit_min_cooccur`:pair 成員的 C_ℓ[i,j] ≥ 門檻才收(v2 再做)。

**eviction**:插入時若 `per_layer_bytes[ℓ] > layer_quota` 先驅逐同層 LRU;
再檢查 `total_bytes > cap_bytes` 驅逐全域 LRU。驅逐只是釋放 pinned buffer,
無任何一致性後果(內容定址、模型權重靜態、永不 stale)。

**尺寸感**:entry = 9.44MB(pair merge 後與單 expert 同形)。c=0.5、ratio=0.2
⇒ 5.8GB ≈ 每層 ~12.8 個 pair。K=16/層、每 cycle 至多 16 個 group(多為 pair),
所以配額 < 一個 cycle 的全部 pair——**hit rate 取決於 pair 集中重現**,
這正是 co-occur/hybrid 低 λ 的用武之地(§4)。

**metrics**:`hits / misses / bytes_h2d / bytes_d2h / evictions / denied_admission`
計數器掛在 cache 上,`cli._dump_profile` 加一段列印;另把命中率寫進
`summary.json["merged_cache"]`(論文表直接取)。

### 3.3 raw expert host cache(C++ 小改 + 審計)

**改動點**:`core/memory/memory_pool.cpp:~67`
`memory_capacity_ = GetTotalSystemMemory() × HOST_MEMORY_RATIO`
→ 改為:env `ARCHER_HOST_MEM_BYTES` 有設就用其絕對值,否則維持原式。
loader(`runtime/loader.py::load_offload`)在 `MoE()` 前由
`cpu_budget_ratio × Mem_exp − merged_cache_bytes` 換算設定(沿 A4 的
「YAML→env、env 可 override」模式;**merged cache 的錢從同一份 B_cpu 扣**,
記帳才誠實)。

**實作前必做的三個審計**(結論寫回本文件,別跳過):
1. `HostMemoryPool::AllocateMemory` 的**所有 call site**:pool 裡除了 expert
   host 副本還有誰(dense 參數? KV? buffer?)。若共用,B_cpu 的語意要改成
   「expert 部分 = c×Mem_exp,非 expert 底噪實測後外加」,否則低 c 會先餓死
   非 expert 分配。
2. **host-miss 行為**:AllocateMemory 拿不到記憶體時回傳什麼、caller 怎麼辦?
   有沒有 host→disk 的驅逐路徑,還是只有 load 期一次性從 disk 進 host?
   (`model_topology.cpp::SetDevice` 的 DISK 分支 + `aio/` + prefetch 這條鏈。)
   **若 demand-load 不存在,fallback 方案**:host-miss 時注入校準過的延遲
   (模擬 disk),論文明講 simulated——先讓實驗能跑,真實 IO 之後補。
3. **pinned-full guard**(前次讀碼發現的 hang 邊界):在 SetPinned 或 fetch
   等待迴圈加斷言/上限——`pinned bytes ≤ device pool − margin(≥1 層 48 slots)`,
   違反直接 FATAL 出清楚訊息,不要無限等。

**smoke 程序**(§6-4,過了才准做 §3.2 的 CPU L2;§3.1 的 GPU L1 不受此限,
隨時可做):tiny 上限(如 8GB)× 1 題 × topm 與
specmoe 各一——驗 (a) `cpu_miss_cnt > 0`(NodeBody 已有計數,接出來印),
(b) 輸出合理(AccR 不歸零),(c) 無 hang(watchdog 20 分鐘),
(d) 對照不限 c 的同 config:AccR 差 < 非確定性帶。

**metrics**:AUG_PROFILE 加 rows `disk_fetch_n/us/bytes`(host-miss 補位的
讀量/時間)與 `cpu_hit/cpu_miss`(NodeBody 現成計數彙總)。

### 3.4 YAML / 管線接線(全表)

```yaml
offload:
  cpu_budget_ratio: 0.5        # B_cpu/Mem_exp;未設 = 不限(現況,disk 不參與)
  merged_cache_ratio: 0.2      # B_merged/B_cpu;僅 merge 家族;0 = 關(預設)
```
- `cli.py::RunConfig`:加 `cpu_budget_ratio: Optional[float]`、
  `merged_cache_ratio: float = 0.0` 兩欄(from_yaml 解析 + 傳入
  `load_offload(...)` 與 `Controller(...)→engine`)。
- 公平性斷言(cli.py,兩行):`cpu_budget_ratio` 有設時強制
  `load_cpu_source=False`(58GB 副本爆預算);specmoe(非 merge 家族)下
  `merged_cache_ratio` 必須為 0,否則 raise。
- `configs/README.md` offload 段補兩欄;PROJECT_GUIDE 同步。

### 3.5 驗收與量測(每項都要能寫進論文)

1. **單元測試**(`tests/unit/test_merged_cache.py`,純 CPU):budget 記帳、
   per-layer 配額驅逐、二次機會 admission、uniform-only 禁用、key 穩定性
   (同 member set 不同順序 → 同 key)。
2. **等價性 smoke**:cache on vs off、c 不限,同 config 各 1 run——AccR 差
   落在非確定性帶內(cache 只該影響速度不影響數值;merged 內容 bit 級相同,
   路徑差異僅浮點搬運)。
3. **待解謎題(量測前先弄清)**:topm 現況的 3.5TB draft_fetch **確切組成**
   ——per-question ~47GB 的規模說明它每題重現,但 merge_during_verify 下
   prefill 就該建好 cache,「首 cycle fallback」解釋存疑。拿 1 題 + DLOG/
   profile 分解,搞清楚它是什麼、cache 能不能吃掉它,再寫進 §5 的收益模型。
4. **主量測矩陣**(§6-7):c ∈ {0.35, 0.5, 0.7} × ratio ∈ {0, 0.1, 0.2}
   × {strong-specmoe(ep 按 §6-0 實測定), topm(freqslice), hybrid(λ 按
   cosine sweep 的 knee,現值 cos_a25)},q15 協定 r1–r3;報 TPS/AccR/
   hit-rate/disk bytes。ratio=0 那欄就是「三層但無 merged cache」的 ablation,
   分離「hierarchy 壓力」與「cache 收益」兩個效應。

## 4. 對 hybrid 的好處

**co-occur 在這個前提下第一次有明確的系統角色。** raw λ-sweep 顯示 co-occur
不幫 acceptance(a50 最差、blend 無 knee);但 **cache hit rate 取決於 pair 穩定性
與時間局部性**,而 C map 正是往「現在一起 fire 的 experts」拉 pair:
- λ 低 → pair 跟著 temporal locality 走 → 同題內重現率高 + source 駐留率高
  (路徑 3 便宜)→ h(λ) 高;
- λ=1 → pair 由凍結的 A + 變動的 𝓜_ℓ 決定,重現率未知(可能也不差,要量)。

λ 從單目標(AccR)變雙目標(AccR × h(λ) × miss 成本)。

**2026-07-06 兩個更新讓這個故事更強**:
1. **cosine sweep 已出現 acceptance knee**:`hybrid_cos_a25 = 0.6724`(n=3)
   勝過兩個端點(a00=0.6603、a100=0.6635)——去掉頻率偏差後 co-occur 在
   acceptance 上就有正貢獻了(見 hybrid_cluster_plan.md 關鍵結論),
   不再需要「掉 AccR 換 hit-rate」的辯護。
2. **GPU L1(§3.1)讓 h(λ) 不依賴 CPU 受限前提**:L1 hit rate = pair 重現率 h,
   任何 regime 都在兌現(省 merge 重算 + 跨題重用);CPU L2 的 h 收益則在
   hierarchy 壓力下疊加。co-occur 的系統效用因此有兩個可量測的落點。

**先做免費驗證(不用寫任何 C++)**:`AUG_DUMP_PAIRS` + `analyze_pair_reuse.py`
已存在——對 λ∈{0,0.25,0.5,1}(cos 變體)各跑 1 個 dump run,直接量 pair
重現率 h(λ) 的上界(= L1 可 skip 的 merge 比例上界)。若 h(cos_a25) ≫ h(a100),
「co-occur → 重現率 → L1/L2 收益」全鏈成立;若差不多,誠實地把 L1/L2 當
topm 通用優化寫,hybrid 的賣點回到 cos_a25 的 acceptance knee。

## 5. 預期節省(模型 + 待校準參數)

每 run 額外 disk 時間(fetch 無法被 compute 隱藏的部分,保守全計):

```
T_disk(方法) ≈ [V_verify × m_v + V_draft × m_d] × (1/BW_disk − 1/BW_pcie)
```

差異項是 draft 流。帶入實測(V_draft: specmoe 14.5TB vs 我們 ~0(cache 吸收)),
假設 c=0.5 → m_d ≈ 15%(待校準)、BW_disk = 2 GB/s、BW_pcie = 50 GB/s:

- specmoe draft-side 額外 disk 時間 ≈ 14.5TB × 0.15 × (0.5−0.02) ≈ **~1,050 s/run**;
- 我們 draft-side ≈ 0;另省掉自己的首 cycle 流(3.5TB→hit,省 ~70–800s 視 miss 率)。
- specmoe 單 run wall ≈ 6,800s ⇒ **相對 TPS 優勢粗估 +10~25%**(在現有 AccR 差距之上),
  m_d 或 disk 更慢時更大;c→1 時趨近 0。

**兩個必須先校準的量**(否則以上只是量級):
1. `BW_disk`:在計算節點上對 offload_output 實測順序讀(fio/dd, sbatch)。
2. `m(c)`:host pool 上限跑通後,從 cpu_miss 計數直接讀,對 c 掃描。

誠實 caveat:verify 流是 common-mode——若 m_v 很大,兩法一起變慢,
相對優勢百分比會被稀釋;所以 c 的選擇(§1 的帶)本身是實驗設計的一部分,
論文要報 c-sweep 而不是單點。

**⚠️ 2026-07-04 二次修正(early_pin,已讀碼+查數據)**:draft_fetch 的候選根因是
時序洞——新 kept 在 verify 該層被抓時尚未 pin(pin 在 refresh、verify 結束後才更新),
之後被同輪後面層擠出,pin 掛上時人已在 host。`early_pin=1` 在 capture 後、該層
dispatch 前就 pin 新 kept,理論上讓 verify 抓完直接保護。

**但這是推論,未有乾淨實測(2026-07-04 誠實標記)**。手上只有兩筆 specmoe profile,
都不是要的組合:
- `q15_specmoe`(pin+no_overload+**early_pin=0**):draft_fetch **14.5TB**(乾淨但沒開 ep)。
- `exp_earlypin_244275`(pin+**overload 開著**+ep A/B):draft_fetch 1900→**1708GB**,
  只降 ~10%。**這筆被 overload confound**(overload_wait 191/168s;overload 路徑
  無視 pin),不是 ep 的乾淨測試,且是 mnt=128/qpc=1 小 run。
→ **strong-baseline specmoe(pin+no_overload+early_pin=1)的 draft_fetch 目前未知。**
§5 的 +10~25% 只對 early_pin=0 成立;strong baseline 的差異項要**先量再說**。
**待辦(§6-0,最高優先):跑 `AUG_EARLY_PIN=1` 的 q15_specmoe(+ ep=2 對照),
看 no_overload 下 draft_fetch 是否真的趨近 0。** 三種可能結果:
(i) 趨近 0 → 差異化改靠 acceptance + merged cache 跨題 warm start(specmoe 每題
仍重建 7.25GB kept 駐留,我們 cache 命中);(ii) 只降部分 → draft-side 流仍是差異項,
§5 估計按實測 draft_fetch 重算;(iii) 幾乎不降 → time-hole 假設錯,要重新診斷 churn。
h(λ) 與 §6-7 矩陣的 specmoe 對照一律用實測出來的最佳 ep 設定,不打稻草人。

**✅ 2026-07-08 §6-0 實測結果(job 254248;q15、qpc=5、mnt=512、pin+no_overload)**:
結果 = 選項 **(ii)**。

| ep | draft_fetch | kept 駐留 | TPS | AccR |
|---|---|---|---|---|
| 0 | 4494 GB | 75% | 3.26 | 0.4749 |
| 1 | 945 GB | 99% | 3.97 | 0.4751 |
| 2 | 965 GB | 99% | 4.11 | 0.4860 |

- draft_fetch −79% 但未趨近 0:剩 ~0.95TB 是 kept churn 的真權重搬運
  (kept_changed ~394 顆/cycle),hierarchy 下這條流打 disk = 乾淨對比點。
  §5 的估計要按 0.95TB(非 4.5TB)重算。
- 駐留 75%→99% ⇒ specmoe 的 kept-N bmm 全程 engage(draft 計算從
  expert_forward 移到 draft_dispatch),這是 TPS +22~26% 的主因。
- AccR 不受 ep 影響(substitute table 相同;ep0 vs ep1 差 0.0002),ep2 的
  +1.1pp 與 ep2−ep1 的 TPS 差都在 run-to-run noise 內(同設定 TPS spread ~0.28),
  單跑不可分。
- → **strong-baseline specmoe = pin + no_overload + `early_pin: 2`**(§6-7 矩陣用;
  若要嚴格分 ep1/ep2 需多次重跑取平均)。注意 `q15_specmoe_r1–r3` 是 ep0 跑的:
  AccR 可沿用,TPS 低估。

## 6. 實作順序與風險

| # | 事項 | 大小 | 風險 |
|---|---|---|---|
| 1 | AUG_DUMP_PAIRS λ∈{0,.5,1} 量 h(λ)(免費,先做) | 3 jobs | 無 |
| 2 | 節點 disk 頻寬校準 | 1 job | 無 |
| 3 | HostMemoryPool 上限 runtime 化 + YAML `cpu_budget_ratio` | C++ 小 + rebuild | 低 |
| 4 | 小上限 smoke:demand-load 路徑 × 我們的 evict 改動(overload 路徑已於 2026-07-08 整個刪除,見 `remove_overload_plan.md`;此項只需驗 demand-load × FindExpertEvict) | 1 job | **高——最可能出 race/deadlock 的點** |
| 4.5 | **GPU L1 delta-rebuild(§3.1,~30 行 Python;不依賴 #3/#4,可隨時先做)** | Python 小 | 低(engine_bmm 審計先行) |
| 5 | Python CPU L2 merged cache(§3.2)+ AUG_PROFILE 新 rows | Python 中 | 中 |
| 6 | `load_cpu_source=False` 斷言 + 記帳審計(B_cpu 真的只有 c×58GB) | 小 | 低 |
| 7 | c × merged_cache_ratio × {specmoe, topm, hybrid λ*} 矩陣實驗 | 多 jobs | — |

注意:#4 過了才做 #5;#1/#2 隨時可先跑。改 C++ 記得 sbatch rebuild
(`moe_infinity/rebuild.sh`),不要在 login node 編譯。

## 7. 狀態追蹤

| 項目 | 狀態 |
|---|---|
| §6-0 early_pin probe(ep=0/1/2 draft_fetch) | ✅ job 254248 完成,2026-07-08 分析:選項 (ii),draft_fetch 4.5TB→~0.95TB(−79%)、TPS +22~26%;strong baseline = ep2(詳見 §5 末實測結果表) |
| §6-1 h(λ) 免費驗證 | ⬜ |
| §6-2 disk 頻寬校準 | ⬜ |
| §6-3 host 上限 runtime 化 | ⬜ |
| §6-4 demand-load smoke | ⬜ |
| §6-5 merged cache | ⬜ |
| §6-6 cpu_source 關閉 + 記帳 | ⬜ |
| §6-7 對比矩陣 | ⬜ |
| 論文 method 第三小節(引用本設計) | ⬜ |
