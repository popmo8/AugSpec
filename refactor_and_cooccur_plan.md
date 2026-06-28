# Refactor + Co-occurrence Clustering / Merge-Cache 實作計劃

> 兩個目標綁在一起寫:**(A) 架構重構**(把寫死的東西抽成 strategy、把放錯地方的搬回去)、
> **(B) 新功能**(co-occurrence 分群 + CPU pairwise-merge cache)。B 需要的接縫剛好
> 是 A 建立的,所以 **A 先做、B 疊上去**。全程維持「一個 YAML 一個實驗」的跑法,
> 不動 adapters、controller、specbench、offload C++ 引擎核心的數值行為。
>
> **adapter 去重取消**:adapter 維持原狀,forward / build_weighted_avg / hf-loop 的重複
> 保留不動。
>
> **驗證鐵則**:每個「行為不變」的重構階段跑完都要對既有 config 驗 **MAT/數值不變**
> (bit-exact 路徑用 q5 對照),才進下一階段。

---

## 0. 狀態總覽

| 階段 | 內容 | 風險 | 狀態 |
|---|---|---|---|
| **A0** | `AUG_CLUSTER_UNIFORM` 群內 uniform merge 實驗開關 | 低 | ✅ DONE；**實驗結論:uniform 掉 6.6pp,群內預設用 freq(見 0.5）** |
| **A1** | draft「名單事實」移到類別屬性，cli 去三重列舉 | 低 | ⬜ TODO |
| **A2** | 整理 merge 入口（**固定線性加權平均，不抽 strategy**）+ 預留 cache 包裝點 | 低 | ⬜ TODO |
| **A3** | `clustering/` strategy registry（freq_slice 原樣搬入，YAML 可選） | 中（碰 merge/cache 路徑） | ⬜ TODO |
| **A4** | env-var → YAML（merged_backend / early_pin / no_overload / cluster_uniform） | 低 | ⬜ TODO |
| **A5** | 把放錯地方的搬回去:specmoe forward 出 adapters/、bmm helper 拆出 | 中（純搬移） | ⬜ TODO |
| **B1** | co-occurrence 統計捕捉（scorer + per-layer accumulator） | 中（碰 capture 路徑） | ⬜ TODO |
| **B2** | `CooccurCluster`（**cannot-link / 圖切割**,非 agglomerative — 見 0.5） | 低（新 strategy） | ⬜ TODO |
| **B3** | `CachedMerge`（CPU、bounded LRU、key=成員集合） | 中（記憶體 + 數值） | ⬜ TODO |

**依賴**:A3 → B2(co-occurrence 分群插在 clustering registry 上);A2 → B3(cache 包在線性
merge 入口外);B1 → B2(分群要吃 co-occurrence 統計)。A1 / A4 / A5 與其餘大致獨立。

---

## 0.5 前置實驗結論(2026-06-29,qpc=1 診斷 + q5 partition A/B)

跑了三組診斷,結論直接改寫了下面幾個段落(影響點隨段落標註):

1. **群內 uniform 加權會傷 acceptance(否決「群內可齊頭式」)。** q5 上把群內權重從頻率改成
   齊頭式(`q5_512_tm_unif` vs `q5_512_tm_on`):acceptance 0.583→0.517(−6.6pp / −11.3%)、
   cycle 數 +18.7%、TPS −13%。傷害集中在 routing 有結構的類別(rag/qa −14~17%),math 幾乎不動。
   → **群內權重預設必須是 freq;uniform 只能當消融開關。** 直接推翻 B 段「先拍板」原本的設計 (i)。

2. **co-occurrence 當「併在一起」(must-link)是錯方向 —— 比隨機分群還差。** q5 partition A/B
   (只換分群、群內仍用 freq):dynamic freq_slice 基準 acceptance≈0.59;靜態隨機分群 −8.6pp;
   **凝聚式 co-occurrence(B2 原版)−28~−40pp,明顯輸給隨機。** 直覺:常共現 = 同一 token 各自
   貢獻,硬併成一顆會同時丟掉兩邊資訊。→ **B2 方向要翻成 cannot-link(把高共現切開)。**
   (cooccur 仍在補完整 q5 數字,但 −28pp 以上的差距已穩健。)

3. **逐 cycle 的 selected set 變動大、但有穩定底盤。** 相鄰 cycle Jaccard≈0.46(約 38% 換新),
   沒有「每 cycle 都在」的硬核(僅~1 顆),但約 17/24 顆會出現在過半 cycle。→ **逐 cycle 的成員
   集合無法快取;靜態 per-layer 分群才是 cache 的前提**(影響 A3 的分群生命週期與 B3)。

> 注:`statfreq` / `cooccur_bal` / `cooccur` 與其 env 標籤 harness(`AUG_CLUSTER_LABELS`、
> `scripts/make_cluster_labels.py`)都是**一次性對照,不納入重構**。它們只負責產出上面的結論;
> registry 只保留 `freq_slice` 與之後的 cannot-link `cooccur`(見 A3 / A4)。

---

## 1. 目標 codebase 結構

```
aug_spec/
  adapters/      # 維持原狀(不去重);只把 bmm helper 與 specmoe forward 移出(A5)
  drafts/        # draft 策略 + specmoe substitute forward(A5 搬入)
  clustering/    # 新:ClusterMethod registry — freq_slice + cooccur(A3 / B2)
  merging/       # 新:固定線性 merge 入口 + CachedMerge(A2 / B3,非 registry)
  kernels/       # 新:bmm helper(A5 從 adapters/base.py 拆出)
  runtime/       # loader, specbench, phase, offload_merge, scorers, profile
  config.py      # 新:RunConfig 從 cli.py 拆出(隨 A4)
  controller.py
  cli.py         # parse → build(strategies 注入)→ run
```

**merge 永遠是線性加權平均,不可配置** —— 未來所有 merge 變體都只是「權重不同」,由
clustering / 群內權重決定,不需要 swappable merge 演算法。所以只有 **clustering** 一個新
registry 對應「未來會長新成員」的維度;`merging/` 只放固定的線性 merge + 可選 cache,不是
registry。YAML 用 `cluster.name:` 選分群。

---

## A. 重構

### A1 — draft 名單事實移到類別屬性
**問題**:cli.py 用「寫死的名字清單」判斷某 draft 屬於哪類,散在三處(`_MERGE_DRAFTS`、
`count_top_k` 自動填的 tuple、registry):
```python
# cli.py 現況
_MERGE_DRAFTS = frozenset({"count","topm_count","softmax","prefill_count",...})
if cfg.draft_name in ("count","pruned_count","topm_count","prefill_count",...):
    draft_args["count_top_k"] = adapter.default_count_top_k(model)
```
→ 加新 draft 時必須記得回這兩個清單各補一次,否則悄悄出錯(不預扣 VRAM / 拿不到
count_top_k)且不報錯。

**做法**:把這些事實變成 draft 類別的屬性,cli 改成問類別:
```python
class DraftStrategy:
    holds_merged_residency: bool = False   # ScoreBasedAvgDraft 系列 = True
    needs_count_top_k: bool = False        # count 系列 = True
# cli: if draft_cls.needs_count_top_k: draft_args["count_top_k"] = ...
```
新 draft 自帶旗標,cli 永不再改。
- **驗證**:既有所有 config 行為不變(只是把名單來源換成屬性)。

### A2 — 整理 merge 入口(固定線性,不可配置)
SVD merge 已刪除,**merge 永遠是線性加權平均**。未來變體只是「權重不同」(由 clustering 與
群內權重決定),不需要 swappable 演算法,因此**不抽 MergeMethod / 不做 registry**。這步只把
入口收乾淨,並預留 B3 的 cache 包裝點:
```python
# merging/linear.py  -> 收斂現有 adapter.build_weighted_avg / offload engine.build 的入口
def linear_merge(adapter, block, member_ids, weights) -> dict[str, Tensor]: ...
```
- **接線**:`_build_one` 已是「engine.build → adapter.build_weighted_avg」單一線性路徑
  (SVD 分支已移除),維持現狀;cache(B3)之後包在這個入口外面。
- **驗證**:既有 config MAT 不變(SVD 移除後已驗過 import/實例化;跑一個 q5 確認數值)。

### A3 — `clustering/` strategy registry(★ B2 的前置)
**現況**:怎麼分群寫死在 `drafts/base.py:_assign_clusters`(frequency-slice)。要加
co-occurrence 分群就得改那個 method / 加 if 分支。**(0.5 的 partition A/B 已證實:換分群
acceptance 差很大 —— 隨機 −8.6pp、co-occurrence −28~−40pp。分群確實值得抽成 registry,A3 不是
過度設計。)**

**做法**:把「分群」抽成可選策略,跟 `draft:` / `adapter:` 一樣用 YAML 選。**介面除了逐 cycle 的
`assign`,要多一個比 per-cycle 粗的生命週期 hook**——因為未來的 cooccur(B2)是在 window/EMA 上
累積、每隔一段才重算分群(不是每 cycle),需要一個「每層(或每 window)準備一次」的入口:
```python
# clustering/base.py
class ClusterContext:           # 一個「統計袋子」,method 各取所需
    active: list[int]
    weights: list[float]        # 逐 cycle count(freq_slice 用)
    cooccur: Optional[Tensor]   # [n,n] 共現(B1 填,cooccur 用)
    l2dist: Optional[Tensor]    # 既有 specmoe 的距離,未來可共用
class ClusterMethod:
    def prepare(self, adapter, blocks) -> None: ...   # 每層/每 window 算一次(cooccur 用;freq_slice no-op)
    def assign(self, ctx: ClusterContext, K: int) -> list[list[int]]: ...
# clustering/freq_slice.py  -> 把現有 _assign_clusters 原樣搬進來(動態,prepare 為 no-op)
```
```yaml
cluster: {name: freq_slice}   # 預設;之後可換 cooccur
```
- **接線**:`_cluster_and_build` 改呼叫 `self.cluster_method.assign(ctx, K)`;A0 的群內
  uniform/freq 開關順勢變成 method 參數(見 A4 的 `cluster.within_weight`)。
- **要清掉的實驗 hack(不收編)**:0.5 的 partition A/B 為了快,在 `_assign_clusters` 塞了一條讀
  標籤檔的分支(`AUG_CLUSTER_LABELS` + `_load_static_labels`)。**那批一次性對照(statfreq /
  cooccur_bal / cooccur)都不納入 registry**;A3 要把這條 inline 分支與 `_load_static_labels`
  移除,讓 registry 乾淨地只留 `freq_slice`(預設)+ 之後的 cannot-link `cooccur`(B2)。
  (診斷 dump 仍需 `layer_idx`,該串接保留;`scripts/make_cluster_labels.py` 與標籤檔屬實驗產物,
  不進主程式。)
- **驗證**:freq_slice 跑既有 K=16 config,分群結果與 MAT 與現在一致。

### A4 — env-var → YAML
現在 import 時讀 `os.environ` 的實驗旋鈕收進 YAML(env 僅當 override):

| 現 env | YAML key |
|---|---|
| `AUG_MERGED_BACKEND` | `model.offload.merged_backend` |
| `AUG_EARLY_PIN` | `draft.early_pin` |
| `AUG_NO_OVERLOAD` | `model.offload.no_overload`（C++ 仍讀 env,由 loader 設) |
| `AUG_CLUSTER_UNIFORM` | `cluster.within_weight: freq \| uniform`（**預設 freq**;uniform 僅消融用,見 0.5） |

- **動機**:一個 YAML 完整描述一次 run(可重現);消除「config 一樣、env 不同」的混淆。
- **不收進 YAML、且要清掉的**:`AUG_CLUSTER_LABELS`(0.5 partition A/B 的一次性 harness)隨 A3
  一起移除,不升級成 YAML;statfreq / cooccur_bal / cooccur 標籤檔不保留。
- **不收進 YAML、但保留的**:純診斷 dump(`AUG_DUMP_CLUSTER_WEIGHTS`、`AUG_DUMP_ACTIVE_SET`)
  留在 env 即可 —— 它們是離線分析的旁路,不是 run 的設定。
- **驗證**:把現有靠 env 跑的 config 改成 YAML 欄位,結果一致。

### A5 — 把放錯地方的程式碼搬回該在的地方(純搬移)
兩件「程式碼住錯檔案」:
1. **SpecMoE 的 forward 住在 adapter 裡**:`_topk_substitute_forward` / `_specmoe_engine_bmm`
   在 adapters/base.py,但它是 **SpecMoE draft 的邏輯**,不是模型 adapter 的。搬到
   `drafts/specmoe.py`(SpecMoeDraft 旁邊)。adapter 只暴露 `gate` / `_dispatch_selected`
   等通用 hook。
2. **adapters/base.py 一檔混太多**:把 bmm helper(`_bmm_swiglu` / `_stack_swiglu_weights`)
   拆去 `kernels/bmm.py`。
- 純搬移、不改任何行為。優先度最低,可最後做。
- **驗證**:specmoe 跑既有 config,MAT 不變。

---

## B. 新功能:co-occurrence 分群 + CPU merge cache

### ⚠️ 先拍板:cache 可重用性決定 merge 語意(**已被 0.5 修正**)
merge 是逐 cycle count-weighted。若 cache key 含逐 cycle 權重 → 幾乎不命中。

**原設計 (i)**(誰一群、群內權重都用穩定共現統計、且群內可齊頭式)**已被 0.5 部分否決**:
A0 的 uniform 實驗顯示「群內齊頭式」掉 6.6pp,所以**群內權重不能是 uniform**。

**修正後 (i′)**:cache 仍成立,但靠的是「**穩定但非均勻**」的群內權重 ——
- 成員集合由**靜態/穩定**統計決定(整層固定,見 A3 的 static lifecycle);
- 群內權重用該成員集合在當前 window 的**穩定 freq**(不是逐 cycle count,也不是 uniform),
  是「成員集合 + window」的確定性函數;
- 因此 cache key = 成員集合(+ window epoch)仍可高命中,且保住了 freq 加權的 6.6pp。

0.5 的 set-shift 也佐證:逐 cycle 成員無法快取(38% 換新),但有穩定底盤 → 靜態分群可快取。

### B1 — co-occurrence 統計捕捉
co-occurrence 需要**每 token 的 top-k 集合**,不只聚合 count。
- **scorers.py**:加 `make_cooccurrence_scorer`:對 softmax 取 top-k → `Σ_token onehot·onehotᵀ` → `[n,n]`。
- **drafts/base.py**:`target_score`(count `[n]`)旁邊加 `self.cooccur: Dict[int, Tensor[n,n]]`,
  用 **window / EMA** 累積(讓它穩定 → 提高 cache 命中)。capture 時一併更新。
- **接線**:把 `cooccur[li]` 灌進 `ClusterContext`。
- **驗證**:不開 cooccur cluster 時行為不變;開了之後印共現矩陣 sanity check。

### B2 — `CooccurCluster`(**方向已翻轉:cannot-link / cut,不是 agglomerative merge**)
**0.5 的結果否決了原本的方向**:把高共現的 expert **併在一起**(凝聚式 must-link),acceptance
比隨機還差(−28~−40pp)。原因見 0.5:常共現 = 同一 token 各自貢獻,併成一顆會同時丟掉兩邊資訊。

- **clustering/cooccur.py**:在 `ctx.cooccur` 上做**圖切割 / 譜分群**,目標是**把高共現的 expert
  切到不同 cluster**(cannot-link),而非併在一起;盡量讓「常一起 fire」的 expert 分散,使每個
  token 的多顆貢獻能落在不同 cluster、被分別選用。回傳 `list[list[int]]`,介面與 freq_slice 一致。
- **相似度**:用 frequency-weighted 共現(cosine / raw),**不要 lift/PMI**(lift 會放大對輸出
  幾乎沒影響的稀有雙人組,見 0.5 的共現分析)。
- **平衡**:0.5 顯示凝聚式會塌成一個巨群;切割法要保持各群大小相近(K=16)。
- **YAML**:`cluster: {name: cooccur, window: 512, ...}`。
- **先打的關卡(在投入 B1 token 級統計前)**:沿用 0.5 那套**一次性 harness**(env 標籤檔)只測
  「反共現切割」這一個新方向對 acceptance 的影響,贏過隨機與 freq_slice 才值得做完整 dynamic 版;
  harness 與標籤檔測完即丟,不進主程式。
- **驗證**:MAT 對比 freq_slice / random;確認切割版 ≥ 兩者。

### B3 — `CachedMerge`(CPU、bounded LRU)
- **merging/cache.py**:包在固定的線性 merge 入口外(A2):
  ```python
  class CachedMerge:
      def __init__(self, inner_merge, max_items, device="cpu"): ...
      def merge(self, adapter, block, member_ids, weights):
          key = frozenset(member_ids)          # (i′):成員集合(+window epoch),權重是其確定函數
          hit = self.store.get(key)
          if hit is not None: return hit.to(target_device)
          out = self.inner_merge(adapter, block, member_ids, weights)
          self.store.put(key, out.cpu())        # CPU 存、用時搬回 GPU
          return out
  ```
- **記憶體**:n=128 → 子群數可能很大,`store` 必須 **bounded LRU**(只留反覆命中的)。
  (i′) 的靜態成員 + 穩定 freq 權重才會讓常用子群重複命中、cache 小而有效。
- **量測**:印 cache hit-rate / size / 省下的 merge 次數;對比有無 cache 的 TPS。
- **YAML**:`merge: {cache: {enabled: true, max_items: 4096}}`(merge 不可配置,只有 cache 開關)。
- **驗證**:hit 時的 merged 結果 == 重算結果(數值一致);hit-rate 與 TPS 增益。

---

## 2. 建議執行順序

1. **A1**(低風險 cli 清理,行為不變)。
2. **A3**(clustering registry;freq_slice 原樣搬入,並**移除** 0.5 的 `AUG_CLUSTER_LABELS`
   實驗分支,行為不變驗)+ **A2**(整理線性 merge 入口)。
3. **A4**(env → YAML;順手把 A0 的 uniform 開關正式化成 `cluster.within_weight`,預設 freq)。
4. **決策關卡(B2 前置,便宜)**:用一次性 harness 測「反共現切割」vs random vs freq_slice 的
   acceptance(harness 測完即丟,不留 statfreq/cooccur_bal/cooccur 那批)。贏不過 → 重想分群訊號
   (或改試 functional-distance);贏得過才往下。
5. **B1 → B2 → B3**(疊在 clustering 接縫 + 線性 merge 入口上,純新增成員;B2 為 cannot-link 版,
   B3 用修正後的 (i′) 穩定 freq 權重)。
6. **A5**(純搬移,可最後做)。

每步一個 commit;碰數值的步驟(A3/B3)附 q5 對照數據再合。
