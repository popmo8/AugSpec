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
| **A1** | draft「名單事實」移到類別屬性，cli 去三重列舉 | 低 | ✅ DONE（drafts 旗標 + cli 去 `_MERGE_DRAFTS`；順手修正 pruned_count 漏列） |
| **A2** | 整理 merge 入口（**固定線性加權平均，不抽 strategy**）+ 預留 cache 包裝點 | 低 | ✅ DONE（`merging/linear.py`；`_build_one` 委派） |
| **A3** | `clustering/` strategy registry（freq_slice 原樣搬入，YAML 可選） | 中（碰 merge/cache 路徑） | ✅ DONE（`clustering/` registry；移除 `_assign_clusters`/`AUG_CLUSTER_LABELS`；freq_slice 對 25k 組與舊式逐一相同；q5 bit-exact 驗證中 job 248144） |
| **A4** | env-var → YAML（merged_backend / early_pin / no_overload / cluster_uniform） | 低 | ✅ DONE（`model.offload.no_overload`/`merged_backend`、`draft.early_pin`、`cluster.within_weight`；env 仍為 override） |
| **A5** | 把放錯地方的搬回去:specmoe forward 出 adapters/、bmm helper 拆出 | 中（純搬移） | ✅ DONE（bmm→`kernels/bmm.py`；specmoe forward + `pairwise_l2`→`drafts/specmoe.py`，adapter 改 lazy import） |
| **B1** | co-occurrence 捕捉（`make_cooccurrence_scorer` + 題級累積器，prefill→decode） | 中（碰 capture 路徑） | ✅ DONE（2026-06-30） |
| **B2** | `cooccur_pair`（**must-link 貪婪配對,max size 2**）+ `within_weight: uniform` | 低（新 strategy） | ✅ DONE（2026-06-30；結果見 0.6） |
| **B3** | `CachedMerge`（純成員 key、immutable、bounded LRU；因 uniform 而最乾淨） | 中（offload engine 整合） | ⬜ TODO |

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
   (5 組各 65 題完整,只換分群、群內仍用 freq;baseline acceptance 0.594):
   隨機 0.508(−8.6pp/−14.5%)> 靜態頻率 statfreq 0.420(−17.4pp/−29.3%)> 平衡共現
   cooccur_bal 0.377(−21.7pp/−36.6%)> 凝聚式共現 cooccur 0.205(−38.9pp/−65.5%)。
   **兩個 co-occurrence 變體都輸給隨機,且平衡版 cooccur_bal 仍低於隨機/statfreq → 不是不平衡
   造成的,是訊號方向本身錯。** 直覺:常共現 = 同一 token 各自貢獻,硬併成一顆會同時丟掉兩邊
   資訊。→ **B2 方向要翻成 cannot-link(把高共現切開)。**（⚠️ 此結論已被 0.6 再翻回:token 級
   動態 + uniform 下,must-link 配對其實 ≈ baseline、贏 random。最終採 must-link `cooccur_pair`。）

3. **逐 cycle 的 selected set 變動大、但有穩定底盤。** 相鄰 cycle Jaccard≈0.46(約 38% 換新),
   沒有「每 cycle 都在」的硬核(僅~1 顆),但約 17/24 顆會出現在過半 cycle。→ **逐 cycle 的成員
   集合無法快取;靜態 per-layer 分群才是 cache 的前提**(影響 A3 的分群生命週期與 B3)。

> 注:`statfreq` / `cooccur_bal` / `cooccur` 與其 env 標籤 harness(`AUG_CLUSTER_LABELS`、
> `scripts/make_cluster_labels.py`)都是**一次性對照,不納入重構**,已刪。它們只負責產出上面的
> 結論。registry 現有 `freq_slice`(預設)/ `random` / `cooccur_pair`(見 0.6)。

## 0.6 co-occurrence 配對的正式結果(2026-06-30,q5,token 級動態 + uniform)

把 0.5 的「靜態 proxy」換成正式版(token 級共現、每題 prefill→decode 累積、貪婪 size-2 配對、
群內 uniform)後,**結論對 0.5 反轉**:

| 分群 + 群內權重 | overall AccR | mt_bench(40q,最可信) | vs baseline |
|---|---|---|---|
| baseline:freq_slice + freq | 0.5829 | 0.4711 | — |
| **cooccur_pair + uniform** | **0.5551** | **0.4659**（≈ baseline） | **−2.8pp** |
| random + uniform | 0.5027 | 0.3943 | −8.0pp |

- **must-link 配對在 token 級 + uniform 下其實 work**:cooccur_pair 贏 random+uniform **+5pp**,
  且在最可信的 mt_bench(40q)**≈ baseline**。0.5 那個「must-link 比隨機差」是靜態 proxy 的假象。
- overall −2.8pp ≈ 1–1.5σ(run-to-run 雜訊 ~±0.02–0.04)→ **可能與 baseline 無顯著差**;n=5 的
  子類別(rag/math)雜訊大,別單看。**要定論需重跑 1–2 次取平均。**
- **意義**:uniform 的 acceptance 代價小到可接受,而它換來乾淨可 cache 的 merged(見 B3)。
  → 走 must-link `cooccur_pair` + uniform + cache 這條線。

---

## 1. 目標 codebase 結構

```
aug_spec/
  adapters/      # 維持原狀(不去重);只把 bmm helper 與 specmoe forward 移出(A5)
  drafts/        # draft 策略 + specmoe substitute forward(A5 搬入)
  clustering/    # 新:ClusterMethod registry — freq_slice / random / cooccur_pair(A3 / B2)
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

**做法**:把「分群」抽成可選策略,跟 `draft:` / `adapter:` 一樣用 YAML 選。介面以逐 cycle 的
`assign(ctx, K)` 為主,另保留一個 `prepare(adapter, blocks)` 作為「每層/每 window 準備一次」的
預留 hook(目前 freq_slice/random/cooccur_pair 都 no-op;cooccur_pair 的共現是在 draft 的
`capture` 累積、經 `ctx.cooccur` 餵入,不走 prepare):
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
  移除,讓 registry 乾淨地只留 `freq_slice`(預設);其餘成員 `random` / `cooccur_pair` 之後加。
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

## B. 新功能:co-occurrence pair 分群(+uniform)+ merged-expert cache

> **現況(2026-06-30)**:B1 + B2 已實作為 `cooccur_pair` + `within_weight: uniform`,q5 結果見
> 0.6;B3(cache)待做。B 段方向經 0.6 二度修正:**must-link 配對(把高共現的併成 pair)在
> token 級動態 + uniform 下其實 work**(≈ baseline、贏 random),所以走 must-link,不做 cannot-link。

### ⚠️ 先拍板:uniform → merged 可 cache(設計 (i) 成立,取代 (i′))
cooccur_pair + uniform 下,一個 cluster 的 merged expert
`merged({i,j}) = (E_i + E_j)/2` —— **只跟成員集合有關,與逐 cycle 權重無關,且對整個 run 不變**。

- 0.5 曾擔心 uniform 掉 acceptance(在 freq-slice 分群上 −6.6pp),一度改採 (i′)「穩定但非均勻
  freq + window-epoch key」。
- **0.6 推翻這個顧慮**:在 **co-occurrence 配對分群**上,uniform 的代價小到可接受(overall
  −2.8pp、mt_bench 40q ≈ baseline、贏 random+uniform +5pp,見 0.6)。
- 所以回到最乾淨的 **設計 (i)**:**cache key = 純成員集合(不含權重、不含 window epoch)**,值
  immutable。uniform 是這個乾淨 cache 的前提,acceptance 的小代價換的就是它。

### B1 — co-occurrence 捕捉 ✅ 已實作(2026-06-30)
- **`runtime/scorers.py`**:`make_cooccurrence_scorer(top_k)` — softmax 取 top-k →
  `Σ_token onehot·onehotᵀ` → `[n,n]`。
- **`drafts/base.py`**:`ScoreBasedAvgDraft.cooccur: Dict[int, Tensor[n,n]]`,**每題從 prefill
  累積到 decode**(不是 EMA/window),`capture()` 累加、`reset()` 每題清空。**prefill 的 capture
  本來就會觸發**(target 階段;`PrefillCountDraft` 即靠此),所以不需新 hook。
- **gate**:只有 `cluster_method.needs_cooccur=True` 才累積 → 對其他分群法零開銷。
- **接線**:`_cluster_and_build` 把 `self.cooccur.get(li)` 灌進 `ClusterContext.cooccur`。

### B2 — `cooccur_pair` 分群 ✅ 已實作(must-link,2026-06-30)
- **`clustering/cooccur.py` `CooccurPairCluster`**:在 `ctx.cooccur` 上做**貪婪最大共現配對**——
  反覆併「共現最高、且兩顆都還沒被配」的 pair,直到剩 K=16 群,所以每群是 **pair 或 singleton
  (max size 2)**。`needs_cooccur=True`。
- 搭 `within_weight: uniform`。YAML:`cluster: {name: cooccur_pair, within_weight: uniform}`。
- **方向(對 0.5 再翻一次)**:0.5 用「靜態 per-cycle-set proxy + freq」測,must-link 輸 random
  → 當時改提 cannot-link;但 0.6 用「**token 級動態 + size-2 + uniform**」測,must-link 配對
  **贏 random +5pp、≈ baseline** → 最終採 **must-link 配對**,cannot-link/cut **不需要了**。

### B3 — merged-expert cache(待做;因 uniform 而最乾淨)
包在 A2 的 `linear_merge` 入口外:
```python
# merging/cache.py
class CachedMerge:
    def merge(self, adapter, block, layer_idx, member_ids, weights):
        key = (layer_idx, frozenset(member_ids))   # uniform → 只用成員集合,不含權重
        hit = self.store.get(key)
        if hit is not None:
            self.store.move_to_end(key); return _to_device(hit, block)
        out = self.inner(adapter, block, member_ids, weights)   # = linear_merge
        self.store[key] = _to_cpu(out)              # 存 CPU,bounded LRU
        if len(self.store) > self.max_items: self.store.popitem(last=False)
        return out
```
- **immutable,無需 invalidation**:`E_i` 是固定模型權重、uniform 平均也固定 → `{i,j}` 這顆
  merged **整個 run 不變**,算一次用一輩子。只有 bounded LRU 因容量汰換。
- **關鍵分離**:每題 `reset` 清的是「**共現表**」(決定這題要配哪些 pair),**不是 cache**
  ——`pair → merged` 是 **run-level 永久**的,跨題重用。
- **max size 2 → key space 小**:每層只可能 pair/singleton;共現穩定後反覆命中 →
  **暖機後 ~100% hit、per-cycle merge 成本趨近 0**。
- **接線**:把 `layer_idx` 串進 `_build_one` → cache;**gate 在 `within_weight=="uniform"`**
  (freq 的 merged 依賴權重,純成員 key 會錯;freq 要嘛關 cache、要嘛把量化權重放進 key)。
- **offload 風險(主要)**:offload 的 merge 走 C++ `engine.build`,它在 archer pool 記了帳;
  cache 命中時繞過 engine、直接 CPU→GPU,要確認餵給 draft forward 的格式一致、不讓 residency
  記帳 desync。→ **先在 hf backend 把 cache + 命中率/正確性跑通,再上 offload 驗 engine 整合。**
- **GPU 常駐 vs CPU 存放**:cache 存 CPU(一顆 merged ≈ 一顆 expert ~9MB);每 cycle 只有
  `draft_top_k×層` 的 merged 上 GPU,沿用既有 `cpu_source` 搬運。
- **YAML**:`merge: {cache: {enabled: true, max_items: 4096}}`(merge 不可配置,只有 cache 開關)。
- **量測(重點是 TPS,不是 acceptance)**:hit-rate / distinct pairs built / 省下的 merge 次數 /
  **有無 cache 的 TPS** / hit==重算(數值一致)。論文命題:**「用 ~0–3pp acceptance 換掉幾乎全部
  per-cycle merge 成本」淨 TPS 是否為正**(offload 上 merge/fetch 主導時間,很可能淨正)。

---

## 2. 進度與後續

**已完成**:A1–A5(2026-06-29)、B1 + B2(2026-06-30,`cooccur_pair` + uniform,結果見 0.6)。

**後續**:
1. **重跑定論(便宜,先做)**:cooccur_pair + 一份新 baseline 各重跑 1–2 次取平均,把「overall
   −2.8pp / mt_bench ≈ baseline」的 acceptance gap 釘住(目前單跑、在雜訊邊緣)。
2. **B3 — merged-expert cache**:先在 **hf backend** 把 `CachedMerge` + 命中率 + 「hit==重算」跑通,
   再上 **offload** 驗 `engine.build` 整合與 **TPS 增益**。這步才是 cooccur_pair + uniform 的真正
   報酬(acceptance 已知,cache 換 TPS)。

每步一個 commit;碰數值的步驟附 q5 對照數據再合(注意:因 offload run-to-run 非確定,bit-exact
不可用,改看 aggregate + 多跑取平均,見 0.6)。
