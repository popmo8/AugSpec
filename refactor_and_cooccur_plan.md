# Refactor + Co-occurrence Clustering / Merge-Cache 實作計劃

> 兩個目標綁在一起寫:**(A) 架構重構**(殺重複、把寫死的東西抽成 strategy)、
> **(B) 新功能**(co-occurrence 分群 + CPU pairwise-merge cache)。B 需要的接縫剛好
> 是 A 建立的,所以 **A 先做、B 疊上去**。全程維持「一個 YAML 一個實驗」的跑法,
> 不動 adapters 的 forward 數值、controller、specbench、offload C++ 引擎核心。
>
> **驗證鐵則**:每個「行為不變」的重構階段跑完都要對既有 config 驗 **MAT/數值不變**
> (bit-exact 路徑用 q5 對照),才進下一階段。

---

## 0. 狀態總覽

| 階段 | 內容 | 風險 | 狀態 |
|---|---|---|---|
| **A0** | `AUG_CLUSTER_UNIFORM` 群內 uniform merge 實驗開關 | 低 | ✅ DONE（drafts/base.py `_cluster_and_build`） |
| **A1** | adapter forward wrapper 上提 base（殺 byte-identical 重複） | 低（已逐字相同） | ⬜ TODO |
| **A2** | `SWIGLU_KEYS` 統一 build_weighted_avg / flat-weights | 中（碰數值，需 bit-exact 驗） | ⬜ TODO |
| **A3** | 單一 `_hf_expert_loop` helper（現有 3 份） | 低 | ⬜ TODO |
| **A4** | draft「名單事實」移到類別屬性，cli 去三重列舉 | 低 | ⬜ TODO |
| **A5** | 整理 merge 入口（**固定線性加權平均，不抽 strategy**）+ 預留 cache 包裝點 | 低 | ⬜ TODO |
| **A6** | `clustering/` strategy registry（freq_slice 原樣搬入） | 中（碰 cache 路徑） | ⬜ TODO |
| **A7** | env-var → YAML（merged_backend / early_pin / no_overload / cluster_uniform） | 低 | ⬜ TODO |
| **A8** | 拆 adapters/base.py、specmoe forward 搬出 adapters/ | 中 | ⬜ TODO |
| **B1** | co-occurrence 統計捕捉（scorer + per-layer accumulator） | 中（碰 capture 路徑） | ⬜ TODO |
| **B2** | `CooccurCluster`（agglomerative on co-occurrence） | 低（新 strategy） | ⬜ TODO |
| **B3** | `CachedMerge`（CPU、bounded LRU、key=成員集合） | 中（記憶體 + 數值） | ⬜ TODO |

**依賴**：B1 → B2（分群要吃 co-occurrence）；A6 → B2（clustering 接縫）；A5 → B3（線性 merge 入口外包 cache）；A1–A4 與 B 互不相依，可先做暖身。

---

## 1. 目標 codebase 結構

```
aug_spec/
  adapters/      # 模型家族:ABC + 共用 forward + 各家(瘦身:SWIGLU_KEYS + hooks)
  drafts/        # draft 策略(registry)
  merging/       # 新:線性加權平均 merge(固定,非 registry)+ CachedMerge 包裝層
  clustering/    # 新:ClusterMethod ABC + freq_slice + cooccur(registry)
  forwards/      # 新:averaged / masked / substitute 三個共用 forward wrapper
  runtime/       # loader, specbench, phase, offload_merge, scorers, profile
  config.py      # 新:RunConfig 從 cli.py 拆出
  controller.py
  cli.py         # 瘦成 parse → build(strategies 注入)→ run
```

**merge 永遠是線性加權平均,不可配置** —— 未來所有 merge 變體都只是「權重不同」,
由 clustering / 群內權重決定,不需要 swappable merge 演算法。所以只有
**clustering / forwards** 兩個新 registry 對應「未來會長新成員」的維度;`merging/`
只放固定的線性 merge + 可選 cache 包裝,不是 registry。YAML 用 `cluster.name:` 選分群。

---

## A. 重構(行為不變優先)

### A1 — adapter forward wrapper 上提 base ✅ 證據已驗
`make_averaged_forward` / `make_masked_forward` 在 qwen3 與 mixtral **去空白後 md5 完全相同**
(`4dd6be03…` / `5582c405…`)。兩者唯一的模型差異都委派給 `self._standard_routing` /
`_run_dense_expert` / `_route_multi_expert` / `self.capture`。

- **做法**:把兩個 wrapper 整個移到 `MoEAdapter`,內部呼叫 `self.*` hook。
  mixtral / qwen3 刪掉自己的副本;gptoss 只 override 它真正不同的部分(bias / 無 offload),
  或把差異也收進 hook。
- **檔案**:adapters/base.py(+)、mixtral.py(−)、qwen3.py(−)、gptoss.py(調整)。
- **驗證**:hf 與 offload 各跑一個既有 config,MAT 與重構前完全一致(這步 byte-identical,最安全)。

### A2 — `SWIGLU_KEYS` 統一方法
`build_weighted_avg` / `expert_flat_weights` 在 mixtral(`w1/w3/w2`)與 qwen3
(`gate_proj/up_proj/down_proj`)是同一套迴圈,只差矩陣名字。

- **做法**:adapter 宣告
  ```python
  SWIGLU_KEYS = ("gate_proj", "up_proj", "down_proj")   # (gate, up, down)
  def _expert_matrices(self, expert): -> (W_gate, W_up, W_down)   # 讀一顆 expert
  ```
  通用的加權和 / flat 放 base,逐 key 跑。adapter 只留 key 名 + `_expert_matrices` +
  offload 分支(qwen3 的 `merge_experts_local` C++ 路徑)+ `_run_dense_expert`。
- **驗證**:**bit-exact** —— 對 q5 freq config 比對重構前後逐 cycle 的 merged 權重或最終 MAT 完全相同。

### A3 — 單一 `_hf_expert_loop`
同一段 one-hot / `index_add_` 迴圈出現 3 次:mixtral `_standard_routing`、qwen3 `_standard_routing`、
base `_topk_substitute_forward`。抽成 base 的 `_hf_expert_loop(block, hs, selected, weights)`。
- **驗證**:hf 路徑 MAT 不變。

### A4 — draft 名單事實移到類別屬性
cli.py 現在三處列舉 draft 名單:`_MERGE_DRAFTS`(reserve 計算)、`count_top_k` 自動填的 tuple、registry。

- **做法**:在 `DraftStrategy` 加類別屬性 / classmethod:
  ```python
  holds_merged_residency: bool = False   # ScoreBasedAvgDraft 系列 = True
  needs_count_top_k: bool = False        # count 系列 = True
  ```
  cli 改成問類別,不 hardcode 名字。新 draft 自帶這些旗標,cli 不用改。
- **驗證**:既有所有 config 行為不變(只是把名單來源換成屬性)。

### A5 — 整理 merge 入口(固定線性,不可配置)
SVD merge 已刪除,**merge 永遠是線性加權平均**。未來的 merge 變體都只是「權重不同」
(由 clustering 與群內權重決定),不需要 swappable 的 merge 演算法,因此**不抽 MergeMethod
strategy / 不做 registry**。這一步只是把 merge 入口整理乾淨,並預留 B3 的 cache 包裝點:
```python
# merging/linear.py  -> 收斂現有 adapter.build_weighted_avg / offload engine.build 的入口
def linear_merge(adapter, block, member_ids, weights) -> dict[str, Tensor]: ...
```
- **接線**:`_build_one` 已經是「engine.build → adapter.build_weighted_avg」的單一線性路徑
  (SVD 分支已移除),維持現狀即可;cache(B3)之後包在這個入口外面。
- **驗證**:既有 config MAT 不變(SVD 移除後已驗過 import/實例化;跑一個 q5 確認數值)。

### A6 — `clustering/` strategy registry
```python
# clustering/base.py
class ClusterContext:           # 一個「統計袋子」,method 各取所需
    active: list[int]
    weights: list[float]        # 逐 cycle count(freq_slice 用)
    cooccur: Optional[Tensor]   # [n,n] 共現(B1 填,cooccur 用)
    l2dist: Optional[Tensor]    # 既有 specmoe 的距離,未來可共用
class ClusterMethod:
    def assign(self, ctx: ClusterContext, K: int) -> list[list[int]]: ...
# clustering/freq_slice.py  -> 包現有 _assign_clusters
```
- **接線**:`_cluster_and_build` 改呼叫 `self.cluster_method.assign(ctx, K)`;A0 的群內
  uniform/freq 開關也順勢變成 method 的參數或 merge 層的職責(見 A7)。
- **驗證**:freq_slice 跑既有 K=16 config,分群結果與 MAT 與現在一致。

### A7 — env-var → YAML
現在 import 時讀 `os.environ` 的實驗旋鈕收進 YAML(env 僅當 override):

| 現 env | YAML key |
|---|---|
| `AUG_MERGED_BACKEND` | `model.offload.merged_backend` |
| `AUG_EARLY_PIN` | `draft.early_pin` |
| `AUG_NO_OVERLOAD` | `model.offload.no_overload`（C++ 仍讀 env,由 loader 設) |
| `AUG_CLUSTER_UNIFORM` | `cluster.within_weight: uniform \| freq` |

- **動機**:一個 YAML 完整描述一次 run(可重現);消除「config 一樣、env 不同」的混淆。
- **驗證**:把現有靠 env 跑的 config 改成 YAML 欄位,結果一致。

### A8 — 拆 adapters/base.py、specmoe forward 搬家
- adapters/base.py 拆成:`adapters/base.py`(ABC + 共用 forward)、`kernels/bmm.py`(`_bmm_swiglu` / `_stack_swiglu_weights`)。
- `_topk_substitute_forward` / `_specmoe_engine_bmm` 是 SpecMoE 專屬,搬到 `forwards/substitute.py`(或 specmoe draft 模組)。adapter 只暴露 `gate` / `_dispatch_selected` 等通用 hook。
- **驗證**:specmoe 跑既有 config,MAT 不變。

---

## B. 新功能:co-occurrence 分群 + CPU merge cache

### ⚠️ 先拍板:cache 可重用性決定 merge 語意
merge 是逐 cycle count-weighted。若 cache key 含逐 cycle 權重 → 幾乎不命中。
**所以採設計 (i)**:co-occurrence 既決定「誰跟誰一群」,也決定「群內權重」(用穩定的共現
統計,而非逐 cycle count)。成員與權重在多個 cycle 內穩定 → 同一成員集合的 merged 結果可重用 →
cache 才有意義。A0 的 uniform 實驗就是在驗證「群內不用逐 cycle 加權」是否可接受;若 MAT 不掉,
(i) 成立。

### B1 — co-occurrence 統計捕捉
co-occurrence 需要**每 token 的 top-k 集合**,不只聚合 count。
- **scorers.py**:加 `make_cooccurrence_scorer`:對 softmax 取 top-k → `Σ_token onehot·onehotᵀ` → `[n,n]`。
- **drafts/base.py**:`target_score`(count `[n]`)旁邊加 `self.cooccur: Dict[int, Tensor[n,n]]`,
  用 **window / EMA** 累積(讓它穩定 → 提高 cache 命中)。capture 時一併更新。
- **接線**:把 `cooccur[li]` 灌進 `ClusterContext`。
- **驗證**:不開 cooccur cluster 時行為不變;開了之後印共現矩陣 sanity check。

### B2 — `CooccurCluster`(agglomerative)
- **clustering/cooccur.py**:在 `ctx.cooccur` 上做**凝聚式分群**(每次併最相關的兩個子群),
  併到剩 K 群。回傳 `list[list[int]]`,介面與 freq_slice 完全一致。
- 「裝兩個 expert 的結果」= agglomerative 的內部節點(一個成員 frozenset),天然對應 B3 的 cache。
- **YAML**:`cluster: {name: cooccur, window: 512, ...}`。
- **驗證**:小 run 看分群是否抓到「常一起出現」的 expert;MAT 對比 freq_slice。

### B3 — `CachedMerge`(CPU、bounded LRU)
- **merging/cache.py**:包在固定的線性 merge 入口外(A5):
  ```python
  class CachedMerge:
      def __init__(self, inner_merge, max_items, device="cpu"): ...
      def merge(self, adapter, block, member_ids, weights):
          key = frozenset(member_ids)          # 設計 (i):只用成員集合
          hit = self.store.get(key)
          if hit is not None: return hit.to(target_device)
          out = self.inner_merge(adapter, block, member_ids, weights)
          self.store.put(key, out.cpu())        # CPU 存、用時搬回 GPU
          return out
  ```
- **記憶體**:n=128 → 子群數可能很大,`store` 必須 **bounded LRU**(只留反覆命中的)。
  設計 (i) + 穩定 co-occurrence 才會讓常用子群重複命中、cache 小而有效。
- **量測**:印 cache hit-rate / size / 省下的 merge 次數;對比有無 cache 的 TPS。
- **YAML**:`merge: {cache: {enabled: true, max_items: 4096}}`(merge 不可配置,只有 cache 開關)。
- **驗證**:hit 時的 merged 結果 == 重算結果(數值一致);hit-rate 與 TPS 增益。

---

## 2. 建議執行順序

1. **A1 → A3 → A4**(低風險暖身,殺重複,行為不變)。
2. **A6**(clustering strategy 接縫;freq_slice 原樣搬入,行為不變驗)+ **A5**(整理線性 merge 入口)。
3. **A7**(env → YAML;順手把 A0 的 uniform 開關正式化成 `cluster.within_weight`)。
4. **B1 → B2 → B3**(疊在 clustering 接縫 + 線性 merge 入口上,純新增成員)。
5. **A2 / A8**(碰數值較多 / 純整理,可最後做)。

每步一個 commit;碰數值的步驟(A2/B3)附 q5 對照數據再合。
