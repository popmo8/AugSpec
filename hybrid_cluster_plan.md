# Hybrid clustering 計劃：α-blended Expert Relation Map（2026-07-03）

> 目標：新增 `cluster.name: hybrid` —— **prefill 收 activation-similarity（L2）、
> decode 收 co-occurrence**，clustering 時用一個 `alpha` 凸組合兩張 map 決定
> 哪兩顆 expert 配對 merge。對應論文的 *Expert Relation Map*（兩張子圖）。
> 動機：actsim_l2 表現穩定但 decode 期 per-cycle capture 吃 latency；
> co-occur 便宜且帶 temporal locality。hybrid 讓 act-sim 的成本一次付清
> （prefill），decode 只付 router 統計的近零成本。
>
> 本文件是給後續 session 的完整實作規格。實作前先讀
> `ARCHITECTURE_REVIEW_AND_PAPER_PLAN.md` 的守則一節。

---

## 1. 方法定義（論文可直接引用的形式）

對每層 ℓ、每題，維護兩張 active-expert pairwise map：

1. **Activation-similarity map `Â_ℓ`（prefill-only）**
   prefill target forward 時由 C++ engine capture 各 expert 的 raw output，
   對 co-routed token 累積 pairwise 距離；`Â_ℓ[i,j] = −mean L2(out_i, out_j)`
   （現有 `_pair_sim_table` 的 l2 分支，未共同觸發的 pair 為 −inf sentinel）。
   **prefill 結束即凍結**（capture 關閉，decode 零開銷）。

2. **Co-occurrence map `Ĉ_ℓ`（decode-only，跨 cycle 累積）**
   decode 的每個 verify forward 用現有 cooccurrence scorer 累積
   `C[i,j]` = 兩者同在該 token top-k 的 token 數（對角線 = 單獨次數），
   再做 **cosine 正規化**：`Ĉ[i,j] = C[i,j] / sqrt(C[i,i]·C[j,j]) ∈ [0,1]`
   （去除熱門 expert 頻率偏差；對應 0629 結論「相似度用 cosine 非 lift」）。

3. **Relation map 與分群**
   `R_ℓ = α·𝒩(Â_ℓ) + (1−α)·𝒩(Ĉ_ℓ)`，`α ∈ [0,1]`，
   𝒩 = 每層、每次 assign 時對 active pair 子矩陣做的同尺度正規化（§2）。
   分群 = 現有 `greedy_pair(R, active, K)`（max-size-2 貪婪配對），
   下游 merge / draft forward / within_weight 完全不變。

**端點語義**（sanity check + 消融欄）：
- `α=1` ⇒ 純 prefill-only act-sim（本身即新變體：actsim_l2 的省 latency 版）。
- `α=0` + `cooccur_norm: raw` ⇒ ranking 與現行 `cooccur_pair` **完全一致**
  （greedy_pair 只看排序，正規化是單調變換）。`cooccur_norm: cosine`（預設）
  則是修正頻率偏差後的 cooccur。

**每題時間線**：
```
question start ─ reset(): 兩張 map 清空; capture ON
prefill forward ─ 逐層: capture expert outputs → Â 累積;
                  (cooccur_scope=decode ⇒ 這次 forward 不進 Ĉ)
                  on_verify_layer 照常 merge（此時 Ĉ 空 → 排序=純 Â，well-defined）
1st draft start ─ capture OFF（此後 decode 零 capture 成本）
decode cycles   ─ 每 verify: Ĉ 累積; 每次 rebuild 用 R = α𝒩(Â)+(1−α)𝒩(Ĉ)
```

## 2. 正規化 𝒩（`norm` 旋鈕，兩種都實作）

兩張 map 量綱不同（負 L2 距離 vs [0,1] cosine），α 要有意義必須先同尺度。
只對「有資料」的 active pair 做；無資料（Â 的 −inf/−1 sentinel、Ĉ 的 0）
一律映成 0 → 在 greedy 裡排最後、只在湊 K 時才用（與現行 sentinel 行為一致）。

- `minmax`（**預設**，論文公式好寫）：`(s − min)/(max − min)`，
  max==min 時全部設 0.5（防除零）。
- `rank`（消融）：值換成資料 pair 內的正規化名次 `rank/(m−1) ∈ [0,1]`
  （m = 有資料 pair 數；m=1 時設 1.0）。尺度無關、對 outlier 穩健；
  短 prompt（translation）Â 稀疏時較穩。

## 3. 實作規格（檔案逐一列出；CLI 零改動）

`cli.py` 的 `cluster_args` 已把 YAML `cluster.*`（name/within_weight 以外）
原樣傳進 ClusterMethod 建構子 —— 所以新旋鈕不用碰 CLI。

### 3.1 `src/aug_spec/clustering/hybrid.py`（新檔，~100 行）
```python
class HybridRelationCluster(ClusterMethod):
    needs_cooccur = True
    needs_activation_sim = True
    act_sim_prefill_only = True          # engine 讀這個決定 prefill 後關 capture
    def __init__(self, alpha=0.5, metric="l2", norm="minmax",
                 cooccur_norm="cosine", cooccur_scope="decode"):
        # 驗證: 0<=alpha<=1; metric in (cosine,l2); norm in (minmax,rank);
        #       cooccur_norm in (cosine,raw); cooccur_scope in (decode,all)
        ...
    def assign(self, ctx, K):
        R = self._blend(ctx)             # [n,n]，無資料 pair = 0
        return greedy_pair(R, list(ctx.active), K)
```
`_blend(ctx)`：
1. `A = ctx.pair_sim`（draft 已按 self.metric 建好：l2 ⇒ 負距離、−inf sentinel）。
   資料遮罩：`torch.isfinite(A) & (A > -1 + 1e-9 if cosine else True)`
   —— 直接照 metric 分支寫清楚，別想統一式。
2. `C = ctx.cooccur`；cosine 正規化用對角線（`d=diag(C)`，
   `Ĉ = C / sqrt(outer(d,d)).clamp_min(1e-12)`，d=0 處結果視為無資料）。
3. 兩張各自過 𝒩（只在 active pair 上），凸組合成 R。任一張整層無資料 ⇒
   該項全 0（等同 α 落到另一邊，行為連續）。

### 3.2 `clustering/base.py`
`ClusterMethod` 加類別預設 `act_sim_prefill_only: bool = False`（一行）。
`ClusterContext` 不用改（cooccur / pair_sim 欄位已存在）。

### 3.3 `clustering/__init__.py`
registry 註冊 `"hybrid": HybridRelationCluster`，`__all__` 補上。

### 3.4 `drafts/base.py` — decode-only cooccur（~8 行）
`_accumulate_cooccur` 開頭：若 `getattr(cluster_method,"cooccur_scope","all")
== "decode"`，跳過每層 reset 後的**第一次** capture（= prefill forward）：
```python
if scope == "decode" and layer_idx not in self._prefill_seen:
    self._prefill_seen.add(layer_idx); return
```
`reset()` 清 `self._prefill_seen = set()`（`__init__` 也初始化）。
注意：判斷要在 cooccur 累積前、但**不可影響** `target_score` 的 capture。

### 3.5 `runtime/offload_merge.py` — capture 的相位開關（~15 行）
- 新增 `on_question_start()`：若 method `needs_activation_sim` ⇒
  `set_capture_expert_out(True)`（每題重新武裝；`attach()` 原有的開啟保留）。
- `on_draft_start()` 開頭：若 method `act_sim_prefill_only` 且 capture 尚未關
  （self._capture_on flag 自己記）⇒ `set_capture_expert_out(False)`，並呼叫一次
  `get_captured_expert_outputs()` 丟棄殘留 buffer（swap 清空，防跨題污染）。
  每題第一次 on_draft_start 即 prefill→draft 邊界（phase.py:66–74 已觸發）。

### 3.6 `controller.py`
`reset()` 尾端：`if self.merge_engine is not None:
self.merge_engine.on_question_start()`（2 行）。

### 3.7 configs + 文件
- `configs/q15_hybrid_a{00,25,50,75,100}_r{1,2,3}.yaml`：拷貝 q15_actsim_l2_r1，
  cluster 段換成 §頂部 YAML（alpha = 0/0.25/0.5/0.75/1.0），label/dir 對應。
- `configs/README.md` cluster 段補 `hybrid` 全旋鈕表。
- PROJECT_GUIDE.md 依 CLAUDE.md 規則同步（新 cluster method + 本計劃連結）。

### 3.8 單元測試（`tests/unit/test_hybrid_cluster.py`，純 CPU）
- 造小 pair_sim / cooccur 固定矩陣，驗：
  (a) α=0 + raw 的 assign 結果 == `CooccurPairCluster.assign`；
  (b) α=1 的結果 == `ActivationSimCluster.assign`（同 pair_sim）；
  (c) 設計一組「Â 偏好 (0,1)、Ĉ 偏好 (0,2)」的表，掃 α 驗證配對在某個
      臨界值翻轉（單調性）；
  (d) sentinel/全無資料/單 pair 等邊界不 crash、輸出仍是合法 partition。
- minmax 與 rank 兩種 norm 都要覆蓋。

## 4. 實驗計劃（沿 q15 協定：qpc=15, T=5, mnt=512, skip mt_bench, uniform, r1–r3）

| 欄 | config | 目的 |
|---|---|---|
| hybrid α=0.25/0.5/0.75 | q15_hybrid_a{25,50,75}_r{1..3} | 主結果：找 α 的 knee |
| α=1（prefill-only actsim） | q15_hybrid_a100_r{1..3} | 對照 q15_actsim_l2：AccR 掉多少、TPS 賺多少（**latency claim 的直接證據**） |
| α=0（cosine cooccur） | q15_hybrid_a00_r{1..3} | cooccur 修正頻率偏差後的對照 |
| 既有 q15_actsim_l2 / q15_cooccur / q15_freqslice / q15_specmoe | 已跑完 | 直接橫比（同協定） |

- 每組跑完用 `analyze_q15.py` 模式聚合 mean±std（非確定性 SD~0.018，單跑不下結論）。
- 一組加 `AUG_PROFILE=1` 對照 q15_actsim_l2：確認 decode 期 capture 成本消失
  （profiling 表 capture/dispatch 相關 ms/cyc）。
- **論文對應**：Expert Relation Map = α·(act-sim map) + (1−α)·(co-occur map)；
  Table 3 加 hybrid 列；ablation 加 α sweep。
  ⚠️ 草稿寫的是 "prefill-stage **attention** patterns"，實作是 prefill-stage
  **activation**（expert output）similarity —— 論文措辭要改成 activation patterns
  / expert output similarity，以免 reviewer 對不上。

## 5. 風險與備註

- **cooccur 是 must-link 訊號**：0629 partition A/B 結論 must-link cooccur 傷
  acceptance，但 q15（greedy max-size-2 + uniform within）下 cooccur 已勝 random
  ——設定不同結論不同。仍預期 α 偏小時 AccR 下滑；hybrid 的賭注是
  「Â 扛 acceptance、Ĉ 補 temporal locality」，由 α sweep 實證 knee 在哪。
- **Â 凍結在 prefill** 假設 prefill 的相似度結構能代表 decode。
  `AUG_DUMP_CYCLE_SIM` + `analyze_cycle_sim.py` 已有 drift 分析工具，可先驗證。
- 短 prompt 的 Â 稀疏（很多 pair 無資料 → 0）⇒ 有效權重自動偏向 Ĉ；
  這是特性不是 bug，但解讀 per-task 結果時要記得（translation 尤其）。
- v1 的 Ĉ 是整題累積和；若要更強的 recency，可做 per-cycle EMA
  （`C ← λ·C + C_cycle`）——列為 extension，v1 不做。
- 實作順序建議：3.1–3.3（純新增，unit test 可立即驗）→ 3.8 → 3.4–3.6
  （相位接線，跑一個 smoke offload config 驗 capture 確實在 prefill 後關閉：
  AUG_PROFILE 或加一行 debug print）→ 3.7 → 送 q15 批次。

## 6. 狀態追蹤

| 項目 | 狀態 | 備註 |
|---|---|---|
| 3.1–3.3 hybrid method + registry | ⬜ | |
| 3.8 unit tests | ⬜ | |
| 3.4 decode-only cooccur | ⬜ | |
| 3.5–3.6 prefill-only capture 接線 | ⬜ | |
| 3.7 configs + 文件 | ⬜ | |
| smoke 驗證（capture 相位正確） | ⬜ | |
| q15 hybrid α sweep r1–r3 | ⬜ | |
| 聚合 + 論文表更新 | ⬜ | |
