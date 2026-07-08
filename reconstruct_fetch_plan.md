# Reconstruction Fetch 計劃：用 cached merged expert 的代數重建取代 verify fetch（2026-07-06 起草）

> 核心想法（用戶提出）：uniform pair merge 下 `C = 0.5A + 0.5B`。verify 若同時
> 需要 A、B 而 C 已駐留 GPU（L1 merged slots），**只 fetch A，用 `B = 2C − A`
> 重建 B**，C 留著給下一個 draft 繼續用。cluster 演算法配合：**持續留在 draft
> 的 pair 優先配對**（stickiness），因為它們的 C 常駐 → 重建機會最多。
>
> 前置依賴：`memory_hierarchy_plan.md` §3.1 的 GPU L1（內容定址 slots +
> delta rebuild）——C 必須「知道自己是誰」（member set）且跨 cycle 存活，
> 重建才有素材。先讀那份的 §3.0 成本階梯。

---

## 1. 為什麼這件事重要（比 draft-side 省更大）

之前所有分析的結論都是「verify 流是 common-mode、無法差異化」（兩法的 verify
都要抓一樣的 expert）。**Reconstruction fetch 打破這一點**：

- verify fetch 是全系統最大成本（實測 70–75 TB/run、1428–1501 s）；
- cached pair {A,B} 的成員在同輪 verify 被共同 route 時（**這正是 co-occur
  分群最大化的事件**——C_ℓ[i,j] 數的就是它），只需其中一顆在 GPU：
  另一顆用 elementwise kernel `2C − A` 重建（~28MB GPU 讀寫 ≈ 50µs），
  取代 9.44MB host fetch（~190µs）或 disk fetch（3–10ms）；
- 收益隨 hierarchy 壓力放大：**重建 = verify 的 disk-miss 吸收器**，
  hierarchy regime 下單次事件省 60–200×；
- specmoe 結構上做不到：它沒有「涵蓋多顆 expert 的駐留合成物」可供重建
  ——kept expert 只代表自己。**這把 merge > prune 的優勢從 draft 側延伸到
  verify 側**，是論文的新賣點。

節省帳（每個「pair 共同 route + 一員 miss」事件）：
| 情境 | 無重建 | 有重建 | 省 |
|---|---|---|---|
| A 駐留、B host-miss | fetch B 9.44MB | kernel 50µs | ~190µs → 有效 ~4× |
| A 駐留、B disk-miss | 3–10ms | kernel 50µs | **60–200×** |
| A、B 都 miss | 2 次 fetch | 1 次 fetch + kernel | bytes 減半 |

覆蓋率上限：K=16 pairs/層涵蓋 32 顆 expert；verify 每 cycle 每層 active
~25–35 顆 → cached pair 若對準 verify 的共同活躍集（co-occur 的工作），
理論上可觸及 verify fetch 的可觀比例。實際覆蓋率 = 本計劃第一個要量的數。

## 2. 數值正確性（本計劃的核心 trade-off，不可迴避）

**問題**：C 以 bf16 儲存時已捨入（相對誤差 2⁻⁹ ≈ 0.2%），重建
`B̂ = 2C − A = B + (A+B)·δ`，|δ| ≤ 2⁻⁹。誤差性質：
- 絕對誤差有界：~0.0028·σ（σ = 權重 std），即 **~0.3% 的權重擾動**；
- 但 |B| 元素很小處有 **cancellation**：相對誤差放大到 |A+B|/|B|·2⁻⁹;
- **B̂ 用於 verify ⇒ 擾動 target 輸出 ⇒ 觸碰論文的 target-exact 保證。**
  這不是 draft 側「錯了頂多 acceptance 降」的性質，是改變輸出本身。

**三種模式**（實作成 `reconstruct.mode`，全部做進 ablation）：

| mode | C 儲存 | 重建精度 | 額外 GPU 記憶體 | 論文語句 |
|---|---|---|---|---|
| `off` | bf16 | — | 0 | target-exact（現況） |
| `lossy` | bf16 | ~0.3% 權重雜訊 | **0** | 「preserves the target distribution up to bf16 rounding noise」——需實驗支持（§5-V2） |
| `exact` | **fp32** | 位元級精確* | 每 fp32 slot +9.44MB | target-exact 不變 |

*fp32 論證：bf16 的 A、B 在 fp32 中精確可表；`0.5A+0.5B` 只有一次 fp32 加法
捨入（2⁻²⁴）；`2C − A` 後捨回 bf16 時，除非 |B| < |A+B|·2⁻¹⁵（實務上不發生
且屆時絕對誤差仍可忽略），round-to-nearest 恢復**位元級的 B**。

`exact` 的記憶體帳：全部 K=16 slots 轉 fp32 → reserve 7.25→14.5GB，ρ 從
12.5%→25%，**不可接受**。可行版本是 **fp32 子集**：每層只給 J 個「最穩定
pair」fp32 資格（J=4 → +1.8GB，verify 窗口 4.95→3.1GB）——J 是 sweep 軸。
**建議路線**：v1 先做 `lossy`（零記憶體、驗證輸出偏移是否可辯護）＋
`exact-J` 原型；哪個進論文由 §5-V2 的實測決定。

**誤差複利 guard**：被重建出的 B̂（lossy 模式）**不得再作為後續 merge 或
再重建的來源**——slot 標記 `reconstructed=true`，只供該次 verify 計算，
用完依正常路徑處置（或直接丟棄讓下次真 fetch）。

## 3. 系統設計

### 3.1 C++ dispatcher：重建分支（主要工程量，~150–250 行）

位置：`expert_dispatcher.cpp::GPUFetchFunc` 的 cache-miss 路徑（現有
evict/fetch 決策點之前）：

```
on miss(layer ℓ, expert e):
  partner, slot = pair_table[ℓ].lookup(e)     # Python 每 cycle 註冊
  if partner != -1 and merged_resident(slot) and resident(ℓ, partner):
      buf = acquire_slot(ℓ, e)                 # e 本來要佔的 cache 槽
      reconstruct_kernel(buf, merged[slot], resident[partner])  # 2C − A
      mark(buf, reconstructed=true)
      count prof_.reconstruct_{n,us,bytes_saved}
      continue as cache-hit                    # 不發 H2D
  else: 原路徑（fetch/evict）
```

- **pair_table**：新 pybind `SetPairTable(layer, flat_pairs, slot_ids)`，
  Python 端在每次 rebuild 後（`_cluster_and_build` 完成時）註冊本 cycle 的
  pair→slot 映射。size-2 群才登記；singleton 不登記。
- **依賴順序**：兩員皆 miss 時，先走正常路徑 fetch 其中一員（選 pair 中
  routing count 較高者），另一員的重建等 partner 的 fetch 完成
  （沿用現有 per-node mutex/cv 等待機制，不新增同步原語）。
- **kernel**：`out = 2*c - a` elementwise，對 gate/up/down 三塊各一次；
  用 torch C++ API 的 `torch::sub(2*c, a, out=buf)` 級別即可，不必手寫 CUDA。
- **與 merged 儲存的介面**：`merged_resident(slot)` 讀 engine_bmm 的駐留
  merged 區（`merge_experts_local` 產物）——**實作前審計**：slot 佈局、
  生命週期（P1 flush 模式會把 merged 清掉 → 重建與 `flush_on_draft_end`
  **互斥**，config 檢查直接 raise）、fp32 子集模式下的雙精度儲存位置。

### 3.2 Python：sticky clustering（配合面，~30 行）

用戶需求的第二半：**讓 cluster 演算法偏好「持續留在 draft 的 pair」**，
因為存活的 pair = C 常駐 = 重建機會。實作為對任何 pair 類 cluster method
的通用加分項（放 `greedy_pair` 的呼叫端或 hybrid 內）：

```
R'[i,j] = R[i,j] + β · 1[{i,j} ∈ 上一 cycle 的 pairs]
```
- YAML：`cluster.stability_bonus: β`（預設 0 = 關；normalized rank 空間中
  β 的量綱就是「名次讓步幅度」，sweep {0.05, 0.1, 0.2}）。
- 上一 cycle 的 pairs 從 L1 的內容定址 slots 直接可得（`indices`）。
- **誠實的張力**：β 太大 → partition 黏死、不追 routing → AccR 掉;
  β 是「重建收益 × 黏性」vs「acceptance × 適應性」的旋鈕。co-occur/hybrid
  低 λ 本身就偏好時間穩定的 pair，β 是把這個偏好顯式化、可控化。
- 與 hybrid 的關係：最終 pair 分數 = λ·𝒩(Â) + (1−λ)·𝒩(Ĉ) + β·sticky。
  論文可把 β 併入 Expert Relation Map 的第三項（穩定性先驗）。

### 3.3 YAML 全表

```yaml
offload:
  reconstruct:
    mode: off | lossy | exact    # 預設 off;exact 需搭配 fp32_slots_per_layer
    fp32_slots_per_layer: 0      # exact-J 模式的 J;lossy 忽略
cluster:
  stability_bonus: 0.0           # β;任何 pair 類 method 通用
```
- config 互斥檢查（cli.py）：`reconstruct.mode != off` 需要
  `merge_offload + merge_during_verify + within_weight=uniform +
  cluster method 產 size≤2 群 + not flush_on_draft_end`，違反直接 raise。
- specmoe 不適用（無 merged）——公平對比時 specmoe 行照常。

## 4. 節省模型（把 §1 的帳落成公式）

每 cycle 每層，設 cached pairs 集合 P（|P|≤K），事件計數：
- `n_both(p)` = pair p 兩員都被本輪 verify route 且至少一員 miss 的次數;
- 省下 bytes ≈ Σ_p 9.44MB ×（miss 且可重建的成員數）;
- 省下時間 ≈ Σ [t_fetch(miss 來源) − t_kernel]，disk-miss 事件下每次省 3–10ms。

**覆蓋率 = h(λ,β) ×「pair 共同 route 率」**——第一項是 pair 存活率
（L1 hit rate），第二項正是 cosine-normalized co-occur 的最佳化目標。
兩個量都能從現有 dump 免費估：`AUG_DUMP_PAIRS`（pair 序列）×
`AUG_DUMP_ACTIVE_SET`（每輪 verify 的 active set）離線交叉 → 在**寫任何
C++ 之前**就能算出「可重建事件率」的上界（§5-V0）。

## 5. 驗證與實驗順序

| # | 事項 | 產出 / 驗收 |
|---|---|---|
| V0 | **免費上界估計**：現有 λ sweep 的 dump 交叉分析（pair 存活 × 共同 route × 假設 miss 率） | 可重建事件率 vs λ/β 的表;決定值不值得寫 C++ |
| V1 | 數值單元測試（純 CPU）：bf16 lossy 誤差分布（含 cancellation 尾巴）、fp32 exact 位元級恢復 | `tests/unit/test_reconstruct.py` |
| V2 | **輸出等價性實驗（lossy 的生死關）**：mode=lossy vs off，同 config n≥3——AccR 差、輸出文本 BLEU/exact-match、下游任務分數 | 若偏移超出非確定性帶（AccR ±0.018）→ lossy 出局，只留 exact-J |
| V3 | pair_table + 重建分支 C++ 實作 + smoke（1 題，`reconstruct_n>0`、無 hang、AccR 正常） | profile 新 rows |
| V4 | 主實驗：c ∈ {0.5, ∞} × mode {off, lossy/exact-J} × β {0, 0.1} × {topm, hybrid cos_a25} | verify fetch bytes ↓、TPS ↑、AccR 持平 |
| V5 | 論文寫入：method 第三小節（merge system optimization）納入 reconstruction + sticky 項 | 對應 `paper/method_relation_merge.tex` 的後續小節 |

## 6. 風險清單（按殺傷力排序）

1. **target-exact 主張**（§2）：lossy 模式動到 verify 權重。緩解：V2 實測 +
   論文措辭改「up to bf16 rounding noise」+ exact-J 備援。**這條沒過，
   整個 lossy 路線放棄，不硬凹。**
2. **engine_bmm merged 儲存審計**（§3.1）：slot 生命週期若與假設不符
   （如每 cycle 全量重灌、draft 後即釋放），重建的素材根本不在——先審計
   再動工（與 memory_hierarchy_plan §3.1 的審計合併做）。
3. **C++ 併發**：重建分支引入「等 partner fetch」的新依賴邊——確保只用
   現有 node mutex/cv 模式，不加新鎖;deadlock smoke 必跑（歷史教訓：
   specmoe 的 lost-wakeup hang）。
4. **β 傷 acceptance**：sticky 讓 partition 落後 routing。V4 同時報 AccR。
5. 誤差複利（§2 guard）與 flush 模式互斥（§3.3）——設計上已擋，實作照做。

## 7. 狀態追蹤

| 項目 | 狀態 |
|---|---|
| V0 免費上界估計 | ⬜ |
| V1 數值單元測試 | ⬜ |
| V2 lossy 輸出等價性 | ⬜ |
| §3.2 sticky clustering(β) | ⬜ |
| §3.1 pair_table + C++ 重建分支 | ⬜ |
| V4 主實驗 | ⬜ |
| V5 論文小節 | ❌ 2026-07-06 **從論文撤下**(用戶決定:機制成立但不支撐主論點)。`method_reconstruct.tex` 改寫為 Merge-Aware System Optimization(hierarchy + pipelining),重建機制不再出現於論文。本計劃降級為**未來工作**;V0–V4 若日後要撿回再執行 |
