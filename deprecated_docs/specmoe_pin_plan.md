# SpecMoE expert pinning — 讓 baseline 忠實於論文

> 我們的 specmoe draft 實作了 kept-N 替換**演算法**，但**沒有把 kept-N pin 在 GPU**
> ——目前是 archer 按需 fetch。這讓它**不是 offload_plan.md §1.4 對比的那個 SpecMoE**
> （論文版靠 pinned experts 讓 draft 零 PCIe）。本 md 記錄現況、gap、與補上 pinning 的計劃。

---

## 0. 狀態

| 項 | 狀態 |
|---|---|
| kept-N + substitute 演算法 | ✅ 已實作（`drafts/specmoe.py` + `_topk_substitute_forward`） |
| kept-N pinned in GPU（draft 0 PCIe） | ❌ **未實作**（draft 走 archer 按需 fetch） |
| VRAM reserve for pinned | ❌ 未實作（specmoe 拿滿 archer pool） |

---

## 1. 現在的 specmoe 做什麼

`adapters/base.py::_topk_substitute_forward`（draft phase）：
1. `block.gate` 算 natural top-`route_top_k` winner。
2. 用 `draft_cache[layer]` 的 substitute table 把每個 winner **重映射**到 kept-N
   （在 mask 內 → 自己；不在 → L2 最近的 kept；mask 是每 cycle 由 count top-N 重建）。
3. **`adapter._dispatch_selected` → `expert_executor.dispatch_local` → archer 按需 fetch**
   那些 (重映射後的) kept expert，算完回傳。

→ kept-N 的**選擇邏輯**對（acceptance 正確），但它們**不常駐 GPU**：每次 draft 要用就
走 archer 的 on-demand fetch（cache 剛好留著就命中、被 evict 就重抓）。

## 2. Gap：沒有 pin

SpecMoE 在 offload 下的核心賣點 = **把 N 顆代表 expert pin 在 GPU → draft 直接讀
resident、零 PCIe**（代價：那 N×L 顆佔 VRAM、兩階段都當 cache、**不可回收**）。

我們現在：

| | kept-N 演算法 | pin（draft 0 PCIe） | VRAM reserve |
|---|---|---|---|
| **論文 SpecMoE** | ✅ | ✅ | N×L×expert |
| **我們現在** | ✅ | ❌ 按需 fetch | 0（拿滿 cache） |

後果：現在的 specmoe **draft 要付 PCIe** 抓 kept-N（比論文版弱），但**不佔 reserve**
（拿滿 12.2GB cache，對它有利）。**兩個偏差方向相反，但都讓它不是論文的 SpecMoE。**

## 3. 為何影響對比（thesis 公平性）

論文核心主張（offload_plan.md §1.4）：**merged-expert draft 贏 SpecMoE + 更省 VRAM**，
靠的是「我們的 merged 可回收、SpecMoE 的 pinned 不可回收」這個不對稱。這**假設 SpecMoE
有 pin**。拿沒 pin 的 specmoe 對比 = 比錯對手：
- 沒 pin 的 specmoe draft 付 PCIe → TPS 被低估（對 SpecMoE 不利）。
- 沒 reserve → 拿滿 cache → verify 較快（對 SpecMoE 有利）。
- 兩個效應糾纏，數字不能直接對應論文。

## 4. 計劃：補上 pinning（讓 specmoe 忠實）

每 cycle（mask 更新後）：
1. **算 kept-N**（已有：count top-N，per layer）。
2. **fetch + pin**：把 kept-N fetch 進 GPU、**標記不被 evict**（常駐到下次 mask 變）。
   draft 的 `_dispatch_selected` 讀到的就是 resident → **0 PCIe**。
3. **reserve**：從預算預扣 `N×L×expert`（與 topm 的 merged 對稱；N=16 → 7.25GB）。

實作元件（複用現有 dispatcher 基礎）：
- **C++ pin set**：一個 per-gpu「pinned key 集合」，`FindExpertEvict` / overload-evict /
  `EvictLayer` **跳過 pinned**（EvictLayer 的反面）。method：`SetPinned(layer, expert_ids)`
  / `ClearPinned`。
- **Python**：specmoe `refresh` 算完 kept-N mask 後，呼叫 `set_pinned`；mask 變時換 pin。
  （pin 的 fetch：第一次 draft 觸發 archer fetch 並 pin，或顯式預取。）
- **reserve**：`cli._MERGE_DRAFTS` 的對稱——specmoe 也算 `N×L×expert` 預扣（加一個
  `compute_pinned_bytes` 或共用 `compute_merged_bytes`，K→N）。flag 控制 on/off ablation。

## 5. 對稱對比 setup（補完後）

b=0.2，兩邊同預算：

| | reserve 7.25GB | verify cache | draft |
|---|---|---|---|
| **specmoe（pinned）** | pinned kept-N | 4.97GB | 讀 pinned，0 PCIe |
| **topm（P2+P3）** | merged | 4.97GB | 讀 merged dense，0 PCIe |

差別收斂到論文真正要比的：**acceptance（merged vs substitute）** + **§1.4 recyclability**
（merged 可在 verify 回收當 cache、pinned 不行——這要等動態回收版才體現，固定 reserve 版
兩邊都靜態切，先不體現）。

## 6. 決策點
- **要不要做 pinning**：要忠實對比論文 SpecMoE 就要。先用現在的 on-demand specmoe 跑也行，
  但數字標「非論文 SpecMoE（on-demand 版）」。
- **N 的選擇**：N=16 對稱 topm K=16；論文 SpecMoE 原生 N 可能不同（offload_plan §1.2 提過
  同預算下 N 受限）。對比時 N 要講清楚。
- 排在 verify_merge_plan.md 的 P2+P3 之後、核心對比之前。
