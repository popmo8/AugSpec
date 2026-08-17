# serial_dispatch 實作計劃（2026-07-16）— 關掉 fetch/forward overlap 的最 naive baseline

> **目標**：一個 config 開關 `model.offload.serial_dispatch: true`，把 offload MoE forward 的
> fetch/exec overlap 關掉——**一次只 dispatch 一個 expert，抓完＋算完才輪下一個**。
> 與非投機三 baseline（moe_caching / moe_precache / moe_ondemand）自由組合；
> **`moe_ondemand + serial_dispatch` = 教科書級 naive offload（無 cache＋無 overlap）**，
> 作為論文的最 naive 下界。支援 B=1 與 B>1。不動 C++。

## 1. 語意

| | 現況（overlap on）| serial_dispatch: true |
|---|---|---|
| 一層的 dispatch | 8（~128@prefill）個 expert 一次 enqueue，引擎 fetch‖exec 交錯 | **逐 expert**：dispatch(e)→wait(e)→dispatch(e+1)→… |
| fetch 位置 | 藏在 exec 底下（重疊）| **完全暴露在 critical path** |
| fetch↔fetch 平行 | 多執行緒同時抓 | 無（naive offload 本來就沒有）|

## 2. 設計（為什麼這樣做）

1. **patch 點 = offloaded block forward**（`moe_infinity/models/qwen.py` `Qwen3MoEBlock.forward`）。
   三個非投機 baseline 都走這同一個 forward → **patch 一處、三者全生效，B=1/B>1 都通**
   （batch_spec 也是呼叫 model.forward）。
2. **實作**：新模組 `src/aug_spec/runtime/serial_dispatch.py`，`install_serial_dispatch(model)`：
   對每個有 `expert_executor` 的 MoE block，把 `block.forward` 換成 serial 版：
   - 沿用 block 自己的 route 準備（`_Qwen3MoEBlock__prepare_expert_route`，name-mangled 私有法）
     → gate 照常被呼叫（**與 moe_precache 的 gate hook 相容**）；
   - 對每個 routed expert e：建單-expert mask/weights（index copy，兩個 tensor op）→
     `dispatch_local` → `wait_dispatch_local` → 線性累加輸出（引擎已套 routing 權重，加總＝原結果）；
   - 回傳 (final, router_logits) 同原介面。
3. **config**：RunConfig 新欄位 `serial_dispatch`（`model.offload.serial_dispatch`，預設 false），
   env `AUG_SERIAL_DISPATCH` 可 override（A4 慣例）。cli 在 load_offload 後呼叫 install。
4. **spec 模式擋掉**：controller.install 會覆蓋 block forward、把 patch 蓋掉（adapter 的 verify
   dispatch 是另一個 call site）→ `spec_mode + serial_dispatch` 直接 **raise**（fail-fast，
   本開關只給非投機 naive baseline；未來要給 Ours 用再另案 patch `_route_offload`）。
5. **與現有機制相容性**：moe_ondemand 的 model-level flush hook（forward 之後）✓；
   moe_precache 的 gate hook（prep 內呼叫 self.gate）✓；moe_caching 無 hook ✓。

## 3. 變更清單（全 Python）

| 檔案 | 變更 |
|---|---|
| `src/aug_spec/runtime/serial_dispatch.py`（新，~70 行）| `install_serial_dispatch(model)`：逐 block 換 forward（`types.MethodType`），回傳 patch 層數 |
| `src/aug_spec/cli.py` | RunConfig 欄位 + parse；load_offload 後 `if serial_dispatch: install + print`；`spec_mode` 併用 → raise |
| `configs/` | `ondemand_ser_smoke.yaml`（B=1, qpc1/mnt32）、`ondemand_ser_smoke_b4.yaml`、正式：`ondemand_ser_b1/b4/b64.yaml`（t1 協議；label 後綴 `_ser`）|
| `PROJECT_GUIDE.md` / 本檔 | 完成後同步（規則 4）|

## 4. 驗證階梯

| # | 測試 | 判準 |
|---|---|---|
| V1 | smoke B=1 + B=4（ondemand+serial, qpc1/mnt32, 2 sbatch）| rc=0；TPS 明顯低於無 serial 的 smoke；輸出正常 |
| V2 | **機制證明**（AUG_PROFILE, B=1, mnt128, ondemand+serial vs 已有的 ondemand profile）| fetch 時間佔 wall 比例暴增（serialization 生效的直接證據）；fetches==forwards 仍成立（zero-cache 不受影響）；確認減速來自 fetch 暴露、非我們加的 Python mask 開銷 |
| V3 | full t1 B=1/4/64（3 sbatch, 8h）→ 「naive offload」列入三方表 | 需用戶同意 |

## 5. 預測（先寫下）

B=1 ondemand+serial：每 token 384 次序列化 fetch ×（~0.2-0.4ms 傳輸 + dispatch 往返）≈
+150~300ms/token → **~2-3.5 tok/s**（vs overlap 版 6.17）。B=64 每 step 6144 次序列化 fetch
→ 掉更兇。fetch_us/wall 佔比從 ~30% → 大幅上升（V2 直接量）。

## 6. 風險與緩解

| 風險 | 緩解 |
|---|---|
| R1 高頻單-expert dispatch 踩引擎邊角（expected_queue=1 切換、notify 開銷、race）| smoke 立即暴露；batch loop 已有修法A sync |
| R2 依賴 moe_infinity 私有方法名（name-mangling）| 集中在一個模組；斷了只影響此開關 |
| R3 bf16 累加順序改變 → 輸出微漂 | 非投機 baseline 無 AccR，TPS ablation 可接受；標註 |
| R4 減速被自加的 Python 開銷污染歸因 | V2 用 profile 的 fetch_us 佔比講故事，不只看 TPS |

## 7. Todo（依序）

- [x] T1 `serial_dispatch.py` + cli 欄位/接線/spec-guard + AST/review — 完成
- [x] T2 V1 smoke（263618/263619）— **綠**：patch 上 48 block、rc=0、零 cache 保留
      （hit 0、fetches==forwards）；**TPS 砍半**（B=1 6.60→3.42、B=4 7.32→3.98, mnt32）→
      serialization 生效、fetch 被暴露。
- [ ] T3（可選）V2 mnt128 profile 對照 fetch_us/wall——但 full t1（mnt512, decode 主導）
      Python 迭代少、歸因更乾淨，可直接看 V3。
- [ ] T4（需同意）V3 full t1 三點（B=1/4/64, `ondemand_ser_b1/b4/b64`）→ naive 下界入表
      + 文件同步（PROJECT_GUIDE、baseline_tables_plan）
