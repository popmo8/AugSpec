# Batched Speculative Decoding 實作計劃（batch_spec_plan.md，2026-07-13 定稿）

> **目標**：實測 `fig:batch_sweep`——{MoE-Caching, SpecMoE, Ours} × batch B∈{1,4,16,64}
> 的 end-to-end TPS，補上 Table 2 的 batch-scaling 故事（crossover 實測）。
> **硬約束**：(1) 完全不影響既有模組——所有既有路徑零行為變化，新功能走獨立檔案
> ＋顯式 opt-in 旗標；(2) B=1 等價驗收是硬門檻，不過就棄用（fallback = trace 模擬圖）。
> **時程**：Table 1 六列（t1b jobs）落地後才開工；工程 ~4 天 + 跑數；
> **abort 準則見 §9**——D2 結束 V1 不綠就停損。

---

## 0. 隔離原則（「不影響原有模組」的具體做法）

| 原則 | 做法 |
|---|---|
| 新程式全部住新檔 | `src/aug_spec/runtime/batch_spec.py`（唯一新模組）＋ `tests/unit/test_batch_spec.py` |
| 顯式 opt-in | `run.batch_loop: true`（預設 **false**）＋ `run.batch_size: N`（預設 1）。batch_loop=false 時 **cli 走原路徑，一行都不繞進新碼** |
| 不碰的檔案 | `specbench.py` 的 HF assisted 路徑、`phase.py`、`adapters/*`、`drafts/*`、`controller.py`、`merged_cache.py`、`offload_merge.py`、**C++ 全部** |
| cli.py 唯二改動 | RunConfig 加 2 個欄位（解析）＋ run_experiment 裡一個 `if cfg.batch_loop:` 分支改呼叫 `run_specbench_batched`（additive） |
| 不 monkey-patch HF | batch loop **不碰** `AssistedCandidateGenerator`；只用 `model.forward` + `DynamicCache`。相位切換直接寫 `controller.in_draft_phase`、手動呼叫 merge engine 的 `on_draft_start/on_draft_end`（語義同 `phase.py`，見 §4.6） |

## 1. 演算法規格（greedy、target-exact，語義 = HF assisted decoding）

### 1.1 名詞
- committed stream：某序列已確定的 token 序列（prompt + 已接受 + bonus）。
- 邏輯位置 L：token 在 committed（＋本 cycle 提案）流中的 0-based index——**position_ids 一律用它**。
- 物理位置 P：token 在 KV cache 第 2 維的 index。物理只增不減；被拒絕的提案留在 cache 成「洞」。

### 1.2 核心不變量（實作與測試都圍繞它）
1. **target cache 恆落後 committed 一個 token**（最新的 bonus 尚未入 cache）；
   下一個 verify forward 的輸入 = `[bonus, p_1..p_T]`（T+1 tokens）。
2. verify 輸出 `logits[j]`（j=0..T）預測 committed 流的下一位：
   `p_{j+1}` 接受 ⟺ `argmax(logits[j]) == p_{j+1}` 且前綴全接受；
   接受數 k = 最長匹配前綴；**新 bonus b' = argmax(logits[k])**；本 cycle 前進 k+1 tokens
   （= 現行 metrics 的 `accept_len = 1 + num_matches`）。
3. cache 追加後的有效性：`[bonus, p_1..p_k]` 有效（mask=1），`p_{k+1}..p_T` 是洞（mask=0）。
4. **assistant（draft）cache**：draft 期間逐 token 追加；verify 後被拒絕的 draft slot 變洞；
   已接受的 p_j 保留 **draft 路由算出的 KV**（= HF assisted 的行為，勿「修正」它）；
   b' 的 assistant-KV 在下一 cycle 的 draft step-1 餵入時才產生。
5. 洞靠 **2D attention_mask 遮 key**；RoPE 正確性靠**顯式 position_ids = 邏輯位置**
   （HF 把 K 旋轉後才進 cache，所以洞不會污染別人，被 mask 掉即可）。
6. 物理追加順序 = 邏輯順序（每 cycle 只在尾端 append），因此 causal mask 天然正確。

### 1.3 每題批次的流程
```
組 batch（§5 取樣）→ 對每序列個別 prefill（batch=1 forward，避開 kMaxTokens）
  → target cache 合批（右對齊補洞，§2.2）；TTFT token = argmax(prompt 最後一個 logit)
  → C-BOOT：controller.update_masks()（prefill 計票 → merged/mask/table，池化整個 batch 的 prompt 統計）
  → KV-copy：assistant cache = deepcopy(target prompt cache)（逐序列，再合批）
迴圈直到全部序列完成：
  draft 相位：T 步，每步 forward([B,1])、greedy argmax → proposals[B,T]
  verify 相位：forward([B, T+1] = [bonus, p_1..p_T]) → 逐序列接受 k_i、bonus b'_i
  簿記：mask/position/committed 更新（§1.2）；完成序列 → repack 縮 batch（§4.4）
```

## 2. 資料結構（`batch_spec.py`）

### 2.1 `BatchState`
```python
@dataclass
class BatchState:
    seq_ids: List[int]            # 目前 batch 內序列 → 原始題目 index（repack 後會縮）
    committed: List[List[int]]    # 每序列 committed token ids（prompt 之後的部分）
    prompt_len: List[int]
    finished: List[bool]
    tgt_cache: DynamicCache       # [B, H, P_t, D]
    ast_cache: DynamicCache
    tgt_mask: Tensor              # [B, P_t] 1=有效 0=洞/pad
    ast_mask: Tensor
    # 邏輯長度（= position_ids 來源）：
    tgt_pos_next: Tensor          # [B] 下一個餵給 target 的 token 的邏輯位置
    ast_pos_next: Tensor
    # per-seq 遙測：cycles, accept_lens, n_prop, n_acc, tokens, eos
```

### 2.2 cache 工具（本計劃最容易出 bug 的三個函式，各配專屬測試）
- `stack_caches(per_seq_caches) -> (DynamicCache, mask)`：不同長度右補（尾端 pad、mask=0），
  逐層 `torch.cat` 出 [B,H,P,D]。**pad 在尾端**（右對齊 = 左放內容），配合「只在尾端 append」。
  ⚠ pad 的 K 未被 mask=1 引用即可，值隨意（用 zeros）。
- `append_mask(mask, valid[B, t]) -> mask`：每次 forward 後把新 t 個 slot 的有效位 append。
  draft 步：`valid = ~finished`；verify 步：`valid[:, 0]=~finished`、`valid[:, 1+j]= (j < k_i)`。
- `repack(state, keep_idx)`：對 cache 每層、mask、所有 per-seq 陣列做 `index_select(0, keep_idx)`。
  DynamicCache 逐層 `key_cache[l] = key_cache[l][keep_idx]`（同 value）。

### 2.3 洞的成長上界（記憶體 sanity）
每 cycle 每序列 target 洞 ≤ T−k、assistant 洞 ≤ T−k → cache 物理長 ≈ 邏輯長 × (T+1)/E[k+1]
≈ 1.5×（MAT≈4）。B=64、邏輯 ~1100（prompt+512+洞）：KV ≈ 48層×2×4heads×128×2B×64×1700 ≈ **11GB**
（GQA 救了我們）；hf backend 61GB 權重 + 11GB KV + activations < 141GB ✓；
offload：非 expert ~3GB + pool 12GB + KV 11GB + slots ✓。V4 pilot 實測確認。

## 3. 逐相位規格

### 3.1 Prefill（每序列 batch=1）
- `moe._configure_hook(ids)`（offload 才有）→ `model(input_ids=prompt, use_cache=True)`。
- **assert prompt_len ≤ 2048**（kMaxTokens；Spec-Bench/HumanEval 皆遠低於此，超過就 raise）。
- 收 TTFT token（prompt 尾 logit 的 argmax）→ committed[0]。
- capture 已在 forward 中發生（verify 相位、in_draft_phase=False）→ 全 batch prefill 完
  呼叫一次 `controller.update_masks()`（C-BOOT 等價；B=1 時與現行完全同語義）。
- per-seq cache 收集後 `stack_caches` 合批；assistant = 逐序列 deepcopy 後合批
  （先 deepcopy 再 stack，避免共享 storage）。

### 3.2 Draft 相位（T 步）
```python
controller.in_draft_phase = True
engine and engine.on_draft_start()
for t in range(T):
    inp = bonus if t == 0 else prev_argmax        # [B,1]；finished 序列餵 pad_id
    out = model(input_ids=inp, past_key_values=ast_cache,
                attention_mask=cat(ast_mask, ones[B,1]),
                position_ids=ast_pos_next[:, None])
    ast_mask = append_mask(ast_mask, valid=~finished)
    ast_pos_next += (~finished)                   # finished 序列邏輯位置凍結
    prev_argmax = out.logits[:, -1].float().argmax(-1)
    proposals[:, t] = prev_argmax
controller.in_draft_phase = False
engine and engine.on_draft_end()
```
⚠ finished 序列的 pad slot 一律 mask=0 且**不推進邏輯位置**——它們的 Q 輸出被丟棄、
K 被遮罩，對其他序列零影響（自己 batch 維本來就互不干擾）。

### 3.3 Verify 相位
```python
inp = cat([bonus[:, None], proposals], dim=1)     # [B, T+1]
out = model(input_ids=inp, past_key_values=tgt_cache,
            attention_mask=cat(tgt_mask, ones[B, T+1]),
            position_ids=tgt_pos_next[:, None] + arange(T+1))
logits = out.logits.float()                        # [B, T+1, V]
pred = logits.argmax(-1)                           # pred[:, j] 預測第 j+1 提案位
match = (pred[:, :T] == proposals)                 # [B, T]
k = ((~match).cumsum(1) == 0).sum(1)               # 最長匹配前綴（向量化）
bonus_next = pred[gather k]                        # pred[i, k_i]
```
- 簿記：`tgt_mask` append `[1(bonus), 1×k_i, 0×(T−k_i)]`；`tgt_pos_next += k_i + 1`；
  committed += `p_1..p_k + bonus_next`；`ast_mask` 把本 cycle draft 的 T 個 slot 中
  第 k_i+1..T 個改成 0（**注意：assistant 的洞是回頭改上一段 append 的位**，
  保留 `ast_slot_start` 索引）；`ast_pos_next = 對齊 committed`（= tgt_pos_next − 1？
  否——assistant 邏輯長 = committed 長 − 1（b' 未入），實作用 committed 長度現算，
  不要維護兩份會漂移的計數）。
- **capture 語義**：verify forward 內 adapter 照常 capture 全部 B×(T+1) 位（含將被拒的）
  ——與 B=1 的 HF 路徑一致（capture 本來就先於接受判定）。finished 序列已被 repack 移除，
  無 pad 污染。
- EOS：committed 中出現 EOS（含 bonus）→ 該序列 finished，本 cycle 有效 token 記到 EOS 為止。
- mnt：committed（不含 prompt）≥ mnt → finished（超出部分不計 tokens）。

### 3.4 kMaxTokens 防線
單次 forward token 數：prefill ≤2048（assert）、draft B、verify B×(T+1)。
**assert B×(T+1) ≤ 2048**（B=64,T=5 → 384 ✓；B 上限 341）。

## 4. 系統整合

### 4.1 三系統共用同一 runner
- `draft: none` → spec=False：跳過 draft 相位，verify 段退化為「每 step 餵 [B,1] 上個 token」
  的普通 batched greedy decode（= MoE-Caching batched）。同一份簿記碼，少走分支。
- SpecMoE / Ours：controller 照常 install（forward 換裝機制不變、天然支援任意 batch）。

### 4.2 cli 接線（additive）
`RunConfig` + `batch_loop: bool`、`batch_size: int`；`run_experiment` 在建好
controller/draft 後：`if cfg.batch_loop: result = run_specbench_batched(...)`
else 走原 `run_specbench`。兩函式回傳同型 `SpecBenchResult`。

### 4.3 metrics 定義（batch 下的誠實口徑）
- **TPS（圖的 y 軸）** = Σ 全部序列 committed tokens ÷ Σ 各 batch 的 wall（不含載入）。
- AccR / MAT：逐序列良定義，照常聚合（per-subtask AccR 可報）。
- **per-subtask TPS 在 B>1 不定義**（同 batch 混類別共享 wall）——輸出檔寫 N/A。
- 記 per-batch：B_start、各序列題號、wall、tokens——figure script 直接吃。
- 靜態 batching（無 continuous batching）＋ repack——在論文 setup 一句話交代。

### 4.4 repack 時機
每 cycle 結束，若有新 finished → `repack(keep=未完成)`；B 縮到 0 → 下一 batch。
（好處：無 pad 污染 capture、省算力；成本：index_select 每層一次，µs–ms 級。）

### 4.5 draft 側每 cycle 更新（Ours/SpecMoE）
verify 後呼叫 `controller.update_masks()`（= 現行 on_cycle 的動作）。
hybrid 的 act-sim 捕捉開關由 engine 的 on_draft_start 相位鉤子處理（§3.2 已呼叫）✓。

### 4.6 與 phase.py 的語義對照表（實作時逐條核對）
| phase.py 行為 | batch loop 對應 |
|---|---|
| get_candidates 進入時 in_draft_phase=True + on_draft_start | draft 相位開頭 |
| 離開時 False + on_draft_end | draft 相位結尾 |
| set_profile_phase(1/0) | 同點呼叫（AUG_PROFILE 時） |
| on_question_start → controller.reset() | 每個 batch 開始（注意：merged cache/統計以 batch 為 reset 單位——B=1 時與現行同） |
| on_cycle → update_masks | 每 verify 後 |
| C-BOOT 空首輪 + KV-copy | §3.1 的 prefill + deepcopy |

## 5. 取樣與 batch 組成
- 沿用 t1 協議：`_sample_questions`（qpc=15、mt_bench_pooled、humaneval）→ 依取樣順序
  切成 `ceil(N/B)` 個 batch（最後一個不足額照跑）。混類別 = 真實 serving，圖只報 overall TPS。
- B=1 時 batch loop 逐題跑，與現行題目順序一致（等價驗收的前提）。

## 6. 驗收（每關都有明確判準，不過不進下一關）

| 關 | 內容 | 判準 |
|---|---|---|
| **V0** | CPU 單元測試：FakeCausalLM（logits=f(token,position) 查表、記錄收到的 position_ids/attention_mask）驗：接受/bonus 索引數學、洞 mask 正確性、repack、B=1/2/3 強制參差接受、EOS/mnt 凍結、`stack_caches` 右補 | pytest 全綠；**接受邏輯另寫一份 20 行純 Python 參考實作對拍** |
全部驗收統一用 **qpc=5、mnt=512、humaneval on、mt_bench_pooled on**（= t1/f15
主力協議；長生成讓 AccR 統計穩、單跑即可比，不必靠 n=3 平均掉 offload 噪音）。

| 關 | 內容 | 判準 |
|---|---|---|
| **V1（硬門檻）** | hf backend、B=1：batch loop vs 現行 HF 路徑同題目 | greedy committed **文字逐 token 相同**（bf16 holes-vs-crop 會誘發極少數近平手 argmax 翻轉→容忍 ≤2 題在首個洞之後岔開，方向須混合）；aggregate AccR 差 ≤0.02 |
| **V2** | offload、B=1：ours + specmoe | AccR 差在非確定性噪音內（≤0.03）；no hang（watchdog） |
| **V3** | B=4：不變量 debug 模式（每 cycle assert：洞數=Σ(T−k)、邏輯長=committed 長、mask 單調、position 連續） | 全部 assert 過；AccR 與 B=1 差 ≤0.05（bf16 批次數值差容忍） |
| **V4** | B=64 pilot：VRAM 峰值、cycle 時間 | 無 OOM、無 stall → 才排全 sweep |

## 7. Sweep 矩陣（V4 過後）
{moecache, specmoe(ep1), ours(hybrid a75)} × B∈{1,4,16,64}，t1 協議（qpc15/mnt512/humaneval/pooled），
各 1 run（B=1 直接用 batch loop 跑，與 t1b 數字互為 sanity）。12 個 job，
高 B 的 wall 反而短。產出 `scripts/plot_batch_sweep.py` → fig:batch_sweep。

## 8. 檔案清單
| 檔 | 動作 |
|---|---|
| `src/aug_spec/runtime/batch_spec.py` | 新增（BatchState、cache 工具、run_specbench_batched，估 600 行） |
| `src/aug_spec/cli.py` | +2 欄位、+1 分支 |
| `tests/unit/test_batch_spec.py` | 新增（V0 全部） |
| `configs/bsweep_{mc,sm,ours}_b{1,4,16,64}.yaml` | 新增 12 個 |
| `scripts/plot_batch_sweep.py` | 新增 |
| 其他一切 | **不動** |

## 9. 時程與停損
| 天 | 內容 |
|---|---|
| D0 | 骨架 + cache 工具 + V0（FakeLM 測試全綠） |
| D1 | hf 真模型接通 + **V1 硬門檻** |
| D2 | offload 接通（engine 鉤子、none 模式）+ V2/V3。**D2 結束 V1/V2 未綠 → 停損**：凍結本計劃，改做 trace 模擬圖（選項 B，1 天） |
| D3 | V4 pilot → 送 12 個 sweep job |
| D4+ | 收數、畫圖；緩衝 |
前置條件：t1b 六列已收數、論文其餘欄位無 blocker。

## 10. 已知偏差聲明（寫進論文 setup 用）
1. 靜態 batching + 完成即 repack（無 continuous batching）——對三系統一視同仁。
2. draft 統計（top-M 計票 / kept-N / hybrid 訊號）在 batch 內**池化共享**——
   這是 batch 下的自然設計（SpecMoE 論文亦然），B=1 退化為現行語義。
3. bf16 批次 kernel 與單序列數值不 bit-exact → 不同 B 的 AccR 有 ≤幾 pp 漂移，
   圖以 TPS 為主張、AccR 僅附註。

## 11. 狀態追蹤
| 項目 | 狀態 | 備註 |
|---|---|---|
| D0 骨架 + V0 | ✅ 2026-07-13 | `runtime/batch_spec.py`（~450 行）+ cli 接線 + `test_batch_spec.py` 11 測試（全接受/全拒/參差/EOS/mnt/T=0/批次=單序列逐 token 一致/洞 mask/邏輯位置/repack/合批），全套 77 綠。設計期抓到的 k==T 角落（p_T 未餵 assistant → step-1 寬 2 補餵）已實作並被 all-accept 測試每 cycle 打到 |
| D1 hf + V1 硬門檻 | ✅ 2026-07-13 | q5/mnt512、hf topm_count、B=1 vs 現行 HF 路徑。byte-exact 逐 token 比對（AUG_DUMP_COMMITTED，jobs 260757）：**12/14 題完全相同**，2 題（395@tok6、95@tok47）在首個洞之後岔開；aggregate MAT/AccR 差極小且方向混合（8 上 6 下）＝ bf16 holes-vs-crop 無害噪音，V0 已在確定性 FakeLM 上鎖住邏輯。判定通過 |
| D2 offload + V2/V3（停損檢查點） | 🔶 partial | q5/mnt512（jobs 260924/260925）：**SpecMoE ✅ 驗證通過**（batch vs legacy ΔAccR −0.003、ΔMAT +0.012，全在噪音內；前次 q3 的 −0.05 確認為 n=21 噪音）。**Ours（cache mode + C3 pipeline `merge_overlap`）❌ silent hang** 於 ~第 22/35 題被 watchdog 砍（rc=124，無 C++ FATAL）→ 之前的「ΔAccR −0.068」是殘缺數據無效。legacy Ours 跑完整 35 題無恙 → 是「batch loop × C3 pipeline」相位驅動時序的死結。診斷 job 261411（心跳 + AUG_HANG_DEBUG）定位中。選項：修 C3 交互（選 1，進行中）／ Ours 退 `merge_overlap: false`（選 2 繞過） |
| D3 V4 pilot + sweep 送出 | ⬜ | |
| 收數 + fig:batch_sweep | ⬜ | |
