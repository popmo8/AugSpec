# GPU Merged-Expert Cache Plan(2026-07-09 起草;同日兩次修訂,現行 = A′ 單一 pool)

> **取代 `memory_hierarchy_plan.md`(CPU 三層設計已作廢)。**
> 規格來源是論文:`paper/method_reconstruct.tex`(Merge-Aware System
> Optimization:merged cache、retention 規則、Fetch-Overlapped Merge
> Pipelining)與 `paper/method_relation_merge.tex`(greedy pairing、
> Eq. weightmerge 的 **uniform 1/|C| 係數**、singleton 定義)。
> 動工前照 §7 狀態表逐項執行;C++ 改動一律 sbatch rebuild。

## 0. 目標

1. **不設容量旋鈕(auto 模式)**:merged expert 盡可能保留,verify
   缺空間時 on-demand 淘汰(discard,零 D2H);唯一的硬限制是內部
   **verify floor**(pin 總量 ≤ pool − c×單層最大工作集,c≈2,內部常數)。
   不新增任何 YAML 容量欄位。
2. **所有 GPU expert 權重收斂到單一 C++ pool(archer)**:原始 expert、
   merged expert、singleton 全在同一個 cache,pin 管工作集、pin-aware LFU
   管其餘 —— **C++ 只加機制,policy 全留 Python**。
3. 落實論文兩個系統機制(content-addressable cache + retention、
   fetch-overlapped merge pipelining)+ singleton 雙重身分優化。

### 決策演進(留檔,避免重走)
- v1(兩空間):merged 留 Python arena、original 留 archer,singleton 走
  archer pin → 發現 pairs/singletons 數量逐 cycle 浮動,固定 ratio 會
  「一邊擠一邊空」。
- v2(統一 slot 模型):singleton identity-複本進 arena,draft 需求恆 K×L
  → 解掉浮動,但 verify 看不到 arena 裡的 singleton 複本,同一份權重
  verify 還是要重抓;且有暫態雙份。
- **v3 = A′(現行)**:單一 C++ pool。決定性理由:**singleton 的雙重身分
  (draft expert = verify expert)只有單一 pool 能兌現** —— pin 在 archer
  的 singleton,verify route 到 = hit、免 fetch。specmoe 的 pinned kept-N
  今天就享有這個優勢(verify route kept-N = hit),topm 不做等於自帶結構性
  劣勢。v1 反對 A 的理由(policy 進 C++、動態內容生命週期)以「C++ 只加
  機制、policy 留 Python、merged 驅逐 = discard(可重建)」化解。
- **v4(同日追加):容量旋鈕 → auto 模式**。引擎的 verify 是逐 expert
  fetch↔compute 串流(fetch thread → exec queue → exec worker),不是
  「fetch 一批再算」——verify 真正需要保障的 buffer 是**常數級**
  (~1–2 層工作集,≤1GB),不是 pool 的比例。所以容量旋鈕不必存在:
  history 無壓力時自然長滿 pool,壓力來了 on-demand discard;唯一硬限制
  是 verify floor(§3)。(曾短暫考慮過 draft 側容量旋鈕,未實作即撤銷——
  不要重新發明它。)

## 1. 現況盤點(2026-07-09 讀碼結論)

- topm 現況兩個空間:archer C++ pool(原始 expert;`FindExpertEvict`
  pin-aware LFU;overload 路徑已刪)+ Python `draft_cache`(K 顆 merged,
  **固定 reserve** `K×L×expert` 在 `cli.py:328` 從 usable 扣掉)。
- merged 零 cache 行為:每 cycle 整層重建;`merging/linear.py` 已預留
  member-set key hook(B3)。
- merge 執行:`dispatcher.merge_experts_local(layer, ids, ws, gpu)` —
  GPU 讀 resident 成員(0 PCIe),回傳新 tensor(**要改成寫進常駐 slot**)。
- clustering:`hybrid` = 論文 R map(λ=`alpha`),`greedy_pair` 產出
  pairs + singletons;|active| < 2K 時 singleton 常見;p + s = K(恆定)。
- **係數前提已滿足(2026-07-09 查證)**:member-set content-addressing 只在
  uniform 下成立——而 **q15 campaign 的 48 份 clustering config 全部
  `within_weight: uniform`**,與論文 Eq.(weightmerge)一致,論文 acceptance
  數字本來就是 uniform 的。code 預設仍是 freq(舊 q5 freq_slice 結論),但
  pair 方法的實務 config 都是 uniform → 不需要 gate 實驗,q15 uniform 數字
  直接當 C1 的 AccR 基準。
- specmoe 的 pin 機具直接複用:`set_pinned` / early_pin(ep2)/
  `kept_bmm_state` resident-gather(99% engage 已驗證)。
- **⚠ `EvictLayer` 不看 pin**(把該層 cached 全踢)→ C1 必改(否則踢掉
  pinned singleton / merged slot)。
- **`evict_layer` 直接由新設計蓋掉(2026-07-09 定案,不另做 on/off A/B)**:
  它的存在理由(避免 cache 填滿觸發 overload race)隨 overload 刪除而消失。
  新 plan 下 footprint 控制移交 demand-driven 驅逐(§2.5 接口調整清單);
  C++ `EvictLayer` 與舊 P2/P4 evict 路徑**保留當 legacy 逃生門**
  (`AUG_LEGACY_MERGE`,正常路徑零呼叫),最終驗收後移除。
- **archer 的角色**:offload 執行引擎(佔位參數、pinned host pool、
  C++ fetch/exec worker、H2D-compute overlap)。verify 每 run 沖 ~16TB,
  這套機械不可能搬 Python —— 所以是「全部進 C++」而不是「全部進 Python」。
- pinned-starvation guard 已上線(warn + 10s drained→fatal),pin 擴張的
  安全網已備;`exec_active_` 計數也已存在。

## 2. 架構:單一 C++ pool(A′)

**C++ = 機制(mechanism),Python = 政策(policy)。**

### 2.1 Expert id 空間
- 每層:`0..n-1` = 原始 expert(有 host backing,行為不變);
  **`n..n+S-1` = merged slot**(合成 id;S = 每層 slot 上限,由
  pool − floor 推得)。
- `experts_` 表在建構時延伸 S 個 slot node;`cached_experts_` / `pinned_`
  的 key 編碼(`layer<<32|id`)天然支援合成 id。
- merged slot node:**無 host backing**,內容可由 re-merge 重建。
  - 進駐:lazily 配 GPU buffer(記入 `cache_sizes_`,同一本帳)。
  - 驅逐:**discard**(free buffer、清 cached 記錄;不做 D2H)——
    `FindExpertEvict` / `EvictLayer` 遇到合成 id 走 discard 分支。
- verify routing 永遠只產生 `0..n-1`,不會 route 到合成 id(gate 輸出
  空間沒變),所以 merged slot 對 verify 是透明的。

### 2.2 新增/修改 C++ API(全部是機制,無 policy)
| API | 內容 |
|---|---|
| `merge_experts_to_slot(layer, slot_id, member_ids, weights, gpu)` | `merge_experts_local` 變體:merge 結果寫進 slot node 的常駐 buffer(首次配置 + 記帳),插入 `cached_experts_`。 |
| ~~`EvictLayer` 跳過 `pinned_`~~ | 作廢——EvictLayer 整個刪除(§2.5 C-DEL),無需 pin-skip。 |
| `FindExpertEvict` / `EvictLayer` 的合成-id discard 分支 | 驅逐 merged slot = free buffer,不 SetDevice(host)。 |
| `FindExpertEvict` 兩段式驅逐(C1 預設) | 先掃 unpinned **合成 id**(merged history = 機會性 tier、可重建),沒有才對原始 expert 走 LFU。判斷只看 id 範圍(機制非 policy)。unpin 後的 ex-singleton 就是普通原始 expert,無特例。 |
| residency probe | 現成:`get_resident_expert_weights(layer, id, gpu)` 空返回 = miss。 |
| (既有)`set_pinned(layer, ids)` | ids 可含合成 id,不用改。 |

### 2.3 Python 政策(全在 Python,C++ 不知道 key/retention)
- **Content index**:`(layer, frozenset(members)) → slot_id`;singleton 的
  key = `(layer, {i})` 但**不佔 slot**——直接 pin 原始 id `i`。
- **兩段式 cache-first partition(2026-07-09 補,pairing 必須遵守保護級
  history,不靠 greedy 自然重現)**:
  1. **adopt**:掃保護級 history,pair 兩成員都 ∈ 本輪 𝓜_ℓ → 直接採用
     為本輪 group(hit:0 fetch 0 merge),成員移出 candidate 池、K 額度
     扣一;同一 expert 落在多個 cached pair 時按 retention 分數取捨,
     落選 pair 留作 history 不刪。
  2. **greedy**:剩餘 candidates 跑 `greedy_pair`,配 K−(adopted) 組。
  依據:greedy 的 tie-break 與 C map 漂移會偶爾翻掉本可 hit 的 pair;
  staleness 由「兩成員都還在 𝓜」擋住大半,uniform 係數下採舊 pair 的品質
  差為二階(Prop. 2 論證仍成立)。**保險絲**:C1 驗證 q15 AccR 不掉;
  掉了降級為 sticky β(cached pair 加 R bonus,β=∞ 即 adopt、β=0 回
  論文自然重現;`reconstruct_fetch_plan.md` 有 β 先例)。
  telemetry 加 `adopted_pair_rate`(hit rate 主成分)。
- **工作集 pin(= specmoe kept-N 的推廣;ep2 語意,精確版)**:
  - pin 的依據是**下一輪 draft 工作集**(新 partition 的 singleton 原始
    id + merged slot id),不是「這輪 verify 有 route 到」——verify-only
    的 expert(routed 但不進新 𝓜,routed>M 時常見)從不 pin,算完留在
    cache 自然淘汰。
  - 時序照 specmoe `early_pin: 2` + `late_unpin`(specmoe.py:165-199,
    ep probe 已驗證):verify 該層 capture 完 pin **舊工作集 ∪ 新工作集**
    (聯集),該層 dispatch 全部算完後 late_unpin 只留新工作集——跌出
    𝓜 的舊 singleton **算完才失去保護**,不會 mid-compute 被擠出。
  - 「cache 裡的先算」是引擎天然行為:Enqueue 時 resident expert 直接進
    exec queue(cpp:167-186),不排 fetch 隊——resident-first 不需另做排程。
  - 淨效果:singleton 由 verify 自己的 fetch 帶進來,下輪 draft 0 PCIe,
    **且 verify route 到它時已是 hit(雙重身分,免重抓——本計劃核心新收益)**。
  - **hook 落點(C1/C2 實作對照)**:
    1. capture(averaged forward target 相位)→ 算新 partition(**stash,
       不重算**)→ probe → `set_pinned(舊∪新)`:singleton 原始 id +
       hit 的 slot id;
    2. dispatch:resident-first 免費、pinned 保護到算完;
    3. `on_verify_layer`(post-dispatch,P2/P3 現有 hook)→ miss 的 pair
       用 stash 的 partition `merge_experts_to_slot`(成員 resident,
       0 fetch)→ pin 新 slot;
    4. late_unpin(該層 dispatch 完)→ pin 縮回新工作集,跌出者變
       unpinned history。
    topm 這邊 **硬編 ep2 語意、不開旋鈕**(零新旋鈕原則);`draft.early_pin`
    維持 specmoe 專用。AccR 不受影響有先例:ep probe 實測 ep0/1/2 AccR
    相同(pin 只動 residency/時序,不動 draft 權重)。
- **Retention(論文 retention 規則——文字敘述、無公式;2026-07-06 改寫後
  該小節「no new equations」,舊 label `eq:retention` 已不存在)**:
  - **保護級 history**(pair 兩成員都 ∈ 當前 𝓜_ℓ)→ pin;但受 verify
    floor 截斷:超出 floor 時按 co-occur / R 分數只 pin 前段。
    (worst case (K+M/2)×L×expert ≈ 14.5GB > pool,截斷是必要的。)
  - **一般 history** → unpin,留在 pool 裡自然存活;verify 缺空間時被
    on-demand 淘汰(對 LFU 天然偏冷 → 比熱的 verify resident 先死,
    順序正確;merged 驅逐 = discard,零成本)。C++ 驅逐後 Python 的
    content index 以 residency probe 對帳(probe miss = 已被淘汰)。
  - question 邊界 stats 清空 → LRU tiebreak。
- ~~跨題 warm start~~ **不做(2026-07-10 定案)**:merged expert 必須反映
  **本題** prefill 的 similarity map(拿別題 A map 分出的 pair 硬套語意
  不乾淨);「省重 merge」的目標已由 prefill-time merge 達成(C-BOOT 空首輪
  + P3:每層 dispatch 完就地用本層剛捕到的統計 merge,藏進 prefill 的
  transfer 流,不存在「prefill 完再重 merge 一輪」)。reset() 照舊每題清
  index/pins;cache 的 reuse 範圍 = 題內跨 cycle。

### 2.4 Draft forward
- 每層 K 組 = merged slots + pinned singletons,全部 GPU resident →
  `kept_bmm_state` 同款 gather + stack 進 engine_bmm;
  stack 以 (layer, partition 版本) memoise。
- **fallback 退役為 ablation-only(2026-07-09 定案;YAML flag
  `run.prefill_warmup`,預設 true)**,設計 = 空首輪 + prefill-merge。
  flag 語意:`true`(預設)= 空首輪 warmup,draft 相位缺 merged 即
  **hard assert**(走不到 fallback);`false` = 論文 ablation 用,回到
  draft 先行 + cycle-1 fallback 的舊行為(fallback 分支因此保留在 code,
  僅此模式可達)。放 `run:` 段(specbench 迴圈層、所有方法一體生效);
  命名與既有 `run.warmup`(計時前 compile 暖身)區分,README 並列說明。
  背景:HF assisted decoding 是 draft 先行(get_candidates 先於任何
  target forward;查證 specbench.py),cycle-1 draft 時 count/partition
  不存在,所以現行 code 靠 fallback(draft 走 `_standard_routing` 真
  routing,qwen3.py:233-245)bootstrap——品質無損但貴,topm
  `draft_fetch` ~3.5TB/run(~47GB/題)幾乎全是它。解法:
  1. **空首輪**:patched `get_candidates`(specbench.py 既有 patch 點)
     首輪回 0 candidates → target 純 prefill:capture(count)+ A map
     照收,**P2/P3 在各層 dispatch 完、resident 未逐時就地 merge
     (0 fetch)**,singleton 同時 ep2-pin;HF 對 0 候選 = target 自己
     生成 1 個 token(TTFT token,精確無損,只少那一輪投機加速)。
  2. **第二輪起** draft(含其 prompt-prefill)每層都有 merged/singleton
     → 全程 bmm,warmup 模式下 fallback 再無觸發條件 → **走到即 hard
     assert**(fail fast)。好處:fallback 今天是靜默退化模式,會把未來
     merge 建構的 regression 吸收成「變慢」;assert 讓它立刻現形。
     (分支本體保留給 `prefill_warmup: false` 的 ablation 模式用。)
  - **map 狀態(查證)**:prefill 結束時 A ✓、count/𝓜 ✓、**C ✗**——
    co-occur 是 decode-only(`cooccur_scope: decode`,code 刻意跳過
    prefill;論文「C grows with every verified token」一致,**不改**)。
    C 空時 R blend 的 unobserved sentinel 使其退化為 A-only 排序
    (= 論文 λ=1 端點),cycle 2 起自動回正常 blend。
  - hf backend 同樣適用(空首輪 patch 在 specbench,backend 無關;該輪
    target forward 後 refresh 建 merged)。
  - **KV-copy(2026-07-10 追加,空首輪的必要配套)**:HF assisted decoding
    的 assistant 有**獨立 KV cache**,不會繼承 target prefill 的 KV;空首輪
    後它的第一個 forward 會在 draft 相位把整個 prompt 用 merged/substitute
    重 encode → context KV 劣化 → AccR 崩到 0.01(jobs 258368/258378,
    兩法兩 backend 同崩)。**設計教訓:「draft 權重不變 ⇒ acceptance 不變」
    只對單步成立,draft 的 context KV 品質是 first-order——歷史上 cycle-1
    fallback 一直默默扮演「真 routing prompt KV」的角色。** 修法:共權重
    單模型下 target 的 prompt KV 就是 draft 的最優 context → 空首輪結束時
    stash target cache(透明包 `target_model.forward`,assistant 呼叫以
    in_draft_call 旗標排除),第一次真 get_candidates 時 deepcopy 注入
    `assistant_kwargs["past_key_values"]` → draft 免重 encode prompt、直接
    投機。修復實測(258444):offload AccR 0.028→0.455、hf 0.011→0.302、
    token 對齊率 0.27→0.665、TPS +11%。殘餘 ~6pp AccR 差距 = 每題首 cycle
    不再是 ~100% accept(fallback=target 本人)的已知語意變化,mnt=64 下
    佔比 ~1/6 被放大,mnt=512 稀釋至 <1pp(最終驗收量化)。
    per-cycle accepted-token 的 re-encode 維持 merged(歷史行為)。
  - stale warm start(沿用上一題 partition)**不做**(同跨題 warm start
    的否決理由:draft 狀態一律出自本題 prefill 統計)。
  - 驗收:draft_fetch GB/題 → ~0;MAT 預期持平(首 token 由 target 生成,
    精確;僅失去單一 token 的投機加速)。

### 2.5 `evict_layer` 接口調整(無縫銜接清單,2026-07-09 定案)

驅逐職責從「每層主動清」移交「demand-driven」(fetch 缺位時才逐,
兩段式 §2.2)。唯一呼叫者是 `OffloadMergeEngine.on_verify_layer`:

| 現在 | 新 plan |
|---|---|
| P2:`_merge()` → `cuda.synchronize()` → `evict_layer(ℓ)` | merge 完不清層;sync 是為 evict 而做,一併移除(draft 在 verify 後才跑,default-stream 順序天然保證;C1 可先留 sync 單獨驗證) |
| P4:側流 merge → `_pending=(ℓ,event)` → 下層時 sync+`evict_layer(前層)`;`_drain_pending` 清尾 | 側流 merge 保留;deferred-evict 簿記整段刪;`_drain_pending` 只剩 draft-start 前 sync merge stream |
| P1 `flush_cache`(on_draft_start) | cache 模式禁用(衝突報 error) |
| C++ `EvictLayer` | **直接刪除(2026-07-09 定案,不留死碼)**:函式 + header 宣告 + pybind 綁定 + `prof_.evict_layer` counters + `_dump_profile` row 一併移除;回滾靠 git(同 overload 前例)。連帶 §2.2 的「EvictLayer pin-skip」項作廢。 |

**EvictLayer 先行刪除(C-DEL,獨立於 C0–C3,可立即做)**:
- 風險低的關鍵證據:**specmoe 每個 run 都跑「pool 滿 + FindExpertEvict
  demand-驅逐」模式**(它無 merge engine、從不呼叫 evict_layer),batch>1
  高併發已被 q15/ep-probe/q5_512 驗證;deadlock timed-wait + starvation
  guard 都在。topm 刪除後走的是同一條已踩實的路。
- 範圍:C++ 三件(函式/宣告/綁定)+ counters + `offload_merge.py` P2/P4
  的 evict 呼叫與 deferred-evict 簿記(**P2 的 per-layer `synchronize`
  第一步保留**,避免同時動兩件事混淆歸因,C1 再移除)+ profile row +
  PROJECT_GUIDE 同步。
- 驗收:rebuild + 雙 smoke + **一次 q15 topm 對照既有基準**(TPS/AccR/
  verify_fetch——把原 C-EV 想量的東西併進這裡量)。
- 其餘無縫銜接確認點:(a) pool 跑滿後每次 miss 帶一次 O(unpinned) ≈ µs
  掃描;(b) 原始 expert 驅逐無 D2H,成本不變;(c) specmoe 零影響;
  (d) draft 相位全 resident 不 fetch;(e) vram_guard 照常稽核
  (pool 常態跑滿 = 設計內的誠實 scarce-VRAM 模擬)。

### 2.6 激活與預設

- **會成為預設**:激活面沿用現有 `merge_offload: true` +
  `merge_during_verify: true`(topm 標準 config 既有設定),引擎實作直接
  升級,**不新增 YAML 欄位**。specmoe / 非 merge draft 走不到這些路徑,
  天然不受影響。
- 過渡期(C0–C3)保留診斷 env **`AUG_LEGACY_MERGE`**(設了走舊的
  「每 cycle 重建、無 cache」merge 路徑,供 stage 級 A/B;**不含**
  evict_layer——那個已在 C-DEL 直接刪除、無逃生門);**最終驗收通過後
  刪除**——生命週期同 `AUG_NO_OVERLOAD`(旋鈕 → 預設 → 刪除)。
- 空首輪(C-BOOT)是 specbench 迴圈層改動,**對所有方法一體生效**
  (對比才公平);由 **`run.prefill_warmup`** 控制(唯一的新 YAML 欄位,
  為論文「有/無 prefill warmup」ablation 而設;**實作完成後預設 true**,
  `false` 走舊 fallback 行為)。

## 3. 預算語意:auto,零新旋鈕

```yaml
model:
  offload:
    vram_budget_ratio: 0.2        # 總預算(不變,單一 pool 全額;無新欄位)
```

**(2026-07-10 C1 實作修訂:「單一預算、兩個 arena」取代「單一本帳」。**
原案讓 slot bytes 記在 archer 帳本(`cache_sizes_`),但 slot 實體是
torch-allocator 記憶體——幽靈扣款抽乾帳本、fetch thread 活鎖(job 258694
法醫傾印定案)。兩個 allocator 無法共用一本帳,劃帳如下:)

- **載入時劃帳**:`slot carve = K′×L×expert`(torch 記憶體,draft slots);
  `archer pool = usable − carve`(原始 expert)。總帳恆 = usable =
  `vram_budget_ratio × model_bytes`,零新旋鈕。
  (0.2× @K=16:usable 12.21 → carve 7.25 + pool 4.97GB。)
- **verify floor**:singleton 的 archer pin 總量 ≤ pool − floor
  (floor = 2×48×expert ≈ 0.91GB;超額的 singleton pin 被拒,只損失
  verify-hit 紅利不損正確性——draft 一律由 slot 供貨,見 §4.1 修訂)。
- **K′ adaptive**:`K′ = min(K, (usable − floor) / (L×expert))`,
  budget 印表明示;`K′ < 1` 才 error。
- **帳實守恆(C++ 兩則,2026-07-10)**:fetch 只在「真插入」時扣帳
  (`insert(key).second`);`FindExpertEvict` 掃描時對殭屍條目(key 在帳、
  node 不在 GPU——pin-blind 的 archer prefetcher 所致)當場收屍還帳。
  歷史註記:`EvictLayer` 過去每層無條件 erase,意外扮演殭屍清道夫;
  C-DEL 刪它之後這個**潛伏帳漏**才現形。
- 公平性:兩法同 `vram_budget_ratio`;specmoe 的 kept-N pin 本來就在
  pool 內,對比自然公平。品質軸的 sweep 用 K(draft 寬度)。
- TODO:vram_guard 的稽核上限尚未對齊新劃帳(audit-only,照舊會印
  超額警告;C2 收尾時校正)。

## 4. 兩個論文機制的落點

### 4.1 Content-addressable cache + retention(C1/C2)
- cycle 流程:**adopt**(保護級 hit 直接成組,§2.3 兩段式)→ 剩餘
  candidates greedy → 逐 group probe(residency check)→ hit 免 fetch
  免 merge;miss → 取 slot(空位或 retention 驅逐)→
  `merge_experts_to_slot`。
- **僅 `within_weight: uniform` 時啟用 pair cache**(= q15/論文的標準設定,
  前提已滿足);singleton 的 key 單成員、權重恆 1,任何設定下都可 cache。
- freq 模式 fallback(僅防呆,標準 config 用不到):每 cycle 重建(現行為),
  cache 停用 + 印提示。

### 4.2 Fetch-Overlapped Merge Pipelining(C3)
- **pipeline 完全下游於 partition(§2.3)**:該層 capture → adopt+greedy →
  probe,結果決定 hit(直接用 cache,不 fetch 不 merge)與 miss(才進下面
  的重排與 merge)——沒有 partition/probe 結果,pipeline 無事可排。
- v1:probe(hit 跳過)+ **enqueue 重排**(下輪 partition 需要且 miss 的
  成員排最前 fetch;input_queue FIFO、單 fetch thread,Python 排序即控制;
  C++ 只加一個可選優先序參數——純機制)+ **非 routed 的 wanted 成員附掛
  prefetch**(miss group 成員這輪沒被 route 到時,附在該層 fetch 流尾端,
  與 compute 重疊——否則 merge 點要同步補抓冷成員,這是 v1 最實在的收益)
  + 沿用 P4 side-stream(merge 藏進下一層 fetch)。
  但書:merge 仍在 dispatch 後(P4 藏一層)時,「routed 內重排」本身收益
  有限(dispatch 等全員到齊);動刀前先 profiling merge 點等待中冷成員
  補抓的佔比。
- v2(profiling 決定):層內 per-group 提前 merge(需 completion event);
  P4 已把 merge 藏一層,先量 `merge(P3)`/drain 佔比再說。
- 與 `flush_on_draft_end`(P1)互斥:cache 模式下停用(config 衝突報 error)。

### 4.3 Telemetry(C1 起)
- AUG_PROFILE 新 rows:`merged_hit / merged_miss / merge_elided_GB /
  singleton_verify_hit / pipeline_reorder_n`。
- **`singleton_verify_hit` 精確定義(C1 驗收指標)**:verify routing 要求
  某 expert 時,`cache_hit ∧ 該 expert ∈ pinned_`(= 當前工作集的 singleton)
  → +1(併計 bytes)。每次命中 = verify 省一次 9.44MB fetch,是「雙重身分」
  的直接證據。實作在 C++ Enqueue/fetch 的 hit 路徑(hit 與 pinned 都已在手,
  一個 set 查詢)。判讀:singleton 是 top-count expert,verify 大機率重複
  route → 數字理應每層每 cycle 數次的量級;**≈0 = pin 時序或驅逐有 bug**,
  比 TPS 更早暴露問題。sanity 基準:同 counter 對 specmoe 的 pinned kept-N
  也有效,兩法 hit 率應同量級。
- budget 印表:模式(auto/ratio)/ floor bytes / S / K′ /
  當前 draft-side pinned 佔用 + unpinned history 佔用。

## 5. 實作階段(每階段 sbatch 驗證後才進下一階段)

| 階段 | 內容 | 驗證 |
|---|---|---|
| C-DEL | EvictLayer 全刪(§2.5;**獨立可立即做**,C++ 三件 + engine evict 呼叫 + profile row) | rebuild + smoke ×2 + q15 topm 對照基準(TPS/AccR/verify_fetch) |
| C-BOOT | 空首輪 + `run.prefill_warmup` flag(預設 true;false = ablation 走舊 fallback)+ warmup 模式下 fallback 改 hard assert(§2.4;**不依賴 C0–C3,現有 P3 就能建 merged,獨立可先跑**) | q15 topm:draft_fetch GB/題 → ~0、MAT 持平、TTFT 下降;flag off 重現舊行為 |
| C0 | C++ 機制:id 空間延伸 + `merge_experts_to_slot` + discard 驅逐分支 + `EvictLayer` pin-skip(**先不接 policy,行為不變**) | rebuild + unit tests + smoke ×2(數字不動) |
| C1 | auto 模式(verify floor)+ Python content index + 工作集 pin(singleton 雙重身分生效)+ 移除舊固定 merged reserve | smoke + q15;singleton_verify_hit 上報 |
| C2 | **題內** retention + history(保護級 pair pin 到 floor、其餘 unpin/discard;不做跨題 warm start——2026-07-10 定案) | q15;merged_hit / elided GB(題內 adopt rate) |
| C3 | pipelining v1(probe + enqueue 重排) | AUG_PROFILE;TPS |
| **最終驗收** | 整個 plan 完成後,以**論文方法設定的 q15 三重複(`q15_hybrid_a{λ}_r1–r3`,qpc=15/mnt=512/T=5/vram 0.2/uniform)**為對照重跑同 config | (1) **AccR/MAT 落在 run-to-run noise 內**(差 <3pp 多跑取平均)——理論上 plan 不動 draft 權重,AccR 掉 = 有 bug,這是「沒有錯」的判準;(2) TPS ≥ 舊值;(3) draft_fetch/題 → ~0、assert 零觸發。跨方法基準 = `q15_specmoe_ep2`(4.105/0.486 與復跑 4.197/0.515 當 noise band) |

- 驗證固定套路:`tests/unit` → `smoke_noov_*` → q15(qpc=5、mnt=512、
  skip mt_bench)比對基準(topm 現行 + `q15_specmoe_ep2` 4.11/0.486);
  非確定性 → <3pp 效果多跑取平均。

## 6. 風險與開放問題

1. ~~uniform 係數 vs acceptance gate~~ **已解決(2026-07-09 查證)**:
   q15 campaign 48 份 clustering config 全部 `within_weight: uniform`,
   論文數字即 uniform 數字,cache 前提是現狀不是假設。舊「uniform −6.6pp」
   結論僅適用 q5 freq_slice 大群 regime(PROJECT_GUIDE 已加範圍註記)。
2. **C++ 手術風險(本計劃最大工程風險)**:在剛清完 race 的 dispatcher 上
   加 slot 機制。緩解:C0 先做純機制(不接 policy、行為不變)單獨 smoke;
   discard 分支要過 batch>1 併發測試;starvation guard / pin-aware LFU /
   `exec_active_` 都已就位。
3. **verify floor 的正確性依賴「逐 expert 串流」假設**:引擎現況成立
   (fetch thread → exec queue 逐顆重疊);若未來加大批 prefetch 深度,
   floor 常數 c 要跟著調。floor 截斷 + starvation guard 雙保險,
   budget 印表列 pin 佔用,pin > pool 70% 印 warning。
4. **history 的存活是「機會性」的(設計語意,非 bug)**:兩段式驅逐
   (§2.2,C1 預設:unpinned merged history 先、原始 expert 走 LFU)下,
   verify 壓力大時 history 會被優先洗掉——無壓力才多留;真正要保的
   (pair⊆𝓜_ℓ)走 pin(floor 內)。若實測 hit rate 過低,調大保護級配額。
5. **主表數字大多可沿用**:q15 sweep 本來就是 uniform,acceptance 數字不因
   cache 而變(cache 只省 fetch/merge,不動 draft 權重);要重測的只有
   **TPS/throughput 行**(cache + pipelining 生效後的系統數字,provenance
   規範照 `tab_main_baselines.tex` 檔內註解)。
6. **組合矩陣**:cache 模式固定 `merge_during_verify=true` + `merge_overlap
   =true` 當新預設;`evict_layer` 停用(§2.5,legacy 逃生門
   `AUG_LEGACY_MERGE`);其餘組合只做消融。

## 7. 狀態追蹤

| 項目 | 狀態 |
|---|---|
| C-DEL EvictLayer 全刪(§2.5) | ✅ 2026-07-10:code 全刪、cdel_tm_off liveness 過(4.95GB pool 沖 1.1TB ≈ 12 萬次 demand-evict 零異常、AccR 0.51 健康);**效能 A/B(TPS/verify_fetch vs 歷史)併入最終驗收** |
| C-BOOT 空首輪 + `run.prefill_warmup` + assert + KV-copy(§2.4) | ✅ 2026-07-10:含兩顆 bug 修復(assistant mask 失同步 → crash;draft 重 encode prompt → AccR 崩,KV-copy 解);診斷 258444 過(offload 0.455 / 對齊率 0.665 / TPS +11%);mnt=512 規模的 MAT 持平驗證併入最終驗收 |
| C0 C++ slot 機制(行為不變) | ✅ 2026-07-10:MergedSlot 表 + 5 API + 兩段式回收 hook;binding 檢查 + 六項功能 probe(258137)全過;smoke 行為不變 |
| C1 auto 模式 + content index + 工作集 pin | ✅ 2026-07-10(job 258696):**AccR 0.7036 / TPS 4.489(+15.6% vs freq/legacy 3.884)**、adopt rate 0.52、merge_elided 10.1TB、singleton_verify_hit 3525/cyc(elided 1TB)、pin-denied 0.8%、fallback 0。途中修四 bug:assistant mask 失同步、pin-too-late TOCTOU、pin-blind prefetcher 翻參照(→ singleton 一律 slot 供貨,pin 降級為 verify-hit best-effort)、帳本幽靈扣款+殭屍(→ 劃帳模型 + 帳實守恆,§3 修訂) |
| C2 題內 retention + history(跨題 warm start 不做) | ⬜ |
| C3 pipelining v1 | ⬜ |
| 最終驗收(q15_hybrid 三重複對照:AccR noise 內持平、TPS ↑、draft_fetch→~0) | ⬜ |
| 論文補 adopt-first 一句(hierarchy 段 "likely to reproduce" 之後;現行敘述是「先 greedy 後 probe、retention 靠自然重現」,與 §2.3 兩段式實作方向相反,C1 落地後必補) | ⬜ |
