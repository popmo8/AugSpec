# Remove-Overload Plan:刪除 moe_infinity overload 路徑、pin-aware evict 成為唯一路徑

> 2026-07-08 起草,待批准後動工。動 C++ 前先讀 `memory_hierarchy_plan.md` §6 風險註記。

## 0. 目標與動機

把 moe_infinity 原生的「overload」路徑(cache 滿 + batch>1 時帳外硬塞一顆、用完即丟、
單 slot 序列化、驅逐標記有 race、無視 pin)**整段刪除**,讓所有 fetch(batch=1 與
batch>1、specmoe 與 topm、prefill 與 verify)一律走 `FindExpertEvict`:
**pin 決定誰不可被 evict(`pinned_` skip),滿了就一定 evict(LFU + timed-wait)**。
`no_overload` 這顆旋鈕隨之失去意義 → 廢棄。

兩法共用同一個 evict 引擎(同一個 `ExpertDispatcher`、同一個 `FindExpertEvict`、
同一份 `pinned_`),所以這是一次性、對兩法同時生效的改動,公平性不受影響。

## 0.5 為什麼安全(等價論證,先讀)

今天 `no_overload: true` 之下,`no_overload_ == true` ⇒ overload 分支
(`expert_dispatcher.cpp:610`)永遠不進 ⇒ `gpu_overload_[gpu_id]` 恆為 false ⇒
- `:719` 的 `if (!gpu_overload_[gpu_id])` 恆真(記帳必然執行);
- `:779` 的 `exec_args.evict` 恆為 false ⇒ OutputFunc `:874` 的 evict 區塊是死碼。

也就是說:**要刪的程式碼在 no_overload=true 下已可證明是死碼**,刪除後留下的
路徑與近月所有 no_overload runs(q15 系列、ep probe、q5_512 系列)跑的
byte-for-byte 是同一條。行為改變只發生在「沒開 no_overload 的舊 config」——
而那正是本次要的 default 翻轉。

## 1. C++ 改動(`moe_infinity/core/parallel/expert_dispatcher.{cpp,h}`)

1. **刪 overload 分支**:`GPUFetchFunc` 的 `if (batch_size > 1 && !no_overload_) {...}`
   (`:610-627`,含 busy-wait 與 overload_wait 計時)整段刪除,留下的
   else 臂(FindExpertEvict + 2ms timed-wait 重試迴圈,即 deadlock fix)無條件執行。
   - `:595` 的區域變數 `batch_size` 此後只剩這裡在用 → 一併刪(避免 unused warning)。
     `:813` / `:871` 是別的函式,不動。
2. **記帳無條件化**:`:719` `if (!gpu_overload_[gpu_id])` 拿掉 guard,
   `cache_sizes_ -=` 與 `cached_experts_.insert` 直接執行。
3. **刪 `ExecArgs.evict`**(`.h:48`):唯一的 true 來源就是 overload。
   - `:185`、`:779` 兩處賦值刪除;
   - OutputFunc `:874-895` 的 `if (args.evict) {...}`(丟回 host + 清旗標)整段刪除。
4. **刪成員與初始化**:`gpu_overload_`(`.h:233`、ctor `:47`)、
   `no_overload_`(`.h:267-271` 含註解、`:54` 的 getenv)。
   順手清掉散落的註解屍體(`:59`、`:74`、`:617`、`:893`、`:898-903` 等
   gpu_overload / futex 舊註解)。
5. **刪 profiling counters**:`overload_wait_n/us`——struct 欄位、reset(`:452`)、
   JSON 匯出(`:470-471`)。計時程式碼隨 #1 一起消失。
6. **不動**:FindExpertEvict 的 pin-skip / LFU / victim-lock 語意、timed-wait
   deadlock fix、phase-tagged fetch profiling、set_pinned 介面。

## 2. Python 改動(`src/aug_spec/`)

1. `runtime/loader.py`:刪 `no_overload` 參數(`:176`)與設 env 的區塊(`:202-206`)。
2. `cli.py`:
   - 刪 `RunConfig.no_overload` 欄位(`:104-105`)與 `:361` 的傳遞;
   - `:201` 解析處改為:YAML 若出現 `offload.no_overload`,印一行 deprecation
     提示(「已成預設且路徑已移除,此欄位無作用」)後忽略——**不報錯**,
     因為 ~15 份既有 config 都寫著它;
   - `AUG_NO_OVERLOAD` env 從此無人讀,自然變 inert(舊 script 的 export 無害)。
3. `_dump_profile`(`cli.py:258`):刪 `overload_wait` 這一 row,
   並同步 `:232` 的 docstring(拿掉 overload_wait 的說明)。
4. grep 收尾:`grep -rn "no_overload\|NO_OVERLOAD\|overload" src/ tests/ moe_infinity/core`
   確認除了歷史 .md 之外零殘留。`tests/` 目前無 no_overload 引用(已確認)。

## 3. 文件同步(CLAUDE.md 規則 4)

- `configs/README.md`:model.offload 表刪 `no_overload` row、env overrides 表刪
  `AUG_NO_OVERLOAD`;在 deprecation 段落記一筆(既有 YAML 的該欄位被忽略)。
- `PROJECT_GUIDE.md`:
  - 「offload config 關鍵欄位」拿掉 `no_overload: true`;
  - 「旋鈕」段的 no_overload bullet 改寫為「已內建為唯一行為並刪除旋鈕(2026-07-xx),
    歷史效果:topm +24% / specmoe +33% TPS」;
  - Profiling 段的 row 列表拿掉 `overload_wait`;
  - 「已知關鍵結論」的公平條件「都開 no_overload」改為「no_overload 已內建」;
  - 舊 script `export AUG_NO_OVERLOAD=1` 的註記改為「inert」。
- `memory_hierarchy_plan.md` 等歷史計劃文件不改寫(保留當時脈絡),只在
  §6-4(demand-load smoke 提到 no_overload/evict 改動)加一行指回本 plan。
- **不清洗既有 YAML**:`no_overload: true` 留在舊 config 裡(歷史可重現性),
  新 config 不再寫。

## 4. 重編與驗證(一律 sbatch,不在 login node 編譯/跑 GPU)

| # | 步驟 | 怎麼跑 | 通過標準 |
|---|---|---|---|
| 1 | 重編 C++ | `sbatch moe_infinity/rebuild.sh`(ninja 增量) | build 無 error/warning |
| 2 | 單元測試 | login node 可:`.venv/bin/python -m pytest tests/unit -q` | 全綠(純 CPU,不碰 dispatcher) |
| 3 | smoke(兩法各一) | sbatch 小 run:specmoe(ep2)+ topm 各 qpc=1/mnt=64,`AUG_PROFILE=1` | 不 hang、跑完;profile 無 `overload_wait` row;specmoe kept 駐留 ~99% |
| 4 | 守門 A/B | 重跑 `q15_specmoe_ep2`(不改 config)比對 job 254248:TPS ~4.1、AccR ~0.486、draft_fetch ~0.95TB | 落在 run-to-run noise 內(AccR ±~0.02、TPS ±~0.3;§0.5 等價論證下預期一致) |
| 5 | 舊-default config 抽查 | 挑一份沒寫 no_overload 的舊 config(如 `cmp_opt_specmoe`)跑 qpc=1 | 行為 = 新路徑(TPS 應明顯優於它的歷史數字)——確認 default 真的翻轉 |

輪詢 job 狀態一律 `sleep 60`(SLURM 30 秒規範)。

## 5. 風險與已知後果

- **主要風險極低**(§0.5:留下的路徑已被近月所有 runs 驗證)。真正的行為改變
  只影響「沒開 no_overload 的舊 config」,那是本意。
- **歷史數字不可再重現**:overload 時代的 runs(如 `cmp_opt_specmoe` TPS 2.60、
  `exp_earlypin_244275`)在新 build 下無法復跑出原數字——它們本來就已被
  no_overload 系列取代,論文不引用,保留 output 目錄當歷史即可。
- **極端 edge case**:若某層 cache 內全部是 pinned/在用的 expert,fetch 會在
  timed-wait 迴圈等到有人釋放——這是「滿了就必須 evict、絕不超帳」的預期語意。
  現行 budget(0.2×)+ N=16 pin 下從未觸發餓死;若未來 pin 集合大幅變大
  (memory_hierarchy 的 kept reserve 實驗),需重新評估,已在該 plan §6 有註記。
  **→ 2026-07-09 已加 pinned-starvation guard**:(a) cache 全 pinned(evict
  零候選,非暫時性鎖)時的 fetch 當下即 `DLOG_WARN`;(b) 若全 pinned 且 exec
  pipeline 已排空(exec queue 空 + in-flight 0,追蹤用新 `exec_active_` 計數)
  **持續 10 秒**(期間每秒再 WARN 一次),即為可證死結——Python 端必然卡在
  `Wait()` 等這個 fetch,unpin 永不可能到來——`DLOG_FATAL` 直接 abort,
  不讓 run 無聲卡死。`FindExpertEvict` 新增 `all_pinned` out-param 區分
  「全 pinned」與「被鎖(等 exec 完就好)」兩種 nullptr。
- **profile 格式改變**:`overload_wait` row 消失,任何解析 profile 文字的舊
  script/分析需知悉(PROJECT_GUIDE Profiling 段同步)。

## 6. 明確不做的事

- 不動 FindExpertEvict 的選擇策略(LFU + pin-skip 維持現狀;換策略是另一個題目)。
- 不動 early_pin / merged_backend 等其他旋鈕。
- 不改舊 YAML、不重寫歷史 plan 文件的當時結論。

## 7. 狀態追蹤

| 項目 | 狀態 |
|---|---|
| §1 C++ 刪除 | ✅ 2026-07-08(expert_dispatcher.{cpp,h};grep 零殘留) |
| §2 Python 刪除 + deprecation 提示 | ✅ 2026-07-08(cli.py/loader.py;有 key 的 YAML 印一次提示,解析驗證過) |
| §3 文件同步 | ✅ 2026-07-08(configs/README、PROJECT_GUIDE、memory_hierarchy_plan §6-4) |
| §4-1 rebuild | ✅ job 257130:build_ext rc=0、import OK |
| §4-2 unit tests | ✅ 15/15 pass(純 CPU,不碰 dispatcher) |
| §4-3 smoke ×2 | ✅ job 257130:specmoe(310 cyc, TPS 3.29)與 topm(244 cyc, TPS 3.30)都跑完無 hang;specmoe kept 駐留 99%(bmm engage);profile 無 overload_wait row |
| §4-4 守門 A/B(q15_specmoe_ep2 重跑) | ✅ job 257131:post-removal MAT 3.552 / AccR 0.5146 / TPS 4.197 / draft_fetch 972GB vs 基準(254248)3.417 / 0.4860 / 4.105 / 965GB——AccR +2.9pp ≈ 1σ、TPS +0.09,noise 內吻合(方向還略優);§0.5 等價論證成立 |
| §4-5 舊-default 抽查 | ✅ 併入 §4-3:兩個 smoke config 刻意不寫 no_overload(舊 default),跑的就是新唯一路徑;加上 overload 程式碼已不存在,無 A 可比 |
| §5 pinned-starvation guard(2026-07-09 追加) | ✅ 實作 + job 257171 驗證:rebuild rc=0、雙 smoke 跑完(specmoe 271 cyc / topm 218 cyc,kept 駐留 99%)、log 零 "fully pinned"(無誤觸) |
