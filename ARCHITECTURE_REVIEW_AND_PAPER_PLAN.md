# 架構 Review + 論文可重現性計劃（2026-07-03）

> 本文件由資深架構視角對整個 `aug_spec` repo 做的一次性完整盤點，
> 並規劃「放進論文、讓人 reproduce」前必須完成的工作。
> **後續 session（Opus 以下模型）請把本文件當作 roadmap 逐項執行**；
> 每完成一項就把該項狀態改成 ✅ 並附 commit hash / job id。
> 執行守則見最後一節「給後續 session 的守則」——**先讀那節再動手**。

---

## 1. 現況架構總覽（這些是對的，不要動）

```
aug_spec/
├── src/aug_spec/            # pip install -e . 的主套件（src layout，正確）
│   ├── cli.py               # 入口：RunConfig.from_yaml → run_experiment（588 行）
│   ├── controller.py        # adapter × draft 接線；install/uninstall forward（135 行）
│   ├── adapters/            # 模型家族 registry（qwen3 / mixtral / gptoss / base）
│   ├── drafts/              # draft 策略 registry（topm_count=我方、specmoe=baseline、
│   │                        #   base.py = ScoreBasedAvgDraft 合併/分群核心 605 行）
│   ├── clustering/          # ClusterMethod registry（freq_slice / random / cooccur /
│   │                        #   weight_sim / activation_sim）
│   ├── merging/linear.py    # 唯一線性 merge 入口（刻意非 registry）
│   ├── kernels/bmm.py       # SwiGLU 批次 bmm
│   └── runtime/             # loader / specbench(655行) / phase / offload_merge / scorers
├── moe_infinity/            # vendored fork（C++ expert offloading 引擎）
│   └── core/                # 對 upstream 的 diff：+577/−20 行、6 檔
├── configs/                 # 106 個 YAML；configs/README.md = 完整 schema 文件（很好）
├── scripts/                 # 67 個 sbatch script + 7 個 analyze_*.py
├── tests/offload/           # ⚠️ 是手動 probe script + .out，不是自動化測試
├── output/                  # 132 個實驗結果目錄（gitignored）
└── PROJECT_GUIDE.md         # session 上手文件（很好，持續維護）
```

**做對的設計決策（保持）**：
- 「一個 YAML = 一個實驗，加實驗不加 Python 檔」的原則，且 `configs/README.md` 完整。
- drafts / clustering / adapters 三個 registry；draft 用 class attribute 宣告自身性質
  （`holds_merged_residency` 等），CLI 不用 hardcode 名單（A1 成果）。
- merge 刻意不做 registry（永遠線性）——已寫進 PROJECT_GUIDE 關鍵結論，別翻案。
- 診斷用 env（`AUG_PROFILE` / `AUG_DUMP_*`）與實驗定義（YAML）刻意分離。
- q15 系列已建立 r1/r2/r3 三重複 + `analyze_q15.py` mean±std 的慣例——這正是論文需要的模式。

---

## 2. 發現的問題（按嚴重度排序）

### 2.1 🔴 Git 衛生：結果無法對應到程式碼版本（最嚴重）
- **有未 commit 的變更**，且其中包含 **C++ 功能碼**（`expert_dispatcher.cpp/.h`、
  `py_archer_prefetch.cpp` 的 activation-capture，+35 行）與多個 Python 檔、
  未追蹤的 `q15_actsim_*` configs。目前 `.so` 是用未 commit 的 source 編的——
  **任何已跑出的 q15 actsim 結果都對應不到任何 commit**。
- `summary.json` 沒有記錄 git commit hash / dirty 狀態 / GPU 型號 / 環境資訊。
  → 132 個 output 目錄裡的結果，事後無法證明是哪版程式跑的。
- 全部歷史只有 18 個 commit（單顆 commit 粒度過大，如 `85f5436 co-occur merge`
  一次 28 檔 +384/−2541）。既成事實無法改，但**從現在起每個邏輯變更一個 commit**。

### 2.2 🔴 Vendored fork 無變更文件
- `moe_infinity/` 對 upstream 的差異共 +577/−20（EvictLayer、no_overload dispatch、
  DispatchBmm、merge_experts_local、GetCapturedExpertOutputs、profiling …），
  但**沒有任何一份文件列出「我們改了什麼、為什麼」**，也沒記 upstream 的
  base version/commit。論文的 system section 和 artifact 審查都需要這份清單。

### 2.3 🔴 沒有自動化測試
- `tests/offload/` 是一次性 probe（m1–m9），還把 `.out` 結果檔 commit 進 repo。
- 純 CPU 可驗的核心邏輯完全沒有 unit test：`linear_merge` 數值正確性、
  `freq_slice`/各 ClusterMethod 的切分性質、`topm_count` 的 top-M 截斷、
  `RunConfig.from_yaml` 解析與錯誤路徑、`_cluster_and_build` 的 mass 正規化。
  這些都不需要 GPU，pytest 幾秒可跑完——是防止後續（較弱模型的）session
  改壞核心邏輯的**唯一防線**，優先度極高。

### 2.4 🟡 可攜性：67 個 script 硬編個人環境
- `scripts/*.sh` 全部硬編 `/work/morrisliu07/aug_spec`、`--account=MST114471`、
  個人 email、`/work/morrisliu07/job_log`。外人 clone 下來一個 script 都跑不動。
- 論文 artifact 需要一個參數化的 sbatch 模板（環境變數或 `env.sh`），
  而不是 67 個複製貼上的變體。

### 2.5 🟡 依賴鎖定不完整
- root `pyproject.toml` 只有寬鬆下限（`torch>=2.1` 等），**root 沒有 lock file**
  （只有 `moe_infinity/uv.lock`）。torch/transformers 小版本就可能改變
  assisted-decoding 行為 → 必須凍結。
- CUTLASS 用 `git clone` 最新版（無 pin）；Spec-Bench `question.jsonl` 抓
  GitHub `main` branch（無 pin）；HF model 沒 pin revision。三個都是上游一動、
  結果就不可重現的破口。
- repo **沒有 LICENSE**（moe_infinity 自帶 Apache-2.0）。公開 artifact 必補。

### 2.6 🟡 configs / output 無法對應論文
- 106 個 YAML 裡多數是歷史一次性實驗（cmp_* / exp_* / prof_* / smoke_*），
  和最終論文表格用的 config 混在同一層。審稿人無法知道「哪 8 個 YAML 對應 Table 2」。
- `paper/sec3_4_verify_merge.tex` 被刪除（未 commit 的刪除）——論文本體放哪裡？
  需要一個明確決定（分離 repo 或 `paper/` 目錄）並寫進 PROJECT_GUIDE。

### 2.7 🟢 程式碼層面（小，非必須，最後做）
- `adapters/base.py` 的 module 全域 `_MERGED_BACKEND`/`_EARLY_PIN` + 
  `apply_offload_settings()` mutation：可運作但脆弱（import 順序敏感）。
  可收成一個 `OffloadSettings` 單例。**風險大於效益時就不動**。
- `drafts/base.py` 裡 4 個 `_maybe_dump_*` 診斷方法約 130 行，稀釋核心可讀性，
  可抽到 `drafts/diagnostics.py`（純搬移，不改邏輯）。
- `cli.py:run_experiment` 244 行一條龍（budget 計算/載入/組裝/跑/寫檔）。
  可拆成 3–4 個函式，但**論文前不值得冒險**。
- `_NamespaceFromDict.__getattr__` 依序掃 model/draft/run/output 四個 section，
  同名 key 會被 model 段遮蔽——加一行註解警告即可。
- mixtral / gptoss adapter 已「退役」但仍在：**留著**（支撐「方法通用」的敘事），
  但在 `adapters/__init__.py` docstring 標注「Qwen3 之外目前 untested」。
- `os._exit(0)`（C++ thread pool 卡 shutdown 的 workaround）：保留，已有註解。

---

## 3. 論文可重現性 Gap 分析

以「審稿人/讀者拿到 artifact 能重現主表」為標準，缺這些：

| # | Gap | 現況 | 需要 |
|---|-----|------|------|
| R1 | 結果↔版本對應 | summary.json 無 hash | 每次 run 記 git hash+dirty、GPU、driver、torch 版本 |
| R2 | 環境凍結 | 無 root lock | `requirements-lock.txt`（pip freeze）+ 記 CUDA/gcc/CUTLASS commit |
| R3 | 資料 pin | Spec-Bench 抓 main | pin commit hash 的 raw URL + 檔案 sha256 檢查 |
| R4 | 模型 pin | 無 revision | YAML `model.revision` 欄位 + 主 config 填上 HF commit hash |
| R5 | 主表 config 集 | 混在 106 個 YAML | `configs/paper/` 目錄，一表一組 config |
| R6 | 一鍵重現 | 67 個個人 script | `reproduce/` 目錄：模板化 sbatch + 產表 script |
| R7 | 統計嚴謹 | 已知非確定性 SD~0.018 | 主表全部 n≥3 重複、報 mean±std（q15 慣例推廣） |
| R8 | fork 說明 | 無 | `moe_infinity/AUGSPEC_CHANGES.md` |
| R9 | 硬體/時間需求 | 無 | README 寫明：1×H200(或等效)、host RAM 需求、201GB disk（offload export）、每組實驗預估時數 |
| R10 | License | 無 | 加 LICENSE（建議 Apache-2.0，與 vendored 依賴相容） |

**已經有、只要包裝的**：offload export 有 `prep_base_offload.sh` 冪等邏輯；
`analyze_q15.py` 已是 mean±std 聚合範本；`configs/README.md` schema 文件完整；
`install.sh` 已能從零建環境（缺 pin 而已）。

---

## 4. 工作計劃（P0→P4，依序執行）

### P0 — 止血：讓「現在」可考證（半天，先做，全部低風險）

1. **Commit 現況**：
   - 檢視 `git status`；未追蹤的 `configs/q15_actsim_*` 等一併加入。
   - 分成語義 commit：(a) C++ activation-capture + clustering actsim/weightsim；
     (b) 文件/舊檔清理（deleted plans、cluster_labels、diary）。
   - 打 tag：`git tag pre-paper-freeze`。
2. **`summary.json` 加 provenance 區塊**（改 `cli.py:run_experiment`，~20 行）：
   ```python
   "provenance": {
     "git_commit": <rev-parse HEAD>, "git_dirty": <bool>,
     "torch": torch.__version__, "transformers": ...,
     "gpu": torch.cuda.get_device_name(0), "hostname": ..., "timestamp_utc": ...,
   }
   ```
   git 資訊用 `subprocess.run(["git", ...], cwd=repo_root)`，失敗時填 `null` 不得中斷 run。
3. **凍結環境**：`.venv/bin/pip freeze > requirements-lock.txt` 並 commit；
   在檔頭註解記錄 `cuda/12.6, gcc/11.5.0, CUTLASS commit <hash>, driver <version>`
   （CUTLASS hash 用 `git -C /work/morrisliu07/cutlass rev-parse HEAD` 取得）。

**驗收**：`git status` 乾淨；跑任一 smoke config 後 summary.json 有 provenance。

### P1 — 測試防線（1 天；在任何進一步改動之前）

建 `tests/unit/`（pytest，純 CPU，不碰 GPU/moe_infinity）：
1. `test_config.py`：`RunConfig.from_yaml` 完整欄位/預設值/錯誤路徑
   （缺 model.id、壞 dtype、壞 within_weight 要 raise）。
2. `test_linear_merge.py`：用 2–3 個小假 expert（純 tensor dict）手算
   加權平均，驗 `linear_merge` 數值（atol=1e-6）。
3. `test_cluster_methods.py`：每個 ClusterMethod 驗性質——
   群覆蓋全部 active、無重疊、群數 = min(K, |active|)；freq_slice 額外驗
   「按頻率排序切片」的確切輸出；random 驗 seed 固定則輸出固定。
4. `test_score_drafts.py`：`topm_count` 給定 count 向量驗 top-M 截斷+renorm；
   `_cluster_and_build` 驗 masses 總和=1、experts 依 mass 降冪。
   （需要一個 ~30 行的 FakeAdapter/FakeBlock fixture，讓 `_build_one` 可以在
   純 CPU tensor 上跑。）
5. `pyproject.toml` 加 `[project.optional-dependencies] test = ["pytest"]`；
   README 加「跑測試：`pytest tests/unit`」。tests/unit 在 login node 跑是允許的
   （純 CPU、秒級，不違反 no-compute 規範；不確定就 sbatch）。

**驗收**：`pytest tests/unit` 全綠；之後任何核心改動先跑它。
（`tests/offload/` 舊 probe 移到 `tests/probes/` 並在目錄加 README 註明
「歷史手動診斷，非測試」；`.out` 從 git 移除。）

### P2 — 論文實驗基礎設施（2–3 天，核心）

1. **`configs/paper/` 目錄**：把最終主張用到的 config 收進來（複製，不搬移，
   舊路徑別的 script 還引用著）。命名 `t<表號>_<描述>_r<重複>.yaml`，例如
   `t1_topm_k16_r1.yaml`。每個檔頭註解寫「對應論文 Table/Figure 幾、哪一列」。
   主對比（topm vs specmoe @ b=0.2, mnt=512, no_overload）**每邊至少 r1–r3**。
2. **YAML 加 `model.revision`**：`cli.py` 傳給 `from_pretrained(revision=...)`
   （hf 與 offload 兩條載入路徑都要），`configs/README.md` 補文件；
   paper configs 全部填上 Qwen3-30B-A3B-Base 當前的 HF commit hash。
3. **Spec-Bench pin**：`specbench.py` 的 URL 改成 pin commit 的 raw URL，
   加下載後 sha256 驗證（不符就報錯教使用者刪 cache）。
4. **`reproduce/` 目錄**：
   - `reproduce/env.sh`：集中 `REPO_ROOT`（自動偵測）、`HF_HOME`、module load、
     `SBATCH_ACCOUNT`/`SBATCH_PARTITION`（環境變數可覆寫，預設值放範例）。
   - `reproduce/submit.sh <config...>`：唯一的 sbatch 模板，source env.sh；
     取代「一實驗一 script」模式。**舊 scripts/ 不動、不刪**（歷史紀錄），
     目錄加 README 註明已被 reproduce/ 取代。
   - `reproduce/make_tables.py`：仿 `analyze_q15.py`，從 `output/` 聚合出
     論文每張主表的 mean±std，輸出 CSV + LaTeX 兩種格式。
   - `reproduce/README.md`：端到端流程——install.sh → prep offload export
     （警告：201GB disk、一次性）→ submit paper configs → make_tables.py；
     每步附預估時間與硬體需求（R9）。
5. **重跑主表**：全部用 pin 好的環境+revision 重跑 n≥3，確認結論
   （topm 贏 specmoe 的幅度）在凍結版本上成立。**論文引用的數字一律出自這批
   帶 provenance 的 run**，不要用 132 個舊 output 裡的歷史數字。

**驗收**：一個乾淨 shell 從 `reproduce/README.md` 走完全流程可產出主表；
`make_tables.py` 輸出的每個數字可回溯到帶 git hash 的 summary.json。

### P3 — Fork 文件與 artifact 打包（1 天）

1. **`moe_infinity/AUGSPEC_CHANGES.md`**：
   - 記 upstream repo URL + fork 時的 base version/commit
     （用 `git log` 考古 `4e53663 Before adding moe_infinity` 附近確認）。
   - 逐項列修改（對照 `git diff 4e53663..HEAD -- moe_infinity`）：
     EvictLayer、no_overload dispatch 路徑、DispatchBmm(engine_bmm)、
     merge_experts_local / get_resident_expert_weights / flush_cache、
     GetCapturedExpertOutputs（actsim 用）、profiling counters、kMaxTokens 128→2048。
     每項一句「做什麼＋為什麼」，這份直接餵論文 system section。
2. **LICENSE**：root 加 Apache-2.0；README 註明 moe_infinity 為 Apache-2.0 fork
   並引用其論文（`moe_infinity/CITATIONS.md` 已有）。
3. **去個人化**：檢查 repo 內 email / account 字串（`grep -rn "hhliu@\|MST114471"
   --include='*.sh' --include='*.py' --include='*.md'`），投稿若需匿名 artifact，
   `reproduce/` 與 README 不得含個人識別（舊 scripts/ 可整目錄排除在 artifact 外）。
4. **fresh-clone 驗證**：在乾淨目錄 clone → `install.sh` → pytest → 一個 smoke
   config sbatch 跑通。這是 artifact evaluation 的實際預演。

### P4 — 程式碼清理（1 天，最後做，全部「可不做」）

按 §2.7：診斷 dump 抽離到 `drafts/diagnostics.py`（純搬移）→ 跑 pytest +
一個 smoke run 驗等價；`_NamespaceFromDict` 加遮蔽警告註解；
adapters `__init__` 標注 mixtral/gptoss untested；configs 根目錄的歷史 YAML
移入 `configs/archive/`（**先 grep scripts/ 確認無引用再搬，或干脆只搬
確定沒被引用的**）。每步一個 commit。

---

## 5. 給後續 session 的守則（Opus 以下模型，務必遵守）

1. **依 P0→P4 順序做，不要跳**。P1 的測試是你們自己的安全網。
2. **不要做本文件沒列的重構**。特別是：不要動 C++（除非有可重現的 bug + 先問用戶）、
   不要把 merge 改成 registry、不要重寫 cli.py/specbench.py、不要換掉
   env-override 機制。§2.7 標「可不做」的，猶豫就不做。
3. 任何核心改動前後：`pytest tests/unit` + 一個 smoke config（sbatch，不要在
   login node 跑 GPU）。等價驗證用「結構等價 + smoke」，**不要嘗試 bit-exact 比對**
   （offload 推論非確定性，PROJECT_GUIDE 已載明）。
4. 一個邏輯變更一個 commit，訊息寫清楚對應本計劃哪一項（如 `P2.3: pin Spec-Bench`）。
5. 完成任一項：更新本文件狀態 + 依 CLAUDE.md 規則同步 PROJECT_GUIDE.md。
6. SLURM 輪詢間隔 ≥30 秒（CLAUDE.md 規則 3）。
7. 論文數字只用 P2.5 之後帶 provenance 的 run；舊 output 僅供方向參考。

## 6. 狀態追蹤

| 項目 | 狀態 | 備註 |
|---|---|---|
| P0.1 commit 現況 + tag | ⬜ | |
| P0.2 summary.json provenance | ⬜ | |
| P0.3 requirements-lock | ⬜ | |
| P1 pytest tests/unit | ⬜ | |
| P2.1 configs/paper/ | ⬜ | |
| P2.2 model.revision | ⬜ | |
| P2.3 Spec-Bench pin + sha256 | ⬜ | |
| P2.4 reproduce/ | ⬜ | |
| P2.5 主表 n≥3 重跑 | ⬜ | |
| P3.1 AUGSPEC_CHANGES.md | ⬜ | |
| P3.2 LICENSE | ⬜ | |
| P3.3 去個人化 | ⬜ | |
| P3.4 fresh-clone 驗證 | ⬜ | |
| P4 清理 | ⬜ | 可選 |
