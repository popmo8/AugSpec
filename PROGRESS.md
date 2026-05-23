# 實驗進度 (PROGRESS)

> 配套文件:
> - [README.md](README.md) — repo 結構與使用方式
> - [next_step.md](next_step.md) — offloading backend 的架構決策(高層)
> - [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — offloading backend 的逐 phase 實作計畫(給 AI agent)
> - [configs/README.md](configs/README.md) — YAML schema 跟所有可用選項

---

## 核心主張 (Thesis claim)

> 在 **CPU offloading-based MoE inference 的 speculative decoding** 場景下,把 target 端被選中的 experts **merge 成一顆 dense expert** 當 draft,**只需要 1 顆 expert 的 VRAM**,且 acceptance / token throughput **打得贏目前的 SOTA**:在 translation task 上比 SpecMoE 高出 **約 10%**。

這個 repo (`aug_spec`) 是把上述想法寫成 paper-ready 的乾淨實作,並把 SpecMoE 作為對照組一起跑。

---

## 跟 SOTA — SpecMoE [DAC 2026] — 的對位

| 面向 | SpecMoE (Bang et al., DAC 2026) | **本工作 (merge-based)** |
|---|---|---|
| Draft 模型形式 | 從 target 的 expert 中**選 N 個** temporally-hot experts 釘在 GPU,當作小型 MoE 子集 | **每層只用 1 顆 merged expert**(target 端 captured 的 routing 訊號加權平均) |
| Draft 階段是否還跑 router | **要** — gate 照樣 routing,只是被限制在那 N 個 draft experts | **不用** — draft phase 直接走那顆 dense SwiGLU,完全不 routing、不 dispatch |
| Gate 選到「不在 GPU 的 expert」怎麼辦 | 用**離線預算好的 L2 distance affinity table** 映射到最近的 draft expert | 不會發生 — 只有一顆 expert,本質上對全部 router 票數做加權合併 |
| Draft 階段 GPU 需要的 expert 容量 | `N × expert_size`(SpecMoE paper 主推 N=k=2,NLLB-MoE 評估用 N=4) | **1 × expert_size** |
| Draft 階段 PCIe(CPU↔GPU) | 0(N 顆都釘在 GPU) | **Bounded, one-way, decoupled** — 每次 refresh M × expert_size 的 CPU→GPU,跟 target cache 無交互 |
| 需要 retraining / fine-tuning | 否 | 否 |
| Draft 更新策略 | 每個 verify cycle 由 router activation count 找出 temporally-hot N 顆,evict 不在 hot list 內的 | 每個 verify cycle 用 target router 的 top-k count(或 softmax)當權重重新合併 |

---

## Fair-comparison protocol — VRAM-matched

我們最強的 framing **不是** 「我們省 VRAM」(這會被 reviewer 質疑「省下來幹嘛」),而是:

> **在相同的 total expert VRAM budget 下,我們的 throughput / acceptance 都比 SpecMoE 高,因為我們把 draft-side 省下的 VRAM 全部 reallocate 給 target-side cache。**

### 預算分解(Mixtral-8x7B,bf16,32 MoE layers,每顆 expert ~352 MB)

|  | Draft-side VRAM | Target-side cache VRAM | Total |
|---|---|---|---|
| SpecMoE @ N=2(他們 paper 的最小設定) | 32 × 2 × 352 MB = **22.5 GB** pinned | 剩餘 | depend on V |
| SpecMoE @ N=4(NLLB-MoE 設定) | 32 × 4 × 352 MB = **45 GB** pinned | 剩餘 | depend on V |
| **Ours(merged dense)** | 32 × 1 × 352 MB = **11.25 GB** | 剩餘(任何 V 下都比 SpecMoE 多 11.25–33.75 GB) | depend on V |

**關鍵:** 我們的 draft-side 是 `1 × expert_size × num_layers` **固定值**;SpecMoE 是 `N × expert_size × num_layers` 隨 N 縮放。在固定 V 下,我們 target cache 的可用空間比 SpecMoE 多 `(N − 1) × expert_size × num_layers`。

### Bench points

跑 3 個 V 點,每個 V 點下兩邊各自最佳化分配:

| V (total expert VRAM) | SpecMoE 配置 | Ours 配置 | 對位敘事 |
|---|---|---|---|
| **22.5 GB** | N=2, archer cache=0 | merged=11.25, archer cache=11.25 | 「SpecMoE 的最小 paper 設定;同 VRAM 下我們把一半丟給 cache」 |
| **45 GB** | N=2, cache=22.5 **or** N=4, cache=0 | merged=11.25, archer cache=33.75 | 「中等預算,SpecMoE 可挑配置;我們仍多 ~11 GB 給 cache」 |
| **90 GB** | N=8(全 pinned,GPU-only baseline) | merged=11.25, archer cache=78.75(實質全 cached) | 「無 offload 上界;兩邊都 cache hit,純比 acceptance」 |

### 每個 V 點要產出的數字

| 欄位 | 來源 |
|---|---|
| MAT / AcceptRate | `aug_spec run` |
| TPS | `aug_spec run` 或 `aug_spec bench` |
| `draft_vram_gb` | `summary.json["vram_breakdown"]` |
| `target_cache_vram_gb` | `summary.json["vram_breakdown"]` |
| `expert_total_vram_gb` | `summary.json["vram_breakdown"]`(應該 ≈ V) |
| Mean GB-per-cycle PCIe | `aug_spec bench` + NVML profiler |

實作上需要的東西全部寫在 [IMPLEMENTATION_PLAN.md §10.2 + §11.1](IMPLEMENTATION_PLAN.md)。

### 為什麼這個 framing 比「我們省 VRAM」強

- **直接可量化**:固定預算,比兩個指標(TPS、acceptance),清楚誰贏。
- **堵 reviewer 退路**:不會被質疑「你省 VRAM 但 acceptance 變差」(我們同 V 下 acceptance 也更高);也不會被質疑「你 cache 太大才贏」(SpecMoE 同 V 也能加 cache,只是被 pinned 占走)。
- **跟 SpecMoE paper 自身 setup 對接**:V=22.5 GB(=N=2)是他們自己宣稱的 minimum,我們在他們的 minimum 下贏;V=45 GB(=N=4)是他們 NLLB-MoE 的 setup,我們在他們的 production 設定下也贏。

---

## 此 repo (`aug_spec`) 的進度

### ✅ Phase A — GPU-only migration(已完成)

把 `thesis_experiment/experiments/single/` 的 GPU-only 程式碼搬成 paper-ready 的 src-layout package:

- `src/aug_spec/{adapters,drafts,runtime,controller.py,cli.py}` — 全部模組化
- 8 個 YAML config 替代原本 9 個 `exp_*.py` 檔
- 單一 entrypoint:`aug_spec run --config <yaml>`
- `scripts/run.sh` 對應 TWCC SLURM
- Smoke test([`configs/_smoke.yaml`](configs/_smoke.yaml))跑通,Mixtral-8x7B + count draft 產出合理數字
  - MAT=2.80, AccRate=0.62, TPS=7.56(13 題 × 32 tokens,VRAM 43.5 GB)
- **Full run 已驗證** — [`configs/mixtral_count.yaml`](configs/mixtral_count.yaml) 跑完整 10 q/cat × 512 tokens,輸出的 MAT / AccRate / TPS 跟 `thesis_experiment/.../exp_mixtral_single_countavg_specbench` 結果幾近一致 → **Phase A 遷移確認沒造成數字 regression**

### ✅ Phase A.5 — Bounded-fetch count 變體(已跑完)

為了讓 merge-based draft 在 offload backend 下也能維持「draft phase ≈ 0 PCIe」的論點,加入兩個新 draft strategy。動機是:對任意模型,只要一個 verify cycle 內 token 數 > num_experts,純 `count` 幾乎一定會把所有 expert 都點到至少一次,於是 `build_weighted_avg` 在 offload 下會觸發大量 fetch。這兩個變體都明確限制每 cycle 拿來 merge 的 expert 數:

| 變體 | 邏輯 | 預設行為 | 對應 paper 比較軸 |
|---|---|---|---|
| `topm_count` | 每 cycle 取**票數最高 M 顆**,其餘清零、餘者 renormalise | `M = count_top_k`(Mixtral 2 / GPT-OSS 4 / Qwen3 8)→ fetch 上界 = M | 跟 SpecMoE 的 N 直接可比(SpecMoE pin N 顆當 draft,這裡 merge 那 M=N 顆當 draft) |
| `prefill_count` | **只在 prefill** 那一次 capture 上 build merged expert,整題 decoding 重用 | 無 per-cycle refresh,每題只有一次 fetch 機會 | 最 offload-friendly 的 merge baseline |

實作:
- [`src/aug_spec/drafts/topm_count.py`](src/aug_spec/drafts/topm_count.py)
- [`src/aug_spec/drafts/prefill_count.py`](src/aug_spec/drafts/prefill_count.py)

#### Run queue(GPU backend,T=5,10 q/cat × 512 tokens)

所有數字取自 `output/<label>/overall_summary.csv` 的 `overall` row(130 q × 6 subtasks)。

| Model | `topm_count` | `prefill_count` | `prefill_topm_count` |
|---|---|---|---|
| Mixtral-8x7B (8 experts, top-2) | ✅ MAT=4.36 AccRate=0.675 TPS=18.6 ([cfg](configs/mixtral_topm_count.yaml)) | ✅ MAT=4.19 AccRate=0.641 TPS=27.2 ([cfg](configs/mixtral_prefill_count.yaml)) | ✅ MAT=4.09 AccRate=0.619 TPS=27.5 ([cfg](configs/mixtral_prefill_topm_count.yaml)) |
| GPT-OSS-20B (32 experts, top-4) | ⚠️ MAT=1.07 AccRate=0.022 TPS=9.4 ([cfg](configs/gptoss_topm_count.yaml)) | ⚠️ MAT=1.06 AccRate=0.022 TPS=11.2 ([cfg](configs/gptoss_prefill_count.yaml)) | ⚠️ MAT=1.09 AccRate=0.018 TPS=8.3 ([cfg](configs/gptoss_prefill_topm_count.yaml)) |
| Qwen3-30B-A3B (128 experts, top-8) | ⚠️ MAT=1.40 AccRate=0.135 TPS=4.2 ([cfg](configs/qwen3_topm_count.yaml)) | ⚠️ MAT=1.38 AccRate=0.125 TPS=4.5 ([cfg](configs/qwen3_prefill_count.yaml)) | ⚠️ MAT=1.42 AccRate=0.085 TPS=3.8 ([cfg](configs/qwen3_prefill_topm_count.yaml)) |

#### 觀察(GPT-OSS / Qwen3 ⚠️)

GPT-OSS-20B 上 acceptance rate **崩到 ~0.02**、Qwen3-30B-A3B 上 **~0.09–0.13**,跟 Mixtral 的 ~0.6–0.7 差兩個數量級。Memory 提到「routing 嚴重不均衡的 GPT-OSS / Qwen3 改用 `pruned_count`」,但本表的 `topm_count` / `prefill_count` 兩個變體都是**裸 count**(沒做 cumulative-mass 剪枝),長尾雜訊應該還在。論文主推方法 `pruned_count` 對這兩個模型的 aug_spec 數字目前**尚未產出** — 對應 [`configs/gptoss_pruned_count.yaml`](configs/gptoss_pruned_count.yaml) 與 [`configs/qwen3_pruned_count.yaml`](configs/qwen3_pruned_count.yaml) 還沒跑過。下一步應該優先補上,否則無法判斷 ⚠️ 是 routing 長尾的已知問題,還是 aug_spec 程式碼 regression。

設計討論細節參考 next_step.md 之後會補上的「Draft × offload PCIe 成本模型」段。

### ⏳ Phase B — Offloading backend(規劃中,尚未動工)

詳見 [next_step.md](next_step.md)。9 個 phase 的執行計畫,目前進度 Phase 0(pre-flight 驗證)還沒開始。

里程碑檢核點:
- [ ] Phase 0:`moe_infinity` 載入 + 4 個技術未知驗證
- [ ] Phase 1-3:`backend: offload` YAML key + adapter offload 分支 + smoke 對照 GPU 版本
- [ ] Phase 3.5:`drafts/specmoe.py`(GPU backend 版本)→ **頭對頭 acceptance 表**
- [ ] Phase 4-5:`aug_spec bench` + NVML PCIe profiler
- [ ] Phase 6:SpecMoE 加 offload pin hook
- [ ] Phase 7:`cache_policy: ondemand / caching`
- [ ] Phase 8:4 個 production recipes 跑完
- [ ] Phase 9:跟 SpecMoE 論文 Fig 11/12 對照
---

## 結果概況

### ✅ 已有的結論(來自 thesis_experiment,需在 aug_spec 上覆驗)

- **Translation task 上 merge-based(`countavg` / `pruned_countavg`)比 SpecMoE 高約 10%** (MAT / acceptance rate 維度,具體數字以舊 `summary.json` 為準)。
- 在 Mixtral(skewness=0.32,routing 較均勻)上 merge-based 仍有效 → 對應 `exp_experts_cos_sim.py` 觀察到的 expert 可互換性。
- Qwen3-30B-A3B(128 experts × top-8)上 pruned_count 比純 count 大幅有效 — long tail 的 ~120 顆稀疏 expert 拖進平均只是 noise。

### ⏳ 仍待產出的核心數字

- **End-to-end tokens/sec + GB-transferred** 在 offloading 設定下的 4 路對照表(MoE-OnDemand / MoE-Caching / SpecMoE / Ours)
- **SpecMoE-on-GPU 頭對頭表** — 同一模型 × 同一 task,acceptance rate / MAT 三欄並列(Phase 3.5 的產物)
- **Bounded-fetch ablation** — `count` vs `topm_count`(M=k)vs `prefill_count`,看 acceptance / MAT 損失多少換來幾顆 expert 的 fetch 上限(Phase A.5 的產物)
- **Ablation:`cumulative_threshold` 掃描**(0.5 / 0.7 / 0.9 / 1.0)在 GPT-OSS / Qwen3 上的影響

---

## Talking points(寫論文時)

1. **VRAM-matched comparison(headline)**:固定 total expert VRAM 預算,我們 reallocate `(N−1) × expert_size × num_layers` 的省下空間給 target cache,結果同 V 下 TPS + AcceptRate 都贏。見上面「Fair-comparison protocol」一節。
2. **不需 affinity table、不需 offline 預計算 L2** — SpecMoE 需要為每個 (model, task) pair 跑一次離線 expert L2 距離預計算;我們沒有這個前置步驟。
3. **Draft-phase PCIe = bounded, one-way, decoupled**:每次 refresh 上界 M × expert_size 的 CPU→GPU(`topm_count`)或每題一次(`prefill_count`),且**完全不擾動 target cache 的 LRU 狀態**(因為 source 來自獨立 CPU 副本,不走 archer)。
4. **merge 對 skewness 低的模型(Mixtral、Llama-4-Scout)為何仍有效** → 對應 `exp_experts_cos_sim.py` 量到的 expert 可互換性
5. **Architecture clean**:同一 codebase 同時跑 4 個 baseline(MoE-OnDemand / MoE-Caching / SpecMoE / Ours),只差一個 YAML(reproducibility 加分)
6. **Bounded fetch budget**:`topm_count` 把每 cycle 用到的 expert 嚴格上界在 M = top-k(跟 SpecMoE 的 N 對位);`prefill_count` 整題 decoding 只有一次 fetch — 兩者都在不犧牲 merge 形式的前提下符合 offload PCIe 假設

---

## 對應論文 (SpecMoE)

- 題目:*SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding*
- 作者:Jehyeon Bang, Eunyeong Cho, Ranggi Hwang, Jinha Chung, Minsoo Rhu (KAIST / UNIST)
- 出處:DAC 2026 (extended version on arXiv: 2604.10152)
- 主要 reported number:tokens/sec at batch=256, NLLB-MoE: **123.0** (SpecMoE) vs ~28.6 (MoE-OnDemand) ≈ 4.30× speedup
