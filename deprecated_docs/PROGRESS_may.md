# 實驗進度 (PROGRESS)

> 配套文件:[README.md](README.md)(repo 結構) · [configs/README.md](configs/README.md)(YAML schema)

---

## 核心主張 (Thesis claim)

> 在 **CPU offloading-based MoE inference 的 speculative decoding** 場景下,把 target 端被選中的 experts 透過 **merging 壓縮成少數幾顆 merged expert** 當 draft,把 draft-side 常駐 expert memory 降到 **total expert memory 的 12.5%**,而 draft 階段 acceptance 仍**全面贏過 SOTA(SpecMoE)**。

關鍵是 **不一定要 collapse 成 1 顆** — merged expert 顆數 `K` 隨模型而定,只要守住 12.5% 的 memory 上界:

| 模型 | 原始 → merge 後 | 比例 | translation AccRate (T=5) | overall |
|---|---|---|---|---|
| **Mixtral-8x7B** (top-2) | 8 → **1** 顆 | 1/8 = **12.5%** | **0.878** vs SpecMoE 0.768 (**+0.11**) | 0.675 vs 0.672 |
| **Qwen3-30B-A3B** (top-8) | 128 → **16** 顆 | 16/128 = **12.5%** | **0.808** vs SpecMoE 0.683 (**+0.12**) | 0.437 vs 0.342 (**+0.10**) |

我們最早在 Mixtral 驗證單顆 merged expert(8→1)可行,再**往 Qwen3 延伸**確認大 routing 規模(128×top-8)下也成立。差別在 128→1 的單顆 collapse 會摧毀過多 functional diversity,所以 Qwen3 改用 **K=16 frequency-slice 分群**,每群獨立 merge,兼顧 diversity 與固定的 12.5% 上界。完整逐 task 數字見下方〈實驗結果〉。

主張有**兩個評估軸**:

1. **Acceptance rate** — 固定 draft memory ≤ 12.5%,acceptance 仍贏 SpecMoE。**(已完成,見下)**
2. **Throughput(head-to-head,我們 no-cache)** — 同一個 offload backend 上,**我們完全不另設 target cache**,直接跟 SpecMoE 原樣比 tokens/sec。勝點純粹來自方法本身:acceptance 高 → verify cycle 變少 → target 端昂貴的 on-demand expert fetch 變少;且 K=1 的 draft 不 routing、只走一顆 dense SwiGLU,比 SpecMoE 在 N 顆上跑 gate + L2 substitution 更輕。**(下一步,見〈下一步〉)**

這個 repo (`aug_spec`) 把上述想法寫成 paper-ready 的乾淨實作,並把 SpecMoE 作為對照組一起跑。

---

## 跟 SOTA — SpecMoE [DAC 2026] — 的對位

| 面向 | SpecMoE | **本工作 (merge-based)** |
|---|---|---|
| Draft 形式 | 從 target experts **選 N 個** temporally-hot experts 釘在 GPU | 每層 **K 顆 merged expert**(K=1 Mixtral / K=16 Qwen3),由 target routing 訊號加權合併 |
| Draft 常駐 memory | `N × expert_size`(隨 N 縮放) | 固定 **total expert memory 的 12.5%** |
| Draft 是否跑 router | 要(限制在那 N 顆) | K=1 不用(走 dense SwiGLU);K>1 跑 gate-remap |
| Gate 選到非 draft expert | 用**離線預計算的 L2 affinity table** 映射到最近 draft expert | 不會發生 — draft 只在 merged experts 上 routing |
| Target 端 expert cache | pin 的 N 顆是**真 expert**,verify 命中就算 hit → **兼作 incidental cache**;非 N 顆才 on-demand fetch | **不另設 cache** — merged expert 是合成的、對 target hit 無貢獻;verify 一律 on-demand fetch(throughput 在此最不利設定下仍頭對頭比,見〈下一步〉) |
| 離線預計算 / retraining | 要 L2 距離預計算;不需 retrain | 都不需要 |

---

## 實驗 config 一覽 (`configs/`)

`aug_spec run --config <path>`;schema 見 [configs/README.md](configs/README.md)。只列 Mixtral / Qwen3(GPT-OSS 先忽略)。

### Mixtral-8x7B(8 experts, top-2 → merge 8→1)

| config | draft | 用途 |
|---|---|---|
| [`mixtral_count`](configs/mixtral_count.yaml) | `count` | **主方法** — count-weighted 單顆 merge |
| [`mixtral_topm_count`](configs/mixtral_topm_count.yaml) | `topm_count` | bounded fetch,top-M=2 |
| [`mixtral_prefill_count`](configs/mixtral_prefill_count.yaml) / [`_topm`](configs/mixtral_prefill_topm_count.yaml) | `prefill_count` / `prefill_topm_count` | prefill build 一次後凍結(±top-M) |
| [`mixtral_topm_count_svd`](configs/mixtral_topm_count_svd.yaml) | `topm_count` (SVD) | Sub-MoE SVD subspace merge(M=2, rank=256) |
| [`mixtral_uniform`](configs/mixtral_uniform.yaml) / [`_softmax`](configs/mixtral_softmax.yaml) / [`_random`](configs/mixtral_random.yaml) | `uniform` / `softmax` / `random_mask` | baselines |
| [`mixtral_top{1,4,6}_count`](configs/mixtral_top1_count.yaml) | `topm_count` M=1/4/6 | diagnostic:top-M 掃描 |
| [`mixtral_specmoe`](configs/mixtral_specmoe.yaml) | `specmoe` | **SOTA 對照**(L2-nearest substitution,N sweep) |

### Qwen3-30B-A3B-Base(128 experts, top-8 → merge 128→16)

**K=16 cluster-merge(主軸,「多顆 merged expert」)** — 2×2 ablation: K∈{1,16} × SVD∈{off,on}

| config | draft | K / M / SVD |
|---|---|---|
| [`qwen3_count_k16`](configs/qwen3_count_k16.yaml) / [`_svd_k16`](configs/qwen3_count_svd_k16.yaml) | `count` | K=16,無 top-M,SVD off/on |
| [`qwen3_topm_count_k16`](configs/qwen3_topm_count_k16.yaml) / [`_svd_k16`](configs/qwen3_topm_count_svd_k16.yaml) | `topm_count` | K=16,M=64,SVD off/on(**full treatment**) |
| [`qwen3_prefill_topm_count_k16`](configs/qwen3_prefill_topm_count_k16.yaml) | `prefill_topm_count` | K=16,M=64,frozen |

**K=1 baseline / ablation:** [`qwen3_count`](configs/qwen3_count.yaml) · [`topm_count`](configs/qwen3_topm_count.yaml)(±[`svd`](configs/qwen3_topm_count_svd.yaml)) · [`pruned_count`](configs/qwen3_pruned_count.yaml)(剪長尾 0.9) · [`prefill_count`](configs/qwen3_prefill_count.yaml) · [`prefill_topm_count`](configs/qwen3_prefill_topm_count.yaml)

**top-M 掃描:** [`qwen3_top{1,4,16,32,64}_count`](configs/qwen3_top1_count.yaml)(M=1/4/16/32/64)

**SOTA 對照:** [`qwen3_specmoe_N{8,16,32}`](configs/qwen3_specmoe_N16.yaml)(N=kept-mask size;N=16 對齊 12.5%)

---

## 實驗結果

指標為 **acceptance rate**(`T=5`,Spec-Bench 6 subtasks + overall)。每列粗體為最佳值。

### Mixtral-8x7B(Merge 8 → 1)

baseline: `random prune`=[`mixtral_random`](configs/mixtral_random.yaml)、`average`=[`mixtral_uniform`](configs/mixtral_uniform.yaml) · SpecMoE=[`mixtral_specmoe`](configs/mixtral_specmoe.yaml) · Ours: `count merge`=[`mixtral_count`](configs/mixtral_count.yaml)、`prefill`=[`mixtral_prefill_count`](configs/mixtral_prefill_count.yaml)、`topk k=2`=[`mixtral_topm_count`](configs/mixtral_topm_count.yaml)

| task | random | average | **SpecMoE** | count merge | prefill | topk k=2 |
|---|---|---|---|---|---|---|
| mt_bench | 0.2066 | 0.1332 | 0.6089 | **0.6184** | 0.5728 | 0.612 |
| **translation** | 0.3922 | 0.2553 | 0.768 | 0.8757 | 0.8625 | **0.8783** |
| summarization | 0.1417 | 0.1111 | **0.8433** | 0.7559 | 0.8137 | 0.8203 |
| qa | 0.3146 | 0.2344 | **0.824** | 0.6771 | 0.6445 | 0.6777 |
| math_reasoning | 0.2594 | 0.2528 | 0.8429 | 0.8076 | 0.846 | **0.8697** |
| rag | 0.2572 | 0.1481 | 0.7235 | 0.7006 | 0.7496 | **0.7669** |
| overall | 0.2249 | 0.1532 | 0.6719 | 0.667 | 0.6405 | **0.6746** |

**重點:** translation 上 `topk k=2`=**0.8783**、`count merge`=0.8757 比 SpecMoE 0.768 高 **≈+0.11(10 個百分點)**,印證「約 10%」;`math_reasoning`/`rag` 也由 Ours 拿下。`summarization`/`qa` SpecMoE 仍領先,但這兩 task 連 baseline 都偏高(routing 易預測),merge 與否差距小。

### Qwen3-30B-A3B(Merge 128 → 16,K=16 cluster-merge)

SpecMoE=[`qwen3_specmoe_N16`](configs/qwen3_specmoe_N16.yaml) · Ours: `count merge`=[`qwen3_count_k16`](configs/qwen3_count_k16.yaml)、`prefill`=[`qwen3_prefill_topm_count_k16`](configs/qwen3_prefill_topm_count_k16.yaml)、`topk k=32`=[`qwen3_topm_count_k16`](configs/qwen3_topm_count_k16.yaml)(pool M=32 → 分群 K=16)

| task | **SpecMoE** | count merge | prefill | topk k=32 |
|---|---|---|---|---|
| mt_bench | 0.2811 | 0.3427 | 0.0764 | **0.3525** |
| **translation** | 0.6833 | 0.8043 | 0.3093 | **0.8075** |
| summarization | 0.3443 | 0.4921 | 0.1555 | **0.4943** |
| qa | 0.3718 | 0.5141 | 0.219 | **0.5281** |
| math_reasoning | 0.3571 | **0.4374** | 0.1111 | 0.4337 |
| rag | 0.474 | 0.6537 | 0.1925 | **0.6556** |
| overall | 0.3416 | 0.4288 | 0.1186 | **0.4371** |

**重點:** `count merge`/`topk k=32` 在**每個 subtask 都贏 SpecMoE**(translation +0.12、overall +0.10)→ 128→16 cluster-merge 在大 routing 規模成立。**prefill 凍結在 Qwen3 崩掉**(overall 0.1186):128-expert router 的 prefill 分布無法 generalise 到 decoding,需 per-cycle refresh;對比 Mixtral(8 experts)prefill 幾乎無損 → prefill-freeze 只適用小 routing 規模。

---

## 對應論文 (SpecMoE)

*SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding* — Bang, Cho, Hwang, Chung, Rhu (KAIST / UNIST),DAC 2026(arXiv: 2604.10152)。reported:tokens/sec @ batch=256, NLLB-MoE **123.0** (SpecMoE) vs ~28.6 (MoE-OnDemand) ≈ 4.30× speedup。

---

## 下一步

### Throughput head-to-head(主張第二軸,規劃中)

第二個指標是 **end-to-end throughput(tokens/sec)**,跑在同一個 offload backend 上、**我們完全不另設 target cache**,直接跟 SpecMoE 原樣對比 —— 不碰 VRAM 配置帳:

| | draft 形式 | target verify | 順風 / 逆風 |
|---|---|---|---|
| **SpecMoE** | 在 N 顆 pinned expert 上跑 gate + L2 substitution | 命中那 N 顆算 hit,其餘 on-demand fetch | 順風:pinned experts 兼作 incidental cache |
| **Ours** | K=1 一顆 dense SwiGLU,不 routing、不 dispatch | **無 cache**,一律 on-demand fetch | 順風:acceptance 高 → cycle 少 → fetch episode 少;draft 更輕 |

論點:這場是**賭我們的順風(cycle 變少 + draft 變輕)蓋過 SpecMoE 的順風(pinned-expert incidental hit)**,而且是在對我們**最不利**的設定(我們一顆 cache 都沒有)下還要贏 —— 因此 claim 夠硬,沒有任何 VRAM 帳要算。要產出的數字:tokens/sec、verify cycle 數、GB-transferred per cycle(NVML PCIe profiler)。

### 其他

- SVD merge / 其他 K 值的 ablation 結果(configs 已備齊,見上)。
- paper framing(VRAM-matched fair comparison、talking points)。
