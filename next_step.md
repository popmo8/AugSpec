# `next_step.md` — Offloading-backed pipeline plan

Status of the repo: the GPU-only half of aug_spec is migrated and
green (smoke test `configs/_smoke.yaml` produces sensible
MAT / AccRate / TPS via spec-bench). This document plans the next
piece — adding an **offloading backend backed by `moe_infinity`** so
we can reproduce the SpecMoE paper's throughput / PCIe baselines and
run the thesis' merge-based draft in the regime where it actually
matters.

The work is laid out as **independent phases** with a sanity check
between each. Don't move on to the next phase until the current
phase's check is green.

---

## 1. Goal

Add a second backend that loads MoE models with expert offloading
(non-expert layers on GPU, experts on host memory) so the same
`aug_spec` codebase can:

1. Reproduce the four baselines the SpecMoE paper compares against —
   **MoE-OnDemand**, **MoE-Caching**, **SpecMoE**, plus our
   **merge-based** draft.
2. Measure both **acceptance-rate / MAT** (existing spec-bench
   harness) and **tokens-per-second + GB-transferred-over-PCIe**
   (new `aug_spec bench` subcommand + NVML profiler).
3. Keep adding new experiments as **YAML files only**, no new Python.

The existing 9 GPU-only configs and their entire code path are
preserved; offload is purely additive.

---

## 2. Architectural decisions

### 2.1 Backend is a property of the loader, not of the API

```yaml
model:
  id: mistralai/Mixtral-8x7B-v0.1
  backend: hf                          # default
  # OR
  backend: offload
  offload:
    path: /work/morrisliu07/cache/offload/mixtral
    device_memory_ratio: 0.15
    cache_policy: ondemand             # ondemand | caching
```

- `runtime/loader.py` adds `load_offload(...)` that goes through
  `moe_infinity.MoE(...)`.
- **Adapter / Draft / Controller do not know which backend was used.**
- Existing configs stay valid — default `backend: hf` keeps GPU-only
  behaviour.

**Memory model in offload mode (what `device_memory_ratio` actually
controls):** `device_memory_ratio` is a hard **expert VRAM budget cap**.
moe_infinity's archer engine reserves `device_memory_ratio × GPU_size`
bytes for the expert cache; whenever a needed expert is not in that
cache, it's lazily fetched H2D and an in-cache expert is evicted (by
LRU/`min(incache_visit_count)`). Experts past the budget reside on host
memory and stream in on demand. The router gate, attention layers, KV
cache, and our merged dense expert are **outside** this budget and live
permanently on GPU.

### 2.2 Adapter handles backend differences only on the routing path

```python
class MixtralAdapter:
    def _standard_routing(self, block, ...):
        if isinstance(block, SyncMixtralSparseMoeBlock):
            return self._route_offload(block, ...)
        return self._route_hf(block, ...)
```

- The **draft phase** (merged dense SwiGLU) is identical on both
  backends — it bypasses expert dispatch entirely. The merged expert
  is a regular fp32 tensor stored in `controller.draft_cache`,
  permanently resident on GPU; **it is NOT in archer's expert cache
  and is not subject to eviction**.
- The **target verify phase** is the only spot that diverges:
  HF loops `block.experts[i](state)`; offload calls
  `block.expert_executor.dispatch_local(...)`. The offload path is
  exactly the per-expert batched dispatch you want — for each expert
  with ≥1 routed token, load (if not cached), gather all tokens
  routed to that expert, run forward as one batch, then move to the
  next expert.
- One adapter class per family; an `isinstance` branch is acceptable
  for paper-readiness given the limited scope.

### 2.3 Draft strategies: which are offload-safe, which aren't

Draft strategies fall into two groups by how they read expert weights
at merge time.

| Draft | GPU backend | Offload backend | Why |
|---|---|---|---|
| `uniform`, `count`, `pruned_count`, `softmax` | ✅ | ⚠️ **uncontrolled fetches** | `build_weighted_avg` reads `block.experts[e].w*.weight` for every non-zero-weight expert. In offload, those reads are on placeholder tensors — either return garbage or trigger a sync fetch. Without explicit pinning, draft-side PCIe is unbounded. |
| `topm_count`, `prefill_count`, `prefill_topm_count` | ✅ | ✅ | Hard upper bound on number of experts merged (M ≤ count_top_k, or "only at prefill"). Safe to explicitly pin those M experts before merging. |
| `random_mask` | ✅ | ✅ | Single random expert per layer; trivially bounded. |
| `none` (new) | n/a | ✅ | no spec decoding; lets `aug_spec bench` reuse the same plumbing for MoE-OnDemand / MoE-Caching baselines |
| `specmoe` (new) | ✅ | ✅ | top-N pinned + L2-affinity substitute. Pure algorithm on GPU backend; on offload backend it additionally calls `archer_engine.replace_cache_candidates(...)` after each cycle |

**Recommended offload draft = `topm_count` (M = count_top_k) or
`prefill_count`.** These are the variants Phase A.5 already added
specifically for the offload regime. On offload, "merge-based" in our
paper claims refers to these, not to plain `count`.

`build_weighted_avg` itself needs a backend-aware variant — see §2.7.

SpecMoE is **not** offload-only: on `backend: hf` it acts as a
"limited routing" baseline that produces head-to-head acceptance
numbers against our merge-based draft without any offload
infrastructure.

### 2.4 `cache_policy` governs **target verify**, not the draft

The merge-based draft lives entirely on GPU (one merged expert per
layer; nothing in the archer cache). `cache_policy` only controls
what the target side does with its archer cache during verify:

| Value | Behaviour | Paper analogue |
|---|---|---|
| `ondemand` | small cache, archer-native LRU, no pre-pinning | MoE-OnDemand baseline; also the default for SpecMoE / merge-based, because SpecMoE pins via its draft hook and merge-based doesn't need cache at all |
| `caching` | moe_infinity's tracer prefetcher pre-pins the top ~10% hottest experts | MoE-Caching baseline |

`by_draft` is removed — the SpecMoE draft pins through
`replace_cache_candidates` whenever it wants, independently of
`cache_policy`.

### 2.5 Throughput / PCIe measurement is its own subcommand

| Command | Purpose | Output |
|---|---|---|
| `aug_spec run --config X.yaml` | spec decoding (existing) | MAT, AccRate, per-question CSV, `expert_weights_history.json` |
| `aug_spec bench --config X.yaml` (new) | pure generation; no draft | tokens/sec, latency_ms, peak VRAM, GB transferred over PCIe |

MoE-OnDemand and MoE-Caching baselines pair with `aug_spec bench`
since they have no draft side at all. SpecMoE and merge-based use
`aug_spec run` (acceptance/MAT) and **also** `aug_spec bench` (to get
throughput numbers comparable with the baselines).

### 2.7 How the merged expert is materialised in offload mode

This is the technical crux. On GPU backend, [`adapter.build_weighted_avg(block, weights)`](src/aug_spec/adapters/mixtral.py#L35) just reads `block.experts[e].w1.weight` directly. In offload mode this **does not work** — moe_infinity replaces every offloaded `param.data` with a shape-`(1,)` zero placeholder ([model_offload.py:213-221](moe_infinity/moe_infinity/runtime/model_offload.py#L213)); the real weights are managed inside the archer C++ engine and addressed by `tensor_id`, never round-tripped back to the Python `nn.Parameter`. Even after a `fetch_experts_lock_cache(...)` pin, the Python-side `experts[e].w1.weight` still returns `(1,)` zeros — the pin only nominates tensors for archer's cache candidates, it does **not** update the Python `.data` pointer.

So the merge step cannot use the offloaded model's expert weights at all. The chosen design (**Option A**): **keep a second, CPU-resident copy of the model purely as the source for draft-side merging.**

#### Loader returns two objects

```python
hf_model, tokenizer, moe, cpu_source = load_offload(...)
```

- `hf_model = moe.model` — the offloaded `PreTrainedModel`; target verify runs through this.
- `cpu_source` — a regular `AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=bfloat16)`. Lives entirely on host RAM. Used **only** as a weight source by `build_weighted_avg_offload`. No forward passes ever run on `cpu_source`.

Host RAM cost: ~26 GB for Mixtral-8x7B, ~22 GB for GPT-OSS-20B, ~30 GB for Qwen3-30B-A3B. TWCC nodes have ≫ this.

#### `build_weighted_avg_offload` contract

```
adapter.build_weighted_avg_offload(cpu_block, weights, device) → {w1, w2, w3}
```

implemented as:

1. Pre-allocate fp32 accumulators on `device` (the GPU), shaped like one expert.
2. For each `(e_idx, w)` with `w != 0`: stream `cpu_block.experts[e_idx].w*.weight` H2D as a temporary bf16 tensor, `add_(... .float(), alpha=w)` into the fp32 accumulator, drop the temporary.
3. Cast the fp32 accumulators back to bf16 and return.

**Peak GPU memory during the merge: 1 fp32 accumulator (3 weights) + 1 bf16 transient (3 weights) ≈ 1.75 expert sizes.** Independent of how many experts are being merged — the streaming pattern keeps peak constant.

#### Properties of this design

- **No interaction with archer cache.** Source weights come from CPU RAM, accumulator is our own GPU tensor outside `device_memory_ratio` budget. The merge step does not perturb archer's LRU state at all.
- **Bounded one-way PCIe.** For `topm_count` with M experts merged per cycle: M × 352 MB H2D per cycle (Mixtral). For `prefill_count`: once per question. **CPU→GPU only, no D2H, no eviction.**
- **Decoupled cost model.** Draft-side PCIe and target-side PCIe (archer cache miss → fetch) are separate budgets. We can profile them independently.

This is a refinement of the original paper claim "draft-phase = 0 PCIe". The honest framing is:

> Draft-phase PCIe is **bounded, one-way, and decoupled from target-side cache state** — never causes a target-side cache miss; M × expert_size per refresh for `topm_count`, once-per-question for `prefill_count`.

For `prefill_count` / `prefill_topm_count`, steps 1–3 happen once per question. For `topm_count`, they happen once per verify cycle but only for M ≤ k experts. Either way **draft-phase PCIe is bounded and predictable**; the merged expert itself is one persistent fp32 tensor outside the archer cache.

`uniform`, `count`, `pruned_count`, `softmax` are **not** ported to this contract — they are GPU-only by design. The offload-friendly drafts are `topm_count`, `prefill_count`, `prefill_topm_count`. See §2.3.

### 2.6 Single GPU only

`scripts/run.sh` defaults to `--gpus-per-node=1`. SpecMoE paper Fig.
11 also uses one H100. Multi-GPU is left for future work.

---

## 3. Where `moe_infinity` actually enters the codebase

Only four files import or call into `moe_infinity`:

```
src/aug_spec/
├── runtime/loader.py          # MoE(...) entry point
├── adapters/{mixtral,gptoss,qwen3}.py
│                              # isinstance(block, Sync*Block) +
│                              # block.expert_executor.dispatch_local(...)
├── drafts/specmoe.py          # engine.replace_cache_candidates(...)  (no-op on hf)
└── controller.py              # surfaces archer_engine to the draft at install time
```

Everything else (`specbench.py`, `cli.py`, `profile.py`, the other
five drafts) has no `moe_infinity` imports. The total new code that
talks to `moe_infinity` is expected to be ≤ 100 lines.

---

## 4. Phase plan

Each phase has a sanity check; do not advance past a red one.

| # | Phase | Effort | Risk | Sanity check |
|---|---|---|---|---|
| 0 | **Pre-flight** — scratch script answers the four unknowns in §5 | 1h | 🔴 | All four questions in §5 answered yes/no with evidence |
| 1 | `loader.py` adds `load_offload(...)`; `RunConfig` parses `model.backend` + `model.offload`; existing configs unchanged | 2h | low | `python -c "from aug_spec.runtime.loader import load_offload"` works; existing `aug_spec run --config configs/_smoke.yaml` still passes |
| 2 | Adapter `_standard_routing` adds `_route_offload` branch in each of mixtral / gptoss / qwen3 | 3h | medium | Tiny unit test: a single target forward through one offloaded MoE block returns the same shape as HF |
| 3 | Smoke: write `configs/_smoke_offload.yaml` (mixtral_count + offload), run it. MAT / AccRate must be within bf16 noise of the GPU-only smoke | 1h | 🔴 | `summary.json["overall"]["mean_accept_tokens"]` matches `output/_smoke/summary.json` within ±0.1 |
| 3.5 | `drafts/specmoe.py` — pure-algorithm version on GPU backend. Produces head-to-head acceptance vs our merge-based draft for the paper table | 4h | low | A new config `configs/mixtral_specmoe.yaml` (backend hf) runs and produces sensible MAT numbers |
| 4 | `drafts/none.py` + `aug_spec bench` subcommand + `runtime/bench.py` (pure-generate loop, no spec decoding) | 2h | low | `aug_spec bench --config configs/_smoke.yaml` (with draft=none) runs and reports tokens/sec |
| 5 | `runtime/profile.py` — NVML PCIe RX/TX sampling thread, wired into `bench.py` | 2h | medium | Reported GB transferred is non-zero for offload backend, ~0 for hf backend |
| 6 | `drafts/specmoe.py` gains its `replace_cache_candidates` hook (no-op on hf, active on offload) | 2h | medium | PCIe GB / step on offload backend with `specmoe` draft is strictly lower than `count` draft (or matches the paper's claim) |
| 7 | Cache policy in loader — `ondemand` vs `caching`. `caching` enables the tracer-driven prefetcher; `ondemand` shrinks the cache and disables predictive prefetch | 3h | medium | MoE-Caching baseline (none + caching) shows lower mean PCIe / step than MoE-OnDemand (none + ondemand) |
| 8 | Four production configs under `configs/baselines/`, `configs/baselines_specmoe/`, `configs/ours/`; sbatch sweep | 2h | low | Each config completes; head-to-head table prints |
| 9 | Compare numbers vs (a) thesis_experiment's existing GPU-only acceptance, (b) SpecMoE paper Fig 11/12 | — | — | Reproduction note in PROGRESS.md |

---

## 5. Pre-flight verifications (Phase 0)

Resolve all four **before** writing Phase 1 code. A 50-line scratch
script in a notebook or `tests/sanity_offload.py` is enough.

1. **`adapter.iter_moe()` still works after `moe_infinity.MoE(...)`**
   — does `model.model.layers[i].block_sparse_moe` still resolve, and
   is the type now `SyncMixtralSparseMoeBlock`?

2. **CPU-resident weight source coexists with `MoE(...)`** — load both
   `moe = MoE(model_id, {...})` and
   `cpu_source = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=bfloat16)`
   in the same process. Verify (a) `cpu_source.model.layers[0].block_sparse_moe.experts[0].w1.weight`
   is a real shape-`(intermediate, hidden)` bf16 tensor on CPU,
   non-zero; (b) host RAM usage is ≈ full model size + offload
   workspace, comfortably below TWCC node memory; (c) `cpu_source`
   doesn't get intercepted by moe_infinity's empty-init hooks (it
   shouldn't — moe_infinity's monkey-patches only affect models loaded
   inside the `engine.init(...)` context manager).

3. **Confirm offloaded `experts[i].w1.weight` is shape-`(1,)`
   placeholder** (already evidence-confirmed via
   [model_offload.py:213-221](moe_infinity/moe_infinity/runtime/model_offload.py#L213),
   but re-prove it in the pre-flight script for the record). The
   point is to cement that direct reads of `moe.model.experts[e]`
   weights are useless for merging — `build_weighted_avg_offload`
   must source from `cpu_source` only.

4. **Single-GPU capacity for Mixtral non-expert + cache** — does
   Mixtral-8x7B at `device_memory_ratio=0.15` fit in one 80 GB H100?
   Decides whether the sbatch default `--gpus-per-node=1` is enough
   or we need `=2` for offload Mixtral runs.

---

## 6. Four production recipes

Each recipe is one YAML per model under the matching subfolder of
`configs/`. The four differ **only** in `draft.name` and
`model.offload.cache_policy`; same harness, same metrics.

| Recipe | `draft.name` | `cache_policy` | Subcommand |
|---|---|---|---|
| **MoE-OnDemand** baseline | `none` | `ondemand` | `aug_spec bench` |
| **MoE-Caching** baseline | `none` | `caching` | `aug_spec bench` |
| **SpecMoE** competitor | `specmoe` (args: `N=4`, `gamma=10` for NLLB-style / `gamma=5` for Mixtral, Llama-4) | `ondemand` | both — `run` for acceptance, `bench` for throughput |
| **Ours** — merge-based | `topm_count` (M = count_top_k) or `prefill_count` | `ondemand` | both |

**Note:** plain `count` / `pruned_count` are the GPU-backend versions of merge-based draft. They reproduce thesis_experiment numbers on `backend: hf` but are **not** offload-safe (see §2.3, §2.7). For offload comparisons, use the bounded-fetch variants.

---

## 7. Out of scope (explicit)

- **No C++ edits to `moe_infinity/core/`** in this round. If §5
  question 2 reveals that `replace_cache_candidates` is only
  advisory, revisit the decision at Phase 6 — until then,
  Python-only.
- **No removal of the GPU backend.** It stays as a reference point
  and as the host for the §3.5 SpecMoE-on-GPU head-to-head.
- **No SSD offloading** (SpecMoE §VI-D). Future work.
- **No multi-GPU offload.** Single GPU baseline only, matching the
  paper's primary table.
- **No batch-size sweep.** The paper's Fig. 11 batched numbers come
  later — first land batch-1 correctness, then bigger batches.

---

## 8. Decisions still pending

None blocking Phase 0–3. Phase 7 (cache policy) might want
fine-tuning once we see real PCIe numbers — keep it parameterisable
via `model.offload.device_memory_ratio` so we don't have to recompile
the comparison.
