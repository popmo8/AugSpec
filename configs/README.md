# `configs/` — one YAML per experiment

Every aug_spec experiment is a YAML file in this directory. The same
code path (`aug_spec run --config <path>`) serves every combination of
(model, draft strategy, knobs) — **adding a new experiment means
adding a YAML, not a Python file**.

## Quick reference

```yaml
model:
  id: <huggingface model id>            # REQUIRED
  backend: hf | offload                 # default: hf
  dtype: bfloat16 | float16 | float32   # default: bfloat16
  device_map: auto | "cuda:0" | {…}     # default: auto  (hf backend)
  trust_remote_code: true | false       # default: true
  adapter: mixtral | gptoss | qwen3_moe # default: auto-detect from config.model_type
  offload:                              # backend: offload only — see "model.offload"
    path: <expert dir>                  # REQUIRED for offload
    vram_budget_ratio: 0.2              # usable VRAM / model VRAM (overrides device_memory_ratio)
    merge_offload: true                 # GPU resident-merge via archer dispatcher
    merge_during_verify: true           # per-layer merge during verify (vs after)
    no_overload: true                   # default: false  — C++ no-overload dispatch (A4)
    # merged_backend: engine_bmm        # engine_bmm (default) | dispatch | bmm  (A4)

draft:
  name: uniform | count | pruned_count | topm_count | prefill_count | prefill_topm_count | softmax | random_mask | specmoe   # REQUIRED
  args:
    # strategy-specific — see "Draft strategies" below
  # early_pin: 0                        # default: 0  — SpecMoE early-pin stage 0|1|2 (A4)

cluster:                                # averaged-draft family with K>1 — see "cluster"
  name: freq_slice                      # default: freq_slice  (only method in registry today)
  within_weight: freq                   # default: freq  — freq | uniform  (A4)

run:
  T: 3                                  # default: 3      — speculative tokens per cycle
  questions_per_cat: 10                 # default: 10     — Spec-Bench questions per category
  max_new_tokens: 512                   # default: 512    — generation budget per question
  seed: 0                               # default: 0      — RNG seed for question sampling
  warmup: true                          # default: true   — one tiny generate() before timed eval
  emit_tokens_csv: false                # default: false  — per-cycle CSV; ~100 MB / run when on
  spec_bench_cache: data/spec_bench     # default: <cwd>/data/spec_bench
  reasoning_effort: low                 # GPT-OSS only — injected into chat template

output:
  dir: output/<config-stem>             # default: output/<basename of this yaml, sans extension>
  label: <config-stem>                  # default: <basename>  — appears in stdout headers
```

The only required fields are `model.id` and `draft.name`. Everything
else has a sensible default. Defaults live in
[`src/aug_spec/cli.py`](../src/aug_spec/cli.py) — when in doubt, that's the
source of truth.

## `model`

| key | type | default | notes |
|---|---|---|---|
| `id` | str | — | HuggingFace repo id passed to `from_pretrained`. |
| `backend` | str | `hf` | `hf` (single GPU/sharded copy) or `offload` (moe_infinity expert offloading; needs `model.offload`). |
| `dtype` | str | `bfloat16` | One of `bfloat16` / `float16` / `float32`. |
| `device_map` | str / dict | `auto` | Passed straight to `from_pretrained` (hf backend). `auto` shards across visible GPUs. |
| `trust_remote_code` | bool | `true` | Required for models with custom code (DeepSeek, GPT-OSS, …). |
| `adapter` | str | (auto) | Override the auto-detected adapter. Useful if you fork a model and rename `config.model_type`. |

## `model.offload` (backend: offload only)

Experts live on host RAM and stream into a VRAM budget; non-expert layers stay
on GPU. Only read when `model.backend: offload`.

| key | type | default | notes |
|---|---|---|---|
| `path` | str | — | **Required.** Directory of pre-extracted expert weights. |
| `vram_budget_ratio` | float | (unset) | Usable VRAM as a fraction of the full-model footprint (GPU-independent; thesis 0.2×). Overrides `device_memory_ratio` when set. |
| `device_memory_ratio` | float | `0.15` | Raw archer pool / GPU-size escape hatch; used only when `vram_budget_ratio` is unset. |
| `vram_guard` | bool | `true` | Per-cycle warn when VRAM exceeds budget. |
| `merge_offload` | bool | `false` | GPU resident-merge via the archer dispatcher (alias: `cpp_merge`). |
| `merge_during_verify` | bool | `false` | Merge each layer *during* verify (experts still resident → 0 re-fetch) vs an after-verify refresh. |
| `flush_on_draft_end` | bool | `false` | Phase-exclusive flush (archer@draft-start, merged@draft-end). |
| `merge_overlap` | bool | `false` | Merge on a side stream, overlap with next-layer fetch. |
| `no_overload` | bool | `false` | C++ no-overload dispatch for batch>1 cache-full fetches. **A4** — was `AUG_NO_OVERLOAD`. |
| `merged_backend` | str | `engine_bmm` | Merged-expert draft kernel: `engine_bmm` (C++ DispatchBmm) / `dispatch` (per-expert MoEMLP) / `bmm` (Python). **A4** — was `AUG_MERGED_BACKEND`. |

## `draft`

`draft.name` selects the strategy; `draft.args` is whatever kwargs that
class accepts.

### Strategies

#### `uniform` — fixed 1/n averaged expert

```yaml
draft:
  name: uniform
```

No args. Builds a single averaged expert per layer with equal weights,
once, on first draft-phase use. Cheap baseline; no target-side capture
and no per-cycle refresh.

#### `count` — count-weighted averaged expert

```yaml
draft:
  name: count
  args:
    count_top_k: 2          # optional — auto-defaults to model's num_experts_per_tok
    record_history: false   # default: false
```

| arg | type | default |
|---|---|---|
| `count_top_k` | int | `adapter.default_count_top_k(model)` (2 for Mixtral, 4 for GPT-OSS, 8 for Qwen3) |
| `record_history` | bool | `false` |

Per verify cycle, tallies how many tokens picked each expert in their
top-`count_top_k`. Tallies become weights for the next cycle's
averaged expert. `record_history: true` additionally dumps the raw
per-cycle per-layer per-expert tallies to
`expert_weights_history.json`.

#### `pruned_count` — count, then drop the long tail

```yaml
draft:
  name: pruned_count
  args:
    count_top_k: 4                   # same default rule as `count`
    cumulative_threshold: 0.9        # default: 0.9
    record_history: false
```

Same as `count`, but before each rebuild keeps only the smallest set
of experts whose cumulative normalised count reaches
`cumulative_threshold`; everything else is zeroed and the remainder
re-normalised. `cumulative_threshold: 1.0` ≡ plain `count`. Most
useful on imbalanced routers (GPT-OSS-20B, Qwen3-30B-A3B).

#### `topm_count` — count, then keep top-M

```yaml
draft:
  name: topm_count
  args:
    count_top_k: 2                   # same default rule as `count`
    M: 2                             # default: same value as count_top_k
    K: 16                            # default: 1  — merged experts kept per layer
    draft_top_k: 8                   # default: native top-k  — clusters the draft runs
    record_history: false
```

| arg | type | default |
|---|---|---|
| `count_top_k` | int | `adapter.default_count_top_k(model)` |
| `M` | int | `count_top_k` |
| `K` | int | `1` | number of cluster-merged experts cached per layer (mini-MoE). `K=1` = single dense expert. |
| `draft_top_k` | int | native top-k | how many of the `K` clusters the draft forward runs (only when `K>1`). |
| `record_history` | bool | `false` |

`K` / `draft_top_k` are shared by the whole averaged-draft family (`count`,
`softmax`, `prefill_*` …), not just `topm_count`. When `K>1`, the active
experts are partitioned into `K` clusters and each is merged separately — the
partition method and within-cluster weighting are set in the top-level
[`cluster`](#cluster) section.

Same as `count`, but keeps exactly the top-`M` experts by vote count
each cycle (others zeroed, remainder renormalised). Differs from
`pruned_count` in that the budget is a fixed integer, not a
cumulative-mass threshold — useful when you want a deterministic
upper bound on how many experts contribute to the merged expert.
Default `M = count_top_k` aligns with "merge only what the router
itself would have picked per token".

#### `prefill_count` — count once during prefill, freeze

```yaml
draft:
  name: prefill_count
  args:
    count_top_k: 2                   # same default rule as `count`
```

| arg | type | default |
|---|---|---|
| `count_top_k` | int | `adapter.default_count_top_k(model)` |

Captures the target router's count distribution **only during the
prefill target forward**, builds the merged expert once, then reuses
it unchanged for every speculative-decoding cycle of that question.
Most offload-friendly merge variant: only one fetch episode (the
prefill's expert reads) per question. Does not support `record_history`
since there is only one state per question.

#### `prefill_topm_count` — prefill-only, then keep top-M

```yaml
draft:
  name: prefill_topm_count
  args:
    count_top_k: 2                   # same default rule as `count`
    M: 2                             # default: same value as count_top_k
```

| arg | type | default |
|---|---|---|
| `count_top_k` | int | `adapter.default_count_top_k(model)` |
| `M` | int | `count_top_k` |

Combines `prefill_count` and `topm_count`: capture once at prefill,
keep only the top-M experts by count, build the merged expert, freeze
for the rest of the question. Strongest offload assumption — only M
experts ever read at `build_weighted_avg` time, and only once per
question.

#### `softmax` — softmax-sum weighted averaged expert

```yaml
draft:
  name: softmax
  args:
    record_history: false
```

Like `count` but uses summed softmax mass per expert (continuous
signal: "how often AND how strongly was each expert picked"). Note
that on offload-based backends this draft strategy can in principle
require all experts in GPU at build time (any expert with non-zero
softmax mass gets pulled in) — fine on full-GPU backend, see
[../PROGRESS.md](../PROGRESS.md) for the offloading caveat.

#### `random_mask` — single random expert per layer per cycle (baseline)

```yaml
draft:
  name: random_mask
  args:
    seed: 42               # REQUIRED — different seeds yield different sweeps
    num_experts: 8         # optional — auto-defaults to adapter.num_experts(first block)
```

Picks one uniformly random expert per layer at the start of each
cycle. Sweep `seed` across 42 / 123 / 456 etc. for a stable
randomness baseline.

## `cluster`

Only relevant to the averaged-draft family with `draft.args.K > 1`: it sets how
the active experts are partitioned into the `K` clusters and how each cluster is
merged. Ignored when `K=1` (single dense expert) or for non-averaged drafts.

| key | type | default | notes |
|---|---|---|---|
| `name` | str | `freq_slice` | `ClusterMethod` registry key. Today only `freq_slice` (sort active experts by frequency, cut into `K` contiguous slices). |
| `within_weight` | str | `freq` | Within-cluster merge weighting: `freq` (∝ activation count) or `uniform` (1/\|group\|). **A4** — was `AUG_CLUSTER_UNIFORM`. The partition and cross-cluster mass stay frequency-based either way; only the within-cluster combine changes. |

## `run`

| key | type | default | notes |
|---|---|---|---|
| `T` | int | `3` | Speculative draft tokens per cycle. Held fixed (HF's heuristic ± schedule is disabled). |
| `questions_per_cat` | int | `10` | Per-category sample size from Spec-Bench. |
| `max_new_tokens` | int | `512` | Per-question generation budget. |
| `seed` | int | `0` | RNG for per-category sampling. |
| `warmup` | bool | `true` | Run one tiny `generate()` before timed eval to amortize compile. Disable in smoke tests. |
| `emit_tokens_csv` | bool | `false` | Per-cycle per-position dump. Useful for offline analysis; ~100 MB / full run. |
| `spec_bench_cache` | str | `<cwd>/data/spec_bench` | Override where `question.jsonl` is downloaded / loaded. |
| `reasoning_effort` | str | (none) | **GPT-OSS only** — injected into the chat template by `GptOssAdapter.post_load`. |

## `output`

| key | type | default | notes |
|---|---|---|---|
| `dir` | str | `output/<config-stem>` | Where summary CSV/JSON go. Relative paths are resolved against the cwd `aug_spec` is invoked from. |
| `label` | str | `<config-stem>` | Free-text tag printed in stdout headers and final table. |

## Environment overrides

The goal is "one YAML fully describes one run". The knobs below have YAML
fields (above); their `AUG_*` env vars remain only as **overrides** — when set,
the env value wins over the YAML. Handy for a one-off A/B without editing the
config.

| env var | overrides | values |
|---|---|---|
| `AUG_NO_OVERLOAD` | `model.offload.no_overload` | set (any value) = on |
| `AUG_MERGED_BACKEND` | `model.offload.merged_backend` | `engine_bmm` / `dispatch` / `bmm` |
| `AUG_EARLY_PIN` | `draft.early_pin` | `0` / `1` / `2` |
| `AUG_CLUSTER_UNIFORM` | `cluster.within_weight` | set (any value) = force `uniform` |

Diagnostics / ops are **env-only** (deliberately not in YAML — they are
analysis side-channels, not part of a run's definition):

| env var | effect |
|---|---|
| `AUG_PROFILE` | set = print the engine's per-cycle timing breakdown at the end. |
| `AUG_DUMP_CLUSTER_WEIGHTS=<path>` | append per-cluster within-cluster weights as JSONL. |
| `AUG_DUMP_ACTIVE_SET=<path>` | append the per-(layer,question,cycle) selected expert set as JSONL. |

## Existing configs

The work is now Qwen3-30B-A3B-Base centric (the Mixtral / GPT-OSS configs were
retired). A few canonical entry points:

| file | draft | notes |
|---|---|---|
| `qwen3_count.yaml` | count | plain count-weighted merge baseline (hf backend) |
| `qwen3_pruned_count.yaml` | pruned_count | long-tail pruning (matters at 128 experts × top-8) |
| `qwen3_topm_count.yaml` | topm_count | bounded fetch; M defaults to 8 |
| `qwen3_topm_count_k16.yaml` | topm_count | + `K=16` cluster mini-MoE |
| `qwen3_prefill_topm_count.yaml` | prefill_topm_count | prefill-only + top-M cutoff |
| `q5_512_tm_on.yaml` | topm_count | **offload + K=16 cluster reference**; self-contained A4 knobs (`no_overload`, `cluster`) |
| `q5_512_tm_unif.yaml` | topm_count | as `q5_512_tm_on` but `cluster.within_weight: uniform` (ablation) |
| `base_specmoe_offload.yaml` / `base_topm_offload.yaml` | specmoe / topm_count | offload baselines |

Beyond these, most files are experiment-specific families — `cmp_*`
(backend/kernel A/B), `exp_*` / `p1_*` / `p4_*` (ablations), `prof_*`
(profiling), `q5_*` / `q1_*` (the merged-expert clustering study). Each is one
self-describing YAML; open it and read the header comment.

## Adding a new config

```bash
cp configs/qwen3_topm_count.yaml configs/qwen3_topm_count_T5.yaml
# edit run.T: 3 → 5
aug_spec run --config configs/qwen3_topm_count_T5.yaml
# or under SLURM:
sbatch scripts/run.sh configs/qwen3_topm_count_T5.yaml
```

If you find yourself needing a knob that isn't in this doc:
1. Add the kwarg to the relevant draft / adapter class.
2. Surface it in `src/aug_spec/cli.py` (or just pass it via
   `draft.args` if it's draft-specific — the CLI splats those through).
3. Document it here.
