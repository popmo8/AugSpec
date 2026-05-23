# `configs/` — one YAML per experiment

Every aug_spec experiment is a YAML file in this directory. The same
code path (`aug_spec run --config <path>`) serves every combination of
(model, draft strategy, knobs) — **adding a new experiment means
adding a YAML, not a Python file**.

## Quick reference

```yaml
model:
  id: <huggingface model id>            # REQUIRED
  dtype: bfloat16 | float16 | float32   # default: bfloat16
  device_map: auto | "cuda:0" | {…}     # default: auto
  trust_remote_code: true | false       # default: true
  adapter: mixtral | gptoss | qwen3_moe # default: auto-detect from config.model_type

draft:
  name: uniform | count | pruned_count | topm_count | prefill_count | prefill_topm_count | softmax | random_mask   # REQUIRED
  args:
    # strategy-specific — see "Draft strategies" below

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
| `dtype` | str | `bfloat16` | One of `bfloat16` / `float16` / `float32`. |
| `device_map` | str / dict | `auto` | Passed straight to `from_pretrained`. `auto` shards across visible GPUs. |
| `trust_remote_code` | bool | `true` | Required for models with custom code (DeepSeek, GPT-OSS, …). |
| `adapter` | str | (auto) | Override the auto-detected adapter. Useful if you fork a model and rename `config.model_type`. |

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
    record_history: false
```

| arg | type | default |
|---|---|---|
| `count_top_k` | int | `adapter.default_count_top_k(model)` |
| `M` | int | `count_top_k` |
| `record_history` | bool | `false` |

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

## Existing configs

| file | model | draft | notes |
|---|---|---|---|
| `_smoke.yaml` | Mixtral-8x7B | count | 1 q/cat × 32 tokens — end-to-end sanity check only |
| `mixtral_count.yaml` | Mixtral-8x7B | count | main thesis draft strategy on Mixtral |
| `mixtral_uniform.yaml` | Mixtral-8x7B | uniform | uniform baseline |
| `mixtral_softmax.yaml` | Mixtral-8x7B | softmax | softmax-weighted variant |
| `mixtral_random.yaml` | Mixtral-8x7B | random_mask | random expert baseline (`seed: 42`) |
| `mixtral_topm_count.yaml` | Mixtral-8x7B | topm_count | bounded fetch: keep top-M = count_top_k experts |
| `mixtral_prefill_count.yaml` | Mixtral-8x7B | prefill_count | build merged once on prefill, frozen for decoding |
| `mixtral_prefill_topm_count.yaml` | Mixtral-8x7B | prefill_topm_count | prefill-only + top-M cutoff |
| `gptoss_count.yaml` | GPT-OSS-20B | count | with `reasoning_effort: low` |
| `gptoss_pruned_count.yaml` | GPT-OSS-20B | pruned_count | long-tail pruning at 0.9 |
| `gptoss_topm_count.yaml` | GPT-OSS-20B | topm_count | bounded fetch variant |
| `gptoss_prefill_count.yaml` | GPT-OSS-20B | prefill_count | frozen-after-prefill variant |
| `gptoss_prefill_topm_count.yaml` | GPT-OSS-20B | prefill_topm_count | prefill-only + top-M cutoff |
| `qwen3_count.yaml` | Qwen3-30B-A3B-Base | count | |
| `qwen3_pruned_count.yaml` | Qwen3-30B-A3B-Base | pruned_count | pruning matters more here (128 experts × top-8) |
| `qwen3_topm_count.yaml` | Qwen3-30B-A3B-Base | topm_count | bounded fetch; M defaults to 8 |
| `qwen3_prefill_count.yaml` | Qwen3-30B-A3B-Base | prefill_count | frozen-after-prefill variant |
| `qwen3_prefill_topm_count.yaml` | Qwen3-30B-A3B-Base | prefill_topm_count | prefill-only + top-M cutoff |

## Adding a new config

```bash
cp configs/mixtral_count.yaml configs/mixtral_count_T5.yaml
# edit run.T: 3 → 5
aug_spec run --config configs/mixtral_count_T5.yaml
# or under SLURM:
sbatch scripts/run.sh configs/mixtral_count_T5.yaml
```

If you find yourself needing a knob that isn't in this doc:
1. Add the kwarg to the relevant draft / adapter class.
2. Surface it in `src/aug_spec/cli.py` (or just pass it via
   `draft.args` if it's draft-specific — the CLI splats those through).
3. Document it here.
