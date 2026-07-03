# IMPLEMENTATION_PLAN.md — Offload backend for `aug_spec`

> **Audience.** A future AI agent picking up the offload backend work
> on this repo. You do **not** need to have seen prior conversations.
> Read this top-to-bottom once, then start at Phase 0.
>
> **Companion docs:**
> - [next_step.md](next_step.md) — high-level architectural decisions
> - [PROGRESS.md](PROGRESS.md) — current state of experiments
> - [configs/README.md](configs/README.md) — YAML schema
>
> When this doc conflicts with next_step.md, **this doc wins** for
> implementation details; next_step.md wins for architectural intent.

---

## 1. What you are building

A second model-loading backend in `aug_spec` that uses
[`moe_infinity.MoE(...)`](moe_infinity/moe_infinity/entrypoints/big_modeling.py)
to expert-offload MoE checkpoints, so that **the same speculative-
decoding harness** (Spec-Bench, fixed-T, per-cycle telemetry) can run
in the offloaded regime where the paper's claims actually apply.

### 1.1 Behavioural invariants (non-negotiable)

These are what the user actually wants. Each invariant is the
acceptance test for the relevant code path.

1. **One model object for spec decoding.** Target and draft are the
   same HF `PreTrainedModel` instance (the `.model` attribute of the
   `MoE` wrapper). Shared-weights spec decoding.
2. **A second, CPU-resident copy of the model is loaded alongside**
   purely as the weight source for `build_weighted_avg_offload`.
   No forward passes ever run on this copy; it occupies host RAM only
   (~26 GB Mixtral bf16). See §6.
3. **Fixed expert VRAM budget for target side.** Only
   `device_memory_ratio × GPU_size` bytes of GPU memory are reserved
   for the archer expert cache (target verify). Experts beyond the
   budget live on host memory and stream in on demand.
4. **Lazy fetch on cache miss (target side).** When the router selects
   an expert that's not currently in the GPU cache, archer fetches it
   H2D and evicts the lowest-`incache_visit_count` expert to make room.
5. **Verify phase = per-expert batched dispatch.** For each expert
   with ≥1 routed token in the current forward: load (if not cached) →
   gather all hidden states routed to that expert → run forward in one
   batch → move to the next expert. This is what
   [`expert_executor.dispatch_local`](moe_infinity/moe_infinity/distributed/expert_executor.py#L32)
   already does; do not re-implement it.
6. **Draft phase has bounded, one-way, decoupled PCIe.** The merged
   dense expert is a regular fp32 tensor stored in
   [`controller.draft_cache`](src/aug_spec/controller.py#L48). It lives
   **outside** the archer cache, on GPU, permanently for the duration
   of a question. The draft-phase forward bypasses
   `block_sparse_moe.expert_executor` entirely. Building this merged
   expert streams M expert weights CPU→GPU from the CPU-resident copy
   (no D2H, no archer state perturbation). Not literally "0 PCIe"
   but **bounded** (M × expert_size per refresh) and **decoupled** from
   target-side cache state.
7. **Bounded-fetch merge.** The cost of *building* the merged expert
   (reading source expert weights and accumulating) must be predictable
   and bounded — that means only `topm_count`, `prefill_count`, and
   `prefill_topm_count` are valid offload draft strategies. Plain
   `count` / `pruned_count` / `softmax` are not.
8. **Phase transitions are free.** Switching between verify and draft
   is a Python flag flip on `controller.in_draft_phase`. No memory
   movement, no eviction, no load. Both the archer cache and the
   merged dense experts coexist in VRAM permanently throughout a run.
9. **GPU backend unchanged.** Every existing config under
   [`configs/`](configs/) must continue to produce numbers within bf16
   noise of its current `output/<label>/overall_summary.csv`.

If any of these break during your work, stop and diagnose before
moving forward.

---

## 2. Mental model: what changes vs what doesn't

| Component | GPU backend (today) | Offload backend (new) |
|---|---|---|
| Model load | `AutoModelForCausalLM.from_pretrained` → live weights on GPU | `moe_infinity.MoE(ckpt, {offload_path, device_memory_ratio})` → wrapper owns `.model` (HF PreTrainedModel) with expert weights backed by archer engine |
| Block type | e.g. `MixtralSparseMoeBlock` | `SyncMixtralSparseMoeBlock` (moe_infinity replacement, has `expert_executor` attr) |
| Adapter `iter_moe(model)` | `model.model.layers[i].block_sparse_moe` | same path (verify in pre-flight Q1) |
| Adapter `_standard_routing` (verify) | hand-rolled for-loop over experts | delegate to `block.expert_executor.dispatch_local` |
| Adapter `make_averaged_forward` (draft) | run merged fp32 SwiGLU on GPU | **unchanged** — merged expert is on GPU regardless of backend |
| Adapter `build_weighted_avg` | read `block.experts[e].w*.weight` directly | **new** `build_weighted_avg_offload`: pin via archer → read → unpin |
| Spec decoding driver | HF `target.generate(..., assistant_model=draft)` | **unchanged** — `moe.model` is still a `PreTrainedModel` |
| Per-generation init | nothing | call `moe._configure_hook(input_ids)` before each `target.generate(...)` to set up archer tracer |
| Controller / draft logic | unchanged | unchanged |
| `aug_spec run` CLI | unchanged | unchanged |

The total new Python code is expected to be **<200 lines**. No C++
edits to `moe_infinity/core/` unless pre-flight forces it (see §3).

---

## 3. Pre-flight (Phase 0) — DO THIS FIRST

Write a single scratch script `tests/sanity_offload.py` (or
inline in a notebook). It must answer **four** questions before any
Phase 1 code is written. Estimated: 1 hour.

### 3.1 Skeleton

```python
# tests/sanity_offload.py
import torch
from moe_infinity import MoE

CKPT = "mistralai/Mixtral-8x7B-v0.1"
OFFLOAD_DIR = "/work/morrisliu07/aug_spec/cache/offload/mixtral"

moe = MoE(CKPT, {
    "offload_path": OFFLOAD_DIR,
    "device_memory_ratio": 0.15,
})
hf_model = moe.model
engine = moe.engine

print("--- Q1: adapter path still resolves ---")
# Mirrors src/aug_spec/adapters/mixtral.py::MixtralAdapter.iter_moe
for i, layer in enumerate(hf_model.model.layers):
    block = getattr(layer, "block_sparse_moe", None)
    print(i, type(block).__name__,
          hasattr(block, "gate"), hasattr(block, "experts"),
          hasattr(block, "expert_executor"))
    if i >= 2:
        break

print("--- Q2: pin API ---")
# Find and exercise the pin call. Candidates:
#   engine.expert_prefetcher.fetch_experts_lock_cache(layer_id, ids)
#   engine.archer_engine.replace_cache_candidates(...)
# Print the available methods on engine.expert_prefetcher to discover.
print(dir(engine.expert_prefetcher))

print("--- Q3: weight readability before/after pin ---")
block0 = hf_model.model.layers[0].block_sparse_moe
print("before pin:", block0.experts[0].w1.weight.device,
      block0.experts[0].w1.weight.shape,
      block0.experts[0].w1.weight.abs().sum().item())
# Pin expert 0 of layer 0 via whatever Q2 revealed.
# engine.expert_prefetcher.fetch_experts_lock_cache(0, torch.tensor([[0]]))
print("after pin:", block0.experts[0].w1.weight.device,
      block0.experts[0].w1.weight.abs().sum().item())

print("--- Q4: VRAM fit ---")
# Run one tiny forward to populate things.
toks = torch.tensor([[1, 2, 3, 4, 5]], device="cuda:0")
moe._configure_hook(toks)
with torch.no_grad():
    _ = hf_model(toks)
print("peak VRAM:", torch.cuda.max_memory_allocated() / 1e9, "GB")
```

### 3.2 Acceptance criteria for Phase 0

| Q | Pass condition | If fails |
|---|---|---|
| Q1 | `block` is `SyncMixtralSparseMoeBlock`, has `.gate`, `.experts`, `.expert_executor` | The adapter `iter_moe` needs a fallback path |
| Q2 | At least one of `fetch_experts_lock_cache` / `replace_cache_candidates` exists and accepts a `(layer_id, tensor[ids])` signature | Document what *does* exist; choose the closest API; falling back to a no-op hint is acceptable but document it |
| Q3 | After pin, reading `experts[e].w1.weight` returns the same shape as bf16 GPU backend and is non-zero | If reads return zeros or the tensor stays on CPU/meta, Q2's pin API is wrong — try another |
| Q4 | `peak VRAM ≤ 50 GB` for Mixtral-8x7B at `device_memory_ratio=0.15` on H100 80GB | If >70GB, drop ratio to 0.10; if even that OOMs, single-GPU offload doesn't fit and the sbatch default needs `--gpus-per-node=2` |

Write the answers as a comment block at the top of
`tests/sanity_offload.py` and **commit them**. Subsequent phases
depend on them.

---

## 4. Phase 1 — `loader.load_offload` + YAML schema

**Files touched:** `src/aug_spec/runtime/loader.py`,
`src/aug_spec/cli.py`, optionally `configs/_smoke_offload.yaml`.

### 4.1 Extend `RunConfig`

In [`src/aug_spec/cli.py`](src/aug_spec/cli.py#L73), add to `RunConfig`:

```python
@dataclass
class RunConfig:
    ...
    backend: str                    # "hf" | "offload"
    offload_path: Optional[Path]
    offload_device_memory_ratio: float
    offload_cache_policy: str       # "ondemand" | "caching"
```

In `RunConfig.from_yaml(...)`, parse `model.backend` (default `"hf"`)
and `model.offload.*` (only validated when `backend == "offload"`).
**Existing configs must remain valid** — `backend: hf` is the implicit
default.

### 4.2 Add `load_offload` to loader

In [`src/aug_spec/runtime/loader.py`](src/aug_spec/runtime/loader.py),
add a sibling to `load_model`. Note it returns **four** objects — the
`cpu_source` is the CPU-resident weight source for draft-side merge
(invariant 2 in §1).

```python
def load_offload(
    model_id: str,
    offload_path: Path,
    dtype: torch.dtype = torch.bfloat16,
    device_memory_ratio: float = 0.15,
    cache_policy: str = "ondemand",
    trust_remote_code: bool = True,
) -> Tuple[nn.Module, Any, Any, nn.Module]:
    """Returns (hf_model, tokenizer, moe_wrapper, cpu_source).

    hf_model    = moe.model — what controller/adapter/spec-bench see.
    moe_wrapper = MoE wrapper; caller calls _configure_hook(...) per gen.
    cpu_source  = parallel HF model on CPU, used ONLY by
                  adapter.build_weighted_avg_offload as a weight source.
                  Lives in host RAM, no forward passes.
    """
    from moe_infinity import MoE

    # 1) The offloaded model (target verify path).
    moe = MoE(model_id, {
        "offload_path": str(offload_path),
        "device_memory_ratio": device_memory_ratio,
    })
    moe.engine.expert_cache.set_cache_policy(
        "priority" if cache_policy == "caching" else "lru"
    )

    # 2) The CPU-resident weight source (draft merge path).
    #    Loaded OUTSIDE moe_infinity's engine context, so its weights
    #    are NOT replaced with shape-(1,) placeholders.
    cpu_source = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    cpu_source.eval()
    for p in cpu_source.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = moe.model
    hf_model.eval()
    return hf_model, tokenizer, moe, cpu_source
```

**Memory cost:** Mixtral-8x7B bf16 ≈ 26 GB host RAM; GPT-OSS-20B ≈ 22 GB;
Qwen3-30B-A3B ≈ 30 GB. TWCC nodes have ≥ 300 GB host RAM per node, so
this is fine. Pre-flight Q2 verifies this fits.

### 4.3 Wire backend into `run_experiment`

In [`cli.py::run_experiment`](src/aug_spec/cli.py#L159):

```python
if cfg.backend == "offload":
    model, tokenizer, moe, cpu_source = load_offload(
        cfg.model_id, cfg.offload_path,
        dtype=cfg.dtype,
        device_memory_ratio=cfg.offload_device_memory_ratio,
        cache_policy=cfg.offload_cache_policy,
        trust_remote_code=cfg.trust_remote_code,
    )
else:
    model, tokenizer = load_model(...)
    moe = None
    cpu_source = None
```

Pass `moe` to specbench (so it can call `_configure_hook` per question)
and `cpu_source` to the controller (so `build_weighted_avg_offload`
can read source weights — see §6).

### 4.4 Sanity check (Phase 1 gate)

```bash
python -c "from aug_spec.runtime.loader import load_offload; print('ok')"
aug_spec run --config configs/_smoke.yaml   # must still pass
```

Both must pass before Phase 2. The smoke is critical — if you broke
the HF path, you'll spend Phase 3 debugging the wrong thing.

---

## 5. Phase 2 — Adapter `_route_offload` branch

**Files touched:** `src/aug_spec/adapters/{mixtral,gptoss,qwen3}.py`.

Each adapter's `_standard_routing` currently hand-rolls the
expert-dispatch loop. In offload mode, delegate to
`block.expert_executor.dispatch_local`.

### 5.1 Mixtral

In [`src/aug_spec/adapters/mixtral.py`](src/aug_spec/adapters/mixtral.py#L68),
add a branch:

```python
def _standard_routing(self, block, hs_flat, gate_logits,
                      batch_size, sequence_length, hidden_dim):
    # NEW: dispatch path depends on block type
    if _is_offload_block(block):
        return self._route_offload(
            block, hs_flat, gate_logits,
            batch_size, sequence_length, hidden_dim)
    return self._route_hf(
        block, hs_flat, gate_logits,
        batch_size, sequence_length, hidden_dim)

def _route_offload(self, block, hs_flat, gate_logits,
                   batch_size, sequence_length, hidden_dim):
    """Mirror of SyncMixtralSparseMoeBlock.forward but operates on
    pre-computed router_logits so we share the upstream gate call.
    """
    import torch.nn.functional as F
    routing_weights = F.softmax(gate_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(
        routing_weights, block.top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hs_flat.dtype)

    router_mask = F.one_hot(
        selected_experts, num_classes=block.num_experts)
    routing_weights_mask = (
        routing_weights[:, :, None] * router_mask).permute(0, 2, 1)
    router_mask = router_mask.permute(0, 2, 1)
    router_mask = torch.logical_or(router_mask[:, :, 0], router_mask[:, :, 1])
    routing_weights_mask = torch.sum(routing_weights_mask, dim=-1)

    block.expert_executor.dispatch_local(
        block.layer_id, hs_flat, router_mask, routing_weights_mask)
    final = block.expert_executor.wait_dispatch_local()
    return final.view(batch_size, sequence_length, hidden_dim).to(hs_flat.dtype)
```

Rename the original `_standard_routing` body to `_route_hf` and call
it from the new branch.

```python
def _is_offload_block(block):
    return type(block).__name__ == "SyncMixtralSparseMoeBlock"
```

(Using `type().__name__` instead of `isinstance` avoids an unconditional
import of moe_infinity in the GPU-backend code path.)

### 5.2 GPT-OSS, Qwen3

Mirror the same structure. Look at
`moe_infinity/moe_infinity/models/{qwen,gptoss}.py` for the
equivalent of `SyncMixtralSparseMoeBlock.forward` — copy the routing
math, replace the dispatch call. The class name to check for is
discoverable via Phase 0 Q1.

### 5.3 Sanity check (Phase 2 gate)

A tiny unit test in `tests/test_offload_routing.py`:

```python
def test_one_block_forward_shape():
    moe = MoE(CKPT, {"offload_path": OFFLOAD_DIR,
                     "device_memory_ratio": 0.15})
    moe._configure_hook(torch.zeros((1, 4), dtype=torch.long, device="cuda:0"))
    adapter = MixtralAdapter()
    _, block = next(iter(adapter.iter_moe(moe.model)))
    hs = torch.randn(1, 4, block.hidden_dim,
                     dtype=torch.bfloat16, device="cuda:0")
    out = block(hs)[0] if isinstance(block(hs), tuple) else block(hs)
    assert out.shape == (1, 4, block.hidden_dim)
```

If this works, the offload routing path is correct in isolation.

---

## 6. Phase 2.5 — `build_weighted_avg_offload` (Option A: CPU-resident source)

**Files touched:** `src/aug_spec/adapters/{mixtral,gptoss,qwen3}.py`,
`src/aug_spec/controller.py`, `src/aug_spec/cli.py`,
draft files under `src/aug_spec/drafts/`.

Read §2.7 of next_step.md first. The key fact: in offload mode,
`moe.model.layers[i].block_sparse_moe.experts[e].w*.weight` returns
shape-`(1,)` zero placeholders (moe_infinity replaces all offloaded
`param.data` at load time). Reading them directly — even after any
"pin" call — does **not** materialize the real weight. The merge must
source from the separate CPU-resident model loaded in §4.2.

### 6.1 Contract

```python
def build_weighted_avg(self, block, weights, *, cpu_block=None):
    """
    weights:   List[float], len == num_experts in this block.
               Non-zero entries are the experts to merge.
    cpu_block: The corresponding MoE block on the CPU-resident copy of
               the model (cpu_source.model.layers[i].block_sparse_moe).
               None for HF backend.
    Returns:   dict with keys "w1", "w2", "w3" (Mixtral) or family-specific
               equivalents, all bf16, on the offloaded model's device.
    """
```

When `cpu_block is None`: existing behaviour, no change (reads
`block.experts[e].w*.weight` from the GPU-resident model).

When `cpu_block is not None`: stream from CPU.

```python
def build_weighted_avg_offload(cpu_block, weights, device, dtype):
    # 1. Pre-alloc fp32 accumulators on GPU, sized like one expert.
    ref_w1 = cpu_block.experts[0].w1.weight        # shape (interm, hidden)  on CPU
    ref_w2 = cpu_block.experts[0].w2.weight        # shape (hidden, interm)  on CPU
    ref_w3 = cpu_block.experts[0].w3.weight        # shape (interm, hidden)  on CPU
    w1_sum = torch.zeros(ref_w1.shape, dtype=torch.float32, device=device)
    w2_sum = torch.zeros(ref_w2.shape, dtype=torch.float32, device=device)
    w3_sum = torch.zeros(ref_w3.shape, dtype=torch.float32, device=device)

    # 2. Stream each nonzero-weight expert CPU→GPU, accumulate, drop.
    for e_idx, w in enumerate(weights):
        if w == 0.0:
            continue
        w1 = cpu_block.experts[e_idx].w1.weight.to(device, non_blocking=True).float()
        w2 = cpu_block.experts[e_idx].w2.weight.to(device, non_blocking=True).float()
        w3 = cpu_block.experts[e_idx].w3.weight.to(device, non_blocking=True).float()
        w1_sum.add_(w1, alpha=w)
        w2_sum.add_(w2, alpha=w)
        w3_sum.add_(w3, alpha=w)
        del w1, w2, w3   # release transients before next iteration

    # 3. Cast back to bf16 (storage form for the merged expert).
    return {
        "w1": w1_sum.to(dtype),
        "w2": w2_sum.to(dtype),
        "w3": w3_sum.to(dtype),
    }
```

**Peak GPU memory during this call: ~1.75 expert sizes** (the fp32
accumulator + one bf16 transient at any moment). Independent of how
many experts contribute — streaming keeps peak flat.

### 6.2 Plumbing the CPU source through

The adapter needs the corresponding `cpu_block` for the layer it's
merging. Pass `cpu_source` via the controller:

In [`src/aug_spec/controller.py::Controller.__init__`](src/aug_spec/controller.py#L35):

```python
def __init__(self, model, adapter, draft, *, cpu_source=None):
    ...
    self.cpu_source = cpu_source

    # Pre-resolve (layer_idx → cpu_block) for fast lookup during refresh.
    self.cpu_blocks: Dict[int, nn.Module] = {}
    if cpu_source is not None:
        for layer_idx, _block in adapter.iter_moe(cpu_source):
            self.cpu_blocks[layer_idx] = _block
```

In `cli.py::run_experiment`:

```python
controller = Controller(model, adapter, draft,
                        cpu_source=cpu_source)   # None on HF backend
```

In each draft's `refresh(...)` / `lazy_build(...)`, forward the
`cpu_block` kwarg:

```python
cpu_block = (controller.cpu_blocks.get(layer_idx)
             if controller.cpu_source is not None else None)
avg = adapter.build_weighted_avg(block, weights, cpu_block=cpu_block)
```

This is one extra kwarg on each `build_weighted_avg` call site —
there are ~5 across [src/aug_spec/drafts/](src/aug_spec/drafts/).

### 6.3 Merge frequency by draft

| Draft | When `build_weighted_avg_offload` runs | Cost per question |
|---|---|---|
| `topm_count` | Every verify cycle (in `controller.update_masks`) | M × 352 MB H2D × num_cycles × num_moe_layers |
| `prefill_count` | Once during prefill (in `draft.prepopulate`) | (# nonzero experts in prefill) × 352 MB × num_moe_layers, **once per question** |
| `prefill_topm_count` | Once during prefill, with top-M cap | M × 352 MB × num_moe_layers, **once per question** |

`topm_count` is the high-frequency path. For Mixtral M=2 / 32 layers /
~50 cycles per question: 2 × 32 × 50 = 3200 H2D copies of 352 MB =
~1.1 TB transferred per question over PCIe Gen5 (64 GB/s) ≈ 17.5
seconds *if* fully serialised. In practice torch's `.to(device, non_blocking=True)`
overlaps with the accumulate, and the CPU-bound source means the H2D
happens off the GPU compute critical path. **Profile this in Phase 3
smoke** — if it's the bottleneck, switch the default draft to
`prefill_count` for the production runs.

### 6.4 Sanity check (Phase 2.5 gate)

```bash
# Same merge result whether reading from offloaded model + cpu_source
# (in offload mode) or directly from the HF model (in hf mode).
python tests/test_build_weighted_avg_offload.py
```

The test should: load both backends with identical `model_id`, build a
merged expert with the same `weights` on each, compare
`(w1_off - w1_hf).abs().max() < 1e-3`. This catches any divergence
introduced by the CPU→GPU streaming path.

---

## 7. Phase 3 — Smoke offload (`configs/_smoke_offload.yaml`)

**Files added:** `configs/_smoke_offload.yaml`.

```yaml
model:
  id: mistralai/Mixtral-8x7B-v0.1
  dtype: bfloat16
  backend: offload
  offload:
    path: /work/morrisliu07/aug_spec/cache/offload/mixtral
    device_memory_ratio: 0.15
    cache_policy: ondemand

draft:
  name: topm_count                 # offload-safe; see next_step.md §2.3
  args: {}

run:
  T: 3
  questions_per_cat: 1
  max_new_tokens: 32
  warmup: false

output:
  dir: output/_smoke_offload
  label: _smoke_offload
```

### 7.1 Plumbing for `_configure_hook`

In [`src/aug_spec/runtime/specbench.py`](src/aug_spec/runtime/specbench.py),
add an optional `moe_wrapper` kwarg. Before each
`target_model.generate(...)` call (including the warmup):

```python
if moe_wrapper is not None:
    moe_wrapper._configure_hook(inputs["input_ids"])
```

This is what makes the archer expert tracer aware of the new
sequence. Without it, the offload path probably still produces
**numerically correct** outputs (the dispatcher works statelessly) but
the tracer's caching / prefetch logic is stale.

### 7.2 Acceptance (Phase 3 gate)

```bash
aug_spec run --config configs/_smoke_offload.yaml
```

Acceptance:
- Generates the same number of tokens as the HF smoke (within ±2 due to EOS jitter).
- `summary.json["overall"]["mean_accept_tokens"]` matches
  `output/_smoke/summary.json` ± 0.1.
- Wall time is **slower** than HF (expected; we now pay PCIe).
- `peak_vram_gb` is dramatically lower than the HF run (the whole
  point — Mixtral HF takes ~43 GB, offload should sit under
  `device_memory_ratio × 80 + non-expert` ≈ 20–25 GB).

If MAT differs by >0.1, suspect:
- `_route_offload` re-uses gate logits but moe_infinity's
  `SyncMixtralSparseMoeBlock.forward` re-computes them. Either path
  should yield identical numerics — if not, you reused stale logits.
- `build_weighted_avg_offload` reads pre-pin (garbage) weights.

If wall time is >10× HF, suspect:
- Pin/unpin is being called per-token, not per-cycle/per-question.
- The merged expert is being rebuilt every forward instead of cached.

---

## 8. Phase 3.5 — `drafts/specmoe.py` (GPU first)

This phase has **no offload dependency** — it gives us a head-to-head
acceptance comparison against our merge-based draft using only the HF
backend. See `next_step.md §3.5` and `§4`.

Skeleton:

```python
class SpecMoEDraft(DraftStrategy):
    cache_kind = "masked"   # we restrict routing to N experts

    def __init__(self, N=4, gamma=5, affinity_table_path=None):
        self.N = N
        self.gamma = gamma
        self.affinity = (
            load_affinity(affinity_table_path) if affinity_table_path
            else None)

    def refresh(self, adapter, blocks, draft_cache):
        # Pick top-N hottest experts from accumulated count.
        # Map gate-selected experts to their N-closest via L2 (when
        # affinity table present) or simply mask everything outside N
        # (no-affinity baseline).
        ...
```

Add `configs/mixtral_specmoe.yaml`, run, confirm it produces a sane
MAT (probably 1.5–2.5 — not as high as merge-based on Mixtral).

This phase produces the **head-to-head acceptance table** that the
paper's core claim ("merge-based > SpecMoE by ~10%") rests on. Land it
before any throughput work.

---

## 9. Phase 4 — `aug_spec bench` + `drafts/none.py`

**Files touched:** new `src/aug_spec/drafts/none.py`,
new `src/aug_spec/runtime/bench.py`, `src/aug_spec/cli.py`.

### 9.1 `drafts/none.py`

```python
class NoneDraft(DraftStrategy):
    """No-op draft — for MoE-OnDemand / MoE-Caching baselines."""
    cache_kind = "none"   # special-case in Controller
```

In `Controller.install`, if `cache_kind == "none"`, skip the forward
replacement entirely. The model runs vanilla.

### 9.2 `runtime/bench.py`

```python
def run_bench(model, tokenizer, *, questions_per_cat, max_new_tokens,
              spec_bench_cache, output_dir, label, moe_wrapper=None,
              warmup=True):
    """Pure generate loop, no spec decoding. Reports tokens/sec,
    latency_ms, peak_vram_gb. Same Spec-Bench prompt set as
    run_specbench so numbers are comparable."""
    ...
```

The structure is a stripped-down `run_specbench` with no patches and
no per-cycle telemetry.

### 9.3 CLI

```python
sub = p.add_subparsers(dest="command", required=True)
sub.add_parser("run", ...)     # existing
sub.add_parser("bench", ...)   # new — same --config flag
```

### 9.4 Acceptance (Phase 4 gate)

```bash
aug_spec bench --config configs/_smoke.yaml          # works on HF
aug_spec bench --config configs/_smoke_offload.yaml  # works on offload
```

Both report sane tokens/sec.

---

## 10. Phase 5 — NVML PCIe profiler + VRAM accounting

**Files touched:** new `src/aug_spec/runtime/profile.py`,
`runtime/bench.py`, `runtime/specbench.py` (for summary fields),
`cli.py` (for `run_experiment` summary).

### 10.1 PCIe profiler

```python
class PCIeProfiler:
    """Background thread polling NVML PCIe RX/TX counters at 50 Hz.
    Reports GB transferred during the timed region."""
    ...
```

Use `pynvml.nvmlDeviceGetPcieThroughput` or
`nvmlDeviceGetPcieReplayCounter` depending on driver. Sample every
20ms; on `__exit__`, integrate to total bytes RX/TX.

Wire into both `aug_spec bench` (whole-question scope) and
`aug_spec run` (per-question + per-cycle scope, with breakdowns).

### 10.2 VRAM accounting fields in `summary.json`

The VRAM-matched comparison in PROGRESS.md needs these to be reported.
Compute at end of run and emit alongside existing `peak_vram_gb`:

```python
def _compute_vram_breakdown(controller, moe, gpu_total_gb=80.0):
    """Returns dict with the 3 fields the comparison protocol needs."""
    # Draft side: sum of bytes in controller.draft_cache across layers.
    draft_bytes = 0
    for layer_idx, payload in controller.draft_cache.items():
        if isinstance(payload, dict):           # "averaged" cache_kind
            draft_bytes += sum(t.numel() * t.element_size()
                               for t in payload.values())
        elif torch.is_tensor(payload):          # "masked" cache_kind
            draft_bytes += payload.numel() * payload.element_size()
    # Target side: archer cache budget.
    target_cache_bytes = (
        moe.engine.config.device_memory_ratio * gpu_total_gb * 1024**3
        if moe is not None else 0)
    return {
        "draft_vram_gb": draft_bytes / 1024**3,
        "target_cache_vram_gb": target_cache_bytes / 1024**3,
        "expert_total_vram_gb": (draft_bytes + target_cache_bytes) / 1024**3,
    }
```

Add to `summary.json` (in `cli.py::run_experiment`):

```json
{
  ...
  "peak_vram_gb": 28.3,
  "vram_breakdown": {
    "draft_vram_gb": 11.25,
    "target_cache_vram_gb": 11.25,
    "expert_total_vram_gb": 22.5
  },
  ...
}
```

For HF backend: `target_cache_vram_gb = 0`, and `expert_total_vram_gb`
equals whatever the GPU-resident model occupies for experts.

### 10.3 Acceptance
- `bench` on HF backend reports ~0 GB transferred (everything resident).
- `bench` on offload backend reports a strictly positive number.
- `summary.json` on offload backend has the three `vram_breakdown`
  fields populated and they round-trip back to the YAML's declared
  budget (used by PROGRESS.md's VRAM-matched table — see protocol
  in PROGRESS.md).

---

## 11. Phases 6–9

Follow [next_step.md §4](next_step.md) table for milestone gates.

- **Phase 6**: Add `engine.replace_cache_candidates` hook to
  `specmoe` draft for offload backend. Verify it actually reduces
  PCIe vs `count`.
- **Phase 7**: `cache_policy: caching` enables tracer prefetcher.
  MoE-Caching baseline should show lower mean PCIe / step than
  MoE-OnDemand.
- **Phase 8**: VRAM-matched paired production recipes — see §11.1.
- **Phase 9**: Compare numbers vs SpecMoE paper Fig 11/12.

### 11.1 VRAM-matched paired recipes (Phase 8)

The headline paper claim is **"same total expert VRAM, higher
throughput + acceptance"**. To produce that table, configs must be
grouped by their total expert VRAM budget so they sit on the same
bench-point. The protocol is defined in PROGRESS.md ("VRAM-matched
comparison protocol"); here is the file layout.

```
configs/vram_22gb/                       # V = 22.5 GB total expert VRAM
├── mixtral_ours_topm_count.yaml         # ours: merged 11.25 + cache 11.25
├── mixtral_ours_prefill_count.yaml      # ours, prefill variant
├── mixtral_specmoe_N2.yaml              # SpecMoE @ N=2, cache=0
└── mixtral_baseline_ondemand.yaml       # MoE-OnDemand at same V

configs/vram_45gb/                       # V = 45 GB total expert VRAM
├── mixtral_ours_topm_count.yaml         # merged 11.25 + cache 33.75
├── mixtral_specmoe_N2.yaml              # N=2 + cache 22.5
├── mixtral_specmoe_N4.yaml              # N=4 + cache 0
└── mixtral_baseline_caching.yaml        # MoE-Caching at same V

configs/vram_90gb/                       # V = full (no offload)
├── mixtral_ours_topm_count.yaml         # GPU backend, all in VRAM
└── mixtral_specmoe_N4.yaml              # GPU backend, N=4 pinned
```

The same triplet for Qwen3-30B-A3B and GPT-OSS-20B goes under
`configs/vram_*gb/qwen3_*` / `gptoss_*`.

#### YAML metadata field for the protocol

Add an optional `comparison` block to the YAML schema. **This block is
metadata only** — it does not affect runtime behaviour, it just lets
the analysis scripts in Phase 9 group configs by V-point without
parsing the file path.

```yaml
comparison:
  vram_budget_gb: 22.5             # total expert VRAM at this bench-point
  group: vram_22gb                 # human-readable group label
  competitor: specmoe_N2           # or "ours_topm" / "baseline_ondemand"
```

Plumbing in [`src/aug_spec/cli.py`](src/aug_spec/cli.py):

```python
@dataclass
class RunConfig:
    ...
    comparison: Optional[Dict[str, Any]] = None   # raw passthrough
```

Echo it into `summary.json` under a top-level `comparison` key, so the
analysis layer can pivot on it.

#### Sanity check (Phase 8 gate)

For each `configs/vram_*gb/*.yaml`:
- The run completes.
- `summary.json["vram_breakdown"]["expert_total_vram_gb"]` is within
  ±0.5 GB of `comparison.vram_budget_gb` (proves we honoured the
  budget at runtime).
- The head-to-head table renders (see PROGRESS.md §"VRAM-matched
  comparison protocol" for the expected schema).

---

## 12. Common pitfalls (read before you debug)

- **`build_weighted_avg` returning zeros / wrong shapes** on offload —
  almost certainly the pin call was a no-op. Phase 0 Q3 should have
  caught this; if it didn't, re-run the pre-flight before any other
  debugging.
- **`shared_model_phase_patch` not flipping `in_draft_phase`** —
  monkey-patches are fragile across HF versions. If draft phase isn't
  triggering, print `controller.in_draft_phase` from inside the patched
  `forward` to confirm. (Bug source: `transformers >= 4.45` reshuffled
  `AssistedCandidateGenerator`.)
- **Wall time worse than expected** — the dispatcher runs two threads
  per GPU (`GPUFetchFunc`, `GPUExecFunc`). If your forward serialises
  them (e.g. you call `wait_dispatch_local` per expert instead of
  once per layer), throughput collapses. Always batch the
  `dispatch_local` call across all experts in a layer, then `wait`
  once.
- **`_configure_hook` not called per question** — symptom is that
  acceptance rates degrade over the course of a sweep as stale tracer
  state accumulates. Each `target.generate(...)` should be preceded by
  one `_configure_hook(input_ids)` call.
- **Pre-computed gate logits vs SyncBlock's own gate** — moe_infinity's
  `SyncMixtralSparseMoeBlock.forward` recomputes `block.gate(hidden_states)`.
  Our adapter computes it once and passes it down. These should give
  identical numerics, but if you accidentally pass already-softmaxed
  weights, you'll see acceptance collapse.
- **Don't edit `moe_infinity/core/` C++** unless Phase 0 Q2 forces it.
  Re-running `install.sh` after C++ edits is slow (~5 min). Python
  edits are live.

---

## 13. When you're done

A successful Phase 9 produces one `summary.json` per cell in the
VRAM-matched grid (see PROGRESS.md "VRAM-matched comparison protocol"
for the canonical layout). Minimum to claim the paper headline:

```
output/vram_22gb/mixtral_ours_topm_count/summary.json
output/vram_22gb/mixtral_specmoe_N2/summary.json
output/vram_22gb/mixtral_baseline_ondemand/summary.json
output/vram_45gb/mixtral_ours_topm_count/summary.json
output/vram_45gb/mixtral_specmoe_N4/summary.json
output/vram_45gb/mixtral_baseline_caching/summary.json
```

Then update PROGRESS.md with the head-to-head table per V-point:

```
### V = 22.5 GB total expert VRAM
| Method            | draft VRAM | cache VRAM | MAT  | AccRate | TPS  | GB/cycle |
|---|---|---|---|---|---|---|
| MoE-OnDemand      | 0          | 22.5       | n/a  | n/a     | …    | …        |
| SpecMoE @ N=2     | 22.5       | 0          | …    | …       | …    | …        |
| Ours (topm_count) | 11.25      | 11.25      | …    | …       | …    | …        |
```

If "Ours" wins on TPS at both V points (with equal acceptance or
higher), the paper's core claim reproduces. Save the summary tables
and stop — additional ablation work belongs in a follow-up branch.
