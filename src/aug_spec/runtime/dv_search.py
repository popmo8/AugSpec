"""Draft&Verify skip-layer BO search core (Table 1 "Draft&Verify" row).

Selects which MoE sublayers the `draft_verify` draft keeps, via the
paper's Bayesian optimisation (zhang2024draftverify §3.3) over a
continuous relaxation of the skip vector, with the objective adapted to
the acceptance-only Table 1 protocol: greedy draft/target token
agreement, teacher-forced over full-model greedy continuations of C4
prompts (same shard + sampler as B1's calibration). Under greedy
verification every accepted position conditions on the exact target
prefix, so per-position agreement on target-forced context is the
per-token acceptance probability — one forward pass per candidate
instead of a full generation loop.

Two entry points share this module:
  * `scripts/search_draft_verify.py` — standalone GPU job
    (scripts/run_search_dv.sh), the offline route;
  * `cli.py` — when a `draft_verify` config names no spec/keep-set, the
    run resolves `default_spec_path()` and, if the file is missing, calls
    `search_and_save()` in-run before the controller installs (the wrap
    is restored in a finally, so the controller sees clean forwards).
    Works on the offload backend too, but fetch-bound — prefer the
    standalone script there.

The budget (how MANY layers to keep) is not searched: `resolve_num_keep`
derives it from the shared 12.5% draft-side expert-memory fraction
(Qwen3 48→6), or an explicit `num_keep` override.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

# Shared search hyperparameter defaults (script flags mirror these).
KEEP_FRAC = 0.125
C4_FILE = "data/c4/c4-train.00000-of-01024.json.gz"
NUM_PROMPTS = 8
PROMPT_LEN = 64
GEN_LEN = 192
INIT_POINTS = 20
ITERATIONS = 380
SEED = 0


def default_spec_path(model_id: str, num_keep: int) -> Path:
    return Path("output/draft_verify") / (
        f"{model_id.split('/')[-1]}_L{num_keep}.json")


def resolve_num_keep(n_moe: int, num_keep=None, keep_frac=None) -> int:
    """Kept-layer budget: explicit `num_keep`, else `keep_frac` (default
    the shared 12.5% expert-memory fraction) of the MoE layer count."""
    if num_keep is not None:
        return int(num_keep)
    frac = KEEP_FRAC if keep_frac is None else float(keep_frac)
    return max(1, round(frac * n_moe))


def keep_from_params(params: Dict[str, float], moe_ids: Sequence[int],
                     num_keep: int) -> Tuple[int, ...]:
    """BO's continuous point → kept-layer set: the `num_keep` MoE layers
    with the highest scores (ties broken toward earlier layers, so the
    mapping is deterministic)."""
    ranked = sorted(moe_ids, key=lambda li: (-params[f"l{li}"], li))
    return tuple(sorted(ranked[:num_keep]))


def params_for_keep(keep: Sequence[int], moe_ids: Sequence[int],
                    ) -> Dict[str, float]:
    """Inverse of keep_from_params for probing heuristic seeds."""
    keep_set = set(keep)
    return {f"l{li}": 1.0 if li in keep_set else 0.0 for li in moe_ids}


def heuristic_keeps(moe_ids: Sequence[int],
                    num_keep: int) -> Dict[str, Tuple[int, ...]]:
    """Seed points: first-k (speed-equivalent depth prefix), evenly spaced,
    last-k (the paper found skips cluster in the latter half, i.e. KEEPS
    cluster early — first-k should be a strong seed)."""
    n = len(moe_ids)
    return {
        "first_k": tuple(moe_ids[:num_keep]),
        "even_k":  tuple(moe_ids[i * n // num_keep]
                         for i in range(num_keep)),
        "last_k":  tuple(moe_ids[-num_keep:]),
    }


def count_agreement(pred: torch.Tensor, ids: torch.Tensor, prompt_len: int,
                    gen_lens: Sequence[int]) -> Tuple[int, int]:
    """(matches, total) over the generated region. `pred[b, t]` is the
    argmax prediction for position t+1; sequence b's generated tokens sit
    at ids[b, prompt_len : prompt_len + gen_lens[b]]."""
    matches = total = 0
    for b, g in enumerate(gen_lens):
        if g <= 0:
            continue
        p = pred[b, prompt_len - 1:prompt_len + g - 1]
        t = ids[b, prompt_len:prompt_len + g]
        matches += int((p == t).sum())
        total += g
    return matches, total


def run_search(model, tokenizer, adapter, num_keep: int, model_id: str, *,
               c4_file: str = C4_FILE, num_prompts: int = NUM_PROMPTS,
               prompt_len: int = PROMPT_LEN, gen_len: int = GEN_LEN,
               init_points: int = INIT_POINTS, iterations: int = ITERATIONS,
               seed: int = SEED) -> Dict[str, object]:
    """BO-search the kept MoE-sublayer set on an already-loaded model.

    The MoE block forwards are wrapped only for the candidate-evaluation
    phase and restored in a finally, so callers (the CLI's controller)
    can install their own swaps afterwards. Returns the spec dict
    consumed by the `draft_verify` draft.
    """
    from bayes_opt import BayesianOptimization   # deferred: search-only dep

    # collect_calibration.py is a script, not a package module — reuse its
    # C4 sampler the same way its sibling scripts do.
    scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from collect_calibration import sample_c4_sequences

    from .loader import get_model_device
    device = get_model_device(model)
    blocks = list(adapter.iter_moe(model))
    moe_ids = [li for li, _ in blocks]
    if not 0 < num_keep < len(moe_ids):
        raise ValueError(f"num_keep {num_keep} out of range for "
                         f"{len(moe_ids)} MoE layers")

    # ── dev set: full-model greedy continuations of C4 prompts ─────────
    prompts = sample_c4_sequences(tokenizer, Path(c4_file), num_prompts,
                                  prompt_len, seed)
    print(f"[dv] generating references: {len(prompts)} prompts × "
          f"{gen_len} tokens (full model, greedy)")
    seqs: List[torch.Tensor] = []
    gen_lens: List[int] = []
    t0 = time.time()
    with torch.no_grad():
        for ids in prompts:
            inp = ids.unsqueeze(0).to(device)
            outp = model.generate(
                inp, attention_mask=torch.ones_like(inp),
                max_new_tokens=gen_len, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)
            seqs.append(outp[0].cpu())
            gen_lens.append(outp.shape[1] - prompt_len)
    print(f"[dv] references done in {time.time() - t0:.0f}s, "
          f"gen lens: {gen_lens}")
    if all(g <= 0 for g in gen_lens):
        raise RuntimeError("[dv] every reference generation was empty")

    T = max(s.shape[0] for s in seqs)
    ids_cpu = torch.full((len(seqs), T), tokenizer.pad_token_id,
                         dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids_cpu[b, :s.shape[0]] = s
        mask[b, :s.shape[0]] = 1
    batch = ids_cpu.to(device)
    mask = mask.to(device)

    # ── candidate evaluation: skip-wrap the MoE sublayers ──────────────
    # Wrapped once; per-candidate we only mutate `state["keep"]`.
    state: Dict[str, object] = {"keep": None}

    def make_wrap(li: int, orig):
        def fwd(hidden_states, *a, **kw):
            keep = state["keep"]
            if keep is not None and li not in keep:
                # Zero contribution in the decoder layer's return
                # convention (adapter-supplied; see MoEAdapter.mlp_skip_output).
                return adapter.mlp_skip_output(hidden_states)
            return orig(hidden_states, *a, **kw)
        return fwd

    n_evals = 0
    t_search = time.time()

    @torch.no_grad()
    def evaluate(keep: Tuple[int, ...]) -> float:
        nonlocal n_evals
        n_evals += 1
        state["keep"] = keep
        try:
            logits = model(input_ids=batch, attention_mask=mask,
                           use_cache=False).logits
        finally:
            state["keep"] = None
        pred = logits.argmax(dim=-1).cpu()
        m, t = count_agreement(pred, ids_cpu, prompt_len, gen_lens)
        if n_evals % 25 == 0:
            print(f"[dv] eval {n_evals} "
                  f"({time.time() - t_search:.0f}s elapsed)")
        return m / t

    # ── Bayesian optimisation over the continuous skip relaxation ──────
    cache: Dict[Tuple[int, ...], float] = {}
    best: Dict[str, object] = {"keep": None, "agreement": -1.0}

    def objective(**params) -> float:
        keep = keep_from_params(params, moe_ids, num_keep)
        score = cache.get(keep)
        if score is None:
            score = cache[keep] = evaluate(keep)
            if score > best["agreement"]:
                best.update(keep=keep, agreement=score)
                print(f"[dv] eval {n_evals}: NEW BEST {score:.4f} "
                      f"keep={list(keep)}")
        return score

    pbounds = {f"l{li}": (0.0, 1.0) for li in moe_ids}
    optimizer = BayesianOptimization(
        f=objective, pbounds=pbounds, random_state=seed,
        verbose=0, allow_duplicate_points=True)
    heuristics = heuristic_keeps(moe_ids, num_keep)
    for keep in heuristics.values():
        optimizer.probe(params=params_for_keep(keep, moe_ids), lazy=True)

    originals = [(block, block.forward) for _, block in blocks]
    for (li, block), (_, orig) in zip(blocks, originals):
        block.forward = make_wrap(li, orig)
    try:
        optimizer.maximize(init_points=init_points, n_iter=iterations)
    finally:
        for block, orig in originals:
            block.forward = orig
    print(f"[dv] search done: {n_evals} unique evals in "
          f"{time.time() - t_search:.0f}s")

    keep = list(best["keep"])
    print(f"[dv] best agreement {best['agreement']:.4f} keep={keep}")
    print(f"[dv] heuristics: "
          + ", ".join(f"{n}={cache[k]:.4f}" for n, k in heuristics.items()))
    return {
        "model_id": model_id,
        "num_layers": len(model.model.layers),
        "num_keep": num_keep,
        "mlp_keep": keep,
        "attn_skip": [],
        "objective": "greedy token agreement on C4 dev continuations",
        "agreement": best["agreement"],
        "heuristic_baselines": {
            name: cache[k] for name, k in heuristics.items()},
        "search": {"init_points": init_points, "iterations": iterations,
                   "seed": seed, "unique_evals": len(cache)},
        "data": {"c4_file": str(c4_file), "num_prompts": num_prompts,
                 "prompt_len": prompt_len, "gen_len": gen_len,
                 "gen_lens": gen_lens},
    }


def search_and_save(model, tokenizer, adapter, num_keep: int, model_id: str,
                    out_path: Path, **knobs) -> Dict[str, object]:
    """run_search + write the spec json (parents created)."""
    spec = run_search(model, tokenizer, adapter, num_keep, model_id, **knobs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"[dv] spec → {out_path}")
    return spec
