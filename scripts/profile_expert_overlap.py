"""Diagnostic: do moe_precache's prefill-pinned top-k% experts overlap with the
DECODE-hot experts (the demand moe_caching's LRU keeps resident)?

Runs the offloaded model on spec-bench prompts with plain greedy generate and
counts, per MoE layer, per expert, the routing SEPARATELY for:
  * prefill  (the prompt forward, >1 token)  → A = prefill top-k%  = the set
    moe_precache pins.
  * decode   (each generated token, 1 token) → B = decode  top-k%  = the demand
    a cache serves; a large-budget LRU (moe_caching) keeps ~this set resident.

Routing is CACHE-POLICY-INDEPENDENT — the gate selects experts regardless of
where the weights live — so a single plain run characterises both A and B.

Reports, per layer and aggregated over layers:
  overlap              = |A ∩ B| / |A|                 (set overlap)
  decode_mass_cover_A  = Σ decode_count[A] / Σ decode_count   (what fraction of
                         DECODE routings the prefill-pinned set actually serves
                         — the direct "why precache saves fetches" number)
  prefill_mass_cover_A = Σ prefill_count[A] / Σ prefill_count (sanity, ~high)
  decode_mass_cover_B  = Σ decode_count[B] / Σ decode_count   (the ceiling: a
                         perfect decode-informed pin)

Writes a per-layer CSV and prints the aggregate summary.

Usage:
  python scripts/profile_expert_overlap.py --config configs/profile_overlap.yaml \
      [--device-memory-ratio 0.25] [--out output/overlap/per_layer.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch

from aug_spec.adapters import adapter_for_config, get_adapter
from aug_spec.cli import RunConfig
from aug_spec.runtime.loader import get_model_device, load_offload
from aug_spec.runtime.specbench import (
    _format_chat_prompt, _load_humaneval_questions,
    _load_spec_bench_questions, _sample_questions,
)


class _Profiler:
    """Per-layer prefill vs decode routing tallies via gate forward hooks."""

    def __init__(self, model, adapter, count_top_k):
        self.count_top_k = count_top_k
        self.layers = []          # (layer_id, num_experts)
        self.handles = []
        self.prefill = {}         # layer_id -> [E] float
        self.decode = {}          # layer_id -> [E] float
        for enum_i, block in adapter.iter_moe(model):
            lid = int(getattr(block, "layer_id", enum_i))
            E = adapter.num_experts(block)
            self.layers.append((lid, E))
            self.handles.append(
                block.gate.register_forward_hook(self._hook(lid, E)))

    def _hook(self, lid, E):
        def _h(_m, _i, out):
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(logits) or logits.dim() != 2:
                return
            n_tok = logits.shape[0]
            k = min(self.count_top_k, logits.shape[-1])
            top = torch.topk(logits.detach(), k, dim=-1).indices
            c = torch.bincount(top.reshape(-1),
                               minlength=E).to(torch.float32).cpu()
            tgt = self.decode if n_tok == 1 else self.prefill
            prev = tgt.get(lid)
            tgt[lid] = c if prev is None else prev + c
        return _h

    def remove(self):
        for h in self.handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device-memory-ratio", type=float, default=0.25,
                    help="archer pool as a fraction of GPU memory (routing is "
                         "cache-independent, so pick one that just runs fast)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(Path(args.config))
    pin_fraction = float(cfg.draft_args.get("pin_fraction", 0.10))
    out_path = Path(args.out or (cfg.output_dir / "expert_overlap.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[profile] loading {cfg.model_id} (device_memory_ratio="
          f"{args.device_memory_ratio}) ...", flush=True)
    model, tokenizer, moe, _ = load_offload(
        cfg.model_id, cfg.offload_path,
        device_memory_ratio=args.device_memory_ratio,
        dtype=cfg.dtype, trust_remote_code=cfg.trust_remote_code,
        load_cpu_source=False)
    model.eval()
    adapter = (get_adapter(cfg.adapter_name) if cfg.adapter_name
               else adapter_for_config(model.config))
    ctk = adapter.default_count_top_k(model)
    prof = _Profiler(model, adapter, ctk)

    cache_dir = (Path(cfg.spec_bench_cache) if cfg.spec_bench_cache
                 else Path.cwd() / "data" / "spec_bench")
    all_q = _load_spec_bench_questions(cache_dir)
    if cfg.humaneval:
        # Mirror run_specbench: fold HumanEval in BEFORE sampling so the shared
        # per-category sampler draws questions_per_cat of them (category name
        # "humaneval"), exactly like the benchmark.
        all_q = all_q + _load_humaneval_questions(cache_dir.parent / "humaneval")
    questions, _skip = _sample_questions(
        all_q, cfg.questions_per_cat, cfg.seed, cfg.skip_categories,
        cfg.mt_bench_pooled)
    print(f"[profile] {len(questions)} questions, count_top_k={ctk}, "
          f"pin_fraction={pin_fraction:.0%}", flush=True)

    dev = get_model_device(model)
    for qi, q in enumerate(questions):
        user_msg = q["turns"][0]
        prompt = (user_msg if q["category"] == "humaneval"
                  else _format_chat_prompt(tokenizer, [], user_msg))
        inputs = tokenizer(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            if moe is not None:
                moe._configure_hook(inputs["input_ids"])
            model.generate(**inputs, max_new_tokens=cfg.max_new_tokens,
                           do_sample=False)
        if (qi + 1) % 5 == 0:
            print(f"[profile] {qi + 1}/{len(questions)} done", flush=True)

    prof.remove()

    # ── per-layer overlap / coverage ──────────────────────────────────
    rows = []
    agg = {k: [] for k in ("overlap", "decode_cover_A",
                           "prefill_cover_A", "decode_cover_B")}
    for lid, E in prof.layers:
        pf = prof.prefill.get(lid)
        dc = prof.decode.get(lid)
        if pf is None or dc is None or dc.sum() == 0 or pf.sum() == 0:
            continue
        n_pin = max(1, int((pin_fraction * E) + 0.999))   # ceil
        A = set(torch.topk(pf, n_pin).indices.tolist())
        B = set(torch.topk(dc, n_pin).indices.tolist())
        idxA = torch.tensor(sorted(A))
        idxB = torch.tensor(sorted(B))
        overlap = len(A & B) / n_pin
        dcover_A = float(dc[idxA].sum() / dc.sum())
        pcover_A = float(pf[idxA].sum() / pf.sum())
        dcover_B = float(dc[idxB].sum() / dc.sum())
        rows.append({"layer": lid, "num_experts": E, "n_pin": n_pin,
                     "overlap": round(overlap, 4),
                     "decode_cover_A": round(dcover_A, 4),
                     "prefill_cover_A": round(pcover_A, 4),
                     "decode_cover_B": round(dcover_B, 4)})
        agg["overlap"].append(overlap)
        agg["decode_cover_A"].append(dcover_A)
        agg["prefill_cover_A"].append(pcover_A)
        agg["decode_cover_B"].append(dcover_B)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def stats(v):
        t = torch.tensor(v)
        return (float(t.mean()), float(t.median()),
                float(t.min()), float(t.max()))

    print("\n" + "=" * 68)
    print(f"  expert-overlap profile — {len(rows)} MoE layers, "
          f"pin_fraction={pin_fraction:.0%}")
    print("=" * 68)
    print(f"{'metric':22s} {'mean':>7s} {'median':>7s} {'min':>7s} {'max':>7s}")
    labels = {
        "overlap": "set overlap |A∩B|/|A|",
        "decode_cover_A": "decode mass by A(pin)",
        "prefill_cover_A": "prefill mass by A",
        "decode_cover_B": "decode mass by B(ceil)",
    }
    for k, lbl in labels.items():
        m, md, lo, hi = stats(agg[k])
        print(f"{lbl:22s} {m:7.3f} {md:7.3f} {lo:7.3f} {hi:7.3f}")
    print("=" * 68)
    print(f"  A = prefill top-{pin_fraction:.0%} (= moe_precache pins); "
          f"B = decode top-{pin_fraction:.0%} (= LRU demand).")
    print(f"  → 'decode mass by A' = fraction of DECODE routings the "
          f"prefill-pinned set serves.")
    print(f"  per-layer CSV → {out_path}", flush=True)


if __name__ == "__main__":
    main()
