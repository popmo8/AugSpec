#!/usr/bin/env python
"""DV1 — Draft&Verify skip-layer Bayesian-optimisation search
(Table 1 "Draft&Verify" row; produces the spec for the `draft_verify`
draft, mirroring search_naee.py → static_mask).

Thin CLI over `aug_spec.runtime.dv_search` (method rationale + objective
documented there). Loads the model on GPU (hf backend), runs the BO
search, writes the spec json the `draft_verify` draft consumes. A
`draft_verify` config that names no spec/keep-set resolves the same
default path and, if the file is missing, runs this search in-run — this
script is the offline route (preferred for offload-backend benchmarks,
where an in-run search would be fetch-bound).

Usage (sbatch; see scripts/run_search_dv.sh):
  python scripts/search_draft_verify.py --model-id Qwen/Qwen3-30B-A3B-Base
  # → output/draft_verify/<tag>_L<num-keep>.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aug_spec.adapters import adapter_for_config, get_adapter
from aug_spec.runtime import dv_search
from aug_spec.runtime.loader import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--c4-file", default=dv_search.C4_FILE)
    ap.add_argument("--num-keep", type=int, default=None,
                    help="MoE sublayers the draft executes (default: "
                         f"{dv_search.KEEP_FRAC:.1%} of the MoE layer "
                         "count — the shared draft budget)")
    ap.add_argument("--num-prompts", type=int, default=dv_search.NUM_PROMPTS)
    ap.add_argument("--prompt-len", type=int, default=dv_search.PROMPT_LEN)
    ap.add_argument("--gen-len", type=int, default=dv_search.GEN_LEN)
    ap.add_argument("--init-points", type=int, default=dv_search.INIT_POINTS)
    ap.add_argument("--iterations", type=int, default=dv_search.ITERATIONS)
    ap.add_argument("--seed", type=int, default=dv_search.SEED)
    ap.add_argument("--out", default=None,
                    help="spec json path (default: output/draft_verify/"
                         "<model tag>_L<num-keep>.json)")
    args = ap.parse_args()

    print(f"[dv] loading {args.model_id} (hf, bf16)")
    model, tokenizer = load_model(args.model_id, dtype=torch.bfloat16,
                                  device=torch.device("cuda:0"))
    model.eval()
    adapter = (get_adapter(args.adapter) if args.adapter
               else adapter_for_config(model.config))

    n_moe = sum(1 for _ in adapter.iter_moe(model))
    num_keep = dv_search.resolve_num_keep(n_moe, args.num_keep)
    out = Path(args.out) if args.out else dv_search.default_spec_path(
        args.model_id, num_keep)
    print(f"[dv] num_keep={num_keep}/{n_moe} → {out}")

    dv_search.search_and_save(
        model, tokenizer, adapter, num_keep, args.model_id, out,
        c4_file=args.c4_file, num_prompts=args.num_prompts,
        prompt_len=args.prompt_len, gen_len=args.gen_len,
        init_points=args.init_points, iterations=args.iterations,
        seed=args.seed)


if __name__ == "__main__":
    main()
