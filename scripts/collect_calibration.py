#!/usr/bin/env python
"""B1 — calibration collection for the static baselines
(baseline_tables_plan.md WS-B; consumed by build_hc_smoe.py (C1) and
search_naee.py (D1)).

Runs the target model (hf backend) over C4 calibration sequences and, per
MoE layer, captures:
  * freq        [n]      — top-k vote counts over ALL calibration tokens
                           (HC-SMoE's frequency-weighted merging weights)
  * input       [T, D]   — block-input hidden states for a fixed global
                           subsample of T tokens (same positions per layer)
  * output      [T, D]   — the block's MoE output for those tokens
                           (NAEE's reconstruction-loss reference)
  * router_logits [T, n] — for those tokens (NAEE re-routes kept subsets)
then computes, with the model still loaded:
  * o           [n, D]   — mean dense output of EVERY expert over the
                           subsample (HC-SMoE Eq.4), via batched bmm.

Usage (sbatch; see scripts/run_collect_calib.sh):
  python scripts/collect_calibration.py --model-id Qwen/Qwen3-30B-A3B-Base
  # → output/calibration/<tag>/{meta.json, expert_stats.pt, layer_<li>.pt}
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from pathlib import Path

import torch

from aug_spec.adapters import adapter_for_config, get_adapter
from aug_spec.runtime.loader import get_model_device, load_model


def sample_c4_sequences(tokenizer, c4_file: Path, n_seq: int, seq_len: int,
                        seed: int):
    """Wanda/NAEE recipe: shuffle the shard's documents (seeded), tokenize,
    keep a random seq_len window of each doc long enough. Deterministic."""
    texts = []
    with gzip.open(c4_file, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 20000:          # plenty of candidates, bounded parse time
                break
            texts.append(json.loads(line)["text"])
    rng = random.Random(seed)
    rng.shuffle(texts)
    seqs = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        if ids.shape[0] <= seq_len:
            continue
        start = rng.randint(0, ids.shape[0] - seq_len - 1)
        seqs.append(ids[start:start + seq_len])
        if len(seqs) == n_seq:
            return seqs
    raise RuntimeError(
        f"only found {len(seqs)}/{n_seq} documents with > {seq_len} tokens")


def stack_expert_weights(adapter, block):
    """[n, D, I] gate/up and [n, I, D] down stacks (bmm operand layout) from
    a block's RAW expert modules. gptoss needs the fused-tensor path (G4 —
    baseline_tables_plan.md WS-G)."""
    name = adapter.name
    if name == "qwen3_moe":
        keys = ("gate_proj", "up_proj", "down_proj")
    elif name == "mixtral":
        keys = ("w1", "w3", "w2")           # gate, up, down
    else:
        raise NotImplementedError(
            f"stack_expert_weights: adapter {name!r} not wired yet "
            f"(gptoss = plan item G3)")
    g, u, d = (torch.stack([getattr(e, k).weight for e in block.experts])
               for k in keys)
    return (g.transpose(1, 2).contiguous(), u.transpose(1, 2).contiguous(),
            d.transpose(1, 2).contiguous())


@torch.no_grad()
def mean_expert_outputs(adapter, block, x: torch.Tensor,
                        expert_chunk: int = 32,
                        token_chunk: int = 1024) -> torch.Tensor:
    """o[j] = mean_t E_j(x_t) for every expert (HC-SMoE Eq.4), computed as
    chunked batched bmm on x's device. Returns [n, D] fp32 (cpu).

    no_grad is LOAD-BEARING: expert weights have requires_grad=True, and
    without it every chunk's bmm graph stays alive across layers — job
    259816 OOM'd a 141GB H200 at layer 21 exactly this way."""
    from aug_spec.kernels.bmm import bmm_swiglu
    gw, uw, dw = stack_expert_weights(adapter, block)
    n = gw.shape[0]
    acc = torch.zeros(n, x.shape[1], dtype=torch.float64, device=x.device)
    for t0 in range(0, x.shape[0], token_chunk):
        xt = x[t0:t0 + token_chunk]
        for e0 in range(0, n, expert_chunk):
            out = bmm_swiglu(xt, gw[e0:e0 + expert_chunk],
                             uw[e0:e0 + expert_chunk],
                             dw[e0:e0 + expert_chunk])     # [e, t, D]
            acc[e0:e0 + expert_chunk] += out.double().sum(dim=1)
    return (acc / x.shape[0]).float().cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--c4-file",
                    default="data/c4/c4-train.00000-of-01024.json.gz")
    ap.add_argument("--n-seq", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--store-tokens", type=int, default=4096,
                    help="global token subsample stored per layer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None,
                    help="default: output/calibration/<model tag>")
    args = ap.parse_args()

    tag = args.model_id.split("/")[-1]
    out_dir = Path(args.out_dir or f"output/calibration/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[B1] loading {args.model_id} (hf) ...", flush=True)
    model, tokenizer = load_model(args.model_id, dtype=torch.bfloat16,
                                  device_map="auto")
    adapter = (get_adapter(args.adapter) if args.adapter
               else adapter_for_config(model.config))
    blocks = list(adapter.iter_moe(model))
    count_top_k = adapter.default_count_top_k(model)
    n_experts = adapter.num_experts(blocks[0][1])
    print(f"[B1] adapter={adapter.name} layers={len(blocks)} "
          f"experts={n_experts} count_top_k={count_top_k}", flush=True)

    seqs = sample_c4_sequences(tokenizer, Path(args.c4_file),
                               args.n_seq, args.seq_len, args.seed)
    total_tokens = args.n_seq * args.seq_len
    # Fixed GLOBAL subsample — identical token positions for every layer.
    g = torch.Generator()
    g.manual_seed(args.seed)
    sel_global = torch.randperm(total_tokens, generator=g)
    sel_global = sel_global[:args.store_tokens].sort().values
    sel_by_seq = {
        s: (sel_global[(sel_global // args.seq_len) == s] % args.seq_len)
        for s in range(args.n_seq)}

    freq = {li: torch.zeros(n_experts, dtype=torch.long) for li, _ in blocks}
    store = {li: {"input": [], "output": [], "router_logits": []}
             for li, _ in blocks}
    state = {"sel": torch.empty(0, dtype=torch.long)}

    def make_hook(li):
        def hook(module, inputs, output):
            hs = inputs[0]
            out, logits = output[0], output[1]
            flat_in = hs.reshape(-1, hs.shape[-1])
            flat_out = out.reshape(-1, out.shape[-1])
            logits = logits.reshape(-1, logits.shape[-1])
            top = logits.topk(count_top_k, dim=-1).indices
            freq[li] += torch.bincount(
                top.flatten(), minlength=n_experts).cpu()
            sel = state["sel"]
            if sel.numel():
                sel_d = sel.to(flat_in.device)
                store[li]["input"].append(
                    flat_in[sel_d].to("cpu", torch.bfloat16))
                store[li]["output"].append(
                    flat_out[sel_d].to("cpu", torch.bfloat16))
                store[li]["router_logits"].append(
                    logits[sel_d].to("cpu", torch.float32))
        return hook

    handles = [block.register_forward_hook(make_hook(li))
               for li, block in blocks]
    device = get_model_device(model)
    t0 = time.perf_counter()
    with torch.no_grad():
        for s, ids in enumerate(seqs):
            state["sel"] = sel_by_seq[s]
            model(input_ids=ids.unsqueeze(0).to(device))
            print(f"[B1] capture seq {s + 1}/{args.n_seq} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()

    # Phase 2 — per-layer files + dense mean expert outputs (model still
    # loaded; weights come straight off the blocks).
    o_all = {}
    for li, block in blocks:
        x_cpu = torch.cat(store[li]["input"])
        torch.save({"input": x_cpu,
                    "output": torch.cat(store[li]["output"]),
                    "router_logits": torch.cat(store[li]["router_logits"])},
                   out_dir / f"layer_{li}.pt")
        o_all[li] = mean_expert_outputs(
            adapter, block, x_cpu.to(device, torch.bfloat16))
        torch.cuda.empty_cache()      # per-layer weight stacks are transient
        print(f"[B1] layer {li}: stored {x_cpu.shape[0]} tokens, o_j done "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    torch.save({"o": o_all, "freq": {li: f for li, f in freq.items()},
                "count_top_k": count_top_k}, out_dir / "expert_stats.pt")
    meta = {
        "model_id": args.model_id, "adapter": adapter.name,
        "num_layers": len(blocks), "num_experts": n_experts,
        "count_top_k": count_top_k, "n_seq": args.n_seq,
        "seq_len": args.seq_len, "stored_tokens": int(sel_global.numel()),
        "seed": args.seed, "c4_file": str(args.c4_file),
        "layer_indices": [li for li, _ in blocks],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[B1] done → {out_dir} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
