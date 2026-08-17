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
import torch.nn.functional as F

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
    a block's RAW expert modules. qwen3/mixtral only — gptoss's fused
    clamped-GLU layout goes through `expert_output_fn` instead (G3)."""
    name = adapter.name
    if name in ("qwen3_moe", "deepseek_moe"):
        keys = ("gate_proj", "up_proj", "down_proj")
    elif name == "mixtral":
        keys = ("w1", "w3", "w2")           # gate, up, down
    else:
        raise NotImplementedError(
            f"stack_expert_weights: adapter {name!r} has no raw expert "
            f"modules; use expert_output_fn")
    g, u, d = (torch.stack([getattr(e, k).weight for e in block.experts])
               for k in keys)
    return (g.transpose(1, 2).contiguous(), u.transpose(1, 2).contiguous(),
            d.transpose(1, 2).contiguous())


@torch.no_grad()
def gptoss_expert_outputs(experts, x: torch.Tensor,
                          e0: int, e1: int) -> torch.Tensor:
    """[e, t, D] raw (pre-routing-weight) outputs of experts e0:e1 on x —
    the fused clamped-GLU dense path, op-for-op the inference branch of
    `GptOssExperts.forward` (interleaved gate/up, clamp limits, biases)."""
    alpha = float(getattr(experts, "alpha", 1.702))
    limit = float(getattr(experts, "limit", 7.0))
    gw = experts.gate_up_proj[e0:e1]                       # [e, D, 2I]
    xt = x.unsqueeze(0).expand(gw.shape[0], -1, -1)        # [e, t, D]
    gate_up = torch.bmm(xt, gw) + experts.gate_up_proj_bias[e0:e1, None, :]
    gate = gate_up[..., ::2].clamp(max=limit)
    up = gate_up[..., 1::2].clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    out = torch.bmm((up + 1) * glu, experts.down_proj[e0:e1])
    return out + experts.down_proj_bias[e0:e1, None, :]


def expert_output_fn(adapter, block):
    """Per-family dense expert-output kernel: returns
    (callable(x [t, D], e0, e1) → [e, t, D] raw outputs, num_experts).
    qwen3/mixtral stack raw modules for `bmm_swiglu`; gptoss cannot reuse it
    (clamped GLU + biases, not silu) and runs its own fused path."""
    if adapter.name == "gptoss":
        experts = block.experts
        return (lambda x, e0, e1: gptoss_expert_outputs(experts, x, e0, e1),
                experts.gate_up_proj.shape[0])
    from aug_spec.kernels.bmm import bmm_swiglu
    gw, uw, dw = stack_expert_weights(adapter, block)
    return (lambda x, e0, e1: bmm_swiglu(x, gw[e0:e1], uw[e0:e1], dw[e0:e1]),
            gw.shape[0])


@torch.no_grad()
def mean_expert_outputs(adapter, block, x: torch.Tensor,
                        expert_chunk: int = 32,
                        token_chunk: int = 1024) -> torch.Tensor:
    """o[j] = mean_t E_j(x_t) for every expert (HC-SMoE Eq.4), computed as
    chunked batched bmm on x's device. Returns [n, D] fp32 (cpu).

    no_grad is LOAD-BEARING: expert weights have requires_grad=True, and
    without it every chunk's bmm graph stays alive across layers — job
    259816 OOM'd a 141GB H200 at layer 21 exactly this way."""
    fn, n = expert_output_fn(adapter, block)
    acc = torch.zeros(n, x.shape[1], dtype=torch.float64, device=x.device)
    for t0 in range(0, x.shape[0], token_chunk):
        xt = x[t0:t0 + token_chunk]
        for e0 in range(0, n, expert_chunk):
            out = fn(xt, e0, min(e0 + expert_chunk, n))    # [e, t, D]
            acc[e0:e0 + out.shape[0]] += out.double().sum(dim=1)
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

    is_gptoss = adapter.name == "gptoss"
    is_deepseek = adapter.name == "deepseek_moe"

    def make_hook(li):
        def hook(module, inputs, output):
            hs = inputs[0]
            flat_in = hs.reshape(-1, hs.shape[-1])
            out_t = output if torch.is_tensor(output) else output[0]
            flat_out = out_t.reshape(-1, out_t.shape[-1])
            if is_gptoss:
                # Native gpt-oss blocks return the post-scatter router
                # SCORES (zeros outside the natural top-4) as output[1] —
                # NAEE's kept-subset re-route needs FULL logits, so
                # recompute them from the block input (one linear).
                r = module.router
                logits = F.linear(flat_in.to(r.weight.dtype),
                                  r.weight, r.bias)
            elif is_deepseek:
                # DeepseekMoE returns a single tensor and its MoEGate
                # returns (topk_idx, topk_weight, aux) — recompute the FULL
                # logits the way the gate does (fp32 linear).
                logits = F.linear(flat_in.type(torch.float32),
                                  module.gate.weight.type(torch.float32))
            else:
                logits = output[1].reshape(-1, output[1].shape[-1])
            top = logits.topk(count_top_k, dim=-1).indices
            freq[li] += torch.bincount(
                top.flatten(), minlength=n_experts).cpu()
            sel = state["sel"]
            if sel.numel():
                sel_d = sel.to(flat_in.device)
                out_sel = flat_out[sel_d]
                if is_deepseek:
                    # The block output includes the always-on shared
                    # experts; NAEE reconstructs the ROUTED part only, so
                    # store the routed-only reference (shared cancels out
                    # of every candidate identically otherwise — but the
                    # stored reference must match what is reconstructed).
                    out_sel = out_sel - module.shared_experts(flat_in[sel_d])
                store[li]["input"].append(
                    flat_in[sel_d].to("cpu", torch.bfloat16))
                store[li]["output"].append(
                    out_sel.to("cpu", torch.bfloat16))
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
