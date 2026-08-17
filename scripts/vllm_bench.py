"""vLLM decode-throughput ceiling for Qwen3-30B-A3B (all-resident, fused MoE
kernels) — the "is the 108ms/token forward floor a naive-implementation
artifact or a hardware limit?" control (bottleneck report).

Loads the model fully on GPU (no expert offload) and measures end-to-end
tokens/s at B=1/4/64 with greedy decode. Output tokens >> prompt so tok/s is
decode-dominated; comparable to the moe_infinity offload numbers (6-9 tok/s).
Prints a small table.
"""

from __future__ import annotations

import argparse
import time

from vllm import LLM, SamplingParams


PROMPTS = [
    "Write a detailed explanation of how a mixture-of-experts transformer routes tokens to experts, and why it saves compute.",
    "Explain step by step how to implement a binary search tree in Python, with insertion and lookup.",
    "Describe the causes and consequences of the fall of the Western Roman Empire.",
    "Summarize the key ideas behind speculative decoding for large language models.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Base")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--cpu-offload-gb", type=float, default=0.0,
                    help="GB of weights to stream from CPU (0 = all-resident "
                         "ceiling). Model is ~61GB; e.g. 48 keeps ~13GB on GPU "
                         "to roughly match the moe_infinity 0.2x budget.")
    ap.add_argument("--batches", default="1,4,64")
    args = ap.parse_args()

    print(f"[vllm_bench] loading {args.model} "
          f"(cpu_offload_gb={args.cpu_offload_gb}, gpu_mem_util="
          f"{args.gpu_mem_util}) ...", flush=True)
    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util,
              cpu_offload_gb=args.cpu_offload_gb,
              enforce_eager=False, trust_remote_code=True, max_model_len=2048)

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                        ignore_eos=True)  # force full max_tokens for clean tok/s

    print(f"\n{'batch':>6s} {'gen_tokens':>11s} {'wall_s':>8s} "
          f"{'tok/s':>8s} {'tok/s/seq':>10s}")
    for b in [int(x) for x in args.batches.split(",")]:
        prompts = [(PROMPTS * ((b // len(PROMPTS)) + 1))[:b]]
        prompts = prompts[0]
        # warmup (esp. B=1 CUDA graph capture) — untimed
        llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8,
                                              ignore_eos=True), use_tqdm=False)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        wall = time.perf_counter() - t0
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"{b:>6d} {gen:>11d} {wall:>8.2f} {gen / wall:>8.2f} "
              f"{gen / wall / b:>10.2f}", flush=True)


if __name__ == "__main__":
    main()
