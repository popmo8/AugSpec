"""M9a (v2) — 五個基礎時間量測

1. T_expert_fwd  單顆 expert SiLU-MLP forward (GPU, 實際 hidden-state input)
2. T_layer_fwd   整個 MoE block forward (GPU, 含 gate + dispatch, hook 計時)
3. T_cpu_merge   單次 CPU merge: M 顆 experts → 1 averaged expert (CPU 累加，結果留在 CPU)
4. T_gpu_merge   單次 GPU merge: M 顆 experts 已在 GPU → GPU 累加 (不含 PCIe)
5. T_pcie        單顆 expert H2D transfer (CPU → GPU, 3 matrices)

Usage:
    sbatch tests/offload/m9a_profile.sh [--num-merge 32] [--n-iters 10]
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(REPO_ROOT, "tests", "offload", "m9a_profile.out")
PROJS = ("gate_proj", "up_proj", "down_proj")

_report: list[str] = []


def log(m: str = "") -> None:
    print(m, flush=True)
    _report.append(m)


def gpu_used_gb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True)
        return int(out.splitlines()[0]) / 1024
    except Exception:
        return -1.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--offload-dir",
                   default=os.path.join(REPO_ROOT, "moe_infinity",
                                        "offload_output", "Qwen3-30B-A3B"))
    p.add_argument("--device-memory-ratio", type=float, default=0.2)
    p.add_argument("--num-merge", type=int, default=2,
                   help="M: number of experts to merge into one")
    p.add_argument("--n-iters", type=int, default=10,
                   help="measurement iterations for GPU/PCIe ops")
    p.add_argument("--n-layer-iters", type=int, default=10,
                   help="full-model forwards used to time one MoE layer")
    args = p.parse_args()

    import torch
    import torch.nn.functional as F
    from aug_spec.adapters.qwen3 import Qwen3MoeAdapter
    from aug_spec.runtime.loader import load_offload

    log("=" * 70)
    log("M9a (v2) — 五個基礎時間量測")
    log(f"  model={args.model}  ratio={args.device_memory_ratio}  M={args.num_merge}")
    log("=" * 70)

    model, tokenizer, moe, cpu_source = load_offload(
        args.model, args.offload_dir,
        device_memory_ratio=args.device_memory_ratio,
        load_cpu_source=True)
    adapter = Qwen3MoeAdapter()
    blocks = list(adapter.iter_moe(model))
    cpu_blocks = dict(adapter.iter_moe(cpu_source))
    hidden = model.config.hidden_size
    L = len(blocks)
    M = args.num_merge

    li0, blk0 = blocks[0]
    cb0 = cpu_blocks[li0]
    expert_mb = sum(getattr(cb0.experts[0], pr).weight.numel()
                    for pr in PROJS) * 2 / 1e6
    n_exp = adapter.num_experts(blk0)
    dtype = cb0.experts[0].gate_proj.weight.dtype

    log(f"  MoE layers={L}  hidden={hidden}  expert_size={expert_mb:.2f} MB  "
        f"total experts={n_exp}  M={M}")

    # Uniform weights for M experts
    weights = [1.0 / M if e < M else 0.0 for e in range(n_exp)]

    # ── Warmup ──────────────────────────────────────────────────────────────
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to("cuda:0")
    moe._configure_hook(ids)
    with torch.no_grad():
        moe.generate(ids, max_new_tokens=8, do_sample=False,
                     pad_token_id=tokenizer.eos_token_id)

    # ── Capture realistic hidden states from layer 0 ─────────────────────
    hs_captured: dict[int, torch.Tensor] = {}

    def _capture_hook(m, inp, _li=li0):
        hs_captured[_li] = inp[0].detach()

    h_cap = blk0.register_forward_pre_hook(_capture_hook)
    moe._configure_hook(ids)
    with torch.no_grad():
        model(ids)
    h_cap.remove()

    hs_flat = hs_captured[li0].reshape(-1, hidden)  # (num_tokens, hidden) on GPU

    # ── Helper: median timer ─────────────────────────────────────────────
    def median_ms(fn, n=args.n_iters, warmup=2, sync=True):
        for _ in range(warmup):
            fn()
            if sync:
                torch.cuda.synchronize()
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            if sync:
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(samples)

    # ── 3. T_cpu_merge: CPU merge (M experts → 1, 結果留在 CPU) ──────────
    # 先量 cpu_merge 因為不需要 GPU，可以與後面的 GPU 準備工作分開
    def do_cpu_merge():
        adapter.build_weighted_avg(cb0, weights)

    log("\n[量測中] T_cpu_merge ...")
    t_cpu_merge = median_ms(do_cpu_merge, sync=False)

    # ── 4. T_gpu_merge: GPU merge (M experts 已在 GPU，GPU 累加) ─────────
    # Pre-upload M expert weights to GPU (這個上傳時間不算入 T_gpu_merge)
    log("[量測中] 上傳 M 顆 expert 到 GPU (用於 GPU merge 計時，不計入結果) ...")
    gpu_experts: list[dict[str, torch.Tensor]] = []
    for e_idx in range(M):
        e = cb0.experts[e_idx]
        gpu_experts.append({
            "gate_proj": e.gate_proj.weight.to("cuda:0"),
            "up_proj":   e.up_proj.weight.to("cuda:0"),
            "down_proj": e.down_proj.weight.to("cuda:0"),
        })
    torch.cuda.synchronize()

    def do_gpu_merge():
        g_s = torch.zeros_like(gpu_experts[0]["gate_proj"], dtype=torch.float32)
        u_s = torch.zeros_like(gpu_experts[0]["up_proj"],   dtype=torch.float32)
        d_s = torch.zeros_like(gpu_experts[0]["down_proj"], dtype=torch.float32)
        w = 1.0 / M
        for ge in gpu_experts:
            g_s.add_(ge["gate_proj"].float(), alpha=w)
            u_s.add_(ge["up_proj"].float(),   alpha=w)
            d_s.add_(ge["down_proj"].float(), alpha=w)
        return {"gate_proj": g_s.to(dtype),
                "up_proj":   u_s.to(dtype),
                "down_proj": d_s.to(dtype)}

    log("[量測中] T_gpu_merge ...")
    t_gpu_merge = median_ms(do_gpu_merge)

    # ── 5. T_pcie: 單顆 expert H2D (CPU → GPU) ───────────────────────────
    cpu_e0 = cb0.experts[0]

    def do_pcie():
        return [getattr(cpu_e0, pr).weight.to("cuda:0") for pr in PROJS]

    log("[量測中] T_pcie ...")
    t_pcie = median_ms(do_pcie)

    # ── 1. T_expert_fwd: 單顆 expert forward (GPU) ───────────────────────
    merged_gpu = do_gpu_merge()

    def do_expert_fwd():
        adapter._run_dense_expert(merged_gpu, hs_flat)

    log("[量測中] T_expert_fwd ...")
    t_expert_fwd = median_ms(do_expert_fwd)

    # ── 2. T_layer_fwd: 整個 MoE block forward (hook 計時) ───────────────
    # Hook 只掛在 layer 0 (blk0)；每次 full forward 量一次，共 n_layer_iters 次。
    # 在 pre/post hook 裡 synchronize，確保 GPU 時間邊界清楚。
    log(f"[量測中] T_layer_fwd ({args.n_layer_iters} full-model forwards) ...")
    _pre_t: list[float] = []
    _post_t: list[float] = []

    def _pre_hook(m, inp):
        torch.cuda.synchronize()
        _pre_t.append(time.perf_counter())

    def _post_hook(m, inp, out):
        torch.cuda.synchronize()
        _post_t.append(time.perf_counter())

    h1 = blk0.register_forward_pre_hook(_pre_hook)
    h2 = blk0.register_forward_hook(_post_hook)

    # Warmup
    for _ in range(2):
        moe._configure_hook(ids)
        with torch.no_grad():
            model(ids)
    _pre_t.clear()
    _post_t.clear()

    for _ in range(args.n_layer_iters):
        moe._configure_hook(ids)
        with torch.no_grad():
            model(ids)

    h1.remove()
    h2.remove()

    layer_dts = [(post - pre) * 1e3
                 for pre, post in zip(_pre_t, _post_t)]
    t_layer_fwd = statistics.median(layer_dts) if layer_dts else -1.0

    # ── 輸出 ──────────────────────────────────────────────────────────────
    fetch_bw = expert_mb / t_pcie if t_pcie > 0 else 0.0

    log("")
    log("=" * 70)
    log(f"結果 (median,  n_tokens={hs_flat.shape[0]},  M={M}):")
    log("")
    log(f"  1. T_expert_fwd  單顆 expert forward        (GPU)  : {t_expert_fwd:8.3f} ms")
    log(f"  2. T_layer_fwd   整個 MoE block forward     (GPU)  : {t_layer_fwd:8.2f} ms")
    log(f"  3. T_cpu_merge   CPU merge ({M:2d} experts → 1)  (CPU)  : {t_cpu_merge:8.2f} ms")
    log(f"  4. T_gpu_merge   GPU merge ({M:2d} experts → 1)  (GPU)  : {t_gpu_merge:8.2f} ms")
    log(f"  5. T_pcie        單顆 expert H2D ({expert_mb:.1f} MB)    : {t_pcie:8.3f} ms"
        f"  ({fetch_bw:.1f} GB/s)")
    log("")
    log("比值分析:")
    log(f"  T_cpu_merge / T_layer_fwd  = {t_cpu_merge:.2f} / {t_layer_fwd:.2f}"
        f" = {t_cpu_merge / t_layer_fwd:.2f}"
        f"  → {'可藏進單層 ✓' if t_cpu_merge < t_layer_fwd else '需跨多層 overlap'}")
    log(f"  T_gpu_merge / T_layer_fwd  = {t_gpu_merge:.2f} / {t_layer_fwd:.2f}"
        f" = {t_gpu_merge / t_layer_fwd:.2f}")
    log(f"  T_pcie / T_layer_fwd       = {t_pcie:.3f} / {t_layer_fwd:.2f}"
        f" = {t_pcie / t_layer_fwd:.3f}  (單顆 expert 上傳 vs 一層 forward)")
    log(f"  T_pcie × M / T_layer_fwd   = {t_pcie * M:.2f} / {t_layer_fwd:.2f}"
        f" = {t_pcie * M / t_layer_fwd:.2f}  (M 顆 expert 上傳 vs 一層 forward)")
    log("")
    log(f"  GPU used (ratio={args.device_memory_ratio}) : {gpu_used_gb():.1f} GB")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        for ln in traceback.format_exc().rstrip().splitlines():
            log(ln)
    finally:
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w") as f:
            f.write("\n".join(_report) + "\n")
        print(f"\nreport → {OUT_PATH}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
