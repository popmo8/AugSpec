"""M4 — offload_plan.md merge 原型:三變體計時 + 精度比對（選項二）。

merge 完全不碰 archer / dispatch:來源是 cpu_source 的真實權重、目標是
我們自己的 GPU tensor。故 M4 **只載 cpu_source、不載 MoE 引擎**(同進程
共存已 M1 Q2 驗過),專心量 merge 本身。

§1.3 原本只留 (c)(d) 兩個合規變體並淘汰 (a),但那是 Mixtral 的記憶體
帳(expert 352 MB → (a) 累加器 1 GB 撐爆工作區)。Qwen3 expert 僅
9.4 MB,(a) 累加器才 ~19 MB **不撐爆**,淘汰理由不成立 → 三個全測,
讓數據決定 Qwen3 主線該用哪個(可能與 Mixtral 不同)。

三變體(都對齊現有 hf 版 build_weighted_avg 的 fp32 累加順序):
  (a) gpu_full     整顆 fp32 累加器,M 顆 expert 各整顆 H2D。PCIe = M 顆
  (c) gpu_chunked  chunk 外層 / expert 內層,暫存只兩個 chunk。PCIe = M 顆
  (d) cpu_merge    host 上 fp32 合完,只傳 1 顆結果 H2D。PCIe = 1 顆,CPU 算

每變體量:數值(對 reference 逐位元 / allclose)、單層 wall(暖機後
median)、GPU 暫存峰值、PCIe 理論傳輸量。

reference = aug_spec 現有 Qwen3MoeAdapter.build_weighted_avg(純 CPU fp32),
即要對齊的權威,也是 (d) 的核心。

Usage:
    .venv/bin/python tests/offload/m4_merge.py [--num-merge 16] [--chunk-rows 256]
    sbatch tests/offload/m4_merge.sh
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(REPO_ROOT, "tests", "offload", "m4_merge.out")

PROJS = ("gate_proj", "up_proj", "down_proj")

_report_lines: list[str] = []
_failures: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _report_lines.append(msg)


# ────────────────────────────────────────────────────────────────────
# 三個 merge 變體
# ────────────────────────────────────────────────────────────────────

def variant_a_gpu_full(cpu_block, nonzero):
    """整顆 fp32 累加器:每顆 expert 整顆 H2D,GPU 上 fp32 累加。"""
    import torch
    out = {}
    for proj in PROJS:
        ref = getattr(cpu_block.experts[0], proj).weight
        acc = torch.zeros(tuple(ref.shape), dtype=torch.float32, device="cuda:0")
        for e, w in nonzero:
            src = getattr(cpu_block.experts[e], proj).weight   # CPU bf16
            acc.add_(src.to("cuda:0").float(), alpha=w)
        out[proj] = acc.to(torch.bfloat16)
    return out


def variant_c_gpu_chunked(cpu_block, nonzero, chunk_rows):
    """分塊 fp32:chunk 外層、expert 內層,暫存只兩個 chunk。
    每個元素的累加順序(expert 0,1,2...)與 (a)/reference 完全相同,
    只是把張量切塊處理 → fp32 結果應逐位元一致。"""
    import torch
    out = {}
    for proj in PROJS:
        ref = getattr(cpu_block.experts[0], proj).weight
        rows, cols = ref.shape
        res = torch.empty((rows, cols), dtype=torch.bfloat16, device="cuda:0")
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            buf = torch.zeros((end - start, cols), dtype=torch.float32,
                              device="cuda:0")
            for e, w in nonzero:
                src = getattr(cpu_block.experts[e], proj).weight[start:end]
                buf.add_(src.to("cuda:0").float(), alpha=w)
            res[start:end] = buf.to(torch.bfloat16)
        out[proj] = res
    return out


def variant_d_cpu_merge(adapter, cpu_block, weights):
    """CPU 上 fp32 合(= reference),只把結果傳 GPU。"""
    cpu_out = adapter.build_weighted_avg(cpu_block, weights)
    return {k: v.to("cuda:0") for k, v in cpu_out.items()}


# ────────────────────────────────────────────────────────────────────
# 量測工具
# ────────────────────────────────────────────────────────────────────

def _sync():
    import torch
    torch.cuda.synchronize()


def _time_median(fn, n=10, warmup=2):
    """暖機(觸 mmap 頁 + cuda init)後計時 n 次取 median 秒。"""
    for _ in range(warmup):
        fn()
        _sync()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), min(samples)


def _peak_mb(fn):
    """跑一次,回報該變體的 GPU 配置峰值(MB)。我們的 buffer 走 torch
    allocator,量得到;含輸出 merged expert(~9.4MB)+ 暫存。"""
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return (peak - base) / (1024 ** 2)


def _max_abs_diff(out, ref_cpu):
    import torch
    d = 0.0
    for proj in PROJS:
        d = max(d, float((out[proj].cpu().float()
                          - ref_cpu[proj].float()).abs().max()))
    return d


# ────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--layer-idx", type=int, default=0)
    p.add_argument("--num-merge", type=int, default=16,
                   help="要合的 expert 數 M(對齊 PROGRESS.md K=16)")
    p.add_argument("--chunk-rows", type=int, default=256,
                   help="(c) 變體的分塊大小(列)")
    p.add_argument("--n-iters", type=int, default=10)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM
    from aug_spec.adapters.qwen3 import Qwen3MoeAdapter

    log("=" * 68)
    log("M4 probe — merge 三變體計時 + 精度(Qwen3,只載 cpu_source)")
    log(f"  model      : {args.model}")
    log(f"  layer_idx  : {args.layer_idx}   M(num_merge): {args.num_merge}")
    log(f"  chunk_rows : {args.chunk_rows}   n_iters: {args.n_iters}")
    log(f"  torch CPU threads : {torch.get_num_threads()}")
    log("=" * 68)

    log("\nloading cpu_source (device_map='cpu', bf16) ...")
    cpu_source = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cpu", torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    adapter = Qwen3MoeAdapter()
    cpu_block = cpu_source.model.layers[args.layer_idx].mlp
    num_experts = len(cpu_block.experts)

    # 一組 weights:選 M 顆均勻權重 1/M,其餘 0(merge 機制驗證,內容不重要)
    M = args.num_merge
    chosen = list(range(M))
    weights = [0.0] * num_experts
    for e in chosen:
        weights[e] = 1.0 / M
    nonzero = [(e, weights[e]) for e in chosen]

    expert_bytes = sum(
        getattr(cpu_block.experts[0], proj).weight.numel() for proj in PROJS
    ) * 2  # bf16
    log(f"\n  num_experts={num_experts}  expert_size={expert_bytes/1e6:.2f} MB")

    # reference:純 CPU fp32 權威
    ref_cpu = adapter.build_weighted_avg(cpu_block, weights)

    variants = {
        "(a) gpu_full   ": (
            lambda: variant_a_gpu_full(cpu_block, nonzero), M),
        "(c) gpu_chunked": (
            lambda: variant_c_gpu_chunked(cpu_block, nonzero, args.chunk_rows), M),
        "(d) cpu_merge  ": (
            lambda: variant_d_cpu_merge(adapter, cpu_block, weights), 1),
    }

    rows = []
    for name, (fn, pcie_experts) in variants.items():
        log(f"\n[{name.strip()}]")
        try:
            out = fn()
            diff = _max_abs_diff(out, ref_cpu)
            bit_exact = all(
                torch.equal(out[proj].cpu(), ref_cpu[proj]) for proj in PROJS)
            del out
            torch.cuda.empty_cache()

            peak = _peak_mb(fn)
            median, best = _time_median(fn, n=args.n_iters)
            pcie_mb = pcie_experts * expert_bytes / 1e6

            verdict = "逐位元" if bit_exact else f"max|Δ|={diff:.2e}"
            ok = bit_exact or diff < 1e-2
            log(f"    數值 vs ref : {verdict}  {'PASS' if ok else 'FAIL'}")
            log(f"    wall/layer  : median {median*1e3:.2f} ms  (best {best*1e3:.2f})")
            log(f"    ×48 層估計  : {median*48*1e3:.1f} ms / merge cycle")
            log(f"    GPU 暫存峰值: {peak:.1f} MB")
            log(f"    PCIe H2D    : {pcie_mb:.1f} MB ({pcie_experts} expert)")
            if not ok:
                _failures.append(name.strip())
            rows.append((name, median, peak, pcie_mb, verdict))
        except Exception:
            log(f"    FAIL")
            for ln in traceback.format_exc().rstrip().splitlines():
                log(f"    {ln}")
            _failures.append(name.strip())

    # 對比表
    log("\n" + "=" * 68)
    log("對比表(單層 merge,M=%d)" % M)
    log(f"  {'variant':<16} {'wall(ms)':>10} {'peak(MB)':>10} {'PCIe(MB)':>10}  數值")
    for name, median, peak, pcie_mb, verdict in rows:
        log(f"  {name:<16} {median*1e3:>10.2f} {peak:>10.1f} {pcie_mb:>10.1f}  {verdict}")

    log("\n" + "=" * 68)
    if _failures:
        log(f"RESULT: {len(_failures)} FAILED — {', '.join(_failures)}")
    else:
        log("RESULT: ALL PASS（三變體數值皆對齊 reference）")
    log("=" * 68)
    return 1 if _failures else 0


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
            f.write("\n".join(_report_lines) + "\n")
        print(f"\nreport → {OUT_PATH}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
