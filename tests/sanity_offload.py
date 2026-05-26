"""Phase 0 pre-flight for the offload backend.

Answers the four questions in IMPLEMENTATION_PLAN.md §3.2:

  Q1. After `moe_infinity.MoE(ckpt, cfg)` loads, does
      `model.model.layers[i].block_sparse_moe` still resolve, and is
      the block type now `SyncMixtralSparseMoeBlock` with `.gate`,
      `.experts`, `.expert_executor` attributes?

  Q2. Can a CPU-resident weight source (`AutoModelForCausalLM
      .from_pretrained(model_id, device_map="cpu", torch_dtype=bf16)`)
      coexist in the same process as the MoE wrapper, and are its
      expert weights real (non-zero, correct shape, on CPU)?

  Q3. In the offloaded model, is `block.experts[e].w1.weight` a
      shape-(1,) zero placeholder (which means we cannot read it
      directly during merge, and must source from the CPU copy)?

  Q4. Does Mixtral-8x7B fit on a single H100-80GB at
      `device_memory_ratio=0.15` (non-expert + cache budget combined)?

Run from the repo root:

    source .venv/bin/activate
    export HF_HOME=/work/morrisliu07/.cache/huggingface
    python tests/sanity_offload.py

Or under SLURM:

    sbatch scripts/run.sh    # won't work — wrapper expects an aug_spec config.
    # Use sanity_offload directly:
    sbatch --partition=normal2 --account=MST114471 --time=00:30:00 \
        --gpus-per-node=1 --cpus-per-task=4 --wrap \
        "source /work/morrisliu07/aug_spec/.venv/bin/activate && \
         export HF_HOME=/work/morrisliu07/.cache/huggingface && \
         python /work/morrisliu07/aug_spec/tests/sanity_offload.py"

The script writes a machine-readable verdict to
`tests/sanity_offload_results.json` so subsequent phases can grep it.
"""

from __future__ import annotations

import json
import os
import resource
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import torch

# Make sure HF_HOME is honoured before any HF import.
HF_HOME = os.environ.get("HF_HOME", "/work/morrisliu07/.cache/huggingface")
os.environ["HF_HOME"] = HF_HOME

CKPT = "mistralai/Mixtral-8x7B-v0.1"
OFFLOAD_DIR = "/work/morrisliu07/aug_spec/cache/offload/mixtral"
DEVICE_MEMORY_RATIO = 0.15
RESULTS_PATH = Path(__file__).parent / "sanity_offload_results.json"


# =============================================================================
# Helpers
# =============================================================================

def banner(s: str) -> None:
    print("\n" + "=" * 72, flush=True)
    print(f"  {s}", flush=True)
    print("=" * 72, flush=True)


def host_ram_used_gb() -> float:
    """RSS of this process, in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    peaks = []
    for i in range(torch.cuda.device_count()):
        try:
            peaks.append(torch.cuda.max_memory_allocated(i) / (1024**3))
        except RuntimeError:
            continue
    return max(peaks) if peaks else 0.0


def reset_vram_peak() -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(i)
            except RuntimeError:
                pass


# =============================================================================
# Load order: CPU source FIRST (untouched by moe_infinity), then MoE wrapper.
# =============================================================================

def load_cpu_source() -> Any:
    """Load a regular HF Mixtral on CPU as the weight source for merging.

    Done BEFORE MoE(...) so moe_infinity's empty-init / hook
    machinery cannot intercept this load.
    """
    from transformers import AutoModelForCausalLM

    print(f"  Loading CPU-resident source: {CKPT} (bf16, device_map=cpu) ...")
    t0 = time.perf_counter()
    cpu_source = AutoModelForCausalLM.from_pretrained(
        CKPT,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,  # Mixtral is in core transformers
    )
    cpu_source.eval()
    for p in cpu_source.parameters():
        p.requires_grad_(False)
    print(f"    done in {time.perf_counter() - t0:.1f}s, "
          f"host RAM ~ {host_ram_used_gb():.1f} GB")
    return cpu_source


def load_moe_wrapper() -> Any:
    """Load the offloaded MoE wrapper."""
    from moe_infinity import MoE

    Path(OFFLOAD_DIR).mkdir(parents=True, exist_ok=True)
    print(f"  Loading offloaded MoE wrapper: offload_path={OFFLOAD_DIR}, "
          f"device_memory_ratio={DEVICE_MEMORY_RATIO}")
    t0 = time.perf_counter()
    moe = MoE(CKPT, {
        "offload_path": OFFLOAD_DIR,
        "device_memory_ratio": DEVICE_MEMORY_RATIO,
    })
    print(f"    done in {time.perf_counter() - t0:.1f}s")
    return moe


# =============================================================================
# Q1
# =============================================================================

def q1_adapter_path_resolves(moe: Any) -> Dict[str, Any]:
    """Walk moe.model.model.layers[i].block_sparse_moe — same path the
    MixtralAdapter.iter_moe uses."""
    banner("Q1: adapter.iter_moe path still resolves on the offloaded model")

    hf_model = moe.model
    findings: Dict[str, Any] = {
        "hf_model_type": type(hf_model).__name__,
        "has_model_attr": hasattr(hf_model, "model"),
        "first_blocks": [],
    }

    inner = getattr(hf_model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    findings["has_model_layers"] = layers is not None
    findings["num_layers"] = len(layers) if layers is not None else 0

    if layers is None:
        findings["verdict"] = "FAIL: hf_model.model.layers does not exist"
        return findings

    for i, layer in enumerate(layers[:3]):
        block = getattr(layer, "block_sparse_moe", None)
        info = {
            "layer_idx": i,
            "block_type": type(block).__name__ if block is not None else None,
            "has_gate": hasattr(block, "gate") if block is not None else False,
            "has_experts": hasattr(block, "experts") if block is not None else False,
            "has_expert_executor": (
                hasattr(block, "expert_executor") if block is not None else False),
            "has_layer_id": hasattr(block, "layer_id") if block is not None else False,
            "num_experts": (
                len(block.experts) if block is not None
                and hasattr(block, "experts") else None),
        }
        findings["first_blocks"].append(info)

    # Verdict: at least the first block must be Sync* with .gate/.experts/.expert_executor.
    if findings["first_blocks"]:
        b0 = findings["first_blocks"][0]
        if (b0["block_type"] in ("SyncMixtralSparseMoeBlock",)
                and b0["has_gate"] and b0["has_experts"]
                and b0["has_expert_executor"]):
            findings["verdict"] = "PASS"
        else:
            findings["verdict"] = (
                f"FAIL: block_type={b0['block_type']} "
                f"gate={b0['has_gate']} experts={b0['has_experts']} "
                f"expert_executor={b0['has_expert_executor']}")
    else:
        findings["verdict"] = "FAIL: no MoE blocks found"

    for k, v in findings.items():
        if k != "first_blocks":
            print(f"  {k}: {v}")
    print("  first_blocks:")
    for info in findings["first_blocks"]:
        print(f"    {info}")
    print(f"  >>> {findings['verdict']}")
    return findings


# =============================================================================
# Q2
# =============================================================================

def q2_cpu_source_coexists(cpu_source: Any) -> Dict[str, Any]:
    """The CPU-resident model has real expert weights, on CPU, non-zero."""
    banner("Q2: CPU-resident weight source has real expert weights")

    findings: Dict[str, Any] = {
        "cpu_source_type": type(cpu_source).__name__,
        "host_ram_gb": host_ram_used_gb(),
    }
    try:
        block0 = cpu_source.model.layers[0].block_sparse_moe
        w1 = block0.experts[0].w1.weight
        findings["block_type"] = type(block0).__name__
        findings["num_experts"] = len(block0.experts)
        findings["expert0_w1_shape"] = list(w1.shape)
        findings["expert0_w1_dtype"] = str(w1.dtype)
        findings["expert0_w1_device"] = str(w1.device)
        findings["expert0_w1_abs_sum"] = float(w1.abs().sum().item())
        findings["expert0_w1_isnan"] = bool(torch.isnan(w1).any().item())

        # Must be: real shape (intermediate, hidden), bf16, on CPU, non-zero, no nans.
        ok_shape = len(w1.shape) == 2 and w1.shape[0] > 1024
        ok_device = w1.device.type == "cpu"
        ok_dtype = w1.dtype == torch.bfloat16
        ok_nonzero = findings["expert0_w1_abs_sum"] > 0
        ok_clean = not findings["expert0_w1_isnan"]
        if ok_shape and ok_device and ok_dtype and ok_nonzero and ok_clean:
            findings["verdict"] = "PASS"
        else:
            findings["verdict"] = (
                f"FAIL: shape={w1.shape} device={w1.device} dtype={w1.dtype} "
                f"|w|={findings['expert0_w1_abs_sum']:.2f} "
                f"nan={findings['expert0_w1_isnan']}")
    except Exception as exc:
        findings["verdict"] = f"FAIL: {type(exc).__name__}: {exc}"
        traceback.print_exc()

    for k, v in findings.items():
        print(f"  {k}: {v}")
    print(f"  >>> {findings['verdict']}")
    return findings


# =============================================================================
# Q3
# =============================================================================

def q3_offloaded_weight_is_placeholder(moe: Any) -> Dict[str, Any]:
    """Inside the MoE wrapper, expert weight tensors are shape-(1,) zeros."""
    banner("Q3: offloaded moe.model expert weights are shape-(1,) placeholders")

    findings: Dict[str, Any] = {}
    try:
        block0 = moe.model.model.layers[0].block_sparse_moe
        w1 = block0.experts[0].w1.weight
        findings["expert0_w1_shape"] = list(w1.shape)
        findings["expert0_w1_dtype"] = str(w1.dtype)
        findings["expert0_w1_device"] = str(w1.device)
        findings["expert0_w1_abs_sum"] = float(w1.abs().sum().item())

        # Expectation from model_offload.py:213-221: shape (1,), zeros.
        is_placeholder = (
            list(w1.shape) == [1]
            and findings["expert0_w1_abs_sum"] == 0.0
        )
        findings["is_placeholder"] = is_placeholder
        findings["verdict"] = (
            "PASS (placeholder as expected — direct reads useless for merge)"
            if is_placeholder
            else f"UNEXPECTED: shape={w1.shape} |w|={findings['expert0_w1_abs_sum']:.4f}"
        )
    except Exception as exc:
        findings["verdict"] = f"FAIL: {type(exc).__name__}: {exc}"
        traceback.print_exc()

    for k, v in findings.items():
        print(f"  {k}: {v}")
    print(f"  >>> {findings['verdict']}")
    return findings


# =============================================================================
# Q4
# =============================================================================

def q4_vram_fits_single_h100(moe: Any) -> Dict[str, Any]:
    """Run a tiny forward through the offloaded model and report peak VRAM."""
    banner("Q4: VRAM peak for non-expert + archer cache on single H100")

    findings: Dict[str, Any] = {
        "device_count": torch.cuda.device_count(),
        "device_memory_ratio": DEVICE_MEMORY_RATIO,
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            findings[f"gpu_{i}"] = (
                f"{props.name} {props.total_memory / 1024**3:.1f} GB")

    reset_vram_peak()

    try:
        device = "cuda:0"
        toks = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]],
                            dtype=torch.long, device=device)
        moe._configure_hook(toks)

        # NOTE: use no_grad, NOT inference_mode. moe_infinity's C++
        # ExpertDispatcher does in-place index_add_ on the output, which
        # crashes if the tensor was created under inference_mode.
        with torch.no_grad():
            t0 = time.perf_counter()
            out = moe.model(toks)
            wall = time.perf_counter() - t0
        findings["forward_wall_s"] = round(wall, 3)
        findings["output_shape"] = list(out.logits.shape) if hasattr(out, "logits") else "no-logits"
        findings["peak_vram_gb"] = round(peak_vram_gb(), 2)

        # Pass criterion: ≤ 50 GB on an 80 GB H100 at device_memory_ratio=0.15.
        if findings["peak_vram_gb"] <= 50.0:
            findings["verdict"] = (
                f"PASS — peak {findings['peak_vram_gb']:.1f} GB / 80 GB H100")
        else:
            findings["verdict"] = (
                f"WARN — peak {findings['peak_vram_gb']:.1f} GB exceeds 50 GB "
                "soft target; may need --gpus-per-node=2 for production runs")
    except Exception as exc:
        findings["verdict"] = f"FAIL: {type(exc).__name__}: {exc}"
        traceback.print_exc()

    for k, v in findings.items():
        print(f"  {k}: {v}")
    print(f"  >>> {findings['verdict']}")
    return findings


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    banner("PHASE 0 — sanity_offload.py")
    print(f"  CKPT      = {CKPT}")
    print(f"  HF_HOME   = {HF_HOME}")
    print(f"  OFFLOAD   = {OFFLOAD_DIR}")
    print(f"  RATIO     = {DEVICE_MEMORY_RATIO}")
    print(f"  GPUs      = {torch.cuda.device_count()}")

    results: Dict[str, Any] = {
        "ckpt": CKPT,
        "offload_dir": OFFLOAD_DIR,
        "device_memory_ratio": DEVICE_MEMORY_RATIO,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Step 1: load CPU source FIRST.
    banner("Loading CPU-resident source (before any moe_infinity activity)")
    try:
        cpu_source = load_cpu_source()
    except Exception as exc:
        traceback.print_exc()
        results["fatal"] = f"load_cpu_source: {type(exc).__name__}: {exc}"
        RESULTS_PATH.write_text(json.dumps(results, indent=2))
        return 1

    # Step 2: load MoE wrapper.
    banner("Loading offloaded MoE wrapper")
    try:
        moe = load_moe_wrapper()
    except Exception as exc:
        traceback.print_exc()
        results["fatal"] = f"load_moe_wrapper: {type(exc).__name__}: {exc}"
        RESULTS_PATH.write_text(json.dumps(results, indent=2))
        return 1

    # Step 3–6: Q1–Q4.
    results["q1"] = q1_adapter_path_resolves(moe)
    results["q2"] = q2_cpu_source_coexists(cpu_source)
    results["q3"] = q3_offloaded_weight_is_placeholder(moe)
    results["q4"] = q4_vram_fits_single_h100(moe)

    # Final summary.
    banner("SUMMARY")
    all_pass = True
    for k in ("q1", "q2", "q3", "q4"):
        v = results[k]["verdict"]
        ok = v.startswith("PASS")
        all_pass = all_pass and ok
        print(f"  {k}: {v}")
    results["all_pass"] = all_pass
    results["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n  Results written to {RESULTS_PATH}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    _rc = main()
    # moe_infinity's C++ background threads don't shut down cleanly on
    # Python exit — process hangs until SLURM kills it. Force exit
    # after results have been written to disk.
    import sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(_rc)
