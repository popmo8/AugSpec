"""C1 — 驗證新增的 C++ binding `get_resident_expert_weights` 能正確讀到
verify 剛 fetch 上 GPU 的 expert 權重，且與 cpu_source 逐位元相符。

流程:
  1. load_offload + cpu_source
  2. configure_hook + 一次完整 verify forward（讓 dispatcher cache 填滿 resident expert）
  3. 從最後一層往前掃，找出哪些 (layer, expert) 目前 resident
  4. 對前幾顆 resident expert，把回傳的 3 個 GPU tensor 與 cpu_source 的
     {gate_proj, up_proj, down_proj} 比對（順序自動判定），要求 torch.equal

驗收: 找得到 resident expert 且全部逐位元相符 → RESULT: PASS

Usage:
    sbatch tests/offload/c1_resident.sh
"""

from __future__ import annotations

import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(REPO_ROOT, "tests", "offload", "c1_resident.out")
PROJS = ("gate_proj", "up_proj", "down_proj")
_report: list[str] = []


def log(m: str = "") -> None:
    print(m, flush=True)
    _report.append(m)


def main() -> int:
    import torch
    from aug_spec.adapters.qwen3 import Qwen3MoeAdapter
    from aug_spec.runtime.loader import load_offload

    model_id = "Qwen/Qwen3-30B-A3B"
    offload_dir = os.path.join(REPO_ROOT, "moe_infinity", "offload_output",
                               "Qwen3-30B-A3B")

    log("=" * 70)
    log("C1 — get_resident_expert_weights 正確性驗證")
    log("=" * 70)

    model, tokenizer, moe, cpu_source = load_offload(
        model_id, offload_dir, device_memory_ratio=0.2, load_cpu_source=True)
    adapter = Qwen3MoeAdapter()
    blocks = list(adapter.iter_moe(model))
    cpu_blocks = dict(adapter.iter_moe(cpu_source))
    n_exp = adapter.num_experts(blocks[0][1])
    log(f"  MoE layers={len(blocks)}  experts/layer={n_exp}")

    # 取 dispatcher handle（任一 block 共用同一個）
    disp = blocks[0][1].expert_executor.expert_dispatcher

    # verify forward 填 cache
    ids = tokenizer("The capital of France is a beautiful and historic city",
                    return_tensors="pt").input_ids.to("cuda:0")
    moe._configure_hook(ids)
    with torch.no_grad():
        model(ids)

    # 從最後一層往前掃，找 resident expert（最近 fetch 的最可能還在 cache）
    resident: list[tuple[int, int]] = []
    for li, _ in reversed(blocks):
        for e in range(n_exp):
            w = disp.get_resident_expert_weights(li, e, 0)
            if len(w) == 3:
                resident.append((li, e))
        if len(resident) >= 8:
            break

    log(f"  掃到 resident expert 數: {len(resident)}（取樣前 8 顆驗證）")
    if not resident:
        log("  FAIL: 沒有任何 resident expert（cache 太小或時機不對）")
        log("RESULT: FAIL")
        return 1

    all_ok = True
    inferred_order = None
    for (li, e) in resident[:8]:
        gpu_w = disp.get_resident_expert_weights(li, e, 0)
        cpu_e = cpu_blocks[li].experts[e]
        refs = {pr: getattr(cpu_e, pr).weight for pr in PROJS}

        # 對每個回傳 tensor，找出哪個 proj 逐位元相符
        order = []
        for t in gpu_w:
            t_cpu = t.detach().to("cpu")
            match = None
            for pr, ref in refs.items():
                if t_cpu.shape == ref.shape and torch.equal(t_cpu, ref):
                    match = pr
                    break
            order.append(match)
        ok = all(o is not None for o in order) and len(set(order)) == 3
        if not ok:
            all_ok = False
            log(f"  L{li} E{e}: MISMATCH order={order} "
                f"shapes={[tuple(t.shape) for t in gpu_w]}")
        else:
            if inferred_order is None:
                inferred_order = order
            log(f"  L{li} E{e}: OK  order={order}")

    log("")
    log(f"  推定 tensor_ids 順序 = {inferred_order}")
    log("RESULT: " + ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


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
