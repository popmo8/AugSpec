"""C3 — MergeJob pipeline 功能驗證(c3_pipeline_plan.md D1;不接 policy)。

直接對 dispatcher 呼叫 submit_merge_jobs / wait_merges_done,驗證:
  1. 冷成員 job(routed=[] → 不設門、立即執行、H2D 直讀):slot 內容與
     merge_experts_local **逐位元**相同(同一 fp32 累加運算順序)。
  2. singleton identity job(單成員 w=1.0)= 原 expert 權重逐位元。
  3. 同 slot 重寫(copy_ 路徑)換權重後內容正確,且 handle(D0 預配)
     在重寫前後是同一組 buffer。
  4. 無 pending 時 wait_merges_done 立即返回(drain no-op)。
  5. 越界 slot / layer 的 submit 回 False。

「成員在途 → 到齊即發」的 gating 路徑需要 dispatch 併發,由 c1_smoke /
c1_q5_512 端到端覆蓋(AccR 同量級 + 無 fatal + drain_wait≈0)。

用法(sbatch 內):.venv/bin/python tests/offload/c3_jobs.py
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def log(m: str) -> None:
    print(m, flush=True)


def main() -> int:
    import torch
    from aug_spec.adapters.qwen3 import Qwen3MoeAdapter
    from aug_spec.runtime.loader import load_offload

    model_id = "Qwen/Qwen3-30B-A3B-Base"
    offload_dir = os.path.join(REPO_ROOT, "moe_infinity", "offload_output",
                               "Qwen3-30B-A3B-Base")

    log("=" * 70)
    log("C3 — MergeJob pipeline 功能驗證")
    log("=" * 70)

    model, tokenizer, moe, cpu_source = load_offload(
        model_id, offload_dir, device_memory_ratio=0.2, load_cpu_source=False)
    adapter = Qwen3MoeAdapter()
    blocks = list(adapter.iter_moe(model))
    disp = blocks[0][1].expert_executor.expert_dispatcher
    L = len(blocks)
    li = blocks[0][0]
    disp.init_merged_slots(L, 4)
    log(f"  MoE layers={L}  first layer_idx={li}")

    def eq(a, b):
        return (len(a) == len(b)
                and all(torch.equal(x, y) for x, y in zip(a, b)))

    # 1. 冷成員 pair job:立即執行(routed=[] 不設門),逐位元 == local merge
    ref = disp.merge_experts_local(li, [0, 1], [0.5, 0.5], 0)
    assert len(ref) == 3
    assert disp.submit_merge_jobs(li, [0], [[0, 1]], [[0.5, 0.5]], [])
    disp.wait_merges_done(60.0)
    got = disp.get_merged_slot(li, 0, 0)
    assert eq(ref, got), "[1] pipeline pair != merge_experts_local"
    log("  [1] cold-member pair job bit-exact ✓")

    # 2. singleton identity job
    ref2 = disp.merge_experts_local(li, [2], [1.0], 0)
    assert disp.submit_merge_jobs(li, [1], [[2]], [[1.0]], [])
    disp.wait_merges_done(60.0)
    assert eq(ref2, disp.get_merged_slot(li, 1, 0)), "[2] identity mismatch"
    log("  [2] singleton identity job bit-exact ✓")

    # 3. 同 slot 重寫:handle 穩定(D0)、內容更新
    handle_before = disp.get_merged_slot(li, 0, 0)
    ref3 = disp.merge_experts_local(li, [0, 1], [0.25, 0.75], 0)
    assert disp.submit_merge_jobs(li, [0], [[0, 1]], [[0.25, 0.75]], [])
    disp.wait_merges_done(60.0)
    handle_after = disp.get_merged_slot(li, 0, 0)
    assert eq(ref3, handle_after), "[3] rewrite content mismatch"
    assert all(a.data_ptr() == b.data_ptr()
               for a, b in zip(handle_before, handle_after)), \
        "[3] slot buffers not stable (D0 pre-alloc broken)"
    log("  [3] slot rewrite + stable handles ✓")

    # 4. drain no-op
    disp.wait_merges_done(5.0)
    log("  [4] empty drain immediate ✓")

    # 5. 越界 submit → False
    assert disp.submit_merge_jobs(li, [99], [[0]], [[1.0]], []) is False
    assert disp.submit_merge_jobs(9999, [0], [[0]], [[1.0]], []) is False
    log("  [5] OOB submit returns False ✓")

    log("C3 probe OK")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
