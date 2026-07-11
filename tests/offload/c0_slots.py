"""C0 — merged-slot 機制功能驗證(merged_cache_plan.md C0;不接 policy)。

直接對 dispatcher 呼叫 slot API,驗證(D0 預配版,c3_pipeline_plan.md):
  1. init_merged_slots 建表並**預配全部 buffer**(重複 init 為 no-op);
     未寫入的 slot handle 即有效(3 個 CUDA zero tensor)——D2 plan 前移
     的前提。
  2. merge_experts_to_slot 的內容 == 同輸入的 merge_experts_local
     (共用 MergeAccumulate,應 bit-exact)。
  3. 同 slot rebuild(copy_ 路徑)換成員後內容正確。
  4. get_merged_slot:越界 = [];未寫入 slot = 有效 zero handle。
  5. discard_merged_slot 已刪除(無 discard 路徑);set_merged_slot_pinned
     可呼叫。
  6. 越界 merge_experts_to_slot 回 False。

用法(sbatch 內):.venv/bin/python tests/offload/c0_slots.py
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
    log("C0 — merged-slot 機制功能驗證")
    log("=" * 70)

    model, tokenizer, moe, cpu_source = load_offload(
        model_id, offload_dir, device_memory_ratio=0.2, load_cpu_source=False)
    adapter = Qwen3MoeAdapter()
    blocks = list(adapter.iter_moe(model))
    disp = blocks[0][1].expert_executor.expert_dispatcher
    L = len(blocks)
    li = blocks[0][0]
    log(f"  MoE layers={L}  first layer_idx={li}")

    def eq(a, b):
        return (len(a) == len(b)
                and all(torch.equal(x, y) for x, y in zip(a, b)))

    # 1. init(+ 重複 init no-op)+ D0:未寫入 handle 即有效且為零
    disp.init_merged_slots(L, 4)
    disp.init_merged_slots(L, 4)          # 應印 WARN、不改表
    pre = disp.get_merged_slot(li, 0, 0)
    assert len(pre) == 3, f"[1] pre-write handle invalid: {len(pre)} tensors"
    assert all(t.is_cuda for t in pre), "[1] pre-alloc buffers not on CUDA"
    assert all((t == 0).all().item() for t in pre), "[1] pre-alloc not zeroed"
    log("  [1] init_merged_slots + pre-write handle valid (D0) ✓")

    # 2. 首次 merge-into-slot == merge_experts_local(配置路徑,冷成員 →
    #    覆蓋 MergeAccumulate 的 host->GPU 暫時讀取分支)
    ref = disp.merge_experts_local(li, [0, 1], [0.5, 0.5], 0)
    assert len(ref) == 3, f"merge_experts_local returned {len(ref)} tensors"
    ok = disp.merge_experts_to_slot(li, 0, [0, 1], [0.5, 0.5], 0)
    got = disp.get_merged_slot(li, 0, 0)
    assert ok and eq(ref, got), "[2] slot content != merge_experts_local"
    log("  [2] merge_experts_to_slot(alloc) == merge_experts_local ✓")

    # 3. rebuild(copy_ 路徑):同 slot 換成員
    ref2 = disp.merge_experts_local(li, [2, 3], [0.25, 0.75], 0)
    ok = disp.merge_experts_to_slot(li, 0, [2, 3], [0.25, 0.75], 0)
    got2 = disp.get_merged_slot(li, 0, 0)
    assert ok and eq(ref2, got2), "[3] rebuild content mismatch"
    log("  [3] rebuild (copy_ path) ✓")

    # 4. 越界 = miss;未寫入 slot = 有效 zero handle(D0 契約)
    assert disp.get_merged_slot(li, 99, 0) == [], "[4] OOB slot not miss"
    assert disp.get_merged_slot(9999, 0, 0) == [], "[4] OOB layer not miss"
    unwritten = disp.get_merged_slot(li, 1, 0)
    assert len(unwritten) == 3 and all(
        (t == 0).all().item() for t in unwritten), "[4] unwritten != zeros"
    log("  [4] OOB miss + unwritten-slot zero handle ✓")

    # 5. pin API 可呼叫;discard 已刪除(D0:無 discard 路徑)
    disp.set_merged_slot_pinned(li, [0], 0)
    assert not hasattr(disp, "discard_merged_slot"), \
        "[5] discard_merged_slot should be deleted"
    log("  [5] set_merged_slot_pinned ✓ / discard 已刪 ✓")

    # 6. 越界 merge 回 False
    assert disp.merge_experts_to_slot(li, 99, [0], [1.0], 0) is False
    assert disp.merge_experts_to_slot(9999, 0, [0], [1.0], 0) is False
    log("  [6] OOB merge returns False ✓")

    log("C0 probe OK")
    return 0


if __name__ == "__main__":
    rc = main()
    # moe_infinity's C++ worker threads hang interpreter shutdown (blocked in
    # ThreadSafeQueue::Pop) — force-exit after the verdict is printed, same as
    # cli.py's offload path.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
