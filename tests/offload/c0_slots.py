"""C0 — merged-slot 機制功能驗證(merged_cache_plan.md C0;不接 policy)。

直接對 dispatcher 呼叫五個新 API,驗證:
  1. init_merged_slots 建表(重複 init 為 no-op)。
  2. merge_experts_to_slot(首次 = 配置路徑)的內容 == 同輸入的
     merge_experts_local(共用 MergeAccumulate,應 bit-exact)。
  3. 同 slot rebuild(copy_ 路徑)換成員後內容正確。
  4. get_merged_slot:空 slot / 越界 = [](probe miss)。
  5. discard_merged_slot 後 get = [];set_merged_slot_pinned 可呼叫。
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

    # 1. init(+ 重複 init no-op)
    disp.init_merged_slots(L, 4)
    disp.init_merged_slots(L, 4)          # 應印 WARN、不改表
    log("  [1] init_merged_slots OK")

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

    # 4. probe miss:空 slot 與越界
    assert disp.get_merged_slot(li, 1, 0) == [], "[4] empty slot not miss"
    assert disp.get_merged_slot(li, 99, 0) == [], "[4] OOB slot not miss"
    assert disp.get_merged_slot(9999, 0, 0) == [], "[4] OOB layer not miss"
    log("  [4] probe miss (empty / out-of-bounds) ✓")

    # 5. pin API + discard → miss
    disp.set_merged_slot_pinned(li, [0], 0)
    disp.discard_merged_slot(li, 0, 0)
    assert disp.get_merged_slot(li, 0, 0) == [], "[5] discard didn't clear"
    log("  [5] set_merged_slot_pinned / discard_merged_slot ✓")

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
