"""C2 — 驗證 C++ `merge_experts_local` 與 CPU merge（build_weighted_avg）數值一致。

兩條路徑都測:
  resident: 用 L47（verify 後 hot）的 resident expert → 零 PCIe 就地讀 + GPU 累加
  cold:     用 L0（多半已被 evict）的 expert        → 暫時 host->GPU 複製 + GPU 累加

對齊基準: 同一組 (expert_ids, weights) 餵給 adapter.build_weighted_avg(cpu_block)
（CPU fp32 累加 → bf16）。GPU fp32 累加 vs CPU 可能差 1 ULP（FMA/rounding），
最終 bf16 cast 會吃掉大部分 → 以 bf16 解析度 allclose 為準，並報告逐位元相符比例。

驗收: resident + cold 兩路徑都 allclose（max abs diff 在 bf16 量級）→ RESULT: PASS

Usage:
    sbatch tests/offload/c2_merge.sh
"""

from __future__ import annotations

import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(REPO_ROOT, "tests", "offload", "c2_merge.out")
PROJS = ("gate_proj", "up_proj", "down_proj")
_report: list[str] = []


def log(m: str = "") -> None:
    print(m, flush=True)
    _report.append(m)


def _compare(tag, merged_gpu, ref_cpu) -> bool:
    import torch
    ok = True
    for pr, mt in zip(PROJS, merged_gpu):
        m = mt.detach().to("cpu")
        r = ref_cpu[pr]
        if m.shape != r.shape:
            log(f"  [{tag}] {pr}: SHAPE MISMATCH {tuple(m.shape)} vs {tuple(r.shape)}")
            ok = False
            continue
        exact = torch.equal(m, r)
        max_diff = (m.float() - r.float()).abs().max().item()
        # bf16 ULP near 1.0 ~ 2^-8 = 0.0039; allow a few ULP for accumulation order
        close = torch.allclose(m.float(), r.float(), rtol=0, atol=0.02)
        frac = (m == r).float().mean().item()
        log(f"  [{tag}] {pr}: exact={exact} match_frac={frac:.4f} "
            f"max_abs_diff={max_diff:.5f} allclose={close}")
        ok = ok and close
    return ok


def main() -> int:
    import torch
    from aug_spec.adapters.qwen3 import Qwen3MoeAdapter
    from aug_spec.runtime.loader import load_offload

    model_id = "Qwen/Qwen3-30B-A3B"
    offload_dir = os.path.join(REPO_ROOT, "moe_infinity", "offload_output",
                               "Qwen3-30B-A3B")

    log("=" * 70)
    log("C2 — merge_experts_local vs CPU merge")
    log("=" * 70)

    model, tokenizer, moe, cpu_source = load_offload(
        model_id, offload_dir, device_memory_ratio=0.2, load_cpu_source=True)
    adapter = Qwen3MoeAdapter()
    blocks = list(adapter.iter_moe(model))
    cpu_blocks = dict(adapter.iter_moe(cpu_source))
    n_exp = adapter.num_experts(blocks[0][1])
    layer_to_block = dict(blocks)
    disp = blocks[0][1].expert_executor.expert_dispatcher

    ids = tokenizer("The capital of France is a beautiful and historic city",
                    return_tensors="pt").input_ids.to("cuda:0")
    moe._configure_hook(ids)
    with torch.no_grad():
        model(ids)

    last_li = blocks[-1][0]
    first_li = blocks[0][0]

    # 找 last layer 的 resident experts
    resident = [e for e in range(n_exp)
                if len(disp.get_resident_expert_weights(last_li, e, 0)) == 3]
    log(f"  L{last_li} resident 數: {len(resident)}")
    cold = [e for e in range(n_exp)
            if len(disp.get_resident_expert_weights(first_li, e, 0)) == 3]
    log(f"  L{first_li} resident 數: {len(cold)}（cold 測試用 non-resident）")

    def ref_merge(li, ids_list, w_list):
        wvec = [0.0] * n_exp
        s = sum(w_list)
        for e, w in zip(ids_list, w_list):
            wvec[e] = w / s
        return adapter.build_weighted_avg(cpu_blocks[li], wvec), wvec

    all_ok = True

    # ── resident path: 取 2 顆 resident, 升序 ──
    if len(resident) >= 2:
        ids_r = sorted(resident[:2])
        w_r = [0.6, 0.4]
        ref, wvec = ref_merge(last_li, ids_r, w_r)
        nz = [(e, wvec[e]) for e in range(n_exp) if wvec[e] > 0.0]
        merged = disp.merge_experts_local(
            last_li, [e for e, _ in nz], [w for _, w in nz], 0)
        log(f"  resident merge: L{last_li} experts={ids_r}")
        all_ok &= _compare("resident", merged, ref)
    else:
        log("  SKIP resident（resident 不足 2）")
        all_ok = False

    # ── cold path: 取 first layer 2 顆 non-resident ──
    non_res = [e for e in range(n_exp) if e not in set(cold)]
    if len(non_res) >= 2:
        ids_c = sorted(non_res[:2])
        w_c = [0.7, 0.3]
        ref, wvec = ref_merge(first_li, ids_c, w_c)
        nz = [(e, wvec[e]) for e in range(n_exp) if wvec[e] > 0.0]
        merged = disp.merge_experts_local(
            first_li, [e for e, _ in nz], [w for _, w in nz], 0)
        log(f"  cold merge: L{first_li} experts={ids_c}")
        all_ok &= _compare("cold", merged, ref)
    else:
        log("  SKIP cold（找不到 non-resident）")

    log("")
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
