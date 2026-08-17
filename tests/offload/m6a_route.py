"""M6a — offload_plan.md:M3 的 route_offload 搬進 qwen3 adapter 後重驗。

M3 用 standalone 函式證明 dispatch 路徑數值正確;M6a 確認搬進
`Qwen3MoeAdapter._route_offload` + `_standard_routing` 分流後,行為不變。
(等同 M3 R1/R4,但走 adapter method。)

檢查項目:
  A1  _is_offload_block(offload block) == True;adapter._route_offload
      輸出與原生 Qwen3MoEBlock.forward 一致(M3 R1 重跑,走 adapter)
  A2  經 _standard_routing 分流(verify 用法,gate_logits 未動)輸出
      與原生一致 —— 確認分流接線正確
  A3  masked gate(只留一顆 expert)經 adapter 路徑跑得動、輸出有限非零
      (M3 R4 重跑,M6b random_mask draft phase 的前提)

Usage:
    sbatch tests/offload/m6a_route.sh
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(REPO_ROOT, "tests", "offload", "m6a_route.out")

_report_lines: list[str] = []
_failures: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _report_lines.append(msg)


def check(tag: str, desc: str):
    def deco(fn):
        def run(*args, **kwargs):
            log(f"\n[{tag}] {desc}")
            try:
                fn(*args, **kwargs)
                log(f"[{tag}] PASS")
                return True
            except Exception:
                log(f"[{tag}] FAIL")
                for ln in traceback.format_exc().rstrip().splitlines():
                    log(f"    {ln}")
                _failures.append(tag)
                return False
        return run
    return deco


def _diff(a, b):
    return float((a.float() - b.float()).abs().max())


def _inputs(model, layer_idx, seq_len):
    import torch
    block = model.model.layers[layer_idx].mlp
    hidden = model.config.hidden_size
    hs = torch.randn(1, seq_len, hidden, dtype=torch.bfloat16, device="cuda:0")
    hs_flat = hs.view(-1, hidden)
    gate_logits = block.gate(hs_flat)
    return block, hs, hs_flat, gate_logits, hidden


@check("A1", "_is_offload_block True;_route_offload == 原生 forward")
def a1_route(adapter, model, layer_idx, seq_len):
    import torch
    from aug_spec.adapters.qwen3 import _is_offload_block

    block, hs, hs_flat, gate_logits, hidden = _inputs(model, layer_idx, seq_len)
    assert _is_offload_block(block), "offload block 沒被辨識"

    ref, _ = block(hs)
    ref = ref.view(seq_len, hidden)
    mine = adapter._route_offload(block, hs_flat, gate_logits).view(seq_len, hidden)
    d = _diff(ref, mine)
    log(f"    max|ref - route_offload| : {d:.3e}")
    assert torch.equal(ref, mine) or d < 1e-2, f"差異 {d:.3e} 過大"
    log("    逐位元相同 ✓" if torch.equal(ref, mine) else "    < 1e-2(dispatch ulp)")


@check("A2", "經 _standard_routing 分流(verify 用法)== 原生 forward")
def a2_standard_routing(adapter, model, layer_idx, seq_len):
    import torch

    block, hs, hs_flat, gate_logits, hidden = _inputs(model, layer_idx, seq_len)
    ref, _ = block(hs)
    # _standard_routing 內部會 _is_offload_block 分流到 _route_offload
    mine = adapter._standard_routing(block, hs_flat, gate_logits, 1, seq_len, hidden)
    mine = mine.view(seq_len, hidden)
    d = _diff(ref.view(seq_len, hidden), mine)
    log(f"    max|ref - standard_routing| : {d:.3e}")
    assert torch.equal(ref.view(seq_len, hidden), mine) or d < 1e-2, \
        f"分流後差異 {d:.3e} 過大"


@check("A3", "masked gate(只留一顆 expert)經 adapter 跑得動且有限非零")
def a3_masked(adapter, model, layer_idx, seq_len):
    import torch

    block, hs, hs_flat, gate_logits, hidden = _inputs(model, layer_idx, seq_len)
    e = 7
    masked = torch.full_like(gate_logits, float("-inf"))
    masked[:, e] = gate_logits[:, e]
    out = adapter._route_offload(block, hs_flat, masked).view(seq_len, hidden)
    log(f"    output : shape={tuple(out.shape)} dtype={out.dtype}")
    assert torch.isfinite(out).all(), "輸出含 NaN/Inf"
    assert float(out.float().abs().sum()) > 0, "輸出全零"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--offload-dir",
                   default=os.path.join(REPO_ROOT, "moe_infinity",
                                        "offload_output", "Qwen3-30B-A3B"))
    p.add_argument("--device-memory-ratio", type=float, default=0.75)
    p.add_argument("--layer-idx", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=4)
    args = p.parse_args()

    import torch
    from aug_spec.adapters.qwen3 import Qwen3MoeAdapter
    from aug_spec.runtime.loader import load_offload

    log("=" * 68)
    log("M6a probe — route_offload 搬進 adapter 後重驗(Qwen3)")
    log("=" * 68)

    log("\ncalling load_offload(...) ...")
    model, tokenizer, moe, _ = load_offload(
        args.model, args.offload_dir,
        device_memory_ratio=args.device_memory_ratio, load_cpu_source=False)

    # 暖機(engine seq_id / dispatch 狀態就緒,同 M3)
    ids = tokenizer("The capital of France is", return_tensors="pt"
                    ).input_ids.to("cuda:0")
    moe._configure_hook(ids)
    with torch.no_grad():
        moe.generate(ids, max_new_tokens=8, do_sample=False,
                     pad_token_id=tokenizer.eos_token_id)

    adapter = Qwen3MoeAdapter()
    a1_route(adapter, model, args.layer_idx, args.seq_len)
    a2_standard_routing(adapter, model, args.layer_idx, args.seq_len)
    a3_masked(adapter, model, args.layer_idx, args.seq_len)

    log("\n" + "=" * 68)
    log(f"RESULT: {len(_failures)} FAILED — {', '.join(_failures)}"
        if _failures else "RESULT: ALL PASS")
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
