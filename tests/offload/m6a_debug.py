"""M6a debug 第四輪 — dispatch 在「正常完整 forward 序列」裡是否可靠?

前三輪:單獨抓一層、餵 random 輸入、反覆 dispatch,結果又錯又跳(從不
等於真實權重算的 ground truth)。假設:引擎用背景 thread 搬 expert 權重、
wait 沒等搬完;正常逐層跑時層間計算掩蓋了搬運,單層脫序測試則踩到競態。

關鍵推論:_route_offload ≈ 原生 forward(餵 dispatch 的 mask 數值相同),
所以這毛病不是我們引入的;真實 verify(完整序列、每層一次)應可靠。
本輪驗證這個前提:
  F1  完整 model forward 跑兩次,比 logits —— 一致 = 序列內 dispatch 可靠
  F2  generate 跑兩次,比 token(greedy argmax 對噪音較不敏感,當輔證)
  F3  完整 forward 裡抓某層 MoE 輸出兩次比 —— 序列內單層是否確定

F1 ~0 → _route_offload 安全,改測法即可;F1 大 → offload 路線有根本問題。
"""

from __future__ import annotations

import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import torch
    from aug_spec.runtime.loader import load_offload

    model, tokenizer, moe, _ = load_offload(
        "Qwen/Qwen3-30B-A3B",
        os.path.join(REPO_ROOT, "moe_infinity", "offload_output", "Qwen3-30B-A3B"),
        device_memory_ratio=0.75, load_cpu_source=False)

    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to("cuda:0")

    def diff(a, b):
        return float((a.float() - b.float()).abs().max())

    # ── F1:完整 forward 兩次比 logits ──
    with torch.no_grad():
        moe._configure_hook(ids)
        o1 = model(ids).logits.clone()
        moe._configure_hook(ids)
        o2 = model(ids).logits.clone()
    print(f"[F1] 完整 forward 兩次 logits maxdiff: {diff(o1, o2):.3e}  "
          f"(logits norm={float(o1.norm()):.1f})")

    # ── F3:完整 forward 裡抓 layer 0 MoE 輸出兩次比 ──
    captured = []

    def hook(mod, inp, out):
        captured.append(out[0].detach().clone() if isinstance(out, tuple)
                        else out.detach().clone())

    h = model.model.layers[0].mlp.register_forward_hook(hook)
    with torch.no_grad():
        moe._configure_hook(ids)
        model(ids)
        moe._configure_hook(ids)
        model(ids)
    h.remove()
    if len(captured) >= 2:
        print(f"[F3] layer0 MoE 輸出(序列內)兩次 maxdiff: "
              f"{diff(captured[0], captured[1]):.3e}")

    # ── F2:generate 兩次比 token ──
    with torch.no_grad():
        moe._configure_hook(ids)
        g1 = moe.generate(ids, max_new_tokens=16, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
        moe._configure_hook(ids)
        g2 = moe.generate(ids, max_new_tokens=16, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
    same = torch.equal(g1, g2)
    print(f"[F2] generate 兩次 token 完全相同: {same}")
    print(f"     g1: {tokenizer.decode(g1[0][ids.shape[1]:], skip_special_tokens=True)!r}")
    if not same:
        print(f"     g2: {tokenizer.decode(g2[0][ids.shape[1]:], skip_special_tokens=True)!r}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
