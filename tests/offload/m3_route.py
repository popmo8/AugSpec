"""M3 — offload_plan.md `_route_offload` 原型（搬進 adapter 前先單測）。

aug_spec 的 adapter 在 hf backend 上,target verify 走 `_standard_routing`
裡的 expert loop(`block.experts[i](state)`)。offload backend 上那些
expert 是 placeholder,verify 必須改走 `dispatch_local`。本 script 把
這條替代路徑寫成 standalone 函式並驗證數值,產出物之後原封搬進
[qwen3.py](src/aug_spec/adapters/qwen3.py)。

兩個版本(對應 offload_plan.md M3 的兩條路):
  route_offload_kernel  用 block.lib.topk_softmax(C++ kernel)組 mask
                        —— 與原生 forward 同一條路,當對照 ground truth
  route_offload_torch   自己用 torch 組 mask —— 能吃「被 draft 動過的
                        gate logits」(kernel 固定 top-k、不接受 mask,
                        masked draft 如 random_mask 用不了它)。這個才是
                        要搬進 adapter 的版本。

檢查項目:
  R1  route_offload_kernel 輸出 == 原生 Qwen3MoEBlock.forward(同一條
      dispatch 路徑,理應逐位元相同)—— 驗「把 dispatch 拆出來自己呼叫」
      沒接錯線
  R2  route_offload_torch 自組的 (router_mask, routing_weights_mask)
      與 kernel 版一致 —— 驗我們重現了 kernel 的 top-k softmax + renorm
      (含 norm_topk_prob 這個 mixtral 沒有的差異)
  R3  route_offload_torch 輸出 ≈ kernel 版(差異只該來自 torch vs kernel
      的浮點細節)
  R4  masked gate(只留一顆 expert,模擬 random_mask 的 draft phase)
      route_offload_torch 跑得動、輸出有限且非零 —— M6b 第一條 E2E 的前提

Usage:
    .venv/bin/python tests/offload/m3_route.py
    sbatch tests/offload/m3_route.sh
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(REPO_ROOT, "tests", "offload", "m3_route.out")

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


# ────────────────────────────────────────────────────────────────────
# 被測函式:offload 版的 verify routing(回傳已加權合併的 hidden states)
# ────────────────────────────────────────────────────────────────────

def _topk_softmax_masks(block, gate_logits):
    """重現 C++ lib.topk_softmax:top-k softmax + renorm,組
    (router_mask: bool, routing_weights_mask: bf16),兩者 (num_tokens,
    num_experts)。kernel 在 fp32 算 softmax/renorm、最後轉 bf16,順序
    照抄以求數值一致。kernel 固定 renorm(不看 norm_topk_prob);A3B
    norm_topk_prob=True 故一致,這裡仍依 block flag 以便其他模型沿用。"""
    import torch
    import torch.nn.functional as F

    num_tokens = gate_logits.shape[0]
    routing = F.softmax(gate_logits, dim=1, dtype=torch.float32)
    topk_w, topk_idx = torch.topk(routing, block.top_k, dim=-1)
    if block.norm_topk_prob:
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    topk_w = topk_w.to(torch.bfloat16)

    router_mask = torch.zeros(num_tokens, block.num_experts,
                              dtype=torch.bool, device=gate_logits.device)
    router_mask.scatter_(1, topk_idx, True)
    weights_mask = torch.zeros(num_tokens, block.num_experts,
                               dtype=torch.bfloat16, device=gate_logits.device)
    weights_mask.scatter_add_(1, topk_idx, topk_w)
    return router_mask, weights_mask


def _dispatch(block, hs_flat, router_mask, weights_mask):
    """共用尾段:把 mask 丟給 archer engine,回傳已加權合併的結果。"""
    block.expert_executor.dispatch_local(
        block.layer_id, hs_flat, router_mask, weights_mask)
    return block.expert_executor.wait_dispatch_local()


def route_offload_kernel(block, hs_flat, gate_logits):
    """對照組:用 C++ kernel 組 mask(與原生 forward 同路)。"""
    router_mask, weights_mask = block.lib.topk_softmax(gate_logits)
    return _dispatch(block, hs_flat, router_mask, weights_mask)


def route_offload_torch(block, hs_flat, gate_logits):
    """要搬進 adapter 的版本:自己組 mask,能吃 masked gate_logits。"""
    router_mask, weights_mask = _topk_softmax_masks(block, gate_logits)
    return _dispatch(block, hs_flat, router_mask, weights_mask)


# ────────────────────────────────────────────────────────────────────
# 檢查項
# ────────────────────────────────────────────────────────────────────

def _make_inputs(moe, layer_idx, seq_len):
    """造一個 (1, seq_len, hidden) 的 bf16 輸入,並先觸發 gate 物化拿
    gate_logits。回傳 (block, hs_flat, gate_logits)。"""
    import torch

    block = moe.model.model.layers[layer_idx].mlp
    hidden = moe.model.config.hidden_size
    hs = torch.randn(1, seq_len, hidden, dtype=torch.bfloat16, device="cuda:0")
    hs_flat = hs.view(-1, hidden)
    gate_logits = block.gate(hs_flat)   # 走 module __call__,觸發 gate 物化
    return block, hs_flat, gate_logits


def _diff(a, b):
    import torch
    return float((a.float() - b.float()).abs().max())


@check("R1", "route_offload_kernel == 原生 Qwen3MoEBlock.forward")
def r1_kernel_vs_native(moe, layer_idx, seq_len):
    import torch

    block, hs_flat, _ = _make_inputs(moe, layer_idx, seq_len)
    hidden = moe.model.config.hidden_size
    hs = hs_flat.view(1, seq_len, hidden)

    ref, _ = block(hs)                              # 原生 forward,(1,T,hid)
    ref = ref.view(seq_len, hidden)
    gate_logits = block.gate(hs_flat)
    mine = route_offload_kernel(block, hs_flat, gate_logits).view(seq_len, hidden)

    d = _diff(ref, mine)
    log(f"    max|ref - kernel|      : {d:.3e}")
    if torch.equal(ref, mine):
        log("    逐位元相同 ✓")
    else:
        # dispatch 內部浮點合併可能有 ulp 級非確定性,降級為極嚴 allclose
        assert d < 1e-2, f"差異 {d:.3e} 超出 dispatch 非確定性可解釋範圍"
        log("    非逐位元但 < 1e-2(dispatch 浮點合併的 ulp 級差異)")


@check("R2", "route_offload_torch 自組 mask == kernel 組的 mask")
def r2_mask_equiv(moe, layer_idx, seq_len):
    import torch

    block, _, gate_logits = _make_inputs(moe, layer_idx, seq_len)
    k_mask, k_w = block.lib.topk_softmax(gate_logits)
    t_mask, t_w = _topk_softmax_masks(block, gate_logits)

    log(f"    router_mask  shape/dtype: {tuple(k_mask.shape)} {k_mask.dtype}")
    log(f"    weights_mask shape/dtype: {tuple(k_w.shape)} {k_w.dtype}")
    assert torch.equal(k_mask, t_mask), "router_mask(選中哪些 expert)不一致"
    dw = _diff(k_w, t_w)
    log(f"    max|kernel_w - torch_w| : {dw:.3e}")
    assert dw < 1e-3, f"routing 權重差異 {dw:.3e} 過大(renorm 順序?)"


@check("R3", "route_offload_torch 輸出 ≈ route_offload_kernel")
def r3_torch_vs_kernel(moe, layer_idx, seq_len):
    block, hs_flat, gate_logits = _make_inputs(moe, layer_idx, seq_len)
    k_out = route_offload_kernel(block, hs_flat, gate_logits).view(seq_len, -1)
    t_out = route_offload_torch(block, hs_flat, gate_logits).view(seq_len, -1)
    d = _diff(k_out, t_out)
    log(f"    max|kernel - torch|    : {d:.3e}")
    assert d < 1e-2, f"輸出差異 {d:.3e} 過大"


@check("R4", "masked gate(只留一顆 expert)dispatch 跑得動且輸出有限非零")
def r4_masked(moe, layer_idx, seq_len):
    import torch

    block, hs_flat, gate_logits = _make_inputs(moe, layer_idx, seq_len)
    # 模擬 random_mask 的 draft phase:每個 token 只留同一顆 expert e,
    # 其餘設 -inf。softmax+renorm 後該 expert 權重為 1。
    e = 7
    masked = torch.full_like(gate_logits, float("-inf"))
    masked[:, e] = gate_logits[:, e]
    r_mask, w_mask = _topk_softmax_masks(block, masked)

    log(f"    selected experts/token : {int(r_mask[0].sum())}"
        f"(top_k={block.top_k},但只有 e={e} 有非零權重)")
    assert bool(r_mask[:, e].all()), f"expert {e} 應對每個 token 都被選"
    nonzero_w = (w_mask.float().abs() > 0).sum(dim=1)
    log(f"    nonzero weight/token   : {int(nonzero_w[0])}(應為 1)")
    assert int(nonzero_w[0]) == 1, "只該有一顆 expert 帶非零權重"

    out = route_offload_torch(block, hs_flat, masked).view(seq_len, -1)
    log(f"    output                 : shape={tuple(out.shape)} dtype={out.dtype}")
    assert torch.isfinite(out).all(), "輸出含 NaN/Inf"
    assert float(out.float().abs().sum()) > 0, "輸出全零(該 expert 沒生效?)"


# ────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--offload-dir",
                   default=os.path.join(REPO_ROOT, "moe_infinity",
                                        "offload_output", "Qwen3-30B-A3B"))
    p.add_argument("--device-memory-ratio", type=float, default=0.75,
                   help="M3 驗 routing 數值,容量非重點,用 M0 已知良好的 0.75")
    p.add_argument("--layer-idx", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=4)
    args = p.parse_args()

    import torch
    from transformers import AutoTokenizer
    from moe_infinity import MoE

    log("=" * 68)
    log("M3 probe — _route_offload 原型(Qwen3)")
    log(f"  model      : {args.model}")
    log(f"  layer_idx  : {args.layer_idx}   seq_len: {args.seq_len}")
    log("=" * 68)

    log("\nloading MoE(...) ...")
    moe = MoE(args.model, {
        "offload_path": args.offload_dir,
        "device_memory_ratio": args.device_memory_ratio,
    })
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 暖機:跑一次短 generate 讓 engine 的 seq_id / dispatch 狀態就緒
    # (與 M1 FW 同模式:單獨 dispatch 需 engine 已被 generate 初始化過)。
    log("warming up engine with a 8-token generate ...")
    ids = tokenizer("The capital of France is", return_tensors="pt"
                    ).input_ids.to("cuda:0")
    with torch.no_grad():
        moe.generate(ids, max_new_tokens=8, do_sample=False,
                     pad_token_id=tokenizer.eos_token_id)

    r1_kernel_vs_native(moe, args.layer_idx, args.seq_len)
    r2_mask_equiv(moe, args.layer_idx, args.seq_len)
    r3_torch_vs_kernel(moe, args.layer_idx, args.seq_len)
    r4_masked(moe, args.layer_idx, args.seq_len)

    log("\n" + "=" * 68)
    if _failures:
        log(f"RESULT: {len(_failures)} FAILED — {', '.join(_failures)}")
    else:
        log("RESULT: ALL PASS")
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
    # moe_infinity 的 C++ thread pool 在正常 interpreter shutdown 會 hang
    os._exit(rc)
