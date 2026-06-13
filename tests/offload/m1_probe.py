"""M1 — offload_plan.md 結構探勘（read-only probe）。

載入 `moe_infinity.MoE(...)` 後逐項回答 next_step.md §5 的四個問題，
並再確認 offload_plan.md §2 的事實。每項獨立 try/except：單項紅了
繼續跑完其他項，結尾統一回報並以 exit code 區分全綠 / 有紅。

預設模型 = Qwen3-30B-A3B（offload 主線）。Mixtral 在本 fork 上輸出
亂碼（M0 job 234500 重現，見 offload_plan.md §2），但仍可用
`--model mistralai/Mixtral-8x7B-v0.1 --skip-generate` 做純結構探勘。

完整輸出同步寫到 tests/offload/m1_probe.out 備查。

Usage（互動式 GPU 節點）:
    .venv/bin/python tests/offload/m1_probe.py
或 SLURM:
    sbatch tests/offload/m1_probe.sh

檢查項目：
  Q1  MoE(...) 後 adapter 的 block 路徑（mixtral: layer.block_sparse_moe
      / qwen3: layer.mlp，靠 gate+experts 屬性辨識）是否仍解析得到、
      type 是否為對應的 Sync/MoE block、forward 簽名是否仍是
      (hidden_states)
  Q3  offload 後 expert 與 gate 的權重靜止時是否皆為 shape-(1,) 零
      placeholder（本 fork 連 dense 權重都由 per-module forward hook
      在前向瞬間物化 —— M1 第一輪 job 234581 的發現）
  EX  expert_executor / lib / engine hook（dispatch_local、
      topk_softmax、replace_cache_candidates 等 M3/M10 介面）是否存在
  Q4  device_memory_ratio（預設 0.15）下單卡能否完成一次短 generate；
      回報 GPU used / host RSS
  FW  單獨呼叫一個 MoE block 的 forward（M3 的前提）—— 回傳
      (final, router_logits) 且 shape 正確
  Q2  CPU-resident 權重源與 MoE(...) 同進程共存：cpu_source 的 expert
      權重為真實 bf16、非零（未被 moe_infinity 的 empty-init hook 攔截）；
      回報載入前後 host RSS 增量
"""

from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(REPO_ROOT, "tests", "offload", "m1_probe.out")

# offload 後預期的 block 類別（family → type name）
EXPECTED_BLOCK_TYPES = {"Qwen3MoEBlock", "SyncMixtralSparseMoeBlock"}

_report_lines: list[str] = []
_failures: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _report_lines.append(msg)


def rss_gb() -> float:
    """本進程 host RSS（GB），讀 /proc，不依賴 psutil。"""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024 ** 2)
    return -1.0


def gpu_used_gb() -> float:
    """nvidia-smi 回報的 GPU used（GB）。archer engine 的配置不走
    torch allocator，torch.cuda.memory_allocated 看不到，必須問 driver。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True)
        return int(out.splitlines()[0]) / 1024
    except (subprocess.SubprocessError, ValueError, IndexError):
        return -1.0


def find_moe_blocks(model):
    """與 aug_spec adapter 同邏輯的 block 搜尋：mixtral 在
    layer.block_sparse_moe、qwen3 在 layer.mlp，皆以 gate+experts 辨識。"""
    blocks = []
    for i, layer in enumerate(model.model.layers):
        for attr in ("block_sparse_moe", "mlp"):
            block = getattr(layer, attr, None)
            if block is not None and hasattr(block, "gate") \
                    and hasattr(block, "experts"):
                blocks.append((i, block))
                break
    return blocks


def check(tag: str, desc: str):
    """每個檢查項的裝飾器：印標頭、抓例外、記錄 PASS/FAIL。"""
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
# 檢查項
# ────────────────────────────────────────────────────────────────────

@check("Q1", "MoE(...) 後 block 路徑 / type / forward 簽名")
def q1_structure(moe):
    import torch.nn as nn

    layers = moe.model.model.layers
    blocks = find_moe_blocks(moe.model)
    assert blocks, "gate+experts 的 block 解析不到（adapter iter_moe 會失敗）"
    log(f"    MoE layers found       : {len(blocks)} / {len(layers)}")

    _, block = blocks[0]
    tname = type(block).__name__
    log(f"    block type             : {type(block).__module__}.{tname}")
    assert tname in EXPECTED_BLOCK_TYPES, \
        f"預期 {EXPECTED_BLOCK_TYPES}，拿到 {tname}"
    assert isinstance(block.experts, nn.ModuleList)

    sig = list(inspect.signature(block.forward).parameters)
    log(f"    forward signature      : ({', '.join(sig)})")
    assert sig == ["hidden_states"], f"forward 簽名變了: {sig}"
    log(f"    top_k / num_experts    : {block.top_k} / {block.num_experts}")
    if hasattr(block, "norm_topk_prob"):
        log(f"    norm_topk_prob         : {block.norm_topk_prob}"
            f"  (M3 要驗 lib.topk_softmax 是否處理了 renorm)")


@check("Q3", "offload 後 expert 與 gate 權重靜止時皆為 (1,) placeholder")
def q3_placeholder(moe):
    # 第一輪 M1（job 234581）發現：不只 expert，連 gate/dense 權重
    # 靜止時也是 placeholder（CPU 上）。moe_infinity 在每個 module 掛
    # forward pre/post hook，前向瞬間 archer_engine.begin/end 物化/釋放
    # 該 module 自己的參數（model_offload.py:962-1050）。在替換後的
    # block.forward 內呼叫 block.gate(...) 走 module __call__，hook
    # 照常觸發 —— 對 aug_spec 安全。
    block = find_moe_blocks(moe.model)[0][1]
    for name, p in block.experts[0].named_parameters():
        log(f"    experts[0].{name:<22}: shape={tuple(p.shape)} "
            f"dtype={p.dtype} dev={p.device} sum={float(p.float().sum()):.3e}")
        assert tuple(p.shape) == (1,), \
            f"expert 權重 {name} 不是 (1,) placeholder？設計前提變了"
        assert float(p.float().abs().sum()) == 0.0

    g = block.gate.weight
    log(f"    gate.weight            : shape={tuple(g.shape)} "
        f"dtype={g.dtype} dev={g.device}")
    assert tuple(g.shape) == (1,), (
        f"gate 權重不是 placeholder（shape={tuple(g.shape)}）—— "
        "與 M1 第一輪的發現矛盾，hook 機制可能變了")


@check("EX", "expert_executor / lib / engine 介面盤點（M3 / M10 依賴）")
def ex_interfaces(moe):
    block = find_moe_blocks(moe.model)[0][1]
    for attr in ("expert_executor", "lib", "archer_engine", "layer_id",
                 "expert_tensor_ids"):
        log(f"    block.{attr:<18} : {'YES' if getattr(block, attr, None) is not None else 'MISSING'}")
    assert getattr(block, "expert_executor", None) is not None

    ex = block.expert_executor
    for attr in ("dispatch_local", "wait_dispatch_local"):
        ok = callable(getattr(ex, attr, None))
        log(f"    executor.{attr:<19}: {'YES' if ok else 'MISSING'}")
        assert ok, f"expert_executor.{attr} 不存在 —— M3 的 _route_offload 走不通"

    lib = getattr(block, "lib", None)
    ok = lib is not None and callable(getattr(lib, "topk_softmax", None))
    log(f"    lib.topk_softmax           : {'YES' if ok else 'MISSING'}"
        f"  (qwen3 routing 的 fused kernel)")

    eng = moe.engine
    for attr in ("expert_prefetcher", "expert_tracer", "expert_dispatcher"):
        log(f"    engine.{attr:<21}: {'YES' if getattr(eng, attr, None) is not None else 'MISSING'}")
    pf = getattr(eng, "expert_prefetcher", None)
    for attr in ("fetch_experts_lock_cache", "replace_cache_candidates"):
        found = []
        for owner, name in ((eng, "engine"), (pf, "prefetcher"),
                            (getattr(eng, "archer_engine", None), "archer_engine")):
            if owner is not None and callable(getattr(owner, attr, None)):
                found.append(name)
        log(f"    {attr:<28}: {('on ' + ','.join(found)) if found else 'NOT FOUND (M10 再追)'}")


@check("Q4", "短 generate 跑通（單卡容量 @ 此 device_memory_ratio）")
def q4_generate(moe, tokenizer, max_new_tokens):
    import torch

    ids = tokenizer("The capital of France is", return_tensors="pt"
                    ).input_ids.to("cuda:0")
    with torch.no_grad():
        out = moe.generate(ids, max_new_tokens=max_new_tokens,
                           do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    log(f"    generated ({max_new_tokens} tok)   : {text!r}")
    log(f"    GPU used               : {gpu_used_gb():.1f} GB")
    log(f"    host RSS               : {rss_gb():.1f} GB")
    assert out.shape[1] > ids.shape[1], "沒有產生任何新 token"


@check("FW", "單獨呼叫一個 MoE block forward（M3 前提）")
def fw_single_block(moe):
    import torch

    block = find_moe_blocks(moe.model)[0][1]
    # 權重靜止時是 (1,) placeholder（Q3），維度只能從 config 拿；
    # 且必須走 block(hs)（module __call__）讓物化 hook 觸發，
    # 不能直呼 block.forward(hs)。
    hidden_dim = moe.model.config.hidden_size
    hs = torch.randn(1, 4, hidden_dim, dtype=torch.bfloat16,
                     device="cuda:0")
    with torch.no_grad():
        out = block(hs)
    assert isinstance(out, tuple) and len(out) == 2, \
        f"預期回傳 (final, router_logits)，拿到 {type(out)}"
    final, router_logits = out
    log(f"    final                  : shape={tuple(final.shape)} dtype={final.dtype}")
    log(f"    router_logits          : shape={tuple(router_logits.shape)}")
    assert tuple(final.shape) == (1, 4, hidden_dim)
    assert tuple(router_logits.shape) == (4, block.num_experts)
    assert torch.isfinite(final).all(), "輸出含 NaN/Inf"


@check("Q2", "CPU-resident 權重源與 MoE(...) 同進程共存")
def q2_cpu_source(model_id):
    import torch
    from transformers import AutoModelForCausalLM

    rss_before = rss_gb()
    log(f"    host RSS before        : {rss_before:.1f} GB")
    log("    loading cpu_source (device_map='cpu', bf16) ...")
    cpu_source = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    rss_after = rss_gb()
    log(f"    host RSS after         : {rss_after:.1f} GB  (Δ {rss_after - rss_before:+.1f} GB)")

    expert0 = find_moe_blocks(cpu_source)[0][1].experts[0]
    for name, p in expert0.named_parameters():
        log(f"    cpu experts[0].{name:<18}: shape={tuple(p.shape)} "
            f"dtype={p.dtype} dev={p.device} absmean={float(p.float().abs().mean()):.3e}")
        assert p.dim() == 2 and tuple(p.shape) != (1,), \
            f"cpu_source 的 {name} 也變 placeholder —— 被 moe_infinity 的 init hook 攔截了"
        assert p.dtype == torch.bfloat16 and p.device.type == "cpu"
        assert float(p.float().abs().mean()) > 0, f"cpu_source {name} 全零"
    del cpu_source


# ────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--offload-dir",
                   default=os.path.join(REPO_ROOT, "moe_infinity",
                                        "offload_output", "Qwen3-30B-A3B"))
    p.add_argument("--device-memory-ratio", type=float, default=0.15,
                   help="next_step.md §5 Q4 指定 0.15 驗單卡容量")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--skip-cpu-copy", action="store_true",
                   help="跳過 Q2（cpu_source 載入要再吃約一個模型的 host RAM）")
    p.add_argument("--skip-generate", action="store_true",
                   help="跳過 Q4/FW（純結構檢查，最快；Mixtral 探勘時建議加）")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from moe_infinity import MoE

    log("=" * 68)
    log("M1 probe — offload 結構探勘")
    log(f"  model               : {args.model}")
    log(f"  offload_dir         : {args.offload_dir}")
    log(f"  device_memory_ratio : {args.device_memory_ratio}")
    log("=" * 68)

    log(f"\nhost RSS at start       : {rss_gb():.1f} GB")
    log("loading MoE(...) ...")
    moe = MoE(args.model, {
        "offload_path": args.offload_dir,
        "device_memory_ratio": args.device_memory_ratio,
    })
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    log(f"host RSS after MoE load : {rss_gb():.1f} GB")
    log(f"GPU used after MoE load : {gpu_used_gb():.1f} GB")

    q1_structure(moe)
    q3_placeholder(moe)
    ex_interfaces(moe)
    if args.skip_generate:
        log("\n[Q4] [FW] skipped (--skip-generate)")
    else:
        q4_generate(moe, tokenizer, args.max_new_tokens)
        fw_single_block(moe)   # 放在 generate 後（engine 狀態已暖機）
    if args.skip_cpu_copy:
        log("\n[Q2] skipped (--skip-cpu-copy)")
    else:
        q2_cpu_source(args.model)

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
        # 報告一定要先落地，才能 os._exit
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w") as f:
            f.write("\n".join(_report_lines) + "\n")
        print(f"\nreport → {OUT_PATH}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # moe_infinity 的 C++ thread pool 在正常 interpreter shutdown 會 hang
    # （同 examples/mixtral_example.py 的處理方式）
    os._exit(rc)
