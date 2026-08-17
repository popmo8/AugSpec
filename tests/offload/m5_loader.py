"""M5 — offload_plan.md `load_offload()` 驗證(不接 CLI)。

驗證 loader.py 新增的 load_offload 把 M0–M4 驗過的東西(MoE 載入 +
device patch + cpu_source)正確包成一個函式,且回傳的 model 能照
aug_spec 的用法(`model.generate(...)`,非 moe wrapper)跑起來。

檢查項目:
  L1  import + 回傳形狀:(model, tokenizer, moe, cpu_source) 四個都對
  L2  device patch 生效:model.device == cuda:0(M2 的 assisted
      generation 前提;原生由 placeholder 推得 cpu)
  L3  model.generate 跑通(先手動 moe._configure_hook,模擬 aug_spec
      specbench 的用法)—— 產出合理 token
  L4  cpu_source 是真實權重源:expert 非 placeholder、bf16、在 CPU

Usage:
    .venv/bin/python tests/offload/m5_loader.py
    sbatch tests/offload/m5_loader.sh
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(REPO_ROOT, "tests", "offload", "m5_loader.out")

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


@check("L1", "回傳形狀:(model, tokenizer, moe, cpu_source)")
def l1_shape(bundle):
    model, tokenizer, moe, cpu_source = bundle
    import torch.nn as nn
    assert isinstance(model, nn.Module), "model 不是 nn.Module"
    assert hasattr(moe, "_configure_hook"), "moe 缺 _configure_hook"
    assert model is moe.model, "model 應為 moe.model 同一物件"
    assert cpu_source is not None, "load_cpu_source=True 卻回 None"
    assert tokenizer.pad_token is not None, "pad_token 未設"
    log(f"    model type   : {type(model).__name__}")
    log(f"    cpu_source   : {type(cpu_source).__name__}")


@check("L2", "device patch:model.device == cuda:0")
def l2_device(bundle):
    import torch
    model = bundle[0]
    log(f"    model.device : {model.device}")
    assert model.device == torch.device("cuda:0"), \
        "device patch 沒生效 —— assisted generation 會把 input_ids 搬去 cpu"


@check("L3", "model.generate 跑通(手動 _configure_hook,aug_spec 用法)")
def l3_generate(bundle):
    import torch
    model, tokenizer, moe, _ = bundle
    ids = tokenizer("The capital of France is", return_tensors="pt"
                    ).input_ids.to("cuda:0")
    moe._configure_hook(ids)                 # specbench 每次 generate 前必呼叫
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=16, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    log(f"    generated    : {text!r}")
    assert out.shape[1] > ids.shape[1], "沒產生新 token"


@check("L4", "cpu_source expert 是真實權重(非 placeholder)")
def l4_cpu_source(bundle):
    import torch
    cpu_source = bundle[3]
    block = cpu_source.model.layers[0].mlp
    w = block.experts[0].gate_proj.weight
    log(f"    experts[0].gate_proj : shape={tuple(w.shape)} "
        f"dtype={w.dtype} dev={w.device} absmean={float(w.float().abs().mean()):.3e}")
    assert w.dim() == 2 and tuple(w.shape) != (1,), "cpu_source expert 是 placeholder"
    assert w.dtype == torch.bfloat16 and w.device.type == "cpu"
    assert float(w.float().abs().mean()) > 0, "cpu_source expert 全零"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--offload-dir",
                   default=os.path.join(REPO_ROOT, "moe_infinity",
                                        "offload_output", "Qwen3-30B-A3B"))
    p.add_argument("--device-memory-ratio", type=float, default=0.15)
    args = p.parse_args()

    from aug_spec.runtime.loader import load_offload

    log("=" * 68)
    log("M5 probe — load_offload()")
    log(f"  model               : {args.model}")
    log(f"  device_memory_ratio : {args.device_memory_ratio}")
    log("=" * 68)

    log("\ncalling load_offload(...) ...")
    bundle = load_offload(
        args.model, args.offload_dir,
        device_memory_ratio=args.device_memory_ratio)

    l1_shape(bundle)
    l2_device(bundle)
    l3_generate(bundle)
    l4_cpu_source(bundle)

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
    os._exit(rc)
