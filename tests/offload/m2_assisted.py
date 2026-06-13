"""M2 — offload_plan.md assisted generation 探針（全計畫最大未知）。

spec-bench 的本質是 `model.generate(assistant_model=model)`（同一物件、
shared weights）。這在 offloaded model 上沒人驗證過 —— HF 的
AssistedCandidateGenerator 會做 KV-cache crop / re-forward，可能跟
archer engine 的狀態管理互咬。本 script 在不碰 aug_spec 程式碼的前提下
直接驗證這件事。

關鍵正確性判準：greedy 下 target==draft 的 assisted generation，draft
提案永遠等於 target 的 argmax → 全數接受 → **輸出必須與普通 greedy
generate 逐 token 相同**。不同就是 KV/狀態互咬的鐵證。

第一輪（job 234621）發現：offload 模型參數靜止時全是 CPU placeholder
→ `model.device` 回報 cpu → HF 的 AssistedCandidateGenerator 把
input_ids 搬去 CPU（candidate_generator.py:211）→ device mismatch +
device-side assert 毒化 CUDA context，A2/A3 連帶陣亡。因此：
  * 預設套用 device 屬性 workaround（`--no-device-patch` 可重現原 bug）
  * 三個檢查改為各自獨立進程跑（wrapper 負責），毒化不再串染

檢查項目：
  A1  短 prompt（~30 tok）：assisted == plain greedy（token 級全等）
  A2  長 prompt（≥1024 tok）：assisted 跑得完且 == plain greedy ——
      interface_example.py 曾把 input+output 壓在 128 token，要確認
      那是保守設定而不是 kernel 真實上限（Spec-Bench prompt 動輒上千）
  A3  連續 3 個 prompt、每次 generate 前重呼叫 _configure_hook ——
      驗證 hook 重複呼叫 / engine 狀態跨 question 不累積壞

Usage:
    .venv/bin/python tests/offload/m2_assisted.py
    sbatch tests/offload/m2_assisted.sh
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(REPO_ROOT, "tests", "offload", "m2_assisted.out")

_report_lines: list[str] = []
_failures: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _report_lines.append(msg)


def gpu_used_gb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True)
        return int(out.splitlines()[0]) / 1024
    except (subprocess.SubprocessError, ValueError, IndexError):
        return -1.0


def patch_device_property(model) -> None:
    """offload 模型的參數靜止時全是 CPU placeholder，`model.device`
    因此回報 cpu；HF assisted generation 用它決定把 input_ids 搬去哪
    （candidate_generator.py:113/211）。把 device 屬性釘在 cuda:0。
    M5 的 load_offload 必須帶同樣的 workaround。"""
    import torch
    cls = type(model)
    cls.device = property(lambda self: torch.device("cuda:0"))
    log(f"    [workaround] {cls.__name__}.device → 恆為 cuda:0"
        "（原生由 placeholder 參數推得 cpu）")


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

SHORT_PROMPT = ("A mixture-of-experts model differs from a dense "
                "transformer in that")

# 長 prompt 用重複段落堆出 ≥1024 token（內容通順即可，重點是長度）
LONG_PARAGRAPH = (
    "The development of large language models has fundamentally changed "
    "how we approach natural language processing. Mixture-of-experts "
    "architectures route each token to a small subset of expert networks, "
    "which keeps the computational cost per token low while scaling the "
    "total parameter count. However, serving such models on a single GPU "
    "requires offloading expert weights to host memory and fetching them "
    "on demand, which makes the PCIe link the dominant bottleneck during "
    "autoregressive decoding. ")


def _generate(model, tokenizer, ids, max_new_tokens, assistant=None):
    import torch
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                  pad_token_id=tokenizer.eos_token_id)
    if assistant is not None:
        kwargs["assistant_model"] = assistant
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, **kwargs)
    dt = time.perf_counter() - t0
    return out[0][ids.shape[1]:].tolist(), dt


def _compare_one(moe, tokenizer, prompt_ids, max_new_tokens, label):
    """同一 prompt 跑 plain greedy 與 assisted greedy，必須 token 全等。"""
    moe._configure_hook(prompt_ids)
    plain, t_plain = _generate(moe.model, tokenizer, prompt_ids,
                               max_new_tokens)
    moe._configure_hook(prompt_ids)
    assisted, t_assist = _generate(moe.model, tokenizer, prompt_ids,
                                   max_new_tokens, assistant=moe.model)
    log(f"    [{label}] plain   : {len(plain)} tok in {t_plain:.1f}s")
    log(f"    [{label}] assisted: {len(assisted)} tok in {t_assist:.1f}s")
    log(f"    [{label}] text    : "
        f"{tokenizer.decode(assisted, skip_special_tokens=True)[:120]!r}")
    assert assisted == plain, (
        f"assisted 與 plain greedy 輸出不同（前者 {len(assisted)} tok、"
        f"後者 {len(plain)} tok；首個分歧位置 "
        f"{next((i for i, (a, b) in enumerate(zip(assisted, plain)) if a != b), min(len(assisted), len(plain)))}）"
        " —— KV-cache crop / engine 狀態互咬的證據")


@check("A1", "短 prompt：assisted == plain greedy")
def a1_short(moe, tokenizer, max_new_tokens):
    ids = tokenizer(SHORT_PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    log(f"    prompt len = {ids.shape[1]} tok")
    _compare_one(moe, tokenizer, ids, max_new_tokens, "short")


@check("A2", "長 prompt（≥1024 tok）：assisted 跑得完且 == plain greedy")
def a2_long(moe, tokenizer, max_new_tokens):
    text = LONG_PARAGRAPH * 20
    ids = tokenizer(text, return_tensors="pt",
                    truncation=True, max_length=1200
                    ).input_ids.to("cuda:0")
    assert ids.shape[1] >= 1024, f"長 prompt 只有 {ids.shape[1]} tok，堆料不夠"
    log(f"    prompt len = {ids.shape[1]} tok")
    _compare_one(moe, tokenizer, ids, max_new_tokens, "long")


@check("A3", "連續 3 個 prompt、每次重呼叫 _configure_hook")
def a3_repeat(moe, tokenizer, max_new_tokens):
    prompts = [
        "The capital of France is",
        "In computer architecture, a cache miss occurs when",
        "The main difference between TCP and UDP is",
    ]
    for i, p in enumerate(prompts, 1):
        ids = tokenizer(p, return_tensors="pt").input_ids.to("cuda:0")
        moe._configure_hook(ids)
        toks, dt = _generate(moe.model, tokenizer, ids, max_new_tokens,
                             assistant=moe.model)
        log(f"    [{i}/3] {len(toks)} tok in {dt:.1f}s: "
            f"{tokenizer.decode(toks, skip_special_tokens=True)[:80]!r}")
        assert len(toks) > 0, f"第 {i} 個 prompt 沒產生任何 token"


# ────────────────────────────────────────────────────────────────────

def main() -> int:
    global OUT_PATH

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--offload-dir",
                   default=os.path.join(REPO_ROOT, "moe_infinity",
                                        "offload_output", "Qwen3-30B-A3B"))
    p.add_argument("--device-memory-ratio", type=float, default=0.75,
                   help="M2 驗的是 assisted generation 機制本身，用 M0 "
                        "已知良好的 0.75 隔離容量變因（容量歸 M1 Q4 管）")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--only", choices=["A1", "A2", "A3"], default=None,
                   help="只跑單一檢查（wrapper 用它做進程隔離，CUDA "
                        "context 毒化不串染）；報告改寫 m2_assisted_<項>.out")
    p.add_argument("--no-device-patch", action="store_true",
                   help="不套 device workaround（重現第一輪的原始 bug 用）")
    args = p.parse_args()

    if args.only:
        OUT_PATH = OUT_PATH.replace(".out", f"_{args.only}.out")

    from transformers import AutoTokenizer
    from moe_infinity import MoE

    log("=" * 68)
    log("M2 probe — assisted generation on offloaded model")
    log(f"  model               : {args.model}")
    log(f"  offload_dir         : {args.offload_dir}")
    log(f"  device_memory_ratio : {args.device_memory_ratio}")
    log(f"  only                : {args.only or 'A1+A2+A3（同進程）'}")
    log("=" * 68)

    log("\nloading MoE(...) ...")
    moe = MoE(args.model, {
        "offload_path": args.offload_dir,
        "device_memory_ratio": args.device_memory_ratio,
    })
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    log(f"GPU used after load     : {gpu_used_gb():.1f} GB")
    if not args.no_device_patch:
        patch_device_property(moe.model)

    if args.only in (None, "A1"):
        a1_short(moe, tokenizer, args.max_new_tokens)
    if args.only in (None, "A2"):
        a2_long(moe, tokenizer, args.max_new_tokens)
    if args.only in (None, "A3"):
        a3_repeat(moe, tokenizer, args.max_new_tokens)

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
