"""SpecBench evaluation runner using HuggingFace assisted decoding.

`run_specbench()` is a self-contained driver that:

  * Loads SpecBench's question.jsonl (once, cached locally) and samples
    `questions_per_cat` per category.
  * Forces a fixed-T speculative schedule (defeats HF's heuristic ±
    adjustment and ConfidenceCriteria early stop). `actual_T` falls
    below T only when the draft emits EOS or the budget runs out.
  * Runs `target.generate(..., assistant_model=draft)` per question.
  * Captures per-cycle (draft tokens, target argmax, num_matches, top-k
    logits) and per-question (cycles, new_tokens, wall_time, MAT,
    AccRate, TPS) telemetry.
  * Writes `per_question_summary.csv`, `overall_summary.csv`, and
    optionally `tokens.csv` to `output_dir`.
  * Invokes `on_cycle(qres, cycle_stats)` after every verify cycle so
    callers can layer their own telemetry without copy-pasting the
    driver.
"""

from __future__ import annotations

import csv
import json
import random
import time
import urllib.request
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .loader import get_model_device


LM_TOPK_DEFAULT = 8

SPEC_BENCH_QUESTION_URL = (
    "https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/"
    "data/spec_bench/question.jsonl"
)

SPEC_BENCH_MT_BENCH_CATS = frozenset({
    "writing", "roleplay", "reasoning", "math",
    "coding", "extraction", "stem", "humanities",
})

# Subtask order from Spec-Bench's evaluation/speed.py::get_single_speedup.
SPEC_BENCH_SUBTASKS: Tuple[str, ...] = (
    "mt_bench", "translation", "summarization",
    "qa", "math_reasoning", "rag", "overall",
)


# =============================================================================
# Dataset + prompt helpers
# =============================================================================

def _load_spec_bench_questions(cache_dir: Path) -> List[Dict[str, Any]]:
    """Download (once) and load Spec-Bench's question.jsonl."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "question.jsonl"
    if not cache_file.exists():
        print(f"  Downloading Spec-Bench questions → {cache_file}")
        urllib.request.urlretrieve(SPEC_BENCH_QUESTION_URL, cache_file)
    questions: List[Dict[str, Any]] = []
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def _vicuna_fallback_prompt(messages: List[Dict[str, str]]) -> str:
    """Vicuna-style prompt for tokenizers without a chat_template
    (e.g. Mixtral-8x7B-v0.1). Spec-Bench itself uses this via fastchat."""
    header = (
        "A chat between a curious user and an artificial intelligence "
        "assistant. The assistant gives helpful, detailed, and polite "
        "answers to the user's questions."
    )
    parts = [header]
    for m in messages:
        role = "USER" if m["role"] == "user" else "ASSISTANT"
        parts.append(f"{role}: {m['content']}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _format_chat_prompt(tokenizer, history: List[Dict[str, str]],
                        new_user_msg: str) -> str:
    messages = history + [{"role": "user", "content": new_user_msg}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass
    return _vicuna_fallback_prompt(messages)


def _category_matches(q_cat: str, subtask: str) -> bool:
    if subtask == "overall":
        return True
    if subtask == "mt_bench":
        return q_cat in SPEC_BENCH_MT_BENCH_CATS
    return q_cat == subtask


# =============================================================================
# Result dataclasses
# =============================================================================

@dataclass
class CycleStats:
    """Per-cycle output of one verify step. Passed to `on_cycle`."""

    cycle_idx: int
    actual_T: int
    num_matches: int
    draft_tokens: List[int]
    real_tokens: List[int]
    accepted: List[bool]
    target_top1_margin: List[float]
    target_top2_id: List[int]
    # Top-k per draft proposal position (shape [actual_T, K], cpu fp16/int64).
    draft_top_vals: torch.Tensor
    draft_top_ids: torch.Tensor
    target_top_vals: torch.Tensor
    target_top_ids: torch.Tensor


@dataclass
class QuestionResult:
    qid: str
    category: str
    num_cycles: int
    num_new_tokens: int
    wall_time_s: float
    mean_accept_length: float
    acceptance_rate: float
    tokens_per_second: float
    n_proposed: int = 0
    n_accepted: int = 0
    accept_lengths: List[int] = field(default_factory=list)


@dataclass
class SpecBenchResult:
    per_question: List[QuestionResult]
    per_subtask: Dict[str, Dict[str, float]]
    overall: Dict[str, float]


# =============================================================================
# Patched assisted-decoding generator
# =============================================================================

@contextmanager
def _locked_assist_patch(T: int,
                         lm_topk: int,
                         on_verify: Callable[[CycleStats], None]):
    """Monkey-patch AssistedCandidateGenerator to:
      - lock `num_assistant_tokens = T` (defeat HF's heuristic ±),
      - clip if HF ever returns more than T candidates,
      - capture per-cycle CycleStats and call `on_verify` after each
        `update_candidate_strategy`.
    """
    from transformers.generation.candidate_generator import (
        AssistedCandidateGenerator,
    )

    state: Dict[str, Any] = {
        "cycle_idx": 0,
        "draft_tokens": None,        # List[int]
        "draft_top_vals": None,      # [T, K] fp16 cpu
        "draft_top_ids": None,       # [T, K] int64 cpu
    }

    orig_get = AssistedCandidateGenerator.get_candidates
    orig_upd = AssistedCandidateGenerator.update_candidate_strategy

    def patched_get(self, input_ids):
        self.num_assistant_tokens = T
        cand, lg = orig_get(self, input_ids)
        target_len = input_ids.shape[1] + T
        if cand.shape[1] > target_len:
            cand = cand[:, :target_len]
            if lg is not None:
                lg = lg[:, :T]
        actual_T = cand.shape[1] - input_ids.shape[1]
        if actual_T > 0:
            state["draft_tokens"] = cand[
                0,
                input_ids.shape[1]:input_ids.shape[1] + actual_T,
            ].tolist()
            if lg is not None and lg.shape[1] >= actual_T:
                d_top = torch.topk(lg[0, :actual_T, :], k=lm_topk, dim=-1)
                state["draft_top_vals"] = (
                    d_top.values.detach().to(torch.float16).cpu())
                state["draft_top_ids"] = d_top.indices.detach().cpu()
            else:
                state["draft_top_vals"] = None
                state["draft_top_ids"] = None
        else:
            state["draft_tokens"] = []
            state["draft_top_vals"] = None
            state["draft_top_ids"] = None
        return cand, lg

    def patched_upd(self, input_ids, scores, num_matches):
        draft_tokens: List[int] = state["draft_tokens"] or []
        actual_T = len(draft_tokens)
        if actual_T > 0:
            if scores is not None and scores.shape[1] >= actual_T:
                t_top = torch.topk(
                    scores[0, :actual_T, :], k=lm_topk, dim=-1)
                t_vals = t_top.values.detach().to(torch.float16).cpu()
                t_ids = t_top.indices.detach().cpu()
                real_tokens = t_ids[:, 0].tolist()
                target_top1_margin = (t_vals[:, 0] - t_vals[:, 1]).tolist()
                target_top2_id = t_ids[:, 1].tolist()
            else:
                t_vals = torch.full((actual_T, lm_topk),
                                    float("nan"), dtype=torch.float16)
                t_ids = torch.full((actual_T, lm_topk), -1, dtype=torch.int64)
                real_tokens = [-1] * actual_T
                target_top1_margin = [float("nan")] * actual_T
                target_top2_id = [-1] * actual_T

            d_vals = state["draft_top_vals"]
            d_ids = state["draft_top_ids"]
            if d_vals is None or d_ids is None:
                d_vals = torch.full((actual_T, lm_topk),
                                    float("nan"), dtype=torch.float16)
                d_ids = torch.full((actual_T, lm_topk), -1, dtype=torch.int64)

            nm = int(num_matches)
            cs = CycleStats(
                cycle_idx=state["cycle_idx"],
                actual_T=actual_T,
                num_matches=nm,
                draft_tokens=draft_tokens,
                real_tokens=real_tokens,
                accepted=[t < nm for t in range(actual_T)],
                target_top1_margin=target_top1_margin,
                target_top2_id=target_top2_id,
                draft_top_vals=d_vals,
                draft_top_ids=d_ids,
                target_top_vals=t_vals,
                target_top_ids=t_ids,
            )
            on_verify(cs)
            state["cycle_idx"] += 1

        result = orig_upd(self, input_ids, scores, num_matches)
        self.num_assistant_tokens = T
        return result

    AssistedCandidateGenerator.get_candidates = patched_get
    AssistedCandidateGenerator.update_candidate_strategy = patched_upd
    try:
        yield
    finally:
        AssistedCandidateGenerator.get_candidates = orig_get
        AssistedCandidateGenerator.update_candidate_strategy = orig_upd


# =============================================================================
# CSV schemas
# =============================================================================

_TOKENS_FIELDS = [
    "question_id", "category", "cycle_idx", "pos_idx",
    "draft_token_id", "real_token_id", "accepted",
    "target_top1_margin", "target_top2_id",
    "draft_top8_vals", "draft_top8_ids",
    "target_top8_vals", "target_top8_ids",
]
_PER_Q_FIELDS = [
    "question_id", "category", "num_cycles", "num_new_tokens",
    "wall_time_s", "mean_accept_length",
    "acceptance_rate", "tokens_per_second",
]
_OVERALL_FIELDS = [
    "subtask", "num_questions", "total_cycles",
    "mean_accept_tokens", "acceptance_rate", "tokens_per_second",
]


def _write_tokens_row(writer: csv.DictWriter,
                      qid: str, category: str, cs: CycleStats) -> None:
    for t in range(cs.actual_T):
        writer.writerow({
            "question_id": qid,
            "category": category,
            "cycle_idx": cs.cycle_idx,
            "pos_idx": t,
            "draft_token_id": int(cs.draft_tokens[t]),
            "real_token_id": int(cs.real_tokens[t]),
            "accepted": int(cs.accepted[t]),
            "target_top1_margin": f"{cs.target_top1_margin[t]:.6f}",
            "target_top2_id": int(cs.target_top2_id[t]),
            "draft_top8_vals": ";".join(
                f"{float(v):.4f}" for v in cs.draft_top_vals[t].tolist()),
            "draft_top8_ids": ";".join(
                str(int(i)) for i in cs.draft_top_ids[t].tolist()),
            "target_top8_vals": ";".join(
                f"{float(v):.4f}" for v in cs.target_top_vals[t].tolist()),
            "target_top8_ids": ";".join(
                str(int(i)) for i in cs.target_top_ids[t].tolist()),
        })


# =============================================================================
# Aggregation helpers
# =============================================================================

def _aggregate_subtask(per_q: List[QuestionResult],
                       subtask: str) -> Optional[Dict[str, Any]]:
    filt = [q for q in per_q if _category_matches(q.category, subtask)]
    if not filt:
        return None
    accs: List[int] = []
    n_prop = 0
    n_acc = 0
    tps_list: List[float] = []
    for q in filt:
        accs.extend(q.accept_lengths)
        n_prop += q.n_proposed
        n_acc += q.n_accepted
        if q.wall_time_s > 0:
            tps_list.append(q.num_new_tokens / q.wall_time_s)
    return {
        "num_questions": len(filt),
        "total_cycles": len(accs),
        "mean_accept_tokens": (sum(accs) / len(accs)) if accs else 0.0,
        "acceptance_rate": (n_acc / n_prop) if n_prop > 0 else 0.0,
        "tokens_per_second": (sum(tps_list) / len(tps_list)) if tps_list else 0.0,
    }


def _print_final_table(label: str,
                       per_subtask: Dict[str, Dict[str, Any]]) -> None:
    title = f"[{label}] FINAL" if label else "FINAL"
    print(f"\n{title}")
    print(f"  {'subtask':<16} {'MAT':>8} {'AccRate':>8} {'TPS':>8}")
    for subtask in SPEC_BENCH_SUBTASKS:
        m = per_subtask.get(subtask)
        if m is None:
            continue
        print(f"  {subtask:<16} "
              f"{m['mean_accept_tokens']:>8.4f} "
              f"{m['acceptance_rate']:>8.4f} "
              f"{m['tokens_per_second']:>8.4f}")


# =============================================================================
# Main entry point
# =============================================================================

def run_specbench(
    target_model,
    draft_model,
    tokenizer,
    num_speculative: int = 3,
    questions_per_cat: int = 10,
    max_new_tokens: int = 512,
    output_dir: Optional[Path] = None,
    label: str = "",
    seed: int = 0,
    spec_bench_cache: Optional[Path] = None,
    emit_tokens_csv: bool = True,
    progress_every: int = 5,
    warmup: bool = True,
    lm_topk: int = LM_TOPK_DEFAULT,
    on_cycle: Optional[Callable[[QuestionResult, CycleStats], None]] = None,
    on_question_start: Optional[Callable[[Dict[str, Any]], None]] = None,
    before_generate: Optional[Callable[[Any], None]] = None,
    vram_limit_bytes: Optional[int] = None,
    vram_guard: bool = False,
) -> SpecBenchResult:
    """Run SpecBench eval with a fixed-T speculative schedule.

    See module docstring for the full pipeline. Returns a `SpecBenchResult`
    with `per_question`, `per_subtask`, and `overall` aggregates.

    `before_generate(input_ids)` (optional) runs immediately before every
    `target_model.generate(...)` call (warmup + each question). The offload
    backend wires `moe._configure_hook` here — moe_infinity needs fresh
    expert-tracer sequence entries per generation. No-op (None) on hf.
    """
    target_model.generation_config.num_assistant_tokens = num_speculative
    target_model.generation_config.num_assistant_tokens_schedule = "constant"
    draft_model.generation_config.assistant_confidence_threshold = 0

    # VRAM budget audit + guard (verify_merge_plan.md §0.1). Driver-level total
    # GPU memory (includes archer's cudaMalloc pool, which torch peak misses) is
    # sampled per verify cycle: track the run peak, and warn once if it crosses
    # the budget (×1.05 slack). This only DETECTS a breach — it never enforces;
    # the budget is held structurally (archer cache cap + flush). No-op when
    # vram_limit_bytes is None (hf backend / no budget set).
    from aug_spec.runtime.loader import get_gpu_used_bytes
    _vram_peak = [0]
    _vram_warned = [False]

    def _vram_sample(phase: str = "") -> None:
        if vram_limit_bytes is None:
            return
        used = get_gpu_used_bytes(0)
        if used < 0:
            return
        if used > _vram_peak[0]:
            _vram_peak[0] = used
        if (vram_guard and not _vram_warned[0]
                and used > vram_limit_bytes * 1.05):
            _vram_warned[0] = True
            print(f"  [vram_guard] WARNING used={used / 1e9:.2f}GB > "
                  f"limit={vram_limit_bytes / 1e9:.2f}GB ×1.05 (phase={phase}) — "
                  f"total VRAM over budget (e.g. merged not flushed / leak). "
                  f"Audit only; not enforced.", flush=True)

    cache_dir = (spec_bench_cache if spec_bench_cache is not None
                 else Path.cwd() / "data" / "spec_bench")
    all_q = _load_spec_bench_questions(cache_dir)
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for q in all_q:
        by_cat[q["category"]].append(q)
    rng = random.Random(seed)
    questions: List[Dict[str, Any]] = []
    for cat in sorted(by_cat):
        pool = list(by_cat[cat])
        rng.shuffle(pool)
        questions.extend(pool[:questions_per_cat])

    print("=" * 70)
    title = f"  SpecBench [{label}]" if label else "  SpecBench"
    print(title)
    print(f"  T (fixed)     : {num_speculative}")
    print(f"  Q/cat         : {questions_per_cat}")
    print(f"  Total Q       : {len(questions)} ({len(by_cat)} categories)")
    if output_dir is not None:
        print(f"  Output        : {output_dir}")
    print("=" * 70)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    tokens_writer: Optional[csv.DictWriter] = None
    tokens_f = None
    per_q_writer: Optional[csv.DictWriter] = None
    per_q_f = None
    if output_dir is not None:
        if emit_tokens_csv:
            tokens_f = open(output_dir / "tokens.csv", "w",
                            newline="", encoding="utf-8")
            tokens_writer = csv.DictWriter(tokens_f, fieldnames=_TOKENS_FIELDS)
            tokens_writer.writeheader()
        per_q_f = open(output_dir / "per_question_summary.csv", "w",
                       newline="", encoding="utf-8")
        per_q_writer = csv.DictWriter(per_q_f, fieldnames=_PER_Q_FIELDS)
        per_q_writer.writeheader()

    if warmup:
        print("  Warmup (compile + first spec call) ...")
        warm_prompt = _format_chat_prompt(
            tokenizer, [], "Briefly introduce yourself.")
        warm_in = tokenizer(warm_prompt, return_tensors="pt").to(
            get_model_device(target_model))
        # no_grad (not inference_mode): the offload backend's dispatcher does an
        # in-place index_add_ on hidden_states inside its C++ thread; under
        # inference_mode those are "inference tensors" and the in-place update
        # aborts. The main loop already runs generate under plain no_grad, so
        # warmup must match. (hf is unaffected either way.)
        with torch.no_grad():
            if before_generate is not None:
                before_generate(warm_in["input_ids"])
            target_model.generate(
                **warm_in, max_new_tokens=8, do_sample=False,
                assistant_model=draft_model,
            )
        print("    warmup done")

    per_question: List[QuestionResult] = []

    try:
        t_sweep = time.perf_counter()
        for qi, q in enumerate(questions):
            qid = str(q["question_id"])
            category = q["category"]

            if on_question_start is not None:
                on_question_start(q)

            user_msg = q["turns"][0]
            prompt = _format_chat_prompt(tokenizer, [], user_msg)
            inputs = tokenizer(prompt, return_tensors="pt").to(
                get_model_device(target_model))
            input_len = int(inputs["input_ids"].shape[1])

            cycle_stats: List[Tuple[int, int]] = []  # (actual_T, num_matches)

            qres_partial = QuestionResult(
                qid=qid, category=category, num_cycles=0, num_new_tokens=0,
                wall_time_s=0.0, mean_accept_length=0.0,
                acceptance_rate=0.0, tokens_per_second=0.0,
            )

            def _on_verify(cs: CycleStats,
                           _qres=qres_partial,
                           _qid=qid, _cat=category):
                cycle_stats.append((cs.actual_T, cs.num_matches))
                if tokens_writer is not None:
                    _write_tokens_row(tokens_writer, _qid, _cat, cs)
                if on_cycle is not None:
                    on_cycle(_qres, cs)
                _vram_sample("verify_cycle")

            t0 = time.perf_counter()
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if before_generate is not None:
                    before_generate(inputs["input_ids"])
                with _locked_assist_patch(
                    num_speculative, lm_topk, _on_verify,
                ):
                    out = target_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        assistant_model=draft_model,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                wall = time.perf_counter() - t0

                new_tokens = int(out.shape[1] - input_len)
                accept_lens = [1 + nm for _, nm in cycle_stats]
                n_prop = sum(n for n, _ in cycle_stats)
                n_acc = sum(nm for _, nm in cycle_stats)
                mat = (sum(accept_lens) / len(accept_lens)
                       if accept_lens else 0.0)
                acc_rate = (n_acc / n_prop) if n_prop > 0 else 0.0
                tps = (new_tokens / wall) if wall > 0 else 0.0

                qres = QuestionResult(
                    qid=qid, category=category,
                    num_cycles=len(cycle_stats),
                    num_new_tokens=new_tokens,
                    wall_time_s=wall,
                    mean_accept_length=mat,
                    acceptance_rate=acc_rate,
                    tokens_per_second=tps,
                    n_proposed=n_prop,
                    n_accepted=n_acc,
                    accept_lengths=accept_lens,
                )
                per_question.append(qres)

                if per_q_writer is not None:
                    per_q_writer.writerow({
                        "question_id": qid,
                        "category": category,
                        "num_cycles": qres.num_cycles,
                        "num_new_tokens": qres.num_new_tokens,
                        "wall_time_s": f"{qres.wall_time_s:.4f}",
                        "mean_accept_length": f"{qres.mean_accept_length:.4f}",
                        "acceptance_rate": f"{qres.acceptance_rate:.4f}",
                        "tokens_per_second": f"{qres.tokens_per_second:.4f}",
                    })
                if tokens_f is not None:
                    tokens_f.flush()
                if per_q_f is not None:
                    per_q_f.flush()
            except Exception as e:
                print(f"  [WARN] qid={qid} failed: "
                      f"{type(e).__name__}: {e}")

            if (qi + 1) % progress_every == 0 or qi + 1 == len(questions):
                elapsed = time.perf_counter() - t_sweep
                eta = elapsed / (qi + 1) * (len(questions) - qi - 1)
                print(f"  [{qi+1:>3}/{len(questions)}] "
                      f"cat={category:<14} "
                      f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")
    finally:
        if tokens_f is not None:
            tokens_f.close()
        if per_q_f is not None:
            per_q_f.close()

    per_subtask: Dict[str, Dict[str, Any]] = {}
    for subtask in SPEC_BENCH_SUBTASKS:
        m = _aggregate_subtask(per_question, subtask)
        if m is not None:
            per_subtask[subtask] = m

    if output_dir is not None:
        with open(output_dir / "overall_summary.csv", "w",
                  newline="", encoding="utf-8") as f:
            ow = csv.DictWriter(f, fieldnames=_OVERALL_FIELDS)
            ow.writeheader()
            for subtask in SPEC_BENCH_SUBTASKS:
                m = per_subtask.get(subtask)
                if m is None:
                    continue
                ow.writerow({
                    "subtask": subtask,
                    "num_questions": m["num_questions"],
                    "total_cycles": m["total_cycles"],
                    "mean_accept_tokens": f"{m['mean_accept_tokens']:.4f}",
                    "acceptance_rate": f"{m['acceptance_rate']:.4f}",
                    "tokens_per_second": f"{m['tokens_per_second']:.4f}",
                })

    _print_final_table(label, per_subtask)
    if vram_limit_bytes is not None and _vram_peak[0] > 0:
        over = _vram_peak[0] > vram_limit_bytes
        print(f"\n  [vram] peak={_vram_peak[0] / 1e9:.2f}GB  "
              f"limit={vram_limit_bytes / 1e9:.2f}GB  "
              f"{'OVER BUDGET ⚠' if over else 'within budget ✓'}")
    if output_dir is not None:
        print(f"\n  → CSVs saved to {output_dir}")

    overall = per_subtask.get("overall", {
        "mean_accept_tokens": 0.0,
        "acceptance_rate": 0.0,
        "tokens_per_second": 0.0,
        "num_questions": 0,
        "total_cycles": 0,
    })
    return SpecBenchResult(
        per_question=per_question,
        per_subtask=per_subtask,
        overall=overall,
    )
