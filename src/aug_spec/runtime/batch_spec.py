"""Batched speculative decoding loop (batch_spec_plan.md).

Opt-in via `run.batch_loop: true` — NOTHING here is imported or executed by
the legacy HF-assisted path. The loop re-implements HF assisted greedy
decoding semantics for batch B with a "dirty KV" scheme: rejected draft
positions stay in the cache as HOLES masked out by the 2D attention mask,
and correctness of RoPE comes from explicit LOGICAL position_ids (HF caches
keys post-rotation, so holes cannot contaminate valid tokens).

Core invariants (batch_spec_plan.md §1.2 — the tests assert these):
  I1  The target cache always lags the committed stream by exactly ONE
      token (the newest bonus). Each verify feeds [bonus, p_1..p_T].
  I2  verify logits[j] predicts committed position after p_j; k = longest
      matching prefix; new bonus b' = argmax(logits[k]); the sequence
      advances k+1 tokens.
  I3  After verify, slots [bonus, p_1..p_k] are valid, p_{k+1}..p_T holes.
  I4  The assistant keeps its OWN draft-routing KV for accepted tokens
      (HF behaviour — do not "fix" it). When k == T, p_T was never fed to
      the assistant: the next draft step-1 feeds [p_T, b'] (width 2, with
      a masked pad slot for sequences that don't need it).
  I5  mask-sum invariants (debug):
        tgt_mask.sum == prompt_len + len(committed) - 1
        ast_mask.sum == prompt_len + len(committed) - 1 - has_missing
  I6  Physical append order == logical order, so causality is preserved
      by construction; holes only ever need key-masking.

`draft: none` runs the same loop with T=0 (verify degenerates to plain
batched greedy, one token per step) — the MoE-Caching baseline.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers import DynamicCache


# =========================================================================
# Cache utilities
# =========================================================================

def stack_caches(caches: List[DynamicCache]) -> Tuple[DynamicCache,
                                                      torch.Tensor]:
    """Merge per-sequence (batch-1) caches into one batched cache.

    Content is LEFT-aligned (pad at the tail, zeros); returns the batched
    cache plus the validity mask [B, P_max]. Later appends always go to the
    physical tail, so the pad becomes an interior hole handled by the mask.
    """
    n_layers = len(caches[0].layers)
    lens = [c.get_seq_length() for c in caches]
    P = max(lens)
    legacy = []
    for li in range(n_layers):
        ks, vs = [], []
        for c, L in zip(caches, lens):
            k, v = c.layers[li].keys, c.layers[li].values
            if L < P:
                pad = k.new_zeros(k.shape[0], k.shape[1], P - L, k.shape[3])
                k = torch.cat([k, pad], dim=2)
                v = torch.cat([v, pad.clone()], dim=2)
            ks.append(k)
            vs.append(v)
        legacy.append((torch.cat(ks, dim=0), torch.cat(vs, dim=0)))
    mask = torch.zeros(len(caches), P, dtype=torch.long,
                       device=legacy[0][0].device)
    for i, L in enumerate(lens):
        mask[i, :L] = 1
    return DynamicCache.from_legacy_cache(tuple(legacy)), mask


# =========================================================================
# Batch state
# =========================================================================

@dataclass
class SeqRecord:
    """Per-sequence telemetry, survives repacking (indexed by question)."""
    qid: str = ""
    category: str = ""
    prompt_len: int = 0
    committed: List[int] = field(default_factory=list)  # incl. every bonus
    accept_lens: List[int] = field(default_factory=list)
    n_prop: int = 0
    n_acc: int = 0
    finished: bool = False


@dataclass
class BatchState:
    """Live tensors for the sequences still decoding. All batch-dim arrays
    are repacked together when sequences finish."""
    seq_idx: List[int]                    # position → index into records
    records: List[SeqRecord]              # full batch, NOT repacked
    prompt_len: torch.Tensor              # [B] long
    bonus: torch.Tensor                   # [B] long — newest committed token
    missing: torch.Tensor                 # [B] long — p_T not yet in the
                                          # assistant cache, else -1 (I4)
    tgt_cache: DynamicCache
    tgt_mask: torch.Tensor                # [B, P_t] long
    ast_cache: Optional[DynamicCache]     # None when spec is off (T=0)
    ast_mask: Optional[torch.Tensor]

    def committed_len(self, device=None) -> torch.Tensor:
        t = torch.tensor([len(self.records[i].committed)
                          for i in self.seq_idx], dtype=torch.long)
        return t.to(device) if device is not None else t

    def repack(self, keep: torch.Tensor) -> None:
        """Drop finished sequences (keep = long indices into the batch)."""
        self.seq_idx = [self.seq_idx[i] for i in keep.tolist()]
        self.prompt_len = self.prompt_len[keep]
        self.bonus = self.bonus[keep]
        self.missing = self.missing[keep]
        self.tgt_cache.batch_select_indices(keep.to(self.tgt_mask.device))
        self.tgt_mask = self.tgt_mask[keep]
        if self.ast_cache is not None:
            self.ast_cache.batch_select_indices(
                keep.to(self.ast_mask.device))
            self.ast_mask = self.ast_mask[keep]

    def assert_invariants(self) -> None:
        """I5 mask-sum invariants — debug mode only."""
        cl = self.committed_len(self.tgt_mask.device)
        pl = self.prompt_len.to(self.tgt_mask.device)
        want_t = pl + cl - 1
        got_t = self.tgt_mask.sum(dim=1)
        assert torch.equal(got_t, want_t), \
            f"I5 target mask-sum: got {got_t.tolist()} want {want_t.tolist()}"
        if self.ast_mask is not None:
            has_missing = (self.missing.to(self.tgt_mask.device) >= 0).long()
            want_a = want_t - has_missing
            got_a = self.ast_mask.sum(dim=1)
            assert torch.equal(got_a, want_a), \
                f"I5 assistant mask-sum: got {got_a.tolist()} " \
                f"want {want_a.tolist()}"


# =========================================================================
# Core loop
# =========================================================================

# Quiesce the archer offload engine between forwards. The batch loop drives
# forwards back-to-back (draft_1..draft_T, verify) with none of the Python
# overhead HF's generate has between steps; the archer fetch/exec/merge
# threads never get to settle before the next forward's dispatch/merge/bmm
# starts, and the CUDA-stream/thread race that opens intermittently wedges
# the device after ~100+ cycles (a STOCHASTIC hang — same config completes on
# reruns; batch_hang_debug.md). A full device sync after each forward forces
# every engine kernel to land before the next forward touches the engine.
# Off (AUG_BATCH_NOSYNC=1) reproduces the race for diagnosis. hf backend has
# no engine so this is a cheap no-op there; at B>1 the sync amortises.
_BATCH_SYNC = os.environ.get("AUG_BATCH_NOSYNC") is None


def _forward(model, input_ids, cache, mask, cur_valid, position_ids):
    """One cached forward with explicit logical positions. `cur_valid`
    [B, t] marks which of the newly fed slots are real (pads/holes-to-be
    are still fed as valid=1 unless known-invalid at feed time, e.g. the
    width-2 draft step-1 pad slot)."""
    attn = torch.cat([mask, cur_valid], dim=1)
    out = model(input_ids=input_ids, past_key_values=cache,
                attention_mask=attn, position_ids=position_ids,
                use_cache=True)
    logits = out.logits
    if _BATCH_SYNC and logits.is_cuda:
        torch.cuda.synchronize(logits.device)
    return logits, torch.cat([mask, cur_valid], dim=1)


@torch.no_grad()
def spec_decode_batch(model, controller, state: BatchState, *, T: int,
                      max_new_tokens: int, eos_id: Optional[int],
                      pad_id: int, engine=None,
                      on_cycle: Optional[Callable[[], None]] = None,
                      debug_invariants: bool = False) -> None:
    """Run synchronized speculative cycles until every sequence finishes.
    Mutates `state` (and its `records`). `controller` only needs a writable
    `in_draft_phase` attribute (None allowed when T == 0)."""
    dev = state.tgt_mask.device
    n_cycles = 0
    while state.seq_idx:
        B = len(state.seq_idx)
        # Heartbeat: keeps the stall watchdog fed during long batches and
        # localises a hang to a cycle index (a batch prints nothing else
        # until it completes — job 260924 stalled invisibly without this).
        n_cycles += 1
        if n_cycles % 25 == 0:
            done = sum(len(state.records[i].committed)
                       for i in state.seq_idx)
            print(f"    [cycle {n_cycles}] B={B} committed≈{done}",
                  flush=True)
        cl = state.committed_len(dev)
        pl = state.prompt_len.to(dev)

        # ── draft phase ──────────────────────────────────────────────
        proposals = torch.full((B, T), pad_id, dtype=torch.long, device=dev)
        if T > 0:
            controller.in_draft_phase = True
            if engine is not None:
                engine.on_draft_start()
            # step-1, width 2: [missing p_T (if any), bonus]  (I4)
            has0 = (state.missing >= 0)
            ast_len = pl + cl - 1 - has0.long()   # assistant logical length
            col0 = torch.where(has0, state.missing, state.bonus)  # pad value
            inp = torch.stack([col0, state.bonus], dim=1)          # [B, 2]
            pos0 = ast_len
            pos1 = ast_len + has0.long()
            positions = torch.stack([pos0, pos1], dim=1)
            valid = torch.stack(
                [has0.long(), torch.ones(B, dtype=torch.long, device=dev)],
                dim=1)
            logits, state.ast_mask = _forward(
                model, inp, state.ast_cache, state.ast_mask, valid, positions)
            prev = logits[:, -1].float().argmax(dim=-1)
            proposals[:, 0] = prev
            next_pos = pos1 + 1
            for t in range(1, T):
                ones = torch.ones(B, 1, dtype=torch.long, device=dev)
                logits, state.ast_mask = _forward(
                    model, prev[:, None], state.ast_cache, state.ast_mask,
                    ones, next_pos[:, None])
                prev = logits[:, -1].float().argmax(dim=-1)
                proposals[:, t] = prev
                next_pos = next_pos + 1
            # slots of p_1..p_{T-1}: fed at steps 1..T-1 → the last T-1
            # appended columns. Remember where they start for hole-flips.
            ast_pstart = state.ast_mask.shape[1] - (T - 1) if T > 1 else None
            controller.in_draft_phase = False
            if engine is not None:
                engine.on_draft_end()

        # ── verify phase ─────────────────────────────────────────────
        inp = torch.cat([state.bonus[:, None], proposals], dim=1)  # [B,T+1]
        positions = (pl + cl - 1)[:, None] + torch.arange(
            T + 1, dtype=torch.long, device=dev)[None]
        ones = torch.ones(B, T + 1, dtype=torch.long, device=dev)
        tgt_vstart = state.tgt_mask.shape[1]
        logits, state.tgt_mask = _forward(
            model, inp, state.tgt_cache, state.tgt_mask, ones, positions)
        pred = logits.float().argmax(dim=-1)                       # [B,T+1]
        if T > 0:
            match = pred[:, :T] == proposals
            k = ((~match).cumsum(dim=1) == 0).sum(dim=1)           # [B]
        else:
            k = torch.zeros(B, dtype=torch.long, device=dev)
        bonus_next = pred.gather(1, k[:, None]).squeeze(1)

        # ── bookkeeping (I2/I3/I4) ───────────────────────────────────
        # target holes: slot j (1-based among p) valid iff j <= k
        jj = torch.arange(1, T + 1, device=dev)[None]              # [1,T]
        state.tgt_mask[:, tgt_vstart + 1:] = (jj <= k[:, None]).long()
        if T > 1:
            # assistant slots hold p_1..p_{T-1}; p_j committed iff j <= k
            jja = torch.arange(1, T, device=dev)[None]             # [1,T-1]
            state.ast_mask[:, ast_pstart:] = (jja <= k[:, None]).long()
        state.missing = torch.where(
            k == T, proposals[:, T - 1] if T > 0 else state.missing,
            torch.full_like(state.missing, -1))

        finished_now = []
        for b in range(B):
            rec = state.records[state.seq_idx[b]]
            ki = int(k[b])
            ext = proposals[b, :ki].tolist() + [int(bonus_next[b])]
            if T > 0:
                rec.n_prop += T
                rec.n_acc += ki
                rec.accept_lens.append(ki + 1)
            # EOS: cut at the first EOS (inclusive), then freeze.
            if eos_id is not None and eos_id in ext:
                ext = ext[:ext.index(eos_id) + 1]
                rec.finished = True
            # max_new_tokens budget (committed counts every generated token
            # incl. the prefill TTFT token).
            room = max_new_tokens - len(rec.committed)
            if len(ext) >= room:
                ext = ext[:room]
                rec.finished = True
            rec.committed.extend(ext)
            if rec.finished:
                finished_now.append(b)
                # a truncated ext desyncs bonus/caches — irrelevant, the
                # sequence is repacked out below and never runs again.
        state.bonus = bonus_next

        if debug_invariants and not finished_now:
            state.assert_invariants()

        if on_cycle is not None:
            on_cycle()

        if finished_now:
            keep = torch.tensor([b for b in range(B)
                                 if b not in set(finished_now)],
                                dtype=torch.long)
            if keep.numel() == 0:
                state.seq_idx = []
                return
            state.repack(keep)


# =========================================================================
# Driver: SpecBench over batches
# =========================================================================

@torch.no_grad()
def _prefill_one(model, ids: torch.Tensor,
                 before_generate=None) -> Tuple[DynamicCache, int]:
    """batch-1 prefill (kMaxTokens guard: never feed a whole batch of
    prompts through one MoE forward). Returns (cache, TTFT token)."""
    assert ids.shape[1] <= 2048, \
        f"prompt of {ids.shape[1]} tokens exceeds the engine kMaxTokens=2048"
    if before_generate is not None:
        before_generate(ids)
    cache = DynamicCache()
    out = model(input_ids=ids, past_key_values=cache,
                attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(ids.shape[1],
                                          device=ids.device)[None],
                use_cache=True)
    ttft = int(out.logits[0, -1].float().argmax())
    return cache, ttft


def run_specbench_batched(
    target_model,
    tokenizer,
    controller,                        # None → non-speculative (T=0)
    *,
    num_speculative: int,
    batch_size: int,
    questions_per_cat: int,
    max_new_tokens: int,
    output_dir,
    label: str = "",
    seed: int = 0,
    spec_bench_cache=None,
    skip_categories=None,
    include_humaneval: bool = False,
    mt_bench_pooled: bool = False,
    before_generate=None,
    debug_invariants: bool = False,
):
    """Batched analogue of `run_specbench` (same question protocol, same
    greedy target-exact semantics). Returns a `SpecBenchResult`.

    Metric caveats (batch_spec_plan.md §4.3): per-question TPS =
    seq_tokens / batch_wall (co-scheduled); the honest headline number is
    `overall.tokens_per_second` = Σtokens / Σbatch_walls. Per-subtask TPS
    is NOT defined for B > 1 and is reported as 0.0.
    """
    from pathlib import Path

    from .loader import get_model_device
    from .specbench import (
        QuestionResult, SpecBenchResult, _aggregate_subtask,
        _format_chat_prompt, _load_humaneval_questions,
        _load_spec_bench_questions, _print_final_table, _sample_questions,
        SPEC_BENCH_SUBTASKS)

    spec = controller is not None
    T = num_speculative if spec else 0
    device = get_model_device(target_model)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    cache_dir = (spec_bench_cache if spec_bench_cache is not None
                 else Path.cwd() / "data" / "spec_bench")
    all_q = _load_spec_bench_questions(cache_dir)
    if include_humaneval:
        all_q = all_q + _load_humaneval_questions(cache_dir.parent / "humaneval")
    questions, skip = _sample_questions(all_q, questions_per_cat, seed,
                                        skip_categories, mt_bench_pooled)
    print(f"  [batch_loop] B={batch_size} spec={spec} T={T} "
          f"questions={len(questions)} "
          f"batches={-(-len(questions) // batch_size)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_question: List[QuestionResult] = []
    batch_walls: List[float] = []
    engine = getattr(controller, "merge_engine", None) if spec else None

    # Per-question CSV is written INCREMENTALLY (flushed after every batch) so
    # partial results survive a kill/timeout and progress is watchable live —
    # matching the legacy run_specbench. Final aggregation happens after.
    import csv
    _CSV_FIELDS = ["question_id", "category", "num_cycles", "num_new_tokens",
                   "wall_time_s", "mean_accept_length", "acceptance_rate",
                   "tokens_per_second", "batch_size"]
    per_q_f = open(output_dir / "per_question_summary.csv", "w", newline="",
                   encoding="utf-8")
    per_q_writer = csv.DictWriter(per_q_f, fieldnames=_CSV_FIELDS)
    per_q_writer.writeheader()
    per_q_f.flush()

    def _write_qres(qr: "QuestionResult") -> None:
        per_q_writer.writerow({
            "question_id": qr.qid, "category": qr.category,
            "num_cycles": qr.num_cycles, "num_new_tokens": qr.num_new_tokens,
            "wall_time_s": f"{qr.wall_time_s:.4f}",
            "mean_accept_length": f"{qr.mean_accept_length:.4f}",
            "acceptance_rate": f"{qr.acceptance_rate:.4f}",
            "tokens_per_second": f"{qr.tokens_per_second:.4f}",
            "batch_size": batch_size})

    # AUG_BATCH_REPLICATE (diagnostic, batch_hang_debug/affinity §1): make
    # every batch B identical copies of ONE question (diversity = 0). If
    # acceptance then returns to the B=1 level, the batch AccR drop is purely
    # the pooled-draft-over-diverse-sequences effect (not a batching bug),
    # and affinity batching can recover it. One batch per question here.
    _replicate = os.environ.get("AUG_BATCH_REPLICATE") is not None
    _iter = ([[q] * batch_size for q in questions] if _replicate
             else [questions[b0:b0 + batch_size]
                   for b0 in range(0, len(questions), batch_size)])

    for _bi, batch_q in enumerate(_iter):
        b0 = _bi * batch_size
        t0 = time.perf_counter()
        if spec:
            controller.reset()          # per-batch reset (B=1 ≡ legacy)

        records, tgt_caches, ttfts, plens = [], [], [], []
        for q in batch_q:
            user_msg = q["turns"][0]
            prompt = (user_msg if q["category"] == "humaneval"
                      else _format_chat_prompt(tokenizer, [], user_msg))
            ids = tokenizer(prompt, return_tensors="pt"
                            ).input_ids.to(device)
            cache, ttft = _prefill_one(target_model, ids, before_generate)
            rec = SeqRecord(qid=str(q["question_id"]),
                            category=q["category"],
                            prompt_len=ids.shape[1], committed=[ttft])
            records.append(rec)
            tgt_caches.append(cache)
            ttfts.append(ttft)
            plens.append(ids.shape[1])

        if spec:
            controller.update_masks()   # C-BOOT: build draft state from
                                        # the pooled prefill captures
            ast_caches = [copy.deepcopy(c) for c in tgt_caches]  # KV-copy
            ast_cache, ast_mask = stack_caches(ast_caches)
        else:
            ast_cache, ast_mask = None, None
        tgt_cache, tgt_mask = stack_caches(tgt_caches)

        state = BatchState(
            seq_idx=list(range(len(records))), records=records,
            prompt_len=torch.tensor(plens, dtype=torch.long, device=device),
            bonus=torch.tensor(ttfts, dtype=torch.long, device=device),
            missing=torch.full((len(records),), -1, dtype=torch.long,
                               device=device),
            tgt_cache=tgt_cache, tgt_mask=tgt_mask.to(device),
            ast_cache=ast_cache,
            ast_mask=ast_mask.to(device) if ast_mask is not None else None)

        on_cycle = controller.update_masks if spec else None
        spec_decode_batch(target_model, controller, state, T=T,
                          max_new_tokens=max_new_tokens, eos_id=eos_id,
                          pad_id=pad_id, engine=engine, on_cycle=on_cycle,
                          debug_invariants=debug_invariants)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        batch_walls.append(wall)

        if os.environ.get("AUG_DUMP_COMMITTED"):
            import json as _json
            with open(output_dir / "committed.jsonl", "a") as _f:
                for rec in records:
                    _f.write(_json.dumps(
                        {"qid": rec.qid, "committed": rec.committed}) + "\n")
        for rec in records:
            n_cyc = len(rec.accept_lens)
            mat = (sum(rec.accept_lens) / n_cyc) if n_cyc else 0.0
            qr = QuestionResult(
                qid=rec.qid, category=rec.category, num_cycles=n_cyc,
                num_new_tokens=len(rec.committed), wall_time_s=wall,
                mean_accept_length=mat,
                acceptance_rate=(rec.n_acc / rec.n_prop
                                 if rec.n_prop else 0.0),
                tokens_per_second=len(rec.committed) / wall,
                n_proposed=rec.n_prop, n_accepted=rec.n_acc,
                accept_lengths=rec.accept_lens)
            per_question.append(qr)
            _write_qres(qr)
        per_q_f.flush()                         # survive kill/timeout
        done = (_bi + 1) if _replicate else b0 + len(batch_q)
        total = len(questions) if _replicate else len(questions)
        print(f"  [batch {_bi + 1}] {done}/{total} q, "
              f"wall={wall:.1f}s, "
              f"tok={sum(len(r.committed) for r in records)}", flush=True)

    per_q_f.close()

    # ── aggregation: pooled acc metrics; TPS from batch walls only ──
    per_subtask: Dict[str, Dict[str, Any]] = {}
    for subtask in SPEC_BENCH_SUBTASKS:
        m = _aggregate_subtask(per_question, subtask)
        if m is not None:
            if batch_size > 1 and subtask != "overall":
                m["tokens_per_second"] = 0.0   # undefined under co-scheduling
            per_subtask[subtask] = m
    total_tokens = sum(qr.num_new_tokens for qr in per_question)
    if "overall" in per_subtask:
        per_subtask["overall"]["tokens_per_second"] = (
            total_tokens / sum(batch_walls) if batch_walls else 0.0)

    # overall_summary.csv (per-subtask aggregates) — written once at the end,
    # like the legacy runner; per_question_summary.csv was streamed above.
    with open(output_dir / "overall_summary.csv", "w", newline="",
              encoding="utf-8") as f:
        ow = csv.DictWriter(f, fieldnames=[
            "subtask", "num_questions", "total_cycles",
            "mean_accept_tokens", "acceptance_rate", "tokens_per_second"])
        ow.writeheader()
        for subtask in SPEC_BENCH_SUBTASKS:
            m = per_subtask.get(subtask)
            if m is None:
                continue
            ow.writerow({
                "subtask": subtask, "num_questions": m["num_questions"],
                "total_cycles": m["total_cycles"],
                "mean_accept_tokens": f"{m['mean_accept_tokens']:.4f}",
                "acceptance_rate": f"{m['acceptance_rate']:.4f}",
                "tokens_per_second": f"{m['tokens_per_second']:.4f}"})

    _print_final_table(label, per_subtask)
    print(f"  [batch_loop] overall TPS (Σtok/Σwall) = "
          f"{per_subtask.get('overall', {}).get('tokens_per_second', 0):.4f}")
    overall = per_subtask.get("overall", {
        "mean_accept_tokens": 0.0, "acceptance_rate": 0.0,
        "tokens_per_second": 0.0, "num_questions": 0, "total_cycles": 0})
    return SpecBenchResult(per_question=per_question,
                           per_subtask=per_subtask, overall=overall)
