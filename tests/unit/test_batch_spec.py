"""V0 tests for the batched speculative loop (batch_spec_plan.md §6).

A deterministic FakeCausalLM (logits = one-hot of f(token, position),
distinct draft/target tables, records every call's ids/positions/masks)
drives `spec_decode_batch`; results are checked token-by-token against
two independent pure-Python references:
  * `rollout`  — target greedy: spec decoding MUST reproduce it exactly,
                 whatever the draft does (target-exact invariant).
  * `ref_spec` — a 30-line reference speculative simulator for per-cycle
                 accept lengths.
Pure CPU — login-node safe.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, List

import pytest
import torch
from transformers import DynamicCache

from aug_spec.runtime.batch_spec import (
    BatchState, SeqRecord, _prefill_one, spec_decode_batch, stack_caches)

VOCAB = 32
EOS = 31
PAD = 0


class Phase:
    in_draft_phase = False


class FakeLM:
    """logits[b, j] = one-hot(f(input_ids[b, j], position_ids[b, j])).
    f switches on the phase flag (draft vs target), mimicking the shared
    model whose MoE forwards branch on controller.in_draft_phase."""

    def __init__(self, target_f: Callable, draft_f: Callable, phase: Phase,
                 n_layers: int = 2):
        self.target_f, self.draft_f, self.phase = target_f, draft_f, phase
        self.n_layers = n_layers
        self.calls: List[dict] = []

    def __call__(self, input_ids=None, past_key_values=None,
                 attention_mask=None, position_ids=None, use_cache=True):
        f = self.draft_f if self.phase.in_draft_phase else self.target_f
        B, t = input_ids.shape
        nxt = torch.tensor(
            [[f(int(input_ids[b, j]), int(position_ids[b, j]))
              for j in range(t)] for b in range(B)])
        logits = torch.nn.functional.one_hot(nxt, VOCAB).float() * 7.0
        for li in range(self.n_layers):
            past_key_values.update(torch.zeros(B, 1, t, 1),
                                   torch.zeros(B, 1, t, 1), li)
        self.calls.append({
            "draft": self.phase.in_draft_phase,
            "ids": input_ids.clone(), "pos": position_ids.clone(),
            "mask": attention_mask.clone()})
        return SimpleNamespace(logits=logits,
                               past_key_values=past_key_values)


# ── pure-python references ──────────────────────────────────────────────

def rollout(target_f, prompt, mnt, eos=None):
    committed, tok, pos = [], prompt[-1], len(prompt) - 1
    while len(committed) < mnt:
        tok = target_f(tok, pos)
        pos += 1
        committed.append(tok)
        if eos is not None and tok == eos:
            break
    return committed


def ref_spec(target_f, draft_f, prompt, T, mnt, eos=None):
    """Reference speculative sim → (committed, accept_lens)."""
    committed = [target_f(prompt[-1], len(prompt) - 1)]     # TTFT token
    accept_lens = []
    if (eos is not None and committed[0] == eos) or mnt <= 1:
        return committed[:mnt], accept_lens
    while True:
        base = len(prompt) + len(committed) - 1              # bonus position
        tok, pos, props = committed[-1], base, []
        for _ in range(T):
            tok = draft_f(tok, pos)
            props.append(tok)
            pos += 1
        tok_t, pos_t, k = committed[-1], base, 0
        for j in range(T):
            if props[j] == target_f(tok_t, pos_t):
                k += 1
                tok_t, pos_t = props[j], pos_t + 1
            else:
                break
        ext = props[:k] + [target_f(tok_t, pos_t)]
        accept_lens.append(k + 1)
        done = False
        if eos is not None and eos in ext:
            ext = ext[:ext.index(eos) + 1]
            done = True
        room = mnt - len(committed)
        if len(ext) >= room:
            ext, done = ext[:room], True
        committed += ext
        if done:
            return committed, accept_lens


# ── harness ─────────────────────────────────────────────────────────────

def make_state(model, prompts, spec=True):
    records, caches, ttfts = [], [], []
    for i, p in enumerate(prompts):
        ids = torch.tensor(p)[None]
        cache, ttft = _prefill_one(model, ids)
        records.append(SeqRecord(qid=str(i), category="qa",
                                 prompt_len=len(p), committed=[ttft]))
        caches.append(cache)
        ttfts.append(ttft)
    tgt_cache, tgt_mask = stack_caches(caches)
    if spec:
        import copy
        ast_cache, ast_mask = stack_caches([copy.deepcopy(c) for c in caches])
    else:
        ast_cache, ast_mask = None, None
    return BatchState(
        seq_idx=list(range(len(prompts))), records=records,
        prompt_len=torch.tensor([len(p) for p in prompts]),
        bonus=torch.tensor(ttfts),
        missing=torch.full((len(prompts),), -1, dtype=torch.long),
        tgt_cache=tgt_cache, tgt_mask=tgt_mask,
        ast_cache=ast_cache, ast_mask=ast_mask)


def run_case(target_f, draft_f, prompts, T, mnt, eos=EOS, spec=True):
    phase = Phase()
    model = FakeLM(target_f, draft_f, phase)
    state = make_state(model, prompts, spec=spec)
    spec_decode_batch(model, phase, state, T=T, max_new_tokens=mnt,
                      eos_id=eos, pad_id=PAD, debug_invariants=True)
    return state.records, model


def check_against_refs(records, target_f, draft_f, prompts, T, mnt, eos=EOS):
    for rec, p in zip(records, prompts):
        assert rec.committed == rollout(target_f, p, mnt, eos), \
            f"target-exact violated for prompt {p}"
        if T > 0:
            _, ref_lens = ref_spec(target_f, draft_f, p, T, mnt, eos)
            assert rec.accept_lens == ref_lens, \
                f"accept_lens mismatch for prompt {p}"


# deterministic token maps, values in 1..30 (no accidental EOS/PAD)
def tf(tok, pos):
    return (tok * 7 + pos * 3 + 5) % 29 + 1


def df_same(tok, pos):
    return tf(tok, pos)


def df_wrong(tok, pos):
    return (tf(tok, pos) % 29) + 1        # ≠ tf everywhere


def df_mixed(tok, pos):
    return tf(tok, pos) if pos % 3 else df_wrong(tok, pos)


PROMPTS = [[3, 5, 9], [4, 4, 4, 4, 4], [7]]


class TestSpecDecodeBatch:
    def test_all_accept_exercises_missing_pT(self):
        # draft == target → k=T every cycle → the k==T "p_T never fed to
        # the assistant" corner (I4) fires on EVERY cycle.
        recs, _ = run_case(tf, df_same, PROMPTS, T=3, mnt=17)
        check_against_refs(recs, tf, df_same, PROMPTS, 3, 17)
        assert all(l == 4 for r in recs for l in r.accept_lens[:-1])

    def test_all_reject(self):
        recs, _ = run_case(tf, df_wrong, PROMPTS, T=3, mnt=10)
        check_against_refs(recs, tf, df_wrong, PROMPTS, 3, 10)
        assert all(l == 1 for r in recs for l in r.accept_lens)

    def test_ragged_mixed_batch(self):
        recs, _ = run_case(tf, df_mixed, PROMPTS, T=4, mnt=23)
        check_against_refs(recs, tf, df_mixed, PROMPTS, 4, 23)
        lens = {l for r in recs for l in r.accept_lens}
        assert len(lens) > 1                     # genuinely ragged

    def test_eos_freezes_sequence(self):
        def tf_eos(tok, pos):
            return EOS if pos == 6 else tf(tok, pos)
        recs, _ = run_case(tf_eos, df_mixed, PROMPTS, T=3, mnt=50)
        check_against_refs(recs, tf_eos, df_mixed, PROMPTS, 3, 50)
        for rec in recs:
            assert rec.committed[-1] == EOS
            assert EOS not in rec.committed[:-1]

    def test_mnt_budget_exact(self):
        recs, _ = run_case(tf, df_same, PROMPTS, T=5, mnt=13, eos=None)
        for rec in recs:
            assert len(rec.committed) == 13

    def test_T0_plain_greedy(self):
        # spec off (MoE-Caching mode): committed == target rollout,
        # no proposals counted.
        recs, model = run_case(tf, df_wrong, PROMPTS, T=0, mnt=9, spec=False)
        for rec, p in zip(recs, PROMPTS):
            assert rec.committed == rollout(tf, p, 9, EOS)
            assert rec.n_prop == 0 and rec.accept_lens == []
        assert not any(c["draft"] for c in model.calls)

    def test_batch_matches_single(self):
        # B=3 must produce byte-identical streams to three B=1 runs.
        recs_b, _ = run_case(tf, df_mixed, PROMPTS, T=3, mnt=15)
        for p, rec_b in zip(PROMPTS, recs_b):
            recs_1, _ = run_case(tf, df_mixed, [p], T=3, mnt=15)
            assert rec_b.committed == recs_1[0].committed
            assert rec_b.accept_lens == recs_1[0].accept_lens


class TestMaskAndPositions:
    def test_verify_masks_have_holes(self):
        # Single sequence, draft wrong at every step (k=0): after cycle 1
        # the T rejected proposal slots must be holes in cycle 2's mask.
        prompts = [[3, 5, 9]]
        recs, model = run_case(tf, df_wrong, prompts, T=3, mnt=8)
        verifies = [c for c in model.calls if not c["draft"]
                    and c["ids"].shape == (1, 4)]
        m2 = verifies[1]["mask"][0]
        P = 3                                    # prompt slots
        # layout: [prompt ×3 | v1: bonus p1 p2 p3 | v2 feed ×4]
        assert m2[:P].tolist() == [1, 1, 1]
        assert m2[P:P + 4].tolist() == [1, 0, 0, 0]     # k=0: only bonus
        assert m2[P + 4:].tolist() == [1, 1, 1, 1]      # current feed

    def test_positions_are_logical_not_physical(self):
        # With k=0 cycles, physical length outgrows logical; verify
        # positions must keep following the LOGICAL stream.
        prompts = [[2, 6]]
        recs, model = run_case(tf, df_wrong, prompts, T=2, mnt=6)
        verifies = [c for c in model.calls if not c["draft"]
                    and c["ids"].shape == (1, 3)]
        for i, c in enumerate(verifies):
            # cycle i: bonus logical position = prompt(2) + i cycles ×1 tok
            assert c["pos"][0].tolist() == [2 + i, 3 + i, 4 + i]

    def test_repack_shrinks_batch(self):
        def tf_eos(tok, pos):
            return EOS if (pos == 5 and tok % 2) else tf(tok, pos)
        recs, model = run_case(tf_eos, df_same, PROMPTS, T=2, mnt=30)
        check_against_refs(recs, tf_eos, df_same, PROMPTS, 2, 30)
        b_sizes = {c["ids"].shape[0] for c in model.calls}
        assert len(b_sizes) > 1                  # batch actually shrank


class TestCacheUtils:
    def test_stack_caches_alignment(self):
        caches = []
        for L in (3, 5):
            c = DynamicCache()
            k = torch.arange(float(L)).reshape(1, 1, L, 1)
            c.update(k, k.clone(), 0)
            caches.append(c)
        merged, mask = stack_caches(caches)
        assert tuple(merged.layers[0].keys.shape) == (2, 1, 5, 1)
        assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        assert merged.layers[0].keys[0, 0, :3, 0].tolist() == [0., 1., 2.]
        assert merged.layers[0].keys[0, 0, 3:, 0].tolist() == [0., 0.]
