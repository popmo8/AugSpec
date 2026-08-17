#!/usr/bin/env python
"""V1 gate check (batch_spec_plan.md §6): per-question exact comparison of
(num_new_tokens, num_cycles, MAT, AccR) between the legacy HF path and the
batch loop at B=1. Any greedy token divergence necessarily shifts cycle
boundaries, so exact equality of these four is text-equality evidence."""
import csv, sys
def load(p):
    with open(p) as f:
        return {r["question_id"]: r for r in csv.DictReader(f)}
a = load(sys.argv[1]); b = load(sys.argv[2])
assert a.keys() == b.keys(), f"question sets differ: {a.keys() ^ b.keys()}"
bad = []
for qid in a:
    fields = ("num_new_tokens", "num_cycles", "mean_accept_length",
              "acceptance_rate")
    diffs = {f: (a[qid][f], b[qid][f]) for f in fields
             if a[qid][f] != b[qid][f]}
    if diffs:
        bad.append((qid, diffs))
print(f"[V1] {len(a) - len(bad)}/{len(a)} questions identical")
for qid, d in bad:
    print(f"  MISMATCH {qid}: {d}")
n = len(a)
ok = (n - len(bad)) >= n - 1        # tolerance: ≤1 divergent question
print("[V1] GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
