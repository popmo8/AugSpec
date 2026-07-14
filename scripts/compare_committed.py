#!/usr/bin/env python
"""V1 definitive check: byte-identical committed token streams between the
legacy HF path and the batch loop (B=1). Target-exact greedy on the
deterministic hf backend ⇒ these MUST match token-for-token regardless of
how the draft behaves. Reports first divergence position per question."""
import json, sys
def load(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line); d[r["qid"]] = r["committed"]
    return d
a = load(sys.argv[1]); b = load(sys.argv[2])
assert a.keys() == b.keys(), f"qid sets differ: {a.keys() ^ b.keys()}"
n_ident = 0
for qid in sorted(a):
    ta, tb = a[qid], b[qid]
    if ta == tb:
        n_ident += 1; continue
    div = next((i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]),
               min(len(ta), len(tb)))
    print(f"  DIVERGE {qid}: len {len(ta)} vs {len(tb)}, first diff @tok {div}"
          f"  legacy…{ta[max(0,div-2):div+2]}  batch…{tb[max(0,div-2):div+2]}")
print(f"[V1-committed] {n_ident}/{len(a)} token-identical")
ok = n_ident == len(a)
print("[V1-committed] GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
