#!/usr/bin/env python3
"""Cache-reuse analysis for cooccur_pair (from AUG_DUMP_PAIRS JSONL).

Per (layer, question), over consecutive refresh cycles t -> t+1:

  * both-present rate  — of the pairs {A,B} formed at cycle t, the fraction
    whose BOTH members are in cycle t+1's active set M (so the pair *could* be
    re-formed → the requested metric);
  * re-paired rate     — fraction of cycle-t pairs that cycle t+1 actually forms
    again (the strict consecutive cache-hit);

and run-level:

  * distinct pairs vs total pair-builds — with the immutable member-keyed cache
    (B3), only distinct (layer, {A,B}) pairs are ever merged; everything else is
    a hit. hit-rate = 1 - distinct/total.

Usage: analyze_pair_reuse.py <pairs.jsonl> [out_dir]
"""
import collections
import json
import statistics as s
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    dump = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else dump.parent

    seqs = collections.defaultdict(dict)        # (layer,qid) -> {cycle: rec}
    total_pairbuilds = 0
    distinct = set()                            # (layer, frozenset{A,B}) seen
    repeat_builds = 0
    with open(dump) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            seqs[(r["layer"], r["qid"])][r["cycle"]] = r
            for p in r["pairs"]:
                total_pairbuilds += 1
                key = (r["layer"], frozenset(p))
                if key in distinct:
                    repeat_builds += 1
                else:
                    distinct.add(key)

    both_present = repaired = total_prevpairs = 0
    per_layer = collections.defaultdict(lambda: [0, 0])   # layer -> [bp, total]
    for (layer, qid), cyc in seqs.items():
        order = sorted(cyc)
        for a, b in zip(order, order[1:]):
            prev = cyc[a]; cur = cyc[b]
            cur_active = set(cur["active"])
            cur_pairs = {frozenset(p) for p in cur["pairs"]}
            for p in prev["pairs"]:
                fp = frozenset(p)
                total_prevpairs += 1
                per_layer[layer][1] += 1
                if fp <= cur_active:
                    both_present += 1
                    per_layer[layer][0] += 1
                if fp in cur_pairs:
                    repaired += 1

    lines = []
    def out(x=""):
        lines.append(x); print(x)

    out("=" * 70)
    out("  cooccur_pair CACHE-REUSE analysis")
    out(f"  source: {dump}")
    out("=" * 70)
    nrec = sum(len(c) for c in seqs.values())
    out(f"  records {nrec} | layers {len({l for l,_ in seqs})} | "
        f"questions {len({q for _,q in seqs})}")
    out("")
    out("-" * 70)
    out("  CONSECUTIVE-CYCLE (t -> t+1, within a question)")
    out("-" * 70)
    out(f"  cycle-t pairs examined : {total_prevpairs}")
    if total_prevpairs:
        out(f"  both members still in M next cycle : {both_present} "
            f"({100*both_present/total_prevpairs:.1f}%)   ← requested metric")
        out(f"  pair actually re-formed next cycle : {repaired} "
            f"({100*repaired/total_prevpairs:.1f}%)   (strict consecutive hit)")
        bl = [100*bp/tot for bp, tot in per_layer.values() if tot]
        out(f"  per-layer both-present rate: mean={s.mean(bl):.1f}% "
            f"min={min(bl):.1f}% max={max(bl):.1f}%")
    out("")
    out("-" * 70)
    out("  RUN-LEVEL (immutable member-keyed cache, B3)")
    out("-" * 70)
    out(f"  total pair-builds      : {total_pairbuilds}")
    out(f"  distinct (layer,{{A,B}}) : {len(distinct)}")
    if total_pairbuilds:
        out(f"  cache hit-rate = 1 - distinct/total = "
            f"{100*(1 - len(distinct)/total_pairbuilds):.1f}%")
        out(f"  (= fraction of pair-merges a permanent cache would serve "
            f"without recompute)")

    out_dir.mkdir(parents=True, exist_ok=True)
    rep = out_dir / "pair_reuse_stats.txt"
    rep.write_text("\n".join(lines) + "\n")
    print(f"\n[analyze] wrote {rep}")


if __name__ == "__main__":
    main()
