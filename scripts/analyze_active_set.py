#!/usr/bin/env python3
"""Two questions about the per-cycle selected expert sets.

Reads the JSONL from AUG_DUMP_ACTIVE_SET (one record per (layer, question,
cycle): the selected expert ids = experts kept after the top-M cutoff). Expert
ids are layer-local, so everything is computed *within* a layer.

Q1  Co-occurring expert pairs
    For each layer, over all cycles, count how often each pair (i,j) is jointly
    selected. Report both raw co-occurrence and *lift* (a.k.a. observed/expected
    ratio): lift(i,j) = P(i,j) / (P(i)P(j)). lift=1 means the two experts are
    selected together exactly as often as chance predicts from their individual
    rates; lift>>1 means a genuine pairing beyond just "both are popular".

Q2  Cycle-to-cycle set shift
    Within each (layer, question) the count vector is overwritten each cycle
    (not accumulated), so consecutive cycles are independent verify windows.
    For each adjacent cycle pair compute Jaccard overlap, #added, #removed.
    Compare the observed mean overlap to the random baseline (two sets of the
    same sizes drawn uniformly from n experts) to judge whether the set is
    "sticky" (persists) or genuinely reshuffles.

Usage: analyze_active_set.py <active_set.jsonl> [out_dir]
"""
import collections
import itertools
import json
import math
import statistics as st
import sys
from pathlib import Path


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[int(k)] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    dump = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else dump.parent

    # layer -> list of (qid, cycle, set(active))
    per_layer = collections.defaultdict(list)
    n_experts = 0
    n_degen = 0
    total = 0
    with open(dump) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1
            n_experts = max(n_experts, r["n"])
            if r.get("degenerate"):
                n_degen += 1
                continue
            per_layer[r["layer"]].append(
                (r["qid"], r["cycle"], set(r["active"])))

    if not per_layer:
        print(f"[analyze] no usable records in {dump}")
        sys.exit(1)

    lines = []
    def out(s=""):
        lines.append(s); print(s)

    out("=" * 74)
    out("  PER-CYCLE SELECTED EXPERT SETS — co-occurrence & set-shift")
    out(f"  source: {dump}")
    out("=" * 74)
    out(f"  records            : {total}  (degenerate/all-zero dropped: {n_degen})")
    out(f"  experts per layer  : {n_experts}")
    out(f"  layers             : {len(per_layer)}")
    set_sizes = [len(s) for recs in per_layer.values() for _, _, s in recs]
    out(f"  selected-set size  : mean={st.mean(set_sizes):.2f} "
        f"median={pct(set_sizes,50):.0f} min={min(set_sizes)} max={max(set_sizes)}")
    out("")

    # ── Q1: co-occurrence ────────────────────────────────────────────────
    out("-" * 74)
    out("  Q1. CO-OCCURRING EXPERT PAIRS")
    out("-" * 74)
    MIN_SUPPORT = 30                 # ignore pairs co-selected < this many cycles
    best_pairs = []                  # (lift, layer, i, j, co, cycles, pi, pj)
    per_layer_maxlift = []
    for layer, recs in per_layer.items():
        C = len(recs)
        solo = collections.Counter()
        pair = collections.Counter()
        for _, _, s in recs:
            for e in s:
                solo[e] += 1
            for i, j in itertools.combinations(sorted(s), 2):
                pair[(i, j)] += 1
        layer_max = 0.0
        for (i, j), co in pair.items():
            if co < MIN_SUPPORT:
                continue
            pi, pj, pij = solo[i] / C, solo[j] / C, co / C
            lift = pij / (pi * pj)
            layer_max = max(layer_max, lift)
            best_pairs.append((lift, layer, i, j, co, C, pi, pj))
        per_layer_maxlift.append(layer_max)

    lifts = [b[0] for b in best_pairs]
    out(f"  pairs with co-support >= {MIN_SUPPORT}: {len(best_pairs)}")
    if lifts:
        out(f"  lift over those pairs : mean={st.mean(lifts):.3f} "
            f"median={pct(lifts,50):.3f} p90={pct(lifts,90):.3f} "
            f"p99={pct(lifts,99):.3f} max={max(lifts):.3f}")
        out(f"  (lift=1.0 => co-occur exactly as chance predicts)")
        frac_strong = sum(1 for x in lifts if x >= 1.5) / len(lifts)
        out(f"  fraction of pairs with lift>=1.5: {100*frac_strong:.1f}%")
        out(f"  per-layer max lift    : mean={st.mean(per_layer_maxlift):.3f} "
            f"max={max(per_layer_maxlift):.3f}")
    out("")
    out("  Top 20 pairs by lift (genuine pairings beyond individual popularity):")
    out(f"  {'layer':>5s} {'e_i':>4s} {'e_j':>4s} {'lift':>6s} {'co':>6s} "
        f"{'cyc':>5s} {'P(i)':>5s} {'P(j)':>5s} {'P(j|i)':>6s}")
    for lift, layer, i, j, co, C, pi, pj in sorted(best_pairs, reverse=True)[:20]:
        out(f"  {layer:5d} {i:4d} {j:4d} {lift:6.2f} {co:6d} {C:5d} "
            f"{pi:5.2f} {pj:5.2f} {co/ (pi*C):6.2f}")
    out("")
    out("  Top 10 pairs by RAW co-occurrence (often just two popular experts):")
    out(f"  {'layer':>5s} {'e_i':>4s} {'e_j':>4s} {'co':>6s} {'cyc':>5s} "
        f"{'co/cyc':>6s} {'lift':>6s}")
    for lift, layer, i, j, co, C, pi, pj in sorted(
            best_pairs, key=lambda b: -b[4])[:10]:
        out(f"  {layer:5d} {i:4d} {j:4d} {co:6d} {C:5d} {co/C:6.2f} {lift:6.2f}")
    out("")

    # ── Q2: cycle-to-cycle set shift ─────────────────────────────────────
    out("-" * 74)
    out("  Q2. CYCLE-TO-CYCLE SET SHIFT (consecutive cycles, same question)")
    out("-" * 74)
    jaccards = []
    turnovers = []          # (added+removed) / union
    added_fr = []           # fraction of new set that is freshly added
    exp_jaccards = []       # random baseline per transition
    n_trans = 0
    for layer, recs in per_layer.items():
        byq = collections.defaultdict(list)
        for qid, cyc, s in recs:
            byq[qid].append((cyc, s))
        for qid, seq in byq.items():
            seq.sort()
            for (_, a), (_, b) in zip(seq, seq[1:]):
                if not a and not b:
                    continue
                inter = len(a & b)
                union = len(a | b)
                if union == 0:
                    continue
                n_trans += 1
                jaccards.append(inter / union)
                turnovers.append((len(a - b) + len(b - a)) / union)
                added_fr.append((len(b - a) / len(b)) if b else 0.0)
                # random baseline: E[|a∩b|] = |a||b|/n
                ei = len(a) * len(b) / n_experts
                exp_jaccards.append(ei / (len(a) + len(b) - ei)
                                    if (len(a) + len(b) - ei) > 0 else 0.0)

    out(f"  transitions analysed : {n_trans}")
    out(f"  Jaccard overlap      : mean={st.mean(jaccards):.3f} "
        f"median={pct(jaccards,50):.3f} p10={pct(jaccards,10):.3f} "
        f"p90={pct(jaccards,90):.3f}")
    out(f"  random-baseline Jacc : mean={st.mean(exp_jaccards):.3f}  "
        f"(if selection were i.i.d. uniform over {n_experts} experts)")
    out(f"  turnover (changed/union): mean={st.mean(turnovers):.3f} "
        f"median={pct(turnovers,50):.3f}")
    out(f"  newly-added fraction : mean={st.mean(added_fr):.3f} "
        f"(share of next cycle's set that was NOT in the previous cycle)")
    stick = st.mean(jaccards) / st.mean(exp_jaccards) if st.mean(exp_jaccards) else float('nan')
    out("")
    out(f"  observed/random overlap ratio = {stick:.2f}x")
    if stick > 1.3:
        out("  => sets are STICKIER than random: a stable core persists across cycles,")
        out("     but the rest still churns (see turnover above).")
    else:
        out("  => sets reshuffle roughly as much as random selection would.")

    # ── write artefacts ──────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "active_set_stats.txt"
    report.write_text("\n".join(lines) + "\n")

    summary = {
        "n_records": total, "n_degenerate": n_degen,
        "n_experts": n_experts, "n_layers": len(per_layer),
        "selected_set_size": {
            "mean": st.mean(set_sizes), "median": pct(set_sizes, 50),
            "min": min(set_sizes), "max": max(set_sizes)},
        "cooccurrence": {
            "min_support": MIN_SUPPORT,
            "n_pairs": len(best_pairs),
            "lift_mean": (st.mean(lifts) if lifts else None),
            "lift_median": (pct(lifts, 50) if lifts else None),
            "lift_p99": (pct(lifts, 99) if lifts else None),
            "lift_max": (max(lifts) if lifts else None),
            "frac_lift_ge_1.5": (sum(1 for x in lifts if x >= 1.5) / len(lifts)
                                 if lifts else None),
        },
        "set_shift": {
            "n_transitions": n_trans,
            "jaccard_mean": st.mean(jaccards),
            "jaccard_median": pct(jaccards, 50),
            "random_jaccard_mean": st.mean(exp_jaccards),
            "stickiness_ratio": stick,
            "turnover_mean": st.mean(turnovers),
            "added_fraction_mean": st.mean(added_fr),
        },
    }
    (out_dir / "active_set_stats.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    import csv
    with open(out_dir / "top_cooccur_pairs.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["layer", "e_i", "e_j", "lift", "co_count",
                     "n_cycles", "p_i", "p_j"])
        for lift, layer, i, j, co, C, pi, pj in sorted(best_pairs, reverse=True):
            wr.writerow([layer, i, j, f"{lift:.4f}", co, C,
                         f"{pi:.4f}", f"{pj:.4f}"])

    print(f"\n[analyze] wrote {report}")
    print(f"[analyze] wrote {out_dir / 'active_set_stats.json'}")
    print(f"[analyze] wrote {out_dir / 'top_cooccur_pairs.csv'}")


if __name__ == "__main__":
    main()
