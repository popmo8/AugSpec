#!/usr/bin/env python3
"""Detailed adjacent-cycle overlap of the per-cycle selected expert sets.

Reads the AUG_DUMP_ACTIVE_SET JSONL (one record per (layer, question, cycle):
the selected expert ids = experts kept after the top-M cutoff). Everything is
within a layer (expert ids are layer-local) and within a question (the count
vector is overwritten, not accumulated, each cycle and cleared between
questions). Reports:

  1. adjacent-cycle (gap=1) overlap: Jaccard, retained/added/removed counts,
     overlap coefficient, fraction of the previous set that survives;
  2. overlap vs cycle gap (1..G): how fast the set "forgets";
  3. overlap by layer depth and by position-in-question;
  4. stable core: how many experts persist across all / >=80% / >=50% of a
     question's cycles, vs the mean per-cycle set size.

A random baseline (two sets of the same sizes drawn uniformly from n experts)
is reported for context.

Usage: analyze_cycle_overlap.py <active_set.jsonl> [out_dir]
"""
import collections
import json
import math
import statistics as st
import sys
from pathlib import Path

MAX_GAP = 10


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

    # (layer, qid) -> {cycle: set(active)}
    seqs = collections.defaultdict(dict)
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
            seqs[(r["layer"], r["qid"])][r["cycle"]] = set(r["active"])

    if not seqs:
        print(f"[analyze] no usable records in {dump}")
        sys.exit(1)

    lines = []
    def out(s=""):
        lines.append(s); print(s)

    set_sizes = [len(s) for cyc in seqs.values() for s in cyc.values()]
    n_layers = len({l for l, _ in seqs})
    n_q = len({q for _, q in seqs})
    out("=" * 76)
    out("  ADJACENT-CYCLE OVERLAP of per-cycle selected expert sets")
    out(f"  source: {dump}")
    out("=" * 76)
    out(f"  records {total} (degenerate dropped {n_degen}) | "
        f"layers {n_layers} | questions {n_q} | experts/layer {n_experts}")
    out(f"  per-cycle set size: mean={st.mean(set_sizes):.2f} "
        f"median={pct(set_sizes,50):.0f} min={min(set_sizes)} max={max(set_sizes)}")
    out("")

    # ── gap=1 adjacent metrics ───────────────────────────────────────────
    jac, overlap_coef, retained, added, removed = [], [], [], [], []
    frac_prev_survives, frac_next_isnew = [], []
    exp_jac = []
    by_layer = collections.defaultdict(list)        # layer -> jaccards
    by_pos = collections.defaultdict(list)          # cycle-pos bucket -> jaccards
    # gap -> list of jaccards
    by_gap = collections.defaultdict(list)

    for (layer, qid), cyc in seqs.items():
        order = sorted(cyc)
        for idx in range(len(order)):
            c0 = order[idx]
            a = cyc[c0]
            # gaps
            for g in range(1, MAX_GAP + 1):
                if idx + g < len(order):
                    b = cyc[order[idx + g]]
                    u = len(a | b)
                    if u:
                        by_gap[g].append(len(a & b) / u)
            # adjacent (gap=1) detailed
            if idx + 1 < len(order):
                b = cyc[order[idx + 1]]
                inter = len(a & b); union = len(a | b)
                if union == 0:
                    continue
                jac.append(inter / union)
                overlap_coef.append(inter / min(len(a), len(b)))
                retained.append(inter)
                added.append(len(b - a))
                removed.append(len(a - b))
                frac_prev_survives.append(inter / len(a) if a else 0.0)
                frac_next_isnew.append(len(b - a) / len(b) if b else 0.0)
                ei = len(a) * len(b) / n_experts
                exp_jac.append(ei / (len(a) + len(b) - ei)
                               if (len(a) + len(b) - ei) > 0 else 0.0)
                by_layer[layer].append(inter / union)
                by_pos[c0 // 5].append(inter / union)

    out("-" * 76)
    out("  1. ADJACENT CYCLE (gap = 1)")
    out("-" * 76)
    out(f"  transitions: {len(jac)}")
    def line(name, xs, fmt="{:.3f}"):
        out(f"  {name:24s} mean={fmt.format(st.mean(xs))} "
            f"median={fmt.format(pct(xs,50))} p10={fmt.format(pct(xs,10))} "
            f"p90={fmt.format(pct(xs,90))}")
    line("Jaccard |A∩B|/|A∪B|", jac)
    line("overlap |A∩B|/min", overlap_coef)
    line("retained |A∩B|", retained, "{:.2f}")
    line("added |B\\A|", added, "{:.2f}")
    line("removed |A\\B|", removed, "{:.2f}")
    line("prev-survives |A∩B|/|A|", frac_prev_survives)
    line("next-is-new |B\\A|/|B|", frac_next_isnew)
    out(f"  random-baseline Jaccard  mean={st.mean(exp_jac):.3f}  "
        f"(i.i.d. uniform over {n_experts})")
    out(f"  observed/random ratio    = {st.mean(jac)/st.mean(exp_jac):.2f}x")
    out("")

    # ── overlap vs gap ───────────────────────────────────────────────────
    out("-" * 76)
    out("  2. OVERLAP vs CYCLE GAP (does the set keep forgetting?)")
    out("-" * 76)
    out(f"  {'gap':>4s} {'n':>8s} {'meanJacc':>9s}")
    for g in range(1, MAX_GAP + 1):
        if by_gap[g]:
            out(f"  {g:4d} {len(by_gap[g]):8d} {st.mean(by_gap[g]):9.3f}")
    out("")

    # ── by layer depth ───────────────────────────────────────────────────
    out("-" * 76)
    out("  3a. ADJACENT JACCARD by LAYER (depth trend)")
    out("-" * 76)
    out(f"  {'layer':>5s} {'n':>6s} {'meanJacc':>9s}")
    for layer in sorted(by_layer):
        g = by_layer[layer]
        out(f"  {layer:5d} {len(g):6d} {st.mean(g):9.3f}")
    out("")

    # ── by position in question ──────────────────────────────────────────
    out("-" * 76)
    out("  3b. ADJACENT JACCARD by POSITION-IN-QUESTION (cycle buckets of 5)")
    out("-" * 76)
    out(f"  {'cycles':>9s} {'n':>8s} {'meanJacc':>9s}")
    for bucket in sorted(by_pos):
        g = by_pos[bucket]
        out(f"  {bucket*5:3d}-{bucket*5+4:<3d}   {len(g):8d} {st.mean(g):9.3f}")
    out("")

    # ── stable core ──────────────────────────────────────────────────────
    out("-" * 76)
    out("  4. STABLE CORE per (layer, question)")
    out("-" * 76)
    core_all, core80, core50, mean_size, ncyc = [], [], [], [], []
    for (layer, qid), cyc in seqs.items():
        C = len(cyc)
        if C < 2:
            continue
        freq = collections.Counter()
        for s in cyc.values():
            for e in s:
                freq[e] += 1
        core_all.append(sum(1 for v in freq.values() if v == C))
        core80.append(sum(1 for v in freq.values() if v >= 0.8 * C))
        core50.append(sum(1 for v in freq.values() if v >= 0.5 * C))
        mean_size.append(st.mean(len(s) for s in cyc.values()))
        ncyc.append(C)
    out(f"  questions x layers analysed: {len(core_all)} "
        f"(mean {st.mean(ncyc):.0f} cycles each)")
    out(f"  mean per-cycle set size        : {st.mean(mean_size):.2f}")
    out(f"  core present in 100% of cycles : {st.mean(core_all):.2f} experts "
        f"({100*st.mean(core_all)/st.mean(mean_size):.0f}% of a typical set)")
    out(f"  core present in >=80% of cycles: {st.mean(core80):.2f} experts "
        f"({100*st.mean(core80)/st.mean(mean_size):.0f}%)")
    out(f"  core present in >=50% of cycles: {st.mean(core50):.2f} experts "
        f"({100*st.mean(core50)/st.mean(mean_size):.0f}%)")
    out("")
    out("  reading: a small always-on core persists; most of each cycle's set")
    out("  sits outside it and churns (consistent with the gap-1 numbers above).")

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "cycle_overlap_stats.txt"
    report.write_text("\n".join(lines) + "\n")

    summary = {
        "n_transitions_gap1": len(jac),
        "jaccard": {"mean": st.mean(jac), "median": pct(jac, 50),
                    "p10": pct(jac, 10), "p90": pct(jac, 90)},
        "overlap_coef_mean": st.mean(overlap_coef),
        "retained_mean": st.mean(retained),
        "added_mean": st.mean(added),
        "removed_mean": st.mean(removed),
        "prev_survives_mean": st.mean(frac_prev_survives),
        "next_is_new_mean": st.mean(frac_next_isnew),
        "random_jaccard_mean": st.mean(exp_jac),
        "stickiness_ratio": st.mean(jac) / st.mean(exp_jac),
        "jaccard_by_gap": {g: st.mean(by_gap[g]) for g in range(1, MAX_GAP + 1)
                           if by_gap[g]},
        "core": {
            "mean_set_size": st.mean(mean_size),
            "core_100pct": st.mean(core_all),
            "core_80pct": st.mean(core80),
            "core_50pct": st.mean(core50),
        },
    }
    (out_dir / "cycle_overlap_stats.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"\n[analyze] wrote {report}")
    print(f"[analyze] wrote {out_dir / 'cycle_overlap_stats.json'}")


if __name__ == "__main__":
    main()
