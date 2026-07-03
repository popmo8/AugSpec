#!/usr/bin/env python3
"""Cross-cycle drift of per-cycle expert similarity (from AUG_DUMP_CYCLE_SIM).

Each record is one (layer, question, cycle) with that cycle's STANDALONE
pairwise output-cosine (computed from only that cycle's tokens). Question: for a
given expert pair, how much does its similarity swing cycle-to-cycle? If it's
stable, you can compute similarity once (calibrate/freeze) instead of
accumulating every cycle.

Per (layer, question, pair) series of per-cycle sims (len >= 2):
  std, range (max-min), mean |consecutive Δ|.
Reported as distributions across all such series, plus by layer, plus a
"stable" fraction (std < 0.05). Also the spread of per-cycle sim vs the
accumulated (mean) sim.

Usage: analyze_cycle_sim.py <cycle_sim.jsonl> [out_dir]
"""
import collections
import json
import math
import statistics as s
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
        print(__doc__); sys.exit(1)
    dump = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else dump.parent

    # (layer, qid, (i,j)) -> {cycle: sim}
    series = collections.defaultdict(dict)
    n_rec = 0
    layers = set()
    with open(dump) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_rec += 1
            layers.add(r["layer"])
            for i, j, v in r["sims"]:
                series[(r["layer"], r["qid"], i, j)][r["cycle"]] = v

    stds, ranges, dabs, means = [], [], [], []
    per_layer = collections.defaultdict(list)     # layer -> [std,...]
    for (layer, qid, i, j), cyc in series.items():
        if len(cyc) < 2:
            continue
        order = sorted(cyc)
        vals = [cyc[c] for c in order]
        st = s.pstdev(vals)
        stds.append(st)
        ranges.append(max(vals) - min(vals))
        dabs.append(s.mean(abs(b - a) for a, b in zip(vals, vals[1:])))
        means.append(s.mean(vals))
        per_layer[layer].append(st)

    lines = []
    def out(x=""):
        lines.append(x); print(x)

    out("=" * 72)
    out("  PER-CYCLE expert-similarity DRIFT across cycles")
    out(f"  source: {dump}")
    out("=" * 72)
    out(f"  records {n_rec} | layers {len(layers)} | "
        f"pair-series (>=2 cycles) {len(stds)}")
    if not stds:
        out("  (no multi-cycle pair series)");
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (out_dir / "cycle_sim_stats.txt").write_text("\n".join(lines) + "\n")
        return
    out("")
    out("  per-pair similarity across its cycles (lower = more stable):")
    out(f"  std            mean={s.mean(stds):.4f} median={pct(stds,50):.4f} "
        f"p90={pct(stds,90):.4f} max={max(stds):.4f}")
    out(f"  range(max-min) mean={s.mean(ranges):.4f} median={pct(ranges,50):.4f} "
        f"p90={pct(ranges,90):.4f} max={max(ranges):.4f}")
    out(f"  |Δ| consec     mean={s.mean(dabs):.4f} median={pct(dabs,50):.4f} "
        f"p90={pct(dabs,90):.4f}")
    out(f"  mean per-cycle sim itself: mean={s.mean(means):.4f} "
        f"range=[{min(means):.3f},{max(means):.3f}]")
    stable = sum(1 for x in stds if x < 0.05) / len(stds)
    out(f"  fraction of pairs with std < 0.05 (stable): {100*stable:.1f}%")
    out(f"  fraction with std < 0.10                  : "
        f"{100*sum(1 for x in stds if x<0.10)/len(stds):.1f}%")
    out("")
    out("  by layer (mean std):")
    bl = {l: s.mean(v) for l, v in per_layer.items() if v}
    out(f"    across layers: mean={s.mean(list(bl.values())):.4f} "
        f"min={min(bl.values()):.4f} max={max(bl.values()):.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rep = out_dir / "cycle_sim_stats.txt"
    rep.write_text("\n".join(lines) + "\n")
    print(f"\n[analyze] wrote {rep}")


if __name__ == "__main__":
    main()
