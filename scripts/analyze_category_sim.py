#!/usr/bin/env python3
"""Does expert-pair similarity differ by question category?

Reads AUG_DUMP_CYCLE_SIM JSONL (per layer/question/cycle pairwise output-cosine)
+ the run's per_question_summary.csv (qid order == category order; qpc=1 so one
question per category). For each (layer, pair):
  per-question sim = mean of its per-cycle sims within that question;
  cross-category std = std of those per-question means across questions.
Compares that to the cross-CYCLE std (within a question) — if cross-category
std >> cross-cycle std, similarity is category-dependent (a single frozen table
won't transfer; per-question prefill warmup is warranted).

NOTE: at qpc=1, "category" and "question" are confounded (1 q/cat) — this is a
first look; a clean within-vs-between needs qpc>=2.

Usage: analyze_category_sim.py <cycle_sim.jsonl> <per_question_summary.csv> [out]
"""
import collections
import json
import math
import statistics as s
import sys
import csv
from pathlib import Path


def pct(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[int(k)] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    dump = Path(sys.argv[1]); pqs = Path(sys.argv[2])
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else dump.parent

    cats = [r["category"] for r in csv.DictReader(open(pqs))]   # qid -> category
    # (layer, qid, i, j) -> [sum_sim, n_cycles]   (mean per-cycle within question)
    qsum = collections.defaultdict(lambda: [0.0, 0])
    with open(dump) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            li, q = r["layer"], r["qid"]
            for i, j, v in r["sims"]:
                a = qsum[(li, q, i, j)]
                a[0] += v; a[1] += 1

    # (layer,i,j) -> {qid: per-question mean sim}
    pair = collections.defaultdict(dict)
    for (li, q, i, j), (sm, nc) in qsum.items():
        pair[(li, i, j)][q] = sm / nc

    xcat_std, xcat_range = [], []
    examples = []   # (spread, layer, i, j, {cat: sim})
    for (li, i, j), perq in pair.items():
        if len(perq) < 2:
            continue
        vals = list(perq.values())
        st = s.pstdev(vals)
        xcat_std.append(st)
        xcat_range.append(max(vals) - min(vals))
        if len(perq) >= 5:
            examples.append((max(vals) - min(vals), li, i, j,
                             {cats[q]: round(v, 3) for q, v in sorted(perq.items())}))

    lines = []
    def out(x=""):
        lines.append(x); print(x)
    out("=" * 72)
    out("  EXPERT-PAIR SIMILARITY by QUESTION CATEGORY (qpc=1)")
    out(f"  categories ({len(cats)}): {cats}")
    out("=" * 72)
    out(f"  (layer,pair) series across >=2 questions: {len(xcat_std)}")
    out("")
    out("  cross-CATEGORY std of a pair's similarity (per-question means):")
    out(f"    mean={s.mean(xcat_std):.4f} median={pct(xcat_std,50):.4f} "
        f"p90={pct(xcat_std,90):.4f} max={max(xcat_std):.4f}")
    out(f"    range(max-min): mean={s.mean(xcat_range):.4f} "
        f"median={pct(xcat_range,50):.4f} p90={pct(xcat_range,90):.4f}")
    out("")
    out("  COMPARE — cross-CYCLE std (within a question, from analyze_cycle_sim): "
        "~0.0187")
    ratio = s.mean(xcat_std) / 0.0187
    out(f"  cross-category mean std / cross-cycle = {ratio:.2f}x")
    out("  >1.3x => similarity varies by category beyond within-question noise.")
    out("")
    out("  biggest cross-category swings (pairs seen in >=5 categories):")
    for spread, li, i, j, percat in sorted(examples, reverse=True)[:12]:
        out(f"    L{li} ({i},{j}) range={spread:.3f}: {percat}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "category_sim_stats.txt").write_text("\n".join(lines) + "\n")
    print(f"\n[analyze] wrote {out_dir / 'category_sim_stats.txt'}")


if __name__ == "__main__":
    main()
