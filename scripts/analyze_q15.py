#!/usr/bin/env python3
"""Aggregate the q15 sweep: per-method mean +/- std over r1/r2/r3.

Reads output/q15_<method>_r{1,2,3}/overall_summary.csv for the 7 methods and
reports overall AccR / MAT / TPS as mean +/- std (and min..max) across the
available repeats — so the run-to-run variance is averaged out and methods are
comparable. Also per-subtask mean. Missing/unfinished repeats are skipped.

Usage: analyze_q15.py
"""
import csv
import os
import statistics as s

METHODS = ["freqslice", "random", "cooccur", "actsim_cos", "actsim_l2",
           "weightsim_cos", "weightsim_l2", "specmoe",
           "hybrid_a00", "hybrid_a25", "hybrid_a50", "hybrid_a75",
           "hybrid_a100",
           # cosine-normalized co-occur variant (a100 == raw a100: cooccur
           # weight is 0 at lambda=1, so reuse the raw run rather than rerun).
           "hybrid_cos_a00", "hybrid_cos_a25", "hybrid_cos_a50",
           "hybrid_cos_a75"]
SUBS = ["translation", "summarization", "qa", "math_reasoning", "rag", "overall"]


def read(method, r):
    p = f"output/q15_{method}_r{r}/overall_summary.csv"
    if not os.path.exists(p):
        return None
    rows = {row["subtask"]: row for row in csv.DictReader(open(p))}
    return rows


def main():
    print(f"{'method':16s} {'n':>2s}  {'AccR mean':>9s} {'std':>6s} "
          f"{'min':>6s} {'max':>6s}  {'MAT':>6s} {'TPS':>6s}")
    summary = []
    for m in METHODS:
        reps = [read(m, r) for r in (1, 2, 3)]
        reps = [x for x in reps if x is not None]
        if not reps:
            print(f"{m:16s}  0  (no runs yet)")
            continue
        accr = [float(x["overall"]["acceptance_rate"]) for x in reps]
        mat = [float(x["overall"]["mean_accept_tokens"]) for x in reps]
        tps = [float(x["overall"]["tokens_per_second"]) for x in reps]
        std = s.pstdev(accr) if len(accr) > 1 else 0.0
        print(f"{m:16s} {len(accr):2d}  {s.mean(accr):9.4f} {std:6.4f} "
              f"{min(accr):6.3f} {max(accr):6.3f}  "
              f"{s.mean(mat):6.3f} {s.mean(tps):6.3f}")
        summary.append((s.mean(accr), std, m))

    if summary:
        print("\nranked by mean AccR (uniform merge, no mt_bench, qpc=15):")
        for a, st, m in sorted(summary, reverse=True):
            print(f"  {m:16s} {a:.4f} +/- {st:.4f}")

    # per-subtask mean per method
    print("\nper-subtask mean AccR:")
    print(f"{'method':16s} " + " ".join(f"{x[:8]:>8s}" for x in SUBS))
    for m in METHODS:
        reps = [read(m, r) for r in (1, 2, 3)]
        reps = [x for x in reps if x is not None]
        if not reps:
            continue
        cells = []
        for sub in SUBS:
            vals = [float(x[sub]["acceptance_rate"]) for x in reps if sub in x]
            cells.append(f"{s.mean(vals):8.3f}" if vals else f"{'-':>8s}")
        print(f"{m:16s} " + " ".join(cells))


if __name__ == "__main__":
    main()
