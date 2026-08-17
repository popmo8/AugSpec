#!/usr/bin/env python3
"""Quantify the non-uniform within-cluster merge weights.

Reads the JSONL produced by AUG_DUMP_CLUSTER_WEIGHTS (one record per merged
cluster) and reports how far the frequency-based within-cluster weights spread
relative to the uniform reference (1/|cluster|). This is exactly the difference
between the q5_512_tm_on (freq) and q5_512_tm_unif (uniform) merge weighting.

Per cluster of size s with weights w (sorted desc, sum=1):
  top_share     = w[0]                       uniform ref = 1/s
  top_vs_unif   = w[0] * s                    x times the uniform share
  max_minus_min = w[0] - w[-1]               absolute spread
  max_over_min  = w[0] / w[-1]               ratio of heaviest:lightest
  norm_entropy  = H(w)/log(s)                1.0 = perfectly uniform
  max_abs_dev   = max_i |w_i - 1/s|          largest deviation from uniform

Usage: analyze_cluster_weights.py <dump.jsonl> [out_dir]
"""
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
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(name, xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return f"  {name:16s}  (no data)"
    return (f"  {name:16s}  mean={st.mean(xs):8.4f}  median={pct(xs,50):8.4f}  "
            f"p90={pct(xs,90):8.4f}  max={max(xs):8.4f}  min={min(xs):8.4f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    dump = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else dump.parent

    records = []
    with open(dump) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        print(f"[analyze] no records in {dump}")
        sys.exit(1)

    rows = []          # per multi-expert (size>1) cluster metrics
    n_size1 = 0
    sizes = []
    for r in records:
        w = r["weights"]
        s = r["size"]
        sizes.append(s)
        if s <= 1:
            n_size1 += 1
            continue
        unif = 1.0 / s
        top = w[0]
        bot = w[-1]
        H = -sum(p * math.log(p) for p in w if p > 0)
        rows.append({
            "layer": r["layer"], "cycle": r["cycle"], "cluster": r["cluster"],
            "size": s,
            "top_share": top,
            "top_vs_unif": top / unif,
            "max_minus_min": top - bot,
            "max_over_min": (top / bot) if bot > 0 else float("inf"),
            "norm_entropy": H / math.log(s),
            "max_abs_dev": max(abs(x - unif) for x in w),
        })

    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    out("=" * 72)
    out("  NON-UNIFORM within-cluster merge-weight spread")
    out(f"  source: {dump}")
    out("=" * 72)
    out(f"  total cluster builds : {len(records)}")
    out(f"  distinct layers      : {len({r['layer'] for r in records})}")
    out(f"  refresh cycles/layer : up to {max(r['cycle'] for r in records) + 1}")
    out(f"  cluster size         : mean={st.mean(sizes):.2f} "
        f"min={min(sizes)} max={max(sizes)}")
    out(f"  size==1 clusters     : {n_size1}/{len(records)} "
        f"({100*n_size1/len(records):.1f}%) -> trivially uniform, excluded below")
    out("")
    out(f"  multi-expert clusters analysed: {len(rows)}")
    out("")
    out("  How concentrated are the freq weights inside one cluster?")
    out("  (uniform merge would force every metric below to its uniform value)")
    out("")
    out("  metric            (uniform value)")
    out(summarize("top_share", [r["top_share"] for r in rows]))
    out(summarize("top_vs_unif (1)", [r["top_vs_unif"] for r in rows]))
    out(summarize("max_minus_min(0)", [r["max_minus_min"] for r in rows]))
    finite_ratio = [r["max_over_min"] for r in rows if math.isfinite(r["max_over_min"])]
    out(summarize("max_over_min (1)", finite_ratio))
    out(summarize("norm_entropy (1)", [r["norm_entropy"] for r in rows]))
    out(summarize("max_abs_dev  (0)", [r["max_abs_dev"] for r in rows]))
    out("")

    # breakdown by cluster size bucket
    out("  by cluster size:")
    out(f"  {'size':>6s} {'n':>5s} {'top_share':>10s} {'top_vs_unif':>12s} "
        f"{'max/min':>9s} {'norm_ent':>9s}")
    by_size = {}
    for r in rows:
        by_size.setdefault(r["size"], []).append(r)
    for s in sorted(by_size):
        g = by_size[s]
        fr = [r["max_over_min"] for r in g if math.isfinite(r["max_over_min"])]
        out(f"  {s:6d} {len(g):5d} "
            f"{st.mean(r['top_share'] for r in g):10.4f} "
            f"{st.mean(r['top_vs_unif'] for r in g):12.4f} "
            f"{(st.mean(fr) if fr else float('nan')):9.3f} "
            f"{st.mean(r['norm_entropy'] for r in g):9.4f}")
    out("")

    # breakdown by cluster rank (0 = heaviest-frequency slice)
    out("  by cluster index (0 = heaviest-frequency slice):")
    out(f"  {'idx':>4s} {'n':>5s} {'top_share':>10s} {'top_vs_unif':>12s} "
        f"{'norm_ent':>9s}")
    by_idx = {}
    for r in rows:
        by_idx.setdefault(r["cluster"], []).append(r)
    for c in sorted(by_idx):
        g = by_idx[c]
        out(f"  {c:4d} {len(g):5d} "
            f"{st.mean(r['top_share'] for r in g):10.4f} "
            f"{st.mean(r['top_vs_unif'] for r in g):12.4f} "
            f"{st.mean(r['norm_entropy'] for r in g):9.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "cluster_weight_stats.txt"
    report.write_text("\n".join(lines) + "\n")

    # machine-readable
    summary = {
        "n_cluster_builds": len(records),
        "n_size1_clusters": n_size1,
        "n_multi_clusters": len(rows),
        "cluster_size": {"mean": st.mean(sizes), "min": min(sizes), "max": max(sizes)},
        "metrics": {},
    }
    for key in ["top_share", "top_vs_unif", "max_minus_min", "norm_entropy",
                "max_abs_dev"]:
        xs = [r[key] for r in rows]
        summary["metrics"][key] = {
            "mean": st.mean(xs), "median": pct(xs, 50),
            "p90": pct(xs, 90), "max": max(xs), "min": min(xs)}
    fr = [r["max_over_min"] for r in rows if math.isfinite(r["max_over_min"])]
    summary["metrics"]["max_over_min"] = {
        "mean": st.mean(fr), "median": pct(fr, 50), "p90": pct(fr, 90),
        "max": max(fr), "min": min(fr)} if fr else None
    (out_dir / "cluster_weight_stats.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    # per-cluster CSV for further plotting
    import csv
    with open(out_dir / "cluster_weight_rows.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n[analyze] wrote {report}")
    print(f"[analyze] wrote {out_dir / 'cluster_weight_stats.json'}")
    print(f"[analyze] wrote {out_dir / 'cluster_weight_rows.csv'}")


if __name__ == "__main__":
    main()
