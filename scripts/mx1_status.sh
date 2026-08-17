#!/bin/bash
# ONE-SHOT status read-out for the mx1 (Mixtral Table-1) campaign.
# Runs exactly one squeue query per invocation — safe under the Nano5
# SLURM-query policy (no polling; do NOT wrap this in a tight loop).
#   bash scripts/mx1_status.sh
cd /work/morrisliu07/aug_spec

echo "== queued/running mx1 jobs (one squeue call) =="
squeue -u "$USER" -o "%.9i %.20j %.8T %.11l %.11M %.20E" | grep -E "JOBID|mx1|collect_calib|search_dv" || echo "(none)"

echo
echo "== prerequisite artifacts =="
for f in output/calibration/Mixtral-8x7B-v0.1/meta.json \
         output/naee/Mixtral-8x7B-v0.1_r1.json \
         output/hc_smoe/Mixtral-8x7B-v0.1_K1.json \
         output/mc_smoe/Mixtral-8x7B-v0.1_K1.json \
         output/draft_verify/Mixtral-8x7B-v0.1_L4.json; do
    [ -f "$f" ] && echo "  OK      $f" || echo "  MISSING $f"
done

echo
echo "== run outputs (overall_summary = finished) =="
for m in randmask specmoe enum speed dv randmerge hcsmoe mcsmoe ours; do
    for pfx in mx1_smoke_ mx1_; do
        d=output/${pfx}${m}
        if [ -f "$d/overall_summary.csv" ]; then s="DONE"
        elif [ -f "$d/per_question_summary.csv" ]; then
            s="PARTIAL($(($(wc -l < "$d/per_question_summary.csv") - 1)) q)"
        elif [ -d "$d" ]; then s="STARTED"
        else s="-"; fi
        printf "  %-22s %s\n" "${pfx}${m}" "$s"
    done
done

echo
echo "== mean7 acceptance for finished FULL runs =="
.venv/bin/python - <<'EOF'
import csv, os
SUBS = ["mt_bench", "translation", "summarization", "qa",
        "math_reasoning", "rag", "humaneval"]
for m in ["randmask", "specmoe", "enum", "speed", "dv",
          "randmerge", "hcsmoe", "mcsmoe", "ours"]:
    p = f"output/mx1_{m}/overall_summary.csv"
    if not os.path.exists(p):
        continue
    rows = {r["subtask"]: r for r in csv.DictReader(open(p))
            if r.get("subtask")}
    try:
        cells = [float(rows[s]["acceptance_rate"]) for s in SUBS]
        print(f"  mx1_{m:10s} mean7={sum(cells)/len(cells):.4f}  "
              + " ".join(f"{s.split('_')[0]}={c:.3f}"
                         for s, c in zip(SUBS, cells)))
    except KeyError as e:
        print(f"  mx1_{m:10s} (missing subtask {e} in overall_summary)")
EOF
