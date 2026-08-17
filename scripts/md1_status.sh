#!/bin/bash
# ONE-SHOT status read-out for the md1 (DeepSeek-MoE-16B-base Table-1)
# campaign. Runs exactly one squeue query per invocation — safe under the
# Nano5 SLURM-query policy (no polling; do NOT wrap this in a tight loop).
#   bash scripts/md1_status.sh
cd /work/morrisliu07/aug_spec

echo "== queued/running md1 jobs (one squeue call) =="
squeue -u "$USER" -o "%.9i %.20j %.8T %.11l %.11M %.20E" | grep -E "JOBID|md1|collect_calib|search_dv" || echo "(none)"

echo
echo "== prerequisite artifacts =="
for f in output/calibration/deepseek-moe-16b-base/meta.json \
         output/naee/deepseek-moe-16b-base_r8.json \
         output/hc_smoe/deepseek-moe-16b-base_K8.json \
         output/mc_smoe/deepseek-moe-16b-base_K8.json \
         output/draft_verify/deepseek-moe-16b-base_L3.json; do
    [ -f "$f" ] && echo "  OK      $f" || echo "  MISSING $f"
done

echo
echo "== run outputs (overall_summary = finished) =="
for m in randmask specmoe enum speed dv randmerge hcsmoe mcsmoe ours; do
    for pfx in md1_smoke_ md1_; do
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
    p = f"output/md1_{m}/overall_summary.csv"
    if not os.path.exists(p):
        continue
    rows = {r["subtask"]: r for r in csv.DictReader(open(p))
            if r.get("subtask")}
    try:
        cells = [float(rows[s]["acceptance_rate"]) for s in SUBS]
        print(f"  md1_{m:10s} mean7={sum(cells)/len(cells):.4f}  "
              + " ".join(f"{s.split('_')[0]}={c:.3f}"
                         for s, c in zip(SUBS, cells)))
    except KeyError as e:
        print(f"  md1_{m:10s} (missing subtask {e} in overall_summary)")
EOF
