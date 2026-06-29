#!/bin/bash
#SBATCH --job-name=q5_repro
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/q5_repro_%j.log
#SBATCH -e /work/morrisliu07/job_err/q5_repro_%j.err
#
# Determinism check: re-run the SAME q5_512_tm_on settings with the SAME
# (current) code into output/q5_512_tm_repro. Comparing repro vs verify (both
# refactored code) isolates run-to-run non-determinism from any refactor effect;
# also compared against the original baseline.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"

echo "[q5_repro] node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "======== topm no_overload, mnt=512 (rerun, same code) ========"
AUG_NO_OVERLOAD=1 .venv/bin/python -m aug_spec.cli run \
    --config configs/q5_512_tm_repro.yaml || echo "[q5_repro] RUN FAILED"

echo "======== COMPARE repro vs baseline(q5_512_tm_on) and verify ========"
.venv/bin/python - <<'PYEOF'
import csv, os
def rows(d):
    p=f"output/{d}/per_question_summary.csv"
    return {(r["category"],r["question_id"]): r
            for r in csv.DictReader(open(p))} if os.path.exists(p) else {}
cols=["num_cycles","num_new_tokens","mean_accept_length","acceptance_rate"]
repro=rows("q5_512_tm_repro")
for other in ("q5_512_tm_on","q5_512_tm_verify"):
    o=rows(other)
    if not o or not repro:
        print(f"  vs {other}: MISSING (repro={len(repro)} other={len(o)})"); continue
    keys=sorted(set(repro)&set(o))
    qdiff=0; field=0
    for k in keys:
        d=[c for c in cols if repro[k][c]!=o[k][c]]
        if d: qdiff+=1; field+=len(d)
    print(f"  repro vs {other:18s}: {len(keys)} q, {qdiff} q differ, {field} field diffs",
          "=> IDENTICAL" if qdiff==0 else "")
PYEOF
echo "[q5_repro] done"
