#!/bin/bash
#SBATCH --job-name=q5_512_tm
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/q5_512_tm_%j.log
#SBATCH -e /work/morrisliu07/job_err/q5_512_tm_%j.err
#
# topm qpc=5, no_overload ON, mnt=512 (apples-to-apples vs old engine_bmm 512 runs).
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
echo "[q5_512_tm] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== topm no_overload, mnt=512 ========"
AUG_NO_OVERLOAD=1 .venv/bin/python -m aug_spec.cli run --config configs/q5_512_tm_on.yaml || echo "[q5_512_tm] RUN FAILED"

echo "======== RESULT ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
def agg(d):
    try: r=list(csv.DictReader(open(f"output/{d}/per_question_summary.csv")))
    except FileNotFoundError: return None
    return (len(r), s.mean(float(x["mean_accept_length"]) for x in r),
            s.mean(float(x["acceptance_rate"]) for x in r),
            s.mean(float(x["tokens_per_second"]) for x in r))
a=agg("q5_512_tm_on")
print("  q5_512_tm_on "+("(none)" if a is None else f"n={a[0]} MAT={a[1]:.3f} AccR={a[2]:.3f} TPS={a[3]:.3f}"))
PYEOF
echo "[q5_512_tm] done"
