#!/bin/bash
#SBATCH --job-name=q5_512_tm_unif
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/q5_512_tm_unif_%j.log
#SBATCH -e /work/morrisliu07/job_err/q5_512_tm_unif_%j.err
#
# topm qpc=5, mnt=512, no_overload ON, within-cluster merge = UNIFORM
# (AUG_CLUSTER_UNIFORM=1). Compares against the existing freq baseline
# output/q5_512_tm_on (same q5_512_tm_on.yaml settings).
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
echo "[q5_512_tm_unif] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== topm uniform within-cluster merge, mnt=512 ========"
AUG_NO_OVERLOAD=1 AUG_CLUSTER_UNIFORM=1 .venv/bin/python -m aug_spec.cli run \
    --config configs/q5_512_tm_unif.yaml || echo "[q5_512_tm_unif] RUN FAILED"

echo "======== COMPARE (freq baseline vs uniform) ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
def agg(d):
    try: r=list(csv.DictReader(open(f"output/{d}/per_question_summary.csv")))
    except FileNotFoundError: return None
    return (len(r), s.mean(float(x["mean_accept_length"]) for x in r),
            s.mean(float(x["acceptance_rate"]) for x in r),
            s.mean(float(x["tokens_per_second"]) for x in r))
print(f"  {'within-cluster':16s} {'n':>3s} {'MAT':>6s} {'AccR':>6s} {'TPS':>6s}")
for lbl,d in [("freq (baseline)","q5_512_tm_on"),("uniform","q5_512_tm_unif")]:
    a=agg(d)
    print(f"  {lbl:16s} "+("(none)" if a is None else f"{a[0]:3d} {a[1]:6.3f} {a[2]:6.3f} {a[3]:6.3f}"))
PYEOF
echo "[q5_512_tm_unif] done"
