#!/bin/bash
#SBATCH --job-name=engbmm_sm
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/engbmm_sm_%j.log
#SBATCH -e /work/morrisliu07/job_err/engbmm_sm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# v1 engine bmm (specmoe): pinned kept-N through DispatchBmm. Binary current
# (built by engbmm_topm) — no rebuild. 4th quadrant for the bmm-vs-bmm gap.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_MERGED_BACKEND=engine_bmm
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
echo "[engbmm_sm] node=$(hostname) job=${SLURM_JOB_ID:-local} backend=${AUG_MERGED_BACKEND}"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_engbmm_specmoe.yaml || echo "[engbmm_sm] RUN FAILED"
echo "======== RESULT ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
try:
    r=list(csv.DictReader(open("output/cmp_engbmm_specmoe/per_question_summary.csv")))
    print(f"  specmoe-bmm  n={len(r)} MAT={s.mean(float(x['mean_accept_length']) for x in r):.3f} "
          f"AccRate={s.mean(float(x['acceptance_rate']) for x in r):.3f} "
          f"TPS={s.mean(float(x['tokens_per_second']) for x in r):.3f}")
except FileNotFoundError: print("  (no summary)")
PYEOF
echo "[engbmm_sm] done"
