#!/bin/bash
#SBATCH --job-name=full_eb_specmoe
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/full_eb_specmoe_%j.log
#SBATCH -e /work/morrisliu07/job_err/full_eb_specmoe_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# FULL SpecBench (480 Q) — specmoe via engine_bmm (C++ DispatchBmm). Binary
# current (built by engbmm_topm 243107) — no rebuild.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_MERGED_BACKEND=engine_bmm
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
echo "[full_specmoe] node=$(hostname) job=${SLURM_JOB_ID:-local} backend=${AUG_MERGED_BACKEND}"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_full_engbmm_specmoe.yaml || echo "[full_specmoe] RUN FAILED"
echo "======== RESULT ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
try:
    r=list(csv.DictReader(open("output/cmp_full_engbmm_specmoe/per_question_summary.csv")))
    print(f"  specmoe-engbmm  n={len(r)} MAT={s.mean(float(x['mean_accept_length']) for x in r):.3f} "
          f"AccRate={s.mean(float(x['acceptance_rate']) for x in r):.3f} "
          f"TPS={s.mean(float(x['tokens_per_second']) for x in r):.3f}")
except FileNotFoundError: print("  (no summary)")
PYEOF
echo "[full_specmoe] done"
