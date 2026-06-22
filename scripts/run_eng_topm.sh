#!/bin/bash
#SBATCH --job-name=eng_topm
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/eng_topm_%j.log
#SBATCH -e /work/morrisliu07/job_err/eng_topm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# Same-engine comparison (topm half): merged draft via dispatch_merged_local —
# the SAME MoEMLP::forward kernel SpecMoE dispatches. P2+P3, no P1. Default
# backend = dispatch. Binary already current (smoke 242791) — no rebuild.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
echo "[eng_tm] node=$(hostname) job=${SLURM_JOB_ID:-local} backend=${AUG_MERGED_BACKEND:-dispatch}"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_eng_topm.yaml || echo "[eng_tm] RUN FAILED"
echo "======== RESULT ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
try:
    r=list(csv.DictReader(open("output/cmp_eng_topm/per_question_summary.csv")))
    print(f"  topm  n={len(r)}  MAT={s.mean(float(x['mean_accept_length']) for x in r):.3f}  "
          f"AccRate={s.mean(float(x['acceptance_rate']) for x in r):.3f}  "
          f"TPS={s.mean(float(x['tokens_per_second']) for x in r):.3f}")
except FileNotFoundError: print("  (no summary)")
PYEOF
echo "[eng_tm] done"
