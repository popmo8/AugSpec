#!/bin/bash
#SBATCH --job-name=topm_m64
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/topm_m64_%j.log
#SBATCH -e /work/morrisliu07/job_err/topm_m64_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
echo "[m64] node=$(hostname) job=${SLURM_JOB_ID:-local}"
cd "${REPO_ROOT}"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_topm_opt_m64.yaml 2>&1 | grep -E "\[budget\]|\[vram\]|peak|FATAL|Aborted|FAIL" || echo "[m64] FAILED"
echo "=== result ==="; tail -3 output/cmp_topm_opt_m64/per_question_summary.csv 2>/dev/null
echo "[m64] done"
