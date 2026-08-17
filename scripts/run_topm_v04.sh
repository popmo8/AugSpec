#!/bin/bash
#SBATCH --job-name=topm_v04
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/topm_v04_%j.log
#SBATCH -e /work/morrisliu07/job_err/topm_v04_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
echo "[v04] node=$(hostname) job=${SLURM_JOB_ID:-local}"
cd "${REPO_ROOT}"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_opt_topm_v04.yaml 2>&1 | grep -E "\[budget\]|\[vram\]|peak|FATAL|Aborted|FAIL" || echo "[v04] FAILED"
echo "=== overall ==="; cat output/cmp_opt_topm_v04/overall_summary.csv 2>/dev/null | tail -1
echo "[v04] done"
