#!/bin/bash
#SBATCH --job-name=ly_diverg
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/ly_diverg_%j.log
#SBATCH -e /work/morrisliu07/job_err/ly_diverg_%j.err
#
# Layer-by-layer divergence: hook every block_sparse_moe and compare
# (input, output) between offload moe.model and HF cpu_source on same prompt.
# Tells us at WHICH layer the offload path starts going wrong.

set -euo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0
# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

echo "[ly_diverg] node=$(hostname)  job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -u tests/debug_layer_diverge_mixtral.py
