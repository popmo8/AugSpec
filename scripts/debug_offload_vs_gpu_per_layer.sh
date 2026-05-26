#!/bin/bash
#SBATCH --job-name=cmp_layer
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/cmp_layer_%j.log
#SBATCH -e /work/morrisliu07/job_err/cmp_layer_%j.err
#
# Per-layer offload vs GPU-full Mixtral comparison. Both run on the same
# H200; offload should be bit-equivalent to GPU-full.

set -euo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0
# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

echo "[cmp_layer] node=$(hostname)  job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -u tests/debug_offload_vs_gpu_per_layer.py
