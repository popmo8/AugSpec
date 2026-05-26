#!/bin/bash
#SBATCH --job-name=block_cmp
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/block_cmp_%j.log
#SBATCH -e /work/morrisliu07/job_err/block_cmp_%j.err
#
# Single-block side-by-side: offload SyncMixtralSparseMoeBlock vs HF stock
# MixtralSparseMoeBlock. Isolates moe_infinity's dispatch path from spec-decoding.

set -euo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0
# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

echo "[block_cmp] node=$(hostname)  job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -u tests/debug_block_compare_mixtral.py
