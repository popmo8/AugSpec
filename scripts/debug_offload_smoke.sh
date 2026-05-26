#!/bin/bash
#SBATCH --job-name=dbg_off
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/dbg_off_%j.log
#SBATCH -e /work/morrisliu07/job_err/dbg_off_%j.err
#
# Debug-only sbatch: re-run offload smoke and capture the full traceback
# of the per-question failure (we patched specbench's WARN handler to
# print full tracebacks).

set -euo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0
# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

echo "[dbg] node=$(hostname)  job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

aug_spec run --config configs/_smoke_offload.yaml
