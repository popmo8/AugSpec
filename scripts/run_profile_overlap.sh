#!/bin/bash
#SBATCH --job-name=profile_overlap
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/%x_%j.log
#SBATCH -e /work/morrisliu07/job_err/%x_%j.err
#
# Diagnostic: prefill-hot vs decode-hot expert overlap (scripts/profile_expert_overlap.py).
set -uo pipefail
CONFIG="${1:-configs/profile_overlap.yaml}"
DMR="${2:-0.25}"
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
echo "[profile] node=$(hostname) config=${CONFIG} device_memory_ratio=${DMR}"
.venv/bin/python -u scripts/profile_expert_overlap.py --config "${CONFIG}" \
    --device-memory-ratio "${DMR}"
echo "[profile] done rc=$?"
