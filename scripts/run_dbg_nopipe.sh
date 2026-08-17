#!/bin/bash
#SBATCH --job-name=dbg_hang
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/%x_%j.log
#SBATCH -e /work/morrisliu07/job_err/%x_%j.err
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_HANG_DEBUG=600      # dump all thread stacks every 600s after a stall
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
echo "[dbg] node=$(hostname) AUG_HANG_DEBUG=600 heartbeat=every-25-cycles"
.venv/bin/python -u -m aug_spec.cli run --config configs/dbg_ours_nopipe.yaml
