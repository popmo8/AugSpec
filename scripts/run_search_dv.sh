#!/bin/bash
#SBATCH --job-name=search_dv
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/%x_%j.log
#SBATCH -e /work/morrisliu07/job_err/%x_%j.err
#
# DV1 Draft&Verify skip-layer BO search (Table 1 Draft&Verify row).
#   sbatch scripts/run_search_dv.sh Qwen/Qwen3-30B-A3B-Base --num-keep 6
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: run_search_dv.sh <model-id> [args...]" >&2; exit 2; }
MODEL_ID="$1"; shift

REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
echo "[search_dv] node=$(hostname) model=${MODEL_ID} extra=$*"
.venv/bin/python -u scripts/search_draft_verify.py --model-id "${MODEL_ID}" "$@"
