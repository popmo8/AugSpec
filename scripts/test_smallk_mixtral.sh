#!/bin/bash
#SBATCH --job-name=small_k
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/small_k_%j.log
#SBATCH -e /work/morrisliu07/job_err/small_k_%j.err
#
# Hypothesis test: Mixtral logits comparison after forcing the
# Small-K GEMM path in fused_moe_mlp.cu (`use_large_k = false`).
#
# Prerequisite: rebuild moe_infinity locally BEFORE submitting this:
#   cd /work/morrisliu07/aug_spec
#   ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0
#   source .venv/bin/activate
#   export CUTLASS_DIR=/work/morrisliu07/cutlass
#   uv pip install -e ./moe_infinity \
#       --no-build-isolation --no-deps --force-reinstall
#
# Expected: if the Large-K branch was the bug, Mixtral logits should
# now match HF (argmax 'a' / 'Paris'-equivalent, top-K overlap high).

set -euo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0
# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

echo "[small_k] node=$(hostname)  job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo
echo "============================================================"
echo "  Mixtral logits comparison (Small-K forced)"
echo "============================================================"
python -u tests/debug_offload_logits.py --model mixtral
