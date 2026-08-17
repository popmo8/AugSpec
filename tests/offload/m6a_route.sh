#!/bin/bash
#SBATCH --job-name=m6a_route
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/m6a_route_%j.log
#SBATCH -e /work/morrisliu07/job_err/m6a_route_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# M5 — load_offload() 驗證 wrapper。
#
# Usage:
#   sbatch tests/offload/m6a_route.sh
#   bash   tests/offload/m6a_route.sh        # 互動式 GPU 節點
#
# 報告:tests/offload/m6a_route.out(python 端負責)。

set -uo pipefail

REPO_ROOT="/work/morrisliu07/aug_spec"

export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true

echo "[m5] node=$(hostname)  job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/tests/offload/m6a_route.py" "$@"
