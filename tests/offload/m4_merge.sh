#!/bin/bash
#SBATCH --job-name=m4_merge
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/m4_merge_%j.log
#SBATCH -e /work/morrisliu07/job_err/m4_merge_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# M4 — merge 三變體計時 + 精度 wrapper。
#
# cpus-per-task=4 與 run.sh 一致 ——(d) CPU merge 的速度吃核數,
# §1.3 預警「4 核時 (d) 未必贏」,用部署的真實核數量才誠實。
#
# Usage:
#   sbatch tests/offload/m4_merge.sh
#   sbatch tests/offload/m4_merge.sh --num-merge 32     # 掃 M
#   bash   tests/offload/m4_merge.sh                    # 互動式 GPU 節點
#
# 報告:tests/offload/m4_merge.out(python 端負責)。

set -uo pipefail

REPO_ROOT="/work/morrisliu07/aug_spec"

export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true

echo "[m4] node=$(hostname)  job=${SLURM_JOB_ID:-local}  args=$*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/tests/offload/m4_merge.py" "$@"
