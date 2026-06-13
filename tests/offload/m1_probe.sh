#!/bin/bash
#SBATCH --job-name=m1_probe
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/m1_probe_%j.log
#SBATCH -e /work/morrisliu07/job_err/m1_probe_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# M1 — offload_plan.md 結構探勘 wrapper。
#
# Usage:
#   sbatch tests/offload/m1_probe.sh
#   sbatch tests/offload/m1_probe.sh --skip-cpu-copy      # 省 host RAM 的快跑
#   bash   tests/offload/m1_probe.sh                      # 互動式 GPU 節點
#
# 報告固定寫到 tests/offload/m1_probe.out（python 端負責）。
# 注意：HF token 不寫在這裡（tests/ 之後會進 git）；模型應已在
# HF_HOME cache。若 cache 不在，先 export HF_TOKEN 再 submit
# （sbatch 預設繼承提交時的環境變數）。

set -euo pipefail

REPO_ROOT="/work/morrisliu07/aug_spec"

export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true

echo "[m1] node=$(hostname)  job=${SLURM_JOB_ID:-local}  args=$*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/tests/offload/m1_probe.py" "$@"
