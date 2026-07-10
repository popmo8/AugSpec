#!/bin/bash
#SBATCH --job-name=c0_probe
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/c0_probe_%j.log
#SBATCH -e /work/morrisliu07/job_err/c0_probe_%j.err
#
# merged_cache_plan.md C0 功能 probe(tests/offload/c0_slots.py):
# 直接呼叫五個 merged-slot API 驗證正確性(不接 policy)。
# 依賴 c0_smoke job 的 rebuild(.so 已含 C0 機制)。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
cd "${REPO_ROOT}"
echo "[c0_probe] node=$(hostname) job=${SLURM_JOB_ID:-local}"
"${PY}" tests/offload/c0_slots.py
rc=$?
echo "[c0_probe] rc=${rc}"
exit "${rc}"
