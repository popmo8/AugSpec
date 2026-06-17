#!/bin/bash
#SBATCH --job-name=base_spec_retry
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/base_spec_retry_%j.log
#SBATCH -e /work/morrisliu07/job_err/base_spec_retry_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# 重編 _store（含 dispatcher 死鎖修正：cache_cv lost-wakeup -> wait_for 重試迴圈）
# 後重跑 base_specmoe_offload（hf 已完成、topm 已完成，只缺這格）。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
MOE_ROOT="${REPO_ROOT}/moe_infinity"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUTLASS_DIR="${HOME}/cutlass"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
echo "[retry] node=$(hostname) job=${SLURM_JOB_ID:-local} nvcc=$(which nvcc)"

echo "======== rebuild _store (deadlock fix) ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -6
rc=${PIPESTATUS[0]}
echo "[retry] build rc=${rc}"
if [ "${rc}" -ne 0 ]; then echo "[retry] BUILD FAILED"; exit "${rc}"; fi

echo "======== rerun specmoe offload (Base) ========"
cd "${REPO_ROOT}"
.venv/bin/python -m aug_spec.cli run --config configs/base_specmoe_offload.yaml || echo "[retry] offload FAILED"
echo "[retry] done"
