#!/bin/bash
#SBATCH --job-name=c1_resident
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/c1_resident_%j.log
#SBATCH -e /work/morrisliu07/job_err/c1_resident_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# C1: 重編 _store（含新增 get_resident_expert_weights）+ 跑唯讀正確性驗證。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
MOE_ROOT="${REPO_ROOT}/moe_infinity"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUTLASS_DIR="${HOME}/cutlass"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
echo "[c1] node=$(hostname) job=${SLURM_JOB_ID:-local} nvcc=$(which nvcc)"

echo "======== rebuild _store ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -8
rc=${PIPESTATUS[0]}
echo "[c1] build rc=${rc}"
if [ "${rc}" -ne 0 ]; then echo "[c1] BUILD FAILED"; exit "${rc}"; fi

echo "======== C1 test ========"
cd "${REPO_ROOT}"
"${PY}" tests/offload/c1_resident.py
echo "[c1] done"
