#!/bin/bash
#SBATCH --job-name=c3_cppmerge
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/c3_cppmerge_%j.log
#SBATCH -e /work/morrisliu07/job_err/c3_cppmerge_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# C3 E2E: topm m32k16 offload + GPU resident merge (cpp_merge).
# A/B 對照 = cmp_topm_m32k16_offload (CPU merge, MAT 2.42 / TPS 0.647)。
# acceptance 應一致（merge bit-exact, C2）；驗 TPS 是否反超 SpecMoE (2.29)。
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
echo "[c3] node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "======== rebuild _store (incremental safety) ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -5
rc=${PIPESTATUS[0]}
echo "[c3] build rc=${rc}"
if [ "${rc}" -ne 0 ]; then echo "[c3] BUILD FAILED"; exit "${rc}"; fi

echo "======== C3 cpp_merge E2E ========"
cd "${REPO_ROOT}"
.venv/bin/python -m aug_spec.cli run \
    --config configs/cmp_topm_m32k16_offload_cppmerge.yaml \
    || echo "[c3] RUN FAILED"
echo "[c3] done"
