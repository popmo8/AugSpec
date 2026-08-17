#!/bin/bash
#SBATCH --job-name=spec_n2
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/spec_n2_%j.log
#SBATCH -e /work/morrisliu07/job_err/spec_n2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# specmoe N=2 pin @ vram=0.4. 重編確保 pin code 在 .so，再跑。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"; MOE_ROOT="${REPO_ROOT}/moe_infinity"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUTLASS_DIR="${HOME}/cutlass"; export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
echo "[n2] node=$(hostname) job=${SLURM_JOB_ID:-local}"
cd "${MOE_ROOT}"; "${PY}" setup.py build_ext --inplace 2>&1 | tail -4
rc=${PIPESTATUS[0]}; echo "[n2] build rc=${rc}"; [ "${rc}" -ne 0 ] && { echo "[n2] BUILD FAILED"; exit "${rc}"; }
cd "${REPO_ROOT}"
.venv/bin/python -m aug_spec.cli run --config configs/specmoe_n32_v04.yaml 2>&1 | grep -E "\[budget\]|\[vram\]|peak|FATAL|Aborted|FAIL" || echo "[n2] FAILED"
echo "=== result ==="; tail -3 output/specmoe_n32_v04/per_question_summary.csv 2>/dev/null
echo "[n2] done"
