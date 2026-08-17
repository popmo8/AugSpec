#!/bin/bash
#SBATCH --job-name=moe_rebuild
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/moe_rebuild_%j.log
#SBATCH -e /work/morrisliu07/job_err/moe_rebuild_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# 重編 moe_infinity 的 C++ 擴充（_store / _engine），in-place 更新被 import 的 .so。
# ninja 增量編譯：只有改動的 .cpp/.cu 會重編 + relink。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
MOE_ROOT="${REPO_ROOT}/moe_infinity"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export CUTLASS_DIR="${HOME}/cutlass"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
echo "[rebuild] node=$(hostname) job=${SLURM_JOB_ID:-local} nvcc=$(which nvcc)"
cd "${MOE_ROOT}"

PY="${REPO_ROOT}/.venv/bin/python"

echo "======== build_ext --inplace ========"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -40
rc=${PIPESTATUS[0]}
echo "[rebuild] build_ext rc=${rc}"
if [ "${rc}" -ne 0 ]; then
    echo "[rebuild] BUILD FAILED"
    exit "${rc}"
fi

echo "======== import smoke ========"
"${PY}" - <<'PYEOF'
import moe_infinity, moe_infinity._store as s
print("[rebuild] moe_infinity import OK from", moe_infinity.__file__)
ed = getattr(s, "expert_dispatcher", None)
print("[rebuild] expert_dispatcher methods:",
      sorted(m for m in dir(ed) if not m.startswith("__")) if ed else "MISSING")
PYEOF
echo "[rebuild] done"
