#!/bin/bash
#SBATCH --job-name=rb_smoke
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/rb_smoke_%j.log
#SBATCH -e /work/morrisliu07/job_err/rb_smoke_%j.err
#
# Recompile moe_infinity's C++ extension (after editing kMaxTokens in
# expert_module.cpp) WITHOUT touching any Python deps, then run smoke.
#
# Lesson from job 212255: `uv pip install -e ./moe_infinity ...` re-resolves
# moe_infinity's declared deps and silently downgrades transformers
# 4.57.6 → 4.53.0 (a version pin in moe_infinity's setup.py). To avoid
# that, we bypass pip entirely for the rebuild — `python setup.py
# build_ext --inplace` only compiles the .so and drops it into the
# source tree where the existing editable install expects it.
#
# Step 1 restores transformers/tokenizers to the versions Phase 0 +
# HF smoke validated on (since the prior failed run downgraded them).

set -euo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0

# locate CUTLASS — same logic as install.sh
if [[ -z "${CUTLASS_DIR:-}" ]]; then
    for candidate in "${HOME}/cutlass" "/work/${USER}/cutlass"; do
        if [[ -d "${candidate}/include/cutlass" ]]; then
            export CUTLASS_DIR="${candidate}"
            break
        fi
    done
fi
echo "[rb_smoke] CUTLASS_DIR=${CUTLASS_DIR:-NOT-FOUND}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

echo "[rb_smoke] node=$(hostname)  job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo
echo "============================================================"
echo "  Step 1/3: restore transformers==4.57.6 + tokenizers"
echo "  (undo the prior accidental downgrade)"
echo "============================================================"
uv pip install 'transformers==4.57.6' 'tokenizers>=0.22' \
    --force-reinstall --no-deps 2>&1 | tail -10

echo
echo "============================================================"
echo "  Step 2/3: rebuild moe_infinity .so via build_ext --inplace"
echo "  (kMaxTokens 128 -> 2048; no pip, no dep resolve)"
echo "============================================================"
cd "${REPO_ROOT}/moe_infinity"
python -u setup.py build_ext --inplace 2>&1 | tail -30
cd "${REPO_ROOT}"

# Confirm post-rebuild state.
python -u -c "
import transformers
import moe_infinity
from moe_infinity import MoE
print('transformers:', transformers.__version__)
print('moe_infinity import: OK')
print('MoE class: OK')
"

echo
echo "============================================================"
echo "  Step 3/3: offload smoke"
echo "============================================================"
aug_spec run --config configs/_smoke_offload.yaml
