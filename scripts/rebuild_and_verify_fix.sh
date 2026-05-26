#!/bin/bash
#SBATCH --job-name=rb_verify
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/rb_verify_%j.log
#SBATCH -e /work/morrisliu07/job_err/rb_verify_%j.err
#
# Clean rebuild moe_infinity (after fixing the use-after-free in
# expert_module.cpp's SetTensorsFromIds → torch::empty instead of
# from_blob+DoNothingDeleter), then immediately verify by running:
#   1. debug_generate_mixtral.py — does plain generate produce English?
#   2. debug_offload_vs_gpu_per_layer.py — does every MoE layer match GPU-full?
#
# Previous rebuild attempts failed silently — build.ninja was overwritten by
# the _engine extension and expert_module.o never got recompiled. To avoid
# any chance of stale artifacts being reused, this script removes build/
# and the in-tree .so files BEFORE rebuilding.

set -euo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0

if [[ -z "${CUTLASS_DIR:-}" ]]; then
    for candidate in "${HOME}/cutlass" "/work/${USER}/cutlass"; do
        if [[ -d "${candidate}/include/cutlass" ]]; then
            export CUTLASS_DIR="${candidate}"
            break
        fi
    done
fi
echo "[rb_verify] CUTLASS_DIR=${CUTLASS_DIR:-NOT-FOUND}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"

echo "[rb_verify] node=$(hostname)  job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo
echo "============================================================"
echo "  Step 0/4: pre-clean (remove build/ and in-tree .so)"
echo "============================================================"
cd "${REPO_ROOT}/moe_infinity"
rm -rf build/
rm -f moe_infinity/_engine.cpython-310-x86_64-linux-gnu.so
rm -f moe_infinity/_store.cpython-310-x86_64-linux-gnu.so
echo "  build/ removed: $([[ -d build ]] && echo NO || echo YES)"
echo "  in-tree _store.so removed: $([[ -f moe_infinity/_store.cpython-310-x86_64-linux-gnu.so ]] && echo NO || echo YES)"

echo
echo "============================================================"
echo "  Step 1/4: confirm source has the fix"
echo "============================================================"
grep -n "torch::empty(tensor_shape, options)" core/parallel/expert_module.cpp | head -5
grep -n "torch::empty(data_shape, options)" core/parallel/expert_module.cpp | head -5
echo "  source mtime: $(stat -c %y core/parallel/expert_module.cpp)"

echo
echo "============================================================"
echo "  Step 2/4: rebuild moe_infinity from scratch"
echo "============================================================"
t0=$SECONDS
python -u setup.py build_ext --inplace 2>&1 | tail -40
echo "  rebuild took $((SECONDS - t0))s"

echo
echo "  --- verifying expert_module.o was actually compiled ---"
EM_O=$(find build/temp.linux-x86_64-cpython-310 -name "expert_module.o" 2>/dev/null || true)
if [[ -z "${EM_O}" ]]; then
    echo "  FATAL: expert_module.o NOT FOUND after rebuild — build is broken"
    ls -la build/temp.linux-x86_64-cpython-310/core/parallel/ 2>/dev/null || true
    exit 1
fi
echo "  expert_module.o: $(ls -la ${EM_O})"

echo "  --- verifying .so was newly linked ---"
ls -la moe_infinity/_store.cpython-310-x86_64-linux-gnu.so

cd "${REPO_ROOT}"

echo
echo "============================================================"
echo "  Step 3/4: plain generate on Mixtral (English vs gibberish?)"
echo "============================================================"
python -u tests/debug_generate_mixtral.py 2>&1 | tail -80

echo
echo "============================================================"
echo "  Step 4/4: per-layer offload vs GPU full (numerical match?)"
echo "============================================================"
python -u tests/debug_offload_vs_gpu_per_layer.py 2>&1 | tail -80

echo
echo "============================================================"
echo "  DONE"
echo "============================================================"
