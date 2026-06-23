#!/bin/bash
#SBATCH --job-name=aug_prof
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/aug_prof_%j.log
#SBATCH -e /work/morrisliu07/job_err/aug_prof_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# Profiling (AUG_PROFILE=1): per-cycle time breakdown for topm vs specmoe to
# validate H1 (topm verify overload-serialises fetch) / H2 (evict churn) /
# H5 (specmoe draft re-fetch from late pinning). Rebuilds with the profiling
# counters AND the Enqueue race fix (wait-retry), so the runs don't abort.
#   Track A — counters: prof_*_cnt (mnt=128) → [AUG_PROFILE] table in the log.
#   Track B — nsys timeline: prof_*_ns (mnt=24) → .nsys-rep (open in Nsight
#             Systems to see H2D vs kernel overlap) + nsys stats summary.
# Backend = engine_bmm (the optimised C++ DispatchBmm both drafts share) — we
# profile the version we actually run. Verify-side hypotheses (H1 overload,
# H2 evict, race) are draft-backend-independent; H5 (SpecMoE draft re-fetch)
# still shows via the kept_bmm_state fallback to dispatch on a non-resident kept.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"; MOE_ROOT="${REPO_ROOT}/moe_infinity"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUTLASS_DIR="${HOME}/cutlass"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export AUG_PROFILE=1
export AUG_MERGED_BACKEND=engine_bmm   # profile the optimised (C++ bmm) path
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
NSYS="/work/HPC_software/LMOD/nvidia/packages/cuda-12.6/bin/nsys"
PROF_DIR="${REPO_ROOT}/output/profiling"; mkdir -p "${PROF_DIR}"
echo "[prof] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== rebuild engine (profiling counters + Enqueue race fix) ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -5
rc=${PIPESTATUS[0]}; echo "[prof] build rc=${rc}"
if [ "${rc}" -ne 0 ]; then echo "[prof] BUILD FAILED"; exit "${rc}"; fi
cd "${REPO_ROOT}"

echo "======== Track A: counters (mnt=128) ========"
echo "-------- topm --------"
.venv/bin/python -m aug_spec.cli run --config configs/prof_topm_cnt.yaml || echo "[prof] topm_cnt FAILED"
echo "-------- specmoe --------"
.venv/bin/python -m aug_spec.cli run --config configs/prof_specmoe_cnt.yaml || echo "[prof] specmoe_cnt FAILED"

echo "======== Track B: nsys timelines (mnt=24) ========"
if [ -x "${NSYS}" ]; then
  for who in topm specmoe; do
    echo "-------- nsys ${who} --------"
    "${NSYS}" profile -o "${PROF_DIR}/prof_${who}" --trace=cuda --force-overwrite true \
      "${PY}" -m aug_spec.cli run --config configs/prof_${who}_ns.yaml \
      || echo "[prof] nsys ${who} FAILED"
    echo "==== nsys stats: ${who} (memcpy + kernel summary) ===="
    "${NSYS}" stats --report cuda_gpu_mem_time_sum,cuda_gpu_kern_sum \
      "${PROF_DIR}/prof_${who}.nsys-rep" 2>/dev/null | head -40 || true
  done
else
  echo "[prof] nsys not found at ${NSYS} — skipped Track B (counters still valid)"
fi
echo "[prof] done — counter tables above; .nsys-rep in ${PROF_DIR}/"
