#!/bin/bash
#SBATCH --job-name=q5_actsim
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/%x_%j.log
#SBATCH -e /work/morrisliu07/job_err/%x_%j.err
#
# activation_similarity end-to-end: (1) rebuild the moe_infinity C++ extension
# ON THE COMPUTE NODE (never the login node — adds the per-expert-output capture
# binding), (2) verify the new binding imports, (3) run the actsim config with
# AUG_DUMP_PAIRS under a stall watchdog. Dumps pairs for the cross-cycle overlap
# analysis (same as cooccur_pair).
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
export AUG_DUMP_PAIRS="${REPO_ROOT}/output/q5_512_tm_actsim/pairs.jsonl"
STALL="${STALL_SECONDS:-1200}"
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
echo "[actsim] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== (1) build C++ extension (compute node) ========"
cd "${REPO_ROOT}/moe_infinity"
"${PY}" setup.py build_ext --inplace || { echo "[actsim] BUILD FAILED"; exit 2; }

echo "======== (2) verify compiled ext imports ========"
cd "${REPO_ROOT}"
"${PY}" -c "import moe_infinity; print('[actsim] moe_infinity ext import OK')" \
    || { echo "[actsim] EXT IMPORT FAILED"; exit 3; }

echo "======== (3) run actsim under stall watchdog ========"
mkdir -p "${REPO_ROOT}/output/q5_512_tm_actsim"
RUNLOG="/work/morrisliu07/job_log/actsim_run_${SLURM_JOB_ID:-local}.out"
: > "${RUNLOG}"
"${PY}" -u -m aug_spec.cli run --config configs/q5_512_tm_actsim.yaml > "${RUNLOG}" 2>&1 &
PID=$!
tail -f "${RUNLOG}" & TAILPID=$!
rc=0
while kill -0 "${PID}" 2>/dev/null; do
    sleep 60
    age=$(( $(date +%s) - $(stat -c %Y "${RUNLOG}" 2>/dev/null || echo 0) ))
    if [ "${age}" -gt "${STALL}" ]; then
        echo "[actsim] !!! STALL ${age}s > ${STALL}s — killing ${PID}" >&2
        kill -9 "${PID}" 2>/dev/null; pkill -9 -P "${PID}" 2>/dev/null
        rc=124; break
    fi
done
[ "${rc}" -eq 0 ] && { wait "${PID}"; rc=$?; }
kill "${TAILPID}" 2>/dev/null; wait "${TAILPID}" 2>/dev/null || true
echo "[actsim] run exited rc=${rc}"
exit "${rc}"
