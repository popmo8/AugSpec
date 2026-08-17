#!/bin/bash
#SBATCH --job-name=q1_actsim_cyclesim
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/%x_%j.log
#SBATCH -e /work/morrisliu07/job_err/%x_%j.err
#
# Per-cycle similarity DRIFT experiment, qpc=1. Uses the already-built C++ ext
# (NO rebuild). Runs activation_similarity with AUG_DUMP_CYCLE_SIM: each cycle's
# standalone pairwise output-cosine is dumped, so we can measure how much a
# pair's similarity swings cycle-to-cycle. Stall-watchdogged.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
export AUG_DUMP_CYCLE_SIM="${REPO_ROOT}/output/q1_512_tm_actsim/cycle_sim.jsonl"
STALL="${STALL_SECONDS:-1200}"
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
cd "${REPO_ROOT}"
mkdir -p output/q1_512_tm_actsim
echo "[q1_actsim] node=$(hostname) job=${SLURM_JOB_ID:-local} (no rebuild; using existing .so)"

RUNLOG="/work/morrisliu07/job_log/q1actsim_run_${SLURM_JOB_ID:-local}.out"
: > "${RUNLOG}"
"${PY}" -u -m aug_spec.cli run --config configs/q1_512_tm_actsim.yaml > "${RUNLOG}" 2>&1 &
PID=$!
tail -f "${RUNLOG}" & TAILPID=$!
rc=0
while kill -0 "${PID}" 2>/dev/null; do
    sleep 60
    age=$(( $(date +%s) - $(stat -c %Y "${RUNLOG}" 2>/dev/null || echo 0) ))
    if [ "${age}" -gt "${STALL}" ]; then
        echo "[q1_actsim] !!! STALL ${age}s > ${STALL}s — killing ${PID}" >&2
        kill -9 "${PID}" 2>/dev/null; pkill -9 -P "${PID}" 2>/dev/null
        rc=124; break
    fi
done
[ "${rc}" -eq 0 ] && { wait "${PID}"; rc=$?; }
kill "${TAILPID}" 2>/dev/null; wait "${TAILPID}" 2>/dev/null || true
echo "[q1_actsim] run exited rc=${rc}"

echo "======== PER-CYCLE SIMILARITY DRIFT ANALYSIS ========"
[ -f "${AUG_DUMP_CYCLE_SIM}" ] && "${PY}" scripts/analyze_cycle_sim.py \
    "${AUG_DUMP_CYCLE_SIM}" output/q1_512_tm_actsim || echo "(no cycle_sim dump)"
echo "[q1_actsim] done"
