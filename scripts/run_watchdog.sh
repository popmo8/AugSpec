#!/bin/bash
#SBATCH --job-name=watchdog
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
# Run aug_spec config(s) under a STALL WATCHDOG. The moe_infinity offload engine
# can dead-lock silently mid-run (no traceback, no error — the process just sits
# RUNNING and wastes the GPU to the time limit). This wrapper kills it and exits
# non-zero the moment progress stalls, so a hang is reported instead of ignored.
#
#   sbatch --job-name=q5_randunif scripts/run_watchdog.sh configs/X.yaml
#   STALL_SECONDS=900 sbatch ... scripts/run_watchdog.sh configs/X.yaml
#   sbatch --exclude=hgpn39 --job-name=... scripts/run_watchdog.sh configs/X.yaml
#
# Mechanism: run output is unbuffered (-u) to a per-config file; every printed
# line (progress + per-question HF logs) bumps its mtime. If the file goes
# untouched for STALL_SECONDS (default 1200 = 20 min, safely above the healthy
# per-question cadence) the run is killed (SIGKILL + child reap) and rc=124.
set -uo pipefail
[ $# -ge 1 ] || { echo "[watchdog] usage: run_watchdog.sh <config.yaml> [more...]" >&2; exit 2; }

REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
STALL="${STALL_SECONDS:-1200}"
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
echo "[watchdog] node=$(hostname) job=${SLURM_JOB_ID:-local} stall=${STALL}s cfgs=$*"

overall=0
for cfg in "$@"; do
    echo "======================== ${cfg} ========================"
    RUNLOG="/work/morrisliu07/job_log/wd_${SLURM_JOB_ID:-local}_$(basename "${cfg}" .yaml).out"
    : > "${RUNLOG}"
    .venv/bin/python -u -m aug_spec.cli run --config "${cfg}" > "${RUNLOG}" 2>&1 &
    PID=$!
    tail -f "${RUNLOG}" & TAILPID=$!

    rc=0
    while kill -0 "${PID}" 2>/dev/null; do
        sleep 60
        last=$(stat -c %Y "${RUNLOG}" 2>/dev/null || echo 0)
        age=$(( $(date +%s) - last ))
        if [ "${age}" -gt "${STALL}" ]; then
            echo ""
            echo "[watchdog] !!!!!! STALL DETECTED: no output for ${age}s (> ${STALL}s)."
            echo "[watchdog] offload engine almost certainly dead-locked; killing PID ${PID} (cfg=${cfg})." >&2
            kill -9 "${PID}" 2>/dev/null
            pkill -9 -P "${PID}" 2>/dev/null
            rc=124
            break
        fi
    done
    if [ "${rc}" -eq 0 ]; then wait "${PID}"; rc=$?; fi
    kill "${TAILPID}" 2>/dev/null; wait "${TAILPID}" 2>/dev/null || true
    echo "[watchdog] ${cfg} exited rc=${rc}"
    [ "${rc}" -ne 0 ] && overall="${rc}"
done

echo "[watchdog] all done; overall rc=${overall}"
exit "${overall}"
