#!/bin/bash
#SBATCH --job-name=dbg_gdb
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/%x_%j.log
#SBATCH -e /work/morrisliu07/job_err/%x_%j.err
#
# Native-stack capture of the batch-loop hang (batch_spec_plan.md Phase-1 T2).
# Runs the hanging config; when the run log's mtime freezes (stall), attaches
# gdb to dump EVERY thread's native backtrace — this is the only way to see the
# archer C++ fetch/exec/merge threads (Python faulthandler shows only Python
# frames). Also snapshots GPU + process memory. Then kills.
set -uo pipefail
CONFIG="${1:-configs/dbg_ours_batch1.yaml}"      # default: cache-mode ON target
STALL="${STALL_SECONDS:-150}"

REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"

RUNLOG="/work/morrisliu07/job_log/gdb_${SLURM_JOB_ID:-local}_run.out"
DUMP="/work/morrisliu07/job_log/gdb_${SLURM_JOB_ID:-local}_stacks.txt"
: > "${RUNLOG}"
echo "[dbg_gdb] node=$(hostname) config=${CONFIG} stall=${STALL}s"
echo "[dbg_gdb] run log  -> ${RUNLOG}"
echo "[dbg_gdb] stacks   -> ${DUMP}"

.venv/bin/python -u -m aug_spec.cli run --config "${CONFIG}" > "${RUNLOG}" 2>&1 &
PID=$!
echo "[dbg_gdb] python PID=${PID}"

captured=0
while kill -0 "${PID}" 2>/dev/null; do
    sleep 30
    last=$(stat -c %Y "${RUNLOG}" 2>/dev/null || echo 0)
    age=$(( $(date +%s) - last ))
    # only arm after the run has actually started decoding (heartbeat present)
    started=$(grep -c "\[cycle \|\[batch_loop\]" "${RUNLOG}" 2>/dev/null || echo 0)
    if [ "${captured}" -eq 0 ] && [ "${started}" -gt 0 ] && [ "${age}" -gt "${STALL}" ]; then
        echo "[dbg_gdb] STALL: ${age}s no output — capturing native stacks of PID ${PID}"
        {
            echo "==================== $(date) ===================="
            echo "==== last heartbeat ===="
            grep "\[cycle " "${RUNLOG}" | tail -3
            echo "==== nvidia-smi ===="
            nvidia-smi 2>&1 | head -25
            echo "==== /proc/${PID}/status (mem) ===="
            grep -E "VmRSS|VmHWM|Threads" /proc/${PID}/status 2>/dev/null
            echo "==== gdb: thread apply all bt ===="
            gdb -p "${PID}" -batch \
                -ex "set pagination off" \
                -ex "set print pretty on" \
                -ex "info threads" \
                -ex "thread apply all bt" \
                -ex "detach" -ex "quit" 2>&1
        } > "${DUMP}" 2>&1
        echo "[dbg_gdb] stacks captured -> ${DUMP}"
        captured=1
        sleep 5
        kill -9 "${PID}" 2>/dev/null
        pkill -9 -P "${PID}" 2>/dev/null
        break
    fi
done
echo "[dbg_gdb] done (captured=${captured})"
