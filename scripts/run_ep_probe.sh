#!/bin/bash
#SBATCH --job-name=ep_probe
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/%x_%j.log
#SBATCH -e /work/morrisliu07/job_err/%x_%j.err
#
# memory_hierarchy_plan.md §6-0: does early_pin drive specmoe draft_fetch -> 0
# under the CLEAN strong-baseline setup (pin + no_overload)? Runs ep=0/1/2 at
# qpc=5 so draft_fetch totals are directly comparable. AUG_PROFILE on.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
STALL="${STALL_SECONDS:-1200}"
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
cd "${REPO_ROOT}"
echo "[ep_probe] node=$(hostname) job=${SLURM_JOB_ID:-local}"
for ep in 0 1 2; do
  cfg="configs/q15_specmoe_ep${ep}.yaml"
  echo "======================== ${cfg} ========================"
  RUNLOG="/work/morrisliu07/job_log/epprobe_${SLURM_JOB_ID:-local}_ep${ep}.out"
  : > "${RUNLOG}"
  "${PY}" -u -m aug_spec.cli run --config "${cfg}" > "${RUNLOG}" 2>&1 &
  PID=$!; tail -f "${RUNLOG}" & TAILPID=$!
  rc=0
  while kill -0 "${PID}" 2>/dev/null; do
    sleep 60
    age=$(( $(date +%s) - $(stat -c %Y "${RUNLOG}" 2>/dev/null || echo 0) ))
    if [ "${age}" -gt "${STALL}" ]; then
      echo "[ep_probe] !!! STALL ${age}s -- killing ${PID} (ep${ep})" >&2
      kill -9 "${PID}" 2>/dev/null; pkill -9 -P "${PID}" 2>/dev/null; rc=124; break
    fi
  done
  [ "${rc}" -eq 0 ] && { wait "${PID}"; rc=$?; }
  kill "${TAILPID}" 2>/dev/null; wait "${TAILPID}" 2>/dev/null || true
  echo "[ep_probe] ep${ep} exited rc=${rc}"
  echo "----- ep${ep} draft_fetch line -----"
  grep -E "draft_fetch|verify_fetch|fetched .* GB total|kept_changed" "${RUNLOG}" | tail -5
done
echo "[ep_probe] all done"
