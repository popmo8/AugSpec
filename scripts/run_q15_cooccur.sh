#!/bin/bash
#SBATCH --job-name=q15_cooccur
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/%x_%j.log
#SBATCH -e /work/morrisliu07/job_err/%x_%j.err
#
# 3 repeats (r1,r2,r3) of cooccur -- UNIFORM merge, skip mt_bench, qpc=15.
# Each repeat runs under a stall watchdog (offload deadlock -> killed + reported).
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
echo "[q15_cooccur] node=$(hostname) job=${SLURM_JOB_ID:-local}"
for cfg in configs/q15_cooccur_r1.yaml configs/q15_cooccur_r2.yaml configs/q15_cooccur_r3.yaml; do
  echo "======================== ${cfg} ========================"
  RUNLOG="/work/morrisliu07/job_log/q15run_${SLURM_JOB_ID:-local}_$(basename "${cfg}" .yaml).out"
  : > "${RUNLOG}"
  "${PY}" -u -m aug_spec.cli run --config "${cfg}" > "${RUNLOG}" 2>&1 &
  PID=$!
  tail -f "${RUNLOG}" & TAILPID=$!
  rc=0
  while kill -0 "${PID}" 2>/dev/null; do
    sleep 60
    age=$(( $(date +%s) - $(stat -c %Y "${RUNLOG}" 2>/dev/null || echo 0) ))
    if [ "${age}" -gt "${STALL}" ]; then
      echo "[q15_cooccur] !!! STALL ${age}s > ${STALL}s -- killing ${PID} (${cfg})" >&2
      kill -9 "${PID}" 2>/dev/null; pkill -9 -P "${PID}" 2>/dev/null; rc=124; break
    fi
  done
  [ "${rc}" -eq 0 ] && { wait "${PID}"; rc=$?; }
  kill "${TAILPID}" 2>/dev/null; wait "${TAILPID}" 2>/dev/null || true
  echo "[q15_cooccur] ${cfg} exited rc=${rc}"
done
echo "[q15_cooccur] all done"
