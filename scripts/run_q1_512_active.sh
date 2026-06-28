#!/bin/bash
#SBATCH --job-name=q1_512_active
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/q1_512_active_%j.log
#SBATCH -e /work/morrisliu07/job_err/q1_512_active_%j.err
#
# Diagnostic: q_per_cat=1, mnt=512. Dumps the per-(layer,question,cycle)
# selected expert set (AUG_DUMP_ACTIVE_SET) and runs the co-occurrence +
# cycle-to-cycle set-shift analyses.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"

OUT_DIR="output/q1_512_tm_active"
DUMP="${OUT_DIR}/active_set.jsonl"
mkdir -p "${OUT_DIR}"
rm -f "${DUMP}"

echo "[q1_512_active] node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "======== topm, qpc=1, mnt=512, dumping selected expert sets ========"
AUG_NO_OVERLOAD=1 AUG_DUMP_ACTIVE_SET="${DUMP}" \
    .venv/bin/python -m aug_spec.cli run \
    --config configs/q1_512_tm_active.yaml || echo "[q1_512_active] RUN FAILED"

echo "======== CO-OCCURRENCE + SET-SHIFT ANALYSIS ========"
.venv/bin/python scripts/analyze_active_set.py "${DUMP}" "${OUT_DIR}"

echo "[q1_512_active] done"
