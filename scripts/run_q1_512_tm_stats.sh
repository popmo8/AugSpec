#!/bin/bash
#SBATCH --job-name=q1_512_tm_stats
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/q1_512_tm_stats_%j.log
#SBATCH -e /work/morrisliu07/job_err/q1_512_tm_stats_%j.err
#
# Diagnostic: q_per_cat=1, mnt=512, no_overload ON, within-cluster merge =
# FREQUENCY (non-uniform / default). Dumps every per-cluster within-cluster
# weight vector (AUG_DUMP_CLUSTER_WEIGHTS) then runs the spread analysis so we
# can see how far the non-uniform weights sit from the uniform 1/|cluster|.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"

OUT_DIR="output/q1_512_tm_stats"
DUMP="${OUT_DIR}/cluster_weights.jsonl"
mkdir -p "${OUT_DIR}"
rm -f "${DUMP}"          # fresh dump each run (analysis appends otherwise)

echo "[q1_512_tm_stats] node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "======== topm freq within-cluster merge, qpc=1, mnt=512 ========"
AUG_NO_OVERLOAD=1 AUG_DUMP_CLUSTER_WEIGHTS="${DUMP}" \
    .venv/bin/python -m aug_spec.cli run \
    --config configs/q1_512_tm_stats.yaml || echo "[q1_512_tm_stats] RUN FAILED"

echo "======== CLUSTER-WEIGHT SPREAD ANALYSIS ========"
.venv/bin/python scripts/analyze_cluster_weights.py "${DUMP}" "${OUT_DIR}"

echo "[q1_512_tm_stats] done"
