#!/bin/bash
#SBATCH --job-name=mx1_artifacts
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
# Mixtral Table-1 static-baseline artifacts (mixtral_acceptance_plan.md):
#   1. NAEE kept-set search, EXACT enumeration (C(8,1)=8/layer) — needs GPU.
#   2. HC-SMoE grouping spec, K=1 (CPU).
#   3. MC-SMoE grouping spec, average K=1 (CPU).
# Requires B1 calibration at output/calibration/Mixtral-8x7B-v0.1
# (scripts/run_collect_calib.sh mistralai/Mixtral-8x7B-v0.1) — submit this
# job with --dependency=afterok:<calib job id>.
set -euo pipefail

REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"

CALIB=output/calibration/Mixtral-8x7B-v0.1
[ -f "${CALIB}/meta.json" ] || {
    echo "[mx1_artifacts] missing B1 calibration at ${CALIB}" >&2; exit 3; }

echo "[mx1_artifacts] 1/3 NAEE exact search (r=1)"
.venv/bin/python -u scripts/search_naee.py \
    --model-id mistralai/Mixtral-8x7B-v0.1 --calib-dir "${CALIB}" --r 1

echo "[mx1_artifacts] 2/3 HC-SMoE spec (K=1)"
.venv/bin/python -u scripts/build_hc_smoe.py --calib-dir "${CALIB}" --K 1

echo "[mx1_artifacts] 3/3 MC-SMoE spec (K=1)"
.venv/bin/python -u scripts/build_mc_smoe.py --calib-dir "${CALIB}" --K 1

echo "[mx1_artifacts] done:"
ls -l output/naee/Mixtral-8x7B-v0.1_r1.json \
      output/hc_smoe/Mixtral-8x7B-v0.1_K1.json \
      output/mc_smoe/Mixtral-8x7B-v0.1_K1.json
