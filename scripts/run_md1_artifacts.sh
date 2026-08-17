#!/bin/bash
#SBATCH --job-name=md1_artifacts
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
# DeepSeek-MoE-16B-base Table-1 static-baseline artifacts (deepseek_plan.md):
#   1. NAEE kept-set search, SAMPLED 1e5 subsets (C(64,8)~4.4e9) — needs GPU.
#   2. HC-SMoE grouping spec, K=8 (CPU).
#   3. MC-SMoE grouping spec, average K=8 (CPU).
# Requires B1 calibration at output/calibration/deepseek-moe-16b-base
# (scripts/run_collect_calib.sh deepseek-ai/deepseek-moe-16b-base) — submit
# this job with --dependency=afterok:<calib job id>.
set -euo pipefail

REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"

CALIB=output/calibration/deepseek-moe-16b-base
[ -f "${CALIB}/meta.json" ] || {
    echo "[md1_artifacts] missing B1 calibration at ${CALIB}" >&2; exit 3; }

echo "[md1_artifacts] 1/3 NAEE sampled search (r=8, 1e5 candidates)"
.venv/bin/python -u scripts/search_naee.py \
    --model-id deepseek-ai/deepseek-moe-16b-base --calib-dir "${CALIB}" --r 8

echo "[md1_artifacts] 2/3 HC-SMoE spec (K=8)"
.venv/bin/python -u scripts/build_hc_smoe.py --calib-dir "${CALIB}" --K 8

echo "[md1_artifacts] 3/3 MC-SMoE spec (K=8)"
.venv/bin/python -u scripts/build_mc_smoe.py --calib-dir "${CALIB}" --K 8

echo "[md1_artifacts] done:"
ls -l output/naee/deepseek-moe-16b-base_r8.json \
      output/hc_smoe/deepseek-moe-16b-base_K8.json \
      output/mc_smoe/deepseek-moe-16b-base_K8.json
