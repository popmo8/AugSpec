#!/bin/bash
#SBATCH --job-name=cmp_hf
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/cmp_hf_%j.log
#SBATCH -e /work/morrisliu07/job_err/cmp_hf_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# 跑兩個 hf backend 比較 — #1 specmoe N=16, #3 topm m32 k16。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
echo "[cmp_hf] node=$(hostname) job=${SLURM_JOB_ID:-local}"
cd "${REPO_ROOT}"
echo "======== #1 hf SpecMoE N=16 ========"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_specmoe_N16_hf.yaml || echo "[cmp_hf] specmoe FAILED"
echo "======== #3 hf topM m32 k16 ========"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_topm_m32k16_hf.yaml || echo "[cmp_hf] topm FAILED"
echo "[cmp_hf] done"
