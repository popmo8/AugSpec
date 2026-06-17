#!/bin/bash
#SBATCH --job-name=base_specmoe
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/base_specmoe_%j.log
#SBATCH -e /work/morrisliu07/job_err/base_specmoe_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# 四方對比（Base 模型）腳本 1/2 — SpecMoE N=16: hf + offload. qpc=3, mnt=256.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
echo "[base_specmoe] node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cd "${REPO_ROOT}"
echo "======== #1 specmoe hf (Base) ========"
.venv/bin/python -m aug_spec.cli run --config configs/base_specmoe_hf.yaml || echo "[base_specmoe] hf FAILED"
echo "======== #2 specmoe offload (Base) ========"
.venv/bin/python -m aug_spec.cli run --config configs/base_specmoe_offload.yaml || echo "[base_specmoe] offload FAILED"
echo "[base_specmoe] done"
