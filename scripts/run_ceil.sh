#!/bin/bash
#SBATCH --job-name=ceil_rep
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/%x_%j.log
#SBATCH -e /work/morrisliu07/job_err/%x_%j.err
set -uo pipefail
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_BATCH_REPLICATE=1     # every B=4 batch = one question replicated 4x
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd /work/morrisliu07/aug_spec
echo "[ceil] AUG_BATCH_REPLICATE=1 (identical B=4 batches)"
.venv/bin/python -u -m aug_spec.cli run --config configs/ceil_ours_b4rep.yaml
