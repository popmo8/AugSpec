#!/bin/bash
#SBATCH --job-name=gen_mxt
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/gen_mxt_%j.log
#SBATCH -e /work/morrisliu07/job_err/gen_mxt_%j.err
#
# Plain text generation on offload Mixtral via moe.model.generate. No
# spec-decoding, no aug_spec, no comparisons. Just: does it produce
# coherent English text or gibberish?

set -euo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0
# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

echo "[gen_mxt] node=$(hostname)  job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -u tests/debug_generate_mixtral.py
