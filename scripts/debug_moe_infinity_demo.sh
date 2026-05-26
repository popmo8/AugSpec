#!/bin/bash
#SBATCH --job-name=moe_demo
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/moe_demo_%j.log
#SBATCH -e /work/morrisliu07/job_err/moe_demo_%j.err
#
# Run moe_infinity's official Qwen3 demo to check whether moe_infinity
# itself produces correct outputs. This is the cleanest signal of
# whether our problem is:
#   (a) moe_infinity is broken on this system → demo also produces garbage
#   (b) moe_infinity is fine but its Mixtral path is bad → demo works,
#       but our Mixtral logits are wrong (already confirmed)
#   (c) we're missing some setup → demo's setup teaches us what

set -euo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0
# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

echo "[moe_demo] node=$(hostname)  job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Make the offload dir writable + ensure it exists.
mkdir -p /work/morrisliu07/aug_spec/cache/offload/qwen3_demo

# Run the official demo. Override offload_dir to a path we own.
python -u moe_infinity/examples/readme_example.py \
    --checkpoint Qwen/Qwen3-30B-A3B \
    --offload_dir /work/morrisliu07/aug_spec/cache/offload/qwen3_demo
