#!/bin/bash
#SBATCH --job-name=aug_spec
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#
# Resource note:
#   aug_spec runs as ONE Python process — ntasks must stay 1.
#   gpus-per-node=1 fits gpt-oss-20b (~40 GB). For Mixtral-8x7B
#   (94 GB bf16) or Qwen3-30B-A3B override at submit time:
#       sbatch --gpus-per-node=2 scripts/run.sh configs/mixtral_count.yaml
#   normal2 caps cpus-per-node at 12; total here = 1 × 4 = 4.
#SBATCH -o /work/morrisliu07/job_log/aug_spec_%j.log
#SBATCH -e /work/morrisliu07/job_err/aug_spec_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# Run one (or many) aug_spec experiments on TWCC under SLURM.
#
# Usage:
#   sbatch scripts/run.sh configs/mixtral_count.yaml
#   sbatch scripts/run.sh configs/mixtral_count.yaml configs/qwen3_count.yaml
#   sbatch --job-name=mx_count scripts/run.sh configs/mixtral_count.yaml
#
# Each YAML is run sequentially in the same job; the job aborts on the
# first failing experiment (set -e). Outputs go to whatever `output.dir`
# says in each YAML (default: output/<config stem>).

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "[run.sh] No config provided. Pass one or more .yaml paths." >&2
    exit 1
fi

REPO_ROOT="/work/morrisliu07/aug_spec"

export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0

# Use the repo's own venv (NOT the old ~/.conda/envs/moe).
# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"

cd "${REPO_ROOT}"

echo "[run.sh] node=$(hostname)  job=${SLURM_JOB_ID:-local}  cfgs=$#"
echo "[run.sh] aug_spec at $(which aug_spec)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for cfg in "$@"; do
    echo
    echo "========================================================================"
    echo "[run.sh] $(date -Iseconds)  cfg=${cfg}"
    echo "========================================================================"
    aug_spec run --config "${cfg}"
done

echo
echo "[run.sh] $(date -Iseconds)  all configs completed"
