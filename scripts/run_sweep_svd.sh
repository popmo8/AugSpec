#!/bin/bash
#SBATCH --job-name=qwen3_count_k16
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/qwen3_count_k16_%j.log
#SBATCH -e /work/morrisliu07/job_err/qwen3_count_k16_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# SVD subspace merge sweep: runs each model's SVD topm_count against its
# plain topm_count baseline so results are directly comparable.
#
# gpus-per-node=2 is required for Mixtral-8x7B and Qwen3-30B-A3B (≥ 94 GB
# bf16).  GPT-OSS-20B fits on 1 GPU but we keep 2 here for a uniform node.
# Override at submit time if you only want the GPT-OSS pair:
#
#     sbatch --gpus-per-node=1 --time=4:00:00 scripts/run_sweep_svd.sh
#
# Pair layout (baseline → SVD):
#   mixtral_topm_count     → mixtral_topm_count_svd
#   gptoss_topm_count      → gptoss_topm_count_svd   (label: gptoss_top8_count)
#   qwen3_topm_count       → qwen3_topm_count_svd

set -euo pipefail

REPO_ROOT="/work/morrisliu07/aug_spec"

# ════════════════════════════════════════════════════════════════════════
#  ✏  Edit me: configs to run in this sweep (in order).
#     Baselines run first so the SVD run's output/*.json can be diffed
#     immediately after the job finishes.
# ════════════════════════════════════════════════════════════════════════
CONFIGS=(
    # configs/mixtral_topm_count_svd.yaml
    # configs/gptoss_topm_count_svd.yaml
    # configs/qwen3_topm_count_svd.yaml
    # configs/qwen3_topm_count_k16.yaml
    # configs/qwen3_topm_count_svd_k16.yaml
    configs/qwen3_count_k16.yaml
    # configs/qwen3_count_svd_k16.yaml
)
# ════════════════════════════════════════════════════════════════════════

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "[svd_sweep] CONFIGS array is empty. Edit $0 to add configs." >&2
    exit 1
fi

export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0

# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"

cd "${REPO_ROOT}"

# Pre-flight: every config must exist before we burn GPU time on a typo.
for cfg in "${CONFIGS[@]}"; do
    if [[ ! -f "${cfg}" ]]; then
        echo "[svd_sweep] not found: ${cfg}" >&2
        exit 1
    fi
done

echo "[svd_sweep] node=$(hostname)  job=${SLURM_JOB_ID:-local}  n=${#CONFIGS[@]}"
echo "[svd_sweep] aug_spec at $(which aug_spec)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for cfg in "${CONFIGS[@]}"; do
    echo
    echo "========================================================================"
    echo "[svd_sweep] $(date -Iseconds)  cfg=${cfg}"
    echo "========================================================================"
    aug_spec run --config "${cfg}"
done

echo
echo "[svd_sweep] $(date -Iseconds)  all ${#CONFIGS[@]} configs completed"
