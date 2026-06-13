#!/bin/bash
#SBATCH --job-name=top_b2
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/top1_sweep_%j.log
#SBATCH -e /work/morrisliu07/job_err/top1_sweep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# Single-job sweep: edit the CONFIGS list below, then
#
#     sbatch scripts/run_sweep_top1.sh
#
# Every config runs sequentially in the SAME job; on the first failure
# the rest are skipped (set -e). Pick `--gpus-per-node` at the top to
# match the largest model in the list — Mixtral / Qwen3-30B need 2,
# pure GPT-OSS sweeps can drop to 1 via `sbatch --gpus-per-node=1 ...`.

set -euo pipefail

REPO_ROOT="/work/morrisliu07/aug_spec"

# ════════════════════════════════════════════════════════════════════════
#  ✏  Edit me: configs to run in this sweep (in order).
# ════════════════════════════════════════════════════════════════════════
CONFIGS=(
    # configs/gptoss_top8_count.yaml
    # configs/gptoss_top16_count.yaml
    # configs/mixtral_top4_count.yaml
    # configs/mixtral_top6_count.yaml
    configs/qwen3_top4_count.yaml
    configs/qwen3_top16_count.yaml
    configs/qwen3_top32_count.yaml
    configs/qwen3_top64_count.yaml
)
# ════════════════════════════════════════════════════════════════════════

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "[run_sweep] CONFIGS array is empty. Edit $0 to add configs." >&2
    exit 1
fi

export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0

# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"

cd "${REPO_ROOT}"

# Pre-flight: every config must exist before we burn an hour of GPU time
# discovering a typo in the middle of the list.
for cfg in "${CONFIGS[@]}"; do
    if [[ ! -f "${cfg}" ]]; then
        echo "[run_sweep] not found: ${cfg}" >&2
        exit 1
    fi
done

echo "[run_sweep] node=$(hostname)  job=${SLURM_JOB_ID:-local}  n=${#CONFIGS[@]}"
echo "[run_sweep] aug_spec at $(which aug_spec)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for cfg in "${CONFIGS[@]}"; do
    echo
    echo "========================================================================"
    echo "[run_sweep] $(date -Iseconds)  cfg=${cfg}"
    echo "========================================================================"
    aug_spec run --config "${cfg}"
done

echo
echo "[run_sweep] $(date -Iseconds)  all ${#CONFIGS[@]} configs completed"
