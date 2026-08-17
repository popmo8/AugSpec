#!/bin/bash
#SBATCH --job-name=bmm_diag
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/bmm_diag_%j.log
#SBATCH -e /work/morrisliu07/job_err/bmm_diag_%j.err
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
echo "[diag] node=$(hostname)"
AUG_EARLY_PIN=1 AUG_NO_OVERLOAD=1 .venv/bin/python -m aug_spec.cli run --config configs/exp_noov.yaml 2>&1 | grep -E "kept_bmm|kept_changed|draft_fetch|MAT|build" || echo FAILED
echo "[diag] done"
