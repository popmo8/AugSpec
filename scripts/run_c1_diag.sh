#!/bin/bash
#SBATCH --job-name=c1_diag
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/c1_diag_%j.log
#SBATCH -e /work/morrisliu07/job_err/c1_diag_%j.err
#
# C1 mixed-device 診斷:c1_smoke 單支,merged_cache 的 fail-fast assert
# 會帶出 layer/group/src 脈絡。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
export AUG_HANG_DEBUG=180
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
cd "${REPO_ROOT}"
echo "[c1_diag] node=$(hostname) job=${SLURM_JOB_ID:-local}"
"${PY}" -m aug_spec.cli run --config configs/c1_smoke.yaml || echo "[c1_diag] RUN FAILED (see err for the tagged assert)"
echo "[c1_diag] done"
