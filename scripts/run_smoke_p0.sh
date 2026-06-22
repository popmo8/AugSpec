#!/bin/bash
#SBATCH --job-name=smoke_p0
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/smoke_p0_%j.log
#SBATCH -e /work/morrisliu07/job_err/smoke_p0_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# Smoke: OffloadMergeEngine 重構後 merge_offload 路徑端到端（純 Python 改動，不重編）。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
echo "[smoke] node=$(hostname) job=${SLURM_JOB_ID:-local}"
cd "${REPO_ROOT}"
.venv/bin/python -m aug_spec.cli run --config configs/smoke_p0_budget.yaml && echo "[smoke] PASS" || echo "[smoke] FAIL"
echo "[smoke] done"
