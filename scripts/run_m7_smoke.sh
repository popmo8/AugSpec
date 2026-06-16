#!/bin/bash
#SBATCH --job-name=m7_smoke
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/m7_smoke_%j.log
#SBATCH -e /work/morrisliu07/job_err/m7_smoke_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# M7 — 第一條 offload 端到端 + hf 對照。
# 先跑 hf 對照(_smoke_count)建立 count 的 acceptance 基準,
# 再跑 offload(_smoke_offload_count),兩者 MAT/AccRate 應在 bf16 noise 內。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
echo "[m7] node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cd "${REPO_ROOT}"
echo "======== hf 對照 _smoke_count ========"
.venv/bin/python -m aug_spec.cli run --config configs/_smoke_count.yaml || echo "[m7] hf FAILED"
echo "======== offload _smoke_offload_count ========"
.venv/bin/python -m aug_spec.cli run --config configs/_smoke_offload_count.yaml || echo "[m7] offload FAILED"
echo "[m7] done"
