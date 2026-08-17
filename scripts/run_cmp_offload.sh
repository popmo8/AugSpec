#!/bin/bash
#SBATCH --job-name=cmp_offload
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/cmp_offload_%j.log
#SBATCH -e /work/morrisliu07/job_err/cmp_offload_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# 只重跑兩個 offload(#2 specmoe, #4 topm) — hf 已成功。warmup inference_mode
# → no_grad 修了 offload dispatcher 的 in-place index_add_ 崩潰。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
echo "[cmp_offload] node=$(hostname) job=${SLURM_JOB_ID:-local}"
cd "${REPO_ROOT}"
echo "======== #2 offload SpecMoE N=16 ========"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_specmoe_N16_offload.yaml || echo "[cmp_offload] specmoe FAILED"
echo "======== #4 offload topM m32 k16 ========"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_topm_m32k16_offload.yaml || echo "[cmp_offload] topm FAILED"
echo "[cmp_offload] done"
