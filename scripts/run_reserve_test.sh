#!/bin/bash
#SBATCH --job-name=reserve
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/reserve_%j.log
#SBATCH -e /work/morrisliu07/job_err/reserve_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# Test merged-reserve: b=0.2 + merge_during_verify (P2+P3) + 預扣 merged.
# 驗 (1) 跑得動 (sparse cache 1.97GB 夠), (2) NVML peak 從 24→~18GB (archer 縮),
# (3) 不 crash. 純 Python 改動，不重編。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
echo "[reserve] node=$(hostname) job=${SLURM_JOB_ID:-local}"
cd "${REPO_ROOT}"
.venv/bin/python -m aug_spec.cli run --config configs/p1_dv_on.yaml 2>&1 | grep -E "\[budget\]|\[vram\]|peak|FATAL|Aborted|OutOfMemory|FAIL" || echo "[reserve] FAILED"
echo "=== final ==="
wc -l output/p1_dv_on/per_question_summary.csv 2>/dev/null
echo "[reserve] done"
