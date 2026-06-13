#!/bin/bash
#SBATCH --job-name=m2_assisted
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/m2_assisted_%j.log
#SBATCH -e /work/morrisliu07/job_err/m2_assisted_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# M2 — offload_plan.md assisted generation 探針 wrapper。
#
# 第一輪（job 234621）教訓：device-side assert 會毒化整個 CUDA
# context，同進程的後續檢查全部連帶亂報 → A1/A2/A3 改成各自獨立
# 進程跑（--only），互不串染。CUDA_LAUNCH_BLOCKING=1 讓 assert
# 落在正確的 stack 行上，便於診斷（M2 只產 32 token，慢一點無所謂）。
#
# Usage:
#   sbatch tests/offload/m2_assisted.sh
#   bash   tests/offload/m2_assisted.sh        # 互動式 GPU 節點
#
# 報告：tests/offload/m2_assisted_{A1,A2,A3}.out（python 端負責）。

set -uo pipefail   # 故意不開 -e：單項紅了還要跑完其他項

REPO_ROOT="/work/morrisliu07/aug_spec"

export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true

echo "[m2] node=$(hostname)  job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

FAILED=""
for CHK in A1 A2 A3; do
    echo
    echo "========================================================"
    echo "[m2] running ${CHK} in its own process ..."
    echo "========================================================"
    "${REPO_ROOT}/.venv/bin/python" \
        "${REPO_ROOT}/tests/offload/m2_assisted.py" --only "${CHK}" "$@" \
        || FAILED="${FAILED} ${CHK}"
done

echo
if [[ -n "${FAILED}" ]]; then
    echo "[m2] RESULT: FAILED —${FAILED}"
    exit 1
fi
echo "[m2] RESULT: ALL PASS (A1 A2 A3)"
