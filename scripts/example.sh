#!/bin/bash
#SBATCH --job-name=m0_offload
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/m0_offload_%j.log
#SBATCH -e /work/morrisliu07/job_err/m0_offload_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# M0 — offload_plan.md 第一階：原生 moe_infinity example 跑通（環境驗證）。
#
# 模型 = Qwen3-30B-A3B。原本測 Mixtral-8x7B，但 job 234500 重現了已知
# 問題：moe_infinity 跑 Mixtral 輸出亂碼（offload_plan.md §2 有記錄），
# Qwen3 路徑依既往經驗正常 → offload 主線全面改用 Qwen3。
# mixtral_example.py 雖以 Mixtral 命名，內容是通用的
# tokenizer + MoE + greedy generate，直接換 checkpoint 即可。
#
# Usage:
#   sbatch scripts/example.sh          # SLURM
#   bash scripts/example.sh            # 互動式 GPU 節點直接跑
#
# 驗收（offload_plan.md M0）：
#   1. 印出一段合理的續寫文字（不是亂碼 / 不是重複同一 token）
#   2. 全程無 crash
#   3. peak VRAM 記錄供參考 —— Qwen3 bf16 全量僅 ~61 GB，
#      ratio=0.75 下 expert 可能整包進 cache，這裡不當 offload 證據；
#      offload 行為的硬證據由 M1（ratio=0.15）提供

set -euo pipefail

REPO_ROOT="/work/morrisliu07/aug_spec"

export HF_TOKEN="${HF_TOKEN:?Please set HF_TOKEN in your environment before running}"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true

PYTHON="${REPO_ROOT}/.venv/bin/python"
MODEL="Qwen/Qwen3-30B-A3B"
OFFLOAD_DIR="${REPO_ROOT}/moe_infinity/offload_output/Qwen3-30B-A3B"

echo "[m0] node=$(hostname)  job=${SLURM_JOB_ID:-local}"
echo "[m0] model=${MODEL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── 背景 VRAM 取樣（每 5 秒一筆），結尾回報峰值 ──────────────────────
VRAM_LOG="$(mktemp /tmp/m0_vram.XXXXXX)"
(
    while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            >> "${VRAM_LOG}" 2>/dev/null || true
        sleep 5
    done
) &
SAMPLER_PID=$!
trap 'kill "${SAMPLER_PID}" 2>/dev/null || true; rm -f "${VRAM_LOG}"' EXIT

# ── 跑通用 example（greedy、64 個新 token）─────────────────────────
"${PYTHON}" "${REPO_ROOT}/moe_infinity/examples/mixtral_example.py" \
    --checkpoint "${MODEL}" \
    --offload_dir "${OFFLOAD_DIR}" \
    --device_memory_ratio 0.75 \
    --max_new_tokens 64

# ── 驗收輸出 ────────────────────────────────────────────────────────
kill "${SAMPLER_PID}" 2>/dev/null || true
PEAK_MIB="$(sort -n "${VRAM_LOG}" | tail -1)"
echo
echo "[m0] peak VRAM (sampled every 5s): ${PEAK_MIB:-?} MiB"
echo "[m0] PASS — generation completed without crash."
echo "[m0] 人工確認：續寫文字必須合理（Mixtral 的失敗模式就是這裡輸出亂碼）"
