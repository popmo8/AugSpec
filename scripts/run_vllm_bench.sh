#!/bin/bash
#SBATCH --job-name=vllm_bench
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/%x_%j.log
#SBATCH -e /work/morrisliu07/job_err/%x_%j.err
#
# vLLM all-resident decode-throughput ceiling (bottleneck report control).
set -uo pipefail
export HF_HOME=/work/morrisliu07/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=WARNING
ml load cuda/12.6 2>/dev/null || true
cd /work/morrisliu07/aug_spec
CPU_OFFLOAD_GB="${1:-48}"     # 0 = all-resident ceiling; 48 ≈ moe_infinity 0.2x budget
echo "[vllm_bench] node=$(hostname) cpu_offload_gb=${CPU_OFFLOAD_GB}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
/work/morrisliu07/vllm_env/bin/python -u scripts/vllm_bench.py \
    --model Qwen/Qwen3-30B-A3B-Base --max-tokens 256 --batches 1,4,64 \
    --cpu-offload-gb "${CPU_OFFLOAD_GB}"
echo "[vllm_bench] done rc=$?"
