#!/bin/bash
#SBATCH --job-name=base_topm
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/base_topm_%j.log
#SBATCH -e /work/morrisliu07/job_err/base_topm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# 四方對比（Base 模型）腳本 2/2 — top-M count M=32 K=16: hf + offload(cpp_merge).
# qpc=3, mnt=256.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
echo "[base_topm] node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cd "${REPO_ROOT}"
echo "======== #3 topm hf (Base) ========"
.venv/bin/python -m aug_spec.cli run --config configs/base_topm_hf.yaml || echo "[base_topm] hf FAILED"
echo "======== #4 topm offload cpp_merge (Base) ========"
.venv/bin/python -m aug_spec.cli run --config configs/base_topm_offload.yaml || echo "[base_topm] offload FAILED"
echo "[base_topm] done"
