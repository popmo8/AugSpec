#!/bin/bash
#SBATCH --job-name=cmp_topm
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/cmp_topm_%j.log
#SBATCH -e /work/morrisliu07/job_err/cmp_topm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# 對照 #3 + #4: topM m32 k16, hf vs offload (qpc=5). 兩者 acceptance 應一致。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
echo "[cmp_topm] node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cd "${REPO_ROOT}"
echo "======== #3 hf topM m32 k16 ========"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_topm_m32k16_hf.yaml || echo "[cmp_topm] hf FAILED"
echo "======== #4 offload topM m32 k16 ========"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_topm_m32k16_offload.yaml || echo "[cmp_topm] offload FAILED"
echo "[cmp_topm] done"
