#!/bin/bash
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/%x_%j.log
#SBATCH -e /work/morrisliu07/job_err/%x_%j.err
export HF_HOME=/work/morrisliu07/.cache/huggingface
export HF_HUB_OFFLINE=1
cd /work/morrisliu07/aug_spec
exec .venv/bin/python -u scripts/expert_sim_l2.py --model "$1"
