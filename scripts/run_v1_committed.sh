#!/bin/bash
#SBATCH --job-name=v1_bytes
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/%x_%j.log
#SBATCH -e /work/morrisliu07/job_err/%x_%j.err
# V1 byte-exact check (batch_spec_plan.md §6): rerun both V1 configs with
# AUG_DUMP_COMMITTED=1 and diff the committed token streams.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_DUMP_COMMITTED=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
rm -f output/v1_legacy/committed.jsonl output/v1_batch1/committed.jsonl
echo "[v1] legacy ..."
.venv/bin/python -u -m aug_spec.cli run --config configs/v1_legacy.yaml
echo "[v1] batch1 ..."
.venv/bin/python -u -m aug_spec.cli run --config configs/v1_batch1.yaml
echo "[v1] compare ..."
.venv/bin/python scripts/compare_committed.py \
    output/v1_legacy/committed.jsonl output/v1_batch1/committed.jsonl
rc=$?
echo "[v1] committed compare rc=${rc}"
exit ${rc}
