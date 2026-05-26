#!/bin/bash
#SBATCH --job-name=smoke_p13
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/smoke_p13_%j.log
#SBATCH -e /work/morrisliu07/job_err/smoke_p13_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# Phase 1-3 smoke test: validate the offload backend end-to-end.
#
# Steps:
#   1) Import sanity     — catches Python-side syntax / signature errors
#                          before paying for any model load.
#   2) HF regression     — configs/_smoke.yaml. Must still pass; proves
#                          Phase 1-3 didn't break the existing GPU path.
#   3) Offload smoke     — configs/_smoke_offload.yaml. The actual
#                          Phase 3 acceptance test: MAT / AccRate within
#                          ±0.1 of the HF smoke (see config file for the
#                          reference numbers).
#
# Submit:
#   sbatch scripts/smoke_phase1to3.sh

set -euo pipefail

REPO_ROOT="/work/morrisliu07/aug_spec"

export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0

# shellcheck source=/dev/null
source "${REPO_ROOT}/.venv/bin/activate"

cd "${REPO_ROOT}"

echo "[smoke] node=$(hostname)  job=${SLURM_JOB_ID:-local}"
echo "[smoke] aug_spec at $(which aug_spec)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo
echo "============================================================"
echo "  Step 1/3: import check"
echo "============================================================"
python -u -c "
import aug_spec.cli
import aug_spec.runtime.loader
import aug_spec.runtime.specbench
import aug_spec.controller
import aug_spec.adapters.mixtral
import aug_spec.drafts.base
import aug_spec.drafts.count
import aug_spec.drafts.topm_count
import aug_spec.drafts.prefill_count
import aug_spec.drafts.uniform
print('all imports OK')
print('load_offload exists:', hasattr(aug_spec.runtime.loader, 'load_offload'))
from aug_spec.cli import RunConfig
print('RunConfig backend/offload fields:', [
    f.name for f in RunConfig.__dataclass_fields__.values()
    if 'offload' in f.name or 'backend' in f.name])
"

echo
echo "============================================================"
echo "  Step 2/3: HF regression — configs/_smoke.yaml"
echo "============================================================"
aug_spec run --config configs/_smoke.yaml

echo
echo "============================================================"
echo "  Step 3/3: offload smoke — configs/_smoke_offload.yaml"
echo "============================================================"
aug_spec run --config configs/_smoke_offload.yaml

echo
echo "============================================================"
echo "  Done. Inspect:"
echo "    output/_smoke/summary.json"
echo "    output/_smoke_offload/summary.json"
echo "============================================================"
