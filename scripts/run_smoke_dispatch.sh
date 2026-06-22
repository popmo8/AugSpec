#!/bin/bash
#SBATCH --job-name=smoke_disp
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/smoke_disp_%j.log
#SBATCH -e /work/morrisliu07/job_err/smoke_disp_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# B smoke: rebuild engine (SetTensorsDirect + DispatchMergedLocal binding) then
# run a short topM offload — merged draft now dispatches through the same
# MoEMLP::forward kernel as SpecMoE. Pass = no crash + acceptance sane (a wrong
# gate/up/down order would tank acceptance to ~0).
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
MOE_ROOT="${REPO_ROOT}/moe_infinity"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUTLASS_DIR="${HOME}/cutlass"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
echo "[smoke] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== rebuild engine (dispatch_merged_local) ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -6
rc=${PIPESTATUS[0]}; echo "[smoke] build rc=${rc}"
if [ "${rc}" -ne 0 ]; then echo "[smoke] BUILD FAILED"; exit "${rc}"; fi

cd "${REPO_ROOT}"
echo "======== run topM (dispatch_merged_local) ========"
.venv/bin/python -m aug_spec.cli run --config configs/smoke_dispatch_merged.yaml || echo "[smoke] RUN FAILED"

echo "======== RESULT ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
try:
    r = list(csv.DictReader(open("output/smoke_dispatch_merged/per_question_summary.csv")))
    print(f"  n={len(r)}  MAT={s.mean(float(x['mean_accept_length']) for x in r):.3f}  "
          f"AccRate={s.mean(float(x['acceptance_rate']) for x in r):.3f}  "
          f"TPS={s.mean(float(x['tokens_per_second']) for x in r):.3f}")
except FileNotFoundError:
    print("  (no summary — run crashed)")
PYEOF
echo "[smoke] done"
