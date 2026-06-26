#!/bin/bash
#SBATCH --job-name=exp_earlypin
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/exp_earlypin_%j.log
#SBATCH -e /work/morrisliu07/job_err/exp_earlypin_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# Stage 1 (SpecMoE early-pin) before/after:同 config、AUG_PROFILE,只差
# AUG_EARLY_PIN。期望 after: draft_fetch→0, MAT 不變, 看 overload/TPS 淨效果.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"; MOE_ROOT="${REPO_ROOT}/moe_infinity"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUTLASS_DIR="${HOME}/cutlass"; export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
echo "[ep] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== rebuild (clean .so) ========"
cd "${MOE_ROOT}"; "${PY}" setup.py build_ext --inplace 2>&1 | tail -4
rc=${PIPESTATUS[0]}; echo "[ep] build rc=${rc}"; [ "${rc}" -ne 0 ] && { echo "[ep] BUILD FAILED"; exit "${rc}"; }
cd "${REPO_ROOT}"

echo "======== BEFORE (early-pin OFF) ========"
.venv/bin/python -m aug_spec.cli run --config configs/exp_ep_off.yaml || echo "[ep] off FAILED"
echo "======== AFTER (early-pin ON) ========"
AUG_EARLY_PIN=1 .venv/bin/python -m aug_spec.cli run --config configs/exp_ep_on.yaml || echo "[ep] on FAILED"

echo "======== COMPARE ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
def agg(d):
    try: r=list(csv.DictReader(open(f"output/{d}/per_question_summary.csv")))
    except FileNotFoundError: return None
    return (len(r), s.mean(float(x["mean_accept_length"]) for x in r),
            s.mean(float(x["acceptance_rate"]) for x in r),
            s.mean(float(x["tokens_per_second"]) for x in r))
print(f"  {'':14s} {'n':>3s} {'MAT':>6s} {'AccR':>6s} {'TPS':>6s}")
for lbl,d in [("early-pin OFF","exp_ep_off"),("early-pin ON","exp_ep_on")]:
    a=agg(d)
    print(f"  {lbl:14s} "+("(none)" if a is None else f"{a[0]:3d} {a[1]:6.3f} {a[2]:6.3f} {a[3]:6.3f}"))
print("\n  → 看各 run 的 [AUG_PROFILE] 表裡 draft_fetch / overload_wait 的變化")
PYEOF
echo "[ep] done"
