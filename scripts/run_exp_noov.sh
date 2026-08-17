#!/bin/bash
#SBATCH --job-name=exp_noov
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/exp_noov_%j.log
#SBATCH -e /work/morrisliu07/job_err/exp_noov_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# (a) 殺 overload(batch>1 改走 FindExpertEvict,尊重 pin)+ early-pin.
# 對照之前 244275 的 off / on(early-pin, overload 還在). 期望:
#   draft_fetch→0, draft_dispatch(bmm)>0, overload_wait→0, TPS↑, MAT 不變.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"; MOE_ROOT="${REPO_ROOT}/moe_infinity"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUTLASS_DIR="${HOME}/cutlass"; export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
echo "[noov] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== rebuild (AUG_NO_OVERLOAD) ========"
cd "${MOE_ROOT}"; "${PY}" setup.py build_ext --inplace 2>&1 | tail -4
rc=${PIPESTATUS[0]}; echo "[noov] build rc=${rc}"; [ "${rc}" -ne 0 ] && { echo "[noov] BUILD FAILED"; exit "${rc}"; }
cd "${REPO_ROOT}"

echo "======== early-pin + NO-OVERLOAD ========"
AUG_EARLY_PIN=1 AUG_NO_OVERLOAD=1 .venv/bin/python -m aug_spec.cli run --config configs/exp_noov.yaml || echo "[noov] FAILED"

echo "======== COMPARE (off / early-pin / early-pin+no-overload) ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s, re
def agg(d):
    try: r=list(csv.DictReader(open(f"output/{d}/per_question_summary.csv")))
    except FileNotFoundError: return None
    return (len(r), s.mean(float(x["mean_accept_length"]) for x in r),
            s.mean(float(x["acceptance_rate"]) for x in r),
            s.mean(float(x["tokens_per_second"]) for x in r))
print(f"  {'':24s} {'n':>3s} {'MAT':>6s} {'AccR':>6s} {'TPS':>6s}")
for lbl,d in [("off","exp_ep_off"),("early-pin(overload on)","exp_ep_on"),
              ("early-pin + NO-overload","exp_noov")]:
    a=agg(d)
    print(f"  {lbl:24s} "+("(none)" if a is None else f"{a[0]:3d} {a[1]:6.3f} {a[2]:6.3f} {a[3]:6.3f}"))
print("\n  → exp_noov 的 [AUG_PROFILE] 表看 draft_fetch / draft_dispatch(bmm) / overload_wait")
PYEOF
echo "[noov] done"
