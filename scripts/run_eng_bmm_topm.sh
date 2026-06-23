#!/bin/bash
#SBATCH --job-name=engbmm_topm
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/engbmm_topm_%j.log
#SBATCH -e /work/morrisliu07/job_err/engbmm_topm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# v1 engine bmm (topm): rebuild (DispatchBmm) + run with AUG_MERGED_BACKEND=
# engine_bmm. Validate against the Python-bmm run (bmm_ctrl): MAT/AccRate match.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"; MOE_ROOT="${REPO_ROOT}/moe_infinity"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUTLASS_DIR="${HOME}/cutlass"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export AUG_MERGED_BACKEND=engine_bmm
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
echo "[engbmm] node=$(hostname) job=${SLURM_JOB_ID:-local} backend=${AUG_MERGED_BACKEND}"
echo "======== rebuild engine (DispatchBmm) ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -4
rc=${PIPESTATUS[0]}; echo "[engbmm] build rc=${rc}"
if [ "${rc}" -ne 0 ]; then echo "[engbmm] BUILD FAILED"; exit "${rc}"; fi
cd "${REPO_ROOT}"
echo "======== topM engine_bmm ========"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_engbmm_topm.yaml || echo "[engbmm] RUN FAILED"
echo "======== RESULT (vs bmm_ctrl Python-bmm) ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
def agg(d):
    try: r=list(csv.DictReader(open(f"output/{d}/per_question_summary.csv")))
    except FileNotFoundError: return None
    return (len(r), s.mean(float(x["mean_accept_length"]) for x in r),
            s.mean(float(x["acceptance_rate"]) for x in r),
            s.mean(float(x["tokens_per_second"]) for x in r))
for lbl,d in [("engine_bmm","cmp_engbmm_topm"),("python_bmm","cmp_topm_bmm_ctrl")]:
    a=agg(d)
    print(f"  {lbl:11s} (none)" if a is None else
          f"  {lbl:11s} n={a[0]:2d} MAT={a[1]:.3f} AccRate={a[2]:.3f} TPS={a[3]:.3f}")
PYEOF
echo "[engbmm] done"
