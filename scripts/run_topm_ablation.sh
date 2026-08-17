#!/bin/bash
#SBATCH --job-name=topm_abl
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/topm_abl_%j.log
#SBATCH -e /work/morrisliu07/job_err/topm_abl_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# topM ablation 3-way: 無 P2+P3 (after-verify refresh) vs P2+P3 (during-verify
# merge + EvictLayer) vs P2+P3+P4 (+ side-stream overlap). 都 offload @ b=0.2、
# merged-reserve 自動同預算。qpc=3, mnt=256。隔離各階段效果。
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
echo "[abl] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== rebuild _store (ensure latest) ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -4
rc=${PIPESTATUS[0]}; echo "[abl] build rc=${rc}"
if [ "${rc}" -ne 0 ]; then echo "[abl] BUILD FAILED"; exit "${rc}"; fi

cd "${REPO_ROOT}"
echo "======== 無 P2+P3 (after-verify refresh) ========"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_topm_noopt.yaml || echo "[abl] noopt FAILED"
echo "======== P2+P3 (during-verify + evict) ========"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_topm_opt.yaml || echo "[abl] opt FAILED"
echo "======== P2+P3+P4 (+ overlap) ========"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_topm_p4.yaml || echo "[abl] p4 FAILED"

echo "======== RESULT ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
def agg(d):
    try: r=list(csv.DictReader(open(f"output/{d}/per_question_summary.csv")))
    except FileNotFoundError: return None
    return (len(r), s.mean(float(x["mean_accept_length"]) for x in r),
            s.mean(float(x["acceptance_rate"]) for x in r),
            s.mean(float(x["tokens_per_second"]) for x in r))
print(f"  {'topM':16s} {'n':>3s} {'MAT':>7s} {'AccRate':>8s} {'TPS':>7s}")
res={}
for lbl,d in [("無 P2+P3","cmp_topm_noopt"),("P2+P3","cmp_topm_opt"),
              ("P2+P3+P4","cmp_topm_p4")]:
    a=agg(d); res[lbl]=a
    print(f"  {lbl:16s} {'?':>3s} (crashed)" if a is None else
          f"  {lbl:16s} {a[0]:3d} {a[1]:7.3f} {a[2]:8.3f} {a[3]:7.3f}")
base=res["無 P2+P3"]
if base:
    for lbl in ("P2+P3","P2+P3+P4"):
        a=res[lbl]
        if a: print(f"  {lbl} TPS vs 無: {a[3]/base[3]:.2f}x")
PYEOF
echo "[abl] done"
