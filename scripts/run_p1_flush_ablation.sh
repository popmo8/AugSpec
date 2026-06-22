#!/bin/bash
#SBATCH --job-name=p1_flush
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/p1_flush_%j.log
#SBATCH -e /work/morrisliu07/job_err/p1_flush_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# P1 flush ablation: 重編 _store（含 FlushCache）+ off vs on @ b=0.2.
# 驗 (1) MAT/AccRate bit-exact (flush 不改 acceptance), (2) NVML peak 下降 (phase-
# exclusive — merged 與 archer cache 不再並存).
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
echo "[p1f] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== rebuild _store (FlushCache) ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -6
rc=${PIPESTATUS[0]}; echo "[p1f] build rc=${rc}"
if [ "${rc}" -ne 0 ]; then echo "[p1f] BUILD FAILED"; exit "${rc}"; fi

cd "${REPO_ROOT}"
echo "======== FLUSH OFF ========"
.venv/bin/python -m aug_spec.cli run --config configs/p1_flush_off.yaml 2>&1 | grep -E "\[vram\]|\[budget\]|peak|FAIL" || echo "[p1f] off FAILED"
echo "======== FLUSH ON ========"
.venv/bin/python -m aug_spec.cli run --config configs/p1_flush_on.yaml 2>&1 | grep -E "\[vram\]|\[budget\]|peak|FAIL" || echo "[p1f] on FAILED"

echo "======== compare ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
def agg(d):
    r = list(csv.DictReader(open(f"output/{d}/per_question_summary.csv")))
    return (len(r),
            s.mean(float(x["mean_accept_length"]) for x in r),
            s.mean(float(x["acceptance_rate"]) for x in r),
            s.mean(float(x["tokens_per_second"]) for x in r))
no = agg("p1_flush_off"); yes = agg("p1_flush_on")
print(f"  OFF  n={no[0]:2d}  MAT={no[1]:.4f}  AccRate={no[2]:.4f}  TPS={no[3]:.4f}")
print(f"  ON   n={yes[0]:2d}  MAT={yes[1]:.4f}  AccRate={yes[2]:.4f}  TPS={yes[3]:.4f}")
ok = abs(no[1]-yes[1]) < 1e-6 and abs(no[2]-yes[2]) < 1e-6
print(f"  MAT/AccRate bit-exact: {'YES ✓' if ok else 'NO ✗'}")
PYEOF
echo "[p1f] done"
