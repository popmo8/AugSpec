#!/bin/bash
#SBATCH --job-name=p2p3
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/p2p3_%j.log
#SBATCH -e /work/morrisliu07/job_err/p2p3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# P2+P3: 重編 _store（含 EvictLayer）+ merge_during_verify off vs on.
# on 現在 = 逐層 merge + 逐層 evict（防 overload → 不撞競態 + footprint 低）.
# 驗 (1) on 不再 crash, (2) NVML peak: on < off, (3) TPS.
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
echo "[p2p3] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== rebuild _store (EvictLayer) ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -6
rc=${PIPESTATUS[0]}; echo "[p2p3] build rc=${rc}"
if [ "${rc}" -ne 0 ]; then echo "[p2p3] BUILD FAILED"; exit "${rc}"; fi

cd "${REPO_ROOT}"
echo "======== OFF (after-verify refresh merge) ========"
.venv/bin/python -m aug_spec.cli run --config configs/p1_dv_off.yaml 2>&1 | grep -E "\[vram\]|\[budget\]|peak|FATAL|Aborted" || echo "[p2p3] off FAILED"
echo "======== ON (during-verify merge + per-layer evict) ========"
.venv/bin/python -m aug_spec.cli run --config configs/p1_dv_on.yaml 2>&1 | grep -E "\[vram\]|\[budget\]|peak|FATAL|Aborted" || echo "[p2p3] on FAILED"

echo "======== compare ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
def agg(d):
    try:
        r = list(csv.DictReader(open(f"output/{d}/per_question_summary.csv")))
    except FileNotFoundError:
        return None
    return (len(r),
            s.mean(float(x["mean_accept_length"]) for x in r),
            s.mean(float(x["acceptance_rate"]) for x in r),
            s.mean(float(x["tokens_per_second"]) for x in r))
for lbl, d in [("OFF","p1_dv_off"), ("ON ","p1_dv_on")]:
    a = agg(d)
    if a is None: print(f"  {lbl}  (no csv — crashed?)")
    else: print(f"  {lbl}  n={a[0]:2d}  MAT={a[1]:.3f}  AccRate={a[2]:.3f}  TPS={a[3]:.3f}")
PYEOF
echo "[p2p3] done"
