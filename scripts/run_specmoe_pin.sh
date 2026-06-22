#!/bin/bash
#SBATCH --job-name=specmoe_pin
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/specmoe_pin_%j.log
#SBATCH -e /work/morrisliu07/job_err/specmoe_pin_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# SpecMoE pin ablation: 重編 _store（含 SetPinned/ClearPinned/FindExpertEvict skip）
# + pin off vs on @ b=0.2. 驗 (1) 不 crash, (2) MAT/AccRate 同 (pin 純記憶體),
# (3) TPS: on > off (draft 讀 resident kept-N、0 PCIe).
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
echo "[smp] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== rebuild _store (pin) ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -6
rc=${PIPESTATUS[0]}; echo "[smp] build rc=${rc}"
if [ "${rc}" -ne 0 ]; then echo "[smp] BUILD FAILED"; exit "${rc}"; fi

cd "${REPO_ROOT}"
echo "======== PIN OFF (on-demand) ========"
.venv/bin/python -m aug_spec.cli run --config configs/specmoe_pin_off.yaml 2>&1 | grep -E "\[vram\]|\[budget\]|peak|FATAL|Aborted|FAIL" || echo "[smp] off FAILED"
echo "======== PIN ON (resident kept-N) ========"
.venv/bin/python -m aug_spec.cli run --config configs/specmoe_pin_on.yaml 2>&1 | grep -E "\[vram\]|\[budget\]|peak|FATAL|Aborted|FAIL" || echo "[smp] on FAILED"
echo "======== compare ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
def agg(d):
    try: r=list(csv.DictReader(open(f"output/{d}/per_question_summary.csv")))
    except FileNotFoundError: return None
    return (len(r), s.mean(float(x["mean_accept_length"]) for x in r),
            s.mean(float(x["acceptance_rate"]) for x in r),
            s.mean(float(x["tokens_per_second"]) for x in r))
for lbl,d in [("OFF","specmoe_pin_off"),("ON ","specmoe_pin_on")]:
    a=agg(d)
    print(f"  {lbl}  (crashed?)" if a is None else
          f"  {lbl}  n={a[0]:2d}  MAT={a[1]:.3f}  AccRate={a[2]:.3f}  TPS={a[3]:.3f}")
PYEOF
echo "[smp] done"
