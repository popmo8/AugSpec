#!/bin/bash
#SBATCH --job-name=exp_late
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/exp_late_%j.log
#SBATCH -e /work/morrisliu07/job_err/exp_late_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# Early-pin variant 2 (算完才解 pin, AUG_EARLY_PIN=2). 純 Python,不重編(用
# 244275 build 的 .so). 跑完做 3-way 比較:off / on(Stage1) / late(variant2).
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
echo "[late] node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "======== variant 2: 算完才解 pin (AUG_EARLY_PIN=2) ========"
AUG_EARLY_PIN=2 .venv/bin/python -m aug_spec.cli run --config configs/exp_ep_late.yaml || echo "[late] FAILED"

echo "======== 3-WAY COMPARE (off / on=Stage1 / late=variant2) ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
def agg(d):
    try: r=list(csv.DictReader(open(f"output/{d}/per_question_summary.csv")))
    except FileNotFoundError: return None
    return (len(r), s.mean(float(x["mean_accept_length"]) for x in r),
            s.mean(float(x["acceptance_rate"]) for x in r),
            s.mean(float(x["tokens_per_second"]) for x in r))
print(f"  {'':22s} {'n':>3s} {'MAT':>6s} {'AccR':>6s} {'TPS':>6s}")
for lbl,d in [("off (refresh-pin)","exp_ep_off"),
              ("on  (Stage1 early)","exp_ep_on"),
              ("late(算完才解pin)","exp_ep_late")]:
    a=agg(d)
    print(f"  {lbl:22s} "+("(none)" if a is None else f"{a[0]:3d} {a[1]:6.3f} {a[2]:6.3f} {a[3]:6.3f}"))
print("\n  → 各 run 的 [AUG_PROFILE] 表看 draft_fetch / draft_dispatch(bmm) / overload_wait")
PYEOF
echo "[late] done"
