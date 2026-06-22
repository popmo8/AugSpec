#!/bin/bash
#SBATCH --job-name=p1_ablation
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/p1_ablation_%j.log
#SBATCH -e /work/morrisliu07/job_err/p1_ablation_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# P1 ablation: merge_during_verify off vs on @ b=0.2. 驗 (1) MAT/AccRate bit-exact
# (correctness — 只時機不同), (2) on 的 TPS > off (0 re-fetch 的 win).
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
echo "[p1] node=$(hostname) job=${SLURM_JOB_ID:-local}"
cd "${REPO_ROOT}"
echo "======== OFF (after-verify refresh) ========"
.venv/bin/python -m aug_spec.cli run --config configs/p1_dv_off.yaml || echo "[p1] off FAILED"
echo "======== ON (during-verify) ========"
.venv/bin/python -m aug_spec.cli run --config configs/p1_dv_on.yaml || echo "[p1] on FAILED"

echo "======== compare ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
def agg(d):
    r = list(csv.DictReader(open(f"output/{d}/per_question_summary.csv")))
    return (len(r),
            s.mean(float(x["mean_accept_length"]) for x in r),
            s.mean(float(x["acceptance_rate"]) for x in r),
            s.mean(float(x["tokens_per_second"]) for x in r))
no = agg("p1_dv_off"); yes = agg("p1_dv_on")
print(f"  OFF  n={no[0]:2d}  MAT={no[1]:.4f}  AccRate={no[2]:.4f}  TPS={no[3]:.4f}")
print(f"  ON   n={yes[0]:2d}  MAT={yes[1]:.4f}  AccRate={yes[2]:.4f}  TPS={yes[3]:.4f}")
mat_ok = abs(no[1]-yes[1]) < 1e-6 and abs(no[2]-yes[2]) < 1e-6
print(f"  MAT/AccRate bit-exact: {'YES ✓' if mat_ok else 'NO ✗ (acceptance changed — bug!)'}")
print(f"  TPS speedup ON/OFF: {yes[3]/no[3]:.2f}×")
PYEOF
echo "[p1] done"
