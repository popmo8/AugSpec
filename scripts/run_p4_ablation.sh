#!/bin/bash
#SBATCH --job-name=p4_overlap
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /work/morrisliu07/job_log/p4_overlap_%j.log
#SBATCH -e /work/morrisliu07/job_err/p4_overlap_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# P4 ablation: merge_overlap off (p1_dv_on) vs on (p4_overlap_on), both
# merge_during_verify=true @ b=0.2. 純 Python，不重編。驗 (1) 不 crash,
# (2) MAT/AccRate 同 (overlap 不改結果), (3) TPS↑ (拿掉 per-layer 全裝置 sync).
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
echo "[p4] node=$(hostname) job=${SLURM_JOB_ID:-local}"
cd "${REPO_ROOT}"
echo "======== OVERLAP OFF ========"
.venv/bin/python -m aug_spec.cli run --config configs/p1_dv_on.yaml 2>&1 | grep -E "\[vram\]|\[budget\]|peak|FATAL|Aborted|FAIL" || echo "[p4] off FAILED"
echo "======== OVERLAP ON ========"
.venv/bin/python -m aug_spec.cli run --config configs/p4_overlap_on.yaml 2>&1 | grep -E "\[vram\]|\[budget\]|peak|FATAL|Aborted|FAIL" || echo "[p4] on FAILED"
echo "======== compare ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
def agg(d):
    try: r=list(csv.DictReader(open(f"output/{d}/per_question_summary.csv")))
    except FileNotFoundError: return None
    return (len(r), s.mean(float(x["mean_accept_length"]) for x in r),
            s.mean(float(x["acceptance_rate"]) for x in r),
            s.mean(float(x["tokens_per_second"]) for x in r))
for lbl,d in [("OFF","p1_dv_on"),("ON ","p4_overlap_on")]:
    a=agg(d)
    print(f"  {lbl}  (crashed?)" if a is None else
          f"  {lbl}  n={a[0]:2d}  MAT={a[1]:.3f}  AccRate={a[2]:.3f}  TPS={a[3]:.3f}")
PYEOF
echo "[p4] done"
