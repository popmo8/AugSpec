#!/bin/bash
#SBATCH --job-name=q5_sm
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/q5_sm_%j.log
#SBATCH -e /work/morrisliu07/job_err/q5_sm_%j.err
#
# specmoe qpc=5, isolate no_overload: both runs early-pin, toggle AUG_NO_OVERLOAD.
# Confirms at higher q whether no_overload's MAT/TPS gain is real (not 13q noise).
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
echo "[q5sm] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== specmoe WITHOUT no_overload (early-pin, overload on) ========"
AUG_EARLY_PIN=1 .venv/bin/python -m aug_spec.cli run --config configs/q5_noov_sm_off.yaml || echo "[q5sm] off FAILED"

echo "======== specmoe WITH no_overload (early-pin + no_overload) ========"
AUG_EARLY_PIN=1 AUG_NO_OVERLOAD=1 .venv/bin/python -m aug_spec.cli run --config configs/q5_noov_sm_on.yaml || echo "[q5sm] on FAILED"

echo "======== COMPARE ========"
.venv/bin/python - "$0" <<'PYEOF'
import csv, statistics as s, re, glob, sys
def agg(d):
    try: r=list(csv.DictReader(open(f"output/{d}/per_question_summary.csv")))
    except FileNotFoundError: return None
    return (len(r), s.mean(float(x["mean_accept_length"]) for x in r),
            s.mean(float(x["acceptance_rate"]) for x in r),
            s.mean(float(x["tokens_per_second"]) for x in r))
def cyc(d):
    try:
        r=[x for x in csv.DictReader(open(f"output/{d}/overall_summary.csv")) if x['subtask']=='overall']; return int(r[0]['total_cycles'])
    except: return 1
print(f"  {'':12s} {'n':>3s} {'MAT':>6s} {'AccR':>6s} {'TPS':>6s} {'cyc':>5s}")
for lbl,d in [("no_ov OFF","q5_sm_off"),("no_ov ON","q5_sm_on")]:
    a=agg(d); c=cyc(d)
    print(f"  {lbl:12s} "+("(none)" if a is None else f"{a[0]:3d} {a[1]:6.3f} {a[2]:6.3f} {a[3]:6.3f} {c:5d}"))
# profiling: match the row whose FIRST token == label (no draft_dispatch regex bug)
log=open(glob.glob("/work/morrisliu07/job_log/q5_sm_*.log")[-1]).read()
b=log.split("[AUG_PROFILE] per-cycle breakdown")
def tot(blk,label):
    for line in blk.splitlines():
        m=re.match(r"\s+(\S+)\s+[\d.]+ /cyc.*?\(\s*([\d.]+) s total\)",line)
        if m and m.group(1)==label: return float(m.group(2))
    return 0.0
if len(b)>2:
    Coff,Con=cyc("q5_sm_off"),cyc("q5_sm_on")
    print(f"\n  {'ms/cyc':16s} {'OFF':>8s} {'ON':>8s}")
    for k in ["verify_fetch","draft_fetch","overload_wait","expert_forward","draft_dispatch","evict"]:
        print(f"  {k:16s} {tot(b[1],k)*1000/Coff:8.1f} {tot(b[2],k)*1000/Con:8.1f}")
PYEOF
echo "[q5sm] done"
