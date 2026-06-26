#!/bin/bash
#SBATCH --job-name=topm_noov
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/topm_noov_%j.log
#SBATCH -e /work/morrisliu07/job_err/topm_noov_%j.err
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
echo "[tov] node=$(hostname)"
echo "======== topm WITHOUT no_overload ========"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_topmov_off.yaml || echo "[tov] off FAILED"
echo "======== topm WITH no_overload ========"
AUG_NO_OVERLOAD=1 .venv/bin/python -m aug_spec.cli run --config configs/cmp_topmov_on.yaml || echo "[tov] on FAILED"
echo "======== COMPARE ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s, re, glob
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
print(f"  {'':12s} {'n':>3s} {'MAT':>6s} {'AccR':>6s} {'TPS':>6s}")
for lbl,d in [("no_ov OFF","cmp_topmov_off"),("no_ov ON","cmp_topmov_on")]:
    a=agg(d); print(f"  {lbl:12s} "+("(none)" if a is None else f"{a[0]:3d} {a[1]:6.3f} {a[2]:6.3f} {a[3]:6.3f}"))
# profiling key rows from this log
log=open(glob.glob("/work/morrisliu07/job_log/topm_noov_*.log")[-1]).read()
b=log.split("[AUG_PROFILE] per-cycle breakdown")
def tot(blk,k):
    m=re.search(rf"\s+{k}\b.*?\(\s*([\d.]+) s total\)",blk); return float(m.group(1)) if m else 0.0
if len(b)>2:
    Coff,Con=cyc("cmp_topmov_off"),cyc("cmp_topmov_on")
    print(f"\n  {'ms/cyc':16s} {'OFF':>8s} {'ON':>8s}")
    for k in ["verify_fetch","overload_wait","expert_forward","merge","dispatch"]:
        nm={"dispatch":"draft_dispatch","merge":"merge(P3)"}.get(k,k)
        print(f"  {nm:16s} {tot(b[1],k)*1000/Coff:8.1f} {tot(b[2],k)*1000/Con:8.1f}")
PYEOF
echo "[tov] done"
