#!/bin/bash
#SBATCH --job-name=q5_clab
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/q5_clab_%j.log
#SBATCH -e /work/morrisliu07/job_err/q5_clab_%j.err
#
# Partition A/B test: q5 (qpc=5, mnt=512), topm M=32 K=16, freq within-cluster
# weights (NOT uniform). ONLY the clustering partition changes, via a static
# per-layer label file (AUG_CLUSTER_LABELS). Compare acceptance to the live
# dynamic freq-slice baseline output/q5_512_tm_on.
#   sbatch --export=ALL,METHOD=cooccur scripts/run_q5_cluster_ab.sh
# METHOD in {cooccur, cooccur_bal, random, statfreq}.
set -uo pipefail
: "${METHOD:?set METHOD=cooccur|cooccur_bal|random|statfreq}"
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"

LABELS="cluster_labels/labels_${METHOD}_K16.json"
CFG="configs/q5_512_tm_${METHOD}.yaml"
[ -f "$LABELS" ] || { echo "[q5_clab] missing $LABELS"; exit 1; }
[ -f "$CFG" ]    || { echo "[q5_clab] missing $CFG"; exit 1; }

echo "[q5_clab:${METHOD}] node=$(hostname) job=${SLURM_JOB_ID:-local} labels=${LABELS}"
echo "======== partition=${METHOD}, q5, freq within-cluster ========"
AUG_NO_OVERLOAD=1 AUG_CLUSTER_LABELS="${LABELS}" \
    .venv/bin/python -m aug_spec.cli run --config "${CFG}" \
    || echo "[q5_clab:${METHOD}] RUN FAILED"

echo "======== COMPARE vs freq-slice baseline (q5_512_tm_on) ========"
.venv/bin/python - "$METHOD" <<'PYEOF'
import csv, sys, statistics as s
method=sys.argv[1]
def agg(d):
    try: r=list(csv.DictReader(open(f"output/{d}/per_question_summary.csv")))
    except FileNotFoundError: return None
    return (len(r), s.mean(float(x["mean_accept_length"]) for x in r),
            s.mean(float(x["acceptance_rate"]) for x in r),
            s.mean(float(x["tokens_per_second"]) for x in r))
print(f"  {'partition':16s} {'n':>3s} {'MAT':>6s} {'AccR':>6s} {'TPS':>6s}")
for lbl,d in [("freq_slice(base)","q5_512_tm_on"),(method,f"q5_512_tm_{method}")]:
    a=agg(d)
    print(f"  {lbl:16s} "+("(none)" if a is None else f"{a[0]:3d} {a[1]:6.3f} {a[2]:6.3f} {a[3]:6.3f}"))
PYEOF
echo "[q5_clab:${METHOD}] done"
