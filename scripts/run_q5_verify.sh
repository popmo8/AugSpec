#!/bin/bash
#SBATCH --job-name=q5_verify
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/q5_verify_%j.log
#SBATCH -e /work/morrisliu07/job_err/q5_verify_%j.err
#
# Full A1-A5 verify at q5 (qpc=5). Deliberately does NOT export AUG_NO_OVERLOAD
# — no_overload comes from the YAML (tests A4's YAML->env path end-to-end).
# Compares aggregate AccR/MAT against the q5_512_tm_on baseline; per-question
# diffs are expected (pipeline is run-to-run non-deterministic, ~0.018 aggregate
# noise at 65 q), so the check is whether the aggregate lands in that band.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"

echo "[q5_verify] node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "======== A1-A5 verify, q5, no_overload from YAML ========"
.venv/bin/python -m aug_spec.cli run --config configs/q5_512_tm_verify.yaml \
    || echo "[q5_verify] RUN FAILED"

echo "======== COMPARE (aggregate) vs q5_512_tm_on baseline ========"
.venv/bin/python - <<'PYEOF'
import csv, os, statistics as s
def rows(d):
    p=f"output/{d}/per_question_summary.csv"
    return {(r["category"],r["question_id"]):r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}
new=rows("q5_512_tm_verify"); base=rows("q5_512_tm_on")
if not new:
    print("  MISSING q5_512_tm_verify output"); raise SystemExit
keys=sorted(set(new)&set(base))
na=s.mean(float(new[k]["acceptance_rate"]) for k in keys)
ba=s.mean(float(base[k]["acceptance_rate"]) for k in keys)
nm=s.mean(float(new[k]["mean_accept_length"]) for k in keys)
bm=s.mean(float(base[k]["mean_accept_length"]) for k in keys)
print(f"  matched questions: {len(keys)}")
print(f"  q5_verify (A1-A5): AccR={na:.4f}  MAT={nm:.3f}")
print(f"  q5 baseline (old): AccR={ba:.4f}  MAT={bm:.3f}")
print(f"  dAccR={na-ba:+.4f}  (aggregate run-to-run noise ~0.018 at 65 q)")
PYEOF
echo "[q5_verify] done"
