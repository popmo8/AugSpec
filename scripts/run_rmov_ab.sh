#!/bin/bash
#SBATCH --job-name=rmov_ab
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/rmov_ab_%j.log
#SBATCH -e /work/morrisliu07/job_err/rmov_ab_%j.err
#
# remove_overload_plan.md §4-4 guard A/B: rerun the q15 specmoe ep2 setup on
# the post-removal engine (config q15_specmoe_ep2_rmov, fresh output dir) and
# print it next to the job-254248 baseline. §0.5 equivalence argument says the
# surviving code path is identical, so expect a noise-level match
# (AccR ±~0.02, TPS ±~0.3, draft_fetch ~0.95TB).
# Submit with --dependency=afterok:<rmov_smoke job id> — needs the new build.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
cd "${REPO_ROOT}"
echo "[rmov_ab] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== q15_specmoe_ep2_rmov (post-removal engine) ========"
"${PY}" -m aug_spec.cli run --config configs/q15_specmoe_ep2_rmov.yaml || echo "[rmov_ab] RUN FAILED"

echo "======== A/B vs job 254248 baseline ========"
"${PY}" - <<'PYEOF'
import csv
def overall(d):
    try:
        rows = list(csv.DictReader(open(f"output/{d}/overall_summary.csv")))
        return [r for r in rows if r["subtask"] == "overall"][0]
    except FileNotFoundError:
        return None
base, new = overall("q15_specmoe_ep2"), overall("q15_specmoe_ep2_rmov")
for tag, ov in (("baseline(254248)", base), ("post-removal", new)):
    if ov is None:
        print(f"  {tag}: NO OUTPUT")
    else:
        print(f"  {tag}: cycles={ov['total_cycles']} MAT={ov['mean_accept_tokens']} "
              f"AccR={ov['acceptance_rate']} TPS={ov['tokens_per_second']}")
print("  (pass = noise-level match: AccR ±~0.02, TPS ±~0.3;"
      " draft_fetch ~0.95TB in the AUG_PROFILE dump above)")
PYEOF
echo "[rmov_ab] done"
