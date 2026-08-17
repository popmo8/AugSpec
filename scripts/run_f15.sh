#!/bin/bash
#SBATCH --job-name=f15
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/f15_%j.log
#SBATCH -e /work/morrisliu07/job_err/f15_%j.err
#
# f15 最終驗收通用 runner:sbatch scripts/run_f15.sh <config.yaml>
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
cfg="$1"
name=$(basename "${cfg}" .yaml)
echo "[f15] node=$(hostname) job=${SLURM_JOB_ID:-local} cfg=${name}"
cd "${REPO_ROOT}"
"${PY}" -m aug_spec.cli run --config "${cfg}" || echo "[f15] ${name} RUN FAILED"
echo "======== VERDICT ========"
"${PY}" - "$name" <<'PYEOF'
import csv, sys
d = sys.argv[1]
try:
    rows = list(csv.DictReader(open(f"output/{d}/overall_summary.csv")))
    ov = [r for r in rows if r["subtask"] == "overall"][0]
    print(f"  {d}: cycles={ov['total_cycles']} MAT={ov['mean_accept_tokens']} "
          f"AccR={ov['acceptance_rate']} TPS={ov['tokens_per_second']}")
except FileNotFoundError:
    print(f"  {d}: NO OUTPUT (run failed?)")
PYEOF
echo "[f15] done"
