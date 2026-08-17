#!/bin/bash
#SBATCH --job-name=rmov_smoke
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/rmov_smoke_%j.log
#SBATCH -e /work/morrisliu07/job_err/rmov_smoke_%j.err
#
# remove_overload_plan.md §4-1 + §4-3: rebuild the C++ engine after the
# overload-path removal, then smoke specmoe(ep2) + topm on the new build.
# Both smoke configs deliberately omit offload.no_overload — evict-on-full
# is the built-in behaviour now. Pass = both runs finish (no hang), specmoe
# kept-residency ~99% in the profile dump, and no overload_wait row exists.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
MOE_ROOT="${REPO_ROOT}/moe_infinity"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUTLASS_DIR="${HOME}/cutlass"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
echo "[rmov_smoke] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== build_ext --inplace (rebuild after overload removal) ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -40
rc=${PIPESTATUS[0]}
echo "[rmov_smoke] build_ext rc=${rc}"
if [ "${rc}" -ne 0 ]; then
    echo "[rmov_smoke] BUILD FAILED"
    exit "${rc}"
fi

echo "======== import smoke ========"
"${PY}" -c "import moe_infinity; print('[rmov_smoke] import OK from', moe_infinity.__file__)" || exit 1

cd "${REPO_ROOT}"
for cfg in configs/smoke_noov_specmoe.yaml configs/smoke_noov_topm.yaml; do
  name=$(basename "${cfg}" .yaml)
  echo "======== smoke ${name} ========"
  "${PY}" -m aug_spec.cli run --config "${cfg}" || echo "[rmov_smoke] ${name} RUN FAILED"
done

echo "======== VERDICT ========"
"${PY}" - <<'PYEOF'
import csv
for d in ("smoke_noov_specmoe", "smoke_noov_topm"):
    try:
        rows = list(csv.DictReader(open(f"output/{d}/overall_summary.csv")))
        ov = [r for r in rows if r["subtask"] == "overall"][0]
        print(f"  {d}: cycles={ov['total_cycles']} "
              f"MAT={ov['mean_accept_tokens']} TPS={ov['tokens_per_second']}")
    except FileNotFoundError:
        print(f"  {d}: NO OUTPUT (run failed?)")
PYEOF
echo "[rmov_smoke] done"
