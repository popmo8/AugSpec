#!/bin/bash
#SBATCH --job-name=c3_val
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/c3_val_%j.log
#SBATCH -e /work/morrisliu07/job_err/c3_val_%j.err
#
# c3_pipeline_plan.md D0-D3 一次驗證:rebuild → binding(含 pipeline API)→
# D0 probe(c0_slots:預配 handle)→ D1 probe(c3_jobs:MergeJob 逐位元)→
# c1_smoke + c1_q5_512(pipeline 預設開,端到端 + drain KPI)。
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
echo "[c3_val] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== rebuild (merged-slot mechanism) ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -30
rc=${PIPESTATUS[0]}
echo "[c3_val] build_ext rc=${rc}"
if [ "${rc}" -ne 0 ]; then
    echo "[c3_val] BUILD FAILED"
    exit "${rc}"
fi

echo "======== binding check ========"
"${PY}" - <<'PYEOF'
import moe_infinity  # noqa: F401  (ensures package import side effects)
found = None
for modname in ("moe_infinity._engine", "moe_infinity._store"):
    import importlib
    try:
        mod = importlib.import_module(modname)
    except Exception:
        continue
    cls = getattr(mod, "expert_dispatcher", None)
    if cls is not None:
        found = (modname, cls)
        break
assert found, "expert_dispatcher binding not found in _engine/_store"
modname, cls = found
need = ["init_merged_slots", "merge_experts_to_slot", "get_merged_slot",
        "set_merged_slot_pinned", "submit_merge_jobs", "wait_merges_done"]
missing = [m for m in need if not hasattr(cls, m)]
assert not missing, f"missing bindings: {missing}"
print(f"[c3_val] binding OK on {modname}: {need}")
PYEOF

cd "${REPO_ROOT}"
echo "======== D0 probe (tests/offload/c0_slots.py) ========"
"${PY}" tests/offload/c0_slots.py || { echo "[c3_val] D0 PROBE FAILED"; exit 1; }

echo "======== D1 probe (tests/offload/c3_jobs.py) ========"
"${PY}" tests/offload/c3_jobs.py || { echo "[c3_val] D1 PROBE FAILED"; exit 1; }

for cfg in configs/c3_smoke.yaml configs/c3_q5_512.yaml; do
  name=$(basename "${cfg}" .yaml)
  echo "======== ${name} (pipeline on) ========"
  "${PY}" -m aug_spec.cli run --config "${cfg}" || echo "[c3_val] ${name} RUN FAILED"
done

echo "======== VERDICT ========"
"${PY}" - <<'PYEOF'
import csv
for d in ("c3_smoke", "c3_q5_512"):
    try:
        rows = list(csv.DictReader(open(f"output/{d}/overall_summary.csv")))
        ov = [r for r in rows if r["subtask"] == "overall"][0]
        print(f"  {d}: cycles={ov['total_cycles']} "
              f"MAT={ov['mean_accept_tokens']} AccR={ov['acceptance_rate']} "
              f"TPS={ov['tokens_per_second']}")
    except FileNotFoundError:
        print(f"  {d}: NO OUTPUT (run failed?)")
print("  驗收:probe 全過;mg-pipeline 行上報且 drain wait ≈ 0(KPI);"
      "AccR 與 B variant 同量級;TPS 方向 ↑;無 RuntimeError/fatal")
PYEOF
echo "[c3_val] done"
