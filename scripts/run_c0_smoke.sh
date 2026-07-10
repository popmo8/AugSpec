#!/bin/bash
#SBATCH --job-name=c0_smoke
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/c0_smoke_%j.log
#SBATCH -e /work/morrisliu07/job_err/c0_smoke_%j.err
#
# merged_cache_plan.md C0 驗證:rebuild(merged-slot 機制)→ binding 檢查 →
# smoke ×2。C0 是純機制(InitMergedSlots 沒人呼叫 → 零 slot),行為必須不變。
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
echo "[c0_smoke] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== rebuild (merged-slot mechanism) ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -30
rc=${PIPESTATUS[0]}
echo "[c0_smoke] build_ext rc=${rc}"
if [ "${rc}" -ne 0 ]; then
    echo "[c0_smoke] BUILD FAILED"
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
        "set_merged_slot_pinned", "discard_merged_slot"]
missing = [m for m in need if not hasattr(cls, m)]
assert not missing, f"missing bindings: {missing}"
print(f"[c0_smoke] binding OK on {modname}: {need}")
PYEOF

cd "${REPO_ROOT}"
echo "======== C0 functional probe (tests/offload/c0_slots.py) ========"
"${PY}" tests/offload/c0_slots.py || { echo "[c0_smoke] PROBE FAILED"; exit 1; }

for cfg in configs/smoke_noov_specmoe.yaml configs/smoke_noov_topm.yaml; do
  name=$(basename "${cfg}" .yaml)
  echo "======== ${name} ========"
  "${PY}" -m aug_spec.cli run --config "${cfg}" || echo "[c0_smoke] ${name} RUN FAILED"
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
print("  pass = build rc=0、binding OK、functional probe 全過、"
      "兩支 smoke 跑完(smoke 未 init slot → 行為不變)")
PYEOF
echo "[c0_smoke] done"
