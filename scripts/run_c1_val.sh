#!/bin/bash
#SBATCH --job-name=c1_val
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/c1_val_%j.log
#SBATCH -e /work/morrisliu07/job_err/c1_val_%j.err
#
# merged_cache_plan.md C1 驗證:rebuild(singleton_verify_hit counter)→
# c1_smoke(uniform,cache mode 啟動)→ c1_q5_512(規模)。
# 看 AUG_PROFILE 的 merged_cache / singleton_verify_hit 行 + AccR。
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
echo "[c1_val] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== rebuild (singleton_verify_hit counter) ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -20
rc=${PIPESTATUS[0]}
echo "[c1_val] build_ext rc=${rc}"
if [ "${rc}" -ne 0 ]; then
    echo "[c1_val] BUILD FAILED"
    exit "${rc}"
fi

cd "${REPO_ROOT}"
for cfg in configs/c1_smoke.yaml configs/c1_q5_512.yaml; do
  name=$(basename "${cfg}" .yaml)
  echo "======== ${name} ========"
  "${PY}" -m aug_spec.cli run --config "${cfg}" || echo "[c1_val] ${name} RUN FAILED"
done

echo "======== VERDICT ========"
"${PY}" - <<'PYEOF'
import csv
def overall(d):
    try:
        rows = list(csv.DictReader(open(f"output/{d}/overall_summary.csv")))
        return [r for r in rows if r["subtask"] == "overall"][0]
    except FileNotFoundError:
        return None
for tag, d in (("c1_smoke (uniform, cache mode)", "c1_smoke"),
               ("c1_q5_512 (uniform, cache mode)", "c1_q5_512"),
               ("參考: q5_512_boot_on (freq, legacy)", "q5_512_boot_on")):
    ov = overall(d)
    print(f"  {tag}: " + ("NO OUTPUT" if ov is None else
          f"cycles={ov['total_cycles']} MAT={ov['mean_accept_tokens']} "
          f"AccR={ov['acceptance_rate']} TPS={ov['tokens_per_second']}"))
print("  驗收:AccR 與 uniform 參考(q15_freqslice r1-r3 ≈ 0.44-0.47)同量級;"
      "singleton_verify_hit>0;merged_cache adopt rate 上報;fallback≈0")
PYEOF
echo "[c1_val] done"
