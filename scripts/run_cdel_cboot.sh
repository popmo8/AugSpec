#!/bin/bash
#SBATCH --job-name=cdel_cboot
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/cdel_cboot_%j.log
#SBATCH -e /work/morrisliu07/job_err/cdel_cboot_%j.err
#
# merged_cache_plan.md C-DEL + C-BOOT liveness 驗證(全部 qpc=1/mnt=64):
#   1. rebuild(EvictLayer 已刪)
#   2. smoke ×2(specmoe ep2 + topm,prefill_warmup 預設 ON)
#   3. cdel_tm_off(warmup off:舊 fallback + demand 驅逐能跑)
#   4. cboot_tm_on(warmup on:空首輪能跑、assert 零觸發、draft fetch 縮小)
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
echo "[cdel_cboot] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== rebuild (EvictLayer removed) ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -30
rc=${PIPESTATUS[0]}
echo "[cdel_cboot] build_ext rc=${rc}"
if [ "${rc}" -ne 0 ]; then
    echo "[cdel_cboot] BUILD FAILED"
    exit "${rc}"
fi
"${PY}" -c "import moe_infinity; print('[cdel_cboot] import OK')" || exit 1

cd "${REPO_ROOT}"
for cfg in configs/smoke_noov_specmoe.yaml configs/smoke_noov_topm.yaml \
           configs/cdel_tm_off.yaml configs/cboot_tm_on.yaml; do
  name=$(basename "${cfg}" .yaml)
  echo "======== ${name} ========"
  "${PY}" -m aug_spec.cli run --config "${cfg}" || echo "[cdel_cboot] ${name} RUN FAILED"
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
for tag, d in (("C-DEL only (warmup off)", "cdel_tm_off"),
               ("C-DEL + C-BOOT (warmup on)", "cboot_tm_on"),
               ("smoke specmoe", "smoke_noov_specmoe"),
               ("smoke topm", "smoke_noov_topm")):
    ov = overall(d)
    if ov is None:
        print(f"  {tag}: NO OUTPUT")
    else:
        print(f"  {tag}: cycles={ov['total_cycles']} MAT={ov['mean_accept_tokens']} "
              f"AccR={ov['acceptance_rate']} TPS={ov['tokens_per_second']}")
print("  pass = 四支全跑完、無 RuntimeError/hang;"
      "cboot 的 AUG_PROFILE draft fetch GB 相對 cdel 明顯縮小")
PYEOF
echo "[cdel_cboot] done"
