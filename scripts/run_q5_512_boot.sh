#!/bin/bash
#SBATCH --job-name=q5_512_boot
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/q5_512_boot_%j.log
#SBATCH -e /work/morrisliu07/job_err/q5_512_boot_%j.err
#
# C-BOOT 規模驗證(merged_cache_plan.md §2.4):topm qpc=5 / mnt=512 /
# skip mt_bench,prefill_warmup on vs off。AccR 差距 = C-BOOT 語意代價
# (預期 <1pp);TPS/draft_fetch 差距 = 空首輪收益。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
cd "${REPO_ROOT}"
echo "[q5_512_boot] node=$(hostname) job=${SLURM_JOB_ID:-local}"

for cfg in configs/q5_512_boot_on.yaml configs/q5_512_boot_off.yaml; do
  name=$(basename "${cfg}" .yaml)
  echo "======== ${name} ========"
  "${PY}" -m aug_spec.cli run --config "${cfg}" || echo "[q5_512_boot] ${name} RUN FAILED"
done

echo "======== VERDICT ========"
"${PY}" - <<'PYEOF'
import csv
def rows(d):
    try:
        return list(csv.DictReader(open(f"output/{d}/overall_summary.csv")))
    except FileNotFoundError:
        return None
for tag, d in (("warmup ON  (C-DEL+C-BOOT)", "q5_512_boot_on"),
               ("warmup OFF (C-DEL only)  ", "q5_512_boot_off")):
    rs = rows(d)
    if rs is None:
        print(f"  {tag}: NO OUTPUT")
        continue
    for r in rs:
        if r["subtask"] == "overall":
            print(f"  {tag}: cycles={r['total_cycles']} "
                  f"MAT={r['mean_accept_tokens']} AccR={r['acceptance_rate']} "
                  f"TPS={r['tokens_per_second']}")
print("  參考:歷史 q15 topm 家族 AccR ~0.5(uniform 係數、同軸);"
      "判準 = on/off AccR 差 <~1pp、on 的 TPS ≥ off、draft fetch on ≪ off")
PYEOF
echo "[q5_512_boot] done"
