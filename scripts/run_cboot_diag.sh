#!/bin/bash
#SBATCH --job-name=cboot_diag
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/cboot_diag_%j.log
#SBATCH -e /work/morrisliu07/job_err/cboot_diag_%j.err
#
# C-BOOT AccR 崩潰診斷(merged_cache_plan.md §2.4):
#   1. cboot_diag(offload, warmup on, tokens.csv)→ token 移位分析
#   2. cboot_hf_on / cboot_hf_off(hf backend)→ 隔離引擎
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUG_PROFILE=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
cd "${REPO_ROOT}"
echo "[cboot_diag] node=$(hostname) job=${SLURM_JOB_ID:-local}"

for cfg in configs/cboot_diag.yaml configs/cboot_hf_on.yaml configs/cboot_hf_off.yaml; do
  name=$(basename "${cfg}" .yaml)
  echo "======== ${name} ========"
  "${PY}" -m aug_spec.cli run --config "${cfg}" || echo "[cboot_diag] ${name} RUN FAILED"
done

echo "======== TOKEN SHIFT ANALYSIS ========"
"${PY}" - <<'PYEOF'
import csv
import collections

rows = list(csv.DictReader(open("output/cboot_diag/tokens.csv")))
bycyc = collections.defaultdict(list)
for r in rows:
    bycyc[(r["question_id"], int(r["cycle_idx"]))].append(r)

n = same = sh_prev = sh_next = top2 = 0
for key in sorted(bycyc):
    rs = sorted(bycyc[key], key=lambda r: int(r["pos_idx"]))
    d = [int(r["draft_token_id"]) for r in rs]
    t = [int(r["real_token_id"]) for r in rs]
    t2 = [int(r["target_top2_id"]) for r in rs]
    for i in range(len(d)):
        n += 1
        same += d[i] == t[i]
        top2 += d[i] == t2[i]
        if i >= 1:
            sh_prev += d[i] == t[i - 1]
        if i + 1 < len(d):
            sh_next += d[i] == t[i + 1]
print(f"  positions n={n}")
print(f"  aligned   draft[i]==real[i]   : {same / max(1, n):.3f}")
print(f"  shift+1   draft[i]==real[i-1] : {sh_prev / max(1, n):.3f}")
print(f"  shift-1   draft[i]==real[i+1] : {sh_next / max(1, n):.3f}")
print(f"  top2      draft[i]==target#2  : {top2 / max(1, n):.3f}")

# 前 3 個 cycle 的原始序列(肉眼比對)
shown = 0
for key in sorted(bycyc):
    rs = sorted(bycyc[key], key=lambda r: int(r["pos_idx"]))
    print(f"  {key}: draft={[int(r['draft_token_id']) for r in rs]}")
    print(f"  {' ' * len(str(key))}  real ={[int(r['real_token_id']) for r in rs]}")
    shown += 1
    if shown >= 3:
        break
PYEOF

echo "======== VERDICT ========"
"${PY}" - <<'PYEOF'
import csv
def overall(d):
    try:
        rows = list(csv.DictReader(open(f"output/{d}/overall_summary.csv")))
        return [r for r in rows if r["subtask"] == "overall"][0]
    except FileNotFoundError:
        return None
for tag, d in (("offload warmup-on (diag)", "cboot_diag"),
               ("hf warmup-on", "cboot_hf_on"),
               ("hf warmup-off", "cboot_hf_off")):
    ov = overall(d)
    print(f"  {tag}: " + ("NO OUTPUT" if ov is None else
          f"MAT={ov['mean_accept_tokens']} AccR={ov['acceptance_rate']} "
          f"TPS={ov['tokens_per_second']}"))
print("  判別:hf on≈off → 引擎互動;hf on 也崩 → HF 迴圈同步遺漏;"
      "移位比率高 → 位置錯位")
PYEOF
echo "[cboot_diag] done"
