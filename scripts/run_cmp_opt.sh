#!/bin/bash
#SBATCH --job-name=cmp_opt
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/cmp_opt_%j.log
#SBATCH -e /work/morrisliu07/job_err/cmp_opt_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# 核心對比 — 兩邊都用各自的「最佳化方法」、都 offload @ b=0.2、qpc=5、mnt=512:
#   specmoe : pinned kept-N（忠實 SpecMoE，draft 讀 resident、0 PCIe）
#   topm    : P2+P3 during-verify merge + EvictLayer（0 re-fetch）+ merged-reserve(自動)
# 兩邊同預算（0.2x，~7.25GB reserve + ~2GB verify cache）。看 MAT/AccRate/TPS。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
MOE_ROOT="${REPO_ROOT}/moe_infinity"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUTLASS_DIR="${HOME}/cutlass"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
echo "[cmp] node=$(hostname) job=${SLURM_JOB_ID:-local}"

echo "======== rebuild _store (ensure latest: pin/evict/merge) ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -4
rc=${PIPESTATUS[0]}; echo "[cmp] build rc=${rc}"
if [ "${rc}" -ne 0 ]; then echo "[cmp] BUILD FAILED"; exit "${rc}"; fi

cd "${REPO_ROOT}"
echo "======== SpecMoE (pinned) ========"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_opt_specmoe.yaml || echo "[cmp] specmoe FAILED"
echo "======== topM (P2+P3) ========"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_opt_topm.yaml || echo "[cmp] topm FAILED"

echo "======== RESULT ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
def agg(d):
    try: r=list(csv.DictReader(open(f"output/{d}/per_question_summary.csv")))
    except FileNotFoundError: return None
    return (len(r), s.mean(float(x["mean_accept_length"]) for x in r),
            s.mean(float(x["acceptance_rate"]) for x in r),
            s.mean(float(x["tokens_per_second"]) for x in r))
print(f"  {'method':18s} {'n':>3s} {'MAT':>7s} {'AccRate':>8s} {'TPS':>7s}")
res={}
for lbl,d in [("SpecMoE(pinned)","cmp_opt_specmoe"),("topM(P2+P3)","cmp_opt_topm")]:
    a=agg(d); res[lbl]=a
    print(f"  {lbl:18s} {'?':>3s} (crashed)" if a is None else
          f"  {lbl:18s} {a[0]:3d} {a[1]:7.3f} {a[2]:8.3f} {a[3]:7.3f}")
if all(res.values()):
    sm, tm = res["SpecMoE(pinned)"], res["topM(P2+P3)"]
    print(f"\n  topM vs SpecMoE:  MAT {tm[1]/sm[1]:.2f}x  TPS {tm[3]/sm[3]:.2f}x")
PYEOF
echo "[cmp] done"
