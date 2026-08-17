#!/bin/bash
#SBATCH --job-name=topm_bmm
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/topm_bmm_%j.log
#SBATCH -e /work/morrisliu07/job_err/topm_bmm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# topM merged draft via Python torch.bmm (AUG_MERGED_BACKEND=bmm), P1+P2+P3
# offload-merge @ b=0.2, qpc=5 mnt=512. The bmm A/B partner of the same-engine
# (dispatch_merged_local) run, both vs cmp_opt_specmoe.
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
MOE_ROOT="${REPO_ROOT}/moe_infinity"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUTLASS_DIR="${HOME}/cutlass"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export AUG_MERGED_BACKEND=bmm
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
echo "[bmm] node=$(hostname) job=${SLURM_JOB_ID:-local} backend=${AUG_MERGED_BACKEND}"

echo "======== rebuild engine ========"
cd "${MOE_ROOT}"
"${PY}" setup.py build_ext --inplace 2>&1 | tail -4
rc=${PIPESTATUS[0]}; echo "[bmm] build rc=${rc}"
if [ "${rc}" -ne 0 ]; then echo "[bmm] BUILD FAILED"; exit "${rc}"; fi

cd "${REPO_ROOT}"
echo "======== topM bmm (P1+P2+P3) ========"
.venv/bin/python -m aug_spec.cli run --config configs/cmp_topm_bmm.yaml || echo "[bmm] RUN FAILED"

echo "======== RESULT ========"
.venv/bin/python - <<'PYEOF'
import csv, statistics as s
try:
    r = list(csv.DictReader(open("output/cmp_topm_bmm/per_question_summary.csv")))
    print(f"  n={len(r)}  MAT={s.mean(float(x['mean_accept_length']) for x in r):.3f}  "
          f"AccRate={s.mean(float(x['acceptance_rate']) for x in r):.3f}  "
          f"TPS={s.mean(float(x['tokens_per_second']) for x in r):.3f}")
except FileNotFoundError:
    print("  (no summary — run crashed)")
PYEOF
echo "[bmm] done"
