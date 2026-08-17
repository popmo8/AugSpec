#!/bin/bash
#SBATCH --job-name=prep_base_off
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/prep_base_off_%j.log
#SBATCH -e /work/morrisliu07/job_err/prep_base_off_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# 一次性把 Qwen3-30B-A3B-Base export 成 moe_infinity offload 格式（archer_index/
# archer_param_*/name_id_map.json）。兩個 offload 對比 job 共用、避免並行 export 競態。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
PY="${REPO_ROOT}/.venv/bin/python"
OFFDIR="${REPO_ROOT}/moe_infinity/offload_output/Qwen3-30B-A3B-Base"
echo "[prep] node=$(hostname) job=${SLURM_JOB_ID:-local}  target=${OFFDIR}"
cd "${REPO_ROOT}"

if [ -f "${OFFDIR}/name_id_map.json" ]; then
    echo "[prep] offload dir already exists — skip export"
    exit 0
fi

"${PY}" - <<'PYEOF'
from aug_spec.runtime.loader import load_offload
model_id = "Qwen/Qwen3-30B-A3B-Base"
offdir = "/work/morrisliu07/aug_spec/moe_infinity/offload_output/Qwen3-30B-A3B-Base"
# First MoE() call on a missing offload_path triggers the archer export.
model, tok, moe, _ = load_offload(
    model_id, offdir, device_memory_ratio=0.75, load_cpu_source=False)
import torch
ids = tok("Hello", return_tensors="pt").input_ids.to("cuda:0")
moe._configure_hook(ids)
with torch.no_grad():
    moe.generate(ids, max_new_tokens=4, do_sample=False,
                 pad_token_id=tok.eos_token_id)
print("[prep] export OK")
PYEOF
rc=$?
echo "[prep] rc=${rc}"
ls -la "${OFFDIR}" 2>/dev/null | head
exit "${rc}"
