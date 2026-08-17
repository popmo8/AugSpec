#!/bin/bash
#SBATCH --job-name=dsm_prefetch
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/%x_%j.log
#SBATCH -e /work/morrisliu07/job_err/%x_%j.err
#
# Prefetch the DeepSeek-MoE-16B weights into the shared HF cache (no
# login-node downloads — everything goes through SLURM). Resumes partial
# downloads; prints the key MoE config fields at the end as a sanity check.
set -euo pipefail
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
ml load miniconda3/24.11.1 2>/dev/null || true
cd /work/morrisliu07/aug_spec

.venv/bin/python -u - <<'EOF'
import json
from huggingface_hub import snapshot_download

for m in ["deepseek-ai/deepseek-moe-16b-base",
          "deepseek-ai/deepseek-moe-16b-chat"]:
    p = snapshot_download(m)
    print("DONE", m, "->", p, flush=True)
    cfg = json.load(open(f"{p}/config.json"))
    for k in ["model_type", "n_routed_experts", "num_experts_per_tok",
              "n_shared_experts", "first_k_dense_replace", "norm_topk_prob",
              "scoring_func", "num_hidden_layers", "moe_layer_freq"]:
        print(f"  {k} = {cfg.get(k)}", flush=True)
EOF
echo "[dsm_prefetch] all done"
