#!/bin/bash
#SBATCH --job-name=fused_sanity
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/fused_sanity_%j.log
#SBATCH -e /work/morrisliu07/job_err/fused_sanity_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# Fix A sanity: moe_infinity._engine.expert_fused_mlp vs eager 3×F.linear
#   (1) 數值正確性 — 兩者輸出 max abs diff 應 ~bf16 噪聲
#   (2) microbench — 模擬 draft cluster loop (T=1, K=16, L=48) 的 wall time，
#       看 fused 是否真的贏 eager（kernel 內有 cublasCreate/Destroy + sync 的
#       per-call overhead，小 batch 可能反而慢）。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
echo "[fused] node=$(hostname) job=${SLURM_JOB_ID:-local}"

.venv/bin/python - <<'PYEOF'
import time, torch, torch.nn.functional as F
from moe_infinity import _engine

assert hasattr(_engine, "expert_fused_mlp"), "no expert_fused_mlp binding"
dev = "cuda"
D, INTER = 2048, 768          # Qwen3-30B-A3B hidden / moe_intermediate
torch.manual_seed(0)

def eager(hs, g, u, d):
    gate = F.linear(hs, g); up = F.linear(hs, u)
    return F.linear(F.silu(gate) * up, d)

def fused(hs, g, u, d):
    return _engine.expert_fused_mlp(hs.contiguous(), g, u, d)

# ── (1) correctness across a few token counts ──────────────────────────
print("== correctness (max abs diff vs eager) ==")
for T in (1, 4, 16):
    hs = torch.randn(T, D, device=dev, dtype=torch.bfloat16)
    g  = torch.randn(INTER, D, device=dev, dtype=torch.bfloat16) * 0.02
    u  = torch.randn(INTER, D, device=dev, dtype=torch.bfloat16) * 0.02
    d  = torch.randn(D, INTER, device=dev, dtype=torch.bfloat16) * 0.02
    a = eager(hs, g, u, d).float(); b = fused(hs, g, u, d).float()
    diff = (a - b).abs().max().item()
    rel  = diff / (a.abs().max().item() + 1e-9)
    print(f"  T={T:2d}  maxdiff={diff:.4e}  rel={rel:.4e}  shape_ok={tuple(b.shape)==tuple(a.shape)}")

# ── (2) microbench: simulate one draft step's cluster loops ─────────────
# K=16 clusters/layer, L=48 layers, T=1 token (worst case for per-call overhead).
K, L, T, ITERS = 16, 48, 1, 30
weights = []
for _ in range(K):
    weights.append((
        torch.randn(INTER, D, device=dev, dtype=torch.bfloat16) * 0.02,
        torch.randn(INTER, D, device=dev, dtype=torch.bfloat16) * 0.02,
        torch.randn(D, INTER, device=dev, dtype=torch.bfloat16) * 0.02))
hs = torch.randn(T, D, device=dev, dtype=torch.bfloat16)

def run(fn):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(ITERS):
        for _l in range(L):
            for (g, u, d) in weights:
                _ = fn(hs, g, u, d)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e3   # ms / draft-step

# warmup
for _ in range(3):
    for (g, u, d) in weights:
        eager(hs, g, u, d); fused(hs, g, u, d)
t_eager = run(eager)
t_fused = run(fused)
print("\n== microbench (ms per draft-step, K=16 × L=48 × T=1) ==")
print(f"  eager  = {t_eager:8.3f} ms")
print(f"  fused  = {t_fused:8.3f} ms")
print(f"  fused/eager = {t_fused/t_eager:.3f}x  ({'FASTER' if t_fused<t_eager else 'SLOWER'})")
PYEOF
echo "[fused] done"
