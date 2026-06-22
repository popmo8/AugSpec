#!/bin/bash
#SBATCH --job-name=bmm_sanity
#SBATCH --partition=normal2
#SBATCH --account=MST114471
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o /work/morrisliu07/job_log/bmm_sanity_%j.log
#SBATCH -e /work/morrisliu07/job_err/bmm_sanity_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hhliu@arbor.ee.ntu.edu.tw
#
# Fix A' sanity: batched torch.bmm over K stacked merged experts vs the
# per-cluster eager F.linear loop.
#   (1) 正確性 — bmm 加權和 == 迴圈加權和 (~bf16 噪聲)
#   (2) microbench — K∈{16,64}, T∈{1,512}, L=48: bmm vs loop wall time。
#       T=1 decode 應大贏；T=512 prefill 確認沒爆 (dense over-K 多 K/k× FLOPs)。
set -uo pipefail
REPO_ROOT="/work/morrisliu07/aug_spec"
export HF_HOME=/work/morrisliu07/.cache/huggingface
export PYTHONUNBUFFERED=1
ml load cuda/12.6 miniconda3/24.11.1 gcc/11.5.0 2>/dev/null || true
cd "${REPO_ROOT}"
echo "[bmm] node=$(hostname) job=${SLURM_JOB_ID:-local}"

.venv/bin/python - <<'PYEOF'
import time, torch, torch.nn.functional as F
dev = "cuda"; D, INTER = 2048, 768
torch.manual_seed(0)

def build(K):
    g = [torch.randn(INTER, D, device=dev, dtype=torch.bfloat16)*0.02 for _ in range(K)]
    u = [torch.randn(INTER, D, device=dev, dtype=torch.bfloat16)*0.02 for _ in range(K)]
    d = [torch.randn(D, INTER, device=dev, dtype=torch.bfloat16)*0.02 for _ in range(K)]
    return g, u, d

def loop(hs, g, u, d, weight):
    K = len(g); out = torch.zeros_like(hs)
    for ki in range(K):
        col = weight[:, ki]; mask = col > 0
        if not mask.any(): continue
        x = hs[mask]
        h = F.silu(F.linear(x, g[ki])) * F.linear(x, u[ki])
        out[mask] += F.linear(h, d[ki]) * col[mask].unsqueeze(-1)
    return out

def stack(g, u, d):
    gs = torch.stack(g).transpose(1,2).contiguous()   # [K,D,INTER]
    us = torch.stack(u).transpose(1,2).contiguous()   # [K,D,INTER]
    ds = torch.stack(d).transpose(1,2).contiguous()   # [K,INTER,D]
    return gs, us, ds

def bmm(hs, gs, us, ds, weight):
    K = gs.shape[0]; T = hs.shape[0]
    hsK = hs.unsqueeze(0).expand(K, T, -1)
    h = F.silu(torch.bmm(hsK, gs)) * torch.bmm(hsK, us)
    eo = torch.bmm(h, ds)
    return (eo * weight.t().unsqueeze(-1)).sum(dim=0)

def mkweight(T, K, k):
    sc = torch.rand(T, K, device=dev)
    top_s, top_i = sc.topk(k, dim=-1)
    top_s = top_s / top_s.sum(-1, keepdim=True)
    w = torch.zeros(T, K, device=dev, dtype=torch.bfloat16)
    w.scatter_(1, top_i, top_s.to(torch.bfloat16))
    return w

print("== correctness ==")
for K in (16, 64):
    g,u,d = build(K); gs,us,ds = stack(g,u,d)
    hs = torch.randn(4, D, device=dev, dtype=torch.bfloat16)
    w = mkweight(4, K, min(8,K))
    a = loop(hs,g,u,d,w).float(); b = bmm(hs,gs,us,ds,w).float()
    diff=(a-b).abs().max().item(); rel=diff/(a.abs().max().item()+1e-9)
    print(f"  K={K:2d}  maxdiff={diff:.4e}  rel={rel:.4e}")

print("\n== microbench (ms per draft-step, L=48) ==")
def run(fn, *a, iters=30):
    for _ in range(3): fn(*a)
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters):
        for _l in range(48): fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1e3
for K in (16, 64):
    g,u,d = build(K); gs,us,ds = stack(g,u,d)
    for T in (1, 512):
        hs = torch.randn(T, D, device=dev, dtype=torch.bfloat16)
        w = mkweight(T, K, min(8,K))
        tl = run(loop, hs, g, u, d, w)
        tb = run(bmm,  hs, gs, us, ds, w)
        tag = "FASTER" if tb<tl else "SLOWER"
        print(f"  K={K:2d} T={T:4d}  loop={tl:8.3f}  bmm={tb:8.3f}  bmm/loop={tb/tl:.3f}x ({tag})")
PYEOF
echo "[bmm] done"
