"""serial_dispatch — the most-naive offload baseline: NO fetch/exec overlap.

The offloaded MoE block normally enqueues ALL of a layer's routed experts in one
``dispatch_local`` and the archer engine overlaps their fetch (H2D) with exec
across its fetch/exec threads. This patch replaces the block forward with a
version that dispatches ONE expert at a time — ``dispatch(e) → wait(e) →
dispatch(e+1) → …`` — so each expert's fetch sits fully on the critical path
with zero overlap (fetch↔exec and fetch↔fetch). Combined with ``moe_ondemand``
(no cache) this is textbook naive offload: fetch every routed expert, one at a
time, no cache, no pipeline (serial_dispatch_plan.md).

Correctness: ``SetInputs`` (start of each ``dispatch_local``) zeroes the C++
``final_hidden_states_`` accumulator and ``WaitHiddenStates`` returns that
dispatch's output only (expert_dispatcher.cpp SetInputs/WaitHiddenStates), so a
single-expert dispatch returns that expert's routing-weighted contribution
(0 for tokens not routed to it). Summing over the routed experts reproduces the
one-shot dispatch result exactly (same fp32 accumulation, split across calls).

Non-speculative only: the speculative path installs its own block forward via
the controller, which would silently shadow this patch — the CLI raises if
``serial_dispatch`` is combined with a speculative draft.
"""

from __future__ import annotations

import types

import os

import torch

# Periodic CUDA quiesce inside the serial dispatch loop. Serial mode issues one
# dispatch per routed expert, so a single model forward makes ~n_layers ×
# ~E dispatch/wait pairs (e.g. 48×128 ≈ 6144) between batch_spec's per-forward
# torch.cuda.synchronize — ~128× more archer operations for the stochastic
# fetch/exec-thread race to open than the normal path (48 dispatches/forward).
# That race wedged serial B=128 (job 271365, STALL at cycle 30). Syncing every
# AUG_SERIAL_SYNC_EVERY experts (+ once at block end) shrinks the window back to
# the normal path's scale. 0 disables (old behaviour, for A/B). Cheap: each
# wait_dispatch_local already blocked on that expert, so the sync only drains a
# near-idle stream.
_SERIAL_SYNC_EVERY = int(os.environ.get("AUG_SERIAL_SYNC_EVERY", "8"))


def _serial_moe_forward(self, hidden_states: torch.Tensor):
    bsz, seqlen, hid = hidden_states.shape
    hs = hidden_states.view(-1, hid)
    router_logits = self.gate(hs)                       # gate hook (precache) fires
    router_mask, rw_mask = self.lib.topk_softmax(router_logits)
    num_expert = router_mask.shape[-1]
    counts = router_mask.reshape(-1, num_expert).sum(dim=0)
    routed = (counts > 0).nonzero(as_tuple=False).flatten().tolist()
    out = torch.zeros((hs.shape[0], hid), dtype=torch.float32, device=hs.device)
    sync = _SERIAL_SYNC_EVERY > 0 and hs.is_cuda
    for i, e in enumerate(routed):
        # single-column mask/weights → dispatch_local enqueues ONLY expert e.
        mask_e = torch.zeros_like(router_mask)
        mask_e[:, e] = router_mask[:, e]
        rw_e = torch.zeros_like(rw_mask)
        rw_e[:, e] = rw_mask[:, e]
        self.expert_executor.dispatch_local(self.layer_id, hs, mask_e, rw_e)
        out = out + self.expert_executor.wait_dispatch_local()
        if sync and (i + 1) % _SERIAL_SYNC_EVERY == 0:
            torch.cuda.synchronize(hs.device)   # quiesce archer threads
    if sync:
        torch.cuda.synchronize(hs.device)       # drain at block boundary
    return out.view(bsz, seqlen, hid).to(hs.dtype), router_logits


def _is_offload_moe_block(m) -> bool:
    return (getattr(m, "expert_executor", None) is not None
            and getattr(m, "lib", None) is not None
            and getattr(m, "layer_id", None) is not None
            and hasattr(m, "gate") and hasattr(m, "experts"))


def install_serial_dispatch(model) -> int:
    """Replace every offloaded MoE block's forward with the serial (no-overlap)
    version. Returns the number of blocks patched. Idempotent."""
    n = 0
    for m in model.modules():
        if _is_offload_moe_block(m) and not getattr(m, "_serial_patched", False):
            m._serial_patched = True
            m.forward = types.MethodType(_serial_moe_forward, m)
            n += 1
    return n
