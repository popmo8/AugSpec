"""CLI entrypoint: `aug_spec run --config <path>`.

One YAML = one experiment. The same code path serves every (model, draft
strategy) combination — adding a new experiment means adding a YAML, not
a Python file.

Example YAML:

    model:
      id: mistralai/Mixtral-8x7B-v0.1
      dtype: bfloat16
      device_map: auto
      # adapter: mixtral                # optional; auto-detected if omitted

    draft:
      name: count   # uniform | count | pruned_count | topm_count | prefill* |
                    # softmax | random_mask | random_merge | specmoe |
                    # none (non-speculative baseline, e.g. MoE-Caching)
      args:
        count_top_k: 2                  # optional for count-based drafts
        record_history: true

    run:
      T: 3
      questions_per_cat: 10
      max_new_tokens: 512
      seed: 0
      emit_tokens_csv: false

    output:
      dir: output/mixtral_count         # optional; default: output/<config stem>
      label: mixtral_count

Outputs go to `output.dir`:
  per_question_summary.csv, overall_summary.csv, summary.json
  (+ expert_weights_history.json when the draft supports it & record_history=true)
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from aug_spec import __version__
from aug_spec.adapters import adapter_for_config, get_adapter
from aug_spec.adapters.base import apply_offload_settings
from aug_spec.clustering import get_cluster_method
from aug_spec.controller import Controller
from aug_spec.drafts import (
    ScoreBasedAvgDraft, SpecMoeDraft, get_draft, get_draft_class)
from aug_spec.runtime.loader import (
    compute_merged_bytes, compute_model_vram_bytes, compute_precache_pool_bytes,
    free_model, compute_expert_geometry, get_peak_vram_gb, load_model,
    load_offload,
)

from aug_spec.runtime.phase import shared_model_phase_patch, specbench_callbacks
from aug_spec.runtime.specbench import run_specbench


# =========================================================================
# Config dataclass
# =========================================================================

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _parse_qpc(v: Any) -> int:
    """run.questions_per_cat: int, or "all" → -1 (every Spec-Bench question;
    HumanEval capped at 80 in _sample_questions)."""
    if isinstance(v, str):
        if v.strip().lower() == "all":
            return -1
        raise ValueError(
            f"run.questions_per_cat must be an int or 'all', got {v!r}")
    n = int(v)
    if n <= 0:
        raise ValueError(f"run.questions_per_cat must be > 0 (or 'all'), got {n}")
    return n


@dataclass
class RunConfig:
    raw: Dict[str, Any]
    config_path: Path

    # model
    model_id: str
    dtype: torch.dtype
    device_map: Any
    trust_remote_code: bool
    adapter_name: Optional[str]
    backend: str                         # "hf" (default) | "offload"
    offload_path: Optional[str]          # offload backend: expert dir
    device_memory_ratio: float           # offload: archer pool / GPU (escape hatch)
    vram_budget_ratio: Optional[float]   # offload: usable VRAM / model VRAM (P0);
                                         # overrides device_memory_ratio when set
    vram_guard: bool                     # offload: per-cycle VRAM-over-budget warn
    serial_dispatch: bool                # offload: dispatch experts one-at-a-time
                                         # (no fetch/exec overlap) — naive baseline
    merge_offload: bool                  # offload: GPU resident-merge + opts
                                         # via archer dispatcher (M9b)
    merge_during_verify: bool            # offload-merge: per-layer merge during
                                         # verify (P3) vs after-verify refresh
    flush_on_draft_end: bool             # offload-merge: phase-exclusive flush
                                         # (archer@draft-start, merged@draft-end, P1)
    merge_overlap: bool                  # offload-merge: C3 merge-job
                                         # pipeline(同層 fetch∥forward∥
                                         # merge);false = P2 同步 ablation
    merged_backend: Optional[str]        # offload: merged-expert draft kernel
                                         # (was AUG_MERGED_BACKEND; None=default)
    early_pin: Optional[int]             # SpecMoE early-pin stage (was
                                         # AUG_EARLY_PIN; None=default)

    # draft
    draft_name: str
    draft_args: Dict[str, Any]

    # clustering / within-cluster merge (A3/A4)
    cluster_name: str                    # ClusterMethod registry key
    cluster_within_weight: str           # "freq" | "uniform" (was
                                         # AUG_CLUSTER_UNIFORM)
    cluster_args: Dict[str, Any]         # method-specific kwargs (e.g. metric,
                                         # cache, seed) — forwarded to the method

    # run
    T: int
    questions_per_cat: int               # -1 = YAML "all": every Spec-Bench
                                         # question + HumanEval capped at 80
    max_new_tokens: int
    seed: int
    warmup: bool
    prefill_warmup: bool                 # run.prefill_warmup (C-BOOT): empty
                                         # first candidate round → target pure
                                         # prefill builds the draft state before
                                         # the first real draft. Default TRUE;
                                         # false = paper ablation (old
                                         # draft-first + first-cycle fallback).
                                         # Distinct from run.warmup (compile
                                         # warm-up generate before timing).
    emit_tokens_csv: bool
    spec_bench_cache: Optional[Path]
    skip_categories: List[str]           # run.skip_categories (e.g. ["mt_bench"])
    humaneval: bool                      # run.humaneval: append HumanEval as
                                         # its own "humaneval" category (the
                                         # paper tables' Coding column)
    mt_bench_pooled: bool                # run.mt_bench_pooled: sample the 8
                                         # mt_bench sub-categories as ONE pool
                                         # (qpc questions total, not per subcat)
    batch_loop: bool                     # run.batch_loop (batch_spec_plan.md):
                                         # use the custom batched spec loop
                                         # instead of HF assisted decoding.
                                         # false = legacy path, untouched.
    batch_size: int                      # run.batch_size (batch_loop only)
    batch_by_category: bool              # run.batch_by_category: affinity
                                         # batching — one subtask per batch;
                                         # per-subtask TPS becomes defined.
                                         # DEFAULT TRUE (2026-07-16); set false
                                         # for legacy mixed-category batches.
    batch_fill_repeat: bool              # run.batch_fill_repeat: cycle-repeat a
                                         # short subtask group to batch_size
                                         # (B>80 support, batch_scale_plan §3)
    prompt_token_cap: int                # run.prompt_token_cap: >0 truncates
                                         # each prompt to its first N tokens
                                         # (input-length sweeps; 0 = off)
    debug_invariants: bool               # run.debug_invariants: assert the
                                         # ragged-batch invariants each cycle
                                         # (V3 validation; batch_loop only)

    # output
    output_dir: Path
    label: str

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        model_cfg = raw.get("model") or {}
        offload_cfg = model_cfg.get("offload") or {}
        draft_cfg = raw.get("draft") or {}
        run_cfg = raw.get("run") or {}
        out_cfg = raw.get("output") or {}
        cluster_cfg = raw.get("cluster") or {}

        if "id" not in model_cfg:
            raise ValueError("config: model.id is required")
        if "name" not in draft_cfg:
            raise ValueError("config: draft.name is required")

        within_weight = str(cluster_cfg.get("within_weight", "freq")).lower()
        if within_weight not in ("freq", "uniform"):
            raise ValueError(
                "config: cluster.within_weight must be 'freq' or 'uniform', "
                f"got {within_weight!r}")

        dtype_str = str(model_cfg.get("dtype", "bfloat16")).lower()
        if dtype_str not in _DTYPES:
            raise ValueError(
                f"config: unknown model.dtype={dtype_str!r}; "
                f"choose from {sorted(_DTYPES)}")

        # Output directory: explicit > derived from config stem.
        if out_cfg.get("dir"):
            output_dir = Path(out_cfg["dir"])
        else:
            output_dir = Path("output") / path.stem

        spec_cache = run_cfg.get("spec_bench_cache")
        spec_cache_path = Path(spec_cache) if spec_cache else None

        # remove_overload_plan.md (2026-07): the C++ overload path was deleted,
        # so pin-aware evict-on-full is now the only behaviour. The old knob is
        # accepted-but-ignored so pre-existing YAMLs keep loading.
        if "no_overload" in offload_cfg:
            print("[aug_spec] offload.no_overload is deprecated and ignored: "
                  "the overload path was removed; evict-on-full is always on.")

        return cls(
            raw=raw,
            config_path=path.resolve(),
            model_id=str(model_cfg["id"]),
            dtype=_DTYPES[dtype_str],
            device_map=model_cfg.get("device_map", "auto"),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            adapter_name=(str(model_cfg["adapter"])
                          if model_cfg.get("adapter") else None),
            backend=str(model_cfg.get("backend", "hf")).lower(),
            offload_path=(str(offload_cfg["path"])
                          if offload_cfg.get("path") else None),
            device_memory_ratio=float(
                offload_cfg.get("device_memory_ratio", 0.15)),
            vram_budget_ratio=(float(offload_cfg["vram_budget_ratio"])
                               if offload_cfg.get("vram_budget_ratio") is not None
                               else None),
            vram_guard=bool(offload_cfg.get("vram_guard", True)),
            serial_dispatch=bool(offload_cfg.get("serial_dispatch", False)),
            merge_offload=bool(
                offload_cfg.get("merge_offload",
                                offload_cfg.get("cpp_merge", False))),
            merge_during_verify=bool(
                offload_cfg.get("merge_during_verify", False)),
            flush_on_draft_end=bool(
                offload_cfg.get("flush_on_draft_end", False)),
            merge_overlap=bool(
                offload_cfg.get("merge_overlap", True)),   # D3: cache mode
                                                           # pipeline 預設開
            merged_backend=(str(offload_cfg["merged_backend"]).lower()
                            if offload_cfg.get("merged_backend") else None),
            early_pin=(int(draft_cfg["early_pin"])
                       if draft_cfg.get("early_pin") is not None else None),
            draft_name=str(draft_cfg["name"]),
            draft_args=dict(draft_cfg.get("args") or {}),
            cluster_name=str(cluster_cfg.get("name", "freq_slice")),
            cluster_within_weight=within_weight,
            cluster_args={k: v for k, v in cluster_cfg.items()
                          if k not in ("name", "within_weight")},
            T=int(run_cfg.get("T", 3)),
            questions_per_cat=_parse_qpc(run_cfg.get("questions_per_cat", 10)),
            max_new_tokens=int(run_cfg.get("max_new_tokens", 512)),
            seed=int(run_cfg.get("seed", 0)),
            warmup=bool(run_cfg.get("warmup", True)),
            prefill_warmup=bool(run_cfg.get("prefill_warmup", True)),
            emit_tokens_csv=bool(run_cfg.get("emit_tokens_csv", False)),
            spec_bench_cache=spec_cache_path,
            skip_categories=[str(c) for c in (run_cfg.get("skip_categories") or [])],
            humaneval=bool(run_cfg.get("humaneval", False)),
            mt_bench_pooled=bool(run_cfg.get("mt_bench_pooled", False)),
            batch_loop=bool(run_cfg.get("batch_loop", False)),
            batch_size=int(run_cfg.get("batch_size", 1)),
            batch_by_category=bool(run_cfg.get("batch_by_category", True)),
            batch_fill_repeat=bool(run_cfg.get("batch_fill_repeat", False)),
            prompt_token_cap=int(run_cfg.get("prompt_token_cap", 0)),
            debug_invariants=bool(run_cfg.get("debug_invariants", False)),
            output_dir=output_dir,
            label=str(out_cfg.get("label", path.stem)),
        )


# =========================================================================
# Profiling dump (AUG_PROFILE=1)
# =========================================================================

def _dump_offload_fetch_profile(model) -> None:
    """Offload fetch/evict/pinned-hit totals for the non-speculative baselines
    (moe_caching / moe_precache / moe_ondemand), AUG_PROFILE=1 only. The
    decisive cache-vs-no-cache numbers: high verify_fetch_bytes + high evict_n +
    low pinned-hit rate = experts really are re-fetched on demand. For
    moe_ondemand (no pins) pinned_hit_n must be 0 and evictions ≈ fetches
    (1:1 turnover) — the zero-cache accounting proof (moe_ondemand_plan §4 V2).
    The dispatcher is found from the model (works without a controller/manager)."""
    if os.environ.get("AUG_PROFILE") is None or model is None:
        return
    disp = None
    for m in model.modules():
        ex = getattr(m, "expert_executor", None)
        d = getattr(ex, "expert_dispatcher", None) if ex else None
        if d is not None:
            disp = d
            break
    if disp is None or not hasattr(disp, "dump_profile"):
        return
    p = disp.dump_profile()
    vf_n = p.get("verify_fetch_n", 0)
    vf_gb = p.get("verify_fetch_bytes", 0) / 1e9
    ph_n = p.get("pinned_hit_n", 0)
    ph_gb = p.get("pinned_hit_bytes", 0) / 1e9
    ev_n = p.get("evict_n", 0)
    fwd_n = p.get("forward_n", 0)            # total expert forwards = routings
    print("\n" + "=" * 70)
    print("  [AUG_PROFILE] offload fetch/evict/pin totals")
    print("=" * 70)
    print(f"  expert forwards    : {fwd_n:>12,}   (total routings)")
    print(f"  fetches (miss→H2D) : {vf_n:>12,}   {vf_gb:8.2f} GB")
    print(f"  pinned hits (0 PCIe): {ph_n:>12,}   {ph_gb:8.2f} GB")
    print(f"  evictions          : {ev_n:>12,}")
    if fwd_n:
        # A routing that did NOT fetch was served from residency. Pinned hits
        # are the pinned set; anything else resident is a CACHE hit — must be
        # ~0 for moe_ondemand (and is the pinned 10% only, for moe_precache).
        resident = max(0, fwd_n - vf_n)
        nonpin_hits = max(0, fwd_n - vf_n - ph_n)
        print(f"  overall hit rate   : {resident / fwd_n:.4f}  "
              f"(routings served with 0 fetch)")
        print(f"  CACHE hit rate     : {nonpin_hits / fwd_n:.4f}  "
              f"(non-pinned residency reuse — MUST be ~0 for no-cache)")
    if vf_n:
        # NOTE: evict_n counts only on-demand FindExpertEvict; the moe_ondemand
        # per-step flush_cache clears in BULK (not counted here), so this ratio
        # is < 1 under flushing. The decisive zero-cache proof is fetches ==
        # forwards (CACHE hit rate == 0), not this ratio.
        print(f"  evict/fetch ratio  : {ev_n / vf_n:.3f}  "
              f"(on-demand only; bulk flush not counted)")
    # Time attribution (thread-summed µs — can exceed wall; use ratios).
    print(f"  -- time (thread-summed) --")
    for lbl, k in (("fetch H2D", "verify_fetch_us"), ("expert exec", "forward_us"),
                   ("evict scan", "evict_us"), ("enqueue wait", "enqueue_wait_us"),
                   ("draft dispatch", "dispatch_us")):
        print(f"  {lbl:18s} : {p.get(k, 0) / 1e6:10.2f} s total")
    print(f"  → fetches == forwards + cache-hit == 0 → every routing refetched "
          f"(no cache).")
    print("=" * 70, flush=True)


def _dump_topm_stats(draft, out_dir) -> None:
    """Persist M-saturation stats (topm_stats.json) for the M/K sweeps:
    mean active experts per (layer, rebuild), fraction of rebuilds where M
    was saturated, mean experts dropped by the cutoff."""
    st = getattr(draft, "topm_stats", None)
    if not st or not st.get("calls"):
        return
    c = st["calls"]
    summary = dict(st, mean_active=st["sum_active"] / c,
                   mean_dropped=st["sum_dropped"] / c,
                   filled_frac=st["filled_calls"] / c)
    (Path(out_dir) / "topm_stats.json").write_text(
        json.dumps(summary, indent=2))
    print(f"  TopM     : E={st['n']} M={st['m']} "
          f"mean_active={summary['mean_active']:.1f} "
          f"M_filled={summary['filled_frac']:.1%} "
          f"mean_dropped={summary['mean_dropped']:.1f}")


def _dump_profile(controller) -> None:
    """Print the engine's per-cycle time breakdown (AUG_PROFILE=1 only). Times
    are µs; normalised by refresh cycles so topm/specmoe rows are comparable.
    Answers: where each cycle spends time, what overlaps, and SpecMoE's
    avoidable draft re-fetches (draft_fetch)."""
    if os.environ.get("AUG_PROFILE") is None:
        return
    disp = None
    for _, block in getattr(controller, "blocks", []):
        ex = getattr(block, "expert_executor", None)
        disp = getattr(ex, "expert_dispatcher", None) if ex else None
        if disp is not None:
            break
    if disp is None or not hasattr(disp, "dump_profile"):
        return
    p = disp.dump_profile()
    cyc = max(1, int(getattr(controller, "update_count", 0)))
    print("\n" + "=" * 70)
    print(f"  [AUG_PROFILE] per-cycle breakdown ({cyc} cycles)")
    print("=" * 70)
    def row(label, n_key, us_key):
        n, us = p.get(n_key, 0), p.get(us_key, 0)
        print(f"  {label:18s} {n/cyc:8.2f} /cyc   {us/cyc/1000:8.3f} ms/cyc"
              f"   ({us/1e6:7.2f} s total)")
    print(f"  {'step':18s} {'count':>8s}        {'time':>8s}")
    row("verify_fetch", "verify_fetch_n", "verify_fetch_us")
    row("draft_fetch", "draft_fetch_n", "draft_fetch_us")   # SpecMoE re-fetch
    row("evict", "evict_n", "evict_us")
    row("enqueue_wait", "enqueue_wait_n", "enqueue_wait_us")     # race-fix hits
    row("expert_forward", "forward_n", "forward_us")
    row("merge(P3)", "merge_n", "merge_us")   # pipeline 下為 merge 線程時間
    if p.get("mg_jobs_n", 0):
        print(f"  mg-pipeline: jobs {p['mg_jobs_n']} "
              f"(gated {p.get('mg_gated_n', 0)}), gate-wait "
              f"{p.get('mg_gate_wait_us', 0)/1e6:.2f} s, cold "
              f"{p.get('mg_cold_bytes', 0)/1e9:.2f} GB; drain "
              f"{p.get('drain_n', 0)}x wait "
              f"{p.get('drain_wait_us', 0)/1e6:.3f} s (KPI≈0)")
    row("draft_dispatch", "dispatch_n", "dispatch_us")
    gb = (p.get("verify_fetch_bytes", 0) + p.get("draft_fetch_bytes", 0)) / 1e9
    print(f"  fetched {gb:.2f} GB total "
          f"(verify {p.get('verify_fetch_bytes',0)/1e9:.2f} + "
          f"draft {p.get('draft_fetch_bytes',0)/1e9:.2f})")
    # merged_cache_plan.md §4.3 telemetry.
    ph_n = p.get("pinned_hit_n", 0)
    print(f"  singleton_verify_hit: {ph_n/cyc:.2f} /cyc, elided "
          f"{p.get('pinned_hit_bytes',0)/1e9:.2f} GB "
          f"(verify requests served by pinned residents)")
    mc = getattr(getattr(controller, "merge_engine", None),
                 "merged_cache", None)
    if mc is not None:
        tot = max(1, mc.hit_n + mc.miss_n)
        print(f"  merged_cache: hit {mc.hit_n} / miss {mc.miss_n} "
              f"(adopt rate {mc.hit_n/tot:.2f}), merge_elided "
              f"{mc.elided_bytes/1e9:.2f} GB, singleton resident/slot/hit "
              f"{mc.singleton_pinned_n}/{mc.singleton_slot_n}/"
              f"{mc.singleton_hit_n} "
              f"(pin-budget denied {mc.singleton_budget_denied_n}), "
              f"steals {mc.steal_n} (protected {mc.steal_protected_n}), "
              f"sgl-feas-denied {mc.sgl_feas_denied_n}")
    kc = getattr(getattr(controller, "draft", None), "kept_changed", None)
    if kc:
        print(f"  kept_changed: {sum(kc)/len(kc):.2f} experts/cycle "
              f"(vs draft_fetch {p.get('draft_fetch_n',0)/cyc:.2f}/cyc)")
    draft = getattr(controller, "draft", None)
    if draft is not None and getattr(draft, "_bmm_calls", 0) > 0:
        frac = draft._bmm_res_sum / max(1, draft._bmm_kept_sum)
        print(f"  kept_bmm: {draft._bmm_calls} calls, kept resident "
              f"{draft._bmm_res_sum}/{draft._bmm_kept_sum} = {frac*100:.0f}% "
              f"(need 100%/layer for bmm to engage)")
    print("=" * 70)


# =========================================================================
# Run one experiment
# =========================================================================

def run_experiment(cfg: RunConfig) -> Dict[str, Any]:
    """Execute one experiment end-to-end. Returns the summary dict that
    also gets written to `summary.json`."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  aug_spec   : {__version__}")
    print(f"  Config     : {cfg.config_path}")
    print(f"  Model      : {cfg.model_id}  ({cfg.dtype})")
    print(f"  Draft      : {cfg.draft_name}  args={cfg.draft_args}")
    print(f"  T          : {cfg.T}")
    _qpc_str = "all" if cfg.questions_per_cat == -1 else cfg.questions_per_cat
    print(f"  Spec-Bench : {_qpc_str} q/cat × "
          f"max_new_tokens={cfg.max_new_tokens}")
    print(f"  Output     : {cfg.output_dir}")
    print("=" * 70)

    # Non-speculative baselines (no draft, no controller, no forward swaps —
    # plain target-only generate):
    #   moe_caching  — the renamed "none": full budget as a dynamic archer
    #                  cache, no pinning. "none" kept as a back-compat alias.
    #   moe_precache — prefill-count → pin top-k% experts, no other cache
    #                  (runtime/precache.py). Also non-speculative.
    #   moe_ondemand — ZERO cache lower bound: no pins, pool = a minimal
    #                  streaming buffer only → every routed expert fetched on
    #                  demand and evicted (moe_ondemand_plan.md). Runs the same
    #                  path as moe_caching, just a tiny pool.
    draft_lower = cfg.draft_name.lower()
    NON_SPEC_DRAFTS = {"none", "moe_caching", "moe_precache", "moe_ondemand"}
    spec_mode = draft_lower not in NON_SPEC_DRAFTS
    is_precache = draft_lower == "moe_precache"
    is_ondemand = draft_lower == "moe_ondemand"

    # Apply YAML overrides for the offload knobs that used to be import-time
    # env reads (A4). Must run before any forward; env vars still override.
    apply_offload_settings(merged_backend=cfg.merged_backend,
                           early_pin=cfg.early_pin)

    # ── load model + adapter ───────────────────────────────────────────
    # `moe` is the moe_infinity wrapper on the offload backend (None on hf);
    # its `_configure_hook` must run before every generate — wired below as
    # run_specbench's `before_generate`.
    moe = None
    cpu_source = None
    usable_vram_bytes: Optional[int] = None    # VRAM budget audit/guard limit
    # C1 cache mode (merged_cache_plan.md) — resolved in the offload budget
    # block; False on hf / legacy / device_memory_ratio escape-hatch runs.
    cache_mode = False
    k_prime = 0
    expert_bytes_v = 0
    singleton_pin_budget = None
    if cfg.backend == "offload":
        if not cfg.offload_path:
            raise ValueError(
                "config: model.offload.path is required for backend=offload")
        # VRAM budget (P0, verify_merge_plan.md §0): vram_budget_ratio expresses
        # the usable VRAM as a fraction of the full-model footprint (GPU-indep,
        # matches thesis 0.2x). Derive the archer device_memory_ratio from it;
        # fall back to the raw device_memory_ratio escape hatch when unset.
        gpu_total = torch.cuda.get_device_properties(0).total_memory
        if is_precache:
            # moe_precache defines its OWN budget: the pool holds only the pinned
            # top-k% experts + a MINIMAL streaming buffer. The archer engine
            # streams experts through the pool, so non-pinned experts are fetched
            # on demand and evicted as the forward advances — NO cache.
            # (moe_ondemand does NOT come here: the archer prefetcher can't run a
            # tiny pool without evict-starvation — moe_ondemand_plan §10 — so it
            # uses moe_caching's pool below + a per-step flush instead.)
            pin_fraction = float(cfg.draft_args.get("pin_fraction", 0.10))
            _hr_override = os.environ.get("AUG_PRECACHE_HEADROOM_EXPERTS")
            (pool_bytes, precache_n_pin, precache_n_experts,
             _precache_ebytes) = compute_precache_pool_bytes(
                cfg.model_id, pin_fraction, cfg.batch_size, cfg.dtype,
                cfg.trust_remote_code,
                headroom_experts_override=(int(_hr_override)
                                           if _hr_override else None))
            usable_vram_bytes = pool_bytes
            device_memory_ratio = pool_bytes / gpu_total
            print(f"\n  [budget] moe_precache: pin {pin_fraction:.0%} = "
                  f"{precache_n_pin}/{precache_n_experts} experts/layer "
                  f"resident; archer pool {pool_bytes / 1e9:.2f}GB "
                  f"(pinned set + minimal streaming buffer, NO cache) "
                  f"→ device_memory_ratio={device_memory_ratio:.4f}")
        elif cfg.vram_budget_ratio is not None:
            model_vram = compute_model_vram_bytes(
                cfg.model_id, cfg.dtype, cfg.trust_remote_code)
            usable_vram_bytes = int(cfg.vram_budget_ratio * model_vram)
            wants_merged = (spec_mode and cfg.merge_offload and
                            get_draft_class(cfg.draft_name)
                            .holds_merged_residency)
            K = int(cfg.draft_args.get("K", 1))
            uniform_eff = (cfg.cluster_within_weight == "uniform"
                           or os.environ.get("AUG_CLUSTER_UNIFORM") is not None)
            # Cache mode (merged_cache_plan.md §2.6/§3): merged slots live
            # INSIDE the pool on the C0 ledger, so the pool gets the FULL
            # budget (auto mode, single ledger). Member-set content keys need
            # uniform coefficients; AUG_LEGACY_MERGE is the transition escape
            # hatch (old per-cycle rebuild + fixed reserve).
            cache_mode = (wants_merged and K > 1 and uniform_eff
                          and os.environ.get("AUG_LEGACY_MERGE") is None)
            if cache_mode:
                n_layers, expert_bytes_v = compute_expert_geometry(
                    cfg.model_id, cfg.dtype, cfg.trust_remote_code)
                # verify floor (§3): keep c×(per-layer max verify working set,
                # (T+1)×top-k distinct ≈ 48 experts) forever unpinnable.
                floor_bytes = 2 * 48 * expert_bytes_v
                k_prime = min(K, (usable_vram_bytes - floor_bytes)
                              // (n_layers * expert_bytes_v))
                if k_prime < 1:
                    raise ValueError(
                        f"budget too small for cache mode: floor "
                        f"{floor_bytes / 1e9:.1f}GB leaves no slot room in "
                        f"usable {usable_vram_bytes / 1e9:.1f}GB")
                # Single BUDGET, two arenas (2026-07-10): slots are torch
                # memory carved out of `usable` HERE; the archer pool gets the
                # rest. Slot bytes must never touch the archer ledger — ghost
                # debits there drained it and live-locked the fetch thread.
                slot_carve = k_prime * n_layers * expert_bytes_v
                pool_bytes = usable_vram_bytes - slot_carve
                # Resident-singleton pins are ordinary archer residents; cap
                # their total so the floor stays unpinnable (over-budget
                # singletons fall back to identity slots).
                singleton_pin_budget = pool_bytes - floor_bytes
                device_memory_ratio = pool_bytes / gpu_total
                print(f"\n  [budget] auto cache-mode: "
                      f"vram_budget_ratio={cfg.vram_budget_ratio} × "
                      f"model_vram={model_vram / 1e9:.1f}GB = usable "
                      f"{usable_vram_bytes / 1e9:.2f}GB; slot carve "
                      f"{slot_carve / 1e9:.2f}GB (S=K′={k_prime}"
                      + (f", K={K} capped" if k_prime < K else "")
                      + f") → archer pool {pool_bytes / 1e9:.2f}GB; "
                      f"verify floor {floor_bytes / 1e9:.2f}GB; singleton "
                      f"pin budget {singleton_pin_budget / 1e9:.2f}GB → "
                      f"device_memory_ratio={device_memory_ratio:.4f}")
            else:
                # Legacy: merged tensors live in the torch allocator OUTSIDE
                # the pool ledger, so the fixed reserve keeps pool+merged
                # within the budget (honest scarce-VRAM sim).
                merged_bytes = 0
                if wants_merged:
                    merged_bytes = compute_merged_bytes(
                        cfg.model_id, K, cfg.dtype, cfg.trust_remote_code)
                pool_bytes = usable_vram_bytes - merged_bytes
                if pool_bytes <= 0:
                    raise ValueError(
                        f"budget too small: merged reserve "
                        f"{merged_bytes / 1e9:.1f}GB ≥ usable "
                        f"{usable_vram_bytes / 1e9:.1f}GB (lower K or raise b)")
                device_memory_ratio = pool_bytes / gpu_total
                print(f"\n  [budget] vram_budget_ratio={cfg.vram_budget_ratio} × "
                      f"model_vram={model_vram / 1e9:.1f}GB = "
                      f"usable {usable_vram_bytes / 1e9:.2f}GB; "
                      f"reserve merged {merged_bytes / 1e9:.2f}GB → "
                      f"archer pool {pool_bytes / 1e9:.2f}GB → "
                      f"device_memory_ratio={device_memory_ratio:.4f}")
        else:
            device_memory_ratio = cfg.device_memory_ratio
            usable_vram_bytes = int(device_memory_ratio * gpu_total)
        print(f"\nLoading {cfg.model_id} via moe_infinity offload "
              f"(device_memory_ratio={device_memory_ratio:.4f}) ...")
        # cpu_source = host-resident weights for draft-side merging (M7).
        # Merge runs on CPU and ships one expert to GPU (offload-safe for any
        # merge draft); masked drafts (random_mask) never touch it. Skipped
        # entirely for draft:none — no draft means nothing ever reads it.
        model, tokenizer, moe, cpu_source = load_offload(
            cfg.model_id, cfg.offload_path,
            device_memory_ratio=device_memory_ratio,
            dtype=cfg.dtype, trust_remote_code=cfg.trust_remote_code,
            load_cpu_source=spec_mode,
        )
    else:
        print(f"\nLoading {cfg.model_id} (single copy; target == draft) ...")
        model, tokenizer = load_model(
            cfg.model_id,
            dtype=cfg.dtype,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
        )
    model.eval()

    # serial_dispatch (serial_dispatch_plan.md): naive no-overlap offload — one
    # expert per dispatch. Non-speculative only (the controller would shadow the
    # patched block forward). env AUG_SERIAL_DISPATCH overrides the YAML flag.
    serial_dispatch = (cfg.serial_dispatch
                       or os.environ.get("AUG_SERIAL_DISPATCH") is not None)
    if serial_dispatch:
        if cfg.backend != "offload":
            raise ValueError("serial_dispatch requires backend=offload")
        if spec_mode:
            raise ValueError(
                "serial_dispatch is non-speculative only (a speculative draft "
                "installs its own block forward, shadowing the patch); use it "
                "with moe_caching / moe_precache / moe_ondemand")
        from aug_spec.runtime.serial_dispatch import install_serial_dispatch
        n_ser = install_serial_dispatch(model)
        print(f"  Serial     : no-overlap dispatch installed on {n_ser} MoE "
              f"blocks (one expert per dispatch — naive offload)")

    draft = None
    controller = None
    precache = None
    ondemand = None
    on_cycle_extra = None
    draft_args = dict(cfg.draft_args)
    if spec_mode:
        if cfg.adapter_name is not None:
            adapter = get_adapter(cfg.adapter_name)
        else:
            adapter = adapter_for_config(model.config)
        adapter.post_load(model, tokenizer, _NamespaceFromDict(cfg.raw))
        print(f"  Adapter    : {adapter.name}")
    elif is_precache:
        # moe_precache reads routing via lightweight gate forward hooks and
        # pins hot experts — it runs the NATIVE offloaded forward, so no
        # post_load / controller.install (no forward swap). The adapter is used
        # only for iter_moe / num_experts / default_count_top_k.
        if cfg.adapter_name is not None:
            adapter = get_adapter(cfg.adapter_name)
        else:
            adapter = adapter_for_config(model.config)
        from aug_spec.runtime.precache import PrecacheManager
        precache = PrecacheManager(
            model, adapter,
            pin_fraction=float(cfg.draft_args.get("pin_fraction", 0.10)),
            auto_pin=not cfg.batch_loop)
        n_armed = precache.arm()
        print(f"  Adapter    : {adapter.name} (moe_precache, gate hooks only)")
        print(f"  Precache   : armed {n_armed} MoE layers, "
              f"pin_fraction={precache.pin_fraction:.0%}, "
              f"auto_pin={precache.auto_pin}")
    elif is_ondemand:
        # moe_ondemand: same path/pool as moe_caching, but a post-forward hook
        # flushes the expert cache after every step → cache disabled (§10 Path A).
        adapter = None
        from aug_spec.runtime.precache import OndemandFlusher
        ondemand = OndemandFlusher(model)
        ok = ondemand.arm()
        print(f"  Adapter    : (none — moe_ondemand, cache-disabled baseline)")
        print(f"  Ondemand   : per-step flush_cache hook {'armed' if ok else 'FAILED (no dispatcher)'}")
    else:
        adapter = None
        print("  Adapter    : (none — non-speculative baseline run)")
    print(f"  VRAM       : {get_peak_vram_gb():.2f} GB")

    if spec_mode:
        # ── resolve draft args (auto-fill from adapter per the draft's flags) ─
        draft_cls = get_draft_class(cfg.draft_name)
        if draft_cls.needs_count_top_k and "count_top_k" not in draft_args:
            draft_args["count_top_k"] = adapter.default_count_top_k(model)
        if draft_cls.needs_num_experts and "num_experts" not in draft_args:
            # Pick num_experts from the first MoE block in the model.
            first_block = next(iter(adapter.iter_moe(model)))[1]
            draft_args["num_experts"] = adapter.num_experts(first_block)
        if draft_cls.needs_layer_spec:
            # No spec/keep-set named → resolve the default spec for this
            # model (budget from keep_frac/num_keep, default the shared
            # 12.5%); if the file is missing, run the BO search in-run
            # (before controller.install, so it wraps clean forwards).
            num_keep = draft_args.pop("num_keep", None)
            keep_frac = draft_args.pop("keep_frac", None)
            if "spec_path" not in draft_args and "mlp_keep" not in draft_args:
                from aug_spec.runtime import dv_search
                n_moe = sum(1 for _ in adapter.iter_moe(model))
                k = dv_search.resolve_num_keep(n_moe, num_keep, keep_frac)
                spec_path = dv_search.default_spec_path(cfg.model_id, k)
                if not spec_path.exists():
                    print(f"  [dv] no spec at {spec_path} — running the "
                          "skip-layer BO search in-run (offline route: "
                          "scripts/run_search_dv.sh)")
                    dv_search.search_and_save(model, tokenizer, adapter, k,
                                              cfg.model_id, spec_path)
                draft_args["spec_path"] = str(spec_path)
            elif num_keep is not None or keep_frac is not None:
                raise ValueError(
                    "draft_verify: num_keep/keep_frac apply only to the "
                    "auto-resolved spec — drop them when spec_path or "
                    "mlp_keep is given")

        draft = get_draft(cfg.draft_name, **draft_args)
        # Inject the clustering / within-cluster weighting (A3/A4) for the
        # averaged-draft family; other drafts don't cluster. Set before
        # draft.prepare() so the cluster method's prepare hook runs.
        if isinstance(draft, ScoreBasedAvgDraft):
            draft.cluster_method = get_cluster_method(cfg.cluster_name,
                                                      **cfg.cluster_args)
            draft.within_weight = cfg.cluster_within_weight
            # Cache-mode adaptive K′ (merged_cache_plan.md §3): draft width is
            # capped by the slot budget after the verify floor.
            if cache_mode and 0 < k_prime < draft.K:
                print(f"  [budget] adaptive K′: draft.K {draft.K} → {k_prime}")
                draft.K = k_prime
        print(f"  Resolved   : draft={cfg.draft_name}{draft_args} "
              f"cluster={cfg.cluster_name} "
              f"within_weight={cfg.cluster_within_weight}")

        # ── run ───────────────────────────────────────────────────────
        controller = Controller(model, adapter, draft, cpu_source=cpu_source,
                                merge_offload=cfg.merge_offload,
                                merge_during_verify=cfg.merge_during_verify,
                                flush_on_draft_end=cfg.flush_on_draft_end,
                                merge_overlap=cfg.merge_overlap,
                                prefill_warmup=cfg.prefill_warmup,
                                cache_mode=cache_mode,
                                slots_per_layer=k_prime,
                                expert_bytes=expert_bytes_v,
                                singleton_pin_budget=singleton_pin_budget)

        # One-time, model-derived precomputation (e.g. SpecMoE distances).
        draft.prepare(adapter, controller.blocks)

        if isinstance(draft, ScoreBasedAvgDraft):
            on_cycle_extra = draft.make_on_cycle_tagger()  # None unless history

    t0 = time.perf_counter()
    if controller is not None:
        controller.install()
    try:
        callbacks = ({} if controller is None else
                     specbench_callbacks(controller,
                                         on_cycle_extra=on_cycle_extra))
        if moe is not None:
            callbacks["before_generate"] = moe._configure_hook
        # moe_precache B=1: per-question pin/count reset (auto-pin fires inside
        # the gate hooks at the first decode step).
        if precache is not None and not cfg.batch_loop:
            callbacks["on_question_start"] = precache.reset
        if cfg.batch_loop:
            # batch_spec_plan.md: custom batched loop — no HF assisted
            # decoding, no monkey-patching; phase flips + engine hooks are
            # driven inside the loop itself.
            from aug_spec.runtime.batch_spec import run_specbench_batched
            result = run_specbench_batched(
                model, tokenizer,
                controller if spec_mode else None,
                num_speculative=cfg.T,
                batch_size=cfg.batch_size,
                batch_by_category=cfg.batch_by_category,
                batch_fill_repeat=cfg.batch_fill_repeat,
                prompt_token_cap=cfg.prompt_token_cap,
                questions_per_cat=cfg.questions_per_cat,
                max_new_tokens=cfg.max_new_tokens,
                output_dir=cfg.output_dir,
                label=cfg.label,
                seed=cfg.seed,
                spec_bench_cache=cfg.spec_bench_cache,
                skip_categories=cfg.skip_categories,
                include_humaneval=cfg.humaneval,
                mt_bench_pooled=cfg.mt_bench_pooled,
                debug_invariants=cfg.debug_invariants,
                before_generate=(moe._configure_hook if moe is not None
                                 else None),
                precache=precache,
            )
        else:
            phase_ctx = (shared_model_phase_patch(controller)
                         if controller is not None
                         else contextlib.nullcontext())
            with phase_ctx:
                result = run_specbench(
                    target_model=model,
                    # SAME object — shared weights; None = non-spec run.
                    draft_model=model if spec_mode else None,
                    tokenizer=tokenizer,
                    num_speculative=cfg.T,
                    questions_per_cat=cfg.questions_per_cat,
                    max_new_tokens=cfg.max_new_tokens,
                    output_dir=cfg.output_dir,
                    label=cfg.label,
                    seed=cfg.seed,
                    spec_bench_cache=cfg.spec_bench_cache,
                    emit_tokens_csv=cfg.emit_tokens_csv,
                    warmup=cfg.warmup,
                    prefill_warmup=cfg.prefill_warmup if spec_mode else False,
                    on_prefill_warmup=(controller.update_masks
                                       if controller is not None else None),
                    vram_limit_bytes=usable_vram_bytes,
                    vram_guard=cfg.vram_guard,
                    skip_categories=cfg.skip_categories,
                    include_humaneval=cfg.humaneval,
                    mt_bench_pooled=cfg.mt_bench_pooled,
                    **callbacks,
                )
    finally:
        if controller is not None:
            controller.uninstall()
        if precache is not None:
            precache.disarm()
        if ondemand is not None:
            print(f"  Ondemand   : {ondemand.n_flushes} per-step cache flushes")
            ondemand.disarm()
    wall = time.perf_counter() - t0

    # ── write summary.json ────────────────────────────────────────────
    summary: Dict[str, Any] = {
        "config_path": str(cfg.config_path),
        "config": cfg.raw,
        "label": cfg.label,
        "model_id": cfg.model_id,
        "adapter": adapter.name if adapter is not None else None,
        "draft": {"name": cfg.draft_name, "args": draft_args},
        "T": cfg.T,
        "questions_per_cat": cfg.questions_per_cat,
        "max_new_tokens": cfg.max_new_tokens,
        "num_moe_layers": (controller.num_moe_layers
                           if controller is not None else None),
        "wall_time_s": wall,
        "n_cycles_total": (controller.update_count
                           if controller is not None else 0),
        "peak_vram_gb": get_peak_vram_gb(),
        "vram_budget_ratio": cfg.vram_budget_ratio,
        "usable_vram_gb": (usable_vram_bytes / 1e9
                           if usable_vram_bytes is not None else None),
        "overall": result.overall,
        "per_subtask": result.per_subtask,
    }

    # SpecMoE mask-miss telemetry (top-k target winners outside the old mask).
    if isinstance(draft, SpecMoeDraft):
        misses = draft.cycle_misses
        n_cycles = len(misses)
        mean_miss = (sum(misses) / n_cycles) if n_cycles else 0.0
        num_layers = controller.num_moe_layers
        max_miss = num_layers * draft.route_top_k
        summary["specmoe"] = {
            "N": draft.N,
            "route_top_k": draft.route_top_k,
            "count_top_k": draft.count_top_k,
            "n_miss_cycles": n_cycles,
            "mean_mask_miss_per_cycle": mean_miss,
            "mean_mask_miss_per_layer": (
                mean_miss / num_layers if num_layers else 0.0),
            "mask_miss_fraction": mean_miss / max_miss if max_miss else 0.0,
        }

    summary_path = cfg.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    # ── optionally write expert_weights_history.json ─────────────────
    if isinstance(draft, ScoreBasedAvgDraft) and draft.record_history:
        layer_indices = [li for li, _ in adapter.iter_moe(model)]
        history_payload = draft.export_history(metadata={
            "draft": cfg.draft_name,
            "draft_args": draft_args,
            "model_id": cfg.model_id,
            "adapter": adapter.name,
            "num_moe_layers": controller.num_moe_layers,
            "layer_indices": layer_indices,
            "T": cfg.T,
            "questions_per_cat": cfg.questions_per_cat,
            "max_new_tokens": cfg.max_new_tokens,
        })
        history_path = cfg.output_dir / "expert_weights_history.json"
        history_path.write_text(json.dumps(history_payload, ensure_ascii=False))
        print(f"  Expert weights history → {history_path} "
              f"({len(draft.history)} cycles)")

    # ── final stdout summary ──────────────────────────────────────────
    ov = result.overall
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  MAT      : {ov.get('mean_accept_tokens', 0):.3f}")
    print(f"  AccRate  : {ov.get('acceptance_rate', 0):.4f}")
    print(f"  TPS      : {ov.get('tokens_per_second', 0):.2f}")
    print(f"  Wall     : {wall:.2f} s, refresh cycles: "
          f"{controller.update_count if controller is not None else 0}")
    print(f"\n  Results saved → {summary_path}")

    _dump_profile(controller)
    _dump_offload_fetch_profile(model)
    _dump_topm_stats(draft, cfg.output_dir)

    # Release VRAM before returning (so callers can chain).
    free_model(model)
    if cpu_source is not None:
        free_model(cpu_source)
    gc.collect()

    return summary


class _NamespaceFromDict:
    """Lightweight stand-in for argparse.Namespace so adapter.post_load()
    can read free-form kwargs (e.g. GPT-OSS `reasoning_effort`) from
    `raw.model` / `raw.run` without us having to enumerate them here.
    """

    def __init__(self, raw: Dict[str, Any]):
        self._raw = raw

    def __getattr__(self, key: str) -> Any:
        for section in ("model", "draft", "run", "output"):
            block = self._raw.get(section) or {}
            if key in block:
                return block[key]
        return None


# =========================================================================
# Argparse plumbing
# =========================================================================

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aug_spec",
        description="Augmented speculative decoding for MoE inference.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one experiment from a YAML config.")
    run.add_argument("--config", "-c", type=Path, required=True,
                     help="Path to a YAML config (see configs/).")
    return p


def main(argv: Optional[list] = None) -> int:
    # AUG_HANG_DEBUG=<sec>: dump every thread's Python stack to stderr every
    # <sec> seconds — locates silent hangs (e.g. a pybind call that never
    # returns shows up as the last Python frame). Diagnostic-only env.
    if os.environ.get("AUG_HANG_DEBUG"):
        import faulthandler
        faulthandler.dump_traceback_later(
            int(os.environ["AUG_HANG_DEBUG"]), repeat=True, exit=False)
    parser = _make_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        if not args.config.exists():
            print(f"[aug_spec] config not found: {args.config}",
                  file=sys.stderr)
            return 2
        cfg = RunConfig.from_yaml(args.config)
        try:
            run_experiment(cfg)
        except BaseException:
            import traceback
            traceback.print_exc()
            if cfg.backend == "offload":
                # Same shutdown hang as the success path below — a CRASHED
                # offload run must also force-exit, or the exception turns
                # into a walltime burn (observed: jobs 258082/258090 sat at
                # a traceback until TIMEOUT).
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)
            raise
        if cfg.backend == "offload":
            # moe_infinity's C++ thread pool hangs on interpreter shutdown;
            # force-exit after outputs are written (same as examples/*.py).
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
