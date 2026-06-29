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
      name: count                       # uniform | count | pruned_count | softmax | random_mask
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
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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
    compute_merged_bytes, compute_model_vram_bytes, free_model,
    get_peak_vram_gb, load_model, load_offload,
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
    merge_offload: bool                  # offload: GPU resident-merge + opts
                                         # via archer dispatcher (M9b)
    merge_during_verify: bool            # offload-merge: per-layer merge during
                                         # verify (P3) vs after-verify refresh
    flush_on_draft_end: bool             # offload-merge: phase-exclusive flush
                                         # (archer@draft-start, merged@draft-end, P1)
    merge_overlap: bool                  # offload-merge: merge on side stream,
                                         # overlap with next-layer fetch (P4)
    no_overload: bool                    # offload: C++ no-overload dispatch
                                         # (was AUG_NO_OVERLOAD; A4)
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

    # run
    T: int
    questions_per_cat: int
    max_new_tokens: int
    seed: int
    warmup: bool
    emit_tokens_csv: bool
    spec_bench_cache: Optional[Path]

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
            merge_offload=bool(
                offload_cfg.get("merge_offload",
                                offload_cfg.get("cpp_merge", False))),
            merge_during_verify=bool(
                offload_cfg.get("merge_during_verify", False)),
            flush_on_draft_end=bool(
                offload_cfg.get("flush_on_draft_end", False)),
            merge_overlap=bool(
                offload_cfg.get("merge_overlap", False)),
            no_overload=bool(offload_cfg.get("no_overload", False)),
            merged_backend=(str(offload_cfg["merged_backend"]).lower()
                            if offload_cfg.get("merged_backend") else None),
            early_pin=(int(draft_cfg["early_pin"])
                       if draft_cfg.get("early_pin") is not None else None),
            draft_name=str(draft_cfg["name"]),
            draft_args=dict(draft_cfg.get("args") or {}),
            cluster_name=str(cluster_cfg.get("name", "freq_slice")),
            cluster_within_weight=within_weight,
            T=int(run_cfg.get("T", 3)),
            questions_per_cat=int(run_cfg.get("questions_per_cat", 10)),
            max_new_tokens=int(run_cfg.get("max_new_tokens", 512)),
            seed=int(run_cfg.get("seed", 0)),
            warmup=bool(run_cfg.get("warmup", True)),
            emit_tokens_csv=bool(run_cfg.get("emit_tokens_csv", False)),
            spec_bench_cache=spec_cache_path,
            output_dir=output_dir,
            label=str(out_cfg.get("label", path.stem)),
        )


# =========================================================================
# Profiling dump (AUG_PROFILE=1)
# =========================================================================

def _dump_profile(controller) -> None:
    """Print the engine's per-cycle time breakdown (AUG_PROFILE=1 only). Times
    are µs; normalised by refresh cycles so topm/specmoe rows are comparable.
    Answers: where each cycle spends time, what serialises (overload_wait) vs
    overlaps, and SpecMoE's avoidable draft re-fetches (draft_fetch)."""
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
    row("evict_layer", "evict_layer_n", "evict_layer_us")
    row("overload_wait", "overload_wait_n", "overload_wait_us")  # H1 serialise
    row("enqueue_wait", "enqueue_wait_n", "enqueue_wait_us")     # race-fix hits
    row("expert_forward", "forward_n", "forward_us")
    row("merge(P3)", "merge_n", "merge_us")
    row("draft_dispatch", "dispatch_n", "dispatch_us")
    gb = (p.get("verify_fetch_bytes", 0) + p.get("draft_fetch_bytes", 0)) / 1e9
    print(f"  fetched {gb:.2f} GB total "
          f"(verify {p.get('verify_fetch_bytes',0)/1e9:.2f} + "
          f"draft {p.get('draft_fetch_bytes',0)/1e9:.2f})")
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
    print(f"  Spec-Bench : {cfg.questions_per_cat} q/cat × "
          f"max_new_tokens={cfg.max_new_tokens}")
    print(f"  Output     : {cfg.output_dir}")
    print("=" * 70)

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
    if cfg.backend == "offload":
        if not cfg.offload_path:
            raise ValueError(
                "config: model.offload.path is required for backend=offload")
        # VRAM budget (P0, verify_merge_plan.md §0): vram_budget_ratio expresses
        # the usable VRAM as a fraction of the full-model footprint (GPU-indep,
        # matches thesis 0.2x). Derive the archer device_memory_ratio from it;
        # fall back to the raw device_memory_ratio escape hatch when unset.
        gpu_total = torch.cuda.get_device_properties(0).total_memory
        if cfg.vram_budget_ratio is not None:
            model_vram = compute_model_vram_bytes(
                cfg.model_id, cfg.dtype, cfg.trust_remote_code)
            usable_vram_bytes = int(cfg.vram_budget_ratio * model_vram)
            # Reserve the fixed merged-expert residency out of the budget so the
            # archer pool shrinks to leave room — total (archer pool + merged)
            # stays within 0.2x, an honest scarce-VRAM sim (verify_merge_plan.md
            # P1/P2). Only merge drafts on the offload-merge engine hold merged.
            merged_bytes = 0
            if cfg.merge_offload and \
                    get_draft_class(cfg.draft_name).holds_merged_residency:
                K = int(cfg.draft_args.get("K", 1))
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
        # merge draft); masked drafts (random_mask) never touch it.
        model, tokenizer, moe, cpu_source = load_offload(
            cfg.model_id, cfg.offload_path,
            device_memory_ratio=device_memory_ratio,
            dtype=cfg.dtype, trust_remote_code=cfg.trust_remote_code,
            load_cpu_source=True,
            no_overload=cfg.no_overload,
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

    if cfg.adapter_name is not None:
        adapter = get_adapter(cfg.adapter_name)
    else:
        adapter = adapter_for_config(model.config)
    adapter.post_load(model, tokenizer, _NamespaceFromDict(cfg.raw))
    print(f"  Adapter    : {adapter.name}")
    print(f"  VRAM       : {get_peak_vram_gb():.2f} GB")

    # ── resolve draft args (auto-fill from adapter per the draft's flags) ─
    draft_args = dict(cfg.draft_args)
    draft_cls = get_draft_class(cfg.draft_name)
    if draft_cls.needs_count_top_k and "count_top_k" not in draft_args:
        draft_args["count_top_k"] = adapter.default_count_top_k(model)
    if draft_cls.needs_num_experts and "num_experts" not in draft_args:
        # Pick num_experts from the first MoE block in the model.
        first_block = next(iter(adapter.iter_moe(model)))[1]
        draft_args["num_experts"] = adapter.num_experts(first_block)

    draft = get_draft(cfg.draft_name, **draft_args)
    # Inject the clustering / within-cluster weighting (A3/A4) for the
    # averaged-draft family; other drafts don't cluster. Set before
    # draft.prepare() so the cluster method's prepare hook runs.
    if isinstance(draft, ScoreBasedAvgDraft):
        draft.cluster_method = get_cluster_method(cfg.cluster_name)
        draft.within_weight = cfg.cluster_within_weight
    print(f"  Resolved   : draft={cfg.draft_name}{draft_args} "
          f"cluster={cfg.cluster_name} within_weight={cfg.cluster_within_weight}")

    # ── run ───────────────────────────────────────────────────────────
    controller = Controller(model, adapter, draft, cpu_source=cpu_source,
                            merge_offload=cfg.merge_offload,
                            merge_during_verify=cfg.merge_during_verify,
                            flush_on_draft_end=cfg.flush_on_draft_end,
                            merge_overlap=cfg.merge_overlap)

    # One-time, model-derived precomputation (e.g. SpecMoE expert distances).
    draft.prepare(adapter, controller.blocks)

    on_cycle_extra = None
    if isinstance(draft, ScoreBasedAvgDraft):
        on_cycle_extra = draft.make_on_cycle_tagger()  # None if record_history=False

    t0 = time.perf_counter()
    controller.install()
    try:
        callbacks = specbench_callbacks(controller, on_cycle_extra=on_cycle_extra)
        if moe is not None:
            callbacks["before_generate"] = moe._configure_hook
        with shared_model_phase_patch(controller):
            result = run_specbench(
                target_model=model,
                draft_model=model,                # SAME object — shared weights
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
                vram_limit_bytes=usable_vram_bytes,
                vram_guard=cfg.vram_guard,
                **callbacks,
            )
    finally:
        controller.uninstall()
    wall = time.perf_counter() - t0

    # ── write summary.json ────────────────────────────────────────────
    summary: Dict[str, Any] = {
        "config_path": str(cfg.config_path),
        "config": cfg.raw,
        "label": cfg.label,
        "model_id": cfg.model_id,
        "adapter": adapter.name,
        "draft": {"name": cfg.draft_name, "args": draft_args},
        "T": cfg.T,
        "questions_per_cat": cfg.questions_per_cat,
        "max_new_tokens": cfg.max_new_tokens,
        "num_moe_layers": controller.num_moe_layers,
        "wall_time_s": wall,
        "n_cycles_total": controller.update_count,
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
    print(f"  Wall     : {wall:.2f} s, refresh cycles: {controller.update_count}")
    print(f"\n  Results saved → {summary_path}")

    _dump_profile(controller)

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
    parser = _make_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        if not args.config.exists():
            print(f"[aug_spec] config not found: {args.config}",
                  file=sys.stderr)
            return 2
        cfg = RunConfig.from_yaml(args.config)
        run_experiment(cfg)
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
