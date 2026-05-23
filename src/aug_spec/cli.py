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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from aug_spec import __version__
from aug_spec.adapters import adapter_for_config, get_adapter
from aug_spec.controller import Controller
from aug_spec.drafts import ScoreBasedAvgDraft, get_draft
from aug_spec.runtime.loader import (
    free_model, get_peak_vram_gb, load_model,
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

    # draft
    draft_name: str
    draft_args: Dict[str, Any]

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
        draft_cfg = raw.get("draft") or {}
        run_cfg = raw.get("run") or {}
        out_cfg = raw.get("output") or {}

        if "id" not in model_cfg:
            raise ValueError("config: model.id is required")
        if "name" not in draft_cfg:
            raise ValueError("config: draft.name is required")

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
            draft_name=str(draft_cfg["name"]),
            draft_args=dict(draft_cfg.get("args") or {}),
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

    # ── load model + adapter ───────────────────────────────────────────
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

    # ── resolve draft args (auto-fill count_top_k from adapter if absent) ─
    draft_args = dict(cfg.draft_args)
    if cfg.draft_name in ("count", "pruned_count",
                          "topm_count",
                          "prefill_count", "prefill_topm_count") and \
            "count_top_k" not in draft_args:
        draft_args["count_top_k"] = adapter.default_count_top_k(model)
    if cfg.draft_name == "random_mask" and "num_experts" not in draft_args:
        # Pick num_experts from the first MoE block in the model.
        first_block = next(iter(adapter.iter_moe(model)))[1]
        draft_args["num_experts"] = adapter.num_experts(first_block)

    draft = get_draft(cfg.draft_name, **draft_args)
    print(f"  Resolved   : draft={cfg.draft_name}{draft_args}")

    # ── run ───────────────────────────────────────────────────────────
    controller = Controller(model, adapter, draft)

    on_cycle_extra = None
    if isinstance(draft, ScoreBasedAvgDraft):
        on_cycle_extra = draft.make_on_cycle_tagger()  # None if record_history=False

    t0 = time.perf_counter()
    controller.install()
    try:
        callbacks = specbench_callbacks(controller, on_cycle_extra=on_cycle_extra)
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
        "overall": result.overall,
        "per_subtask": result.per_subtask,
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

    # Release VRAM before returning (so callers can chain).
    free_model(model)
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
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
