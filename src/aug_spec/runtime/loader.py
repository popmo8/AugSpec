"""Model loading + memory utilities.

`load_model` returns a HuggingFace causal LM ready for either standard or
speculative decoding. The compat shims here patch older trust_remote_code
models (e.g. DeepSeek-MoE) that call DynamicCache APIs removed in
transformers >= 4.44.

This module is intentionally backend-neutral: it does plain HF loading.
When the offload backend lands, a sibling `load_offload(...)` will live
beside this and use `moe_infinity.MoE(...)`; adapter / draft / controller
code does not need to know which loader was used.
"""

from __future__ import annotations

import gc
import inspect
import math
import types
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.cache_utils import Cache


# ---------------------------------------------------------------------------
# Compatibility shims for older trust_remote_code models.
# ---------------------------------------------------------------------------
if not hasattr(DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length=0, layer_idx=0):  # type: ignore[no-redef]
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = _get_usable_length

if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

if not hasattr(DynamicCache, "get_max_length"):
    def _get_max_length(self):  # type: ignore[no-redef]
        try:
            return self.max_cache_len
        except (ValueError, AttributeError):
            return None
    DynamicCache.get_max_length = _get_max_length


def _fix_prepare_inputs_for_generation(model: nn.Module) -> None:
    """Patch DeepSeek-MoE-style models whose generated
    `prepare_inputs_for_generation` references an undefined `seq_length`."""
    fn = getattr(model, "prepare_inputs_for_generation", None)
    if fn is None:
        return
    try:
        src = inspect.getsource(fn)
    except (TypeError, OSError):
        return
    if "get_usable_length(seq_length)" not in src:
        return

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_usable_length(
                    input_ids.shape[1])
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            if (attention_mask is not None
                    and attention_mask.shape[1] > input_ids.shape[1]):
                input_ids = input_ids[
                    :, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            if (max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs

    model.prepare_inputs_for_generation = types.MethodType(
        prepare_inputs_for_generation, model)
    print("    [compat] Patched prepare_inputs_for_generation")


def load_model(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
    trust_remote_code: bool = True,
    device_map: Optional[Any] = None,
) -> Tuple[nn.Module, Any]:
    """Load a HuggingFace causal LM + matching tokenizer.

    `device_map` takes precedence over `device`. Passing neither leaves
    placement to the model's defaults.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Canonical kwarg name is `torch_dtype` — accepted across all
    # transformers releases. Newer versions also accept `dtype` but
    # then forward it to the model __init__, which most architectures
    # reject. Stay with the safe name.
    kwargs: dict = dict(torch_dtype=dtype, trust_remote_code=trust_remote_code)
    if device_map is not None:
        kwargs["device_map"] = device_map
    elif device is not None:
        kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if trust_remote_code:
        _fix_prepare_inputs_for_generation(model)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Offload backend (moe_infinity). Additive — the hf path above is untouched.
# ---------------------------------------------------------------------------

def _patch_offload_device(model: nn.Module,
                          device: torch.device) -> None:
    """Pin `model.device` to `device`.

    moe_infinity leaves every parameter as a shape-(1,) CPU placeholder,
    materialising the real weight per-forward via module hooks. The default
    `PreTrainedModel.device` reads it off the first parameter and therefore
    reports `cpu`. HF assisted generation uses `assistant_model.device` to
    move `input_ids`, so without this patch speculative decoding moves the
    ids to CPU and crashes mid-forward (M2 finding, offload_plan.md §2).

    Patches the class (not the instance) because `device` is a read-only
    property; single-model / single-GPU is assumed (offload_plan.md §2.6).
    """
    cls = type(model)
    cls.device = property(lambda self: device)


def load_offload(
    model_id: str,
    offload_path: str,
    device_memory_ratio: float = 0.15,
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    device: Optional[torch.device] = None,
    load_cpu_source: bool = True,
) -> Tuple[nn.Module, Any, Any, Optional[nn.Module]]:
    """Load a MoE model with moe_infinity expert offloading.

    Sibling of `load_model` for the offload backend. Non-expert layers stay
    on GPU; experts live on host RAM and stream in on demand within a VRAM
    budget of `device_memory_ratio × GPU_size` (offload_plan.md §1).

    Returns `(model, tokenizer, moe, cpu_source)`:
      * `model`      — `moe.model`, the offloaded `PreTrainedModel`; target
                       verify runs through this.
      * `tokenizer`
      * `moe`        — the `moe_infinity.MoE` wrapper. **Keep it**: every
                       `generate(...)` must be preceded by
                       `moe._configure_hook(input_ids)` to (re)create the
                       expert-tracer sequence entries (offload_plan.md §2).
      * `cpu_source` — a CPU-resident copy used **only** as the weight
                       source for draft-side merging (offload_plan.md §2.7,
                       merge variant (d) per M4). `None` if
                       `load_cpu_source=False`. No forward ever runs on it.

    `moe_infinity` is imported lazily so an hf-only environment that never
    calls this function does not need it installed.
    """
    from moe_infinity import MoE  # lazy: hf-only env doesn't need it

    if device is None:
        device = torch.device("cuda:0")

    moe = MoE(model_id, {
        "offload_path": offload_path,
        "device_memory_ratio": device_memory_ratio,
    })
    model = moe.model
    model.eval()
    _patch_offload_device(model, device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cpu_source = None
    if load_cpu_source:
        cpu_source = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cpu", torch_dtype=dtype,
            trust_remote_code=trust_remote_code)
        cpu_source.eval()

    return model, tokenizer, moe, cpu_source


def get_model_device(model: nn.Module) -> torch.device:
    dev = next(model.parameters()).device
    # Offload models hold CPU placeholders, so the param device wrongly reads
    # cpu; trust the GPU that `load_offload` pinned onto `model.device` (M2/M5).
    # hf models never hit this branch, so their behaviour is unchanged.
    if dev.type == "cpu":
        md = getattr(model, "device", None)
        if isinstance(md, torch.device):
            return md
    return dev


def reset_memory_stats() -> None:
    """Reset peak-memory counters on all visible CUDA devices.

    Safe to call before any CUDA work has happened — silently skips
    devices whose context isn't initialised yet (some clusters report
    `device_count() > 0` for devices the current process can't touch).
    """
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(i)
        except RuntimeError:
            continue
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass


def get_peak_vram_gb() -> float:
    """Largest peak across visible CUDA devices, in GB. Returns 0 on
    CPU-only or when no device is reachable."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        return 0.0
    peaks = []
    for i in range(torch.cuda.device_count()):
        try:
            peaks.append(torch.cuda.max_memory_allocated(i) / (1024 ** 3))
        except RuntimeError:
            continue
    return max(peaks) if peaks else 0.0


def compute_model_vram_bytes(model_id: str, dtype: torch.dtype,
                             trust_remote_code: bool = True) -> int:
    """Full-model footprint in bytes = total params × dtype-size, built on the
    **meta device** (structure only, no weights allocated). This is the `x`
    ("整個模型佔的 VRAM") for the `vram_budget_ratio` budget — GPU-independent,
    matches the thesis 0.2x rule (verify_merge_plan.md §0)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    with torch.device("meta"):
        meta = AutoModelForCausalLM.from_config(
            cfg, trust_remote_code=trust_remote_code)
    n_params = sum(p.numel() for p in meta.parameters())
    del meta
    bytes_per = torch.finfo(dtype).bits // 8
    return n_params * bytes_per


def compute_expert_geometry(model_id: str, dtype: torch.dtype,
                            trust_remote_code: bool = True) -> Tuple[int, int]:
    """(num_hidden_layers, bytes of ONE expert) from config — no weights.
    Cache-mode budget math (merged_cache_plan.md §3): verify floor and the
    per-layer slot cap are derived from these two numbers."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    n_layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    inter = getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size
    bytes_per = torch.finfo(dtype).bits // 8
    return n_layers, 3 * inter * hidden * bytes_per


def compute_merged_bytes(model_id: str, K: int, dtype: torch.dtype,
                         trust_remote_code: bool = True) -> int:
    """Fixed footprint of the cached merged draft experts = K clusters/layer ×
    MoE layers × one-expert bytes (gate+up+down). Reserved out of the budget so
    the merged residency is bounded within 0.2x rather than growing freely in
    the torch allocator (verify_merge_plan.md P1/P2 — honest scarce-VRAM sim).
    Computed from config (no weights)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    n_layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    inter = getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size
    bytes_per = torch.finfo(dtype).bits // 8
    expert_bytes = 3 * inter * hidden * bytes_per   # gate_proj + up_proj + down_proj
    return int(K) * n_layers * expert_bytes


def compute_precache_pool_bytes(
        model_id: str, pin_fraction: float, batch_size: int,
        dtype: torch.dtype, trust_remote_code: bool = True,
        pipeline_slack_topk: int = 2,
        headroom_experts_override: Optional[int] = None
        ) -> Tuple[int, int, int, int]:
    """Archer pool budget for the ``moe_precache`` / ``moe_ondemand`` baselines
    (config-only, no weights). The pool holds ONLY:

      * the pinned set — top ``pin_fraction`` experts per MoE layer, kept
        resident for the whole question (``pin_fraction=0`` → NO pins, i.e. the
        ``moe_ondemand`` zero-cache lower bound), and
      * a minimal streaming buffer — one dispatch's working set
        (``min(E, batch*top_k)``) plus ``pipeline_slack_topk*top_k`` slack.

    The archer engine STREAMS experts through the pool (Enqueue: resident →
    exec, else fetch; a full pool evicts the least-visited non-pinned expert to
    make room — expert_dispatcher.cpp Enqueue/FindExpertEvict). So a forward
    never needs all routed experts resident at once, only the in-flight
    pipeline. Sizing the headroom to one dispatch's working set (not a
    persistent cache) means every non-pinned expert is fetched on demand and
    evicted as the forward advances — nothing beyond the pins is cached across
    layers or decode steps. This is the whole point of the baseline: a LARGER
    headroom (e.g. a full E×safety) would act as an LRU cache and defeat it.

    Returns ``(pool_bytes, n_pin_per_layer, num_experts, expert_bytes)``."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    n_layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    inter = getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size
    n_experts = (getattr(cfg, "num_experts", None)
                 or getattr(cfg, "num_local_experts", None)
                 or getattr(cfg, "n_routed_experts", None))
    if n_experts is None:
        raise ValueError(
            f"could not read expert count from {model_id!r} config "
            "(tried num_experts / num_local_experts / n_routed_experts)")
    top_k = int(getattr(cfg, "num_experts_per_tok", None)
                or getattr(cfg, "moe_topk", None) or 8)
    bytes_per = torch.finfo(dtype).bits // 8
    expert_bytes = 3 * inter * hidden * bytes_per   # gate + up + down
    # pin_fraction == 0 → moe_ondemand (no pins); else at least one pin.
    n_pin = 0 if pin_fraction <= 0 else max(1, math.ceil(pin_fraction * n_experts))
    pinned_bytes = n_layers * n_pin * expert_bytes
    # Streaming buffer sizing. Two constraints:
    #   (a) decode: hold one decode dispatch's working set (min(E, batch*top_k)).
    #   (b) prefill FLOOR: a prefill layer can route the WHOLE prompt across up
    #       to every expert, so ONE dispatch commits up to E experts in-flight at
    #       once. The archer fetch thread evict-starves (expert_dispatcher.cpp
    #       :1055 "evict starvation") if the pool can't hold a full dispatch —
    #       so the TOTAL pool (pins + headroom) must be >= E + slack. Pins count
    #       toward this floor (that is why moe_precache's 624 pins let its
    #       headroom stay tiny, while moe_ondemand with 0 pins needs headroom
    #       itself to cover the floor).
    # This floor (~E) is still far below one decode STEP's fetch count
    # (n_layers * working >= 48*8 = 384 at B=1), so nothing survives across
    # steps: no cache, only a transient in-flight buffer.
    slack = pipeline_slack_topk * top_k
    working = min(n_experts, max(1, batch_size) * top_k)
    pinned_total = n_layers * n_pin
    prefill_floor = n_experts + slack          # pool must hold one full dispatch
    headroom_experts = max(working + slack, prefill_floor - pinned_total)
    if headroom_experts_override is not None:
        headroom_experts = headroom_experts_override
    headroom_bytes = headroom_experts * expert_bytes
    pool_bytes = pinned_bytes + headroom_bytes
    return pool_bytes, n_pin, int(n_experts), expert_bytes


def get_gpu_used_bytes(device: int = 0) -> int:
    """Driver-level total GPU memory in use, via `cudaMemGetInfo` (free/total).
    **Includes archer's own cudaMalloc pool** — unlike torch
    `max_memory_allocated`, which only sees the torch caching allocator. This is
    the right number for the VRAM budget audit/guard (verify_merge_plan.md
    §0.1). Returns -1 when unavailable."""
    if not torch.cuda.is_available():
        return -1
    try:
        free, total = torch.cuda.mem_get_info(device)
        return total - free
    except Exception:
        return -1


def free_model(model: nn.Module) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
