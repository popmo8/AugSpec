"""Model loading + memory utilities.

`load_model` returns a HuggingFace causal LM ready for either standard or
speculative decoding. The compat shims here patch older trust_remote_code
models (e.g. DeepSeek-MoE) that call DynamicCache APIs removed in
transformers >= 4.44.

`load_offload` is the sibling for the offload backend: it wraps
`moe_infinity.MoE(...)` and ALSO loads a separate CPU-resident copy of
the model that the draft path uses as a weight source for
`adapter.build_weighted_avg`. Two-model load is necessary because
moe_infinity replaces every offloaded `param.data` with a shape-(1,)
zero placeholder (see model_offload.py:213-221), so the experts cannot
be read directly from `moe.model`.
"""

from __future__ import annotations

import gc
import inspect
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


_ROTARY_DEVICE_PATCH_INSTALLED: bool = False


def _install_rotary_device_patches() -> None:
    """Patch each family's RotaryEmbedding.forward to align position_ids
    with hidden_states' device before the inner matmul.

    Why: in offload mode, the spec-decoding assist path sometimes passes
    `position_ids` on a different device than `hidden_states` (typically
    CPU vs cuda:0). The stock rotary forward does `inv_freq.to(x.device)`
    but leaves `position_ids` alone, so the `@ position_ids` matmul
    crashes with a CPU/CUDA mismatch. moe_infinity already patches the
    LATER `apply_rotary_pos_emb` for the same family of bugs but does
    not touch the rotary buffer pre-stage — we add this small fix.

    Patched classes (only those whose transformers module imports cleanly;
    missing ones are silently skipped):
      * transformers.models.mixtral.modeling_mixtral.MixtralRotaryEmbedding
      * transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRotaryEmbedding

    Idempotent (safe to call multiple times).
    """
    global _ROTARY_DEVICE_PATCH_INSTALLED
    if _ROTARY_DEVICE_PATCH_INSTALLED:
        return

    targets = [
        ("transformers.models.mixtral.modeling_mixtral",
         "MixtralRotaryEmbedding"),
        ("transformers.models.qwen3_moe.modeling_qwen3_moe",
         "Qwen3MoeRotaryEmbedding"),
    ]
    import importlib

    def _make_patched(orig_forward):
        def patched_forward(self, x, position_ids):
            if position_ids.device != x.device:
                position_ids = position_ids.to(x.device)
            return orig_forward(self, x, position_ids)
        return patched_forward

    for module_name, class_name in targets:
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
            cls.forward = _make_patched(cls.forward)
        except (ImportError, AttributeError):
            continue

    _ROTARY_DEVICE_PATCH_INSTALLED = True


# Backward-compat alias — old name used in some call sites.
_install_mixtral_rotary_device_patch = _install_rotary_device_patches


def load_offload(
    model_id: str,
    offload_path,
    dtype: torch.dtype = torch.bfloat16,
    device_memory_ratio: float = 0.15,
    cache_policy: str = "ondemand",
    trust_remote_code: bool = True,
) -> Tuple[nn.Module, Any, Any, nn.Module]:
    """Load the offloaded MoE model + a CPU-resident weight source.

    Returns
    -------
    hf_model    : `moe.model` — the offloaded PreTrainedModel. This is what
                  the controller / adapter / spec-bench see. Target verify
                  runs through this.
    tokenizer   : standard HF tokenizer.
    moe         : the `moe_infinity.MoE` wrapper. Caller MUST invoke
                  `moe._configure_hook(input_ids)` once per generation
                  (or per question) so the archer tracer is set up.
    cpu_source  : a plain HF model loaded with `device_map="cpu"` and
                  identical weights. The draft-side merge reads
                  `cpu_source.model.layers[i].block_sparse_moe.experts[e].w*.weight`
                  one-at-a-time, streams CPU→GPU, accumulates into a
                  fp32 buffer, never runs forward. Lives in host RAM
                  (≈ 26 GB for Mixtral-8x7B bf16).

    Load order matters: cpu_source is loaded FIRST, before
    `moe_infinity.MoE(...)` activates its empty-init hooks. This guarantees
    cpu_source's weights are real (not zero placeholders).
    """
    from pathlib import Path

    from moe_infinity import MoE

    # Patch every supported family's rotary to handle CPU/CUDA
    # position_ids mismatch that surfaces in spec-decoding's assistant
    # path under offload (see _install_rotary_device_patches docstring).
    _install_rotary_device_patches()

    # 1) CPU-resident weight source (before any moe_infinity activity).
    cpu_source = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    cpu_source.eval()
    for p in cpu_source.parameters():
        p.requires_grad_(False)

    # 2) The offloaded model (target verify path).
    Path(offload_path).mkdir(parents=True, exist_ok=True)
    moe = MoE(model_id, {
        "offload_path": str(offload_path),
        "device_memory_ratio": device_memory_ratio,
    })

    # cache_policy mapping: moe_infinity's ExpertCache.set_cache_policy
    # accepts "lru" / "priority" / etc.; map our YAML names onto theirs.
    # "ondemand" → "lru" (vanilla LRU, no predictive pre-pinning).
    # "caching"  → "priority" (tracer-driven prefetcher = MoE-Caching baseline).
    if hasattr(moe.engine, "expert_cache") and hasattr(
            moe.engine.expert_cache, "set_cache_policy"):
        moe.engine.expert_cache.set_cache_policy(
            "priority" if cache_policy == "caching" else "lru")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = moe.model
    hf_model.eval()
    return hf_model, tokenizer, moe, cpu_source


def get_model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


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


def free_model(model: nn.Module) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
