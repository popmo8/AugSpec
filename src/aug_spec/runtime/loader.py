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
