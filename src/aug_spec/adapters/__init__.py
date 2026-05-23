"""Adapter registry.

Lookup by explicit name (`get_adapter("mixtral")`) or by HF
`config.model_type` (`adapter_for_config(model.config)`).

To add a new model family:
  1. Drop a `<family>.py` next to `mixtral.py` exporting a subclass of
     `MoEAdapter` with a unique `.name`.
  2. Register it in `_REGISTRY` and (optionally) `_MODEL_TYPE_MAP` below.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import MoEAdapter
from .gptoss import GptOssAdapter
from .mixtral import MixtralAdapter
from .qwen3 import Qwen3MoeAdapter


_REGISTRY: Dict[str, Type[MoEAdapter]] = {
    MixtralAdapter.name: MixtralAdapter,
    GptOssAdapter.name: GptOssAdapter,
    Qwen3MoeAdapter.name: Qwen3MoeAdapter,
}

# HF `config.model_type` → adapter name.
_MODEL_TYPE_MAP: Dict[str, str] = {
    "mixtral": "mixtral",
    "gpt_oss": "gptoss",
    "qwen3_moe": "qwen3_moe",
}


def get_adapter(name: str) -> MoEAdapter:
    """Instantiate an adapter by registered name."""
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown adapter {name!r}; known: {known}")
    return _REGISTRY[name]()


def adapter_for_config(config: Any) -> MoEAdapter:
    """Pick an adapter from a HF model config (`config.model_type`)."""
    mt = getattr(config, "model_type", None)
    if mt is None:
        raise TypeError("config has no `.model_type`; pass adapter name explicitly.")
    if mt not in _MODEL_TYPE_MAP:
        known = ", ".join(sorted(_MODEL_TYPE_MAP))
        raise KeyError(
            f"no adapter registered for model_type={mt!r}; known: {known}")
    return get_adapter(_MODEL_TYPE_MAP[mt])


__all__ = [
    "MoEAdapter",
    "MixtralAdapter",
    "GptOssAdapter",
    "Qwen3MoeAdapter",
    "get_adapter",
    "adapter_for_config",
]
