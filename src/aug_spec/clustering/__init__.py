"""Cluster-method registry.

Add a method: drop a `ClusterMethod` subclass module and register it here.
Selected by `cluster.name` in the YAML (wired in A4); defaults to freq_slice.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import ClusterContext, ClusterMethod
from .freq_slice import FreqSliceCluster


_REGISTRY: Dict[str, Type[ClusterMethod]] = {
    "freq_slice": FreqSliceCluster,
}


def get_cluster_method(name: str = "freq_slice", **kwargs: Any) -> ClusterMethod:
    """Instantiate a cluster method by registered name with **kwargs."""
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown cluster method {name!r}; known: {known}")
    return _REGISTRY[name](**kwargs)


__all__ = [
    "ClusterContext",
    "ClusterMethod",
    "FreqSliceCluster",
    "get_cluster_method",
]
