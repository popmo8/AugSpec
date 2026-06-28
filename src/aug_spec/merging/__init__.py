"""Expert merging.

The merge is **always** a linear weighted average — there is no swappable
merge algorithm (SVD was removed). Future merge variants are only "different
weights", decided by the clustering / within-cluster weighting, not by a
different merge op. So this package is *not* a registry: it holds the single
linear merge entry (`linear_merge`) and, later, an optional cache wrapped
around it (B3).
"""

from __future__ import annotations

from .linear import linear_merge

__all__ = ["linear_merge"]
