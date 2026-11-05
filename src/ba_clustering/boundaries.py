"""Coordinate bounds for centroids, and the clipping boundary policy.

Bounds are the per-feature min and max of the preprocessed data. Candidates are
clipped coordinate-wise. Velocities are never changed by clipping
(DECISIONS.md, ADR-005).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ba_clustering.validation import DataValidationError, FloatArray, validate_data_matrix


@dataclass(frozen=True)
class Bounds:
    """Per-feature coordinate limits, each of shape ``(n_features,)``."""

    lower: FloatArray
    upper: FloatArray

    def __post_init__(self) -> None:
        if self.lower.ndim != 1 or self.lower.shape != self.upper.shape:
            raise DataValidationError(
                "lower and upper must be 1-D with equal shapes, "
                f"got {self.lower.shape} and {self.upper.shape}."
            )
        if not (np.isfinite(self.lower).all() and np.isfinite(self.upper).all()):
            raise DataValidationError("Bounds must be finite.")
        if (self.lower > self.upper).any():
            raise DataValidationError("Every lower bound must be <= its upper bound.")

    @property
    def zero_range_features(self) -> list[int]:
        """Indices of features where lower equals upper; those coordinates are pinned."""
        return [int(index) for index in np.flatnonzero(self.lower == self.upper)]


def compute_bounds(X: object) -> Bounds:
    """Return the per-feature min and max of ``X`` as :class:`Bounds`."""
    data = validate_data_matrix(X)
    return Bounds(lower=data.min(axis=0), upper=data.max(axis=0))


def clip_to_bounds(position: FloatArray, bounds: Bounds) -> tuple[FloatArray, int]:
    """Clip a position to ``bounds`` and report how many coordinates were outside.

    Args:
        position: Array whose last axis has length ``n_features``, e.g. ``(K, D)``.
        bounds: Limits broadcast against ``position``.

    Returns:
        A new clipped array, and the number of coordinates that were out of range.
        NaN coordinates are preserved, so the caller can detect them.
    """
    outside = (position < bounds.lower) | (position > bounds.upper)
    return np.clip(position, bounds.lower, bounds.upper), int(np.count_nonzero(outside))
