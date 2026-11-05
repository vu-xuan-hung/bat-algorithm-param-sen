"""Sum of Squared Errors, the only primary optimisation objective.

``SSE = sum_n min_k ||x_n - mu_k||_2^2``

These functions are pure: they never modify the data or the centroids. A centroid
that owns no samples simply contributes nothing, so SSE stays defined for every
finite centroid configuration (DECISIONS.md, ADR-004).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ba_clustering.validation import FloatArray, validate_centroids, validate_data_matrix


def squared_distances(X: FloatArray, centroids: FloatArray) -> FloatArray:
    """Return squared Euclidean distances, shape ``(n_samples, K)``. Inputs are not validated."""
    distances: FloatArray = cdist(X, centroids, metric="sqeuclidean")
    return distances


def sse_unchecked(X: FloatArray, centroids: FloatArray) -> float:
    """Return total SSE without validating inputs. For already-validated hot paths."""
    return float(squared_distances(X, centroids).min(axis=1).sum())


def compute_sse(X: object, centroids: object) -> float:
    """Return the total SSE of ``X`` under ``centroids``.

    Args:
        X: Data of shape ``(n_samples, n_features)``.
        centroids: Centroids of shape ``(K, n_features)``.

    Returns:
        A finite, non-negative float.

    Raises:
        DataValidationError: If either input violates the data contract.
        FloatingPointError: If finite inputs overflow during the computation.
    """
    data = validate_data_matrix(X)
    checked = validate_centroids(centroids, data.shape[1])
    value = sse_unchecked(data, checked)
    if not np.isfinite(value):
        raise FloatingPointError(
            f"SSE overflowed to {value} for finite inputs; the coordinates are too large. "
            "Scale the data before clustering."
        )
    return value
