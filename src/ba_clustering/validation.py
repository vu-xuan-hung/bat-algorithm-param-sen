"""Exceptions and array validators shared across the package.

Validators reject bad input with actionable messages. They never repair data:
NaN is not replaced, shapes are not guessed, and strings are not coerced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_NUMERIC_KINDS = frozenset("iuf")


class ConfigurationError(ValueError):
    """A configuration value is missing, unknown, or outside its valid range."""


class DataValidationError(ValueError):
    """An input array or dataset violates the data contract."""


class NotFittedError(RuntimeError):
    """A fitted attribute or method was used before ``fit`` was called."""


def _as_numeric_array(values: Any, name: str) -> FloatArray:
    """Convert ``values`` to a float64 array, rejecting non-numeric content."""
    try:
        array = np.asarray(values)
    except ValueError as error:
        raise DataValidationError(
            f"{name} has inconsistent dimensions and cannot form a rectangular array: {error}"
        ) from error
    if array.dtype.kind not in _NUMERIC_KINDS:
        raise DataValidationError(
            f"{name} must contain only real numeric values, got dtype {array.dtype}. "
            "Encode or remove non-numeric columns before calling this function."
        )
    return array.astype(np.float64)


def validate_data_matrix(X: Any) -> FloatArray:
    """Return ``X`` as a validated float64 array of shape ``(n_samples, n_features)``.

    The input is never modified; the returned array is a new object.

    Raises:
        DataValidationError: If ``X`` is not 2-D, is empty, is non-numeric, or
            contains NaN or infinity.
    """
    array = _as_numeric_array(X, "X")
    if array.ndim != 2:
        raise DataValidationError(
            f"X must be 2-D with shape (n_samples, n_features), got shape {array.shape}."
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise DataValidationError(f"X must not be empty, got shape {array.shape}.")
    if not np.isfinite(array).all():
        n_bad = int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        raise DataValidationError(
            f"X contains {n_bad} non-finite value(s) (NaN or infinity). "
            "This project does not impute; clean the data before fitting."
        )
    return array


def validate_centroids(centroids: Any, n_features: int) -> FloatArray:
    """Return ``centroids`` as a validated float64 array of shape ``(K, n_features)``.

    Raises:
        DataValidationError: If the shape is wrong, the array is empty, or any
            coordinate is non-numeric or non-finite.
    """
    array = _as_numeric_array(centroids, "centroids")
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != n_features:
        raise DataValidationError(
            f"centroids must have shape (K, {n_features}) with K >= 1, got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise DataValidationError("centroids contain NaN or infinity.")
    return array
