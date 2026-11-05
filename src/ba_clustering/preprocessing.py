"""Feature preprocessing. Fitted once per dataset and shared by every algorithm.

Supported methods:

* ``"zscore"``: ``(x - mean) / std`` with the population standard deviation
  (``ddof=0``), equivalent to scikit-learn's ``StandardScaler``. An exactly constant
  feature becomes all zeros: it is centred and its scale is set to 1.
* ``"none"``: data is returned unchanged, as a copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, get_args

import numpy as np

from ba_clustering.validation import (
    ConfigurationError,
    DataValidationError,
    FloatArray,
    validate_data_matrix,
)

PreprocessingMethod = Literal["zscore", "none"]


@dataclass(frozen=True)
class PreprocessingInfo:
    """Fitted preprocessing parameters.

    Attributes:
        method: Method used.
        mean: Per-feature value subtracted, shape ``(n_features,)``.
        scale: Per-feature divisor, shape ``(n_features,)``.
        constant_features: Indices of features with zero range before scaling.
    """

    method: PreprocessingMethod
    mean: FloatArray
    scale: FloatArray
    constant_features: list[int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description, for result manifests."""
        return {
            "method": self.method,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "constant_features": self.constant_features,
        }


def fit_preprocessing(X: object, method: str = "zscore") -> PreprocessingInfo:
    """Compute preprocessing parameters from ``X``, without transforming it.

    Raises:
        ConfigurationError: If ``method`` is not supported.
        DataValidationError: If ``X`` violates the data contract.
    """
    data = validate_data_matrix(X)
    constant = np.ptp(data, axis=0) == 0.0
    constant_features = [int(index) for index in np.flatnonzero(constant)]
    n_features = data.shape[1]
    if method == "zscore":
        mean = data.mean(axis=0)
        scale = data.std(axis=0, ddof=0)
        # The computed mean of a constant can differ from it by rounding. Use the exact
        # value so that a constant feature maps to exactly zero.
        mean[constant] = data[0, constant]
        scale[constant] = 1.0
        return PreprocessingInfo("zscore", mean, scale, constant_features)
    if method == "none":
        return PreprocessingInfo(
            "none", np.zeros(n_features), np.ones(n_features), constant_features
        )
    raise ConfigurationError(
        f"Unknown preprocessing method {method!r}. "
        f"Supported: {list(get_args(PreprocessingMethod))}."
    )


def apply_preprocessing(X: object, info: PreprocessingInfo) -> FloatArray:
    """Apply fitted parameters to ``X`` and return a new array.

    Raises:
        DataValidationError: If ``X`` is invalid or has the wrong feature count.
    """
    data = validate_data_matrix(X)
    if data.shape[1] != info.mean.shape[0]:
        raise DataValidationError(
            f"X has {data.shape[1]} features, but preprocessing was fitted on {info.mean.shape[0]}."
        )
    transformed: FloatArray = (data - info.mean) / info.scale
    return validate_data_matrix(transformed)


def preprocess(X: object, method: str = "zscore") -> tuple[FloatArray, PreprocessingInfo]:
    """Fit preprocessing on ``X`` and return the transformed data and the fitted parameters."""
    info = fit_preprocessing(X, method)
    return apply_preprocessing(X, info), info
