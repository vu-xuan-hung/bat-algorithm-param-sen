"""Evaluation metrics and statistics. None of these is an optimisation objective."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from sklearn.metrics import adjusted_rand_score, silhouette_score

from ba_clustering.validation import DataValidationError, FloatArray


def silhouette_or_none(X: FloatArray, labels: NDArray[np.intp]) -> float | None:
    """Return the mean Silhouette score, or ``None`` when it is undefined.

    It is defined only for ``2 <= distinct labels <= n_samples - 1``. A run whose
    centroids leave some clusters empty is scored on the labels actually used.
    """
    n_labels = np.unique(labels).size
    if not 2 <= n_labels <= X.shape[0] - 1:
        return None
    return float(silhouette_score(X, labels, metric="euclidean"))


def adjusted_rand_or_none(
    labels_external: NDArray[np.generic] | None, labels: NDArray[np.intp]
) -> float | None:
    """Return the Adjusted Rand Index, or ``None`` when no external labels exist.

    Evaluation only: external labels must never influence optimisation.
    """
    if labels_external is None:
        return None
    return float(adjusted_rand_score(labels_external, labels))


def evaluations_to_plateau(
    evaluations: Sequence[int], best_fitness: Sequence[float], tolerance: float = 0.0
) -> int:
    """Return the first evaluation count at which best fitness was within ``tolerance`` of final.

    A plateau is only "no further improvement within this run's budget". It does not
    imply that a global optimum was reached.
    """
    if len(evaluations) != len(best_fitness) or not evaluations:
        raise DataValidationError(
            "evaluations and best_fitness must be non-empty and equal length."
        )
    threshold = best_fitness[-1] + tolerance
    for count, value in zip(evaluations, best_fitness, strict=True):
        if value <= threshold:
            return count
    return evaluations[-1]


@dataclass(frozen=True)
class Summary:
    """Descriptive statistics of one metric over repeated runs.

    ``std`` uses ``ddof=1``. ``std`` and the confidence interval are ``None`` for fewer
    than two values. The interval is a two-sided 95% Student-t interval on the mean.
    """

    n: int
    mean: float
    std: float | None
    median: float
    q1: float
    q3: float
    minimum: float
    maximum: float
    ci95_low: float | None
    ci95_high: float | None


def summarize(values: ArrayLike) -> Summary:
    """Return descriptive statistics of ``values``.

    Raises:
        DataValidationError: If ``values`` is empty or contains non-finite numbers.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise DataValidationError("summarize needs a non-empty array of finite values.")
    n = int(array.size)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if n > 1 else None
    if std is None:
        low, high = None, None
    else:
        half_width = float(stats.t.ppf(0.975, n - 1)) * std / np.sqrt(n)
        low, high = mean - half_width, mean + half_width
    q1, median, q3 = np.percentile(array, [25, 50, 75])
    return Summary(
        n=n,
        mean=mean,
        std=std,
        median=float(median),
        q1=float(q1),
        q3=float(q3),
        minimum=float(array.min()),
        maximum=float(array.max()),
        ci95_low=low,
        ci95_high=high,
    )


@dataclass(frozen=True)
class PairedComparison:
    """Paired comparison of two conditions over shared seeds.

    ``mean_difference`` and ``median_difference`` are ``a - b``; for SSE, a negative
    value means ``a`` is lower. ``p_value`` is from the two-sided Wilcoxon signed-rank
    test, and is 1.0 when every difference is zero.
    """

    n: int
    mean_difference: float
    median_difference: float
    n_a_lower: int
    n_b_lower: int
    n_tied: int
    p_value: float


def paired_comparison(a: ArrayLike, b: ArrayLike) -> PairedComparison:
    """Compare paired samples ``a`` and ``b``, whose positions must correspond to the same seed.

    Raises:
        DataValidationError: If shapes differ, or the input is empty or non-finite.
    """
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    if first.shape != second.shape or first.ndim != 1 or first.size == 0:
        raise DataValidationError(
            f"Paired samples must be non-empty 1-D arrays of equal length, "
            f"got {first.shape} and {second.shape}."
        )
    if not (np.isfinite(first).all() and np.isfinite(second).all()):
        raise DataValidationError("Paired samples must be finite.")
    difference = first - second
    p_value = 1.0
    if np.any(difference != 0.0):
        p_value = float(stats.wilcoxon(first, second, zero_method="wilcox").pvalue)
    return PairedComparison(
        n=int(difference.size),
        mean_difference=float(difference.mean()),
        median_difference=float(np.median(difference)),
        n_a_lower=int(np.count_nonzero(difference < 0)),
        n_b_lower=int(np.count_nonzero(difference > 0)),
        n_tied=int(np.count_nonzero(difference == 0)),
        p_value=p_value,
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values, in the input order."""
    n = len(p_values)
    order = sorted(range(n), key=lambda index: p_values[index])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (n - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted
