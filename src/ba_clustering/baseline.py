"""Conventional K-Means baseline using scikit-learn's Lloyd algorithm.

Restarts are run one at a time rather than through ``n_init``, because scikit-learn
reports the iteration count of the best restart only. Counting every restart is
needed to report the computational cost honestly.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from ba_clustering.fitness import sse_unchecked
from ba_clustering.validation import ConfigurationError, FloatArray, validate_data_matrix


@dataclass(frozen=True, kw_only=True)
class KMeansConfig:
    """K-Means settings.

    Attributes:
        n_clusters: Fixed ``K``.
        seed: Non-negative seed from which all restart seeds are derived.
        n_init: Number of independent restarts; the lowest-SSE restart is kept.
        init: ``"k-means++"`` or ``"random"``.
        max_iter: Maximum Lloyd iterations per restart.
        tol: scikit-learn convergence tolerance.
    """

    n_clusters: int
    seed: int
    n_init: int
    init: Literal["k-means++", "random"]
    max_iter: int
    tol: float

    def __post_init__(self) -> None:
        for name in ("n_clusters", "n_init", "max_iter", "seed"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ConfigurationError(f"{name} must be an integer, got {value!r}.")
            if value < minimum:
                raise ConfigurationError(f"{name} must be >= {minimum}, got {value}.")
        if self.init not in ("k-means++", "random"):
            raise ConfigurationError(f"Unsupported K-Means init {self.init!r}.")
        if not (isinstance(self.tol, float) and math.isfinite(self.tol) and self.tol >= 0.0):
            raise ConfigurationError(f"tol must be a finite float >= 0, got {self.tol!r}.")

    def to_dict(self) -> dict[str, Any]:
        """Return all fields as a JSON-serialisable dictionary."""
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class KMeansResult:
    """Outcome of :func:`run_kmeans`.

    Attributes:
        centroids: Centroids of the best restart, shape ``(K, D)``.
        labels: Assignment of each sample to its nearest centroid.
        sse: SSE recomputed by this project's own :func:`~ba_clustering.fitness.compute_sse`
            logic, not taken from scikit-learn's ``inertia_``.
        restart_sse: SSE of every restart, shape ``(n_init,)``.
        restart_seeds: Seed passed to scikit-learn for each restart.
        n_lloyd_iterations: Lloyd iterations summed over all restarts. Each one is
            roughly one full assignment pass over the data; seeding costs are excluded.
        runtime_seconds: Wall-clock time of all restarts.
    """

    centroids: FloatArray
    labels: NDArray[np.intp]
    sse: float
    restart_sse: FloatArray
    restart_seeds: list[int]
    n_lloyd_iterations: int
    runtime_seconds: float


def run_kmeans(X: object, config: KMeansConfig) -> KMeansResult:
    """Run multi-start K-Means on ``X`` and keep the restart with the lowest SSE.

    Ties go to the earliest restart.

    Raises:
        DataValidationError: If ``X`` violates the data contract.
        ConfigurationError: If ``n_clusters`` exceeds ``n_samples``.
    """
    data = validate_data_matrix(X)
    if config.n_clusters > data.shape[0]:
        raise ConfigurationError(
            f"n_clusters={config.n_clusters} exceeds n_samples={data.shape[0]}."
        )
    restart_seeds = [
        int(seed)
        for seed in np.random.default_rng(config.seed).integers(0, 2**31 - 1, size=config.n_init)
    ]

    best: KMeans | None = None
    best_sse = np.inf
    restart_sse: list[float] = []
    n_lloyd_iterations = 0
    start = time.perf_counter()
    for restart_seed in restart_seeds:
        model = KMeans(
            n_clusters=config.n_clusters,
            init=config.init,
            n_init=1,
            max_iter=config.max_iter,
            tol=config.tol,
            algorithm="lloyd",
            random_state=restart_seed,
        ).fit(data)
        value = sse_unchecked(data, model.cluster_centers_)
        restart_sse.append(value)
        n_lloyd_iterations += int(model.n_iter_)
        if value < best_sse:
            best, best_sse = model, value
    runtime = time.perf_counter() - start

    if best is None:
        raise FloatingPointError("Every K-Means restart produced a non-finite SSE.")
    return KMeansResult(
        centroids=np.asarray(best.cluster_centers_, dtype=np.float64),
        labels=np.asarray(best.predict(data), dtype=np.intp),
        sse=float(best_sse),
        restart_sse=np.asarray(restart_sse),
        restart_seeds=restart_seeds,
        n_lloyd_iterations=n_lloyd_iterations,
        runtime_seconds=runtime,
    )
