"""Cluster assignment and the :class:`BatClustering` estimator."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from ba_clustering.bat_algorithm import (
    BAResult,
    ConvergenceHistory,
    StopReason,
    run_bat_algorithm,
)
from ba_clustering.boundaries import Bounds, compute_bounds
from ba_clustering.config import BAConfig
from ba_clustering.fitness import squared_distances, sse_unchecked
from ba_clustering.initialization import initialize_positions
from ba_clustering.validation import (
    ConfigurationError,
    FloatArray,
    NotFittedError,
    validate_centroids,
    validate_data_matrix,
)

Labels = NDArray[np.intp]


def assign_labels(X: object, centroids: object) -> Labels:
    """Return the index of the nearest centroid for each sample, shape ``(n_samples,)``.

    Distance is squared Euclidean. Ties go to the lowest centroid index.
    """
    data = validate_data_matrix(X)
    checked = validate_centroids(centroids, data.shape[1])
    labels: Labels = squared_distances(data, checked).argmin(axis=1)
    return labels


def cluster_sizes(labels: Labels, n_clusters: int) -> NDArray[np.intp]:
    """Return the number of samples per cluster, shape ``(n_clusters,)``."""
    sizes: NDArray[np.intp] = np.bincount(labels, minlength=n_clusters)
    return sizes


def count_empty_clusters(labels: Labels, n_clusters: int) -> int:
    """Return how many of the ``n_clusters`` centroids own no samples."""
    return int(np.count_nonzero(cluster_sizes(labels, n_clusters) == 0))


@dataclass(frozen=True)
class _FittedState:
    """Everything produced by one call to ``fit``."""

    result: BAResult
    bounds: Bounds
    labels: Labels
    runtime_seconds: float


class BatClustering:
    """Optimise ``K`` centroids with the Standard Continuous Bat Algorithm, minimising SSE.

    Each call to :meth:`fit` restarts from ``config.seed``, so refitting the same
    data gives the same result. Class labels are never accepted.

    Example:
        >>> import numpy as np
        >>> from ba_clustering import BAConfig, BatClustering
        >>> config = BAConfig(population_size=5, max_iterations=3, n_clusters=2, A0=0.5,
        ...                   r0=0.5, alpha=0.9, gamma=0.9, f_min=0.0, f_max=2.0, seed=0)
        >>> model = BatClustering(config).fit(np.array([[1.0], [2.0], [8.0], [9.0]]))
        >>> model.centroids_.shape
        (2, 1)
    """

    def __init__(self, config: BAConfig) -> None:
        self.config = config
        self._fitted: _FittedState | None = None

    def fit(self, X: object) -> Self:
        """Optimise centroids on ``X``, shape ``(n_samples, n_features)``.

        Raises:
            DataValidationError: If ``X`` violates the data contract.
            ConfigurationError: If ``n_clusters`` exceeds ``n_samples``.
            FloatingPointError: If the objective becomes non-finite.
        """
        data = validate_data_matrix(X)
        config = self.config
        if config.n_clusters > data.shape[0]:
            raise ConfigurationError(
                f"n_clusters={config.n_clusters} exceeds n_samples={data.shape[0]}."
            )

        start = time.perf_counter()
        bounds = compute_bounds(data)
        rng = np.random.default_rng(config.seed)
        initial = initialize_positions(rng, config.population_size, config.n_clusters, bounds)

        def objective(position: FloatArray) -> float:
            return sse_unchecked(data, position)

        result = run_bat_algorithm(objective, initial, bounds, config, rng)
        runtime = time.perf_counter() - start

        result.best_position.setflags(write=False)
        labels = assign_labels(data, result.best_position)
        self._fitted = _FittedState(result, bounds, labels, runtime)
        return self

    def predict(self, X: object) -> Labels:
        """Return nearest-centroid labels for ``X`` using the learned centroids.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            DataValidationError: If ``X`` is invalid or has a different feature count.
        """
        return assign_labels(X, self.centroids_)

    def fit_predict(self, X: object) -> Labels:
        """Fit on ``X`` and return the labels of ``X``."""
        return self.fit(X).labels_

    def _state(self) -> _FittedState:
        if self._fitted is None:
            raise NotFittedError("Call fit() before using fitted attributes or predict().")
        return self._fitted

    @property
    def centroids_(self) -> FloatArray:
        """Best accepted centroids, read-only, shape ``(K, D)``."""
        return self._state().result.best_position

    @property
    def best_fitness_(self) -> float:
        """SSE of :attr:`centroids_` on the training data."""
        return self._state().result.best_fitness

    @property
    def history_(self) -> ConvergenceHistory:
        """Convergence trace."""
        return self._state().result.history

    @property
    def result_(self) -> BAResult:
        """Full optimiser result, including the final population state."""
        return self._state().result

    @property
    def labels_(self) -> Labels:
        """Training-data labels under :attr:`centroids_`."""
        return self._state().labels

    @property
    def n_evaluations_(self) -> int:
        """Objective evaluations used by the last fit."""
        return self._state().result.final_state.n_evaluations

    @property
    def n_iterations_(self) -> int:
        """Iterations run by the last fit."""
        return self._state().result.n_iterations

    @property
    def stop_reason_(self) -> StopReason:
        """Why the last fit stopped."""
        return self._state().result.stop_reason

    @property
    def n_empty_clusters_(self) -> int:
        """Centroids in :attr:`centroids_` that own no training samples."""
        return count_empty_clusters(self.labels_, self.config.n_clusters)

    @property
    def bounds_(self) -> Bounds:
        """Coordinate bounds computed from the training data."""
        return self._state().bounds

    @property
    def runtime_seconds_(self) -> float:
        """Wall-clock optimisation time of the last fit."""
        return self._state().runtime_seconds
