"""Reproducible initial bat state.

Positions are uniform inside the data's bounding box, velocities are zero, and
every bat starts with ``A_i = A0`` and ``r_i = r0`` (DECISIONS.md, ADR-002, ADR-007).
"""

from __future__ import annotations

import numpy as np

from ba_clustering.boundaries import Bounds
from ba_clustering.validation import FloatArray


def initialize_positions(
    rng: np.random.Generator, population_size: int, n_clusters: int, bounds: Bounds
) -> FloatArray:
    """Return initial positions of shape ``(population_size, n_clusters, n_features)``."""
    shape = (population_size, n_clusters, bounds.lower.shape[0])
    return rng.uniform(bounds.lower, bounds.upper, size=shape)


def initial_pulse_rates(r0: float, population_size: int) -> FloatArray:
    """Return the initial operational pulse rate of each bat, which is ``r0``.

    The schedule ``r0 (1 - exp(-gamma t))`` equals zero at ``t = 0`` and is not
    applied here. It runs only on acceptance, with ``t >= 1``.
    """
    return np.full(population_size, float(r0))
