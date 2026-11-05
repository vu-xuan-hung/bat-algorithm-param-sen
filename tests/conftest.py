"""Shared fixtures. All data is local; no test needs network access."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ba_clustering.config import BAConfig
from ba_clustering.validation import FloatArray


@pytest.fixture
def toy_X() -> FloatArray:
    """The brief's 1-D fixture: two obvious clusters, optimum SSE 1.0 at (1.5, 8.5)."""
    return np.array([[1.0], [2.0], [8.0], [9.0]])


def make_config(**overrides: Any) -> BAConfig:
    """Return a small valid config, with the given fields replaced."""
    values: dict[str, Any] = {
        "population_size": 6,
        "max_iterations": 10,
        "n_clusters": 2,
        "A0": 0.5,
        "r0": 0.5,
        "alpha": 0.9,
        "gamma": 0.9,
        "f_min": 0.0,
        "f_max": 2.0,
        "seed": 42,
    }
    values.update(overrides)
    return BAConfig(**values)


@pytest.fixture
def config_factory() -> Any:
    """Expose :func:`make_config` to tests."""
    return make_config
