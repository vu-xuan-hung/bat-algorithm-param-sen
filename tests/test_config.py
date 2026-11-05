from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from ba_clustering.config import BAConfig
from ba_clustering.validation import ConfigurationError


def test_valid_config_round_trips_through_mapping(config_factory: Any) -> None:
    config = config_factory(tolerance=0.01, patience=5)
    assert BAConfig.from_mapping(config.to_dict()) == config


def test_all_six_control_parameters_are_stored_as_given(config_factory: Any) -> None:
    config = config_factory(A0=0.3, r0=0.4, alpha=0.7, gamma=0.2, f_min=0.5, f_max=1.5)
    assert (config.A0, config.r0, config.alpha, config.gamma, config.f_min, config.f_max) == (
        0.3, 0.4, 0.7, 0.2, 0.5, 1.5,
    )  # fmt: skip


def test_config_is_immutable(config_factory: Any) -> None:
    with pytest.raises(AttributeError):
        config_factory().A0 = 0.1


def test_max_evaluations_counts_initial_population(config_factory: Any) -> None:
    assert config_factory(population_size=30, max_iterations=99).max_evaluations == 3000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("population_size", 0),
        ("population_size", 2.5),
        ("population_size", True),
        ("max_iterations", 0),
        ("n_clusters", 0),
        ("seed", -1),
        ("seed", 1.0),
        ("A0", -0.01),
        ("A0", 1.01),
        ("r0", -0.01),
        ("r0", 1.01),
        ("alpha", 0.0),
        ("alpha", 1.01),
        ("gamma", 0.0),
        ("gamma", -1.0),
        ("f_min", -0.1),
        ("f_max", -0.1),
        ("tolerance", -1e-9),
        ("patience", 0),
        ("patience", 1.5),
        ("A0", "0.5"),
        ("A0", None),
        ("A0", True),
    ],
)
def test_out_of_range_or_wrong_type_is_rejected(
    config_factory: Any, field: str, value: object
) -> None:
    with pytest.raises(ConfigurationError, match=field):
        config_factory(**{field: value})


@pytest.mark.parametrize("field", ["A0", "r0", "alpha", "gamma", "f_min", "f_max", "tolerance"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(config_factory: Any, field: str, value: float) -> None:
    with pytest.raises(ConfigurationError, match=field):
        config_factory(**{field: value})


def test_f_min_above_f_max_is_rejected(config_factory: Any) -> None:
    with pytest.raises(ConfigurationError, match="f_min must be <= f_max"):
        config_factory(f_min=2.0, f_max=1.0)


def test_equal_frequency_bounds_are_allowed(config_factory: Any) -> None:
    assert config_factory(f_min=1.0, f_max=1.0).f_min == 1.0


def test_tolerance_without_patience_is_rejected_not_ignored(config_factory: Any) -> None:
    with pytest.raises(ConfigurationError, match="no effect without patience"):
        config_factory(tolerance=0.1, patience=None)


def test_boundary_values_are_valid(config_factory: Any) -> None:
    config = config_factory(A0=0.0, r0=1.0, alpha=1.0, gamma=1e-9, f_min=0.0, f_max=0.0)
    assert config.alpha == 1.0


def test_numpy_scalars_are_accepted(config_factory: Any) -> None:
    config = config_factory(population_size=np.int64(3), A0=np.float64(0.25))
    assert config.population_size == 3


def test_unknown_mapping_key_is_rejected(config_factory: Any) -> None:
    values = config_factory().to_dict() | {"sign_convention": "toward_best"}
    with pytest.raises(ConfigurationError, match="sign_convention"):
        BAConfig.from_mapping(values)


def test_missing_mapping_key_is_rejected(config_factory: Any) -> None:
    values = config_factory().to_dict()
    del values["gamma"]
    with pytest.raises(ConfigurationError, match="gamma"):
        BAConfig.from_mapping(values)
