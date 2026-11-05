from __future__ import annotations

import numpy as np
import pytest

from ba_clustering.bat_algorithm import run_bat_algorithm
from ba_clustering.boundaries import Bounds, clip_to_bounds, compute_bounds
from ba_clustering.fitness import sse_unchecked
from ba_clustering.initialization import initialize_positions
from ba_clustering.validation import DataValidationError
from tests.conftest import make_config


def test_compute_bounds_uses_per_feature_min_and_max() -> None:
    bounds = compute_bounds(np.array([[1.0, 10.0], [3.0, -5.0], [2.0, 0.0]]))
    np.testing.assert_array_equal(bounds.lower, [1.0, -5.0])
    np.testing.assert_array_equal(bounds.upper, [3.0, 10.0])


def test_clip_is_coordinate_wise_and_counts_only_outside_coordinates() -> None:
    bounds = Bounds(np.array([0.0, 0.0]), np.array([1.0, 2.0]))
    position = np.array([[-1.0, 1.0], [0.5, 5.0]])
    clipped, n_clipped = clip_to_bounds(position, bounds)
    np.testing.assert_array_equal(clipped, [[0.0, 1.0], [0.5, 2.0]])
    assert n_clipped == 2


def test_clip_does_not_mutate_input_and_keeps_in_bound_values() -> None:
    bounds = Bounds(np.array([0.0]), np.array([1.0]))
    position = np.array([[0.25], [2.0]])
    before = position.copy()
    clipped, _ = clip_to_bounds(position, bounds)
    np.testing.assert_array_equal(position, before)
    assert clipped[0, 0] == 0.25


def test_clip_preserves_nan_so_it_can_be_detected() -> None:
    bounds = Bounds(np.array([0.0]), np.array([1.0]))
    clipped, _ = clip_to_bounds(np.array([[np.nan]]), bounds)
    assert np.isnan(clipped[0, 0])


def test_value_exactly_on_a_bound_is_not_counted() -> None:
    bounds = Bounds(np.array([0.0]), np.array([1.0]))
    assert clip_to_bounds(np.array([[0.0], [1.0]]), bounds)[1] == 0


def test_zero_range_feature_is_reported_and_pins_the_coordinate() -> None:
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    bounds = compute_bounds(X)
    assert bounds.zero_range_features == [1]
    clipped, _ = clip_to_bounds(np.array([[9.0, 9.0]]), bounds)
    assert clipped[0, 1] == 5.0


def test_inverted_or_non_finite_bounds_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="lower bound"):
        Bounds(np.array([1.0]), np.array([0.0]))
    with pytest.raises(DataValidationError, match="finite"):
        Bounds(np.array([np.nan]), np.array([0.0]))
    with pytest.raises(DataValidationError, match="1-D"):
        Bounds(np.array([0.0, 0.0]), np.array([1.0]))


def test_initial_positions_have_the_required_shape_and_respect_bounds() -> None:
    bounds = Bounds(np.array([0.0, -2.0, 5.0]), np.array([1.0, 2.0, 5.0]))
    positions = initialize_positions(np.random.default_rng(0), 7, 4, bounds)
    assert positions.shape == (7, 4, 3)
    assert (positions >= bounds.lower).all()
    assert (positions <= bounds.upper).all()
    assert (positions[..., 2] == 5.0).all()


def test_all_accepted_positions_stay_in_bounds_during_a_run() -> None:
    # A large f_max drives global moves far outside the box.
    X = np.random.default_rng(1).normal(size=(40, 3))
    bounds = compute_bounds(X)
    config = make_config(population_size=8, max_iterations=30, n_clusters=3, f_max=50.0, r0=1.0)
    rng = np.random.default_rng(config.seed)
    initial = initialize_positions(rng, 8, 3, bounds)
    result = run_bat_algorithm(lambda p: sse_unchecked(X, p), initial, bounds, config, rng)
    state = result.final_state
    assert (state.positions >= bounds.lower).all()
    assert (state.positions <= bounds.upper).all()
    assert (result.best_position >= bounds.lower).all()
    assert (result.best_position <= bounds.upper).all()
    assert sum(result.history.n_clipped_coordinates) > 0


def test_clipping_does_not_change_velocity() -> None:
    # Velocity after a clipped move keeps its unclipped value: see ADR-005.
    X = np.array([[0.0], [1.0]])
    bounds = compute_bounds(X)
    config = make_config(
        population_size=2, max_iterations=1, n_clusters=1, f_min=10.0, f_max=10.0, r0=1.0
    )
    initial = np.array([[[0.25]], [[0.5]]])
    fitness = [sse_unchecked(X, p) for p in initial]
    best = initial[int(np.argmin(fitness))]
    rng = np.random.default_rng(0)
    result = run_bat_algorithm(lambda p: sse_unchecked(X, p), initial, bounds, config, rng)
    expected = (initial[0] - best) * 10.0  # far outside [0, 1] once added to the position
    np.testing.assert_allclose(result.final_state.velocities[0], expected)
    assert sum(result.history.n_clipped_coordinates) >= 1
