"""Invariants that must hold throughout real runs, whatever the random draws."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ba_clustering.bat_algorithm import (
    draw_iteration_randomness,
    initialize_state,
    run_bat_algorithm,
    run_iteration,
)
from ba_clustering.boundaries import compute_bounds
from ba_clustering.clustering import BatClustering
from ba_clustering.fitness import compute_sse, sse_unchecked
from ba_clustering.initialization import initialize_positions
from ba_clustering.validation import FloatArray
from tests.conftest import make_config


@pytest.fixture
def data() -> FloatArray:
    rng = np.random.default_rng(3)
    return np.vstack([rng.normal(loc, 0.5, size=(30, 2)) for loc in (-3.0, 0.0, 3.0)])


PARAMETER_CASES = [
    {},
    {"A0": 1.0},
    {"A0": 0.0},
    {"r0": 0.0},
    {"r0": 1.0},
    {"alpha": 1.0},
    {"alpha": 0.1},
    {"gamma": 0.01},
    {"f_min": 1.0, "f_max": 1.0},
    {"f_max": 20.0},
    {"population_size": 1},
    {"population_size": 2},
]


@pytest.mark.parametrize("overrides", PARAMETER_CASES)
@pytest.mark.parametrize("seed", [0, 7])
def test_best_so_far_never_increases(
    data: FloatArray, overrides: dict[str, Any], seed: int
) -> None:
    config = make_config(max_iterations=40, n_clusters=3, seed=seed, **overrides)
    history = BatClustering(config).fit(data).history_
    assert all(b <= a for a, b in zip(history.best_fitness, history.best_fitness[1:], strict=False))


@pytest.mark.parametrize("overrides", PARAMETER_CASES)
def test_best_fitness_equals_recomputed_sse_of_stored_centroids(
    data: FloatArray, overrides: dict[str, Any]
) -> None:
    config = make_config(max_iterations=40, n_clusters=3, **overrides)
    model = BatClustering(config).fit(data)
    assert model.best_fitness_ == compute_sse(data, model.centroids_)
    assert model.history_.best_fitness[-1] == model.best_fitness_


@pytest.mark.parametrize("overrides", PARAMETER_CASES)
def test_accepted_fitness_stays_synchronised_with_positions(
    data: FloatArray, overrides: dict[str, Any]
) -> None:
    config = make_config(max_iterations=40, n_clusters=3, **overrides)
    state = BatClustering(config).fit(data).result_.final_state
    recomputed = [sse_unchecked(data, position) for position in state.positions]
    np.testing.assert_array_equal(state.fitness, recomputed)


def test_invariants_hold_after_every_single_iteration(data: FloatArray) -> None:
    config = make_config(population_size=8, max_iterations=30, n_clusters=3)
    bounds = compute_bounds(data)
    rng = np.random.default_rng(config.seed)

    def objective(position: FloatArray) -> float:
        return sse_unchecked(data, position)

    state = initialize_state(
        initialize_positions(rng, 8, 3, bounds), objective, config.r0, config.A0
    )
    for t in range(1, 31):
        previous_fitness = state.fitness.copy()
        previous_loudness = state.loudness.copy()
        previous_positions = state.positions.copy()
        previous_pulse = state.pulse_rates.copy()
        previous_best = state.best_fitness
        draws = draw_iteration_randomness(rng, 8, (3, 2))
        stats = run_iteration(state, t, draws, objective, config, bounds)

        accepted = state.fitness < previous_fitness
        assert accepted.sum() == stats.n_accepted
        # Each bat's accepted fitness never worsens.
        assert (state.fitness <= previous_fitness).all()
        # Loudness and pulse rate change exactly when the bat accepted.
        np.testing.assert_array_equal(state.loudness < previous_loudness, accepted)
        np.testing.assert_array_equal(state.loudness[~accepted], previous_loudness[~accepted])
        np.testing.assert_array_equal(state.pulse_rates[~accepted], previous_pulse[~accepted])
        # A rejected bat keeps its accepted position.
        np.testing.assert_array_equal(state.positions[~accepted], previous_positions[~accepted])
        # The best is the minimum of everything accepted so far.
        assert state.best_fitness <= previous_best
        assert state.best_fitness <= state.fitness.min()
        assert state.best_fitness == objective(state.best_position)
        # Rates stay in range.
        assert ((state.pulse_rates >= 0.0) & (state.pulse_rates <= config.r0)).all()
        assert ((state.loudness >= 0.0) & (state.loudness <= config.A0)).all()


def test_zero_loudness_never_accepts(data: FloatArray) -> None:
    model = BatClustering(make_config(A0=0.0, max_iterations=20, n_clusters=3)).fit(data)
    assert sum(model.history_.n_accepted) == 0
    assert model.history_.best_fitness[0] == model.best_fitness_


def test_r0_one_means_no_local_search_before_any_acceptance(data: FloatArray) -> None:
    config = make_config(r0=1.0, A0=0.0, max_iterations=20, n_clusters=3)
    assert sum(BatClustering(config).fit(data).history_.n_local_candidates) == 0


def test_r0_zero_always_uses_local_search(data: FloatArray) -> None:
    config = make_config(r0=0.0, max_iterations=20, n_clusters=3)
    history = BatClustering(config).fit(data).history_
    assert history.n_local_candidates[1:] == [config.population_size] * 20


def test_evaluation_count_matches_the_formula_and_the_calls(data: FloatArray) -> None:
    config = make_config(population_size=30, max_iterations=99, n_clusters=3)
    calls = {"n": 0}

    def objective(position: FloatArray) -> float:
        calls["n"] += 1
        return sse_unchecked(data, position)

    bounds = compute_bounds(data)
    rng = np.random.default_rng(config.seed)
    initial = initialize_positions(rng, 30, 3, bounds)
    result = run_bat_algorithm(objective, initial, bounds, config, rng)
    assert calls["n"] == 3000 == config.max_evaluations == result.final_state.n_evaluations
    assert result.history.n_evaluations == [30 * (t + 1) for t in range(100)]


def test_objective_cannot_mutate_state_through_the_candidate(data: FloatArray) -> None:
    # Fitness evaluation must be pure: results must match when a spy keeps copies.
    config = make_config(n_clusters=3)
    baseline = BatClustering(config).fit(data)
    bounds = compute_bounds(data)
    rng = np.random.default_rng(config.seed)

    def spying(position: FloatArray) -> float:
        before = position.copy()
        value = sse_unchecked(data, position)
        np.testing.assert_array_equal(position, before)
        return value

    initial = initialize_positions(rng, config.population_size, 3, bounds)
    result = run_bat_algorithm(spying, initial, bounds, config, rng)
    assert result.best_fitness == baseline.best_fitness_


def test_initial_population_defines_the_initial_best(data: FloatArray) -> None:
    model = BatClustering(make_config(n_clusters=3, max_iterations=1)).fit(data)
    bounds = compute_bounds(data)
    rng = np.random.default_rng(42)
    initial = initialize_positions(rng, 6, 3, bounds)
    expected = min(sse_unchecked(data, position) for position in initial)
    assert model.history_.best_fitness[0] == expected
    assert model.history_.iteration[0] == 0
    assert model.history_.n_evaluations[0] == 6


# --- Stopping --------------------------------------------------------------------


def test_default_runs_the_full_budget(data: FloatArray) -> None:
    model = BatClustering(make_config(n_clusters=3, max_iterations=15)).fit(data)
    assert model.n_iterations_ == 15
    assert model.stop_reason_ == "max_iterations"


def test_patience_stops_after_consecutive_stalled_iterations(data: FloatArray) -> None:
    # With A0 = 0 nothing is ever accepted, so every iteration stalls.
    config = make_config(n_clusters=3, A0=0.0, max_iterations=50, patience=4)
    model = BatClustering(config).fit(data)
    assert model.n_iterations_ == 4
    assert model.stop_reason_ == "patience"
    assert model.n_evaluations_ == config.population_size * 5


def test_tolerance_treats_small_gains_as_stalled(data: FloatArray) -> None:
    strict = make_config(n_clusters=3, max_iterations=200, patience=10, tolerance=0.0)
    loose = make_config(n_clusters=3, max_iterations=200, patience=10, tolerance=1e6)
    assert BatClustering(loose).fit(data).n_iterations_ == 10
    # With zero tolerance the early improvements on this data postpone the stop.
    assert BatClustering(strict).fit(data).n_iterations_ > 10


def test_slow_drift_does_not_reset_patience() -> None:
    # A scripted objective improves by 1 per iteration: below the tolerance of 2.5 each
    # time, but past it after three iterations. Improvement is measured from the last
    # meaningful best, so the third iteration resets the counter.
    X = np.array([[0.0], [1.0]])
    bounds = compute_bounds(X)
    config = make_config(
        population_size=1, n_clusters=1, A0=1.0, alpha=1.0, r0=0.0, max_iterations=100,
        tolerance=2.5, patience=3,
    )  # fmt: skip
    values = iter([100.0 - step for step in range(200)])
    rng = np.random.default_rng(0)
    result = run_bat_algorithm(lambda _p: next(values), np.array([[[0.5]]]), bounds, config, rng)
    # Iterations 1 and 2 stall, iteration 3 resets, so patience=3 is never reached. Comparing
    # only consecutive iterations would wrongly stop here, at iteration 3.
    assert result.n_iterations == 100
    assert result.stop_reason == "max_iterations"
    # With patience 2 the same drift stops, because two stalled iterations come first.
    config_short = make_config(
        population_size=1, n_clusters=1, A0=1.0, alpha=1.0, r0=0.0, max_iterations=100,
        tolerance=2.5, patience=2,
    )  # fmt: skip
    values = iter([100.0 - step for step in range(200)])
    short = run_bat_algorithm(
        lambda _p: next(values), np.array([[[0.5]]]), bounds, config_short, rng
    )
    assert short.n_iterations == 2
    assert short.stop_reason == "patience"


def test_full_run_passes_one_based_iteration_counter_to_the_pulse_schedule() -> None:
    # An always-improving objective with A = 1 accepts every candidate, so after each
    # iteration t the pulse rate must be exactly r0 (1 - exp(-gamma t)), starting at t = 1.
    X = np.array([[0.0], [1.0]])
    config = make_config(
        population_size=1, n_clusters=1, A0=1.0, alpha=1.0, r0=0.5, gamma=0.9, max_iterations=5
    )
    values = iter(range(1000, 0, -1))
    rng = np.random.default_rng(0)
    result = run_bat_algorithm(
        lambda _p: float(next(values)), np.array([[[0.5]]]), compute_bounds(X), config, rng
    )
    assert result.history.n_accepted[1:] == [1] * 5
    assert result.history.mean_pulse_rate[0] == 0.5
    for t in range(1, 6):
        assert result.history.mean_pulse_rate[t] == pytest.approx(0.5 * (1 - np.exp(-0.9 * t)))
