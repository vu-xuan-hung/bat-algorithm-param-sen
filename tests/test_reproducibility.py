from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ba_clustering.clustering import BatClustering
from ba_clustering.datasets import load_iris_dataset
from ba_clustering.preprocessing import preprocess
from ba_clustering.validation import FloatArray
from tests.conftest import make_config


@pytest.fixture
def iris() -> FloatArray:
    return preprocess(load_iris_dataset().X, "zscore")[0]


def test_same_seed_gives_identical_histories_and_centroids(iris: FloatArray) -> None:
    config = make_config(population_size=10, max_iterations=30, n_clusters=3, seed=123)
    first = BatClustering(config).fit(iris)
    second = BatClustering(config).fit(iris)
    assert first.history_.to_dict() == second.history_.to_dict()
    np.testing.assert_array_equal(first.centroids_, second.centroids_)
    np.testing.assert_array_equal(first.labels_, second.labels_)


def test_refitting_one_estimator_restarts_from_the_seed(iris: FloatArray) -> None:
    model = BatClustering(make_config(n_clusters=3, seed=5)).fit(iris)
    first = model.history_.to_dict()
    assert model.fit(iris).history_.to_dict() == first


def test_global_numpy_random_state_has_no_effect(iris: FloatArray) -> None:
    config = make_config(n_clusters=3, seed=5)
    np.random.seed(1)
    first = BatClustering(config).fit(iris).best_fitness_
    np.random.seed(2)
    np.random.random(100)
    assert BatClustering(config).fit(iris).best_fitness_ == first


def test_fit_does_not_consume_or_touch_global_random_state(iris: FloatArray) -> None:
    np.random.seed(11)
    expected = np.random.random()
    np.random.seed(11)
    BatClustering(make_config(n_clusters=3)).fit(iris)
    assert np.random.random() == expected


def test_input_data_is_not_modified(iris: FloatArray) -> None:
    before = iris.copy()
    BatClustering(make_config(n_clusters=3)).fit(iris)
    np.testing.assert_array_equal(iris, before)


@pytest.mark.parametrize(
    "overrides",
    [
        {"A0": 1.0},
        {"r0": 0.1},
        {"alpha": 0.5},
        {"gamma": 3.0},
        {"f_min": 1.0},
        {"f_max": 7.0},
    ],
)
def test_control_parameters_share_the_initial_population_and_random_stream(
    iris: FloatArray, overrides: dict[str, Any]
) -> None:
    """Paired-seed comparison depends on this: see ADR-006."""
    reference = make_config(population_size=10, max_iterations=20, n_clusters=3, seed=9)
    changed = make_config(population_size=10, max_iterations=20, n_clusters=3, seed=9, **overrides)
    first = BatClustering(reference).fit(iris).history_
    second = BatClustering(changed).fit(iris).history_
    assert first.best_fitness[0] == second.best_fitness[0]
    assert first.mean_fitness[0] == second.mean_fitness[0]


def test_randomness_consumed_is_independent_of_parameters() -> None:
    """Each iteration draws the same amount, whichever branches run."""
    from ba_clustering.bat_algorithm import draw_iteration_randomness

    first, second = np.random.default_rng(0), np.random.default_rng(0)
    a = draw_iteration_randomness(first, 5, (3, 2))
    b = draw_iteration_randomness(second, 5, (3, 2))
    np.testing.assert_array_equal(a.beta, b.beta)
    np.testing.assert_array_equal(a.pulse_draw, b.pulse_draw)
    np.testing.assert_array_equal(a.epsilon, b.epsilon)
    np.testing.assert_array_equal(a.loudness_draw, b.loudness_draw)
    assert first.bit_generator.state == second.bit_generator.state


def test_draw_order_is_beta_pulse_epsilon_loudness() -> None:
    """Pins the random-number semantics. Reordering these changes every published result."""
    from ba_clustering.bat_algorithm import draw_iteration_randomness

    draws = draw_iteration_randomness(np.random.default_rng(3), 4, (2, 2))
    rng = np.random.default_rng(3)
    np.testing.assert_array_equal(draws.beta, rng.random(4))
    np.testing.assert_array_equal(draws.pulse_draw, rng.random(4))
    np.testing.assert_array_equal(draws.epsilon, rng.uniform(-1.0, 1.0, size=(4, 2, 2)))
    np.testing.assert_array_equal(draws.loudness_draw, rng.random(4))


def test_initial_positions_come_first_from_the_stream(iris: FloatArray) -> None:
    from ba_clustering.boundaries import compute_bounds

    config = make_config(population_size=4, max_iterations=1, n_clusters=3, seed=8, A0=0.0)
    model = BatClustering(config).fit(iris)
    bounds = compute_bounds(iris)
    expected = np.random.default_rng(8).uniform(bounds.lower, bounds.upper, size=(4, 3, 4))
    # With A0 = 0 nothing is accepted, so the final positions are the initial ones.
    np.testing.assert_array_equal(model.result_.final_state.positions, expected)


def test_run_consumes_the_same_stream_whatever_the_parameters(iris: FloatArray) -> None:
    from ba_clustering.bat_algorithm import run_bat_algorithm
    from ba_clustering.boundaries import compute_bounds
    from ba_clustering.fitness import sse_unchecked
    from ba_clustering.initialization import initialize_positions

    bounds = compute_bounds(iris)
    final_states = []
    for overrides in ({"A0": 0.0, "r0": 1.0}, {"A0": 1.0, "r0": 0.0}):
        config = make_config(population_size=5, max_iterations=7, n_clusters=3, **overrides)
        rng = np.random.default_rng(config.seed)
        initial = initialize_positions(rng, 5, 3, bounds)
        run_bat_algorithm(lambda p: sse_unchecked(iris, p), initial, bounds, config, rng)
        final_states.append(rng.bit_generator.state)
    assert final_states[0] == final_states[1]
