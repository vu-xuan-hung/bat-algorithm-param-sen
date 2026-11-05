"""Update rules, tested with injected randomness so every branch is deterministic."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pytest

from ba_clustering.bat_algorithm import (
    BatState,
    IterationDraws,
    draw_iteration_randomness,
    frequency_update,
    initialize_state,
    local_search_candidate,
    loudness_update,
    position_update,
    pulse_rate_update,
    run_iteration,
    should_accept,
    velocity_update,
)
from ba_clustering.boundaries import Bounds, compute_bounds
from ba_clustering.fitness import sse_unchecked
from ba_clustering.initialization import initial_pulse_rates
from ba_clustering.validation import FloatArray

# --- Pure equations ---------------------------------------------------------------


@pytest.mark.parametrize(("beta", "expected"), [(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)])
def test_frequency_formula(beta: float, expected: float) -> None:
    assert frequency_update(1.0, 3.0, beta) == pytest.approx(expected)


def test_velocity_sign_convention_from_the_brief() -> None:
    velocity = velocity_update(
        np.array([0.0, 0.0]), np.array([2.0, 8.0]), np.array([1.5, 8.5]), 1.0
    )
    np.testing.assert_allclose(velocity, [0.5, -0.5])


def test_position_update_from_the_brief() -> None:
    proposed = position_update(np.array([2.0, 8.0]), np.array([0.5, -0.5]))
    np.testing.assert_allclose(proposed, [2.5, 7.5])


def test_velocity_accumulates_previous_velocity() -> None:
    velocity = velocity_update(np.array([1.0]), np.array([3.0]), np.array([1.0]), 0.5)
    np.testing.assert_allclose(velocity, [2.0])


def test_the_global_move_points_away_from_best() -> None:
    # Documents the consequence of the specified sign. See ADR-001.
    position, best = np.array([2.0]), np.array([1.0])
    velocity = velocity_update(np.array([0.0]), position, best, 1.0)
    proposed = position_update(position, velocity)
    assert abs(proposed[0] - best[0]) > abs(position[0] - best[0])


def test_local_search_formula() -> None:
    candidate = local_search_candidate(np.array([1.0, 2.0]), np.array([0.5, -1.0]), 0.5)
    np.testing.assert_allclose(candidate, [1.25, 1.5])


@pytest.mark.parametrize(
    ("candidate", "current", "draw", "loudness", "expected"),
    [
        (1.0, 2.0, 0.1, 0.5, True),  # better and quiet draw
        (1.0, 2.0, 0.9, 0.5, False),  # better but draw fails
        (3.0, 2.0, 0.1, 0.5, False),  # worse
        (2.0, 2.0, 0.1, 0.5, False),  # equal is not strictly better
        (1.0, 2.0, 0.5, 0.5, False),  # draw equal to loudness is not less than
        (1.0, 2.0, 0.0, 0.0, False),  # zero loudness can never accept
    ],
)
def test_acceptance_needs_strict_improvement_and_loudness(
    candidate: float, current: float, draw: float, loudness: float, expected: bool
) -> None:
    assert should_accept(candidate, current, draw, loudness) is expected


def test_loudness_formula() -> None:
    assert loudness_update(0.5, 0.9) == pytest.approx(0.45)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9, 1.0])
def test_loudness_never_increases(alpha: float) -> None:
    loudness = 1.0
    for _ in range(20):
        updated = loudness_update(loudness, alpha)
        assert updated <= loudness
        loudness = updated


@pytest.mark.parametrize(("r0", "gamma", "t"), [(0.5, 0.9, 1), (1.0, 0.1, 7), (0.25, 2.0, 3)])
def test_pulse_rate_formula(r0: float, gamma: float, t: int) -> None:
    assert pulse_rate_update(r0, gamma, t) == pytest.approx(r0 * (1 - math.exp(-gamma * t)))


def test_pulse_schedule_is_zero_at_t_zero() -> None:
    assert pulse_rate_update(0.5, 0.9, 0) == 0.0


def test_pulse_schedule_rejects_negative_t() -> None:
    with pytest.raises(ValueError, match="t must be"):
        pulse_rate_update(0.5, 0.9, -1)


@pytest.mark.parametrize("r0", [0.0, 0.3, 1.0])
def test_pulse_schedule_stays_in_zero_to_r0_and_is_monotone(r0: float) -> None:
    values = [pulse_rate_update(r0, 0.9, t) for t in range(0, 50)]
    assert all(0.0 <= value <= r0 for value in values)
    assert all(a <= b for a, b in itertools.pairwise(values))


def test_documented_initial_pulse_convention_is_r0() -> None:
    np.testing.assert_array_equal(initial_pulse_rates(0.3, 4), [0.3] * 4)


# --- run_iteration with injected randomness ---------------------------------------


def toy_state(
    positions: list[list[list[float]]], X: FloatArray, r0: float = 0.5, A0: float = 0.5
) -> BatState:
    array = np.array(positions, dtype=float)
    return initialize_state(array, lambda p: sse_unchecked(X, p), r0, A0)


def draws(
    beta: list[float], pulse: list[float], epsilon: list[list[list[float]]], loud: list[float]
) -> IterationDraws:
    return IterationDraws(
        beta=np.array(beta),
        pulse_draw=np.array(pulse),
        epsilon=np.array(epsilon, dtype=float),
        loudness_draw=np.array(loud),
    )


ZERO_EPS = [[[0.0], [0.0]]]


def iterate(state: BatState, t: int, d: IterationDraws, X: FloatArray, cfg: Any) -> Any:
    return run_iteration(state, t, d, lambda p: sse_unchecked(X, p), cfg, compute_bounds(X))


def test_brief_trial_move_is_rejected_because_its_sse_is_worse(
    toy_X: FloatArray, config_factory: Any
) -> None:
    # Bat 0 is the brief's [2, 8]; bat 1 is the best at [1.5, 8.5]. With f = 1 the proposal
    # is [2.5, 7.5], whose SSE of 5 is worse than 2, so the accepted position must stay.
    state = toy_state([[[2.0], [8.0]], [[1.5], [8.5]]], toy_X)
    config = config_factory(population_size=2)
    d = draws([0.5, 0.0], [0.0, 0.0], [[[0.0], [0.0]], [[0.0], [0.0]]], [0.0, 0.0])

    stats = iterate(state, 1, d, toy_X, config)

    np.testing.assert_allclose(state.velocities[0], [[0.5], [-0.5]])
    np.testing.assert_array_equal(state.positions[0], [[2.0], [8.0]])
    assert state.fitness[0] == pytest.approx(2.0)
    assert stats.n_accepted == 0


def test_rejected_candidate_leaves_loudness_and_pulse_rate_unchanged(
    toy_X: FloatArray, config_factory: Any
) -> None:
    state = toy_state([[[2.0], [8.0]], [[1.5], [8.5]]], toy_X)
    d = draws([0.5, 0.0], [0.0, 0.0], [[[0.0], [0.0]], [[0.0], [0.0]]], [0.0, 0.0])
    iterate(state, 1, d, toy_X, config_factory(population_size=2))
    np.testing.assert_array_equal(state.loudness, [0.5, 0.5])
    np.testing.assert_array_equal(state.pulse_rates, [0.5, 0.5])


def test_improving_candidate_rejected_by_loudness_changes_nothing_but_velocity(
    toy_X: FloatArray, config_factory: Any
) -> None:
    # Local search proposes the global optimum, but the loudness draw fails. Even though it
    # beats the global best, it must not be recorded: only accepted solutions update the best.
    state = toy_state([[[2.0], [8.0]]], toy_X)
    d = draws([0.0], [0.9], [[[-1.0], [1.0]]], [0.9])
    iterate(state, 1, d, toy_X, config_factory(population_size=1))
    np.testing.assert_array_equal(state.positions[0], [[2.0], [8.0]])
    assert state.fitness[0] == pytest.approx(2.0)
    assert state.best_fitness == pytest.approx(2.0)
    np.testing.assert_array_equal(state.best_position, [[2.0], [8.0]])
    assert state.loudness[0] == 0.5
    assert state.pulse_rates[0] == 0.5


def test_acceptance_updates_position_fitness_loudness_pulse_and_best(
    toy_X: FloatArray, config_factory: Any
) -> None:
    # Local candidate is [2, 8] + [-1, 1] * 0.5 = [1.5, 8.5], SSE 1.0.
    state = toy_state([[[2.0], [8.0]]], toy_X)
    config = config_factory(population_size=1)
    d = draws([0.0], [0.9], [[[-1.0], [1.0]]], [0.1])
    stats = iterate(state, 3, d, toy_X, config)

    np.testing.assert_allclose(state.positions[0], [[1.5], [8.5]])
    assert state.fitness[0] == pytest.approx(1.0)
    assert state.loudness[0] == pytest.approx(0.5 * 0.9)
    assert state.pulse_rates[0] == pytest.approx(0.5 * (1 - math.exp(-0.9 * 3)))
    assert state.best_fitness == pytest.approx(1.0)
    np.testing.assert_allclose(state.best_position, [[1.5], [8.5]])
    assert (stats.n_accepted, stats.n_accepted_local, stats.n_local_candidates) == (1, 1, 1)


def test_iteration_index_enters_the_pulse_rate(toy_X: FloatArray, config_factory: Any) -> None:
    values = []
    for t in (1, 2):
        state = toy_state([[[2.0], [8.0]]], toy_X)
        d = draws([0.0], [0.9], [[[-1.0], [1.0]]], [0.1])
        iterate(state, t, d, toy_X, config_factory(population_size=1))
        values.append(state.pulse_rates[0])
    assert values[0] == pytest.approx(0.5 * (1 - math.exp(-0.9)))
    assert values[1] == pytest.approx(0.5 * (1 - math.exp(-1.8)))


def test_first_acceptance_can_lower_the_pulse_rate_below_r0(
    toy_X: FloatArray, config_factory: Any
) -> None:
    # A documented consequence of ADR-002, not a bug.
    state = toy_state([[[2.0], [8.0]]], toy_X)
    d = draws([0.0], [0.9], [[[-1.0], [1.0]]], [0.1])
    iterate(state, 1, d, toy_X, config_factory(population_size=1))
    assert state.pulse_rates[0] < 0.5


def test_accepting_a_non_best_candidate_keeps_the_global_best(
    toy_X: FloatArray, config_factory: Any
) -> None:
    # Bat 1 improves on its own SSE of 10 but does not beat the best of 1.0.
    state = toy_state([[[1.5], [8.5]], [[3.0], [7.0]]], toy_X)
    d = draws([0.0, 0.0], [0.0, 0.9], [[[0.0], [0.0]], [[1.0], [-1.0]]], [0.9, 0.1])
    stats = iterate(state, 1, d, toy_X, config_factory(population_size=2))
    assert stats.n_accepted == 1
    np.testing.assert_allclose(state.positions[1], [[2.0], [8.0]])
    assert state.fitness[1] == pytest.approx(2.0)
    assert state.best_fitness == pytest.approx(1.0)
    np.testing.assert_allclose(state.best_position, [[1.5], [8.5]])


def test_later_bats_in_the_sweep_see_the_updated_best(
    toy_X: FloatArray, config_factory: Any
) -> None:
    # Bat 0 moves to the optimum. Bat 1's local candidate is built around the *new* best.
    state = toy_state([[[2.0], [8.0]], [[3.0], [7.0]]], toy_X)
    d = draws(
        [0.0, 0.0],
        [0.9, 0.9],
        [[[-1.0], [1.0]], [[0.0], [0.0]]],
        [0.1, 0.1],
    )
    iterate(state, 1, d, toy_X, config_factory(population_size=2))
    np.testing.assert_allclose(state.positions[1], [[1.5], [8.5]])


def test_mean_loudness_is_a_snapshot_from_the_start_of_the_iteration(
    toy_X: FloatArray, config_factory: Any
) -> None:
    # Bat 0's acceptance lowers its loudness. Bat 1 must still use the snapshot mean of 0.5.
    state = toy_state([[[2.0], [8.0]], [[3.0], [7.0]]], toy_X)
    d = draws(
        [0.0, 0.0],
        [0.9, 0.9],
        [[[-1.0], [1.0]], [[0.0], [1.0]]],
        [0.1, 0.1],
    )
    iterate(state, 1, d, toy_X, config_factory(population_size=2))
    np.testing.assert_allclose(state.positions[1], [[1.5], [9.0]])


def test_local_search_uses_the_best_not_the_bats_own_position(
    toy_X: FloatArray, config_factory: Any
) -> None:
    state = toy_state([[[1.5], [8.5]], [[3.0], [7.0]]], toy_X)
    d = draws([0.0, 0.0], [0.0, 0.9], [[[0.0], [0.0]], [[0.2], [-0.2]]], [0.9, 0.1])
    iterate(state, 1, d, toy_X, config_factory(population_size=2))
    np.testing.assert_allclose(state.positions[1], [[1.6], [8.4]])


@pytest.mark.parametrize(
    ("pulse_draw", "pulse_rate", "local"), [(0.6, 0.5, True), (0.5, 0.5, False), (0.4, 0.5, False)]
)
def test_local_search_triggers_only_when_draw_exceeds_pulse_rate(
    toy_X: FloatArray, config_factory: Any, pulse_draw: float, pulse_rate: float, local: bool
) -> None:
    state = toy_state([[[2.0], [8.0]]], toy_X, r0=pulse_rate)
    d = draws([0.0], [pulse_draw], ZERO_EPS, [0.9])
    stats = iterate(state, 1, d, toy_X, config_factory(population_size=1, r0=pulse_rate))
    assert (stats.n_local_candidates == 1) is local


def test_every_bat_costs_exactly_one_evaluation(toy_X: FloatArray, config_factory: Any) -> None:
    calls = []

    def objective(position: FloatArray) -> float:
        calls.append(position.copy())
        return sse_unchecked(toy_X, position)

    state = initialize_state(np.array([[[2.0], [8.0]], [[3.0], [7.0]]]), objective, 0.5, 0.5)
    assert len(calls) == state.n_evaluations == 2
    d = draws([0.5, 0.5], [0.9, 0.1], [[[0.0], [0.0]], [[0.0], [0.0]]], [0.9, 0.9])
    run_iteration(state, 1, d, objective, config_factory(population_size=2), compute_bounds(toy_X))
    assert len(calls) == state.n_evaluations == 4


def test_objective_receives_the_clipped_candidate(toy_X: FloatArray, config_factory: Any) -> None:
    seen = []

    def objective(position: FloatArray) -> float:
        seen.append(position.copy())
        return sse_unchecked(toy_X, position)

    state = initialize_state(np.array([[[2.0], [8.0]]]), objective, 0.5, 0.5)
    d = draws([0.0], [0.9], [[[-1.0], [1.0]]], [0.9])
    config = config_factory(population_size=1, A0=1.0)
    state.loudness[:] = 1.0
    bounds = Bounds(np.array([1.5]), np.array([8.5]))
    run_iteration(state, 1, d, objective, config, bounds)
    np.testing.assert_array_equal(seen[-1], [[1.5], [8.5]])


def test_non_finite_objective_raises(config_factory: Any) -> None:
    with pytest.raises(FloatingPointError, match="Non-finite"):
        initialize_state(np.array([[[0.0]]]), lambda _p: float("nan"), 0.5, 0.5)


def test_draw_shapes_and_ranges() -> None:
    d = draw_iteration_randomness(np.random.default_rng(0), 50, (3, 2))
    assert d.beta.shape == d.pulse_draw.shape == d.loudness_draw.shape == (50,)
    assert d.epsilon.shape == (50, 3, 2)
    assert ((d.epsilon >= -1.0) & (d.epsilon < 1.0)).all()
    for values in (d.beta, d.pulse_draw, d.loudness_draw):
        assert ((values >= 0.0) & (values < 1.0)).all()


def test_documented_toy_iteration(toy_X: FloatArray, config_factory: Any) -> None:
    """The hand calculation in docs/ALGORITHM_SPEC.md, section 11. Keep the two in sync."""
    state = toy_state([[[2.0], [8.0]], [[1.2], [8.6]], [[3.0], [7.0]]], toy_X)
    assert state.fitness.tolist() == pytest.approx([2.0, 1.2, 10.0])
    assert state.best_fitness == pytest.approx(1.2)

    d = draws(
        beta=[0.5, 0.25, 0.75],
        pulse=[0.2, 0.9, 0.7],
        epsilon=[[[0.0], [0.0]], [[0.6], [-0.2]], [[0.2], [-0.2]]],
        loud=[0.1, 0.3, 0.6],
    )
    stats = iterate(state, 1, d, toy_X, config_factory(population_size=3))

    # Bat 1: global move rejected, SSE 6.8 is worse than 2.
    np.testing.assert_allclose(state.velocities[0], [[0.8], [-0.6]])
    np.testing.assert_array_equal(state.positions[0], [[2.0], [8.0]])
    assert sse_unchecked(toy_X, np.array([[2.8], [7.4]])) == pytest.approx(6.8)
    # Bat 2: local move accepted, new global best.
    np.testing.assert_allclose(state.positions[1], [[1.5], [8.5]])
    assert state.fitness[1] == pytest.approx(1.0)
    assert state.loudness[1] == pytest.approx(0.45)
    assert state.pulse_rates[1] == pytest.approx(0.5 * (1 - math.exp(-0.9)))
    assert state.pulse_rates[1] == pytest.approx(0.29672, abs=1e-5)
    # Bat 3: local candidate improves (SSE 1.04 < 10) but fails the loudness draw.
    np.testing.assert_allclose(state.velocities[2], [[2.25], [-2.25]])
    np.testing.assert_array_equal(state.positions[2], [[3.0], [7.0]])
    assert sse_unchecked(toy_X, np.array([[1.6], [8.4]])) == pytest.approx(1.04)
    assert state.fitness[2] == pytest.approx(10.0)
    # Population.
    assert state.best_fitness == pytest.approx(1.0)
    np.testing.assert_allclose(state.best_position, [[1.5], [8.5]])
    assert state.n_evaluations == 6
    assert (stats.n_local_candidates, stats.n_accepted, stats.n_accepted_local) == (2, 1, 1)
    assert state.loudness.mean() == pytest.approx(0.48333, abs=1e-4)
