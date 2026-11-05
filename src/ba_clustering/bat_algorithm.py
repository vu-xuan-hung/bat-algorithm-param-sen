"""Standard Continuous Bat Algorithm for minimisation.

This module knows nothing about clustering, files, or plotting. It minimises any
objective over arrays of a fixed shape. The update rules are small pure functions.
:func:`run_iteration` applies them for one iteration using pre-drawn randomness,
so tests can inject exact values. ``ALGORITHM_SPEC.md`` is the normative description.

Do not change an equation, the draw order, or the acceptance rule without updating
the tests and ``docs/DECISIONS.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ba_clustering.boundaries import Bounds, clip_to_bounds
from ba_clustering.config import BAConfig
from ba_clustering.initialization import initial_pulse_rates
from ba_clustering.validation import DataValidationError, FloatArray

Objective = Callable[[FloatArray], float]
StopReason = Literal["max_iterations", "patience"]


# --- Update equations -------------------------------------------------------------


def frequency_update(f_min: float, f_max: float, beta: float) -> float:
    """Eq. 3.1: ``f = f_min + (f_max - f_min) * beta``."""
    return f_min + (f_max - f_min) * beta


def velocity_update(
    velocity: FloatArray, position: FloatArray, best: FloatArray, frequency: float
) -> FloatArray:
    """Eq. 3.2: ``v_new = v + (X - X*) * f``.

    The sign is intentional and specified (DECISIONS.md, ADR-001). With ``f > 0``
    the added term points away from the best solution. Do not flip it.
    """
    return velocity + (position - best) * frequency


def position_update(position: FloatArray, velocity: FloatArray) -> FloatArray:
    """Eq. 3.3: the proposed ``X_new = X + v_new``, before clipping and acceptance."""
    return position + velocity


def local_search_candidate(
    best: FloatArray, epsilon: FloatArray, mean_loudness: float
) -> FloatArray:
    """Eq. 3.4: ``X_local = X* + epsilon * mean_loudness``."""
    return best + epsilon * mean_loudness


def should_accept(
    candidate_fitness: float, current_fitness: float, loudness_draw: float, loudness: float
) -> bool:
    """Sec. 3.5: accept only if strictly better than the bat's own fitness AND ``draw < A_i``."""
    return candidate_fitness < current_fitness and loudness_draw < loudness


def loudness_update(loudness: float, alpha: float) -> float:
    """Eq. 3.6, applied on acceptance: ``A_new = alpha * A``."""
    return alpha * loudness


def pulse_rate_update(r0: float, gamma: float, t: int) -> float:
    """Eq. 3.7, applied on acceptance: ``r_new = r0 * (1 - exp(-gamma * t))``.

    ``t`` is the 1-based iteration counter during optimisation. The formula gives
    zero at ``t = 0``, which is why ``t = 0`` is reserved for initialisation.
    """
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}.")
    return r0 * (1.0 - math.exp(-gamma * t))


# --- State and randomness ---------------------------------------------------------


@dataclass
class BatState:
    """All mutable optimiser state. ``fitness[i]`` is always the fitness of ``positions[i]``.

    Attributes:
        positions: Accepted positions, shape ``(P, *position_shape)``.
        velocities: Velocities, same shape as ``positions``.
        fitness: Accepted fitness values, shape ``(P,)``.
        loudness: Current loudness ``A_i``, shape ``(P,)``.
        pulse_rates: Current operational pulse rate ``r_i``, shape ``(P,)``.
        best_position: Best accepted position so far. Owns its memory.
        best_fitness: Fitness of ``best_position``.
        n_evaluations: Objective evaluations so far.
    """

    positions: FloatArray
    velocities: FloatArray
    fitness: FloatArray
    loudness: FloatArray
    pulse_rates: FloatArray
    best_position: FloatArray
    best_fitness: float
    n_evaluations: int


@dataclass(frozen=True)
class IterationDraws:
    """All random values for one iteration.

    Attributes:
        beta: Frequency draws in ``[0, 1)``, shape ``(P,)``.
        pulse_draw: Local-search trigger draws in ``[0, 1)``, shape ``(P,)``.
        epsilon: Perturbations in ``[-1, 1)``, shape ``(P, *position_shape)``.
        loudness_draw: Acceptance draws in ``[0, 1)``, shape ``(P,)``.
    """

    beta: FloatArray
    pulse_draw: FloatArray
    epsilon: FloatArray
    loudness_draw: FloatArray


def draw_iteration_randomness(
    rng: np.random.Generator, population_size: int, position_shape: tuple[int, ...]
) -> IterationDraws:
    """Draw one iteration's randomness in a fixed order, whichever branches later run.

    Each iteration consumes the same amount of randomness, so runs with the same
    seed share a random stream across parameter settings. Paired-seed comparison
    relies on this; see DECISIONS.md, ADR-006.
    """
    return IterationDraws(
        beta=rng.random(population_size),
        pulse_draw=rng.random(population_size),
        epsilon=rng.uniform(-1.0, 1.0, size=(population_size, *position_shape)),
        loudness_draw=rng.random(population_size),
    )


def initialize_state(
    initial_positions: FloatArray, objective: Objective, r0: float, A0: float
) -> BatState:
    """Evaluate the initial population and build the starting state.

    The best bat is the first one with the minimum fitness. Velocities start at zero.
    """
    population_size = initial_positions.shape[0]
    positions = initial_positions.copy()
    fitness = np.array([_evaluate(objective, position) for position in positions], dtype=float)
    best_index = int(np.argmin(fitness))
    return BatState(
        positions=positions,
        velocities=np.zeros_like(positions),
        fitness=fitness,
        loudness=np.full(population_size, float(A0)),
        pulse_rates=initial_pulse_rates(r0, population_size),
        best_position=positions[best_index].copy(),
        best_fitness=float(fitness[best_index]),
        n_evaluations=population_size,
    )


def _evaluate(objective: Objective, position: FloatArray) -> float:
    value = float(objective(position))
    if not math.isfinite(value):
        raise FloatingPointError(
            f"The objective returned {value}. Non-finite fitness is never treated as a valid "
            "solution; check the data scale and the velocity magnitudes."
        )
    return value


# --- One iteration ----------------------------------------------------------------


@dataclass(frozen=True)
class IterationStats:
    """Per-iteration counters."""

    n_local_candidates: int
    n_accepted: int
    n_accepted_local: int
    n_clipped_coordinates: int


def run_iteration(
    state: BatState,
    t: int,
    draws: IterationDraws,
    objective: Objective,
    config: BAConfig,
    bounds: Bounds,
) -> IterationStats:
    """Run iteration ``t`` (1-based), updating ``state`` in place.

    Bats are processed in index order, and an accepted improvement is visible to
    later bats in the same iteration. ``mean_loudness`` is a snapshot taken once at
    the start of the iteration.
    """
    mean_loudness = float(state.loudness.mean())
    n_local = n_accepted = n_accepted_local = n_clipped = 0

    for i in range(state.positions.shape[0]):
        frequency = frequency_update(config.f_min, config.f_max, float(draws.beta[i]))
        # Velocity comes from the accepted position and persists even if the candidate
        # is rejected.
        state.velocities[i] = velocity_update(
            state.velocities[i], state.positions[i], state.best_position, frequency
        )
        proposal = position_update(state.positions[i], state.velocities[i])

        # Exactly one candidate per bat per iteration: local search replaces the global move.
        is_local = bool(draws.pulse_draw[i] > state.pulse_rates[i])
        if is_local:
            n_local += 1
            proposal = local_search_candidate(state.best_position, draws.epsilon[i], mean_loudness)

        candidate, clipped = clip_to_bounds(proposal, bounds)
        n_clipped += clipped
        candidate_fitness = _evaluate(objective, candidate)
        state.n_evaluations += 1

        if should_accept(
            candidate_fitness,
            float(state.fitness[i]),
            float(draws.loudness_draw[i]),
            float(state.loudness[i]),
        ):
            n_accepted += 1
            n_accepted_local += int(is_local)
            state.positions[i] = candidate
            state.fitness[i] = candidate_fitness
            state.loudness[i] = loudness_update(float(state.loudness[i]), config.alpha)
            state.pulse_rates[i] = pulse_rate_update(config.r0, config.gamma, t)
            if candidate_fitness < state.best_fitness:
                state.best_position = candidate.copy()
                state.best_fitness = candidate_fitness

    return IterationStats(n_local, n_accepted, n_accepted_local, n_clipped)


# --- Full run ---------------------------------------------------------------------


@dataclass
class ConvergenceHistory:
    """Per-iteration trace.

    Row 0 is the state after initialisation; row ``t`` is the state after iteration ``t``.

    Attributes:
        iteration: Iteration index.
        n_evaluations: Cumulative objective evaluations.
        best_fitness: Best-so-far fitness; never increases.
        mean_fitness: Mean accepted fitness of the population; never increases, because
            each bat's accepted fitness only ever improves.
        mean_loudness: Mean ``A_i`` after the iteration.
        mean_pulse_rate: Mean ``r_i`` after the iteration.
        n_local_candidates: Bats that used local search in the iteration.
        n_accepted: Candidates accepted in the iteration.
        n_accepted_local: Accepted candidates that came from local search.
        n_clipped_coordinates: Candidate coordinates that were out of bounds.
        max_abs_velocity: Largest absolute velocity component after the iteration.
    """

    iteration: list[int] = field(default_factory=list)
    n_evaluations: list[int] = field(default_factory=list)
    best_fitness: list[float] = field(default_factory=list)
    mean_fitness: list[float] = field(default_factory=list)
    mean_loudness: list[float] = field(default_factory=list)
    mean_pulse_rate: list[float] = field(default_factory=list)
    n_local_candidates: list[int] = field(default_factory=list)
    n_accepted: list[int] = field(default_factory=list)
    n_accepted_local: list[int] = field(default_factory=list)
    n_clipped_coordinates: list[int] = field(default_factory=list)
    max_abs_velocity: list[float] = field(default_factory=list)

    def record(self, iteration: int, state: BatState, stats: IterationStats) -> None:
        """Append one row from the current state and counters."""
        self.iteration.append(iteration)
        self.n_evaluations.append(state.n_evaluations)
        self.best_fitness.append(state.best_fitness)
        self.mean_fitness.append(float(state.fitness.mean()))
        self.mean_loudness.append(float(state.loudness.mean()))
        self.mean_pulse_rate.append(float(state.pulse_rates.mean()))
        self.n_local_candidates.append(stats.n_local_candidates)
        self.n_accepted.append(stats.n_accepted)
        self.n_accepted_local.append(stats.n_accepted_local)
        self.n_clipped_coordinates.append(stats.n_clipped_coordinates)
        self.max_abs_velocity.append(float(np.abs(state.velocities).max()))

    def to_dict(self) -> dict[str, list[int] | list[float]]:
        """Return the trace as a JSON-serialisable dictionary of columns."""
        return dict(self.__dict__)


@dataclass(frozen=True)
class BAResult:
    """Outcome of :func:`run_bat_algorithm`.

    Attributes:
        best_position: Best accepted position.
        best_fitness: Objective value at ``best_position``.
        history: Convergence trace.
        final_state: State after the last iteration.
        n_iterations: Iterations actually run.
        stop_reason: ``"max_iterations"`` or ``"patience"``. Stopping on patience means
            improvement stalled. It is not evidence of a global optimum.
    """

    best_position: FloatArray
    best_fitness: float
    history: ConvergenceHistory
    final_state: BatState
    n_iterations: int
    stop_reason: StopReason


def run_bat_algorithm(
    objective: Objective,
    initial_positions: FloatArray,
    bounds: Bounds,
    config: BAConfig,
    rng: np.random.Generator,
) -> BAResult:
    """Minimise ``objective`` starting from ``initial_positions``.

    Args:
        objective: Pure function mapping one position to a finite float.
        initial_positions: Array of shape ``(population_size, *position_shape)``.
        bounds: Coordinate limits broadcast against each position.
        config: Algorithm parameters.
        rng: Random generator; advanced in place.

    Returns:
        The best position, its fitness, the trace, and the final state.

    Raises:
        DataValidationError: If ``initial_positions`` has the wrong population axis.
        FloatingPointError: If the objective returns a non-finite value.
    """
    if initial_positions.ndim < 2 or initial_positions.shape[0] != config.population_size:
        raise DataValidationError(
            f"initial_positions must have shape ({config.population_size}, ...), "
            f"got {initial_positions.shape}."
        )
    position_shape = initial_positions.shape[1:]
    state = initialize_state(initial_positions, objective, config.r0, config.A0)
    history = ConvergenceHistory()
    history.record(0, state, IterationStats(0, 0, 0, 0))

    reference_best = state.best_fitness
    stalled_iterations = 0
    stop_reason: StopReason = "max_iterations"
    n_iterations = 0

    for t in range(1, config.max_iterations + 1):
        draws = draw_iteration_randomness(rng, config.population_size, position_shape)
        stats = run_iteration(state, t, draws, objective, config, bounds)
        history.record(t, state, stats)
        n_iterations = t

        # Compare with the best at the last meaningful improvement, so that a slow drift
        # of tiny improvements cannot keep resetting the patience counter.
        if reference_best - state.best_fitness > config.tolerance:
            reference_best = state.best_fitness
            stalled_iterations = 0
        else:
            stalled_iterations += 1
        if config.patience is not None and stalled_iterations >= config.patience:
            stop_reason = "patience"
            break

    return BAResult(
        best_position=state.best_position.copy(),
        best_fitness=state.best_fitness,
        history=history,
        final_state=state,
        n_iterations=n_iterations,
        stop_reason=stop_reason,
    )
