# API reference

Snapshot of the implementation verified on 2026-11-05. Signatures below are read from
Python introspection, not invented interfaces. Annotations use FloatArray=float64 ndarray,
Labels=intp ndarray, Objective=Callable[[FloatArray],float], and Path=pathlib.Path.
Ellipsis/default factory notation follows inspect.signature; dataclass constructors are
shown with generated fields. Public module-level helpers are included even when not
re-exported by ba_clustering.__all__. Helpers named with a leading underscore are internal.
Low-level pure functions assume valid typed inputs; use BatClustering/compute_sse for
validated inputs. See DATA_CONTRACT.md for shapes, errors and limitations.

## Working example

```python
import numpy as np
from ba_clustering import BAConfig, BatClustering, compute_sse
X = np.array([[1.0], [2.0], [8.0], [9.0]])
config = BAConfig(population_size=5, max_iterations=3, n_clusters=2,
                  A0=0.5, r0=0.5, alpha=0.9, gamma=0.9,
                  f_min=0.0, f_max=2.0, seed=0)
model = BatClustering(config).fit(X)
assert compute_sse(X, model.centroids_) == model.best_fitness_
assert model.predict(X).shape == (4,)
```

BatClustering does not scale X. Fitted properties: centroids_ (read-only), labels_,
best_fitness_, history_, result_, n_evaluations_, n_iterations_, stop_reason_,
n_empty_clusters_, bounds_, runtime_seconds_. predict uses centroid feature units.
A repeated fit restarts the seed. result_ exposes mutable final state for advanced users;
modify neither it nor history if relying on fitted invariants.

```python
from ba_clustering import KMeansConfig, run_kmeans, preprocess
scaled, info = preprocess(X, "zscore")
km = run_kmeans(scaled, KMeansConfig(n_clusters=2, seed=0, n_init=3,
                                   init="k-means++", max_iter=100, tol=1e-4))
assert km.sse == compute_sse(scaled, km.centroids)
```

## Experiment and result use

```python
from ba_clustering.experiments import load_experiment_spec, expand_conditions
spec = load_experiment_spec("configs/experiments.yaml", "smoke")
assert len(expand_conditions(spec)) == 3
```

run_experiment(spec, new_output_root) writes files; resume/overwrite are explicit keyword
arguments. Use a fresh temporary root for tests. generate_plots(directory) regenerates
figures from saved data. CLI main(argv) returns 0 on success, 1 for recorded experiment
failures, 2 for recognized user errors. Unexpected programming/numerical exceptions may
propagate. All detailed signatures and source docstrings follow.

## ba_clustering.baseline

Conventional K-Means baseline using scikit-learn's Lloyd algorithm.

Restarts are run one at a time rather than through ``n_init``, because scikit-learn
reports the iteration count of the best restart only. Counting every restart is
needed to report the computational cost honestly.

### KMeansConfig

```python
KMeansConfig(*, n_clusters: 'int', seed: 'int', n_init: 'int', init: "Literal['k-means++', 'random']", max_iter: 'int', tol: 'float') -> None
```

K-Means settings.

Attributes:
    n_clusters: Fixed ``K``.
    seed: Non-negative seed from which all restart seeds are derived.
    n_init: Number of independent restarts; the lowest-SSE restart is kept.
    init: ``"k-means++"`` or ``"random"``.
    max_iter: Maximum Lloyd iterations per restart.
    tol: scikit-learn convergence tolerance.

```python
KMeansConfig.to_dict(self) -> 'dict[str, Any]'
```

Return all fields as a JSON-serialisable dictionary.

### KMeansResult

```python
KMeansResult(centroids: 'FloatArray', labels: 'NDArray[np.intp]', sse: 'float', restart_sse: 'FloatArray', restart_seeds: 'list[int]', n_lloyd_iterations: 'int', runtime_seconds: 'float') -> None
```

Outcome of :func:`run_kmeans`.

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

### run_kmeans

```python
run_kmeans(X: 'object', config: 'KMeansConfig') -> 'KMeansResult'
```

Run multi-start K-Means on ``X`` and keep the restart with the lowest SSE.

Ties go to the earliest restart.

Raises:
    DataValidationError: If ``X`` violates the data contract.
    ConfigurationError: If ``n_clusters`` exceeds ``n_samples``.

## ba_clustering.bat_algorithm

Standard Continuous Bat Algorithm for minimisation.

This module knows nothing about clustering, files, or plotting. It minimises any
objective over arrays of a fixed shape. The update rules are small pure functions.
:func:`run_iteration` applies them for one iteration using pre-drawn randomness,
so tests can inject exact values. ``ALGORITHM_SPEC.md`` is the normative description.

Do not change an equation, the draw order, or the acceptance rule without updating
the tests and ``docs/DECISIONS.md``.

### frequency_update

```python
frequency_update(f_min: 'float', f_max: 'float', beta: 'float') -> 'float'
```

Eq. 3.1: ``f = f_min + (f_max - f_min) * beta``.

### velocity_update

```python
velocity_update(velocity: 'FloatArray', position: 'FloatArray', best: 'FloatArray', frequency: 'float') -> 'FloatArray'
```

Eq. 3.2: ``v_new = v + (X - X*) * f``.

The sign is intentional and specified (DECISIONS.md, ADR-001). With ``f > 0``
the added term points away from the best solution. Do not flip it.

### position_update

```python
position_update(position: 'FloatArray', velocity: 'FloatArray') -> 'FloatArray'
```

Eq. 3.3: the proposed ``X_new = X + v_new``, before clipping and acceptance.

### local_search_candidate

```python
local_search_candidate(best: 'FloatArray', epsilon: 'FloatArray', mean_loudness: 'float') -> 'FloatArray'
```

Eq. 3.4: ``X_local = X* + epsilon * mean_loudness``.

### should_accept

```python
should_accept(candidate_fitness: 'float', current_fitness: 'float', loudness_draw: 'float', loudness: 'float') -> 'bool'
```

Sec. 3.5: accept only if strictly better than the bat's own fitness AND ``draw < A_i``.

### loudness_update

```python
loudness_update(loudness: 'float', alpha: 'float') -> 'float'
```

Eq. 3.6, applied on acceptance: ``A_new = alpha * A``.

### pulse_rate_update

```python
pulse_rate_update(r0: 'float', gamma: 'float', t: 'int') -> 'float'
```

Eq. 3.7, applied on acceptance: ``r_new = r0 * (1 - exp(-gamma * t))``.

``t`` is the 1-based iteration counter during optimisation. The formula gives
zero at ``t = 0``, which is why ``t = 0`` is reserved for initialisation.

### BatState

```python
BatState(positions: 'FloatArray', velocities: 'FloatArray', fitness: 'FloatArray', loudness: 'FloatArray', pulse_rates: 'FloatArray', best_position: 'FloatArray', best_fitness: 'float', n_evaluations: 'int') -> None
```

All mutable optimiser state. ``fitness[i]`` is always the fitness of ``positions[i]``.

Attributes:
    positions: Accepted positions, shape ``(P, *position_shape)``.
    velocities: Velocities, same shape as ``positions``.
    fitness: Accepted fitness values, shape ``(P,)``.
    loudness: Current loudness ``A_i``, shape ``(P,)``.
    pulse_rates: Current operational pulse rate ``r_i``, shape ``(P,)``.
    best_position: Best accepted position so far. Owns its memory.
    best_fitness: Fitness of ``best_position``.
    n_evaluations: Objective evaluations so far.

### IterationDraws

```python
IterationDraws(beta: 'FloatArray', pulse_draw: 'FloatArray', epsilon: 'FloatArray', loudness_draw: 'FloatArray') -> None
```

All random values for one iteration.

Attributes:
    beta: Frequency draws in ``[0, 1)``, shape ``(P,)``.
    pulse_draw: Local-search trigger draws in ``[0, 1)``, shape ``(P,)``.
    epsilon: Perturbations in ``[-1, 1)``, shape ``(P, *position_shape)``.
    loudness_draw: Acceptance draws in ``[0, 1)``, shape ``(P,)``.

### draw_iteration_randomness

```python
draw_iteration_randomness(rng: 'np.random.Generator', population_size: 'int', position_shape: 'tuple[int, ...]') -> 'IterationDraws'
```

Draw one iteration's randomness in a fixed order, whichever branches later run.

Each iteration consumes the same amount of randomness, so runs with the same
seed share a random stream across parameter settings. Paired-seed comparison
relies on this; see DECISIONS.md, ADR-006.

### initialize_state

```python
initialize_state(initial_positions: 'FloatArray', objective: 'Objective', r0: 'float', A0: 'float') -> 'BatState'
```

Evaluate the initial population and build the starting state.

The best bat is the first one with the minimum fitness. Velocities start at zero.

### IterationStats

```python
IterationStats(n_local_candidates: 'int', n_accepted: 'int', n_accepted_local: 'int', n_clipped_coordinates: 'int') -> None
```

Per-iteration counters.

### run_iteration

```python
run_iteration(state: 'BatState', t: 'int', draws: 'IterationDraws', objective: 'Objective', config: 'BAConfig', bounds: 'Bounds') -> 'IterationStats'
```

Run iteration ``t`` (1-based), updating ``state`` in place.

Bats are processed in index order, and an accepted improvement is visible to
later bats in the same iteration. ``mean_loudness`` is a snapshot taken once at
the start of the iteration.

### ConvergenceHistory

```python
ConvergenceHistory(iteration: 'list[int]' = <factory>, n_evaluations: 'list[int]' = <factory>, best_fitness: 'list[float]' = <factory>, mean_fitness: 'list[float]' = <factory>, mean_loudness: 'list[float]' = <factory>, mean_pulse_rate: 'list[float]' = <factory>, n_local_candidates: 'list[int]' = <factory>, n_accepted: 'list[int]' = <factory>, n_accepted_local: 'list[int]' = <factory>, n_clipped_coordinates: 'list[int]' = <factory>, max_abs_velocity: 'list[float]' = <factory>) -> None
```

Per-iteration trace.

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

```python
ConvergenceHistory.record(self, iteration: 'int', state: 'BatState', stats: 'IterationStats') -> 'None'
```

Append one row from the current state and counters.

```python
ConvergenceHistory.to_dict(self) -> 'dict[str, list[int] | list[float]]'
```

Return the trace as a JSON-serialisable dictionary of columns.

### BAResult

```python
BAResult(best_position: 'FloatArray', best_fitness: 'float', history: 'ConvergenceHistory', final_state: 'BatState', n_iterations: 'int', stop_reason: 'StopReason') -> None
```

Outcome of :func:`run_bat_algorithm`.

Attributes:
    best_position: Best accepted position.
    best_fitness: Objective value at ``best_position``.
    history: Convergence trace.
    final_state: State after the last iteration.
    n_iterations: Iterations actually run.
    stop_reason: ``"max_iterations"`` or ``"patience"``. Stopping on patience means
        improvement stalled. It is not evidence of a global optimum.

### run_bat_algorithm

```python
run_bat_algorithm(objective: 'Objective', initial_positions: 'FloatArray', bounds: 'Bounds', config: 'BAConfig', rng: 'np.random.Generator') -> 'BAResult'
```

Minimise ``objective`` starting from ``initial_positions``.

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

## ba_clustering.boundaries

Coordinate bounds for centroids, and the clipping boundary policy.

Bounds are the per-feature min and max of the preprocessed data. Candidates are
clipped coordinate-wise. Velocities are never changed by clipping
(DECISIONS.md, ADR-005).

### Bounds

```python
Bounds(lower: 'FloatArray', upper: 'FloatArray') -> None
```

Per-feature coordinate limits, each of shape ``(n_features,)``.

Property `zero_range_features`: Indices of features where lower equals upper; those coordinates are pinned.

### compute_bounds

```python
compute_bounds(X: 'object') -> 'Bounds'
```

Return the per-feature min and max of ``X`` as :class:`Bounds`.

### clip_to_bounds

```python
clip_to_bounds(position: 'FloatArray', bounds: 'Bounds') -> 'tuple[FloatArray, int]'
```

Clip a position to ``bounds`` and report how many coordinates were outside.

Args:
    position: Array whose last axis has length ``n_features``, e.g. ``(K, D)``.
    bounds: Limits broadcast against ``position``.

Returns:
    A new clipped array, and the number of coordinates that were out of range.
    NaN coordinates are preserved, so the caller can detect them.

## ba_clustering.cli

Command-line interface. Run ``ba-clustering --help`` for usage.

Exit codes: 0 success; 1 at least one experiment run failed; 2 invalid input.

### build_parser

```python
build_parser() -> 'argparse.ArgumentParser'
```

Return the argument parser.

### main

```python
main(argv: 'Sequence[str] | None' = None) -> 'int'
```

Run the CLI and return an exit code.

## ba_clustering.clustering

Cluster assignment and the :class:`BatClustering` estimator.

### assign_labels

```python
assign_labels(X: 'object', centroids: 'object') -> 'Labels'
```

Return the index of the nearest centroid for each sample, shape ``(n_samples,)``.

Distance is squared Euclidean. Ties go to the lowest centroid index.

### cluster_sizes

```python
cluster_sizes(labels: 'Labels', n_clusters: 'int') -> 'NDArray[np.intp]'
```

Return the number of samples per cluster, shape ``(n_clusters,)``.

### count_empty_clusters

```python
count_empty_clusters(labels: 'Labels', n_clusters: 'int') -> 'int'
```

Return how many of the ``n_clusters`` centroids own no samples.

### BatClustering

```python
BatClustering(config: 'BAConfig') -> 'None'
```

Optimise ``K`` centroids with the Standard Continuous Bat Algorithm, minimising SSE.

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

```python
BatClustering.fit(self, X: 'object') -> 'Self'
```

Optimise centroids on ``X``, shape ``(n_samples, n_features)``.

Raises:
    DataValidationError: If ``X`` violates the data contract.
    ConfigurationError: If ``n_clusters`` exceeds ``n_samples``.
    FloatingPointError: If the objective becomes non-finite.

```python
BatClustering.predict(self, X: 'object') -> 'Labels'
```

Return nearest-centroid labels for ``X`` using the learned centroids.

Raises:
    NotFittedError: If called before :meth:`fit`.
    DataValidationError: If ``X`` is invalid or has a different feature count.

```python
BatClustering.fit_predict(self, X: 'object') -> 'Labels'
```

Fit on ``X`` and return the labels of ``X``.

Property `centroids_`: Best accepted centroids, read-only, shape ``(K, D)``.

Property `best_fitness_`: SSE of :attr:`centroids_` on the training data.

Property `history_`: Convergence trace.

Property `result_`: Full optimiser result, including the final population state.

Property `labels_`: Training-data labels under :attr:`centroids_`.

Property `n_evaluations_`: Objective evaluations used by the last fit.

Property `n_iterations_`: Iterations run by the last fit.

Property `stop_reason_`: Why the last fit stopped.

Property `n_empty_clusters_`: Centroids in :attr:`centroids_` that own no training samples.

Property `bounds_`: Coordinate bounds computed from the training data.

Property `runtime_seconds_`: Wall-clock optimisation time of the last fit.

## ba_clustering.config

Strictly validated configuration for the Bat Algorithm.

Reference parameter values live in ``configs/default.yaml``, not here. Only the
stopping options have defaults, and those defaults mean "fixed iteration budget".

### BAConfig

```python
BAConfig(*, population_size: 'int', max_iterations: 'int', n_clusters: 'int', A0: 'float', r0: 'float', alpha: 'float', gamma: 'float', f_min: 'float', f_max: 'float', seed: 'int', tolerance: 'float' = 0.0, patience: 'int | None' = None) -> None
```

Parameters of one Bat Algorithm clustering run.

Attributes:
    population_size: Number of bats, ``P >= 1``.
    max_iterations: Iteration budget, ``T >= 1``.
    n_clusters: Number of centroids per bat, ``K >= 1``.
    A0: Initial loudness of every bat, in ``[0, 1]``.
    r0: Reference pulse-rate parameter in ``[0, 1]``; also the initial
        operational pulse rate of every bat. See DECISIONS.md, ADR-002.
    alpha: Loudness reduction coefficient, in ``(0, 1]``.
    gamma: Pulse-rate growth coefficient, ``> 0``.
    f_min: Minimum frequency, ``>= 0``.
    f_max: Maximum frequency, ``>= f_min``.
    seed: Non-negative seed for ``numpy.random.default_rng``.
    tolerance: Best-fitness improvement that counts as meaningful, ``>= 0``.
        Values above zero require ``patience``.
    patience: Consecutive iterations without meaningful improvement before
        stopping, or ``None`` for the fixed budget.

```python
BAConfig.from_mapping(cls, values: 'Mapping[str, Any]') -> 'BAConfig'
```

Build a config from a mapping, rejecting unknown and missing keys.

```python
BAConfig.to_dict(self) -> 'dict[str, Any]'
```

Return all fields as a JSON-serialisable dictionary.

Property `max_evaluations`: Objective evaluations in a full run: ``P`` initial plus ``P`` per iteration.

## ba_clustering.datasets

Dataset loading. No function here needs network access.

External class labels are kept in a separate field and are used only for
explicitly identified evaluation metrics. They never reach an optimiser.

### Dataset

```python
Dataset(name: 'str', X: 'FloatArray', n_clusters: 'int', source: 'str', labels_external: 'NDArray[Any] | None' = None, generator: 'Mapping[str, Any] | None' = None) -> None
```

Raw, unscaled data plus metadata.

Attributes:
    name: Dataset identifier.
    X: Feature matrix of shape ``(n_samples, n_features)``.
    n_clusters: The fixed ``K`` to use for this dataset.
    source: Human-readable provenance.
    labels_external: Optional labels for evaluation only.
    generator: Parameters and seed of a synthetic generator, if any.

```python
Dataset.metadata(self) -> 'dict[str, Any]'
```

Return a JSON-serialisable description of the dataset.

### load_iris_dataset

```python
load_iris_dataset(n_clusters: 'int' = 3) -> 'Dataset'
```

Load Iris (150 samples, 4 features) from the copy bundled with scikit-learn.

### load_wine_dataset

```python
load_wine_dataset(n_clusters: 'int' = 3) -> 'Dataset'
```

Load Wine (178 samples, 13 features) from the copy bundled with scikit-learn.

### make_synthetic_dataset

```python
make_synthetic_dataset(n_clusters: 'int', n_samples: 'int', n_features: 'int', cluster_std: 'float', seed: 'int') -> 'Dataset'
```

Generate Gaussian blobs.

``seed`` controls only the data. It is independent of the algorithm seeds, so all
runs of an experiment cluster the same points.

### load_csv_dataset

```python
load_csv_dataset(path: 'str | Path', n_clusters: 'int', label_column: 'str | None' = None, name: 'str | None' = None) -> 'Dataset'
```

Load a numeric CSV file with a header row.

Missing values and non-numeric feature columns are rejected, never imputed.

Args:
    path: CSV file path.
    n_clusters: The fixed ``K``.
    label_column: Column to set aside as evaluation-only labels.
    name: Dataset name; defaults to the file stem.

Raises:
    FileNotFoundError: If the file does not exist.
    DataValidationError: If the content violates the data contract.

### load_dataset

```python
load_dataset(spec: 'Mapping[str, Any]') -> 'Dataset'
```

Build a dataset from a config mapping with a ``loader`` key.

Supported loaders: ``iris``, ``wine``, ``synthetic``, ``csv``. Remaining keys are
passed to the loader; unknown keys raise :class:`ConfigurationError`.

## ba_clustering.experiments

Controlled experiments: baseline comparisons, OFAT sweeps, and factorial grids.

An experiment is declared in ``configs/experiments.yaml`` and expanded into runs. Each
run is fully determined by its dataset, algorithm, configuration and seed, so runs can
be repeated or resumed independently. See ``docs/EXPERIMENT_DESIGN.md``.

### ExperimentSpec

```python
ExperimentSpec(experiment_id: 'str', description: 'str', design: 'str', datasets: 'Mapping[str, Mapping[str, Any]]', preprocessing_method: 'str', seeds: 'tuple[int, ...]', algorithms: 'tuple[str, ...]', ba_base: 'Mapping[str, Any]', kmeans: 'Mapping[str, Any]', factors: 'Mapping[str, tuple[float, ...]]') -> None
```

A fully resolved experiment definition.

Attributes:
    experiment_id: Identifier, also the output directory name.
    description: Purpose of the experiment.
    design: ``"baseline"``, ``"ofat"`` or ``"grid"``.
    datasets: Dataset name mapped to its loader options.
    preprocessing_method: Preprocessing shared by all algorithms.
    seeds: Algorithm seeds.
    algorithms: Subset of ``("ba", "kmeans")``.
    ba_base: BA parameters except ``n_clusters`` and ``seed``.
    kmeans: K-Means settings except ``n_clusters`` and ``seed``.
    factors: Swept parameter mapped to its levels.

```python
ExperimentSpec.to_dict(self) -> 'dict[str, Any]'
```

Return a JSON-serialisable snapshot of the definition.

### Condition

```python
Condition(condition_id: 'str', overrides: 'Mapping[str, float]') -> None
```

One BA parameter setting. ``overrides`` is empty for the reference condition.

### PreparedDataset

```python
PreparedDataset(dataset: 'Dataset', X: 'FloatArray', preprocessing: 'PreprocessingInfo', sha256: 'str') -> None
```

A dataset and the single preprocessed matrix every algorithm uses.

### ExperimentOutcome

```python
ExperimentOutcome(directory: 'Path', n_completed: 'int', n_failed: 'int', n_skipped: 'int') -> None
```

Summary of a finished experiment.

### load_base_config

```python
load_base_config(path: 'str | Path') -> 'dict[str, Any]'
```

Load and check the top-level structure of ``configs/default.yaml``.

### list_experiments

```python
list_experiments(path: 'str | Path') -> 'dict[str, str]'
```

Return experiment IDs and descriptions from an experiments file.

### load_experiment_spec

```python
load_experiment_spec(path: 'str | Path', experiment_id: 'str') -> 'ExperimentSpec'
```

Resolve one experiment from an experiments file and its base config.

Raises:
    ConfigurationError: For any unknown key, unknown reference, or invalid design.

### expand_conditions

```python
expand_conditions(spec: 'ExperimentSpec') -> 'list[Condition]'
```

Expand the design into BA conditions, with the reference always included.

A level equal to the reference value is the reference condition itself, so it is
not run twice. OFAT varies one parameter at a time, holding the rest at reference.

Raises:
    ConfigurationError: If the factors do not suit the design.

### build_ba_config

```python
build_ba_config(spec: 'ExperimentSpec', condition: 'Condition', n_clusters: 'int', seed: 'int') -> 'BAConfig'
```

Return the validated BA config for one run.

### prepare_dataset

```python
prepare_dataset(options: 'Mapping[str, Any]', preprocessing_method: 'str') -> 'PreparedDataset'
```

Load one dataset and apply the shared preprocessing.

### run_ba_once

```python
run_ba_once(prepared: 'PreparedDataset', config: 'BAConfig') -> 'dict[str, Any]'
```

Run BA once and return its metric fields.

### run_kmeans_once

```python
run_kmeans_once(prepared: 'PreparedDataset', config: 'KMeansConfig') -> 'dict[str, Any]'
```

Run K-Means once and return its metric fields.

### run_experiment

```python
run_experiment(spec: 'ExperimentSpec', output_root: 'str | Path', *, overwrite: 'bool' = False, resume: 'bool' = False, progress: 'Callable[[str], None] | None' = None) -> 'ExperimentOutcome'
```

Run an experiment and write its outputs to ``output_root/<experiment_id>/``.

Existing results are never replaced without ``overwrite``. With ``resume``, completed
runs are kept, and failed or missing runs are executed. A run that fails on invalid
data or a numerical error is recorded as failed, and the experiment continues;
programming errors propagate.

Raises:
    ResultsExistError: If results exist and no option was given.
    ResultsMismatchError: If resuming results from a different definition.
    ConfigurationError: If any condition is invalid. This is checked before any run.

### seeds_overlap

```python
seeds_overlap(first: 'Sequence[int]', second: 'Sequence[int]') -> 'list[int]'
```

Return seeds present in both rosters. Discovery and confirmation must not overlap.

## ba_clustering.fitness

Sum of Squared Errors, the only primary optimisation objective.

``SSE = sum_n min_k ||x_n - mu_k||_2^2``

These functions are pure: they never modify the data or the centroids. A centroid
that owns no samples simply contributes nothing, so SSE stays defined for every
finite centroid configuration (DECISIONS.md, ADR-004).

### squared_distances

```python
squared_distances(X: 'FloatArray', centroids: 'FloatArray') -> 'FloatArray'
```

Return squared Euclidean distances, shape ``(n_samples, K)``. Inputs are not validated.

### sse_unchecked

```python
sse_unchecked(X: 'FloatArray', centroids: 'FloatArray') -> 'float'
```

Return total SSE without validating inputs. For already-validated hot paths.

### compute_sse

```python
compute_sse(X: 'object', centroids: 'object') -> 'float'
```

Return the total SSE of ``X`` under ``centroids``.

Args:
    X: Data of shape ``(n_samples, n_features)``.
    centroids: Centroids of shape ``(K, n_features)``.

Returns:
    A finite, non-negative float.

Raises:
    DataValidationError: If either input violates the data contract.
    FloatingPointError: If finite inputs overflow during the computation.

## ba_clustering.initialization

Reproducible initial bat state.

Positions are uniform inside the data's bounding box, velocities are zero, and
every bat starts with ``A_i = A0`` and ``r_i = r0`` (DECISIONS.md, ADR-002, ADR-007).

### initialize_positions

```python
initialize_positions(rng: 'np.random.Generator', population_size: 'int', n_clusters: 'int', bounds: 'Bounds') -> 'FloatArray'
```

Return initial positions of shape ``(population_size, n_clusters, n_features)``.

### initial_pulse_rates

```python
initial_pulse_rates(r0: 'float', population_size: 'int') -> 'FloatArray'
```

Return the initial operational pulse rate of each bat, which is ``r0``.

The schedule ``r0 (1 - exp(-gamma t))`` equals zero at ``t = 0`` and is not
applied here. It runs only on acceptance, with ``t >= 1``.

## ba_clustering.metrics

Evaluation metrics and statistics. None of these is an optimisation objective.

### silhouette_or_none

```python
silhouette_or_none(X: 'FloatArray', labels: 'NDArray[np.intp]') -> 'float | None'
```

Return the mean Silhouette score, or ``None`` when it is undefined.

It is defined only for ``2 <= distinct labels <= n_samples - 1``. A run whose
centroids leave some clusters empty is scored on the labels actually used.

### adjusted_rand_or_none

```python
adjusted_rand_or_none(labels_external: 'NDArray[np.generic] | None', labels: 'NDArray[np.intp]') -> 'float | None'
```

Return the Adjusted Rand Index, or ``None`` when no external labels exist.

Evaluation only: external labels must never influence optimisation.

### evaluations_to_plateau

```python
evaluations_to_plateau(evaluations: 'Sequence[int]', best_fitness: 'Sequence[float]', tolerance: 'float' = 0.0) -> 'int'
```

Return the first evaluation count at which best fitness was within ``tolerance`` of final.

A plateau is only "no further improvement within this run's budget". It does not
imply that a global optimum was reached.

### Summary

```python
Summary(n: 'int', mean: 'float', std: 'float | None', median: 'float', q1: 'float', q3: 'float', minimum: 'float', maximum: 'float', ci95_low: 'float | None', ci95_high: 'float | None') -> None
```

Descriptive statistics of one metric over repeated runs.

``std`` uses ``ddof=1``. ``std`` and the confidence interval are ``None`` for fewer
than two values. The interval is a two-sided 95% Student-t interval on the mean.

### summarize

```python
summarize(values: 'ArrayLike') -> 'Summary'
```

Return descriptive statistics of ``values``.

Raises:
    DataValidationError: If ``values`` is empty or contains non-finite numbers.

### PairedComparison

```python
PairedComparison(n: 'int', mean_difference: 'float', median_difference: 'float', n_a_lower: 'int', n_b_lower: 'int', n_tied: 'int', p_value: 'float') -> None
```

Paired comparison of two conditions over shared seeds.

``mean_difference`` and ``median_difference`` are ``a - b``; for SSE, a negative
value means ``a`` is lower. ``p_value`` is from the two-sided Wilcoxon signed-rank
test, and is 1.0 when every difference is zero.

### paired_comparison

```python
paired_comparison(a: 'ArrayLike', b: 'ArrayLike') -> 'PairedComparison'
```

Compare paired samples ``a`` and ``b``, whose positions must correspond to the same seed.

Raises:
    DataValidationError: If shapes differ, or the input is empty or non-finite.

### holm_adjust

```python
holm_adjust(p_values: 'Sequence[float]') -> 'list[float]'
```

Return Holm step-down adjusted p-values, in the input order.

## ba_clustering.plots

Figures generated from saved results. Nothing here runs an algorithm.

Figures are written to ``<results>/figures/``, separate from the data files. The exact
numbers behind each figure are in ``summary.csv`` and ``runs.csv``.

Figures use matplotlib's object-oriented API, so no global pyplot state is touched.
Series colours follow a validated categorical order and are assigned by level order.

### plot_main_effects

```python
plot_main_effects(records: 'Records', manifest: 'Mapping[str, Any]', dataset: 'str', path: 'Path') -> 'None'
```

Small multiples of mean SSE by level for each OFAT factor, with SD error bars.

### plot_factor_convergence

```python
plot_factor_convergence(records: 'Records', manifest: 'Mapping[str, Any]', dataset: 'str', factor: 'str', path: 'Path') -> 'None'
```

Mean best-so-far SSE against objective evaluations, one line per level of ``factor``.

### plot_baseline

```python
plot_baseline(records: 'Records', dataset: 'str', path: 'Path') -> 'None'
```

Final SSE per seed for each algorithm, with the mean marked.

### interaction_matrix

```python
interaction_matrix(records: 'Records', manifest: 'Mapping[str, Any]', dataset: 'str', factors: 'Sequence[str]') -> 'tuple[list[float], list[float], np.ndarray[Any, Any]]'
```

Return row levels, column levels, and mean SSE per cell for the first two factors.

Cells with no completed runs are NaN.

### plot_interaction

```python
plot_interaction(records: 'Records', manifest: 'Mapping[str, Any]', dataset: 'str', factors: 'Sequence[str]', path: 'Path') -> 'None'
```

Heatmap of mean SSE over the first two grid factors, with every cell labelled.

### generate_plots

```python
generate_plots(results_directory: 'str | Path') -> 'list[Path]'
```

Create every figure relevant to an experiment's design and return their paths.

Existing figures are regenerated, because they derive entirely from saved data.

Raises:
    FileNotFoundError: If the directory has no manifest.

## ba_clustering.preprocessing

Feature preprocessing. Fitted once per dataset and shared by every algorithm.

Supported methods:

* ``"zscore"``: ``(x - mean) / std`` with the population standard deviation
  (``ddof=0``), equivalent to scikit-learn's ``StandardScaler``. An exactly constant
  feature becomes all zeros: it is centred and its scale is set to 1.
* ``"none"``: data is returned unchanged, as a copy.

### PreprocessingInfo

```python
PreprocessingInfo(method: 'PreprocessingMethod', mean: 'FloatArray', scale: 'FloatArray', constant_features: 'list[int]') -> None
```

Fitted preprocessing parameters.

Attributes:
    method: Method used.
    mean: Per-feature value subtracted, shape ``(n_features,)``.
    scale: Per-feature divisor, shape ``(n_features,)``.
    constant_features: Indices of features with zero range before scaling.

```python
PreprocessingInfo.to_dict(self) -> 'dict[str, Any]'
```

Return a JSON-serialisable description, for result manifests.

### fit_preprocessing

```python
fit_preprocessing(X: 'object', method: 'str' = 'zscore') -> 'PreprocessingInfo'
```

Compute preprocessing parameters from ``X``, without transforming it.

Raises:
    ConfigurationError: If ``method`` is not supported.
    DataValidationError: If ``X`` violates the data contract.

### apply_preprocessing

```python
apply_preprocessing(X: 'object', info: 'PreprocessingInfo') -> 'FloatArray'
```

Apply fitted parameters to ``X`` and return a new array.

Raises:
    DataValidationError: If ``X`` is invalid or has the wrong feature count.

### preprocess

```python
preprocess(X: 'object', method: 'str' = 'zscore') -> 'tuple[FloatArray, PreprocessingInfo]'
```

Fit preprocessing on ``X`` and return the transformed data and the fitted parameters.

## ba_clustering.results

Result storage and tabulation. All experiment file I/O lives here.

Layout of ``results/<experiment_id>/``:

* ``manifest.json``: experiment snapshot, environment, and completion status.
* ``runs.jsonl``: one complete record per run, appended as each run finishes.
* ``runs.csv``: the same records, flattened, without per-iteration histories.
* ``summary.csv``: descriptive statistics per dataset, algorithm, condition and metric.
* ``paired_vs_reference.csv``: paired SSE comparisons against the reference condition.
* ``figures/``: plots, written separately by :mod:`ba_clustering.plots`.

See ``docs/RESULTS_GUIDE.md`` for the schemas.

### ResultsExistError

```python
ResultsExistError (inherits the built-in exception constructor)
```

The output directory already holds results and no explicit option was given.

### ResultsMismatchError

```python
ResultsMismatchError (inherits the built-in exception constructor)
```

Results being resumed were produced by a different experiment definition.

### canonical_hash

```python
canonical_hash(value: 'Any') -> 'str'
```

Return the SHA-256 hex digest of ``value`` as canonical JSON.

### git_commit

```python
git_commit() -> 'str | None'
```

Return the current commit of the source tree, or ``None`` outside a git repository.

### environment_info

```python
environment_info() -> 'dict[str, Any]'
```

Return the software environment for the manifest.

### write_manifest

```python
write_manifest(directory: 'Path', manifest: 'Mapping[str, Any]') -> 'None'
```

Write the manifest atomically. Key order is preserved: factor order is meaningful.

### read_manifest

```python
read_manifest(directory: 'Path') -> 'dict[str, Any]'
```

Read the manifest of a results directory.

Raises:
    FileNotFoundError: If the directory has no manifest.

### append_run

```python
append_run(directory: 'Path', record: 'Mapping[str, Any]') -> 'None'
```

Append one run record to ``runs.jsonl`` and flush it to the OS.

### load_runs

```python
load_runs(directory: 'Path') -> 'list[dict[str, Any]]'
```

Load all run records.

A truncated *last* line is ignored, because an interrupted process can leave one.
Corruption anywhere else raises, since it means the file was damaged.

Raises:
    json.JSONDecodeError: If a line other than the last is invalid.

### prepare_directory

```python
prepare_directory(directory: 'Path', spec_hash: 'str', *, overwrite: 'bool' = False, resume: 'bool' = False) -> 'list[dict[str, Any]]'
```

Make ``directory`` ready for a run and return completed records to keep.

Args:
    directory: Target directory.
    spec_hash: Hash of the experiment definition.
    overwrite: Delete existing results first.
    resume: Keep completed runs, provided the experiment definition is unchanged.

Raises:
    ResultsExistError: If results exist and neither option was given, or if an
        overwrite target is not a results directory.
    ResultsMismatchError: If resuming results from a different definition.
    ValueError: If both options are set.

### flatten_record

```python
flatten_record(record: 'Mapping[str, Any]') -> 'dict[str, Any]'
```

Return a CSV row: scalars kept, config expanded to ``config_*``, sweep as JSON text.

### runs_table

```python
runs_table(records: 'Sequence[Mapping[str, Any]]') -> 'pd.DataFrame'
```

Return all records as a flat table.

### summary_table

```python
summary_table(records: 'Sequence[Mapping[str, Any]]') -> 'pd.DataFrame'
```

Return descriptive statistics of completed runs, one row per group and metric.

``n_undefined`` counts runs where the metric was unavailable, such as a Silhouette
score that is undefined for a single non-empty cluster.

### paired_table

```python
paired_table(records: 'Sequence[Mapping[str, Any]]') -> 'pd.DataFrame'
```

Compare the SSE of each condition with the BA reference on shared seeds.

``a`` is the listed condition and ``b`` is the reference, so a negative difference
means the condition has lower SSE. Holm adjustment is applied per dataset.

### write_tables

```python
write_tables(directory: 'Path', records: 'Sequence[Mapping[str, Any]]') -> 'None'
```

Write ``runs.csv``, ``summary.csv`` and ``paired_vs_reference.csv``.

## ba_clustering.validation

Exceptions and array validators shared across the package.

Validators reject bad input with actionable messages. They never repair data:
NaN is not replaced, shapes are not guessed, and strings are not coerced.

### ConfigurationError

```python
ConfigurationError (inherits the built-in exception constructor)
```

A configuration value is missing, unknown, or outside its valid range.

### DataValidationError

```python
DataValidationError (inherits the built-in exception constructor)
```

An input array or dataset violates the data contract.

### NotFittedError

```python
NotFittedError (inherits the built-in exception constructor)
```

A fitted attribute or method was used before ``fit`` was called.

### validate_data_matrix

```python
validate_data_matrix(X: 'Any') -> 'FloatArray'
```

Return ``X`` as a validated float64 array of shape ``(n_samples, n_features)``.

The input is never modified; the returned array is a new object.

Raises:
    DataValidationError: If ``X`` is not 2-D, is empty, is non-numeric, or
        contains NaN or infinity.

### validate_centroids

```python
validate_centroids(centroids: 'Any', n_features: 'int') -> 'FloatArray'
```

Return ``centroids`` as a validated float64 array of shape ``(K, n_features)``.

Raises:
    DataValidationError: If the shape is wrong, the array is empty, or any
        coordinate is non-numeric or non-finite.
