"""Controlled experiments: baseline comparisons, OFAT sweeps, and factorial grids.

An experiment is declared in ``configs/experiments.yaml`` and expanded into runs. Each
run is fully determined by its dataset, algorithm, configuration and seed, so runs can
be repeated or resumed independently. See ``docs/EXPERIMENT_DESIGN.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ba_clustering.baseline import KMeansConfig, run_kmeans
from ba_clustering.clustering import BatClustering, count_empty_clusters
from ba_clustering.config import BAConfig
from ba_clustering.datasets import Dataset, load_dataset
from ba_clustering.metrics import (
    adjusted_rand_or_none,
    evaluations_to_plateau,
    silhouette_or_none,
)
from ba_clustering.preprocessing import PreprocessingInfo, preprocess
from ba_clustering.results import (
    append_run,
    canonical_hash,
    environment_info,
    git_commit,
    load_runs,
    prepare_directory,
    write_manifest,
    write_tables,
)
from ba_clustering.validation import ConfigurationError, DataValidationError, FloatArray

SWEEPABLE_PARAMETERS = ("A0", "r0", "alpha", "gamma", "f_min", "f_max")
DESIGNS = ("baseline", "ofat", "grid")
ALGORITHMS = ("ba", "kmeans")
REFERENCE_CONDITION = "reference"
KMEANS_CONDITION = "kmeans"

_EXPERIMENT_KEYS = frozenset(
    {"description", "design", "datasets", "seeds", "algorithms", "ba_overrides", "factors"}
)
_BASE_KEYS = frozenset({"ba", "preprocessing", "kmeans", "datasets"})
_FILE_KEYS = frozenset({"base_config", "seed_rosters", "experiments"})


@dataclass(frozen=True)
class ExperimentSpec:
    """A fully resolved experiment definition.

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
    """

    experiment_id: str
    description: str
    design: str
    datasets: Mapping[str, Mapping[str, Any]]
    preprocessing_method: str
    seeds: tuple[int, ...]
    algorithms: tuple[str, ...]
    ba_base: Mapping[str, Any]
    kmeans: Mapping[str, Any]
    factors: Mapping[str, tuple[float, ...]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the definition."""
        snapshot = dataclasses.asdict(self)
        snapshot["seeds"] = list(self.seeds)
        snapshot["algorithms"] = list(self.algorithms)
        snapshot["factors"] = {name: list(levels) for name, levels in self.factors.items()}
        return snapshot


@dataclass(frozen=True)
class Condition:
    """One BA parameter setting. ``overrides`` is empty for the reference condition."""

    condition_id: str
    overrides: Mapping[str, float]


@dataclass(frozen=True)
class PreparedDataset:
    """A dataset and the single preprocessed matrix every algorithm uses."""

    dataset: Dataset
    X: FloatArray
    preprocessing: PreprocessingInfo
    sha256: str


@dataclass(frozen=True)
class ExperimentOutcome:
    """Summary of a finished experiment."""

    directory: Path
    n_completed: int
    n_failed: int
    n_skipped: int


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        content = yaml.safe_load(path.read_text())
    except FileNotFoundError as error:
        raise ConfigurationError(
            f"Configuration file not found: {path}. Run from the repository root, or pass "
            "an explicit path."
        ) from error
    if not isinstance(content, dict):
        raise ConfigurationError(f"{path} must contain a YAML mapping.")
    return content


def _require_known_keys(found: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(found) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown key(s) in {where}: {unknown}.")


def load_base_config(path: str | Path) -> dict[str, Any]:
    """Load and check the top-level structure of ``configs/default.yaml``."""
    config_path = Path(path)
    base = _load_yaml(config_path)
    _require_known_keys(base, _BASE_KEYS, str(config_path))
    missing = sorted(_BASE_KEYS - set(base))
    if missing:
        raise ConfigurationError(f"{config_path} is missing section(s): {missing}.")
    return base


def list_experiments(path: str | Path) -> dict[str, str]:
    """Return experiment IDs and descriptions from an experiments file."""
    content = _load_yaml(Path(path))
    return {
        name: str(entry.get("description", ""))
        for name, entry in content.get("experiments", {}).items()
    }


def load_experiment_spec(path: str | Path, experiment_id: str) -> ExperimentSpec:
    """Resolve one experiment from an experiments file and its base config.

    Raises:
        ConfigurationError: For any unknown key, unknown reference, or invalid design.
    """
    experiments_path = Path(path)
    content = _load_yaml(experiments_path)
    _require_known_keys(content, _FILE_KEYS, str(experiments_path))
    experiments = content.get("experiments", {})
    if experiment_id not in experiments:
        raise ConfigurationError(
            f"Unknown experiment {experiment_id!r}. Available: {sorted(experiments)}."
        )
    entry = experiments[experiment_id]
    _require_known_keys(entry, _EXPERIMENT_KEYS, f"experiment {experiment_id!r}")

    base = load_base_config(experiments_path.parent / content["base_config"])

    roster_name = entry.get("seeds")
    rosters = content.get("seed_rosters", {})
    if roster_name not in rosters:
        raise ConfigurationError(
            f"Experiment {experiment_id!r}: unknown seed roster {roster_name!r}. "
            f"Available: {sorted(rosters)}."
        )
    seeds = tuple(rosters[roster_name])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ConfigurationError(f"Seed roster {roster_name!r} must be non-empty and unique.")

    dataset_names = entry.get("datasets", [])
    unknown_datasets = sorted(set(dataset_names) - set(base["datasets"]))
    if unknown_datasets or not dataset_names:
        raise ConfigurationError(
            f"Experiment {experiment_id!r}: datasets must be a non-empty subset of "
            f"{sorted(base['datasets'])}, got {dataset_names}."
        )

    algorithms = tuple(entry.get("algorithms", ALGORITHMS))
    if not algorithms or set(algorithms) - set(ALGORITHMS):
        raise ConfigurationError(
            f"Experiment {experiment_id!r}: algorithms must be a subset of {ALGORITHMS}."
        )

    spec = ExperimentSpec(
        experiment_id=experiment_id,
        description=str(entry.get("description", "")),
        design=str(entry.get("design")),
        datasets={name: dict(base["datasets"][name]) for name in dataset_names},
        preprocessing_method=str(base["preprocessing"]["method"]),
        seeds=seeds,
        algorithms=algorithms,
        ba_base={**base["ba"], **entry.get("ba_overrides", {})},
        kmeans=dict(base["kmeans"]),
        factors={
            name: tuple(float(level) for level in levels)
            for name, levels in entry.get("factors", {}).items()
        },
    )
    expand_conditions(spec)
    return spec


def _condition_id(overrides: Mapping[str, float]) -> str:
    return "|".join(f"{name}={value:g}" for name, value in overrides.items())


def expand_conditions(spec: ExperimentSpec) -> list[Condition]:
    """Expand the design into BA conditions, with the reference always included.

    A level equal to the reference value is the reference condition itself, so it is
    not run twice. OFAT varies one parameter at a time, holding the rest at reference.

    Raises:
        ConfigurationError: If the factors do not suit the design.
    """
    if spec.design not in DESIGNS:
        raise ConfigurationError(f"Unknown design {spec.design!r}. Supported: {DESIGNS}.")
    unknown = sorted(set(spec.factors) - set(SWEEPABLE_PARAMETERS))
    if unknown:
        raise ConfigurationError(
            f"Cannot sweep {unknown}. Sweepable parameters: {SWEEPABLE_PARAMETERS}."
        )
    if spec.design == "baseline" and spec.factors:
        raise ConfigurationError("A baseline design must not declare factors.")
    if spec.design == "ofat" and not spec.factors:
        raise ConfigurationError("An OFAT design needs at least one factor.")
    if spec.design == "grid" and len(spec.factors) < 2:
        raise ConfigurationError("A grid design needs at least two factors.")

    conditions = [Condition(REFERENCE_CONDITION, {})]
    if spec.design == "ofat":
        for name, levels in spec.factors.items():
            for level in levels:
                if level != spec.ba_base[name]:
                    overrides = {name: level}
                    conditions.append(Condition(_condition_id(overrides), overrides))
    elif spec.design == "grid":
        names = list(spec.factors)
        for combination in itertools.product(*(spec.factors[name] for name in names)):
            overrides = dict(zip(names, combination, strict=True))
            if any(value != spec.ba_base[name] for name, value in overrides.items()):
                conditions.append(Condition(_condition_id(overrides), overrides))
    identifiers = [condition.condition_id for condition in conditions]
    if len(set(identifiers)) != len(identifiers):
        raise ConfigurationError(
            f"Levels produce duplicate condition IDs: {sorted(identifiers)}. "
            "Levels must be distinct at six significant digits."
        )
    return conditions


def build_ba_config(
    spec: ExperimentSpec, condition: Condition, n_clusters: int, seed: int
) -> BAConfig:
    """Return the validated BA config for one run."""
    return BAConfig.from_mapping(
        {**spec.ba_base, **condition.overrides, "n_clusters": n_clusters, "seed": seed}
    )


def prepare_dataset(options: Mapping[str, Any], preprocessing_method: str) -> PreparedDataset:
    """Load one dataset and apply the shared preprocessing."""
    dataset = load_dataset(options)
    X, info = preprocess(dataset.X, preprocessing_method)
    return PreparedDataset(dataset, X, info, hashlib.sha256(X.tobytes()).hexdigest())


def _common_fields(
    spec: ExperimentSpec,
    prepared: PreparedDataset,
    dataset_name: str,
    algorithm: str,
    condition_id: str,
    sweep: Mapping[str, float],
    seed: int,
    config: Mapping[str, Any],
    commit: str | None,
    version: str,
) -> dict[str, Any]:
    return {
        "experiment_id": spec.experiment_id,
        "run_id": f"{dataset_name}__{algorithm}__{condition_id}__seed{seed}",
        "dataset": dataset_name,
        "algorithm": algorithm,
        "condition_id": condition_id,
        "design": spec.design,
        "sweep": dict(sweep),
        "seed": seed,
        "config": dict(config),
        "preprocessing": prepared.preprocessing.method,
        "data_sha256": prepared.sha256,
        "software_version": version,
        "git_commit": commit,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def run_ba_once(prepared: PreparedDataset, config: BAConfig) -> dict[str, Any]:
    """Run BA once and return its metric fields."""
    model = BatClustering(config).fit(prepared.X)
    history = model.history_
    return {
        "status": "completed",
        "error": None,
        "sse": model.best_fitness_,
        "silhouette": silhouette_or_none(prepared.X, model.labels_),
        "ari_external": adjusted_rand_or_none(prepared.dataset.labels_external, model.labels_),
        "n_empty_clusters": model.n_empty_clusters_,
        "runtime_seconds": model.runtime_seconds_,
        "n_evaluations": model.n_evaluations_,
        "n_iterations": model.n_iterations_,
        "stop_reason": model.stop_reason_,
        "evaluations_to_plateau": evaluations_to_plateau(
            history.n_evaluations, history.best_fitness
        ),
        "n_local_candidates": sum(history.n_local_candidates),
        "n_accepted": sum(history.n_accepted),
        "n_accepted_local": sum(history.n_accepted_local),
        "n_accepted_global": sum(history.n_accepted) - sum(history.n_accepted_local),
        "n_clipped_coordinates": sum(history.n_clipped_coordinates),
        "max_abs_velocity_final": history.max_abs_velocity[-1],
        "n_lloyd_iterations": None,
        "history": history.to_dict(),
    }


def run_kmeans_once(prepared: PreparedDataset, config: KMeansConfig) -> dict[str, Any]:
    """Run K-Means once and return its metric fields."""
    result = run_kmeans(prepared.X, config)
    return {
        "status": "completed",
        "error": None,
        "sse": result.sse,
        "silhouette": silhouette_or_none(prepared.X, result.labels),
        "ari_external": adjusted_rand_or_none(prepared.dataset.labels_external, result.labels),
        "n_empty_clusters": count_empty_clusters(result.labels, config.n_clusters),
        "runtime_seconds": result.runtime_seconds,
        "n_evaluations": None,
        "n_iterations": None,
        "stop_reason": None,
        "evaluations_to_plateau": None,
        "n_local_candidates": None,
        "n_accepted": None,
        "n_accepted_local": None,
        "n_accepted_global": None,
        "n_clipped_coordinates": None,
        "max_abs_velocity_final": None,
        "n_lloyd_iterations": result.n_lloyd_iterations,
        "restart_sse": result.restart_sse.tolist(),
        "history": None,
    }


def _failure_fields(error: Exception) -> dict[str, Any]:
    return {"status": "failed", "error": f"{type(error).__name__}: {error}", "sse": None}


def run_experiment(
    spec: ExperimentSpec,
    output_root: str | Path,
    *,
    overwrite: bool = False,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ExperimentOutcome:
    """Run an experiment and write its outputs to ``output_root/<experiment_id>/``.

    Existing results are never replaced without ``overwrite``. With ``resume``, completed
    runs are kept, and failed or missing runs are executed. A run that fails on invalid
    data or a numerical error is recorded as failed, and the experiment continues;
    programming errors propagate.

    Raises:
        ResultsExistError: If results exist and no option was given.
        ResultsMismatchError: If resuming results from a different definition.
        ConfigurationError: If any condition is invalid. This is checked before any run.
    """
    conditions = expand_conditions(spec)
    prepared_by_name = {
        name: prepare_dataset(options, spec.preprocessing_method)
        for name, options in spec.datasets.items()
    }
    # Validate every configuration before the first run or any file change.
    for prepared in prepared_by_name.values():
        for seed in spec.seeds:
            for condition in conditions:
                build_ba_config(spec, condition, prepared.dataset.n_clusters, seed)
            if "kmeans" in spec.algorithms:
                KMeansConfig(n_clusters=prepared.dataset.n_clusters, seed=seed, **spec.kmeans)

    snapshot = spec.to_dict()
    snapshot["dataset_sha256"] = {
        name: prepared.sha256 for name, prepared in prepared_by_name.items()
    }
    spec_hash = canonical_hash(snapshot)
    directory = Path(output_root) / spec.experiment_id
    kept = prepare_directory(directory, spec_hash, overwrite=overwrite, resume=resume)
    done = {record["run_id"] for record in kept}

    environment = environment_info()
    manifest: dict[str, Any] = {
        "experiment_id": spec.experiment_id,
        "spec": snapshot,
        "spec_hash": spec_hash,
        "environment": environment,
        "datasets": {
            name: {
                **prepared.dataset.metadata(),
                "preprocessing": prepared.preprocessing.to_dict(),
                "zero_range_features": prepared.preprocessing.constant_features,
                "sha256": prepared.sha256,
            }
            for name, prepared in prepared_by_name.items()
        },
        "conditions": [dataclasses.asdict(condition) for condition in conditions],
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_manifest(directory, manifest)

    commit = git_commit()
    version = str(environment["ba_clustering"])
    n_completed = n_failed = 0
    for name, prepared in prepared_by_name.items():
        n_clusters = prepared.dataset.n_clusters
        for seed in spec.seeds:
            planned: list[tuple[str, str, Mapping[str, float], dict[str, Any]]] = []
            if "kmeans" in spec.algorithms:
                km = KMeansConfig(n_clusters=n_clusters, seed=seed, **spec.kmeans)
                planned.append(("kmeans", KMEANS_CONDITION, {}, km.to_dict()))
            if "ba" in spec.algorithms:
                for condition in conditions:
                    ba = build_ba_config(spec, condition, n_clusters, seed)
                    planned.append(
                        ("ba", condition.condition_id, condition.overrides, ba.to_dict())
                    )
            for algorithm, condition_id, sweep, config in planned:
                fields = _common_fields(
                    spec, prepared, name, algorithm, condition_id, sweep, seed, config,
                    commit, version,
                )  # fmt: skip
                if fields["run_id"] in done:
                    continue
                try:
                    if algorithm == "ba":
                        fields.update(run_ba_once(prepared, BAConfig.from_mapping(config)))
                    else:
                        fields.update(run_kmeans_once(prepared, KMeansConfig(**config)))
                    n_completed += 1
                except (DataValidationError, ConfigurationError, FloatingPointError) as error:
                    fields.update(_failure_fields(error))
                    n_failed += 1
                append_run(directory, fields)
                if progress is not None:
                    progress(f"{fields['run_id']}: {fields['status']} sse={fields['sse']}")

    records = load_runs(directory)
    write_tables(directory, records)
    manifest["status"] = "complete" if n_failed == 0 else "complete_with_failures"
    manifest["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    manifest["n_runs"] = len(records)
    manifest["n_failed"] = n_failed
    write_manifest(directory, manifest)
    return ExperimentOutcome(directory, n_completed, n_failed, len(kept))


def seeds_overlap(first: Sequence[int], second: Sequence[int]) -> list[int]:
    """Return seeds present in both rosters. Discovery and confirmation must not overlap."""
    return sorted(set(first) & set(second))
