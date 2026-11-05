"""End-to-end pipeline: dataset, preprocessing, BA, evaluation, export. No network needed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ba_clustering import (
    BAConfig,
    BatClustering,
    KMeansConfig,
    compute_sse,
    load_iris_dataset,
    load_wine_dataset,
    make_synthetic_dataset,
    preprocess,
    run_kmeans,
)
from ba_clustering.experiments import ExperimentSpec, run_experiment
from ba_clustering.metrics import silhouette_or_none
from ba_clustering.results import load_runs

pytestmark = pytest.mark.integration

REFERENCE = {
    "A0": 0.5, "r0": 0.5, "alpha": 0.9, "gamma": 0.9, "f_min": 0.0, "f_max": 2.0,
}  # fmt: skip


@pytest.mark.parametrize(
    "dataset",
    [load_iris_dataset(), load_wine_dataset(), make_synthetic_dataset(4, 100, 3, 1.0, seed=0)],
    ids=["iris", "wine", "synthetic"],
)
def test_full_pipeline_in_memory(dataset: Any) -> None:
    X, _ = preprocess(dataset.X, "zscore")
    config = BAConfig(
        population_size=15, max_iterations=30, n_clusters=dataset.n_clusters, seed=0, **REFERENCE
    )
    model = BatClustering(config).fit(X)
    labels = model.predict(X)

    assert model.centroids_.shape == (dataset.n_clusters, X.shape[1])
    assert model.best_fitness_ == compute_sse(X, model.centroids_)
    assert labels.shape == (X.shape[0],)
    assert (model.centroids_ >= X.min(axis=0)).all()
    assert (model.centroids_ <= X.max(axis=0)).all()
    assert model.n_evaluations_ == config.max_evaluations
    assert model.history_.best_fitness[-1] <= model.history_.best_fitness[0]

    baseline = run_kmeans(
        X,
        KMeansConfig(
            n_clusters=dataset.n_clusters,
            seed=0,
            n_init=3,
            init="k-means++",
            max_iter=100,
            tol=1e-4,
        ),
    )  # fmt: skip
    assert baseline.sse == compute_sse(X, baseline.centroids)
    assert silhouette_or_none(X, baseline.labels) is not None


def spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="integration",
        description="integration test",
        design="ofat",
        datasets={
            "iris": {"loader": "iris", "n_clusters": 3},
            "synthetic": {
                "loader": "synthetic", "n_clusters": 3, "n_samples": 60, "n_features": 2,
                "cluster_std": 1.0, "seed": 1,
            },
        },
        preprocessing_method="zscore",
        seeds=(0, 1),
        algorithms=("ba", "kmeans"),
        ba_base={
            **REFERENCE, "population_size": 6, "max_iterations": 6, "tolerance": 0.0,
            "patience": None,
        },
        kmeans={"n_init": 2, "init": "k-means++", "max_iter": 50, "tol": 0.0001},
        factors={"A0": (0.25,), "r0": (0.25,), "alpha": (0.5,), "gamma": (0.1,),
                 "f_min": (1.0,), "f_max": (5.0,)},
    )  # fmt: skip


def test_all_six_parameters_reach_the_run_configuration(tmp_path: Path) -> None:
    records = load_runs(run_experiment(spec(), tmp_path).directory)
    for name, value in spec().factors.items():
        matching = [r for r in records if r["condition_id"] == f"{name}={value[0]:g}"]
        assert len(matching) == 4
        assert all(r["config"][name] == value[0] for r in matching)


def test_experiment_results_are_reproducible_end_to_end(tmp_path: Path) -> None:
    def scientific_content(directory: Path) -> list[dict[str, Any]]:
        volatile = {"runtime_seconds", "timestamp_utc"}
        return [
            {key: value for key, value in record.items() if key not in volatile}
            for record in load_runs(directory)
        ]

    first = run_experiment(spec(), tmp_path / "first").directory
    second = run_experiment(spec(), tmp_path / "second").directory
    assert scientific_content(first) == scientific_content(second)


def test_exported_sse_matches_independent_recomputation_from_saved_history(tmp_path: Path) -> None:
    directory = run_experiment(spec(), tmp_path).directory
    runs = pd.read_csv(directory / "runs.csv")
    ba = runs[(runs.algorithm == "ba") & (runs.status == "completed")]
    assert len(ba) > 0
    assert (ba.n_evaluations == 6 * 7).all()

    # Re-run one row from nothing but its saved config and compare.
    record = next(r for r in load_runs(directory) if r["run_id"].startswith("iris__ba__gamma"))
    X, _ = preprocess(load_iris_dataset().X, "zscore")
    model = BatClustering(BAConfig.from_mapping(record["config"])).fit(X)
    assert model.best_fitness_ == record["sse"]
    assert model.history_.best_fitness == record["history"]["best_fitness"]


def test_manifest_is_valid_json_and_marks_completion(tmp_path: Path) -> None:
    directory = run_experiment(spec(), tmp_path).directory
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    # 2 datasets x 2 seeds x (K-Means + BA reference + 6 OFAT levels)
    assert manifest["n_runs"] == 32
