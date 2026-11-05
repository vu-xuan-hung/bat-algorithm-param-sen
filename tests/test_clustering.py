from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ba_clustering.baseline import KMeansConfig, run_kmeans
from ba_clustering.clustering import (
    BatClustering,
    assign_labels,
    cluster_sizes,
    count_empty_clusters,
)
from ba_clustering.datasets import (
    Dataset,
    load_csv_dataset,
    load_dataset,
    load_iris_dataset,
    load_wine_dataset,
    make_synthetic_dataset,
)
from ba_clustering.fitness import compute_sse
from ba_clustering.metrics import (
    adjusted_rand_or_none,
    evaluations_to_plateau,
    holm_adjust,
    paired_comparison,
    silhouette_or_none,
    summarize,
)
from ba_clustering.preprocessing import apply_preprocessing, fit_preprocessing, preprocess
from ba_clustering.validation import (
    ConfigurationError,
    DataValidationError,
    FloatArray,
    NotFittedError,
)
from tests.conftest import make_config

# --- Assignment ------------------------------------------------------------------


def test_assignment_uses_nearest_centroid(toy_X: FloatArray) -> None:
    labels = assign_labels(toy_X, np.array([[2.0], [8.0]]))
    assert labels.tolist() == [0, 0, 1, 1]


def test_assignment_tie_goes_to_lowest_index() -> None:
    labels = assign_labels(np.array([[5.0]]), np.array([[4.0], [6.0]]))
    assert labels.tolist() == [0]


def test_identical_centroids_assign_everything_to_the_first(toy_X: FloatArray) -> None:
    labels = assign_labels(toy_X, np.array([[3.0], [3.0]]))
    assert labels.tolist() == [0, 0, 0, 0]
    assert count_empty_clusters(labels, 2) == 1


def test_empty_cluster_counting(toy_X: FloatArray) -> None:
    labels = assign_labels(toy_X, np.array([[5.0], [100.0], [-100.0]]))
    assert cluster_sizes(labels, 3).tolist() == [4, 0, 0]
    assert count_empty_clusters(labels, 3) == 2


def test_assignment_does_not_move_empty_centroids(toy_X: FloatArray) -> None:
    centroids = np.array([[5.0], [100.0]])
    before = centroids.copy()
    assign_labels(toy_X, centroids)
    compute_sse(toy_X, centroids)
    np.testing.assert_array_equal(centroids, before)


# --- Estimator -------------------------------------------------------------------


def test_public_interface_from_the_brief(toy_X: FloatArray) -> None:
    model = BatClustering(config=make_config())
    assert model.fit(toy_X) is model
    labels = model.predict(toy_X)
    assert labels.shape == (4,)
    assert model.centroids_.shape == (2, 1)
    assert model.best_fitness_ == compute_sse(toy_X, model.centroids_)
    assert len(model.history_.best_fitness) == 11
    np.testing.assert_array_equal(labels, model.labels_)


def test_ba_reaches_the_toy_optimum_given_enough_budget(toy_X: FloatArray) -> None:
    config = make_config(population_size=20, max_iterations=200, A0=1.0, alpha=0.98)
    assert BatClustering(config).fit(toy_X).best_fitness_ == pytest.approx(1.0, abs=0.05)


def test_predict_on_new_data(toy_X: FloatArray) -> None:
    model = BatClustering(make_config()).fit(toy_X)
    assert model.predict(np.array([[0.0]])).shape == (1,)


@pytest.mark.parametrize(
    "attribute", ["centroids_", "best_fitness_", "history_", "labels_", "n_evaluations_"]
)
def test_fitted_attributes_need_fit(attribute: str) -> None:
    with pytest.raises(NotFittedError):
        getattr(BatClustering(make_config()), attribute)


def test_predict_before_fit_raises(toy_X: FloatArray) -> None:
    with pytest.raises(NotFittedError):
        BatClustering(make_config()).predict(toy_X)


def test_predict_with_wrong_feature_count_raises(toy_X: FloatArray) -> None:
    model = BatClustering(make_config()).fit(toy_X)
    with pytest.raises(DataValidationError, match="centroids must have shape"):
        model.predict(np.ones((2, 3)))


def test_centroids_are_read_only(toy_X: FloatArray) -> None:
    model = BatClustering(make_config()).fit(toy_X)
    with pytest.raises(ValueError, match="read-only"):
        model.centroids_[0, 0] = 99.0


# --- Input and edge cases --------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        np.empty((0, 2)),
        np.empty((3, 0)),
        np.array([1.0, 2.0, 3.0]),
        np.ones((2, 2, 2)),
        np.array([[1.0, np.nan], [2.0, 3.0]]),
        np.array([[1.0, np.inf], [2.0, 3.0]]),
        np.array([[1.0, -np.inf], [2.0, 3.0]]),
        np.array([["a", "b"], ["c", "d"]]),
        np.array([[True, False], [False, True]]),
        np.array([[1.0 + 2.0j], [3.0 + 0.0j]]),
        np.array([[1.0, None], [2.0, 3.0]], dtype=object),
    ],
)
def test_invalid_data_is_rejected(bad: FloatArray) -> None:
    with pytest.raises(DataValidationError):
        BatClustering(make_config()).fit(bad)


def test_ragged_input_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="inconsistent dimensions"):
        BatClustering(make_config()).fit([[1.0, 2.0], [3.0]])


def test_nan_is_never_silently_replaced() -> None:
    X = np.array([[1.0], [np.nan]])
    with pytest.raises(DataValidationError, match="non-finite"):
        compute_sse(X, np.array([[0.0]]))


@pytest.mark.parametrize("n_clusters", [5, 100])
def test_k_above_n_samples_is_rejected(toy_X: FloatArray, n_clusters: int) -> None:
    with pytest.raises(ConfigurationError, match="exceeds n_samples"):
        BatClustering(make_config(n_clusters=n_clusters)).fit(toy_X)


def test_k_equal_to_n_samples_works(toy_X: FloatArray) -> None:
    assert BatClustering(make_config(n_clusters=4)).fit(toy_X).centroids_.shape == (4, 1)


def test_k_one_works(toy_X: FloatArray) -> None:
    model = BatClustering(make_config(n_clusters=1)).fit(toy_X)
    assert model.labels_.tolist() == [0, 0, 0, 0]


def test_constant_feature_is_handled_and_stays_pinned() -> None:
    X = np.array([[1.0, 7.0], [2.0, 7.0], [8.0, 7.0], [9.0, 7.0]])
    model = BatClustering(make_config()).fit(X)
    assert (model.centroids_[:, 1] == 7.0).all()
    assert model.bounds_.zero_range_features == [1]
    assert np.isfinite(model.best_fitness_)


def test_all_features_constant_gives_zero_sse() -> None:
    model = BatClustering(make_config()).fit(np.full((5, 2), 3.0))
    assert model.best_fitness_ == 0.0


def test_duplicate_samples_work() -> None:
    X = np.array([[1.0], [1.0], [1.0], [9.0], [9.0]])
    model = BatClustering(make_config(population_size=15, max_iterations=60, A0=1.0)).fit(X)
    assert np.isfinite(model.best_fitness_)
    assert model.best_fitness_ >= 0.0


@pytest.mark.parametrize("population_size", [1, 2])
def test_very_small_populations_work(toy_X: FloatArray, population_size: int) -> None:
    model = BatClustering(make_config(population_size=population_size)).fit(toy_X)
    assert model.n_evaluations_ == population_size * 11


def test_single_bat_has_zero_velocity_forever(toy_X: FloatArray) -> None:
    # A lone bat is always X*, so the velocity term (X - X*) f stays zero.
    model = BatClustering(make_config(population_size=1, max_iterations=30)).fit(toy_X)
    assert max(model.history_.max_abs_velocity) == 0.0


def test_single_sample_works() -> None:
    model = BatClustering(make_config(n_clusters=1)).fit(np.array([[3.0, 4.0]]))
    assert model.best_fitness_ == 0.0


def test_integer_data_is_accepted(toy_X: FloatArray) -> None:
    model = BatClustering(make_config()).fit(np.array([[1], [2], [8], [9]]))
    assert model.centroids_.dtype == np.float64


def test_empty_clusters_are_reported_and_not_repaired() -> None:
    # Choosing K = 4 for two duplicated values forces at least two empty clusters.
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    model = BatClustering(make_config(n_clusters=4, max_iterations=2, A0=0.0)).fit(X)
    assert model.n_empty_clusters_ == 4 - np.unique(model.labels_).size
    assert model.n_empty_clusters_ >= 2
    assert np.isfinite(model.best_fitness_)


# --- Datasets --------------------------------------------------------------------


def test_iris_and_wine_metadata() -> None:
    iris = load_iris_dataset().metadata()
    wine = load_wine_dataset().metadata()
    assert (iris["n_samples"], iris["n_features"], iris["n_clusters"]) == (150, 4, 3)
    assert (wine["n_samples"], wine["n_features"], wine["n_clusters"]) == (178, 13, 3)


def test_synthetic_dataset_is_reproducible_and_records_its_generator() -> None:
    first = make_synthetic_dataset(3, 50, 2, 1.0, seed=1)
    second = make_synthetic_dataset(3, 50, 2, 1.0, seed=1)
    np.testing.assert_array_equal(first.X, second.X)
    assert first.metadata()["generator"]["seed"] == 1


def test_load_dataset_dispatch_and_unknown_options() -> None:
    assert load_dataset({"loader": "iris", "n_clusters": 3}).name == "iris"
    with pytest.raises(ConfigurationError, match="Unknown dataset loader"):
        load_dataset({"loader": "mnist"})
    with pytest.raises(ConfigurationError, match="Invalid options"):
        load_dataset({"loader": "iris", "n_clusters": 3, "colour": "red"})


def test_dataset_rejects_invalid_k_and_label_length() -> None:
    with pytest.raises(DataValidationError, match="n_clusters"):
        Dataset("d", np.ones((2, 1)), 3, "test")
    with pytest.raises(DataValidationError, match="labels"):
        Dataset("d", np.ones((2, 1)), 1, "test", labels_external=np.array([1]))


def test_csv_loader_separates_labels(tmp_path: Any) -> None:
    path = tmp_path / "points.csv"
    path.write_text("a,b,label\n1,2,x\n3,4,y\n")
    dataset = load_csv_dataset(path, n_clusters=2, label_column="label")
    assert dataset.X.shape == (2, 2)
    assert dataset.labels_external is not None
    assert dataset.labels_external.tolist() == ["x", "y"]
    assert dataset.name == "points"


def test_csv_loader_rejects_missing_values_and_text(tmp_path: Any) -> None:
    missing = tmp_path / "missing.csv"
    missing.write_text("a,b\n1,\n3,4\n")
    with pytest.raises(DataValidationError, match="non-finite"):
        load_csv_dataset(missing, n_clusters=1)
    text = tmp_path / "text.csv"
    text.write_text("a,b\n1,x\n3,y\n")
    with pytest.raises(DataValidationError, match="Non-numeric"):
        load_csv_dataset(text, n_clusters=1)
    with pytest.raises(DataValidationError, match="label_column"):
        load_csv_dataset(text, n_clusters=1, label_column="nope")


def test_labels_cannot_reach_the_optimiser() -> None:
    # Changing the external labels must have no effect on the optimisation.
    dataset = load_iris_dataset()
    config = make_config(n_clusters=3)
    first = BatClustering(config).fit(dataset.X).history_.to_dict()
    shuffled = Dataset("iris", dataset.X, 3, "test", labels_external=np.zeros(150, dtype=int))
    assert BatClustering(config).fit(shuffled.X).history_.to_dict() == first


# --- Preprocessing ---------------------------------------------------------------


def test_zscore_matches_standard_scaler() -> None:
    from sklearn.preprocessing import StandardScaler

    data = load_wine_dataset().X
    np.testing.assert_allclose(preprocess(data, "zscore")[0], StandardScaler().fit_transform(data))


def test_zscore_constant_feature_becomes_zero_not_nan() -> None:
    X = np.array([[1.0, 0.1], [2.0, 0.1], [3.0, 0.1]])
    transformed, info = preprocess(X, "zscore")
    assert (transformed[:, 1] == 0.0).all()
    assert info.constant_features == [1]


def test_none_preprocessing_is_identity_and_returns_a_copy(toy_X: FloatArray) -> None:
    transformed, _ = preprocess(toy_X, "none")
    np.testing.assert_array_equal(transformed, toy_X)
    assert transformed is not toy_X


def test_unknown_preprocessing_is_rejected(toy_X: FloatArray) -> None:
    with pytest.raises(ConfigurationError, match="Unknown preprocessing"):
        preprocess(toy_X, "robust")


def test_apply_preprocessing_rejects_wrong_feature_count(toy_X: FloatArray) -> None:
    with pytest.raises(DataValidationError, match="fitted on"):
        apply_preprocessing(np.ones((2, 3)), fit_preprocessing(toy_X))


# --- K-Means baseline ------------------------------------------------------------


def kmeans_config(**overrides: Any) -> KMeansConfig:
    values: dict[str, Any] = {
        "n_clusters": 2, "seed": 0, "n_init": 5, "init": "k-means++", "max_iter": 100, "tol": 1e-4,
    }  # fmt: skip
    return KMeansConfig(**(values | overrides))


def test_kmeans_finds_toy_optimum_and_uses_project_sse(toy_X: FloatArray) -> None:
    result = run_kmeans(toy_X, kmeans_config())
    assert result.sse == pytest.approx(1.0)
    assert result.sse == compute_sse(toy_X, result.centroids)
    assert result.labels.shape == (4,)


def test_kmeans_is_reproducible_and_records_restarts() -> None:
    X = preprocess(load_iris_dataset().X)[0]
    first = run_kmeans(X, kmeans_config(n_clusters=3))
    second = run_kmeans(X, kmeans_config(n_clusters=3))
    np.testing.assert_array_equal(first.centroids, second.centroids)
    assert first.restart_seeds == second.restart_seeds
    assert len(first.restart_seeds) == 5
    assert first.restart_sse.shape == (5,)
    assert first.sse == first.restart_sse.min()
    assert first.n_lloyd_iterations >= 5


def test_kmeans_keeps_the_best_restart() -> None:
    X = preprocess(load_wine_dataset().X)[0]
    many = run_kmeans(X, kmeans_config(n_clusters=3, n_init=20, init="random"))
    assert many.sse == many.restart_sse.min()


def test_kmeans_rejects_bad_settings(toy_X: FloatArray) -> None:
    with pytest.raises(ConfigurationError, match="exceeds n_samples"):
        run_kmeans(toy_X, kmeans_config(n_clusters=9))
    with pytest.raises(ConfigurationError):
        kmeans_config(n_init=0)
    with pytest.raises(ConfigurationError, match="init"):
        kmeans_config(init="forgy")
    with pytest.raises(ConfigurationError, match="tol"):
        kmeans_config(tol="0.0001")


# --- Metrics ---------------------------------------------------------------------


def test_silhouette_is_none_when_undefined(toy_X: FloatArray) -> None:
    assert silhouette_or_none(toy_X, np.zeros(4, dtype=np.intp)) is None
    assert silhouette_or_none(toy_X, np.arange(4, dtype=np.intp)) is None


def test_silhouette_when_defined(toy_X: FloatArray) -> None:
    value = silhouette_or_none(toy_X, np.array([0, 0, 1, 1], dtype=np.intp))
    assert value is not None
    assert 0.0 < value <= 1.0


def test_ari_without_labels_is_none_and_with_labels_is_invariant_to_naming() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.intp)
    assert adjusted_rand_or_none(None, labels) is None
    assert adjusted_rand_or_none(np.array([1, 1, 0, 0]), labels) == pytest.approx(1.0)


def test_evaluations_to_plateau() -> None:
    assert evaluations_to_plateau([10, 20, 30, 40], [5.0, 3.0, 2.0, 2.0]) == 30
    assert evaluations_to_plateau([10, 20], [1.0, 1.0]) == 10
    assert evaluations_to_plateau([10, 20, 30], [5.0, 2.05, 2.0], tolerance=0.1) == 20
    with pytest.raises(DataValidationError):
        evaluations_to_plateau([], [])


def test_summarize_hand_values() -> None:
    summary = summarize([1.0, 2.0, 3.0, 4.0])
    assert summary.n == 4
    assert summary.mean == pytest.approx(2.5)
    assert summary.std == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert summary.median == pytest.approx(2.5)
    assert summary.q1 == pytest.approx(1.75)
    assert summary.q3 == pytest.approx(3.25)
    assert summary.ci95_low is not None
    assert summary.ci95_low < 2.5 < summary.ci95_high  # type: ignore[operator]


def test_summarize_single_value_has_no_spread() -> None:
    summary = summarize([7.0])
    assert summary.std is None
    assert summary.ci95_low is None


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf, 1.0]])
def test_summarize_rejects_empty_and_non_finite(values: list[float]) -> None:
    with pytest.raises(DataValidationError):
        summarize(values)


def test_paired_comparison_direction_and_counts() -> None:
    comparison = paired_comparison([1.0, 2.0, 3.0, 5.0], [2.0, 3.0, 3.0, 4.0])
    assert comparison.n_a_lower == 2
    assert comparison.n_b_lower == 1
    assert comparison.n_tied == 1
    assert comparison.mean_difference == pytest.approx(-0.25)


def test_paired_comparison_identical_samples_give_p_one() -> None:
    assert paired_comparison([1.0, 2.0], [1.0, 2.0]).p_value == 1.0


def test_paired_comparison_rejects_mismatched_lengths() -> None:
    with pytest.raises(DataValidationError):
        paired_comparison([1.0], [1.0, 2.0])


def test_holm_adjustment_hand_values() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert holm_adjust([]) == []


@pytest.mark.parametrize("field", ["n_clusters", "seed", "n_init", "max_iter"])
@pytest.mark.parametrize("value", [True, 1.5, "2", np.nan, np.inf])
def test_kmeans_rejects_noninteger_settings(field: str, value: Any) -> None:
    with pytest.raises(ConfigurationError, match=field):
        kmeans_config(**{field: value})


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan, True, -0.1])
def test_kmeans_rejects_invalid_tolerance(value: Any) -> None:
    with pytest.raises(ConfigurationError, match="tol"):
        kmeans_config(tol=value)
