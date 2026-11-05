from __future__ import annotations

import numpy as np
import pytest

from ba_clustering.fitness import compute_sse, sse_unchecked
from ba_clustering.validation import DataValidationError, FloatArray


@pytest.mark.parametrize(
    ("centroids", "expected"),
    [
        ([[2.0], [8.0]], 2.0),
        ([[1.5], [8.5]], 1.0),
        ([[2.5], [7.5]], 5.0),
    ],
)
def test_brief_toy_sse_values(
    toy_X: FloatArray, centroids: list[list[float]], expected: float
) -> None:
    assert compute_sse(toy_X, np.array(centroids)) == pytest.approx(expected, abs=1e-12)


def test_sse_is_squared_not_plain_euclidean() -> None:
    # Distance 3 from a single centroid: squared gives 9; unsquared would give 3.
    assert compute_sse(np.array([[3.0]]), np.array([[0.0]])) == pytest.approx(9.0)


def test_sse_is_total_not_mean(toy_X: FloatArray) -> None:
    # The per-sample mean would be 0.5.
    assert compute_sse(toy_X, np.array([[2.0], [8.0]])) == pytest.approx(2.0)


def test_sse_multidimensional_hand_value() -> None:
    X = np.array([[0.0, 0.0], [3.0, 4.0], [10.0, 10.0]])
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    # 0 + min(25, 49 + 36) + 0 = 25
    assert compute_sse(X, centroids) == pytest.approx(25.0)


def test_sse_matches_brute_force_on_random_data() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    centroids = rng.normal(size=(4, 3))
    expected = sum(min(((x - c) ** 2).sum() for c in centroids) for x in X)
    assert compute_sse(X, centroids) == pytest.approx(expected)


def test_sse_is_pure(toy_X: FloatArray) -> None:
    X = toy_X.copy()
    centroids = np.array([[2.0], [8.0]])
    centroids_before = centroids.copy()
    compute_sse(X, centroids)
    np.testing.assert_array_equal(X, toy_X)
    np.testing.assert_array_equal(centroids, centroids_before)


def test_sse_returns_plain_finite_nonnegative_float(toy_X: FloatArray) -> None:
    value = compute_sse(toy_X, np.array([[0.0], [0.0]]))
    assert type(value) is float
    assert np.isfinite(value)
    assert value >= 0.0


def test_sse_is_defined_with_an_empty_cluster(toy_X: FloatArray) -> None:
    # The centroid at 100 owns nothing. SSE is still the sum over the other centroid.
    # mean 5, squared deviations 16 + 9 + 9 + 16 = 50.
    assert compute_sse(toy_X, np.array([[5.0], [100.0]])) == pytest.approx(50.0)


def test_identical_centroids_match_a_single_centroid(toy_X: FloatArray) -> None:
    assert compute_sse(toy_X, np.array([[5.0], [5.0]])) == pytest.approx(
        compute_sse(toy_X, np.array([[5.0]]))
    )


def test_duplicate_samples_each_count(toy_X: FloatArray) -> None:
    X = np.vstack([toy_X, toy_X])
    centroids = np.array([[2.0], [8.0]])
    assert compute_sse(X, centroids) == pytest.approx(2 * compute_sse(toy_X, centroids))


def test_unchecked_and_checked_agree(toy_X: FloatArray) -> None:
    centroids = np.array([[1.5], [8.5]])
    assert sse_unchecked(toy_X, centroids) == compute_sse(toy_X, centroids)


@pytest.mark.parametrize(
    "centroids",
    [
        np.array([[np.nan], [1.0]]),
        np.array([[np.inf], [1.0]]),
        np.array([1.0, 2.0]),
        np.array([[1.0, 2.0]]),
        np.empty((0, 1)),
    ],
)
def test_invalid_centroids_are_rejected(toy_X: FloatArray, centroids: FloatArray) -> None:
    with pytest.raises(DataValidationError):
        compute_sse(toy_X, centroids)


def test_overflow_raises_instead_of_returning_infinity() -> None:
    with pytest.raises(FloatingPointError, match="overflowed"):
        compute_sse(np.array([[1e308]]), np.array([[-1e308]]))
