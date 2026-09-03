"""Mathematical tests for Lloyd k-means and its initializers."""

# Test names document the behavior directly; individual docstrings add no information.
# ruff: noqa: D103

import numpy as np
import pytest

from bp_k_means.algos.k_means import (
    KMeans,
    kmeans,
    kmeans_plus_plus_init,
    random_init,
    subsampled_kmeans_plus_plus_init,
)


def test_kmeans_plus_plus_keeps_existing_centroids_and_draws_data_points() -> None:
    X = np.array([[0.0], [2.0], [10.0], [12.0]])
    existing = np.array([[1.0]])

    result = kmeans_plus_plus_init(X, 3, seed=7, existing_centroids=existing)

    assert result.shape == (3, 1)
    np.testing.assert_array_equal(result[0], existing[0])
    assert all(any(np.array_equal(row, point) for point in X) for row in result[1:])


def test_kmeans_plus_plus_rejects_too_many_existing_centroids() -> None:
    with pytest.raises(ValueError, match="Existing centroids"):
        kmeans_plus_plus_init(
            np.array([[0.0], [1.0]]), 1, seed=0,
            existing_centroids=np.array([[0.0], [1.0]]),
        )


def test_random_init_selects_distinct_points_and_preserves_existing_centroids() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    existing = np.array([[1.0]])

    result = random_init(X, 3, seed=4, existing_centroids=existing)

    np.testing.assert_array_equal(result[0], existing[0])
    assert len({tuple(row) for row in result}) == 3
    assert {tuple(row) for row in result[1:]} <= {
        tuple(row) for row in X if row[0] != 1.0
    }


def test_subsampled_initializer_only_returns_points_from_the_input() -> None:
    X = np.arange(20.0).reshape(10, 2)
    result = subsampled_kmeans_plus_plus_init(X, 4, subsample_size=5, seed=12)

    assert result.shape == (4, 2)
    assert all(any(np.array_equal(row, point) for point in X) for row in result)


def test_kmeans_returns_the_exact_means_for_a_separable_dataset() -> None:
    X = np.array([[0.0], [1.0], [9.0], [10.0]])
    initial = np.array([[0.0], [10.0]])

    labels, centroids = kmeans(X, 2, seed=0, init_centroids=initial)

    np.testing.assert_array_equal(labels, [0, 0, 1, 1])
    np.testing.assert_allclose(centroids, [[0.5], [9.5]])


def test_kmeans_accepts_precomputed_squared_norms() -> None:
    X = np.array([[0.0, 0.0], [1.0, 2.0], [9.0, 0.0], [10.0, 1.0]])
    initial = np.array([[0.0, 0.0], [10.0, 1.0]])
    X2 = np.einsum("ij,ij->i", X, X)

    without_cache = kmeans(X, 2, seed=3, init_centroids=initial)
    with_cache = kmeans(X, 2, seed=3, init_centroids=initial, X2=X2)

    np.testing.assert_array_equal(with_cache[0], without_cache[0])
    np.testing.assert_allclose(with_cache[1], without_cache[1])


def test_kmeans_repairs_an_empty_cluster_using_the_worst_point() -> None:
    X = np.array([[0.0], [10.0], [20.0]])
    initial = np.array([[0.0], [0.0], [20.0]])

    labels, centroids = kmeans(X, 3, seed=0, init_centroids=initial)

    np.testing.assert_array_equal(labels, [0, 1, 2])
    np.testing.assert_allclose(centroids, X)


def test_kmeans_rejects_an_initialization_with_the_wrong_cluster_count() -> None:
    with pytest.raises(ValueError, match="Expected 2 centroids"):
        kmeans(np.array([[0.0], [1.0]]), 2, seed=0, init_centroids=np.array([[0.0]]))


def test_kmeans_wrapper_stores_centroids_and_fit_predict_returns_same_labels() -> None:
    X = np.array([[0.0], [1.0], [9.0], [10.0]])
    model = KMeans(seed=5, n_init=3)

    predicted = model.fit_predict(X, y=None, target_k=2)

    assert predicted is model.labels_
    np.testing.assert_allclose(np.sort(model.centroids_, axis=0), [[0.5], [9.5]], atol=1e-12)
    assert np.isclose(np.sum((X - model.centroids_[predicted]) ** 2), 1.0)


def test_kmeans_wrapper_rejects_zero_initializations() -> None:
    with pytest.raises(ValueError, match="n_init"):
        KMeans(seed=0, n_init=0)
