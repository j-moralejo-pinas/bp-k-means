"""Tests for cannot-link assignment, updates, and COP-k-means."""

# Test names document the behavior directly; individual docstrings add no information.
# ruff: noqa: D103, FBT001

import numpy as np
import pytest
from conftest import assert_label_pure, direct_wcss

from bp_k_means.algos.cop_k_means import (
    COPKMeans,
    _assign_points,
    _update_centroids,
    cop_kmeans_by_label,
)


def test_assign_points_uses_the_nearest_feasible_cluster() -> None:
    X = np.array([[0.0], [9.0], [10.0]])
    y = np.array(["a", "b", "a"])
    centroids = np.array([[0.0], [10.0]])
    labels = np.array([-1, -1, -1])

    result = _assign_points(X, y, centroids, labels)

    assert result is not None
    assigned, changed = result
    np.testing.assert_array_equal(assigned, [0, 1, 0])
    assert changed


def test_assign_points_skips_a_nearer_cluster_with_a_conflicting_label() -> None:
    X = np.array([[0.0], [1.0], [10.0]])
    y = np.array(["a", "b", "a"])
    centroids = np.array([[0.0], [10.0]])
    labels = np.array([0, 0, 1])

    result = _assign_points(X, y, centroids, labels)

    assert result is not None
    assigned, _ = result
    np.testing.assert_array_equal(assigned, [1, 0, 1])


def test_assign_points_reports_when_all_clusters_are_infeasible() -> None:
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array(["a", "b", "b", "a"])
    centroids = np.array([[0.0], [10.0]])
    labels = np.array([0, 0, 1, 1])

    assert _assign_points(X, y, centroids, labels) is None


def test_update_centroids_computes_means_and_reseeds_empty_clusters() -> None:
    X = np.array([[0.0], [2.0], [10.0]])
    labels = np.array([0, 0, 1])

    result = _update_centroids(X, labels, np.zeros((3, 1)), np.random.default_rng(1))

    np.testing.assert_allclose(result[:2], [[1.0], [10.0]])
    assert any(np.array_equal(result[2], point) for point in X)


@pytest.mark.parametrize("ensure_label", [True, False])
def test_cop_kmeans_is_label_pure_and_returns_cluster_means(
    two_label_points: tuple[np.ndarray, np.ndarray], ensure_label: bool,
) -> None:
    X, y = two_label_points
    labels, centroids = cop_kmeans_by_label(
        X, y, 2, seed=3, init_ensure_label=ensure_label,
    )

    assert labels is not None
    assert centroids is not None
    assert len(np.unique(labels)) == 2
    assert_label_pure(labels, y)
    np.testing.assert_allclose(centroids, [[0.5], [10.5]])
    assert direct_wcss(X, labels) == pytest.approx(1.0)


def test_cop_kmeans_rejects_fewer_clusters_than_labels() -> None:
    with pytest.raises(ValueError, match="number of labels"):
        cop_kmeans_by_label(np.array([[0.0], [1.0]]), np.array(["a", "b"]), 1, seed=0)


def test_cop_wrapper_requires_labels_and_retains_best_feasible_result(
    two_label_points: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = two_label_points
    model = COPKMeans(seed=0, n_init=2)

    with pytest.raises(ValueError, match="requires original labels"):
        model.fit(X, None, 2)
    assert model.fit_predict(X, y, 2) is model.labels_
    assert_label_pure(model.labels_, y)
    assert model.centroids_ is not None
