"""Tests for optimized bisecting k-means and its split helpers."""

# Test names document the behavior directly; individual docstrings add no information.
# ruff: noqa: D103, ANN001

import numpy as np
import pytest
from conftest import assert_label_pure, direct_wcss

from bp_k_means.algos.bisecting_k_means_m_rl_optimized import (
    BisectingKMeansMRL,
    BisectingKMeansMRLNoRefine,
    ClusterNode,
    bisecting_kmeans_m_rl_by_label_optimized,
    bisecting_kmeans_m_rl_by_label_optimized_no_refine,
)
from bp_k_means.algos.bisecting_k_means_optimized import (
    BisectingKMeans,
    BisectingKMeansNoRefine,
    _best_bisecting_split,
    _push_cluster_candidates,
    _refine_cluster_split,
    _validate_inputs,
    bisecting_kmeans_by_label_optimized,
    bisecting_kmeans_by_label_optimized_no_refine,
)


def test_validate_inputs_normalizes_arrays_and_checks_feasibility() -> None:
    X, y, n_samples, n_labels = _validate_inputs(
        [[0.0], [1.0], [10.0]], ["a", "a", "b"], 2, 1,
    )

    assert X.shape == (3, 1)
    assert y.shape == (3,)
    assert (n_samples, n_labels) == (3, 2)


@pytest.mark.parametrize(
    ("target_k", "n_init", "message"),
    [(1, 1, "target_k"), (2, 0, "n_init")],
)
def test_validate_inputs_rejects_invalid_settings(target_k: int, n_init: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_inputs(
            np.array([[0.0], [1.0], [10.0]]), np.array(["a", "a", "b"]), target_k, n_init,
        )


def test_best_bisecting_split_returns_two_clusters_and_exact_wcss() -> None:
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    X2 = np.einsum("ij,ij->i", X, X)

    labels, centroids, per_cluster, counts = _best_bisecting_split(
        X, X2, 1, np.random.default_rng(0),
    )

    assert sorted(counts.tolist()) == [2, 2]
    assert np.sum(per_cluster) == pytest.approx(1.0)
    assert np.sum((X - centroids[labels]) ** 2) == pytest.approx(1.0)


def test_refine_cluster_split_returns_consistent_counts_and_exact_wcss() -> None:
    pts = np.array([[0.0], [1.0], [10.0], [11.0]])
    X2 = np.einsum("ij,ij->i", pts, pts)

    wcss, labels, centroids, counts = _refine_cluster_split(
        pts, pts, X2, float(X2.sum()), np.array([[5.5]]), 0, 1,
        np.random.default_rng(0),
    )

    assert wcss == pytest.approx(1.0)
    assert np.bincount(labels, minlength=2).tolist() == counts.tolist()
    np.testing.assert_allclose(np.sort(centroids[:, 0]), [0.5, 10.5])


def test_push_cluster_candidates_uses_requested_priority_formula() -> None:
    heap: list[tuple[float, int, int, int]] = []
    _push_cluster_candidates(
        heap, 4, np.array([3, 2, 1]), np.array([9.0, 4.0, 2.0]), 3, 10, 7,
        use_wcss_per_cluster=True,
    )

    assert sorted(heap) == [(-2.25, 4, 0, 7), (-1.0, 4, 1, 7)]


def test_cluster_node_stores_split_candidate_state() -> None:
    node = ClusterNode(3, 1, np.array([True, False]), np.array([[2.0]]), 5.0, 9.0)

    assert node.cluster_id == 3
    assert node.lbl_idx == 1
    assert node.wcss == 5.0
    assert node.X2_sum == 9.0
    assert node.split_info is None


@pytest.mark.parametrize(
    "algorithm",
    [
        bisecting_kmeans_by_label_optimized,
        bisecting_kmeans_by_label_optimized_no_refine,
        bisecting_kmeans_m_rl_by_label_optimized,
        bisecting_kmeans_m_rl_by_label_optimized_no_refine,
    ],
)
def test_all_bisecting_variants_reach_target_and_preserve_labels(
    four_point_groups: tuple[np.ndarray, np.ndarray], algorithm,
) -> None:
    X, y = four_point_groups

    labels = algorithm(X, y, 5, seed=5, n_init=2)

    assert len(np.unique(labels)) == 5
    assert_label_pure(labels, y)
    assert direct_wcss(labels=labels, X=X) <= direct_wcss(
        X, np.array([0] * 4 + [1] * 4),
    )


@pytest.mark.parametrize(
    "algorithm",
    [
        bisecting_kmeans_by_label_optimized,
        bisecting_kmeans_by_label_optimized_no_refine,
        bisecting_kmeans_m_rl_by_label_optimized,
        bisecting_kmeans_m_rl_by_label_optimized_no_refine,
    ],
)
def test_all_bisecting_variants_return_identity_at_maximum_target(
    three_two_point_groups: tuple[np.ndarray, np.ndarray], algorithm,
) -> None:
    X, y = three_two_point_groups

    np.testing.assert_array_equal(
        algorithm(X, y, len(X), seed=0, n_init=1), np.arange(len(X)),
    )


@pytest.mark.parametrize(
    "algorithm",
    [bisecting_kmeans_by_label_optimized, bisecting_kmeans_by_label_optimized_no_refine],
)
def test_bisecting_variants_reject_infeasible_targets(algorithm) -> None:
    X = np.array([[0.0], [1.0], [10.0]])
    y = np.array(["a", "a", "b"])

    with pytest.raises(ValueError, match="target_k"):
        algorithm(X, y, 1, seed=0, n_init=1)
    with pytest.raises(ValueError, match="n_init"):
        algorithm(X, y, 2, seed=0, n_init=0)


@pytest.mark.parametrize(
    "wrapper",
    [
        BisectingKMeans,
        BisectingKMeansNoRefine,
        BisectingKMeansMRL,
        BisectingKMeansMRLNoRefine,
    ],
)
def test_bisecting_wrappers_require_labels_and_store_results(wrapper) -> None:
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array(["a", "a", "b", "b"])
    model = wrapper(seed=0, n_init=1)

    with pytest.raises(ValueError, match="original labels"):
        model.fit(X, None, 2)
    assert model.fit(X, y, 3) is model
    assert model.labels_ is not None
    assert_label_pure(model.labels_, y)
