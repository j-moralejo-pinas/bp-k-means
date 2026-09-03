"""Mathematical and behavioral tests for BP-k-means."""

# Test names document the behavior directly; individual docstrings add no information.
# ruff: noqa: D103

import numpy as np
import pytest
from conftest import assert_label_pure, direct_wcss

from bp_k_means.algos.bp_kmeans import (
    BPKMeans,
    InitAlgorithm,
    InitStrategy,
    RankingMetric,
    _build_init_centroids,
    _compute_metric,
    _run_split,
    _wcss_per_cluster,
    bp_kmeans,
)
from bp_k_means.utils.metrics import overall_wcss


def test_wcss_per_cluster_matches_direct_sum_of_squared_deviations() -> None:
    X = np.array([[0.0], [2.0], [10.0]])
    local_labels = np.array([0, 0, 1])
    centroids = np.array([[1.0], [10.0], [99.0]])
    X2 = np.einsum("ij,ij->i", X, X)

    result = _wcss_per_cluster(local_labels, X2, centroids, k=3)

    np.testing.assert_allclose(result, [2.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (RankingMetric.M_L, 4.0),
        (RankingMetric.M_C, 2.0),
        (RankingMetric.M_ERL, 4.0 / 3.0),
    ],
)
def test_ranking_metrics_follow_their_definitions(metric: RankingMetric, expected: float) -> None:
    X2 = np.array([0.0, 4.0, 100.0, 144.0])
    local_labels = np.array([0, 0, 1, 1])
    centroids = np.array([[1.0], [11.0]])

    assert _compute_metric(
        metric, 4.0, local_labels, X2, centroids, 2, 4,
    ) == pytest.approx(expected)


def test_estimated_reduction_metric_handles_the_last_possible_split() -> None:
    local_labels = np.array([0, 1, 2])
    X2 = np.array([0.0, 1.0, 4.0])
    centroids = np.array([[0.0], [1.0], [2.0]])

    assert _compute_metric(RankingMetric.M_ERL, 8.0, local_labels, X2, centroids, 2, 3) == 8.0
    assert _compute_metric(RankingMetric.M_ERL, 8.0, local_labels, X2, centroids, 3, 3) == 0.0


def test_exact_reduction_metric_is_rejected_by_the_simple_metric_helper() -> None:
    with pytest.raises(ValueError, match="Unsupported ranking metric"):
        _compute_metric(
            RankingMetric.M_RL, 1.0, np.array([0]), np.array([0.0]),
            np.array([[0.0]]), 1, 1,
        )


@pytest.mark.parametrize("algorithm", list(InitAlgorithm))
def test_acl_initialization_keeps_all_existing_centroids(algorithm: InitAlgorithm) -> None:
    pts = np.array([[0.0], [1.0], [10.0], [11.0]])
    existing = np.array([[0.5], [10.5]])

    result = _build_init_centroids(
        InitStrategy.I_ACL, pts, existing, 2, 3, np.random.default_rng(2),
        init_algorithm=algorithm, subsample_size=3,
    )

    np.testing.assert_allclose(result[:2], existing)
    assert result.shape == (3, 1)


def test_lri_with_exactly_new_k_points_returns_all_points() -> None:
    pts = np.array([[0.0], [1.0], [2.0]])
    result = _build_init_centroids(
        InitStrategy.I_LRI, pts, np.array([[0.5]]), 1, 3, np.random.default_rng(0),
        init_algorithm=InitAlgorithm.RANDOM_SAMPLING, subsample_size=2,
    )

    np.testing.assert_array_equal(result, pts)


@pytest.mark.parametrize("strategy", [InitStrategy.I_CRI, InitStrategy.I_ACC])
def test_cluster_initialization_strategies_keep_non_target_centroids(
    strategy: InitStrategy,
) -> None:
    pts = np.array([[0.0], [1.0], [10.0], [11.0]])
    current = np.array([[0.5], [10.5]])
    target_pts = pts[:2]

    result = _build_init_centroids(
        strategy, pts, current, 2, 3, np.random.default_rng(0), target_pts, 0,
        init_algorithm=InitAlgorithm.KMEANS_PLUS_PLUS, subsample_size=3,
    )

    np.testing.assert_allclose(result[0], current[1])
    np.testing.assert_array_equal(result[1:], target_pts)


def test_init_centroids_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="Cannot initialize"):
        _build_init_centroids(
            InitStrategy.I_LRI, np.array([[0.0], [1.0]]), np.array([[0.0]]), 1, 3,
            np.random.default_rng(0), subsample_size=2,
        )


def test_run_split_finds_the_mathematical_two_pair_solution() -> None:
    pts = np.array([[0.0], [1.0], [10.0], [11.0]])
    X2 = np.einsum("ij,ij->i", pts, pts)

    wcss, labels, centroids = _run_split(
        pts, X2, float(X2.sum()), np.zeros(4, dtype=int), np.array([[5.5]]),
        1, 2, 1, np.random.default_rng(0), InitStrategy.I_CRI,
        subsample_size=4,
    )

    assert wcss == pytest.approx(1.0)
    assert len(np.unique(labels)) == 2
    np.testing.assert_allclose(np.sort(centroids[:, 0]), [0.5, 10.5])


@pytest.mark.parametrize("ranking_metric", list(RankingMetric))
@pytest.mark.parametrize("init_strategy", list(InitStrategy))
@pytest.mark.parametrize("init_algorithm", list(InitAlgorithm))
def test_bp_kmeans_returns_target_count_and_label_pure_clusters(
    four_point_groups: tuple[np.ndarray, np.ndarray],
    ranking_metric: RankingMetric,
    init_strategy: InitStrategy,
    init_algorithm: InitAlgorithm,
) -> None:
    X, y = four_point_groups
    labels = bp_kmeans(
        X, y, 5, seed=8, n_init=1, subsample_size=3,
        ranking_metric=ranking_metric, init_strategy=init_strategy,
        init_algorithm=init_algorithm,
    )

    assert labels.shape == (len(X),)
    assert len(np.unique(labels)) == 5
    assert_label_pure(labels, y)
    assert overall_wcss(X, labels) <= overall_wcss(X, np.array([0] * 4 + [1] * 4))


def test_bp_kmeans_returns_identity_when_every_point_is_a_cluster(
    four_point_groups: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = four_point_groups

    np.testing.assert_array_equal(
        bp_kmeans(X, y, len(X), seed=0, n_init=1, subsample_size=2), np.arange(len(X))
    )


def test_bp_kmeans_rejects_infeasible_target_counts(
    two_label_points: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = two_label_points
    with pytest.raises(ValueError, match="larger"):
        bp_kmeans(X, y, len(X) + 1, seed=0, n_init=1, subsample_size=2)
    with pytest.raises(ValueError, match="smaller"):
        bp_kmeans(X, y, 1, seed=0, n_init=1, subsample_size=2)


def test_bp_wrapper_requires_source_labels_and_stores_result(
    two_label_points: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = two_label_points
    model = BPKMeans(seed=0, n_init=1, subsample_size=2)

    with pytest.raises(ValueError, match="original labels"):
        model.fit(X, None, 2)
    fitted = model.fit(X, y, 3)

    assert fitted is model
    assert_label_pure(model.labels_, y)
    assert len(np.unique(model.labels_)) == 3


def test_overall_wcss_is_the_direct_partition_sum() -> None:
    X = np.array([[0.0], [2.0], [10.0], [14.0]])
    labels = np.array([0, 0, 1, 1])

    assert overall_wcss(X, labels) == pytest.approx(direct_wcss(X, labels))
