"""Tests for Ward linkage and nearest-neighbor-chain HAC."""

# Test names document the behavior directly; individual docstrings add no information.
# ruff: noqa: D103, ANN001

import numpy as np
import pytest
from conftest import assert_label_pure, canonical_partition

from bp_k_means.algos.hac import (
    HACWard,
    HACWardNNC,
    _assign_ward_labels,
    _build_ward_queue,
    _merge_nnc_clusters,
    _merge_ward_clusters,
    _nearest_neighbor,
    _next_nnc_merge,
    _ward_distance,
    hac_ward_by_label,
    hac_ward_nnc_by_label,
)


def test_ward_distance_is_the_increase_in_within_cluster_sse() -> None:
    sizes = {0: 2, 1: 3}
    centroids = {0: np.array([0.0]), 1: np.array([5.0])}

    assert _ward_distance(0, 1, sizes, centroids) == pytest.approx(30.0)


def test_build_ward_queue_contains_only_same_label_pairs() -> None:
    centroids = {i: np.array([float(i)]) for i in range(4)}
    queue = _build_ward_queue({"a": [0, 2, 3], "b": [1]}, centroids)

    assert sorted(queue) == [(0.5, 2, 3), (2.0, 0, 2), (4.5, 0, 3)]


def test_merge_ward_clusters_updates_membership_mean_size_and_queue() -> None:
    clusters = {0: [0], 1: [1], 2: [2]}
    cluster_label = {0: "a", 1: "a", 2: "a"}
    active = {0, 1, 2}
    sizes = {0: 1, 1: 1, 2: 1}
    centroids = {i: np.array([float(i)]) for i in range(3)}
    queue: list[tuple[float, int, int]] = []

    _merge_ward_clusters(0, 1, 3, clusters, cluster_label, active, sizes, centroids, queue)

    assert clusters == {2: [2], 3: [0, 1]}
    assert active == {2, 3}
    assert sizes[3] == 2
    np.testing.assert_allclose(centroids[3], [0.5])
    assert queue == [(1.5, 3, 2)]


def test_assign_ward_labels_renumbers_sorted_active_clusters() -> None:
    labels = _assign_ward_labels({4, 8}, {4: [0, 2], 8: [1, 3]}, 4)

    np.testing.assert_array_equal(labels, [0, 1, 0, 1])


def test_nearest_neighbor_uses_ward_cost() -> None:
    sizes = {0: 1, 1: 1, 2: 1}
    centroids = {0: np.array([0.0]), 1: np.array([3.0]), 2: np.array([1.0])}

    assert _nearest_neighbor(0, {0, 1, 2}, sizes, centroids) == 2


def test_next_nnc_merge_returns_a_reciprocal_pair_and_ward_cost() -> None:
    sizes = {0: 1, 1: 1, 2: 1}
    centroids = {0: np.array([0.0]), 1: np.array([3.0]), 2: np.array([1.0])}

    cost, first, second = _next_nnc_merge({0, 1, 2}, sizes, centroids)

    assert (first, second) == (2, 0)
    assert cost == pytest.approx(0.5)


def test_merge_nnc_clusters_updates_active_cluster_state() -> None:
    cluster_ids = {0, 1}
    active = {0, 1}
    sizes = {0: 2, 1: 1}
    centroids = {0: np.array([0.0]), 1: np.array([3.0])}

    _merge_nnc_clusters(0, 1, 2, cluster_ids, active, sizes, centroids)

    assert cluster_ids == {2}
    assert active == {2}
    assert sizes == {2: 3}
    np.testing.assert_allclose(centroids[2], [1.0])


@pytest.mark.parametrize("algorithm", [hac_ward_by_label, hac_ward_nnc_by_label])
def test_hac_variants_make_expected_nearest_pair_merges(algorithm) -> None:
    X = np.array([[0.0], [1.0], [10.0], [100.0], [101.0], [110.0]])
    y = np.array(["a", "a", "a", "b", "b", "b"])

    labels = algorithm(X, y, target_k=4)

    np.testing.assert_array_equal(canonical_partition(labels), [0, 0, 1, 2, 2, 3])
    assert_label_pure(labels, y)


def test_nnc_uses_the_lowest_ward_merge_for_a_requested_cut() -> None:
    X = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 3.0], [1.0, 1.0]])
    y = np.array(["a"] * len(X))

    labels = hac_ward_nnc_by_label(X, y, target_k=3)

    np.testing.assert_array_equal(canonical_partition(labels), [0, 1, 1, 2])


def test_nnc_preserves_merged_memberships_instead_of_reassigning_by_centroid() -> None:
    X = np.array(
        [[0.0, 0.0], [0.0, 2.0], [1.0, 1.0], [2.0, 1.0], [2.0, 3.0], [3.0, 0.0]]
    )
    y = np.array(["a"] * len(X))

    nnc_labels = hac_ward_nnc_by_label(X, y, target_k=3)
    ward_labels = hac_ward_by_label(X, y, target_k=3)

    np.testing.assert_array_equal(canonical_partition(nnc_labels), [0, 0, 1, 1, 2, 1])
    np.testing.assert_array_equal(canonical_partition(nnc_labels), canonical_partition(ward_labels))


@pytest.mark.parametrize("algorithm", [hac_ward_by_label, hac_ward_nnc_by_label])
def test_hac_variants_return_identity_at_one_cluster_per_point(algorithm) -> None:
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array(["a", "a", "b", "b"])

    np.testing.assert_array_equal(algorithm(X, y, len(X)), np.arange(len(X)))


@pytest.mark.parametrize("wrapper", [HACWard, HACWardNNC])
def test_hac_wrappers_require_labels_and_store_labels(wrapper) -> None:
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array(["a", "a", "b", "b"])
    model = wrapper(seed=0, n_init=1)

    with pytest.raises(ValueError, match="requires original labels"):
        model.fit(X, None, 2)
    assert model.fit(X, y, 3) is model
    assert model.labels_ is not None
    assert_label_pure(model.labels_, y)


def test_hac_rejects_a_target_below_the_number_of_labels() -> None:
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array(["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="number of labels"):
        hac_ward_by_label(X, y, 1)
