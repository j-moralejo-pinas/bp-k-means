"""Label-constrained hierarchical agglomerative clustering routines."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import TYPE_CHECKING, Any

import numpy as np

from bp_k_means.algos.base_algo import BaseAlgo

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

MIN_CLUSTER_SIZE = 2


def _build_ward_queue(
    label_to_indices: dict[object, list[int]],
    centroids: dict[int, np.ndarray],
) -> list[tuple[float, int, int]]:
    """
    Build the initial same-label Ward priority queue.

    Parameters
    ----------
    label_to_indices : dict[object, list[int]]
        Mapping from source labels to their point or cluster identifiers.
    centroids : dict[int, np.ndarray]
        Mapping from cluster identifier to centroid.

    Returns
    -------
    list[tuple[float, int, int]]
        Heap entries ``(merge_cost, cluster_a, cluster_b)``.
    """
    pq = []
    for idxs in label_to_indices.values():
        for ii, i in enumerate(idxs):
            for j in idxs[ii + 1 :]:
                diff = centroids[i] - centroids[j]
                cost = 0.5 * np.sum(diff * diff)
                heappush(pq, (cost, i, j))
    return pq


def _merge_ward_clusters(
    a: int,
    b: int,
    new_id: int,
    clusters: dict[int, list[int]],
    cluster_label: dict[int, Any],
    active: set[int],
    sizes: dict[int, int],
    centroids: dict[int, np.ndarray],
    pq: list[tuple[float, int, int]],
) -> None:
    """
    Merge two active Ward clusters and enqueue their new candidates.

    Parameters
    ----------
    a : int
        First active cluster identifier to merge.
    b : int
        Second active cluster identifier to merge.
    new_id : int
        Identifier assigned to the merged cluster.
    clusters : dict[int, list[int]]
        Mutable cluster-membership mapping.
    cluster_label : dict[int, Any]
        Mapping from cluster identifier to source label.
    active : set[int]
        Mutable set of active cluster identifiers.
    sizes : dict[int, int]
        Mutable cluster-size mapping.
    centroids : dict[int, np.ndarray]
        Mutable centroid mapping.
    pq : list[tuple[float, int, int]]
        Mutable Ward priority queue.
    """
    clusters[new_id] = clusters[a] + clusters[b]
    cluster_label[new_id] = cluster_label[a]
    active.remove(a)
    active.remove(b)
    active.add(new_id)

    n_a = sizes[a]
    n_b = sizes[b]
    sizes[new_id] = n_a + n_b
    centroids[new_id] = (n_a * centroids[a] + n_b * centroids[b]) / (n_a + n_b)

    del clusters[a], clusters[b]
    del centroids[a], centroids[b]
    del sizes[a], sizes[b]

    for other in active:
        if other == new_id or cluster_label[other] != cluster_label[new_id]:
            continue
        n_other = sizes[other]
        diff = centroids[new_id] - centroids[other]
        cost = (sizes[new_id] * n_other) / (sizes[new_id] + n_other) * np.sum(diff * diff)
        heappush(pq, (cost, new_id, other))


def _assign_ward_labels(
    active: set[int], clusters: dict[int, list[int]], n_samples: int
) -> np.ndarray:
    """
    Renumber active Ward clusters and assign their member points.

    Parameters
    ----------
    active : set[int]
        Active cluster identifiers.
    clusters : dict[int, list[int]]
        Mapping from cluster identifier to member point indices.
    n_samples : int
        Number of input points.

    Returns
    -------
    np.ndarray
        Dense cluster labels indexed by input point.
    """
    active_list = sorted(active)
    new_id_map = {old: i for i, old in enumerate(active_list)}
    labels_final = np.zeros(n_samples, dtype=int)
    for old_cid in active_list:
        new_cid = new_id_map[old_cid]
        for idx in clusters[old_cid]:
            labels_final[idx] = new_cid
    return labels_final


def hac_ward_by_label(X: np.ndarray, y: np.ndarray, target_k: int) -> np.ndarray:
    """
    Hierarchical agglomerative clustering using Ward's criterion.

    Clusters are restricted to merging only with clusters of the same label.

    Parameters
    ----------
    X : np.ndarray
        Feature array with shape (n, d).
    y : np.ndarray
        Labels with shape (n, e.g. CUSEC).
    target_k : int
        Desired number of clusters.

    Returns
    -------
    np.ndarray
        Cluster ids in [0, target_k - 1].

    Raises
    ------
    ValueError
        If the target is below the number of unique labels or no feasible merge remains.
    RuntimeError
        If the final active-cluster count is inconsistent.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n, _d = X.shape

    # Safety: cannot have fewer clusters than unique labels
    unique_labels = np.unique(y)
    min_clusters = len(unique_labels)
    if target_k < min_clusters:
        msg = (
            f"target_k={target_k} is smaller than the number of labels={min_clusters}. "
            f"With the 'same-label only' constraint you cannot go below {min_clusters} clusters."
        )
        raise ValueError(msg)

    # ---------------------------------------------------
    # INITIAL CLUSTERS: each point is its own cluster
    # ---------------------------------------------------
    clusters = {i: [i] for i in range(n)}  # cluster_id -> list of point indices
    cluster_label = {i: y[i] for i in range(n)}  # cluster_id -> label
    active = set(range(n))  # active cluster ids
    sizes = dict.fromkeys(range(n), 1)  # cluster_id -> size
    centroids = {i: X[i].copy() for i in range(n)}  # cluster_id -> centroid

    label_to_indices = {}
    for idx, lbl in enumerate(y):
        label_to_indices.setdefault(lbl, []).append(idx)
    pq = _build_ward_queue(label_to_indices, centroids)

    next_cid = n  # new cluster ids start after the original n

    # ---------------------------------------------------
    # HAC MERGING LOOP (Ward)
    # ---------------------------------------------------
    while len(active) > target_k:
        if not pq:
            # No more possible same-label merges but still > target_k clusters
            msg = (
                f"No feasible merges left for label-restricted Ward HAC. "
                f"active_clusters={len(active)}, target_k={target_k}."
            )
            raise ValueError(msg)

        _cost, a, b = heappop(pq)

        # Skip outdated clusters
        if a not in active or b not in active:
            continue

        # Just in case, enforce same-label constraint
        if cluster_label[a] != cluster_label[b]:
            continue

        new_id = next_cid
        next_cid += 1
        _merge_ward_clusters(a, b, new_id, clusters, cluster_label, active, sizes, centroids, pq)

    # Sanity check: we must have exactly target_k active clusters
    if len(active) != target_k:
        msg = f"Internal error: len(active)={len(active)} but target_k={target_k}"
        raise RuntimeError(msg)

    return _assign_ward_labels(active, clusters, n)


def _ward_distance(
    a: int, b: int, sizes: dict[int, int], centroids: dict[int, np.ndarray]
) -> float:
    """
    Return Ward's merge cost for two clusters.

    Parameters
    ----------
    a : int
        First cluster identifier.
    b : int
        Second cluster identifier.
    sizes : dict[int, int]
        Cluster sizes indexed by identifier.
    centroids : dict[int, np.ndarray]
        Cluster centroids indexed by identifier.

    Returns
    -------
    float
        Increase in within-cluster sum of squares caused by the merge.
    """
    n_a, n_b = sizes[a], sizes[b]
    diff = centroids[a] - centroids[b]
    return (n_a * n_b) / (n_a + n_b) * np.dot(diff, diff)


def _nearest_neighbor(
    cluster_id: int,
    cluster_ids: set[int],
    sizes: dict[int, int],
    centroids: dict[int, np.ndarray],
) -> int:
    """
    Find the nearest cluster in one label group.

    Parameters
    ----------
    cluster_id : int
        Cluster whose neighbor is requested.
    cluster_ids : set[int]
        Candidate clusters in the same source-label group.
    sizes : dict[int, int]
        Cluster sizes indexed by identifier.
    centroids : dict[int, np.ndarray]
        Cluster centroids indexed by identifier.

    Returns
    -------
    int
        Identifier of the lowest-cost candidate, or ``-1`` if none exists.
    """
    best_cluster = -1
    best_distance = np.inf
    for candidate in cluster_ids:
        if candidate == cluster_id:
            continue
        distance = _ward_distance(cluster_id, candidate, sizes, centroids)
        if distance < best_distance:
            best_distance = distance
            best_cluster = candidate
    return best_cluster


def _next_nnc_merge(
    cluster_ids: set[int],
    sizes: dict[int, int],
    centroids: dict[int, np.ndarray],
) -> tuple[float, int, int]:
    """
    Find a reciprocal nearest-neighbor merge for one label group.

    Parameters
    ----------
    cluster_ids : set[int]
        Active clusters in one source-label group.
    sizes : dict[int, int]
        Cluster sizes indexed by identifier.
    centroids : dict[int, np.ndarray]
        Cluster centroids indexed by identifier.

    Returns
    -------
    tuple[float, int, int]
        Ward cost and the reciprocal cluster identifiers.
    """
    chain = [next(iter(cluster_ids))]
    while True:
        a = chain[-1]
        best_b = _nearest_neighbor(a, cluster_ids, sizes, centroids)
        if len(chain) >= MIN_CLUSTER_SIZE and best_b == chain[-2]:
            return _ward_distance(a, best_b, sizes, centroids), a, best_b
        chain.append(best_b)


def _build_nnc_hierarchy(
    cluster_ids: set[int],
    sizes: dict[int, int],
    centroids: dict[int, np.ndarray],
    cluster_members: dict[int, list[int]],
    next_cid: int,
) -> tuple[list[tuple[float, int, int, int, int]], int]:
    """
    Build one complete nearest-neighbor-chain hierarchy.

    Parameters
    ----------
    cluster_ids : set[int]
        Initial active clusters for one source label.
    sizes : dict[int, int]
        Mutable cluster sizes.
    centroids : dict[int, np.ndarray]
        Mutable cluster centroids.
    cluster_members : dict[int, list[int]]
        Mutable mapping from cluster identifier to point members.
    next_cid : int
        Identifier to use for the first generated parent cluster.

    Returns
    -------
    merge_events : list[tuple[float, int, int, int, int]]
        Merge records ``(cost, depth, left, right, parent)``.
    next_cid : int
        Next unused cluster identifier.
    """
    active = set(cluster_ids)
    depth = dict.fromkeys(cluster_ids, 0)
    merge_events = []

    while len(active) > 1:
        cost, a, b = _next_nnc_merge(active, sizes, centroids)
        cluster_members[next_cid] = cluster_members[a] + cluster_members[b]
        depth[next_cid] = max(depth[a], depth[b]) + 1
        merge_events.append((cost, depth[next_cid], a, b, next_cid))
        _merge_nnc_clusters(a, b, next_cid, active, active, sizes, centroids)
        next_cid += 1

    return merge_events, next_cid


def _merge_nnc_clusters(
    a: int,
    b: int,
    new_id: int,
    cluster_ids: set[int],
    active: set[int],
    sizes: dict[int, int],
    centroids: dict[int, np.ndarray],
) -> None:
    """
    Merge two nearest-neighbor-chain clusters.

    Parameters
    ----------
    a : int
        First cluster identifier to merge.
    b : int
        Second cluster identifier to merge.
    new_id : int
        Identifier for the merged cluster.
    cluster_ids : set[int]
        Mutable cluster set used by the chain.
    active : set[int]
        Mutable active cluster set when distinct from ``cluster_ids``.
    sizes : dict[int, int]
        Mutable cluster sizes.
    centroids : dict[int, np.ndarray]
        Mutable cluster centroids.
    """
    n_a, n_b = sizes[a], sizes[b]
    centroids[new_id] = (n_a * centroids[a] + n_b * centroids[b]) / (n_a + n_b)
    sizes[new_id] = n_a + n_b
    cluster_ids.remove(a)
    cluster_ids.remove(b)
    cluster_ids.add(new_id)
    if active is not cluster_ids:
        active.remove(a)
        active.remove(b)
        active.add(new_id)
    del centroids[a], centroids[b]
    del sizes[a], sizes[b]


def hac_ward_nnc_by_label(X: np.ndarray, y: np.ndarray, target_k: int) -> np.ndarray:
    """
    Hierarchical agglomerative clustering using Ward linkage and a nearest-neighbor chain.

    Merges are restricted to clusters with the same label.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels (merges allowed only within same label)
    target_k : int
        Desired total number of clusters

    Returns
    -------
    np.ndarray
        Cluster labels in [0, target_k-1]

    Raises
    ------
    ValueError
        If the target is below the number of unique labels.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n, _d = X.shape

    labels = np.unique(y)
    if target_k < len(labels):
        msg = "target_k cannot be smaller than number of labels under label-restricted HAC."
        raise ValueError(msg)

    # ------------------------------------------------------------
    # INITIALIZE CLUSTERS
    # ------------------------------------------------------------
    # Global cluster storage
    centroids = {}
    sizes = {}
    active = set()

    # Initial clusters: one per point
    for i in range(n):
        centroids[i] = X[i].copy()
        sizes[i] = 1
        active.add(i)

    next_cid = n

    # Per-label active clusters
    label_clusters = {}
    for lbl in labels:
        label_clusters[lbl] = set(np.where(y == lbl)[0])

    cluster_members = {i: [i] for i in range(n)}
    merge_events = []

    for label in labels:
        label_events, next_cid = _build_nnc_hierarchy(
            label_clusters[label],
            sizes,
            centroids,
            cluster_members,
            next_cid,
        )
        merge_events.extend(label_events)

    merge_events.sort()
    for _cost, _depth, a, b, new_id in merge_events[: n - target_k]:
        active.remove(a)
        active.remove(b)
        active.add(new_id)

    return _assign_ward_labels(active, cluster_members, n)


class _WardPredictor(BaseAlgo):
    """
    Shared prediction rule for fitted Ward hierarchy cuts.

    Notes
    -----
    Prediction assigns each sample to the compatible cluster with the smallest
    Ward insertion cost, rather than simply using Euclidean centroid distance.
    """

    def predict(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> NDArray:
        """
        Assign instances by the Ward cost of joining each fitted cluster.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix to predict.
        y : ArrayLike
            Source label for each input row.

        Returns
        -------
        NDArray
            Predicted cluster identifier for each row.
        """
        X_array, y_array = self._validate_prediction_input(X, y)
        assert self._cluster_sizes is not None
        costs = self._squared_centroid_distances(X_array)
        costs *= self._cluster_sizes / (self._cluster_sizes + 1)
        return self._select_lowest_cost_clusters(costs, y_array)


class HACWard(_WardPredictor):
    """Common-interface wrapper around label-constrained Ward HAC."""

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        target_k: int,
    ) -> HACWard:
        """
        Fit label-constrained Ward hierarchical clustering.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix.
        y : ArrayLike
            Original labels that constrain merges.
        target_k : int
            Requested number of clusters.

        Returns
        -------
        HACWard
            The fitted algorithm instance.
        """
        y_array = np.asarray(y)
        labels = hac_ward_by_label(np.asarray(X), y_array, target_k)
        return self._set_cluster_result(X, y_array, labels)


class HACWardNNC(_WardPredictor):
    """Common-interface wrapper around nearest-neighbor-chain Ward HAC."""

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        target_k: int,
    ) -> HACWardNNC:
        """
        Fit nearest-neighbor-chain Ward hierarchical clustering.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix.
        y : ArrayLike
            Original labels that constrain merges.
        target_k : int
            Requested number of clusters.

        Returns
        -------
        HACWardNNC
            The fitted algorithm instance.
        """
        y_array = np.asarray(y)
        labels = hac_ward_nnc_by_label(np.asarray(X), y_array, target_k)
        return self._set_cluster_result(X, y_array, labels)
