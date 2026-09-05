"""
Unified BP-KMeans: greedy label-constrained clustering.

The module exposes configurable label-selection metrics and initialization strategies.
"""

import heapq
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from bp_k_means.algos.base_algo import BaseAlgo
from bp_k_means.algos.k_means import (
    kmeans,
    kmeans_plus_plus_init,
    random_init,
    subsampled_kmeans_plus_plus_init,
)
from bp_k_means.utils.logging import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RankingMetric(Enum):
    """
    Label-selection metrics.

    M_L:   Total WCSS of the label's clusters.
    M_C:   Maximum single-cluster WCSS within the label.
    M_ERL: Estimated WCSS reduction (label-level), scaled by k_y / (k_y + 1).
    M_RL:  Exact WCSS reduction via precomputed trial split.
    """

    M_L = 1
    M_C = 2
    M_ERL = 3
    M_RL = 4


class InitStrategy(Enum):
    """
    Initialization strategies for centroid expansion.

    I_LRI: Re-initialize all centroids for the label.
    I_ACL: Keep existing centroids, add one new.
    I_CRI: Replace highest-WCSS cluster centroid with two new centroids.
    I_ACC: Keep highest-WCSS cluster centroid, add one new within it.
    """

    I_LRI = 1
    I_ACL = 2
    I_CRI = 3
    I_ACC = 4


class InitAlgorithm(Enum):
    """
    Initialization algorithms used to initialize centroids.

    KMEANS_PLUS_PLUS: Standard k-means++ initialization.
    SUBSAMPLING_KMEANS_PLUS_PLUS: k-means++ initialization on a random subsample of the data.
    RANDOM_SAMPLING: Randomly sample k points from the data as centroids.
    """

    KMEANS_PLUS_PLUS = 1
    SUBSAMPLING_KMEANS_PLUS_PLUS = 2
    RANDOM_SAMPLING = 3


def _wcss_per_cluster(
    local_labels: "NDArray", X2: "NDArray", centroids: "NDArray", k: int
) -> "NDArray":
    """WCSS per cluster: sum(||x||^2 for x in c) - n_c * ||mu_c||^2."""
    return np.bincount(local_labels, weights=X2, minlength=k) - np.bincount(
        local_labels, minlength=k
    ) * np.einsum("ij,ij->i", centroids, centroids)


def _compute_metric(
    ranking_metric: RankingMetric,
    wcss_total: float,
    local_labels: "NDArray",
    X2: "NDArray",
    centroids: "NDArray",
    k_y: int,
    n_y: int,
) -> float:
    """Compute the label-selection metric score."""
    if ranking_metric == RankingMetric.M_L:
        score = wcss_total
    elif ranking_metric == RankingMetric.M_C:
        score = float(np.max(_wcss_per_cluster(local_labels, X2, centroids, k_y)))
    elif ranking_metric == RankingMetric.M_ERL:
        score = 0.0 if k_y >= n_y else wcss_total if k_y == n_y - 1 else wcss_total / (k_y + 1)
    else:
        msg = f"Unsupported ranking metric: {ranking_metric}"
        raise ValueError(msg)
    return score


def _build_init_centroids(
    strategy: InitStrategy,
    pts: "NDArray",
    current_centroids: "NDArray",
    curr_k: int,
    new_k: int,
    rng: np.random.Generator,
    target_pts: "NDArray | None" = None,
    max_wcss_idx: "int | None" = None,
    init_algorithm: InitAlgorithm = InitAlgorithm.KMEANS_PLUS_PLUS,
    *,
    subsample_size: int,
) -> "NDArray":
    """Build initial centroids for a k-means run with new_k clusters."""
    dim = pts.shape[1]

    if strategy in (InitStrategy.I_LRI, InitStrategy.I_ACL):
        if pts.shape[0] < new_k:
            msg = f"Cannot initialize {new_k} centroids with only {pts.shape[0]} points"
            raise ValueError(msg)

        if pts.shape[0] == new_k:
            return pts.copy()

        existing_centroids = current_centroids if strategy == InitStrategy.I_ACL else None
        return _call_init_algorithm(
            init_algorithm, pts, new_k, rng, subsample_size, existing_centroids
        )

    if strategy in (InitStrategy.I_CRI, InitStrategy.I_ACC):
        assert target_pts is not None, (
            "target_pts must be provided for cluster-level init strategies"
        )
        assert max_wcss_idx is not None, (
            "max_wcss_idx must be provided for cluster-level init strategies"
        )

        new_k_cluster = new_k - curr_k + 1

        if len(target_pts) < new_k_cluster:
            msg = f"Not enough points in target cluster to initialize {new_k_cluster} centroids"
            raise ValueError(msg)

        init_c = np.empty((new_k, dim), dtype=pts.dtype)
        init_c[:max_wcss_idx] = current_centroids[:max_wcss_idx]
        init_c[max_wcss_idx:-new_k_cluster] = current_centroids[max_wcss_idx + 1 :]

        if len(target_pts) == new_k_cluster:
            init_c[-new_k_cluster:] = target_pts
            return init_c

        existing_centroids = (
            current_centroids[max_wcss_idx : max_wcss_idx + 1]
            if strategy == InitStrategy.I_ACC
            else None
        )

        init_c[-new_k_cluster:] = _call_init_algorithm(
            init_algorithm, target_pts, new_k_cluster, rng, subsample_size, existing_centroids
        )

        return init_c

    msg = f"Unsupported init strategy: {strategy}"
    raise ValueError(msg)


def _call_init_algorithm(
    init_algorithm: InitAlgorithm,
    pts: "NDArray",
    k: int,
    rng: np.random.Generator,
    subsample_size: int,
    existing_centroids: "NDArray | None",
) -> "NDArray":
    """Dispatch to the appropriate centroid initialisation function."""
    if init_algorithm == InitAlgorithm.KMEANS_PLUS_PLUS:
        return kmeans_plus_plus_init(pts, k, rng, existing_centroids=existing_centroids)
    if init_algorithm == InitAlgorithm.SUBSAMPLING_KMEANS_PLUS_PLUS:
        return subsampled_kmeans_plus_plus_init(
            pts, k, subsample_size, seed=rng, existing_centroids=existing_centroids
        )
    if init_algorithm == InitAlgorithm.RANDOM_SAMPLING:
        return random_init(pts, k, rng, existing_centroids=existing_centroids)
    msg = f"Unsupported init algorithm: {init_algorithm}"
    raise ValueError(msg)


def _run_split(
    pts: "NDArray",
    X2: "NDArray",
    sum_X2: float,
    local_labels: "NDArray",
    current_centroids: "NDArray",
    curr_k: int,
    new_k: int,
    n_init: int,
    rng: np.random.Generator,
    init_strategy: InitStrategy,
    init_algorithm: InitAlgorithm = InitAlgorithm.KMEANS_PLUS_PLUS,
    *,
    subsample_size: int,
) -> "tuple[float, NDArray, NDArray]":
    """
    Run n_init k-means attempts with new_k clusters.

    Returns (best_wcss, best_labels, best_centroids).
    """
    target_pts = None
    max_wcss_idx = None
    if init_strategy in (InitStrategy.I_CRI, InitStrategy.I_ACC):
        wcss_per = _wcss_per_cluster(local_labels, X2, current_centroids, curr_k)
        max_wcss_idx = int(np.argmax(wcss_per))
        target_pts = pts[local_labels == max_wcss_idx]

    best_wcss = float("inf")
    best_labels = local_labels
    best_centroids = current_centroids

    for _ in range(n_init):
        init_centroids = _build_init_centroids(
            init_strategy,
            pts,
            current_centroids,
            curr_k,
            new_k,
            rng,
            target_pts,
            max_wcss_idx,
            init_algorithm,
            subsample_size=subsample_size,
        )
        lbls, ctrs = kmeans(pts, new_k, seed=rng, init_centroids=init_centroids, X2=X2)
        counts = np.bincount(lbls, minlength=new_k)
        wcss = sum_X2 - np.sum(counts * np.einsum("ij,ij->i", ctrs, ctrs))

        if wcss < best_wcss:
            best_wcss = wcss
            best_labels = lbls
            best_centroids = ctrs

    return best_wcss, best_labels, best_centroids


def bp_kmeans(
    X: "NDArray",
    y: "NDArray",
    target_k: int,
    *,
    seed: int | np.random.Generator,
    n_init: int,
    subsample_size: int,
    ranking_metric: RankingMetric = RankingMetric.M_ERL,
    init_strategy: InitStrategy = InitStrategy.I_CRI,
    init_algorithm: InitAlgorithm = InitAlgorithm.KMEANS_PLUS_PLUS,
) -> "NDArray":
    """
    BP-KMeans: greedy label-constrained clustering.

    Iteratively selects the label with the highest ranking score and increases
    its number of clusters by one, re-running k-means on that label's points.

    Parameters
    ----------
    X : NDArray
        Dataset.
    y : NDArray
        Pre-existing categorical labels.
    target_k : int
        Desired total number of clusters.
    seed : int | np.random.Generator
        Random seed.
    n_init : int
        Number of k-means restarts per split.
    ranking_metric : RankingMetric
        Ranking metric used for label selection.
    init_strategy : InitStrategy
        Centroid expansion strategy.
    init_algorithm : InitAlgorithm
        Centroid initialisation algorithm (k-means++, subsampled k-means++, or random).
    subsample_size : int
        Number of points used when ``init_algorithm`` is
        ``InitAlgorithm.SUBSAMPLING_KMEANS_PLUS_PLUS``.

    Returns
    -------
    labels : NDArray
        Cluster assignments in [0, target_k).

    Raises
    ------
    ValueError
        If the requested cluster count or initialization is infeasible.
    """
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    X = np.asarray(X)
    y = np.asarray(y)
    _, y = np.unique(y, return_inverse=True)

    n_samples = X.shape[0]
    labels: list[int] = [int(label) for label in np.unique(y)]
    n_labels = len(labels)

    if target_k > n_samples:
        msg = "target_k cannot be larger than number of data points"
        raise ValueError(msg)
    if target_k < n_labels:
        msg = "target_k cannot be smaller than number of unique labels in y"
        raise ValueError(msg)
    if target_k == n_samples:
        return np.arange(n_samples)

    # Global cluster ID tracking: each label starts with one cluster
    global_clusters_per_label: dict[int, list[int]] = {}
    current_cluster_id = 0
    for label in labels:
        global_clusters_per_label[label] = [current_cluster_id]
        current_cluster_id += 1

    # Pre-organize data by label (sorted for cache-friendly access)
    order = np.argsort(y)
    X_sorted = X[order]
    y_sorted = y[order]

    counts = np.bincount(y_sorted)
    split_indices = np.cumsum(counts)[:-1]

    groups = np.split(X_sorted, split_indices)
    idx_groups = np.split(order, split_indices)
    unique_y_vals = [int(c) for c in np.nonzero(counts)[0]]

    points_per_label = dict(zip(unique_y_vals, groups, strict=True))
    indices_per_label = dict(zip(unique_y_vals, idx_groups, strict=True))

    # Per-label state: centroids, local cluster labels, WCSS, precomputed squared norms
    centroids_per_label: dict[int, NDArray] = {}
    cluster_labels_per_label: dict[int, NDArray] = {}
    wcss_per_label: dict[int, float] = {}
    X2_per_label: dict[int, NDArray] = {}
    sum_X2_per_label: dict[int, float] = {}

    for label in labels:
        pts = points_per_label[label]
        X2 = np.einsum("ij,ij->i", pts, pts)
        X2_per_label[label] = X2
        sum_X2_per_label[label] = float(np.sum(X2))

        centroid = pts.mean(axis=0)
        centroids_per_label[label] = centroid[None, :]

        wcss = float(X2.sum() - pts.shape[0] * (centroid @ centroid))
        wcss_per_label[label] = wcss
        cluster_labels_per_label[label] = np.zeros(pts.shape[0], dtype=int)

    # Dispatch to precomputed variant for exact-reduction ranking
    if ranking_metric == RankingMetric.M_RL:
        return _bp_kmeans_precomputed(
            labels,
            target_k,
            n_init,
            rng,
            init_strategy,
            points_per_label,
            indices_per_label,
            centroids_per_label,
            cluster_labels_per_label,
            wcss_per_label,
            X2_per_label,
            sum_X2_per_label,
            global_clusters_per_label,
            current_cluster_id,
            n_samples,
            init_algorithm,
            subsample_size=subsample_size,
        )

    # Build initial heap (max-heap via negation)
    heap: list[tuple[float, int]] = []
    for label in labels:
        score = _compute_metric(
            ranking_metric,
            wcss_per_label[label],
            cluster_labels_per_label[label],
            X2_per_label[label],
            centroids_per_label[label],
            1,
            points_per_label[label].shape[0],
        )
        heap.append((-score, label))
    heapq.heapify(heap)

    # Main loop: split the highest-ranked label until target_k clusters
    while current_cluster_id < target_k:
        logger.debug("BP-KMeans: %d / %d clusters", current_cluster_id, target_k)

        _, selected_label = heapq.heappop(heap)

        pts = points_per_label[selected_label]
        X2 = X2_per_label[selected_label]
        sum_X2 = sum_X2_per_label[selected_label]
        current_centroids = centroids_per_label[selected_label]
        curr_k = current_centroids.shape[0]
        new_k = curr_k + 1

        best_wcss, best_labels, best_centroids = _run_split(
            pts,
            X2,
            sum_X2,
            cluster_labels_per_label[selected_label],
            current_centroids,
            curr_k,
            new_k,
            n_init,
            rng,
            init_strategy,
            init_algorithm,
            subsample_size=subsample_size,
        )

        cluster_labels_per_label[selected_label] = best_labels
        centroids_per_label[selected_label] = best_centroids
        wcss_per_label[selected_label] = best_wcss

        score = _compute_metric(
            ranking_metric,
            best_wcss,
            best_labels,
            X2,
            best_centroids,
            new_k,
            pts.shape[0],
        )
        heapq.heappush(heap, (-score, selected_label))

        global_clusters_per_label[selected_label].append(current_cluster_id)
        current_cluster_id += 1

    # Reconstruct global labels
    labels_global = np.empty(n_samples, dtype=int)
    for label in labels:
        global_ids = np.array(global_clusters_per_label[label], dtype=int)
        labels_global[indices_per_label[label]] = global_ids[cluster_labels_per_label[label]]

    return labels_global


def _bp_kmeans_precomputed(
    labels: list[int],
    target_k: int,
    n_init: int,
    rng: np.random.Generator,
    init_strategy: InitStrategy,
    points_per_label: dict[int, "NDArray"],
    indices_per_label: dict[int, "NDArray"],
    centroids_per_label: dict[int, "NDArray"],
    cluster_labels_per_label: dict[int, "NDArray"],
    wcss_per_label: dict[int, float],
    X2_per_label: dict[int, "NDArray"],
    sum_X2_per_label: dict[int, float],
    global_clusters_per_label: dict[int, list[int]],
    current_cluster_id: int,
    n_samples: int,
    init_algorithm: InitAlgorithm = InitAlgorithm.KMEANS_PLUS_PLUS,
    *,
    subsample_size: int,
) -> "NDArray":
    """M_RL variant: precompute trial splits to rank by exact WCSS reduction."""
    pending_splits: dict[int, tuple[float, NDArray, NDArray]] = {}
    heap: list[tuple[float, int]] = []

    def precompute_next_split(label: int) -> None:
        pts = points_per_label[label]
        current_centroids = centroids_per_label[label]
        curr_k = current_centroids.shape[0]
        new_k = curr_k + 1

        if pts.shape[0] <= curr_k:
            return

        best_wcss, best_labels, best_centroids = _run_split(
            pts,
            X2_per_label[label],
            sum_X2_per_label[label],
            cluster_labels_per_label[label],
            current_centroids,
            curr_k,
            new_k,
            n_init,
            rng,
            init_strategy,
            init_algorithm,
            subsample_size=subsample_size,
        )

        reduction = wcss_per_label[label] - best_wcss
        if reduction <= 0:
            return
        heapq.heappush(heap, (-reduction, label))
        pending_splits[label] = (best_wcss, best_labels, best_centroids)

    # Initial precomputation for all labels
    for label in labels:
        precompute_next_split(label)

    # Iterative splitting by highest actual WCSS reduction
    while current_cluster_id < target_k:
        if not heap:
            break

        _, selected_label = heapq.heappop(heap)

        if selected_label not in pending_splits:
            continue

        next_wcss, next_labels, next_centroids = pending_splits.pop(selected_label)

        centroids_per_label[selected_label] = next_centroids
        cluster_labels_per_label[selected_label] = next_labels
        wcss_per_label[selected_label] = next_wcss

        global_clusters_per_label[selected_label].append(current_cluster_id)
        current_cluster_id += 1

        logger.debug("BP-KMeans (precomputed): %d / %d clusters", current_cluster_id, target_k)

        precompute_next_split(selected_label)

    # Reconstruct global labels
    labels_global = np.empty(n_samples, dtype=int)
    for label in labels:
        global_ids = np.array(global_clusters_per_label[label], dtype=int)
        labels_global[indices_per_label[label]] = global_ids[cluster_labels_per_label[label]]

    return labels_global


class BPKMeans(BaseAlgo):
    """Common-interface wrapper around the boundary-preserving algorithm."""

    def predict(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
    ) -> "NDArray":
        """Assign instances to BP-KMeans centroids selected for their source label."""
        X_array, y_array = self._validate_prediction_input(X, y)
        distances = self._squared_centroid_distances(X_array)
        return self._select_lowest_cost_clusters(distances, y_array)

    def __init__(
        self,
        ranking_metric: RankingMetric = RankingMetric.M_ERL,
        init_strategy: InitStrategy = InitStrategy.I_CRI,
        init_algorithm: InitAlgorithm = InitAlgorithm.KMEANS_PLUS_PLUS,
        *,
        subsample_size: int,
        seed: int | np.random.Generator,
        n_init: int,
    ) -> None:
        """Initialize a BP-KMeans algorithm.

        Parameters
        ----------
        ranking_metric : RankingMetric
            Ranking metric used to select the next label to split.
        init_strategy : InitStrategy
            Strategy used to initialize each split.
        init_algorithm : InitAlgorithm
            Centroid initialization algorithm.
        subsample_size : int
            Subsample size for subsampled k-means++ initialization.
        seed : int | np.random.Generator
            Seed or random generator used by the algorithm.
        n_init : int
            Number of k-means restarts per split.
        """
        super().__init__(seed=seed, n_init=n_init)
        self.ranking_metric = ranking_metric
        self.init_strategy = init_strategy
        self.init_algorithm = init_algorithm
        self.subsample_size = subsample_size

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None,
        target_k: int,
    ) -> "BPKMeans":
        """Fit BP-KMeans and store the resulting cluster labels.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix.
        y : ArrayLike | None
            Original labels that constrain cluster membership.
        target_k : int
            Requested number of clusters.

        Returns
        -------
        BPKMeans
            The fitted algorithm instance.

        Raises
        ------
        ValueError
            If original labels are not provided.
        """
        if y is None:
            msg = "BPKMeans requires original labels"
            raise ValueError(msg)
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        labels = bp_kmeans(
            X_array,
            y_array,
            target_k,
            seed=self.seed,
            n_init=self.n_init,
            ranking_metric=self.ranking_metric,
            init_strategy=self.init_strategy,
            init_algorithm=self.init_algorithm,
            subsample_size=self.subsample_size,
        )
        return self._set_cluster_result(X_array, y_array, labels)
