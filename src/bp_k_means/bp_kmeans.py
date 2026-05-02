"""
Unified BP-KMeans: greedy label-constrained clustering with configurable ranking
and initialization strategies.
"""

import heapq
from hmac import new
import logging
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from bp_k_means.k_means import (
    kmeans,
    kmeans_plus_plus_init,
    random_init,
    subsampled_kmeans_plus_plus_init,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class RankingStrategy(Enum):
    """Ranking strategies for label selection.

    R_L:   Total WCSS of the label's clusters.
    R_C:   Maximum single-cluster WCSS within the label.
    R_ERL: Estimated WCSS reduction (label-level), scaled by k_y / (k_y + 1).
    R_ERC: Estimated WCSS reduction (cluster-level), scaled by k_y / (k_y + 1).
    R_RL:  Exact WCSS reduction via precomputed trial split.
    """

    R_L = 1
    R_C = 2
    R_ERL = 3
    R_ERC = 4
    R_RL = 5


class InitStrategy(Enum):
    """Initialization strategies for centroid expansion.

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


def _compute_rank(
    ranking: RankingStrategy,
    wcss_total: float,
    local_labels: "NDArray",
    X2: "NDArray",
    centroids: "NDArray",
    k_y: int,
    n_y: int,
) -> float:
    """Compute the ranking score for a label."""
    if ranking == RankingStrategy.R_L:
        return wcss_total

    if ranking == RankingStrategy.R_C:
        return float(np.max(_wcss_per_cluster(local_labels, X2, centroids, k_y)))

    if ranking == RankingStrategy.R_ERL:
        if k_y >= n_y:
            return 0.0
        if k_y == n_y - 1:
            return wcss_total
        return wcss_total * k_y / (k_y + 1)

    if ranking == RankingStrategy.R_ERC:
        if k_y >= n_y:
            return 0.0
        max_wcss = float(np.max(_wcss_per_cluster(local_labels, X2, centroids, k_y)))
        if k_y == n_y - 1:
            return max_wcss
        return max_wcss * k_y / (k_y + 1)

    raise ValueError(f"Unsupported ranking strategy: {ranking}")


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
    subsample_size: int = 1000,
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

    raise ValueError(f"Unsupported init strategy: {strategy}")


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
    raise ValueError(f"Unsupported init algorithm: {init_algorithm}")


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
    subsample_size: int = 1000,
) -> "tuple[float, NDArray, NDArray]":
    """Run n_init k-means attempts with new_k clusters.

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
            subsample_size,
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
    X,
    y,
    target_k,
    seed=42,
    n_init=10,
    *,
    ranking_strategy: RankingStrategy = RankingStrategy.R_ERL,
    init_strategy: InitStrategy = InitStrategy.I_CRI,
    init_algorithm: InitAlgorithm = InitAlgorithm.KMEANS_PLUS_PLUS,
    subsample_size: int = 1000,
):
    """
    BP-KMeans: greedy label-constrained clustering.

    Iteratively selects the label with the highest ranking score and increases
    its number of clusters by one, re-running k-means on that label's points.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Dataset.
    y : array-like, shape (n,)
        Pre-existing categorical labels.
    target_k : int
        Desired total number of clusters.
    seed : int or np.random.Generator
        Random seed.
    n_init : int
        Number of k-means restarts per split.
    ranking_strategy : RankingStrategy
        Label selection strategy.
    init_strategy : InitStrategy
        Centroid expansion strategy.
    init_algorithm : InitAlgorithm
        Centroid initialisation algorithm (k-means++, subsampled k-means++, or random).
    subsample_size : int
        Number of points used when ``init_algorithm`` is
        ``InitAlgorithm.SUBSAMPLING_KMEANS_PLUS_PLUS``.

    Returns
    -------
    labels : ndarray, shape (n,)
        Cluster assignments in [0, target_k).
    """
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    X = np.asarray(X)
    y = np.asarray(y)
    _, y = np.unique(y, return_inverse=True)

    n_samples = X.shape[0]
    classes = np.unique(y)
    n_classes = len(classes)

    if target_k > n_samples:
        raise ValueError("target_k cannot be larger than number of data points")
    if target_k < n_classes:
        raise ValueError("target_k cannot be smaller than number of unique classes in y")
    if target_k == n_samples:
        return np.arange(n_samples)

    # Global cluster ID tracking: each class starts with one cluster
    global_cluster_of_class: dict[int, list[int]] = {}
    current_cluster_id = 0
    for c in classes:
        global_cluster_of_class[c] = [current_cluster_id]
        current_cluster_id += 1

    # Pre-organize data by label (sorted for cache-friendly access)
    order = np.argsort(y)
    X_sorted = X[order]
    y_sorted = y[order]

    counts = np.bincount(y_sorted)
    split_indices = np.cumsum(counts)[:-1]

    groups = np.split(X_sorted, split_indices)
    idx_groups = np.split(order, split_indices)
    unique_y_vals = np.nonzero(counts)[0]

    points_per_class = dict(zip(unique_y_vals, groups, strict=True))
    idx_per_class = dict(zip(unique_y_vals, idx_groups, strict=True))

    # Per-class state: centroids, local labels, WCSS, precomputed squared norms
    centroids_per_class: dict[int, "NDArray"] = {}
    class_labels: dict[int, "NDArray"] = {}
    wcss_per_class: dict[int, float] = {}
    X2_per_class: dict[int, "NDArray"] = {}
    sum_X2_per_class: dict[int, float] = {}

    for c in classes:
        pts = points_per_class[c]
        X2 = np.einsum("ij,ij->i", pts, pts)
        X2_per_class[c] = X2
        sum_X2_per_class[c] = float(np.sum(X2))

        centroid = pts.mean(axis=0)
        centroids_per_class[c] = centroid[None, :]

        wcss = float(X2.sum() - pts.shape[0] * (centroid @ centroid))
        wcss_per_class[c] = wcss
        class_labels[c] = np.zeros(pts.shape[0], dtype=int)

    # Dispatch to precomputed variant for exact-reduction ranking
    if ranking_strategy == RankingStrategy.R_RL:
        return _bp_kmeans_precomputed(
            classes,
            target_k,
            n_init,
            rng,
            init_strategy,
            points_per_class,
            idx_per_class,
            centroids_per_class,
            class_labels,
            wcss_per_class,
            X2_per_class,
            sum_X2_per_class,
            global_cluster_of_class,
            current_cluster_id,
            n_samples,
            init_algorithm,
            subsample_size,
        )

    # Build initial heap (max-heap via negation)
    heap: list[tuple[float, int]] = []
    for c in classes:
        score = _compute_rank(
            ranking_strategy,
            wcss_per_class[c],
            class_labels[c],
            X2_per_class[c],
            centroids_per_class[c],
            1,
            points_per_class[c].shape[0],
        )
        heap.append((-score, c))
    heapq.heapify(heap)

    # Main loop: split the highest-ranked label until target_k clusters
    while current_cluster_id < target_k:
        logger.debug("BP-KMeans: %d / %d clusters", current_cluster_id, target_k)

        _, worst_class = heapq.heappop(heap)

        pts = points_per_class[worst_class]
        X2 = X2_per_class[worst_class]
        sum_X2 = sum_X2_per_class[worst_class]
        current_centroids = centroids_per_class[worst_class]
        curr_k = current_centroids.shape[0]
        new_k = curr_k + 1

        best_wcss, best_labels, best_centroids = _run_split(
            pts,
            X2,
            sum_X2,
            class_labels[worst_class],
            current_centroids,
            curr_k,
            new_k,
            n_init,
            rng,
            init_strategy,
            init_algorithm,
            subsample_size,
        )

        class_labels[worst_class] = best_labels
        centroids_per_class[worst_class] = best_centroids
        wcss_per_class[worst_class] = best_wcss

        score = _compute_rank(
            ranking_strategy,
            best_wcss,
            best_labels,
            X2,
            best_centroids,
            new_k,
            pts.shape[0],
        )
        heapq.heappush(heap, (-score, worst_class))

        global_cluster_of_class[worst_class].append(current_cluster_id)
        current_cluster_id += 1

    # Reconstruct global labels
    labels_global = np.empty(n_samples, dtype=int)
    for c in classes:
        global_ids = np.array(global_cluster_of_class[c], dtype=int)
        labels_global[idx_per_class[c]] = global_ids[class_labels[c]]

    return labels_global


def _bp_kmeans_precomputed(
    classes,
    target_k,
    n_init,
    rng,
    init_strategy,
    points_per_class,
    idx_per_class,
    centroids_per_class,
    class_labels,
    wcss_per_class,
    X2_per_class,
    sum_X2_per_class,
    global_cluster_of_class,
    current_cluster_id,
    n_samples,
    init_algorithm: InitAlgorithm = InitAlgorithm.KMEANS_PLUS_PLUS,
    subsample_size: int = 1000,
):
    """R_RL variant: precompute trial splits to rank by exact WCSS reduction."""
    pending_splits: dict[int, tuple[float, "NDArray", "NDArray"]] = {}
    heap: list[tuple[float, int]] = []

    def precompute_next_split(c: int) -> None:
        pts = points_per_class[c]
        current_centroids = centroids_per_class[c]
        curr_k = current_centroids.shape[0]
        new_k = curr_k + 1

        if pts.shape[0] <= curr_k:
            return

        best_wcss, best_labels, best_centroids = _run_split(
            pts,
            X2_per_class[c],
            sum_X2_per_class[c],
            class_labels[c],
            current_centroids,
            curr_k,
            new_k,
            n_init,
            rng,
            init_strategy,
            init_algorithm,
            subsample_size,
        )

        reduction = wcss_per_class[c] - best_wcss
        if reduction <= 0:
            return
        heapq.heappush(heap, (-reduction, c))
        pending_splits[c] = (best_wcss, best_labels, best_centroids)

    # Initial precomputation for all classes
    for c in classes:
        precompute_next_split(c)

    # Iterative splitting by highest actual WCSS reduction
    while current_cluster_id < target_k:
        if not heap:
            break

        _, best_c = heapq.heappop(heap)

        if best_c not in pending_splits:
            continue

        next_wcss, next_labels, next_centroids = pending_splits.pop(best_c)

        centroids_per_class[best_c] = next_centroids
        class_labels[best_c] = next_labels
        wcss_per_class[best_c] = next_wcss

        global_cluster_of_class[best_c].append(current_cluster_id)
        current_cluster_id += 1

        logger.debug("BP-KMeans (precomputed): %d / %d clusters", current_cluster_id, target_k)

        precompute_next_split(best_c)

    # Reconstruct global labels
    labels_global = np.empty(n_samples, dtype=int)
    for c in classes:
        global_ids = np.array(global_cluster_of_class[c], dtype=int)
        labels_global[idx_per_class[c]] = global_ids[class_labels[c]]

    return labels_global
