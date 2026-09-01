"""Optimized bisecting k-means implementations with label constraints."""

import heapq
import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from bp_k_means.algos.base_algo import BaseAlgo
from bp_k_means.algos.k_means import kmeans, kmeans_plus_plus_init

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)
MIN_SPLIT_POINTS = 2

def _refine_cluster_split(
    pts: "NDArray",
    cluster_pts: "NDArray",
    X2: "NDArray",
    sum_X2: float,
    current_centroids: "NDArray",
    local_idx: int,
    n_init: int,
    rng: np.random.Generator,
) -> tuple[float, "NDArray", "NDArray", "NDArray"]:
    """Find the best refined split for one current cluster."""
    new_k = current_centroids.shape[0] + 1
    dim = pts.shape[1]
    new_centroids = np.empty((new_k, dim), dtype=current_centroids.dtype)
    new_centroids[:local_idx] = current_centroids[:local_idx]
    new_centroids[local_idx:-2] = current_centroids[local_idx + 1 :]
    best_wcss_total = float("inf")
    best_new_labels = None
    best_new_centroids = None
    for _ in range(n_init):
        if cluster_pts.shape[0] == MIN_SPLIT_POINTS:
            new_centroids[-2:] = cluster_pts
        else:
            new_centroids[-2:] = kmeans_plus_plus_init(cluster_pts, MIN_SPLIT_POINTS, rng)

        new_labels, candidate_centroids = kmeans(
            pts,
            k=new_k,
            seed=rng,
            init_centroids=new_centroids,
            X2=X2,
        )
        counts = np.bincount(new_labels, minlength=new_k)
        wcss_total = sum_X2 - np.sum(
            counts * np.einsum("ij,ij->i", candidate_centroids, candidate_centroids)
        )
        if wcss_total < best_wcss_total:
            best_wcss_total = wcss_total
            best_new_labels = new_labels
            best_new_centroids = candidate_centroids

    assert best_new_labels is not None
    assert best_new_centroids is not None
    return best_wcss_total, best_new_labels, best_new_centroids, counts


def _best_bisecting_split(
    cluster_pts: "NDArray",
    cluster_X2: "NDArray",
    n_init: int,
    rng: np.random.Generator,
) -> tuple["NDArray", "NDArray", "NDArray", "NDArray"]:
    """Find the best independent two-way split for one cluster."""
    best_wcss_total = float("inf")
    best_labels = None
    best_centroids = None
    best_wcss_per_cluster = None
    best_counts = None

    for _ in range(n_init):
        init_centroids = (
            cluster_pts
            if cluster_pts.shape[0] == MIN_SPLIT_POINTS
            else kmeans_plus_plus_init(cluster_pts, MIN_SPLIT_POINTS, rng)
        )
        sub_labels, sub_centroids = kmeans(
            cluster_pts,
            k=MIN_SPLIT_POINTS,
            seed=rng,
            init_centroids=init_centroids,
        )
        counts = np.bincount(sub_labels, minlength=MIN_SPLIT_POINTS)
        sub_X2_sums = np.bincount(sub_labels, weights=cluster_X2, minlength=MIN_SPLIT_POINTS)
        wcss_per_cluster = sub_X2_sums - counts * np.einsum(
            "ij,ij->i", sub_centroids, sub_centroids
        )
        total_wcss = np.sum(wcss_per_cluster)
        if total_wcss < best_wcss_total:
            best_wcss_total = total_wcss
            best_labels = sub_labels
            best_centroids = sub_centroids
            best_wcss_per_cluster = wcss_per_cluster
            best_counts = counts

    assert best_labels is not None
    assert best_centroids is not None
    assert best_wcss_per_cluster is not None
    assert best_counts is not None
    return best_labels, best_centroids, best_wcss_per_cluster, best_counts


def _push_cluster_candidates(
    cluster_heap: list[tuple[float, int, int, int]],
    lbl_idx: int,
    counts: "NDArray",
    wcss_per_cluster: "NDArray",
    new_k: int,
    point_count: int,
    generation: int,
    *,
    use_wcss_per_cluster: bool,
) -> None:
    """Add splittable refined clusters to the priority queue."""
    for cluster_idx in range(new_k):
        if counts[cluster_idx] <= 1:
            continue
        priority = (
            -wcss_per_cluster[cluster_idx] / (new_k + 1)
            if use_wcss_per_cluster and point_count > new_k + 1
            else -wcss_per_cluster[cluster_idx]
        )
        heapq.heappush(cluster_heap, (priority, lbl_idx, cluster_idx, generation))


def _validate_inputs(
    X: "NDArray",
    y: "NDArray",
    target_k: int,
    n_init: int,
) -> tuple["NDArray", "NDArray", int, int]:
    """Validate shared bisecting k-means inputs and return normalized arrays."""
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    if n_init < 1:
        msg = "n_init must be >= 1"
        raise ValueError(msg)
    if target_k < n_classes:
        msg = (
            f"target_k={target_k} < number of labels={n_classes}. "
            f"With label-pure clusters you cannot go below that."
        )
        raise ValueError(msg)
    return X, y, n_samples, n_classes


def bisecting_kmeans_by_label_optimized(
    X: "NDArray",
    y: "NDArray",
    target_k: int,
    seed: int | np.random.Generator = 42,
    n_init: int = 10,
    *,
    use_wcss_per_cluster: bool = True,
) -> "NDArray":
    """
    Divisive hierarchical clustering (bisecting k-means) with a label constraint.

    The optimized version uses heaps and pre-calculated squared norms. It uses
    the refine-clusters strategy, re-running k-means on each label subset.

    Parameters
    ----------
    X : NDArray
        Data with shape (n, d).
    y : NDArray
        Labels with shape (n, e.g. CUSEC).
    target_k : int
        Desired final number of clusters.
    seed : int | np.random.Generator
        RNG seed.
    n_init : int
        Number of re-initializations for the split.
    use_wcss_per_cluster : bool
        Whether to prioritize splits using per-cluster WCSS.

    Returns
    -------
    NDArray
        Cluster ids in [0, target_k - 1].

    Raises
    ------
    ValueError
        If the requested cluster count is infeasible.
    """
    X, y, n_samples, n_classes = _validate_inputs(X, y, target_k, n_init)
    rng = np.random.default_rng(seed)
    _, _dim = X.shape

    _, y_inverse = np.unique(y, return_inverse=True)

    if target_k == n_samples:
        return np.arange(n_samples)

    # Pre-calculate squared norms for WCSS optimization
    # We will group data by label first to avoid indexing overhead later
    order = np.argsort(y_inverse)
    X_sorted = X[order]
    y_sorted = y_inverse[order]  # 0..n_labels-1

    counts = np.bincount(y_sorted)
    splits = np.cumsum(counts)[:-1]

    groups = np.split(X_sorted, splits)
    # idx_groups maps back to original indices
    idx_groups = np.split(order, splits)

    # Data structures per label (using 0..n_labels-1 as key)
    points_per_class = groups
    indices_per_class = idx_groups
    X2_per_class = []

    centroids_per_class: list[NDArray] = []
    local_labels_per_class: list[NDArray] = []
    sum_X2_per_class: list[float] = []
    generation_per_class = [0] * n_classes

    cluster_heap = []

    current_total_clusters = 0

    for lbl_idx in range(n_classes):
        current_total_clusters += 1

        pts = points_per_class[lbl_idx]

        local_labels = np.zeros(pts.shape[0], dtype=int)
        local_labels_per_class.append(local_labels)

        # TODO(Javier): We should check that all points are not identical
        if pts.shape[0] < MIN_SPLIT_POINTS:
            X2_per_class.append(None)
            sum_X2_per_class.append(0.0)
            centroids_per_class.append(pts[0][None, :])
            continue

        X2 = np.einsum("ij,ij->i", pts, pts)
        sum_X2 = np.sum(X2)
        X2_per_class.append(X2)
        sum_X2_per_class.append(sum_X2)

        centroid = pts.mean(axis=0)
        centroids_per_class.append(centroid[None, :])

        wcss = sum_X2 - pts.shape[0] * (centroid @ centroid)

        heapq.heappush(
            cluster_heap,
            (
                -wcss / 2
                if (
                    use_wcss_per_cluster
                    and pts.shape[0] > centroids_per_class[lbl_idx].shape[0] + 1
                )
                else -wcss,
                lbl_idx,
                0,
                0,
            ),
        )

    while current_total_clusters < target_k:
        if not cluster_heap:
            msg = "Heap empty but target_k not reached. This shouldn't happen."
            raise ValueError(msg)

        _, lbl_idx, local_idx, gen = heapq.heappop(cluster_heap)

        if gen != generation_per_class[lbl_idx]:
            # Stale entry
            continue

        pts = points_per_class[lbl_idx]
        X2 = X2_per_class[lbl_idx]
        sum_X2 = sum_X2_per_class[lbl_idx]
        current_centroids = centroids_per_class[lbl_idx]

        _best_wcss_total, best_new_labels, best_new_centroids, _ = _refine_cluster_split(
            pts,
            pts[local_labels_per_class[lbl_idx] == local_idx],
            X2,
            sum_X2,
            current_centroids,
            local_idx,
            n_init,
            rng,
        )

        centroids_per_class[lbl_idx] = best_new_centroids
        local_labels_per_class[lbl_idx] = best_new_labels
        generation_per_class[lbl_idx] += 1

        new_k = best_new_centroids.shape[0]
        counts = np.bincount(best_new_labels, minlength=new_k)

        X2_sums = np.bincount(best_new_labels, weights=X2, minlength=new_k)
        wcss_per_cluster = X2_sums - counts * np.einsum(
            "ij,ij->i", best_new_centroids, best_new_centroids
        )

        _push_cluster_candidates(
            cluster_heap,
            lbl_idx,
            counts,
            wcss_per_cluster,
            new_k,
            pts.shape[0],
            generation_per_class[lbl_idx],
            use_wcss_per_cluster=use_wcss_per_cluster,
        )

        current_total_clusters += 1

    labels_final = np.empty(n_samples, dtype=int)

    global_cluster_counter = 0

    for lbl_idx in range(n_classes):
        l_indices = indices_per_class[lbl_idx]
        l_centroids = centroids_per_class[lbl_idx]

        labels_final[l_indices] = local_labels_per_class[lbl_idx] + global_cluster_counter

        global_cluster_counter += l_centroids.shape[0]

    return labels_final


def bisecting_kmeans_by_label_optimized_no_refine(
    X: "NDArray",
    y: "NDArray",
    target_k: int,
    seed: int | np.random.Generator = 42,
    n_init: int = 10,
) -> "NDArray":
    """
    Divisive hierarchical clustering (bisecting k-means) with a label constraint.

    The optimized version uses heaps and pre-calculated squared norms. It uses
    the no-refine strategy, re-clustering only the split cluster.

    Parameters
    ----------
    X : NDArray
        Data with shape (n, d).
    y : NDArray
        Labels with shape (n, e.g. CUSEC).
    target_k : int
        Desired final number of clusters.
    seed : int | np.random.Generator
        RNG seed.
    n_init : int
        Number of re-initializations for the split.

    Returns
    -------
    NDArray
        Cluster ids in [0, target_k - 1].
    """
    X, y, n_samples, n_classes = _validate_inputs(X, y, target_k, n_init)
    rng = np.random.default_rng(seed)
    _, dim = X.shape

    _, y_inverse = np.unique(y, return_inverse=True)

    if target_k == n_samples:
        return np.arange(n_samples)

    # Pre-calculate squared norms for WCSS optimization
    order = np.argsort(y_inverse)
    X_sorted = X[order]
    y_sorted = y_inverse[order]

    counts = np.bincount(y_sorted)
    splits = np.cumsum(counts)[:-1]

    groups = np.split(X_sorted, splits)
    idx_groups = np.split(order, splits)

    points_per_class = groups
    indices_per_class = idx_groups
    X2_per_class = []

    centroids_per_class: list[NDArray] = []
    local_labels_per_class: list[NDArray] = []

    cluster_heap = []
    current_total_clusters = 0

    # Initialization
    for lbl_idx in range(n_classes):
        current_total_clusters += 1
        pts = points_per_class[lbl_idx]

        local_labels = np.zeros(pts.shape[0], dtype=int)
        local_labels_per_class.append(local_labels)

        if pts.shape[0] < MIN_SPLIT_POINTS:
            X2_per_class.append(None)
            centroids_per_class.append(pts[0][None, :])
            continue

        X2 = np.einsum("ij,ij->i", pts, pts)
        sum_X2 = np.sum(X2)
        X2_per_class.append(X2)

        centroid = pts.mean(axis=0)
        centroids_per_class.append(centroid[None, :])

        wcss = sum_X2 - pts.shape[0] * (centroid @ centroid)

        heapq.heappush(cluster_heap, (-wcss, lbl_idx, 0))

    while current_total_clusters < target_k:
        if not cluster_heap:
            # This can happen if all clusters have < 2 points or wcss=0
            break

        _, lbl_idx, local_idx = heapq.heappop(cluster_heap)

        pts = points_per_class[lbl_idx]
        X2 = X2_per_class[lbl_idx]
        # Identify points in this cluster
        mask = local_labels_per_class[lbl_idx] == local_idx
        cluster_pts = pts[mask]
        cluster_X2 = X2[mask]

        if cluster_pts.shape[0] < MIN_SPLIT_POINTS:
            continue

        best_sub_labels, best_sub_centroids, best_wcss_per_cluster, best_counts = (
            _best_bisecting_split(cluster_pts, cluster_X2, n_init, rng)
        )

        # Apply the best split
        # 1. Update centroids
        current_centroids = centroids_per_class[lbl_idx]
        new_centroids_arr = np.empty(
            (current_centroids.shape[0] + 1, dim), dtype=current_centroids.dtype
        )
        new_centroids_arr[: current_centroids.shape[0]] = current_centroids

        new_centroids_arr[local_idx] = best_sub_centroids[0]
        new_centroids_arr[-1] = best_sub_centroids[1]

        centroids_per_class[lbl_idx] = new_centroids_arr

        # 2. Update labels
        new_idx = current_centroids.shape[0]
        indices_to_update = np.flatnonzero(mask)
        indices_to_change = indices_to_update[best_sub_labels == 1]
        local_labels_per_class[lbl_idx][indices_to_change] = new_idx

        # 3. Push new clusters to heap
        if best_counts[0] > 1:
            heapq.heappush(cluster_heap, (-best_wcss_per_cluster[0], lbl_idx, local_idx))

        if best_counts[1] > 1:
            heapq.heappush(cluster_heap, (-best_wcss_per_cluster[1], lbl_idx, new_idx))

        current_total_clusters += 1

    # Finalize labels
    labels_final = np.empty(n_samples, dtype=int)
    global_cluster_counter = 0

    for lbl_idx in range(n_classes):
        l_indices = indices_per_class[lbl_idx]
        l_centroids = centroids_per_class[lbl_idx]

        labels_final[l_indices] = local_labels_per_class[lbl_idx] + global_cluster_counter
        global_cluster_counter += l_centroids.shape[0]

    return labels_final


class BisectingKMeans(BaseAlgo):
    """Common-interface wrapper around refined bisecting k-means."""

    def __init__(
        self,
        seed: int | np.random.Generator = 42,
        n_init: int = 10,
        *,
        use_wcss_per_cluster: bool = True,
    ) -> None:
        """Initialize refined bisecting k-means.

        Parameters
        ----------
        seed : int | np.random.Generator
            Seed or random generator used by the algorithm.
        n_init : int
            Number of re-initializations per split.
        use_wcss_per_cluster : bool
            Whether to prioritize candidates by per-cluster WCSS.
        """
        super().__init__(seed=seed, n_init=n_init)
        self.use_wcss_per_cluster = use_wcss_per_cluster

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None,
        target_k: int,
    ) -> "BisectingKMeans":
        """Fit refined bisecting k-means.

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
        BisectingKMeans
            The fitted algorithm instance.

        Raises
        ------
        ValueError
            If original labels are not provided.
        """
        if y is None:
            msg = "BisectingKMeans requires original labels"
            raise ValueError(msg)
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        labels = bisecting_kmeans_by_label_optimized(
            X_array,
            y_array,
            target_k,
            seed=self.seed,
            n_init=self.n_init,
            use_wcss_per_cluster=self.use_wcss_per_cluster,
        )
        return self._set_result(labels)


class BisectingKMeansNoRefine(BaseAlgo):
    """Common-interface wrapper around non-refined bisecting k-means."""

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None,
        target_k: int,
    ) -> "BisectingKMeansNoRefine":
        """Fit non-refined bisecting k-means.

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
        BisectingKMeansNoRefine
            The fitted algorithm instance.

        Raises
        ------
        ValueError
            If original labels are not provided.
        """
        if y is None:
            msg = "BisectingKMeansNoRefine requires original labels"
            raise ValueError(msg)
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        labels = bisecting_kmeans_by_label_optimized_no_refine(
            X_array,
            y_array,
            target_k,
            seed=self.seed,
            n_init=self.n_init,
        )
        return self._set_result(labels)
