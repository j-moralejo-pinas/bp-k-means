import heapq
import logging
from typing import TYPE_CHECKING

import numpy as np

from bp_k_means.k_means import kmeans, kmeans_plus_plus_init

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def bp_kmeans_optimized(X, y, target_k, seed=42, n_init=10, *, use_wcss_per_cluster=True):
    """
    Optimized BP-KMeans with fixed parameters:
    - reuse_centroids=2

    X: array (n, d)
    y: initial class labels
    target_k: desired total number of clusters
    n_init: number of times to run k-means for each split to find the best result
    """
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    X = np.asarray(X)
    y = np.asarray(y)
    unique_y, y = np.unique(y, return_inverse=True)

    if target_k > X.shape[0]:
        raise ValueError("target_k cannot be larger than number of data points")
    if target_k < len(set(y)):
        raise ValueError("target_k cannot be smaller than number of unique classes in y")
    if target_k == X.shape[0]:
        return np.arange(X.shape[0])

    # Each label starts with exactly 1 cluster
    classes = np.unique(y)

    # Mapping from class label to list of global cluster ids
    # Initially, each class has exactly one cluster, assigned sequentially
    global_cluster_of_class: dict[int, list[int]] = {}
    current_cluster_id = 0
    for c in classes:
        global_cluster_of_class[c] = [current_cluster_id]
        current_cluster_id += 1

    # Initialize WCSS for each class
    metric_per_class: list[tuple[float, int]] = []
    centroids_per_class: dict[int, NDArray] = {}
    class_labels: dict[int, NDArray] = {}

    order = np.argsort(y)
    X_sorted = X[order]
    y_sorted = y[order]

    counts = np.bincount(y_sorted)
    splits = np.cumsum(counts)[:-1]

    groups = np.split(X_sorted, splits)
    idx_groups = np.split(order, splits)
    unique_y = np.nonzero(counts)[0]

    points_per_class = dict(zip(unique_y, groups, strict=True))
    idx_per_class = dict(zip(unique_y, idx_groups, strict=True))
    X2_per_class: dict[int, NDArray] = {}
    sum_X2_per_class: dict[int, float] = {}
    for c in classes:
        pts = points_per_class[c]
        X2 = np.einsum("ij,ij->i", pts, pts)
        X2_per_class[c] = X2
        sum_X2_per_class[c] = np.sum(X2)
        centroid: NDArray = pts.mean(axis=0)
        centroids_per_class[c] = centroid[None, :]
        wcss = X2.sum() - pts.shape[0] * (centroid @ centroid)

        metric_per_class.append(
            (
                -wcss / 2
                if (use_wcss_per_cluster and pts.shape[0] > centroids_per_class[c].shape[0] + 1)
                else -wcss,
                c,
            )
        )
        class_labels[c] = np.zeros(pts.shape[0], dtype=int)
    heapq.heapify(metric_per_class)

    # While total number of clusters < target_k, split the worst class
    while current_cluster_id < target_k:
        logger.debug(
            f"BP-KMeans iteration: current clusters {current_cluster_id}, target {target_k}"
        )

        _, worst_class = heapq.heappop(metric_per_class)

        pts = points_per_class[worst_class]
        X2 = X2_per_class[worst_class]
        sum_X2 = sum_X2_per_class[worst_class]
        local_labels = class_labels[worst_class]
        current_centroids = centroids_per_class[worst_class]
        curr_k = current_centroids.shape[0]
        new_k = curr_k + 1

        best_wcss = float("inf")

        for _ in range(n_init):
            wcss_per_cluster = np.bincount(
                local_labels, weights=X2, minlength=curr_k
            ) - np.bincount(local_labels, minlength=curr_k) * np.einsum(
                "ij,ij->i", current_centroids, current_centroids
            )

            # Compute WCSS per cluster
            max_wcss_idx = np.argmax(wcss_per_cluster)

            init_centroids = np.empty((new_k, X.shape[1]), dtype=X.dtype)
            init_centroids[:max_wcss_idx] = current_centroids[:max_wcss_idx]
            init_centroids[max_wcss_idx:-2] = current_centroids[max_wcss_idx + 1 :]
            target_pts = pts[local_labels == max_wcss_idx]
            if len(target_pts) == 2:
                new_centroids = target_pts
            else:
                new_centroids = kmeans_plus_plus_init(target_pts, 2, rng)

            init_centroids[-2:] = new_centroids

            lbls, ctrs = kmeans(pts, new_k, seed=rng, init_centroids=init_centroids)
            counts = np.bincount(lbls, minlength=new_k)
            wcss = sum_X2 - np.sum(counts * np.einsum("ij,ij->i", ctrs, ctrs))

            if wcss < best_wcss:
                best_wcss = wcss
                best_labels = lbls
                best_centroids: NDArray = ctrs

        class_labels[worst_class] = best_labels

        # Update WCSS and centroids for the modified class
        heapq.heappush(
            metric_per_class,
            (
                -best_wcss / (new_k + 1)
                if (use_wcss_per_cluster and pts.shape[0] > new_k + 1)
                else -best_wcss,
                worst_class,
            ),
        )
        centroids_per_class[worst_class] = best_centroids

        global_cluster_of_class[worst_class].append(current_cluster_id)
        current_cluster_id += 1

    labels_global = np.empty(X.shape[0], dtype=int)
    for c in classes:
        global_ids = np.array(global_cluster_of_class[c], dtype=int)  # shape (k_c,)
        labels_global[idx_per_class[c]] = global_ids[class_labels[c]]

    return labels_global


def precomputed_bp_kmeans_optimized(X, y, target_k, seed=42, n_init=10):
    """
    Optimized BP-KMeans with precomputed next splits.

    This version precalculates the clusters with n+1 clusters for all existing classes.
    It chooses the class with the actual highest wcss reduction.
    """
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    X = np.asarray(X)
    y = np.asarray(y)
    unique_y, y = np.unique(y, return_inverse=True)

    if target_k > X.shape[0]:
        raise ValueError("target_k cannot be larger than number of data points")
    if target_k < len(set(y)):
        raise ValueError("target_k cannot be smaller than number of unique classes in y")

    num_classes = len(np.unique(y))
    if target_k == num_classes:
        pass
    if target_k == X.shape[0]:
        return np.arange(X.shape[0])

    classes = np.unique(y)

    # Mapping from class label to list of global cluster ids
    global_cluster_of_class: dict[int, list[int]] = {}
    current_cluster_id = 0
    for c in classes:
        global_cluster_of_class[c] = [current_cluster_id]
        current_cluster_id += 1

    # State for each class
    centroids_per_class: dict[int, NDArray] = {}
    class_labels: dict[int, NDArray] = {}
    wcss_per_class: dict[int, float] = {}
    X2_per_class: dict[int, NDArray] = {}
    sum_X2_per_class: dict[int, float] = {}

    # Pre-organize data
    order = np.argsort(y)
    X_sorted = X[order]
    y_sorted = y[order]

    counts = np.bincount(y_sorted)
    splits = np.cumsum(counts)[:-1]

    groups = np.split(X_sorted, splits)
    idx_groups = np.split(order, splits)
    unique_y_vals = np.nonzero(counts)[0]

    points_per_class = dict(zip(unique_y_vals, groups, strict=True))
    idx_per_class = dict(zip(unique_y_vals, idx_groups, strict=True))

    # Initialize k=1 state for all classes
    for c in classes:
        pts = points_per_class[c]
        X2 = np.einsum("ij,ij->i", pts, pts)
        X2_per_class[c] = X2
        sum_X2_per_class[c] = np.sum(X2)

        centroid: NDArray = pts.mean(axis=0)
        centroids_per_class[c] = centroid[None, :]

        wcss = X2.sum() - pts.shape[0] * (centroid @ centroid)
        wcss_per_class[c] = wcss
        class_labels[c] = np.zeros(pts.shape[0], dtype=int)

    # Store pending splits: class_id -> (reduction, new_wcss, new_labels, new_centroids)
    pending_splits: dict[int, tuple[float, NDArray, NDArray]] = {}

    # Heap stores (-reduction, class_id)
    heap: list[tuple[float, int]] = []

    def precompute_next_split(c: int):
        pts = points_per_class[c]
        current_centroids = centroids_per_class[c]
        curr_k = current_centroids.shape[0]
        new_k = curr_k + 1

        # Optimization: only split if enough points
        if not (pts.shape[0] > new_k + 1):
            return

        X2 = X2_per_class[c]
        sum_X2 = sum_X2_per_class[c]
        local_labels = class_labels[c]

        best_wcss = float("inf")
        best_labels = None
        best_centroids = None

        # Precompute loop
        for _ in range(n_init):
            # Same initialization strategy as optimized script
            wcss_per_cluster = np.bincount(
                local_labels, weights=X2, minlength=curr_k
            ) - np.bincount(local_labels, minlength=curr_k) * np.einsum(
                "ij,ij->i", current_centroids, current_centroids
            )

            max_wcss_idx = np.argmax(wcss_per_cluster)

            init_centroids = np.empty((new_k, X.shape[1]), dtype=X.dtype)
            init_centroids[:max_wcss_idx] = current_centroids[:max_wcss_idx]
            init_centroids[max_wcss_idx:-2] = current_centroids[max_wcss_idx + 1 :]

            target_pts = pts[local_labels == max_wcss_idx]

            if len(target_pts) == 2:
                new_centroids = target_pts
            elif len(target_pts) < 2:
                continue
            else:
                new_centroids = kmeans_plus_plus_init(target_pts, 2, rng)

            init_centroids[-2:] = new_centroids

            lbls, ctrs = kmeans(pts, new_k, seed=rng, init_centroids=init_centroids)
            counts_k = np.bincount(lbls, minlength=new_k)
            wcss = sum_X2 - np.sum(counts_k * np.einsum("ij,ij->i", ctrs, ctrs))

            if wcss < best_wcss:
                best_wcss = wcss
                best_labels = lbls
                best_centroids = ctrs

        if best_labels is not None:
            current_wcss_val = wcss_per_class[c]
            reduction = current_wcss_val - best_wcss

            # Use negative reduction for min-heap (so we pop max reduction)
            heapq.heappush(heap, (-reduction, c))
            pending_splits[c] = (best_wcss, best_labels, best_centroids)

    # Initial precomputation for all classes
    for c in classes:
        precompute_next_split(c)

    # Iterative splitting
    while current_cluster_id < target_k:
        if not heap:
            break

        _, best_c = heapq.heappop(heap)

        if best_c not in pending_splits:
            continue

        next_wcss, next_labels, next_centroids = pending_splits.pop(best_c)

        # Apply split
        centroids_per_class[best_c] = next_centroids
        class_labels[best_c] = next_labels
        wcss_per_class[best_c] = next_wcss

        # Update global IDs
        # The new cluster adds one id.
        # But wait.
        # Original code: global_ids[class_labels[c]]
        # We need to ensure that class_labels maps to indices in global_cluster_of_class list.
        # When we split, we have `new_k` clusters.
        # The global_cluster_of_class list currently has `curr_k` IDs.
        # We append 1 new ID. Total: `new_k` IDs.
        # `next_labels` has values in 0..new_k-1.
        # So it matches.
        global_cluster_of_class[best_c].append(current_cluster_id)
        current_cluster_id += 1

        # Precompute next split for this modified class
        precompute_next_split(best_c)

    # Reconstruct global labels
    labels_global = np.empty(X.shape[0], dtype=int)
    for c in classes:
        global_ids = np.array(global_cluster_of_class[c], dtype=int)
        labels_global[idx_per_class[c]] = global_ids[class_labels[c]]

    return labels_global
