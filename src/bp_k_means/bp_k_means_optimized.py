import heapq
import logging
from typing import TYPE_CHECKING

import numpy as np

from bp_k_means.k_means import kmeans, kmeans_plus_plus_init

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def bp_kmeans_optimized(X, y, target_k, seed=42, n_init=10):
    """
    Optimized BP-KMeans with fixed parameters:
    - reuse_centroids=2
    - use_wcss_per_cluster=True

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
            (-wcss / 2 if pts.shape[0] > centroids_per_class[c].shape[0] + 1 else -wcss, c)
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
                -best_wcss / (new_k + 1) if pts.shape[0] > new_k + 1 else -best_wcss,
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
