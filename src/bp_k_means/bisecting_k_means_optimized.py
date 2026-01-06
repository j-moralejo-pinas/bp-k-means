import heapq
import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy import cluster

from bp_k_means.k_means import kmeans, kmeans_plus_plus_init

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def bisecting_kmeans_by_label_optimized(X, y, target_k, seed=42, n_init=10):
    """
    Divisive hierarchical clustering (bisecting k-means) with label constraint.
    Optimized version using heaps and pre-calculated squared norms.
    Always uses refine_clusters=True strategy (re-running k-means on the label subset).

    X: (n, d) data
    y: (n,) labels (e.g. CUSEC)
    target_k: desired final number of clusters
    seed: RNG seed
    n_init: number of re-initializations for the split

    Returns:
        labels_final: (n,) cluster ids in [0, target_k - 1]
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, dim = X.shape

    unique_labels, y_inverse = np.unique(y, return_inverse=True)
    n_labels = len(unique_labels)

    if n_init < 1:
        raise ValueError("n_init must be >= 1")

    if target_k < n_labels:
        raise ValueError(
            f"target_k={target_k} < number of labels={n_labels}. "
            f"With label-pure clusters you cannot go below that."
        )

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
    points_per_label = groups
    indices_per_label = idx_groups
    X2_per_label = []

    centroids_per_label: list[NDArray] = []
    local_labels_per_label: list[NDArray] = []
    sum_X2_per_label: list[float] = []
    generation_per_label = [0] * n_labels

    cluster_heap = []

    current_total_clusters = 0

    for lbl_idx in range(n_labels):
        current_total_clusters += 1

        pts = points_per_label[lbl_idx]

        local_labels = np.zeros(pts.shape[0], dtype=int)
        local_labels_per_label.append(local_labels)

        # TODO(Javier): We should check that all points are not identical
        if pts.shape[0] < 2:
            X2_per_label.append(None)
            centroids_per_label.append(pts[0][None, :])
            continue

        X2 = np.einsum("ij,ij->i", pts, pts)
        sum_X2 = np.sum(X2)
        X2_per_label.append(X2)
        sum_X2_per_label.append(sum_X2)

        centroid = pts.mean(axis=0)
        centroids_per_label.append(centroid[None, :])

        wcss = sum_X2 - pts.shape[0] * (centroid @ centroid)

        heapq.heappush(cluster_heap, (-wcss, lbl_idx, 0, 0))

    while current_total_clusters < target_k:
        if not cluster_heap:
            raise ValueError("Heap empty but target_k not reached. This shouldn't happen.")

        _, lbl_idx, local_idx, gen = heapq.heappop(cluster_heap)

        if gen != generation_per_label[lbl_idx]:
            # Stale entry
            continue

        pts = points_per_label[lbl_idx]
        X2 = X2_per_label[lbl_idx]
        sum_X2 = sum_X2_per_label[lbl_idx]
        current_centroids = centroids_per_label[lbl_idx]

        best_wcss_total = float("inf")

        new_centroids = np.empty(
            (current_centroids.shape[0] + 1, dim), dtype=current_centroids.dtype
        )

        new_centroids[:local_idx] = current_centroids[:local_idx]
        new_centroids[local_idx:-2] = current_centroids[local_idx + 1 :]

        cluster_pts = pts[local_labels_per_label[lbl_idx] == local_idx]

        for _ in range(n_init):
            if cluster_pts.shape[0] == 2:
                new_centroids[-2:] = cluster_pts
            else:
                new_centroids[-2:] = kmeans_plus_plus_init(cluster_pts, 2, rng)

            k_new = new_centroids.shape[0]

            new_lbls, new_ctrs = kmeans(pts, k=k_new, seed=rng, init_centroids=new_centroids)

            counts = np.bincount(new_lbls, minlength=k_new)
            wcss_total = sum_X2 - np.sum(counts * np.einsum("ij,ij->i", new_ctrs, new_ctrs))

            if wcss_total < best_wcss_total:
                best_wcss_total = wcss_total
                best_new_labels = new_lbls
                best_new_centroids = new_ctrs

        centroids_per_label[lbl_idx] = best_new_centroids
        local_labels_per_label[lbl_idx] = best_new_labels
        generation_per_label[lbl_idx] += 1

        new_k = best_new_centroids.shape[0]
        counts = np.bincount(best_new_labels, minlength=new_k)

        X2_sums = np.bincount(best_new_labels, weights=X2, minlength=new_k)
        wcss_per_cluster = X2_sums - counts * np.einsum(
            "ij,ij->i", best_new_centroids, best_new_centroids
        )

        for k in range(new_k):
            # TODO(Javier): We should check that all points are not identical
            if counts[k] > 1:
                heapq.heappush(
                    cluster_heap,
                    (-wcss_per_cluster[k], lbl_idx, k, generation_per_label[lbl_idx]),
                )

        current_total_clusters += 1

    labels_final = np.empty(n_samples, dtype=int)

    global_cluster_counter = 0

    for lbl_idx in range(n_labels):
        l_indices = indices_per_label[lbl_idx]
        l_centroids = centroids_per_label[lbl_idx]

        labels_final[l_indices] = local_labels_per_label[lbl_idx] + global_cluster_counter

        global_cluster_counter += l_centroids.shape[0]

    return labels_final


def bisecting_kmeans_by_label_optimized_no_refine(X, y, target_k, seed=42, n_init=10):
    """
    Divisive hierarchical clustering (bisecting k-means) with label constraint.
    Optimized version using heaps and pre-calculated squared norms.
    Uses refine_clusters=False strategy (only the split cluster is re-clustered).

    X: (n, d) data
    y: (n,) labels (e.g. CUSEC)
    target_k: desired final number of clusters
    seed: RNG seed
    n_init: number of re-initializations for the split

    Returns:
        labels_final: (n,) cluster ids in [0, target_k - 1]
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, dim = X.shape

    unique_labels, y_inverse = np.unique(y, return_inverse=True)
    n_labels = len(unique_labels)

    if n_init < 1:
        raise ValueError("n_init must be >= 1")

    if target_k < n_labels:
        raise ValueError(
            f"target_k={target_k} < number of labels={n_labels}. "
            f"With label-pure clusters you cannot go below that."
        )

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

    points_per_label = groups
    indices_per_label = idx_groups
    X2_per_label = []

    centroids_per_label: list[NDArray] = []
    local_labels_per_label: list[NDArray] = []

    cluster_heap = []
    current_total_clusters = 0

    # Initialization
    for lbl_idx in range(n_labels):
        current_total_clusters += 1
        pts = points_per_label[lbl_idx]

        local_labels = np.zeros(pts.shape[0], dtype=int)
        local_labels_per_label.append(local_labels)

        if pts.shape[0] < 2:
            X2_per_label.append(None)
            centroids_per_label.append(pts[0][None, :])
            continue

        X2 = np.einsum("ij,ij->i", pts, pts)
        sum_X2 = np.sum(X2)
        X2_per_label.append(X2)

        centroid = pts.mean(axis=0)
        centroids_per_label.append(centroid[None, :])

        wcss = sum_X2 - pts.shape[0] * (centroid @ centroid)

        heapq.heappush(cluster_heap, (-wcss, lbl_idx, 0))

    while current_total_clusters < target_k:
        if not cluster_heap:
            # This can happen if all clusters have < 2 points or wcss=0
            break

        _, lbl_idx, local_idx = heapq.heappop(cluster_heap)

        pts = points_per_label[lbl_idx]
        X2 = X2_per_label[lbl_idx]

        # Identify points in this cluster
        mask = local_labels_per_label[lbl_idx] == local_idx
        cluster_pts = pts[mask]
        cluster_X2 = X2[mask]

        if cluster_pts.shape[0] < 2:
            continue

        # Split cluster_pts into 2
        best_wcss_total = float("inf")

        # Try n_init times
        for _ in range(n_init):
            # Init 2 centroids
            if cluster_pts.shape[0] == 2:
                init_c = cluster_pts
            else:
                init_c = kmeans_plus_plus_init(cluster_pts, 2, rng)

            sub_labels, sub_centroids = kmeans(
                cluster_pts, k=2, seed=rng, init_centroids=init_c
            )

            # Calculate WCSS for this split
            counts = np.bincount(sub_labels, minlength=2)
            sub_X2_sums = np.bincount(sub_labels, weights=cluster_X2, minlength=2)

            wcss_split = sub_X2_sums - counts * np.einsum(
                "ij,ij->i", sub_centroids, sub_centroids
            )
            total_wcss = np.sum(wcss_split)

            if total_wcss < best_wcss_total:
                best_wcss_total = total_wcss
                best_sub_labels = sub_labels
                best_sub_centroids = sub_centroids
                best_wcss_per_cluster = wcss_split
                best_counts = counts

        # Apply the best split
        # 1. Update centroids
        current_centroids = centroids_per_label[lbl_idx]
        new_centroids_arr = np.empty(
            (current_centroids.shape[0] + 1, dim), dtype=current_centroids.dtype
        )
        new_centroids_arr[: current_centroids.shape[0]] = current_centroids

        new_centroids_arr[local_idx] = best_sub_centroids[0]
        new_centroids_arr[-1] = best_sub_centroids[1]

        centroids_per_label[lbl_idx] = new_centroids_arr

        # 2. Update labels
        new_idx = current_centroids.shape[0]
        indices_to_update = np.flatnonzero(mask)
        indices_to_change = indices_to_update[best_sub_labels == 1]
        local_labels_per_label[lbl_idx][indices_to_change] = new_idx

        # 3. Push new clusters to heap
        if best_counts[0] > 1:
            heapq.heappush(
                cluster_heap, (-best_wcss_per_cluster[0], lbl_idx, local_idx)
            )

        if best_counts[1] > 1:
            heapq.heappush(cluster_heap, (-best_wcss_per_cluster[1], lbl_idx, new_idx))

        current_total_clusters += 1

    # Finalize labels
    labels_final = np.empty(n_samples, dtype=int)
    global_cluster_counter = 0

    for lbl_idx in range(n_labels):
        l_indices = indices_per_label[lbl_idx]
        l_centroids = centroids_per_label[lbl_idx]

        labels_final[l_indices] = local_labels_per_label[lbl_idx] + global_cluster_counter
        global_cluster_counter += l_centroids.shape[0]

    return labels_final
