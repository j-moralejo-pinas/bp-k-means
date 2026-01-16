import heapq
import logging
from typing import TYPE_CHECKING, Tuple, Optional

import numpy as np

from bp_k_means.k_means import kmeans, kmeans_plus_plus_init

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def precomputed_bisecting_kmeans_by_label_optimized(X, y, target_k, seed=42, n_init=10):
    """
    Bisecting K-Means with label constraint and precomputed splits (Refine Cluster strategy).

    In this version (Refine Cluster):
    - We maintain the clustering state for each class label.
    - We precompute the effect of splitting one more cluster for each class.
    - Since we have the whole class data, we can re-run k-means on the whole class with k+1 centroids
      (seeded from previous centroids + split of the worst cluster).
    - The heap stores the ACTUAL reduction in WCSS for the entire class if we increment k by 1.
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
        msg = f"target_k={target_k} < number of labels={n_labels}."
        raise ValueError(msg)
    if target_k == n_samples:
        return np.arange(n_samples)

    # Group data by label
    order = np.argsort(y_inverse)
    X_sorted = X[order]
    y_sorted = y_inverse[order]

    counts = np.bincount(y_sorted)
    splits = np.cumsum(counts)[:-1]

    groups = np.split(X_sorted, splits)
    idx_groups = np.split(order, splits)

    points_per_label = groups
    indices_per_label = idx_groups

    # State per label
    centroids_per_label: dict[int, NDArray] = {}
    local_labels_per_label: dict[int, NDArray] = {}
    wcss_per_label: dict[int, float] = {}
    X2_per_label: dict[int, NDArray] = {}
    sum_X2_per_label: dict[int, float] = {}

    # Pending splits: label_idx -> (new_wcss, new_labels, new_centroids)
    pending_splits: dict[int, Tuple[float, NDArray, NDArray]] = {}

    # Heap: (-reduction, label_idx)
    heap: list[Tuple[float, int]] = []

    current_total_clusters = 0

    # Initialize k=1 for each label
    for lbl_idx in range(n_labels):
        pts = points_per_label[lbl_idx]
        current_total_clusters += 1

        # Calculate X2 once
        X2 = np.einsum("ij,ij->i", pts, pts)
        sum_X2 = np.sum(X2)
        X2_per_label[lbl_idx] = X2
        sum_X2_per_label[lbl_idx] = sum_X2

        # Init centroid
        if pts.shape[0] < 1:
            # Should not happen with valid data usually
            centroid = np.zeros((1, dim))
            wcss = 0.0
        else:
            centroid = pts.mean(axis=0)[None, :]
            wcss = sum_X2 - pts.shape[0] * (centroid[0] @ centroid[0])

        centroids_per_label[lbl_idx] = centroid
        local_labels_per_label[lbl_idx] = np.zeros(pts.shape[0], dtype=int)
        wcss_per_label[lbl_idx] = wcss

    def precompute_next_split_refine(lbl_idx: int):
        pts = points_per_label[lbl_idx]
        current_centroids = centroids_per_label[lbl_idx]
        local_labels = local_labels_per_label[lbl_idx]
        X2 = X2_per_label[lbl_idx]
        sum_X2 = sum_X2_per_label[lbl_idx]

        curr_k = current_centroids.shape[0]
        new_k = curr_k + 1

        if pts.shape[0] < new_k:
            return

        # Simple Bisecting K-Means logic to find candidate split
        # Identical to optimized BP-KMeans logic for finding next state

        # 1. Find cluster with highest WCSS contribution
        wcss_per_cluster = np.bincount(local_labels, weights=X2, minlength=curr_k) - np.bincount(
            local_labels, minlength=curr_k
        ) * np.einsum("ij,ij->i", current_centroids, current_centroids)
        max_wcss_idx = np.argmax(wcss_per_cluster)

        # 2. Split that cluster and refine all
        best_wcss_total = float("inf")
        best_new_labels = None
        best_new_centroids = None

        # Prepare init centroids structure
        init_centroids_base = np.empty((new_k, dim), dtype=X.dtype)
        # Copy existing except the one we split (at max_wcss_idx)
        # We put strict existing centroids at 0..max_wcss_idx, and max_wcss_idx+1..-2
        # Then the 2 new ones at the end (or one at max_wcss_idx and one at end?)
        # Let's follow standard pattern: replace split cluster with one part, add other at end.
        # However, `kmeans` function takes init_centroids.

        # To match previous logic:
        # We keep 0..max_wcss_idx unchanged in slots
        # We keep max_wcss_idx+1..end unchanged in slots
        # The new centroids go into max_wcss_idx and -1?
        # Or usually in these implementations:
        # copy [0:max_wcss_idx] -> [0:max_wcss_idx]
        # copy [max_wcss_idx+1:] -> [max_wcss_idx+1: -2]
        # new -> [-2:]

        init_centroids_base[:max_wcss_idx] = current_centroids[:max_wcss_idx]
        init_centroids_base[max_wcss_idx:-2] = current_centroids[max_wcss_idx + 1 :]

        target_pts = pts[local_labels == max_wcss_idx]
        if len(target_pts) < 2:
            return

        for _ in range(n_init):
            if len(target_pts) == 2:
                new_pair = target_pts
            else:
                new_pair = kmeans_plus_plus_init(target_pts, 2, rng)

            init_centroids = init_centroids_base.copy()
            init_centroids[-2:] = new_pair

            # Refine all clusters
            lbls, ctrs = kmeans(pts, new_k, seed=rng, init_centroids=init_centroids)

            counts = np.bincount(lbls, minlength=new_k)
            wcss = sum_X2 - np.sum(counts * np.einsum("ij,ij->i", ctrs, ctrs))

            if wcss < best_wcss_total:
                best_wcss_total = wcss
                best_new_labels = lbls
                best_new_centroids = ctrs

        if best_new_labels is not None:
            current_wcss = wcss_per_label[lbl_idx]
            reduction = current_wcss - best_wcss_total

            heapq.heappush(heap, (-reduction, lbl_idx))
            pending_splits[lbl_idx] = (best_wcss_total, best_new_labels, best_new_centroids)

    # Initial precompute
    for lbl_idx in range(n_labels):
        precompute_next_split_refine(lbl_idx)

    while current_total_clusters < target_k:
        if not heap:
            break

        neg_red, best_lbl = heapq.heappop(heap)

        if best_lbl not in pending_splits:
            continue

        new_wcss, new_labels, new_centroids = pending_splits.pop(best_lbl)

        centroids_per_label[best_lbl] = new_centroids
        local_labels_per_label[best_lbl] = new_labels
        wcss_per_label[best_lbl] = new_wcss

        current_total_clusters += 1

        precompute_next_split_refine(best_lbl)

    # Reconstruct
    labels_final = np.empty(n_samples, dtype=int)
    global_cluster_counter = 0

    for lbl_idx in range(n_labels):
        l_indices = indices_per_label[lbl_idx]
        l_centroids = centroids_per_label[lbl_idx]

        # local labels are 0..k_local-1
        labels_final[l_indices] = local_labels_per_label[lbl_idx] + global_cluster_counter

        global_cluster_counter += l_centroids.shape[0]

    return labels_final


class ClusterNode:
    id: int
    lbl_idx: int
    indices: "NDArray"  # boolean mask relative to label group
    centroid: "NDArray"
    wcss: float
    X2_sum: float

    # Pending split info
    split_reduction: float
    split_info: Optional[Tuple["NDArray", "NDArray", "NDArray", "NDArray"]] = (
        None  # labels, centroids, wcss_pair, counts
    )

    def __init__(self, id, lbl_idx, indices, centroid, wcss, X2_sum):
        self.id = id
        self.lbl_idx = lbl_idx
        self.indices = indices
        self.centroid = centroid
        self.wcss = wcss
        self.X2_sum = X2_sum


def precomputed_bisecting_kmeans_by_label_optimized_no_refine(X, y, target_k, seed=42, n_init=10):
    """
    Bisecting K-Means with label constraint and precomputed splits (No Refine strategy).

    In this version (No Refine):
    - We treat every cluster as an independent candidate for splitting.
    - We precompute the split of *every* current cluster into 2 sub-clusters.
    - The heap stores the ACTUAL reduction for that specific cluster split.
    - When a split is accepted, the chosen cluster is replaced by 2 new clusters.
    - We must then precompute the potential splits for these 2 new clusters.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, dim = X.shape

    unique_labels, y_inverse = np.unique(y, return_inverse=True)
    n_labels = len(unique_labels)

    if target_k == n_samples:
        return np.arange(n_samples)

    # Group data by label
    order = np.argsort(y_inverse)
    X_sorted = X[order]
    y_sorted = y_inverse[order]

    counts = np.bincount(y_sorted)
    splits = np.cumsum(counts)[:-1]

    groups = np.split(X_sorted, splits)
    idx_groups = np.split(order, splits)

    points_per_label = groups
    indices_per_label = idx_groups
    X2_per_label_list = []

    cluster_map: dict[int, ClusterNode] = {}
    next_cluster_id = 0

    heap: list[Tuple[float, int]] = []

    # Calculate X2 per label upfront
    for lbl_idx in range(n_labels):
        pts = points_per_label[lbl_idx]
        X2 = np.einsum("ij,ij->i", pts, pts)
        X2_per_label_list.append(X2)

    current_total_clusters = 0

    def compute_split_candidate(node: ClusterNode):
        pts = points_per_label[node.lbl_idx]
        X2 = X2_per_label_list[node.lbl_idx]

        cluster_pts = pts[node.indices]
        cluster_X2 = X2[node.indices]

        if cluster_pts.shape[0] < 2:
            node.split_reduction = -1.0
            return

        best_wcss_total = float("inf")
        best_sub_labels = None
        best_sub_centroids = None
        best_wcss_split = None
        best_counts = None

        for _ in range(n_init):
            if cluster_pts.shape[0] == 2:
                init_c = cluster_pts
            else:
                init_c = kmeans_plus_plus_init(cluster_pts, 2, rng)

            sub_lbls, sub_ctrs = kmeans(cluster_pts, k=2, seed=rng, init_centroids=init_c)

            sub_counts = np.bincount(sub_lbls, minlength=2)
            sub_X2_sums = np.bincount(sub_lbls, weights=cluster_X2, minlength=2)

            wcss_split = sub_X2_sums - sub_counts * np.einsum("ij,ij->i", sub_ctrs, sub_ctrs)
            total_wcss = np.sum(wcss_split)

            if total_wcss < best_wcss_total:
                best_wcss_total = total_wcss
                best_sub_labels = sub_lbls
                best_sub_centroids = sub_ctrs
                best_wcss_split = wcss_split
                best_counts = sub_counts

        if best_sub_labels is not None:
            reduction = node.wcss - best_wcss_total
            node.split_reduction = reduction
            node.split_info = (best_sub_labels, best_sub_centroids, best_wcss_split, best_counts)

            heapq.heappush(heap, (-reduction, node.id))

    # Independent Init
    for lbl_idx in range(n_labels):
        pts = points_per_label[lbl_idx]
        indices = np.ones(pts.shape[0], dtype=bool)  # All points initially

        X2 = X2_per_label_list[lbl_idx]
        sum_X2 = np.sum(X2)

        if pts.shape[0] > 0:
            centroid = pts.mean(axis=0)[None, :]
            wcss = sum_X2 - pts.shape[0] * (centroid[0] @ centroid[0])
        else:
            centroid = np.zeros((1, dim))
            wcss = 0.0

        node = ClusterNode(next_cluster_id, lbl_idx, indices, centroid, wcss, sum_X2)
        cluster_map[next_cluster_id] = node
        next_cluster_id += 1
        current_total_clusters += 1

        compute_split_candidate(node)

    while current_total_clusters < target_k:
        if not heap:
            break

        neg_red, cid = heapq.heappop(heap)
        node = cluster_map.get(cid)

        if node is None:
            # Stale or removed
            continue

        if node.split_info is None:
            continue

        # Perform split
        sub_labels, sub_centroids, sub_wcss_pair, sub_counts = node.split_info

        # Remove old node
        del cluster_map[cid]

        # Create 2 new nodes
        # Need to split node.indices based on sub_labels
        # node.indices is a boolean mask of shape (n_class_pts,)

        # Get indices of points in this cluster relative to the class group
        # np.flatnonzero(node.indices) gives indices into pts
        # sub_labels corresponds to these indices

        current_indices = np.flatnonzero(node.indices)

        # Child 0
        mask0 = np.zeros_like(node.indices)
        idx0 = current_indices[sub_labels == 0]
        mask0[idx0] = True

        # Child 1
        mask1 = np.zeros_like(node.indices)
        idx1 = current_indices[sub_labels == 1]
        mask1[idx1] = True

        # X2 sums needed for nodes?
        # We computed sub_X2_sums earlier but didn't store fully in split_info,
        # wait, we computed wcss, but not sum_X2.
        # Actually `best_wcss_split` = sum_X2_part - N * c^2
        # So sum_X2_part = wcss + N * c^2

        sum_X2_0 = sub_wcss_pair[0] + sub_counts[0] * (sub_centroids[0] @ sub_centroids[0])
        sum_X2_1 = sub_wcss_pair[1] + sub_counts[1] * (sub_centroids[1] @ sub_centroids[1])

        node0 = ClusterNode(
            next_cluster_id,
            node.lbl_idx,
            mask0,
            sub_centroids[0][None, :],
            sub_wcss_pair[0],
            sum_X2_0,
        )
        cluster_map[next_cluster_id] = node0
        next_cluster_id += 1

        node1 = ClusterNode(
            next_cluster_id,
            node.lbl_idx,
            mask1,
            sub_centroids[1][None, :],
            sub_wcss_pair[1],
            sum_X2_1,
        )
        cluster_map[next_cluster_id] = node1
        next_cluster_id += 1

        current_total_clusters += 1  # Removed 1, Added 2 -> +1 total

        # Compute candidates
        compute_split_candidate(node0)
        compute_split_candidate(node1)

    # Final reconstruction
    labels_final = np.empty(n_samples, dtype=int)

    # We have cluster nodes. We need to assign IDs to them.
    # Group nodes by label to ensure contiguous IDs for output if desired?
    # Original implementation: "global_cluster_counter" implies order by label.

    # Let's collect nodes by label
    nodes_by_label: dict[int, list[ClusterNode]] = {}
    for node in cluster_map.values():
        if node.lbl_idx not in nodes_by_label:
            nodes_by_label[node.lbl_idx] = []
        nodes_by_label[node.lbl_idx].append(node)

    global_counter = 0
    for lbl_idx in range(n_labels):
        class_nodes = nodes_by_label.get(lbl_idx, [])
        # Order doesn't strictly matter for correctness, but for stability maybe?
        # Let's just iterate

        # Points global indices
        global_indices = indices_per_label[lbl_idx]

        # For each node, mark its points in labels_final
        for node in class_nodes:
            # node.indices is boolean mask relative to class
            # global_indices[node.indices] give global indices
            labels_final[global_indices[node.indices]] = global_counter
            global_counter += 1

    return labels_final
