import numpy as np

from bp_k_means.k_means import kmeans, kmeans_plus_plus_init

# Assumes you have your previous kmeans(X, k, max_iter=..., tol=..., seed=...) defined somewhere


def divisive_kmeans_by_label(X, y, target_k, seed=42, refine_clusters=False):
    """
    Divisive hierarchical clustering (bisecting k-means) with label constraint.

    X: (n, d) data
    y: (n,) labels (e.g. CUSEC)
    target_k: desired final number of clusters
    seed: RNG seed

    Returns:
        labels_final: (n,) cluster ids in [0, target_k - 1]
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y)
    n, d = X.shape

    unique_labels = np.unique(y)
    if target_k < len(unique_labels):
        raise ValueError(
            f"target_k={target_k} < number of labels={len(unique_labels)}. "
            f"With label-pure clusters you cannot go below that."
        )

    def cluster_wcss(indices):
        pts = X[indices]
        centroid = pts.mean(axis=0)
        diff = pts - centroid
        return np.sum(diff * diff)

    # Each label starts as one big cluster
    clusters = []
    for lbl in unique_labels:
        idxs = np.where(y == lbl)[0]
        if idxs.size == 0:
            continue
        clusters.append({"indices": idxs, "label": lbl, "wcss": cluster_wcss(idxs)})

    while len(clusters) < target_k:
        best_idx = None
        best_wcss = -np.inf

        # Choose the cluster with largest WCSS that can be split
        for i, c in enumerate(clusters):
            idxs = c["indices"]
            if idxs.size < 2:
                continue
            wcss = c["wcss"]
            if wcss > best_wcss:
                best_wcss = wcss
                best_idx = i

        if best_idx is None:
            # No cluster large enough to split but we haven't reached target_k
            raise ValueError(
                f"Cannot reach target_k={target_k}: all remaining clusters have size 1. "
                f"Current clusters={len(clusters)}."
            )

        if refine_clusters:
            cluster_to_split = clusters[best_idx]
            lbl = cluster_to_split["label"]

            # Gather other clusters of the same label
            other_clusters = [
                c for i, c in enumerate(clusters) if c["label"] == lbl and i != best_idx
            ]

            # Init 2 new centroids for the split cluster
            pts_split = X[cluster_to_split["indices"]]
            new_centroids = kmeans_plus_plus_init(pts_split, 2, rng)

            # Collect centroids from other clusters
            other_centroids = []
            for c in other_clusters:
                pts_c = X[c["indices"]]
                other_centroids.append(pts_c.mean(axis=0))

            if other_centroids:
                init_centroids = np.vstack(other_centroids + [new_centroids])
            else:
                init_centroids = new_centroids

            # Run global k-means on all points of this label
            lbl_indices = np.where(y == lbl)[0]
            pts_lbl = X[lbl_indices]

            k_new = len(init_centroids)
            seed_val = rng.integers(1_000_000_000)

            new_labels_local, _ = kmeans(
                pts_lbl, k=k_new, seed=seed_val, init_centroids=init_centroids
            )

            # Remove old clusters of this label
            clusters = [c for c in clusters if c["label"] != lbl]

            # Add new clusters
            for k_idx in range(k_new):
                mask = new_labels_local == k_idx
                idxs = lbl_indices[mask]
                if idxs.size > 0:
                    clusters.append(
                        {"indices": idxs, "label": lbl, "wcss": cluster_wcss(idxs)}
                    )
        else:
            cluster = clusters.pop(best_idx)
            idxs = cluster["indices"]
            lbl = cluster["label"]
            pts = X[idxs]

            # Try k-means with k=2 on this cluster's points
            local_labels, _ = kmeans(pts, k=2, seed=rng)

            # Handle degenerate case where k-means puts all points in one child
            mask1 = local_labels == 0
            if mask1.sum() == 0 or mask1.sum() == idxs.size:
                # Fallback: deterministic split by halving with a random permutation
                perm = rng.permutation(idxs.size)
                half = idxs.size // 2
                child1_indices = idxs[perm[:half]]
                child2_indices = idxs[perm[half:]]
            else:
                child1_indices = idxs[mask1]
                child2_indices = idxs[~mask1]

            clusters.append(
                {
                    "indices": child1_indices,
                    "label": lbl,
                    "wcss": cluster_wcss(child1_indices),
                }
            )
            clusters.append(
                {
                    "indices": child2_indices,
                    "label": lbl,
                    "wcss": cluster_wcss(child2_indices),
                }
            )

    # Build final labels [0 .. target_k-1]
    labels_final = np.empty(n, dtype=int)
    for cid, c in enumerate(clusters):
        labels_final[c["indices"]] = cid

    return labels_final
