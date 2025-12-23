import logging
import numpy as np

from bp_k_means.k_means import kmeans, kmeans_plus_plus_init

logger = logging.getLogger(__name__)


def compute_wcss(X, labels, centroids):
    wcss = 0.0
    for c in range(len(centroids)):
        pts = X[labels == c]
        if len(pts) > 0:
            diff = pts - centroids[c]
            wcss += np.sum(diff * diff)
    return wcss


def experimental_bp_kmeans(
    X,
    y,
    target_k,
    seed=42,
    n_init=10,
    reuse_centroids=0,
    use_wcss_per_cluster=False,
):
    """
    X: array (n, d)
    y: initial class labels
    target_k: desired total number of clusters
    n_init: number of times to run k-means for each split to find the best result
    reuse_centroids:
        0 or False: Do not reuse centroids (random init).
        1 or True: Reuse existing centroids and add one random centroid.
        2: Reuse existing centroids, remove the one with highest WCSS, and add 2 new centroids from that cluster.
    use_wcss_per_cluster:
        False: Select class with highest WCSS.
        True: Select class with highest WCSS / (n_clusters + 1).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]

    # Each label starts with exactly 1 cluster
    classes = np.unique(y)

    # Mapping from class label to list of global cluster ids
    # Initially, each class has exactly one cluster, assigned sequentially
    global_cluster_of_class = {}
    current_cluster_id = 0
    for c in classes:
        global_cluster_of_class[c] = [current_cluster_id]
        current_cluster_id += 1

    labels_global = np.zeros(n, dtype=int)

    # Initial assignment: cluster per class
    for c in classes:
        labels_global[y == c] = global_cluster_of_class[c][0]

    # Initialize WCSS for each class
    wcss_per_class = {}
    centroids_per_class = {}
    n_points_per_class = {}
    for c in classes:
        pts = X[y == c]
        n_points_per_class[c] = len(pts)
        if len(pts) == 0:
            wcss_per_class[c] = 0.0
            centroids_per_class[c] = np.zeros((0, X.shape[1]))
            continue
        centroid = pts.mean(axis=0)
        centroids_per_class[c] = centroid.reshape(1, -1)
        diff = pts - centroid
        wcss = np.sum(diff * diff)
        wcss_per_class[c] = wcss

    # While total number of clusters < target_k, split the worst class
    while current_cluster_id < target_k:
        logger.debug(
            f"BP-KMeans iteration: current clusters {current_cluster_id}, target {target_k}"
        )
        # Select class to split
        if not use_wcss_per_cluster:
            worst_class = max(wcss_per_class, key=wcss_per_class.get)
        else:
            scores = {}
            for c in wcss_per_class:
                wcss = wcss_per_class[c]
                n_clusters = len(centroids_per_class[c])
                n_points = n_points_per_class[c]

                if n_clusters + 1 == n_points:
                    scores[c] = wcss
                else:
                    scores[c] = wcss / (n_clusters + 1)
            worst_class = max(scores, key=scores.get)

        pts = X[y == worst_class]
        current_centroids = centroids_per_class[worst_class]
        old_k = len(current_centroids)
        new_k = old_k + 1

        # Local kmeans on only this class
        # Run n_init times and keep the best result
        best_wcss = float("inf")
        best_labels = None
        best_centroids = None

        for i in range(n_init):
            # Pick a new random centroid
            local_rng = np.random.default_rng(rng.integers(2**32))
            init_centroids = None

            if reuse_centroids == 2:
                if len(pts) > 0:
                    # Assign points to current centroids to find local clusters
                    dists = np.sum((pts[:, None, :] - current_centroids[None, :, :]) ** 2, axis=2)
                    local_labels = np.argmin(dists, axis=1)

                    # Compute WCSS per cluster
                    max_wcss = -1.0
                    max_wcss_idx = -1

                    for k_idx in range(len(current_centroids)):
                        cluster_pts = pts[local_labels == k_idx]
                        if len(cluster_pts) > 0:
                            diff = cluster_pts - current_centroids[k_idx]
                            c_wcss = np.sum(diff * diff)
                        else:
                            c_wcss = 0.0

                        if c_wcss > max_wcss:
                            max_wcss = c_wcss
                            max_wcss_idx = k_idx

                    if max_wcss_idx != -1:
                        remaining_centroids = np.delete(current_centroids, max_wcss_idx, axis=0)
                        target_pts = pts[local_labels == max_wcss_idx]

                        if len(target_pts) >= 2:
                            new_centroids = kmeans_plus_plus_init(target_pts, 2, local_rng)
                        elif len(target_pts) == 1:
                            c1 = target_pts[0]
                            c2 = pts[local_rng.integers(0, len(pts))]
                            new_centroids = np.vstack([c1, c2])
                        else:
                            if len(pts) >= 2:
                                idx = local_rng.choice(len(pts), size=2, replace=False)
                                new_centroids = pts[idx]
                            else:
                                new_centroids = np.vstack([pts[0], pts[0]])

                        init_centroids = np.vstack([remaining_centroids, new_centroids])
                    else:
                        init_centroids = None
                else:
                    init_centroids = None

            elif reuse_centroids == 1:
                if len(pts) > 0:
                    init_centroids = kmeans_plus_plus_init(
                        pts,
                        len(current_centroids) + 1,
                        local_rng,
                        existing_centroids=current_centroids,
                    )
                else:
                    init_centroids = None

            lbls, ctrs = kmeans(pts, new_k, seed=seed + i, init_centroids=init_centroids)
            wcss = compute_wcss(pts, lbls, ctrs)

            if wcss < best_wcss:
                best_wcss = wcss
                best_labels = lbls
                best_centroids = ctrs

        if best_labels is None:
            msg = "n_init must be >= 1"
            raise ValueError(msg)

        local_labels = best_labels

        # Update WCSS and centroids for the modified class
        wcss_per_class[worst_class] = best_wcss
        centroids_per_class[worst_class] = best_centroids

        # Assign new global cluster ids for this class
        new_cluster_ids = []
        for _ in range(new_k - old_k):
            new_cluster_ids.append(current_cluster_id)
            current_cluster_id += 1

        cluster_ids = global_cluster_of_class[worst_class]
        cluster_ids.extend(new_cluster_ids)

        new_global_map = dict(enumerate(cluster_ids))

        global_labels_subset = np.array([new_global_map[lbl] for lbl in local_labels])

        labels_global[y == worst_class] = global_labels_subset

    return labels_global
