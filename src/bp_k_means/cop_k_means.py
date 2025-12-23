import logging
import numpy as np

logger = logging.getLogger(__name__)


def cop_kmeans_by_class(X, y, k, max_iter=300, seed=42, init_ensure_class=True):
    """
    X: array (n, d)
    y: class labels, integer or string
    k: number of clusters
    init_ensure_class: if True, ensures at least one centroid per class
    """

    rng = np.random.default_rng(seed)
    n, d = X.shape

    # Map classes to indices for speed
    y = np.asarray(y)
    classes = np.unique(y)

    if k < len(classes):
        msg = "Infeasible: k is smaller than the number of classes."
        raise ValueError(msg)

    if init_ensure_class:
        # Initialize centroids: ensure at least one centroid per class
        initial_indices = []
        for cls in classes:
            indices_in_class = np.where(y == cls)[0]
            chosen = rng.choice(indices_in_class)
            initial_indices.append(chosen)

        remaining_count = k - len(initial_indices)
        if remaining_count > 0:
            all_indices = np.arange(n)
            available_indices = np.setdiff1d(all_indices, initial_indices)
            chosen_rest = rng.choice(available_indices, size=remaining_count, replace=False)
            initial_indices.extend(chosen_rest)

        centroids = X[initial_indices]
    else:
        centroids = X[rng.choice(n, size=k, replace=False)]

    labels = np.full(n, -1, dtype=int)

    # For fast cannot-link checks, we only need: y[i] != y[j].
    # No need to build pairwise lists.

    for i in range(max_iter):
        logger.debug(f"COP-KMeans iteration {i+1}/{max_iter}")
        changed = False

        for idx in range(n):
            dist = np.sum((X[idx] - centroids) ** 2, axis=1)
            order = np.argsort(dist)

            assigned = False
            for c in order:
                # Check if assigning idx -> c violates cannot-link:
                # Any point already in cluster c must share the same class.
                same_cluster = np.where(labels == c)[0]

                # If cluster c already contains points of a different class:
                if len(same_cluster) > 0 and np.any(y[same_cluster] != y[idx]):
                    continue  # not allowed

                # Feasible assignment
                if labels[idx] != c:
                    changed = True
                labels[idx] = c
                assigned = True
                break

            if not assigned:
                logger.warning(
                    "No feasible cluster for point %s. Class: %s. COP-KMeans fails.",
                    idx,
                    y[idx],
                )
                return None, None

        new_centroids = np.zeros_like(centroids)
        for ci in range(k):
            pts = X[labels == ci]
            if len(pts) > 0:
                new_centroids[ci] = pts.mean(axis=0)
            else:
                new_centroids[ci] = X[rng.choice(n)]

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

        if not changed:
            break

    return labels, centroids
