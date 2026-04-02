import numpy as np


def kmeans_plus_plus_init(X, k, seed: int | np.random.Generator = 42, existing_centroids=None):
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    n, d = X.shape
    centroids = np.empty((k, d))

    X2 = np.einsum("ij,ij->i", X, X)

    start_idx = 0
    if existing_centroids is not None and len(existing_centroids) > 0:
        n_existing = existing_centroids.shape[0]
        if n_existing > k:
            msg = f"Existing centroids ({n_existing}) > k ({k})"
            raise ValueError(msg)
        centroids[:n_existing] = existing_centroids
        start_idx = n_existing

        # squared distances to nearest existing centroid
        dists = (
            X2[:, None]
            + np.einsum("ij,ij->i", existing_centroids, existing_centroids)[None, :]
            - 2 * (X @ existing_centroids.T)
        )
        closest_dist_sq = np.min(dists, axis=1)

    else:
        # pick first centroid uniformly at random
        c = X[rng.integers(n)]
        centroids[0] = c
        closest_dist_sq = X2 + (c @ c) - 2 * (X @ c)
        start_idx = 1

    for i in range(start_idx, k):
        closest_dist_sq = np.maximum(closest_dist_sq, 0.0)
        sum_sq = closest_dist_sq.sum()
        if sum_sq > 0:
            r = rng.random() * sum_sq
            idx = np.searchsorted(np.cumsum(closest_dist_sq), r)
        else:
            idx = rng.integers(n)

        c = X[idx]
        centroids[i] = c

        new_dist_sq = X2 + (c @ c) - 2 * (X @ c)
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

    return centroids


def subsampled_kmeans_plus_plus_init(
    X, k, subsample_size, seed: int | np.random.Generator = 42, existing_centroids=None
):
    """K-means++ initialisation on a random subsample of X.

    Selects `subsample_size` points uniformly without replacement, then runs
    standard k-means++ on that subset.  All centroids are drawn from the
    subsample, keeping complexity O(k * subsample_size) instead of O(k * n).
    """
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    n = X.shape[0]

    actual_size = min(subsample_size, n)
    sub_idx = rng.choice(n, size=actual_size, replace=False)
    X_sub = X[sub_idx]

    return kmeans_plus_plus_init(X_sub, k, seed=rng, existing_centroids=existing_centroids)


def random_init(X, k, seed: int | np.random.Generator = 42, existing_centroids=None):
    """Random initialisation: pick k distinct points uniformly at random.

    If `existing_centroids` is provided, only the remaining slots are filled
    with new random points, and no new centroid will duplicate an existing one.
    """
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    n, d = X.shape
    centroids = np.empty((k, d))

    start_idx = 0
    candidate_idx = np.arange(n)

    if existing_centroids is not None and len(existing_centroids) > 0:
        n_existing = existing_centroids.shape[0]
        if n_existing > k:
            msg = f"Existing centroids ({n_existing}) > k ({k})"
            raise ValueError(msg)
        centroids[:n_existing] = existing_centroids
        start_idx = n_existing

        # exclude points that coincide with an existing centroid
        is_existing = np.any(
            np.all(X[:, None, :] == existing_centroids[None, :, :], axis=2), axis=1
        )
        candidate_idx = np.flatnonzero(~is_existing)

    n_new = k - start_idx
    if n_new > 0:
        chosen = rng.choice(candidate_idx, size=n_new, replace=False)
        centroids[start_idx:] = X[chosen]

    return centroids


def kmeans(X, k, max_iter=300, seed: int | np.random.Generator = 42, init_centroids=None, X2=None):
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    n, d = X.shape

    if init_centroids is not None:
        centroids = np.array(init_centroids, copy=True)
        if centroids.shape[0] != k:
            msg = f"Expected {k} centroids, got {centroids.shape[0]}"
            raise ValueError(msg)
    else:
        centroids = kmeans_plus_plus_init(X, k, rng)

    labels = np.full(n, -1, dtype=int)

    if X2 is None:
        X2 = np.einsum("ij,ij->i", X, X)

    for _ in range(max_iter):
        dist = (
            X2[:, None]
            + np.einsum("ij,ij->i", centroids, centroids)[None, :]
            - 2 * (X @ centroids.T)
        )

        new_labels = np.argmin(dist, axis=1)

        if np.array_equal(labels, new_labels):
            labels = new_labels
            break
        labels = new_labels

        centroids = np.zeros((k, d), dtype=X.dtype)

        np.add.at(centroids, labels, X)  # sum points per cluster

        counts = np.bincount(labels, minlength=k)  # (k,)

        nonempty = counts > 0
        centroids[nonempty] /= counts[nonempty, None]

        empty_clusters = np.flatnonzero(~nonempty)

        if empty_clusters.size > 0:
            # distance of each point to its assigned centroid
            point_cost = dist[np.arange(n), labels]

            for ec in empty_clusters:
                # pick the point that is currently worst represented
                wi = np.argmax(point_cost)

                # move that point to the empty cluster
                labels[wi] = ec
                centroids[ec] = X[wi]

                # prevent reusing the same point again
                point_cost[wi] = -np.inf

            # Recompute centroids from updated labels so donor clusters
            # are consistent; without this, a stale centroid can cause
            # a false convergence on the next iteration.
            centroids = np.zeros((k, d), dtype=X.dtype)
            np.add.at(centroids, labels, X)
            counts = np.bincount(labels, minlength=k)
            centroids /= counts[:, None]

    return labels, centroids
