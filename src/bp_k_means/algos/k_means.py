"""Core k-means initialization and clustering routines."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bp_k_means.algos.base_algo import BaseAlgo

Array = NDArray


def kmeans_plus_plus_init(
    X: Array,
    k: int,
    seed: int | np.random.Generator = 42,
    existing_centroids: Array | None = None,
) -> Array:
    """Initialize centroids with the k-means++ strategy."""
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
    X: Array,
    k: int,
    subsample_size: int,
    seed: int | np.random.Generator = 42,
    existing_centroids: Array | None = None,
) -> Array:
    """
    K-means++ initialisation on a random subsample of X.

    Selects `subsample_size` points uniformly without replacement, then runs standard k-means++ on
    that subset.  All centroids are drawn from the subsample, keeping complexity O(k *
    subsample_size) instead of O(k * n).
    """
    rng = np.random.default_rng(seed) if isinstance(seed, int) else seed
    n = X.shape[0]

    actual_size = min(subsample_size, n)
    sub_idx = rng.choice(n, size=actual_size, replace=False)
    X_sub = X[sub_idx]

    return kmeans_plus_plus_init(X_sub, k, seed=rng, existing_centroids=existing_centroids)


def random_init(
    X: Array,
    k: int,
    seed: int | np.random.Generator = 42,
    existing_centroids: Array | None = None,
) -> Array:
    """
    Random initialisation: pick k distinct points uniformly at random.

    If `existing_centroids` is provided, only the remaining slots are filled with new random points,
    and no new centroid will duplicate an existing one.
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


def kmeans(
    X: Array,
    k: int,
    max_iter: int = 300,
    seed: int | np.random.Generator = 42,
    init_centroids: Array | None = None,
    X2: Array | None = None,
) -> tuple[Array, Array]:
    """Run Lloyd's k-means algorithm and return labels and centroids."""
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
    assert X2 is not None

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
                donor = int(labels[wi])

                # move that point to the empty cluster
                labels[wi] = ec
                centroids[ec] = X[wi]
                counts[ec] = 1
                counts[donor] -= 1
                point_cost[wi] = -np.inf

                # Update donor centroid incrementally and refresh costs for
                # its remaining points so subsequent steals stay accurate.
                if counts[donor] > 0:
                    centroids[donor] = (centroids[donor] * (counts[donor] + 1) - X[wi]) / counts[
                        donor
                    ]
                    donor_mask = labels == donor
                    diff = X[donor_mask] - centroids[donor]
                    point_cost[donor_mask] = np.einsum("ij,ij->i", diff, diff)

    return labels, centroids


class KMeans(BaseAlgo):
    """Common-interface wrapper around Lloyd's k-means algorithm."""

    def __init__(
        self,
        max_iter: int = 300,
        seed: int | np.random.Generator = 42,
        n_init: int = 1,
    ) -> None:
        """Initialize a k-means algorithm.

        Parameters
        ----------
        max_iter : int
            Maximum number of Lloyd iterations per initialization.
        seed : int | np.random.Generator
            Seed or random generator used for initialization.
        n_init : int
            Number of independent initializations.
        """
        super().__init__(seed=seed, n_init=n_init)
        self.max_iter = max_iter

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None,  # noqa: ARG002 - accepted for interface compatibility
        target_k: int,
    ) -> "KMeans":
        """Fit k-means and store the best labels and centroids.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix.
        y : ArrayLike | None
            Ignored labels, accepted for interface compatibility.
        target_k : int
            Requested number of clusters.

        Returns
        -------
        KMeans
            The fitted algorithm instance.
        """
        X_array = np.asarray(X)
        rng = np.random.default_rng(self.seed) if isinstance(self.seed, int) else self.seed
        best_wcss = float("inf")
        best_labels = None
        best_centroids = None

        for _ in range(self.n_init):
            labels, centroids = kmeans(
                X_array,
                target_k,
                max_iter=self.max_iter,
                seed=rng,
            )
            wcss = float(np.sum((X_array - centroids[labels]) ** 2))
            if wcss < best_wcss:
                best_wcss = wcss
                best_labels = labels
                best_centroids = centroids

        assert best_labels is not None
        assert best_centroids is not None
        return self._set_result(best_labels, best_centroids)
