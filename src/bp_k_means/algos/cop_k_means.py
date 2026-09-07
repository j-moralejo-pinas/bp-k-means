"""COP-KMeans: a specialized label-constrained K-means implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bp_k_means.algos.base_algo import BaseAlgo
from bp_k_means.utils.logging import logger

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def _assign_points(
    X: NDArray,
    y: NDArray,
    centroids: NDArray,
    labels: NDArray,
) -> tuple[NDArray, bool] | None:
    """
    Assign points to the nearest feasible cluster.

    Parameters
    ----------
    X : NDArray
        Points to assign.
    y : NDArray
        Source label for each point.
    centroids : NDArray
        Current cluster centroids.
    labels : NDArray
        Mutable current cluster assignments.

    Returns
    -------
    tuple[NDArray, bool] | None
        Updated labels and a flag indicating whether any assignment changed.
        ``None`` is returned when no feasible cluster exists for a point.
    """
    changed = False
    for idx in range(len(X)):
        distances = np.sum((X[idx] - centroids) ** 2, axis=1)
        for cluster_idx in np.argsort(distances):
            same_cluster = np.where(labels == cluster_idx)[0]
            if len(same_cluster) > 0 and np.any(y[same_cluster] != y[idx]):
                continue
            if labels[idx] != cluster_idx:
                changed = True
            labels[idx] = cluster_idx
            break
        else:
            logger.warning(
                "No feasible cluster for point %s. Label: %s. COP-KMeans fails.",
                idx,
                y[idx],
            )
            return None
    return labels, changed


def _update_centroids(
    X: NDArray,
    labels: NDArray,
    centroids: NDArray,
    rng: np.random.Generator,
) -> NDArray:
    """
    Recompute centroids and reseed empty clusters.

    Parameters
    ----------
    X : NDArray
        Input points.
    labels : NDArray
        Current cluster assignment for each point.
    centroids : NDArray
        Current centroid matrix, used to determine the number of clusters.
    rng : np.random.Generator
        Generator used to reseed empty clusters.

    Returns
    -------
    NDArray
        Updated centroid matrix.
    """
    new_centroids = np.zeros_like(centroids)
    for cluster_idx in range(len(centroids)):
        points = X[labels == cluster_idx]
        new_centroids[cluster_idx] = (
            points.mean(axis=0) if len(points) > 0 else X[rng.choice(len(X))]
        )
    return new_centroids


def cop_kmeans_cannot_link(
    X: NDArray,
    y: NDArray,
    k: int,
    max_iter: int = 300,
    *,
    seed: int | np.random.Generator,
    init_ensure_label: bool = True,
) -> tuple[NDArray, NDArray] | tuple[None, None]:
    """
    Cluster points while enforcing cannot-link constraints between labels.

    Parameters
    ----------
    X : NDArray
        Input points with shape ``(n_samples, n_features)``.
    y : NDArray
        Source label for each point.
    k : int
        Number of clusters.
    max_iter : int
        Maximum number of assignment/update iterations.
    seed : int | np.random.Generator
        Random seed or generator used for initialization.
    init_ensure_label : bool
        Whether to initialize at least one centroid from every source label.

    Returns
    -------
    tuple[NDArray, NDArray] | tuple[None, None]
        Feasible cluster assignments and centroids, or ``None`` values when
        no feasible assignment exists.

    Raises
    ------
    ValueError
        If ``k`` is smaller than the number of unique source labels.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n, _d = X.shape

    # Map labels to indices for speed
    labels = np.unique(y)

    if k < len(labels):
        msg = "Infeasible: k is smaller than the number of labels."
        raise ValueError(msg)

    if init_ensure_label:
        # Initialize centroids: ensure at least one centroid per label
        initial_indices = [rng.choice(np.flatnonzero(y == label)) for label in labels]

        remaining_count = k - len(initial_indices)
        if remaining_count > 0:
            available_indices = np.setdiff1d(np.arange(n), initial_indices)
            initial_indices.extend(
                rng.choice(available_indices, size=remaining_count, replace=False)
            )

        centroids = X[initial_indices]
    else:
        centroids = X[rng.choice(n, size=k, replace=False)]

    labels = np.full(n, -1, dtype=int)

    for i in range(max_iter):
        logger.debug("COP-KMeans iteration %d/%d", i + 1, max_iter)
        assignment = _assign_points(X, y, centroids, labels)
        if assignment is None:
            return None, None
        labels, changed = assignment
        new_centroids = _update_centroids(X, labels, centroids, rng)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

        if not changed:
            break

    return labels, centroids


class COPKMeansCannotLink(BaseAlgo):
    """
    Specialized K-means with cannot-link constraints between source labels.

    Parameters
    ----------
    max_iter : int
        Maximum number of assignment/update iterations per initialization.
    seed : int | np.random.Generator
        Random seed or generator shared by the fitting implementation.
    n_init : int
        Number of independent fitting attempts.
    init_ensure_label : bool
        Whether initialization includes one centroid per source label.

    Attributes
    ----------
    max_iter : int
        Maximum number of assignment/update iterations per initialization.
    init_ensure_label : bool
        Whether initialization includes one centroid per source label.
    """

    max_iter: int
    init_ensure_label: bool

    def predict(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> NDArray:
        """
        Assign instances to the nearest feasible fitted cluster.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix to predict.
        y : ArrayLike
            Source label for each input row.

        Returns
        -------
        NDArray
            Predicted compatible cluster identifier for each row.
        """
        return self._predict_nearest_centroid(X, y)

    def __init__(
        self,
        max_iter: int = 300,
        *,
        seed: int | np.random.Generator,
        n_init: int,
        init_ensure_label: bool = True,
    ) -> None:
        super().__init__(seed=seed, n_init=n_init)
        self.max_iter = max_iter
        self.init_ensure_label = init_ensure_label

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        target_k: int,
    ) -> COPKMeansCannotLink:
        """
        Fit COP-KMeans and store the best feasible result.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix.
        y : ArrayLike
            Labels used by the cannot-link constraint.
        target_k : int
            Requested number of clusters.

        Returns
        -------
        COPKMeansCannotLink
            The fitted algorithm instance.

        Raises
        ------
        RuntimeError
            If no initialization produces a feasible clustering.

        Attributes
        ----------
        labels_ : NDArray
            Labels from the best feasible initialization.
        centroids_ : NDArray
            Centroids from the best feasible initialization.
        """
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        rng = np.random.default_rng(self.seed) if isinstance(self.seed, int) else self.seed
        best_wcss = float("inf")
        best_labels = None
        best_centroids = None

        for _ in range(self.n_init):
            current_seed = rng.integers(2**32)
            labels, centroids = cop_kmeans_cannot_link(
                X_array,
                y_array,
                target_k,
                max_iter=self.max_iter,
                seed=current_seed,
                init_ensure_label=self.init_ensure_label,
            )
            if labels is None or centroids is None:
                continue
            wcss = float(np.sum((X_array - centroids[labels]) ** 2))
            if wcss < best_wcss:
                best_wcss = wcss
                best_labels = labels
                best_centroids = centroids

        if best_labels is None or best_centroids is None:
            msg = "COPKMeansCannotLink did not produce a feasible clustering"
            raise RuntimeError(msg)
        source_labels = np.asarray(
            [y_array[best_labels == cluster][0] for cluster in np.unique(best_labels)]
        )
        return self._set_result(best_labels, best_centroids, source_labels)
