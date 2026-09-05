"""COP-KMeans implementation with cannot-link constraints between labels."""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from bp_k_means.algos.base_algo import BaseAlgo
from bp_k_means.utils.logging import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _assign_points(
    X: "NDArray",
    y: "NDArray",
    centroids: "NDArray",
    labels: "NDArray",
) -> tuple["NDArray", bool] | None:
    """Assign points to the nearest feasible cluster."""
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
    X: "NDArray",
    labels: "NDArray",
    centroids: "NDArray",
    rng: np.random.Generator,
) -> "NDArray":
    """Recompute centroids and reseed empty clusters."""
    new_centroids = np.zeros_like(centroids)
    for cluster_idx in range(len(centroids)):
        points = X[labels == cluster_idx]
        new_centroids[cluster_idx] = (
            points.mean(axis=0) if len(points) > 0 else X[rng.choice(len(X))]
        )
    return new_centroids


def cop_kmeans_by_label(
    X: "NDArray",
    y: "NDArray",
    k: int,
    max_iter: int = 300,
    *,
    seed: int | np.random.Generator,
    init_ensure_label: bool = True,
) -> tuple["NDArray", "NDArray"] | tuple[None, None]:
    """
    Cluster points while enforcing cannot-link constraints between labels.

    X: array (n, d)
    y: labels, integer or string
    k: number of clusters
    init_ensure_label: if True, ensures at least one centroid per label.
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
        initial_indices = []
        for label in labels:
            indices_in_label = np.where(y == label)[0]
            chosen = rng.choice(indices_in_label)
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


class COPKMeans(BaseAlgo):
    """Common-interface wrapper around COP-KMeans."""

    def predict(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
    ) -> "NDArray":
        """Assign instances to the nearest feasible fitted COP-KMeans cluster."""
        X_array, y_array = self._validate_prediction_input(X, y)
        distances = self._squared_centroid_distances(X_array)
        return self._select_lowest_cost_clusters(distances, y_array)

    def __init__(
        self,
        max_iter: int = 300,
        *,
        seed: int | np.random.Generator,
        n_init: int,
        init_ensure_label: bool = True,
    ) -> None:
        """Initialize COP-KMeans.

        Parameters
        ----------
        max_iter : int
            Maximum number of assignment/update iterations per initialization.
        seed : int | np.random.Generator
            Seed or random generator used by the algorithm.
        n_init : int
            Number of independent initializations.
        init_ensure_label : bool
            Whether initialization must include one centroid per label.
        """
        super().__init__(seed=seed, n_init=n_init)
        self.max_iter = max_iter
        self.init_ensure_label = init_ensure_label

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None,
        target_k: int,
    ) -> "COPKMeans":
        """Fit COP-KMeans and store the best feasible result.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix.
        y : ArrayLike | None
            Labels used by the cannot-link constraint.
        target_k : int
            Requested number of clusters.

        Returns
        -------
        COPKMeans
            The fitted algorithm instance.

        Raises
        ------
        ValueError
            If original labels are not provided.
        RuntimeError
            If no initialization produces a feasible clustering.
        """
        if y is None:
            msg = "COPKMeans requires original labels"
            raise ValueError(msg)

        X_array = np.asarray(X)
        y_array = np.asarray(y)
        rng = np.random.default_rng(self.seed) if isinstance(self.seed, int) else self.seed
        best_wcss = float("inf")
        best_labels = None
        best_centroids = None

        for _ in range(self.n_init):
            current_seed = rng.integers(2**32)
            labels, centroids = cop_kmeans_by_label(
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
            msg = "COPKMeans did not produce a feasible clustering"
            raise RuntimeError(msg)
        source_labels = np.asarray(
            [y_array[best_labels == cluster][0] for cluster in np.unique(best_labels)]
        )
        return self._set_result(best_labels, best_centroids, source_labels)
