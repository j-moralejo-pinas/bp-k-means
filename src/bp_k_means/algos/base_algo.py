"""Shared interface for clustering algorithms."""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BaseAlgo(ABC):
    """Base interface for algorithms that produce cluster labels."""

    def __init__(
        self,
        seed: int | np.random.Generator,
        n_init: int = 1,
    ) -> None:
        """Initialize common algorithm settings.

        Parameters
        ----------
        seed : int | np.random.Generator
            Seed or random generator used by the algorithm.
        n_init : int, default=1
            Number of initialization attempts when supported by the algorithm. Algorithms that do
            not use multiple initializations can rely on the default.

        Raises
        ------
        ValueError
            If ``n_init`` is less than one.
        """
        if n_init < 1:
            msg = "n_init must be >= 1"
            raise ValueError(msg)
        self.seed = seed
        self.n_init = n_init
        self.labels_: NDArray | None = None
        self.centroids_: NDArray | None = None
        self._cluster_ids: NDArray | None = None
        self._cluster_source_labels: NDArray | None = None
        self._cluster_sizes: NDArray | None = None

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None,
        target_k: int,
    ) -> Self:
        """Fit the algorithm and store its cluster labels.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix with shape ``(n_samples, n_features)``.
        y : ArrayLike | None
            Optional labels used by constrained algorithms.
        target_k : int
            Requested number of clusters.

        Returns
        -------
        Self
            The fitted algorithm instance.
        """
        raise NotImplementedError

    def fit_predict(
        self,
        X: ArrayLike,
        y: ArrayLike | None,
        target_k: int,
    ) -> NDArray:
        """Fit the algorithm, retain its prediction state, and return its cluster labels.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix with shape ``(n_samples, n_features)``.
        y : ArrayLike | None
            Optional labels used by constrained algorithms.
        target_k : int
            Requested number of clusters.

        Returns
        -------
        NDArray
            Cluster label for each input row.

        Notes
        -----
        This calls ``fit`` once and returns the labels stored on the fitted instance. Subsequent
        calls to ``predict`` use that retained fitted state without fitting again.
        """
        fitted = self.fit(X, y, target_k)
        if fitted.labels_ is None:
            msg = "Algorithm did not produce cluster labels"
            raise RuntimeError(msg)
        return fitted.labels_

    @abstractmethod
    def predict(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
    ) -> NDArray:
        """Assign clusters while optionally respecting one label per input instance.

        Predictions use the cluster state learned by ``fit``. If ``y`` is provided for a
        label-constrained algorithm, only clusters trained from the corresponding label are
        considered. If ``y`` is omitted, all trained clusters are considered.
        """
        raise NotImplementedError

    def _set_cluster_result(
        self,
        X: ArrayLike,
        y: ArrayLike,
        labels: NDArray,
    ) -> Self:
        """Store labels, centroids, and source-label ownership for a fitted clustering."""
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        cluster_ids = np.unique(labels)
        centroids = np.vstack([X_array[labels == cluster].mean(axis=0) for cluster in cluster_ids])
        source_labels = np.asarray([y_array[labels == cluster][0] for cluster in cluster_ids])
        return self._set_result(labels, centroids, source_labels)

    def _validate_prediction_input(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
    ) -> tuple[NDArray, NDArray | None]:
        """Validate prediction inputs without defining an assignment rule."""
        if self.centroids_ is None or self._cluster_ids is None:
            msg = "The algorithm must be fitted before calling predict"
            raise RuntimeError(msg)

        X_array = np.asarray(X)
        if X_array.ndim != self.centroids_.ndim or X_array.shape[1] != self.centroids_.shape[1]:
            msg = f"Expected input with {self.centroids_.shape[1]} features"
            raise ValueError(msg)

        if y is None:
            return X_array, None
        y_array = np.asarray(y)
        if y_array.shape != (X_array.shape[0],):
            msg = "y must contain one label per input instance"
            raise ValueError(msg)
        return X_array, y_array

    def _squared_centroid_distances(self, X: NDArray) -> NDArray:
        """Calculate squared distances from samples to fitted centroids."""
        assert self.centroids_ is not None
        return np.sum((X[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2)

    def _select_lowest_cost_clusters(
        self,
        costs: NDArray,
        y: NDArray | None = None,
    ) -> NDArray:
        """Select minimum-cost fitted clusters, optionally constrained by source label."""
        assert self._cluster_ids is not None
        if y is not None:
            assert self._cluster_source_labels is not None
            compatible = y[:, None] == self._cluster_source_labels[None, :]
            if not np.all(np.any(compatible, axis=1)):
                msg = "No fitted cluster is available for at least one input label"
                raise ValueError(msg)
            costs = np.where(compatible, costs, np.inf)
        return self._cluster_ids[np.argmin(costs, axis=1)]

    def _set_result(
        self,
        labels: NDArray,
        centroids: NDArray | None = None,
        source_labels: NDArray | None = None,
    ) -> Self:
        """Store fitted labels and optional centroids."""
        self.labels_ = labels
        self.centroids_ = centroids
        self._cluster_ids = np.unique(labels)
        self._cluster_source_labels = source_labels
        self._cluster_sizes = np.asarray(
            [np.count_nonzero(labels == cluster) for cluster in self._cluster_ids]
        )
        return self
