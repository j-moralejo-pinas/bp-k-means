"""Shared interface for clustering algorithms."""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BaseAlgo(ABC):
    """
    Base interface for algorithms that produce cluster labels.

    Parameters
    ----------
    seed : int | np.random.Generator
        Random seed or generator shared by the fitting implementation.
    n_init : int
        Number of independent fitting attempts.

    Attributes
    ----------
    seed : int | np.random.Generator
        Random seed or generator shared by the fitting implementation.
    n_init : int
        Number of independent fitting attempts.
    labels_ : NDArray | None
        Cluster label assigned to each training sample after fitting.
    centroids_ : NDArray | None
        Fitted cluster centroids when the algorithm exposes them.

    Raises
    ------
    ValueError
        If ``n_init`` is less than one.
    """

    seed: int | np.random.Generator
    n_init: int
    labels_: NDArray | None
    centroids_: NDArray | None
    _cluster_ids: NDArray | None
    _cluster_source_labels: NDArray | None
    _cluster_sizes: NDArray | None

    def __init__(
        self,
        seed: int | np.random.Generator,
        n_init: int = 1,
    ) -> None:
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
    def fit(  # noqa: D102
        self,
        X: ArrayLike,
        y: ArrayLike,
        target_k: int,
    ) -> Self: ...

    def fit_predict(
        self,
        X: ArrayLike,
        y: ArrayLike,
        target_k: int,
    ) -> NDArray:
        """
        Fit the algorithm, retain its prediction state, and return its cluster labels.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix with shape ``(n_samples, n_features)``.
        y : ArrayLike
            Labels used by constrained algorithms.
        target_k : int
            Requested number of clusters.

        Returns
        -------
        NDArray
            Cluster label for each input row.

        Raises
        ------
        RuntimeError
            If the concrete algorithm does not store labels during fitting.

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
    def predict(  # noqa: D102
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> NDArray: ...

    def _set_cluster_result(
        self,
        X: ArrayLike,
        y: ArrayLike,
        labels: NDArray,
    ) -> Self:
        """
        Store labels, centroids, and source-label ownership for a fitted clustering.

        Parameters
        ----------
        X : ArrayLike
            Training feature matrix.
        y : ArrayLike
            Source label for each training sample.
        labels : NDArray
            Cluster identifier for each row in ``X``.

        Returns
        -------
        Self
            This algorithm instance.
        """
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        cluster_ids = np.unique(labels)
        centroids = np.vstack([X_array[labels == cluster].mean(axis=0) for cluster in cluster_ids])
        source_labels = np.asarray([y_array[labels == cluster][0] for cluster in cluster_ids])
        return self._set_result(labels, centroids, source_labels)

    def _validate_prediction_input(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> tuple[NDArray, NDArray]:
        """
        Validate prediction inputs.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix to predict.
        y : ArrayLike
            Source label for each row of ``X``.

        Returns
        -------
        tuple[NDArray, NDArray]
            Normalized feature and source-label arrays.

        Raises
        ------
        RuntimeError
            If the algorithm has not been fitted.
        ValueError
            If the feature count or number of source labels is invalid.
        """
        if self.centroids_ is None or self._cluster_ids is None:
            msg = "The algorithm must be fitted before calling predict"
            raise RuntimeError(msg)

        X_array = np.asarray(X)
        if X_array.ndim != self.centroids_.ndim or X_array.shape[1] != self.centroids_.shape[1]:
            msg = f"Expected input with {self.centroids_.shape[1]} features"
            raise ValueError(msg)

        y_array = np.asarray(y)
        if y_array.shape != (X_array.shape[0],):
            msg = "y must contain one label per input instance"
            raise ValueError(msg)
        return X_array, y_array

    def _squared_centroid_distances(self, X: NDArray) -> NDArray:
        """
        Calculate squared distances from samples to fitted centroids.

        Parameters
        ----------
        X : NDArray
            Feature matrix with shape ``(n_samples, n_features)``.

        Returns
        -------
        NDArray
            Squared distances with shape ``(n_samples, n_clusters)``.
        """
        assert self.centroids_ is not None
        return np.sum((X[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2)

    def _predict_nearest_centroid(self, X: ArrayLike, y: ArrayLike) -> NDArray:
        """
        Assign samples to their nearest fitted centroid, respecting labels.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix to predict.
        y : ArrayLike
            Source label for each row of ``X``.

        Returns
        -------
        NDArray
            Predicted cluster identifier for every input row.
        """
        X_array, y_array = self._validate_prediction_input(X, y)
        return self._select_lowest_cost_clusters(self._squared_centroid_distances(X_array), y_array)

    def _select_lowest_cost_clusters(
        self,
        costs: NDArray,
        y: NDArray,
    ) -> NDArray:
        """
        Select minimum-cost fitted clusters for each source label.

        Parameters
        ----------
        costs : NDArray
            Cost matrix with one row per sample and one column per cluster.
        y : NDArray
            Source label for each sample.

        Returns
        -------
        NDArray
            Identifier of the lowest-cost compatible cluster per sample.

        Raises
        ------
        ValueError
            If any source label has no compatible fitted cluster.
        """
        assert self._cluster_ids is not None
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
        """
        Store fitted labels and optional centroids.

        Parameters
        ----------
        labels : NDArray
            Cluster identifier for each training sample.
        centroids : NDArray | None
            Centroid matrix for the fitted clusters.
        source_labels : NDArray | None
            Source label associated with each cluster.

        Returns
        -------
        Self
            This algorithm instance.

        Notes
        -----
        The fitted attributes ``labels_``, ``centroids_``, ``_cluster_ids``,
        ``_cluster_source_labels``, and ``_cluster_sizes`` are updated in place.
        """
        self.labels_ = labels
        self.centroids_ = centroids
        self._cluster_ids = np.unique(labels)
        self._cluster_source_labels = source_labels
        self._cluster_sizes = np.asarray(
            [np.count_nonzero(labels == cluster) for cluster in self._cluster_ids]
        )
        return self
