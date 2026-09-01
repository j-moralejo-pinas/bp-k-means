"""Shared interface for clustering algorithms."""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BaseAlgo(ABC):
    """Base interface for algorithms that produce cluster labels."""

    def __init__(
        self,
        seed: int | np.random.Generator = 42,
        n_init: int = 1,
    ) -> None:
        """Initialize common algorithm settings.

        Parameters
        ----------
        seed : int | np.random.Generator
            Seed or random generator used by the algorithm.
        n_init : int
            Number of initialization attempts when supported by the algorithm.

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
        """Fit the algorithm and return its cluster labels.

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
        """
        fitted = self.fit(X, y, target_k)
        if fitted.labels_ is None:
            msg = "Algorithm did not produce cluster labels"
            raise RuntimeError(msg)
        return fitted.labels_

    def _set_result(
        self,
        labels: NDArray,
        centroids: NDArray | None = None,
    ) -> Self:
        """Store fitted labels and optional centroids."""
        self.labels_ = labels
        self.centroids_ = centroids
        return self
