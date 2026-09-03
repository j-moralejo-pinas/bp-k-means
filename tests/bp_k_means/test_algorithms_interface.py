"""Functional tests for the common algorithm interface."""

import numpy as np
import pytest

from bp_k_means.algos.base_algo import BaseAlgo
from bp_k_means.algos.bisecting_k_means_optimized import (
    BisectingKMeans,
    BisectingKMeansNoRefine,
)
from bp_k_means.algos.bp_kmeans import BPKMeans
from bp_k_means.algos.cop_k_means import COPKMeans
from bp_k_means.algos.hac import HACWard, HACWardNNC
from bp_k_means.algos.k_means import KMeans
from bp_k_means.algos.precomputed_bisecting_k_means_optimized import (
    PrecomputedBisectingKMeans,
    PrecomputedBisectingKMeansNoRefine,
)


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Create a small labeled dataset for interface checks."""
    rng = np.random.default_rng(7)
    X = np.vstack(
        [rng.normal(loc=center, scale=0.2, size=(6, 2)) for center in ([0, 0], [5, 5], [10, 0])]
    )
    y = np.repeat([0, 1, 2], 6)
    return X, y


@pytest.mark.parametrize(
    "algo",
    [
        KMeans(seed=42, n_init=1),
        BPKMeans(seed=42, n_init=1, subsample_size=10),
        BisectingKMeans(seed=42, n_init=1),
        BisectingKMeansNoRefine(seed=42, n_init=1),
        COPKMeans(seed=42, n_init=1),
        HACWard(seed=42, n_init=1),
        HACWardNNC(seed=42, n_init=1),
        PrecomputedBisectingKMeans(seed=42, n_init=1),
        PrecomputedBisectingKMeansNoRefine(seed=42, n_init=1),
    ],
    ids=lambda algo: type(algo).__name__,
)
def test_algorithms_share_fit_predict_interface(
    algo: BaseAlgo,
    sample_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Every algorithm returns labels and stores them on the fitted instance."""
    X, y = sample_data
    labels = algo.fit_predict(X, y, target_k=6)

    assert labels.shape == (X.shape[0],)
    assert len(np.unique(labels)) == 6
    assert algo.labels_ is labels

    if isinstance(algo, (KMeans, COPKMeans)):
        assert algo.centroids_ is not None

    if not isinstance(algo, KMeans):
        assert all(len(np.unique(y[labels == cluster])) == 1 for cluster in np.unique(labels))
