"""Boundary Preserving K-Means algorithms."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from bp_k_means.algos.bp_kmeans import (
    BPKMeans,
    InitAlgorithm,
    InitStrategy,
    RankingMetric,
    bp_kmeans,
)

try:
    __version__ = _distribution_version("bp-k-means")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "BPKMeans",
    "InitAlgorithm",
    "InitStrategy",
    "RankingMetric",
    "__version__",
    "bp_kmeans",
]
