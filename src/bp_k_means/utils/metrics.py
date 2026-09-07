"""Shared clustering metrics."""

import numpy as np


def overall_wcss(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the total within-cluster sum of squares.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix with shape ``(n_samples, n_features)``.
    labels : np.ndarray
        Cluster identifier for each row of ``X``.

    Returns
    -------
    float
        Sum of squared distances from each point to its cluster centroid.
    """
    k = labels.max() + 1
    wcss = 0.0

    for c in range(k):
        pts = X[labels == c]
        if len(pts) > 0:
            centroid = pts.mean(axis=0)
            diff = pts - centroid
            wcss += np.sum(diff * diff)

    return wcss
