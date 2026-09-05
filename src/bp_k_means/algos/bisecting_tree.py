"""Prediction state shared by non-refined bisecting algorithms."""

import numpy as np
from numpy.typing import NDArray


class _BisectingTreeNode:
    """One fitted split in a non-refined bisecting hierarchy."""

    def __init__(self, centroid: NDArray) -> None:
        self.centroid = centroid
        self.children: tuple[_BisectingTreeNode, _BisectingTreeNode] | None = None
        self.cluster_id: int | None = None


def _assign_from_hierarchy(
    X: NDArray,
    y: NDArray | None,
    roots_by_label: dict[object, _BisectingTreeNode],
) -> NDArray:
    """Assign samples by following fitted centroid splits from root to leaf."""
    roots = list(roots_by_label.values())
    predictions = np.empty(X.shape[0], dtype=int)
    for idx, point in enumerate(X):
        if y is None:
            node = min(roots, key=lambda root: np.sum((point - root.centroid) ** 2))
        else:
            label = y[idx]
            if label not in roots_by_label:
                msg = f"No fitted bisecting hierarchy is available for label {label!r}"
                raise ValueError(msg)
            node = roots_by_label[label]
        while node.children is not None:
            node = min(
                node.children,
                key=lambda child: np.sum((point - child.centroid) ** 2),
            )
        assert node.cluster_id is not None
        predictions[idx] = node.cluster_id
    return predictions
