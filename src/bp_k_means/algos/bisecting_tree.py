"""Prediction state shared by non-refined bisecting algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _BisectingTreeNode:
    """
    One fitted split in a non-refined bisecting hierarchy.

    Parameters
    ----------
    centroid : NDArray
        Centroid used to route a point at this node.

    Attributes
    ----------
    centroid : NDArray
        Centroid used to route a point at this node.
    children : tuple[_BisectingTreeNode, _BisectingTreeNode] | None
        Left and right child nodes, or ``None`` for a leaf.
    cluster_id : int | None
        Global cluster identifier assigned to a leaf after fitting.
    """

    centroid: NDArray
    children: tuple[_BisectingTreeNode, _BisectingTreeNode] | None
    cluster_id: int | None

    def __init__(self, centroid: NDArray) -> None:
        self.centroid = centroid
        self.children: tuple[_BisectingTreeNode, _BisectingTreeNode] | None = None
        self.cluster_id: int | None = None


def _assign_from_hierarchy(
    X: NDArray,
    y: NDArray,
    roots_by_label: dict[object, _BisectingTreeNode],
) -> NDArray:
    """
    Assign samples by following fitted centroid splits from root to leaf.

    Parameters
    ----------
    X : NDArray
        Feature matrix to predict.
    y : NDArray
        Source label for each input row.
    roots_by_label : dict[object, _BisectingTreeNode]
        Fitted hierarchy root indexed by source label.

    Returns
    -------
    NDArray
        Leaf cluster identifier for each input row.

    Raises
    ------
    ValueError
        If an input source label has no fitted hierarchy.
    """
    predictions = np.empty(X.shape[0], dtype=int)
    for idx, point in enumerate(X):
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
