"""Small deterministic datasets and assertions shared by algorithm tests."""

import numpy as np
import pytest


@pytest.fixture
def two_label_points() -> tuple[np.ndarray, np.ndarray]:
    """Two labels, each containing two tight one-dimensional pairs."""
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array(["left", "left", "right", "right"])
    return X, y


@pytest.fixture
def four_point_groups() -> tuple[np.ndarray, np.ndarray]:
    """Two labels with enough points to exercise several split steps."""
    X = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
         [10.0, 0.0], [11.0, 0.0], [12.0, 0.0], [13.0, 0.0]]
    )
    y = np.array(["left"] * 4 + ["right"] * 4)
    return X, y


@pytest.fixture
def three_two_point_groups() -> tuple[np.ndarray, np.ndarray]:
    """Three labels with two points each."""
    X = np.array([[0.0], [1.0], [10.0], [11.0], [20.0], [21.0]])
    y = np.array(["a", "a", "b", "b", "c", "c"])
    return X, y


def canonical_partition(labels: np.ndarray) -> np.ndarray:
    """Normalize cluster ids by first appearance for partition comparisons."""
    ids: dict[int, int] = {}
    normalized = []
    for label in labels:
        raw_label = int(label)
        if raw_label not in ids:
            ids[raw_label] = len(ids)
        normalized.append(ids[raw_label])
    return np.asarray(normalized)


def assert_label_pure(labels: np.ndarray, y: np.ndarray) -> None:
    """Assert that no cluster contains points from different source labels."""
    for cluster in np.unique(labels):
        assert len(np.unique(y[labels == cluster])) == 1


def direct_wcss(X: np.ndarray, labels: np.ndarray) -> float:
    """Calculate WCSS directly from each cluster's arithmetic mean."""
    return float(
        sum(
            np.sum((X[labels == c] - X[labels == c].mean(axis=0)) ** 2)
            for c in np.unique(labels)
        )
    )
