import numpy as np
import pytest

from bp_k_means.benchmark import _compute_distance_metrics


def test_distance_metrics_use_closest_cluster_node_as_anchor():
    X = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [8.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 0])
    y = np.array(["label-a", "label-a", "label-a"])

    metrics = _compute_distance_metrics(X, labels, y)

    assert metrics["distance_anchor"] == "nearest_node_to_cluster_centroid"
    assert metrics["representative_node_count"] == 1
    assert metrics["avg_dist_to_representative_node_m"] == pytest.approx(8.0 / 3.0)
    assert metrics["max_dist_to_representative_node_m"] == pytest.approx(6.0)
    assert metrics["mean_max_dist_per_label_to_representative_node_m"] == pytest.approx(6.0)
    assert metrics["avg_dist_to_centroid_m"] == pytest.approx(
        metrics["avg_dist_to_representative_node_m"]
    )
