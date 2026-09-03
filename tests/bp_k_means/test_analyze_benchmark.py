"""Functional tests for benchmark result analysis."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bp_k_means.tools.analyze_benchmark import compute_relative_metrics, load_all_metadata
from bp_k_means.tools.benchmark_analysis.data import select_bp_vs_bisecting_kmeans
from bp_k_means.tools.benchmark_analysis.plotting import add_scatter_legends


def test_regular_metadata_excludes_special_and_hac_benchmarks(tmp_path: Path) -> None:
    """Keep dedicated experiments out of the paper's regular aggregate."""
    common = {
        "algorithm": "example",
        "n_init": 1,
        "k_multiplier": 2,
        "k": 4,
        "n_clusters": 4,
        "n_labels": 2,
        "wcss_total": 10,
        "duration_seconds": 1,
    }
    rows = [
        {**common, "dataset": "regular_nodes", "benchmark_type": "regular"},
        {
            **common,
            "dataset": "com_madrid_osm_drive_nodes_split_split",
            "benchmark_type": "regular",
        },
        {**common, "dataset": "regular_nodes", "benchmark_type": "hac_strength"},
    ]
    for index, row in enumerate(rows):
        run_dir = tmp_path / str(index)
        run_dir.mkdir()
        (run_dir / "metadata.json").write_text(json.dumps(row), encoding="utf-8")

    metadata = load_all_metadata(tmp_path)

    assert metadata["dataset"].tolist() == ["regular_nodes"]


def test_relative_metrics_are_normalized_per_benchmark_case() -> None:
    """Use a separate quality and runtime baseline for every dataset/multiplier pair."""
    raw = pd.DataFrame(
        {
            "dataset": ["a", "a", "b", "b"],
            "k_multiplier": [2.0, 2.0, 2.0, 2.0],
            "wcss": [10.0, 15.0, 100.0, 120.0],
            "time": [4.0, 2.0, 5.0, 10.0],
        }
    )

    relative = compute_relative_metrics(raw)

    assert relative["best_wcss"].tolist() == [10.0, 10.0, 100.0, 100.0]
    assert relative["best_time"].tolist() == [2.0, 2.0, 5.0, 5.0]
    assert relative["relative_wcss"].tolist() == [1.0, 1.5, 1.0, 1.2]
    assert relative["relative_time"].tolist() == [2.0, 1.0, 1.0, 2.0]


def test_select_bp_vs_bisecting_kmeans_keeps_only_requested_algorithms() -> None:
    """Restrict comparison rows to BP-KMeans++ and standard Bisecting KMeans."""
    metadata = pd.DataFrame(
        {
            "algorithm": [
                "Bisecting KMeans",
                "Bisecting KMeans (M_RL)",
                "BP-KMeans (M_L, I_LRI, KMEANS_PLUS_PLUS)",
                "BP-KMeans (M_L, I_LRI, RANDOM_SAMPLING)",
                "HAC Ward (NNC)",
            ],
            "n_init": [1, 1, 1, 1, 1],
        }
    )

    selected = select_bp_vs_bisecting_kmeans(metadata)

    assert selected["algorithm"].tolist() == [
        "Bisecting KMeans",
        "BP-KMeans (M_L, I_LRI, KMEANS_PLUS_PLUS)",
    ]


def test_scatter_legend_shows_single_bisecting_baseline() -> None:
    """Keep the sole baseline visible in the comparison graph legend."""
    figure, axis = plt.subplots()
    add_scatter_legends(
        axis,
        {"baseline_color_entries": [("Bisecting KMeans", (0.1, 0.2, 0.3))]},
    )

    legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]

    assert "Bisecting KMeans" in legend_labels
    plt.close(figure)
