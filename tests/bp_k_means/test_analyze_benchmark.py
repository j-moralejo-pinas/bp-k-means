"""Functional tests for benchmark result analysis."""

import json
from pathlib import Path

import pandas as pd

from bp_k_means.tools.analyze_benchmark import compute_relative_metrics, load_all_metadata


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
