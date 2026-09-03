"""Data loading and aggregation for benchmark analysis."""

import json
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bp_k_means.utils.logging import logger

HAC_STRENGTH_BENCHMARK_TYPE = "hac_strength"
REGULAR_DATASET_EXCLUDE_PATTERNS = ("com_madrid", "castile_and_leon")

LABEL_COUNT_LIMIT = 1_000
LARGE_DATASET_LIMIT = 5_000
SMALL_DATASET_LIMIT = 1_000
SIZE_BIN_LABELS = [
    ">1k labels",
    ">5k nodes <1k labels",
    "1k-5k nodes",
    "<1k nodes",
]

BP_ALGORITHM_PATTERN = re.compile(r"BP-KMeans \((\w+),\s*(\w+),\s*(\w+)\)")


def assign_size_bin(n_instances: float, n_labels: int) -> str:
    """Assign a dataset to a size bin based on its node and label counts."""
    if n_labels > LABEL_COUNT_LIMIT:
        return ">1k labels"
    if n_instances > LARGE_DATASET_LIMIT:
        return ">5k nodes <1k labels"
    if n_instances > SMALL_DATASET_LIMIT:
        return "1k-5k nodes"
    return "<1k nodes"


def read_metadata_files(output_dir: Path) -> list[dict[str, Any]]:
    """Read benchmark metadata in a stable order."""
    metadata = []
    for path in sorted(output_dir.rglob("metadata.json")):
        try:
            with path.open() as metadata_file:
                metadata.append(json.load(metadata_file))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping invalid metadata file %s: %s", path, exc)
    return metadata


def _base_metadata_row(meta: dict[str, Any]) -> dict[str, Any]:
    """Select and normalize fields shared by every benchmark analysis."""
    return {
        "dataset": meta["dataset"],
        "algorithm": meta["algorithm"],
        "n_init": int(meta["n_init"]),
        "k_multiplier": float(meta["k_multiplier"]),
        "k": int(meta["k"]),
        "n_clusters": int(meta["n_clusters"]),
        "n_labels": int(meta["n_labels"]) if "n_labels" in meta else np.nan,
        "wcss": float(meta["wcss_total"]),
        "time": float(meta["duration_seconds"]),
    }


def _is_regular_dataset(dataset: str) -> bool:
    normalized = dataset.lower()
    return not any(pattern in normalized for pattern in REGULAR_DATASET_EXCLUDE_PATTERNS)


def load_all_metadata(output_dir: Path) -> pd.DataFrame:
    """Load regular benchmark metadata files into one DataFrame."""
    rows = [
        _base_metadata_row(meta)
        for meta in read_metadata_files(output_dir)
        if meta.get("benchmark_type") != HAC_STRENGTH_BENCHMARK_TYPE
        and _is_regular_dataset(str(meta.get("dataset", "")))
    ]
    return pd.DataFrame(rows)


def load_hac_strength_metadata(output_dir: Path) -> pd.DataFrame:
    """Load only metadata rows produced by the HAC-strength benchmark."""
    rows = []
    for meta in read_metadata_files(output_dir):
        if meta.get("benchmark_type") != HAC_STRENGTH_BENCHMARK_TYPE:
            continue
        row = _base_metadata_row(meta)
        row.update(
            requested_cluster_multiplier=float(
                meta.get("requested_cluster_multiplier", meta["k_multiplier"])
            ),
            requested_n_clusters=int(meta.get("requested_n_clusters", meta["k"])),
            target_k_was_capped=bool(meta.get("target_k_was_capped", False)),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_dataset_sizes(data_dir: Path) -> dict[str, int]:
    """Read the number of rows from each benchmark dataset file."""
    sizes: dict[str, int] = {}
    for path in data_dir.glob("*nodes.parquet"):
        try:
            sizes[path.stem] = pq.read_metadata(path).num_rows
        except (OSError, ValueError) as exc:
            logger.debug("Could not read metadata from %s: %s", path, exc)
    return sizes


def parse_algorithm_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three BP-KMeans component columns."""
    result = df.copy()
    parsed = result["algorithm"].str.extract(BP_ALGORITHM_PATTERN)
    result["ranking_metric"] = parsed[0]
    result["init_strategy"] = parsed[1]
    result["init_algo"] = parsed[2]
    return result


def select_bp_vs_bisecting_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    """Keep standard Bisecting KMeans and BP-KMeans initialized with k-means++."""
    if df.empty:
        return df.copy()
    parsed = parse_algorithm_components(df)
    is_bisecting = parsed["algorithm"] == "Bisecting KMeans"
    is_bp_kmeans = parsed["algorithm"].str.startswith("BP-KMeans")
    is_kpp_bp = is_bp_kmeans & (parsed["init_algo"] == "KMEANS_PLUS_PLUS")
    return cast("pd.DataFrame", parsed[is_bisecting | is_kpp_bp].copy())


def add_dataset_context(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Add dataset sizes, inferred label counts, and publication size bins."""
    result = df.copy()
    result["n_instances"] = result["dataset"].map(load_dataset_sizes(data_dir).get)
    missing_datasets = sorted(result.loc[result["n_instances"].isna(), "dataset"].unique())
    if missing_datasets:
        logger.warning("Dataset sizes unavailable for: %s", ", ".join(missing_datasets))

    result["n_labels"] = result["n_labels"].fillna(result["n_clusters"] / result["k_multiplier"])
    result["size_bin"] = [
        assign_size_bin(n_instances, int(n_labels)) if pd.notna(n_instances) else pd.NA
        for n_instances, n_labels in zip(result["n_instances"], result["n_labels"], strict=False)
    ]
    return result


def compute_relative_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize WCSS and runtime against the best result for each benchmark case."""
    result = df.drop(
        columns=["best_wcss", "best_time", "relative_wcss", "relative_time"],
        errors="ignore",
    ).copy()
    cases = result.groupby(["dataset", "k_multiplier"])
    result["best_wcss"] = cases["wcss"].transform("min")
    result["best_time"] = cases["time"].transform("min")
    result["relative_wcss"] = result["wcss"] / result["best_wcss"]
    result["relative_time"] = result["time"] / result["best_time"]
    return result


def aggregate_relative_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Average both relative metrics for the requested grouping."""
    return (
        df.groupby(group_cols, observed=True)
        .agg(
            mean_relative_wcss=("relative_wcss", "mean"),
            mean_relative_time=("relative_time", "mean"),
        )
        .reset_index()
    )


def algorithm_label(row: pd.Series) -> str:
    """Build a display label from an algorithm row."""
    return f"{row['algorithm']} | n_init={row['n_init']}"
