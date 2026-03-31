import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bp_k_means.bisecting_k_means_optimized import (
    bisecting_kmeans_by_label_optimized,
    bisecting_kmeans_by_label_optimized_no_refine,
)
from bp_k_means.bp_k_means_optimized import bp_kmeans_optimized, precomputed_bp_kmeans_optimized
from bp_k_means.cop_k_means import cop_kmeans_by_class
from bp_k_means.hac import hac_ward_nnc_by_label
from bp_k_means.main import overall_wcss
from bp_k_means.precomputed_bisecting_k_means_optimized import (
    precomputed_bisecting_kmeans_by_label_optimized,
    precomputed_bisecting_kmeans_by_label_optimized_no_refine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")


def _compute_centroids(X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    unique_labels = np.unique(labels)
    rows = []
    for c in unique_labels:
        pts = X[labels == c]
        if len(pts) > 0:
            cx, cy = pts.mean(axis=0)
            rows.append({"cluster_id": int(c), "x_utm": cx, "y_utm": cy})
    return pd.DataFrame(rows)


def _compute_wcss_per_cluster_array(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return an array of WCSS values, one per unique cluster."""
    unique_clusters = np.unique(labels)
    wcss_values = np.empty(len(unique_clusters))
    for i, c in enumerate(unique_clusters):
        pts = X[labels == c]
        centroid = pts.mean(axis=0)
        diff = pts - centroid
        wcss_values[i] = np.sum(diff * diff)
    return wcss_values


def _compute_wcss_per_label_array(X: np.ndarray, y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return an array of WCSS values, one per unique original label in y.

    Each entry is the sum of squared distances of all points belonging to that
    label from their respective (globally computed) cluster centroids.
    """
    unique_clusters = np.unique(labels)
    centroids = {int(c): X[labels == c].mean(axis=0) for c in unique_clusters}

    unique_y = np.unique(y)
    wcss_values = np.empty(len(unique_y))
    for i, lbl in enumerate(unique_y):
        mask = y == lbl
        pts = X[mask]
        pts_clusters = labels[mask]
        wcss = 0.0
        for c in np.unique(pts_clusters):
            cluster_pts = pts[pts_clusters == c]
            diff = cluster_pts - centroids[int(c)]
            wcss += np.sum(diff * diff)
        wcss_values[i] = wcss
    return wcss_values


def _wcss_stats(values: np.ndarray) -> dict:
    """Compute descriptive statistics for an array of WCSS values."""
    return {
        "avg": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q1": float(np.percentile(values, 25)),
        "q2": float(np.percentile(values, 50)),
        "q3": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def _save_run_outputs(
    dataset_name: str,
    alg_name: str,
    k: int,
    k_mult: float,
    n_init: int,
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    duration: float,
    wcss: float,
) -> None:
    safe_alg = re.sub(r"[^\w]", "_", alg_name).strip("_")
    run_dir = OUTPUT_DIR / dataset_name / safe_alg / f"k{k}_ninit{n_init}"
    run_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = int(len(np.unique(labels)))

    # centroids.csv
    _compute_centroids(X, labels).to_csv(run_dir / "centroids.csv", index=False)

    # per-cluster and per-label WCSS distributions
    wcss_cluster_arr = _compute_wcss_per_cluster_array(X, labels)
    wcss_label_arr = _compute_wcss_per_label_array(X, y, labels)

    # metadata.json
    metadata = {
        "dataset": dataset_name,
        "algorithm": alg_name,
        "k": k,
        "k_multiplier": k_mult,
        "n_init": n_init,
        "duration_seconds": duration,
        "wcss_total": wcss,
        "n_clusters": n_clusters,
        "wcss_per_cluster": _wcss_stats(wcss_cluster_arr),
        "wcss_per_label": _wcss_stats(wcss_label_arr),
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # instances.parquet  — coordinates + original class label + cluster assignment
    instances_df = pd.DataFrame(X, columns=["x_utm", "y_utm"])
    instances_df["class_label"] = y
    instances_df["cluster"] = labels
    instances_df.to_parquet(run_dir / "instances.parquet", index=False)


def run_cop_kmeans(X, y, k, seed, n_init, init_ensure_class):
    rng = np.random.default_rng(seed)
    best_wcss = float("inf")
    best_labels = None

    for i in range(n_init):
        current_seed = rng.integers(2**32)
        labels, _ = cop_kmeans_by_class(
            X, y, k, seed=current_seed, init_ensure_class=init_ensure_class
        )

        if labels is None:
            continue

        wcss = overall_wcss(X, labels)
        if wcss < best_wcss:
            best_wcss = wcss
            best_labels = labels

    return best_labels


def run_benchmark():
    datasets_dir = Path("data/datasets")
    dataset_files = list(datasets_dir.glob("*.parquet"))

    # Exclude if madrid dataset is present
    # dataset_files = [f for f in dataset_files if "madrid" not in f.name.lower()]

    if not dataset_files:
        logger.error(f"No parquet files found in {datasets_dir}")
        return

    n_inits = [1, 2, 4, 8, 16, 32]
    k_multipliers = [1.5, 2, 4, 8, 16, 32]

    algorithms = [
        (
            "BP-KMeans Optimized (WCSS/Cluster)",
            lambda X, y, k, seed, n_init: bp_kmeans_optimized(
                X, y, k, seed=seed, n_init=n_init, use_wcss_per_cluster=True
            ),
        ),
        (
            "BP-KMeans Optimized (Total WCSS)",
            lambda X, y, k, seed, n_init: bp_kmeans_optimized(
                X, y, k, seed=seed, n_init=n_init, use_wcss_per_cluster=False
            ),
        ),
        (
            "Precomputed BP-KMeans Optimized",
            lambda X, y, k, seed, n_init: precomputed_bp_kmeans_optimized(
                X, y, k, seed=seed, n_init=n_init
            ),
        ),
        (
            "Bisecting Optimized (Refine, WCSS/Cluster)",
            lambda X, y, k, seed, n_init: bisecting_kmeans_by_label_optimized(
                X, y, k, seed=seed, n_init=n_init, use_wcss_per_cluster=True
            ),
        ),
        (
            "Bisecting Optimized (Refine, Total WCSS)",
            lambda X, y, k, seed, n_init: bisecting_kmeans_by_label_optimized(
                X, y, k, seed=seed, n_init=n_init, use_wcss_per_cluster=False
            ),
        ),
        (
            "Bisecting Optimized (No Refine)",
            lambda X, y, k, seed, n_init: bisecting_kmeans_by_label_optimized_no_refine(
                X, y, k, seed=seed, n_init=n_init
            ),
        ),
        (
            "Precomputed Bisecting (Refine)",
            lambda X, y, k, seed, n_init: precomputed_bisecting_kmeans_by_label_optimized(
                X, y, k, seed=seed, n_init=n_init
            ),
        ),
        (
            "Precomputed Bisecting (No Refine)",
            lambda X, y, k, seed, n_init: precomputed_bisecting_kmeans_by_label_optimized_no_refine(
                X, y, k, seed=seed, n_init=n_init
            ),
        ),
        (
            "HAC Ward (NNC)",
            lambda X, y, k, seed, n_init: hac_ward_nnc_by_label(X, y, target_k=k),
        ),
    ]

    for dataset_path in dataset_files:
        logger.info(f"Loading dataset: {dataset_path.name}")
        try:
            df = pd.read_parquet(dataset_path)
            X = df[["x_utm", "y_utm"]].values
            y = df["CUSEC"].values

            n_instances = len(X)
            n_labels = len(np.unique(y))
        except Exception as e:
            logger.error(f"Failed to process dataset {dataset_path.name}: {e}")

        logger.info(f"  Instances: {n_instances}, Labels: {n_labels}")

        for k_mult in k_multipliers:
            target_k = int(n_labels * k_mult)

            if target_k > n_instances:
                logger.warning(
                    f"  Skipping k={target_k} (x{k_mult}) because it exceeds n_instances ({n_instances})"
                )
                continue

            for n_init in n_inits:
                for alg_name, alg_func in algorithms:
                    # Skip HAC for n_init > 1 as it is deterministic
                    if "HAC" in alg_name and n_init > 1:
                        continue

                    logger.info(
                        f"  Running {alg_name} | k={target_k} (x{k_mult}) | n_init={n_init}"
                    )

                    start_time = time.time()

                    labels = alg_func(X, y, target_k, seed=42, n_init=n_init)
                    end_time = time.time()
                    duration = end_time - start_time

                    if labels is None:
                        logger.error(f"    {alg_name} failed to produce labels.")
                        wcss = np.nan
                        n_clusters = 0
                    else:
                        wcss = overall_wcss(X, labels)
                        n_clusters = len(np.unique(labels))
                        _save_run_outputs(
                            dataset_name=dataset_path.stem,
                            alg_name=alg_name,
                            k=target_k,
                            k_mult=k_mult,
                            n_init=n_init,
                            X=X,
                            y=y,
                            labels=labels,
                            duration=duration,
                            wcss=wcss,
                        )

                    logger.info(
                        f"    -> Time: {duration:.4f}s | WCSS: {wcss:.4f} | Clusters: {n_clusters}"
                    )


if __name__ == "__main__":
    run_benchmark()
