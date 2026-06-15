import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bp_k_means.bisecting_k_means_optimized import (
    bisecting_kmeans_by_label_optimized_no_refine,
)
from bp_k_means.bp_kmeans import InitAlgorithm, InitStrategy, RankingStrategy, bp_kmeans
from bp_k_means.cop_k_means import cop_kmeans_by_class
from bp_k_means.hac import hac_ward_nnc_by_label
from bp_k_means.main import overall_wcss
from bp_k_means.precomputed_bisecting_k_means_optimized import (
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
    *,
    output_dir: Path = OUTPUT_DIR,
    benchmark_type: str = "regular",
    extra_metadata: dict | None = None,
    run_name: str | None = None,
) -> None:
    safe_alg = re.sub(r"[^\w]", "_", alg_name).strip("_")
    run_dir = output_dir / dataset_name / safe_alg / (
        run_name or f"k{k}_ninit{n_init}"
    )
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
        "benchmark_type": benchmark_type,
        "k": k,
        "k_multiplier": k_mult,
        "n_init": n_init,
        "duration_seconds": duration,
        "wcss_total": wcss,
        "n_clusters": n_clusters,
        "n_labels": int(len(np.unique(y))),
        "wcss_per_cluster": _wcss_stats(wcss_cluster_arr),
        "wcss_per_label": _wcss_stats(wcss_label_arr),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
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


K_MULTIPLIERS = [1.5, 2, 4]
N_INITS = [1, 2, 4, 8, 16, 32]
BENCHMARK_DATASET_EXCLUDE_PATTERNS = ("com_madrid", "castile_and_leon")


def _is_basic_benchmark_dataset(path: Path) -> bool:
    stem = path.stem.lower()
    return not any(pattern in stem for pattern in BENCHMARK_DATASET_EXCLUDE_PATTERNS)


def _build_algorithms() -> list:
    return [
        *[
            (
                f"BP-KMeans ({ranking.name}, {init.name}, {init_algo.name})",
                lambda X, y, k, seed, n_init=n_in, r=ranking, i=init, ia=init_algo: bp_kmeans(
                    X,
                    y,
                    k,
                    seed=seed,
                    n_init=n_init,
                    ranking_strategy=r,
                    init_strategy=i,
                    init_algorithm=ia,
                    subsample_size=10,
                ),
                n_in,
            )
            for init_algo in [InitAlgorithm.KMEANS_PLUS_PLUS, InitAlgorithm.RANDOM_SAMPLING]
            for ranking in RankingStrategy
            for init in InitStrategy
            for n_in in N_INITS
        ],
        *[
            (
                "Bisecting KMeans",
                lambda X, y, k, seed, n_init=n_in: bisecting_kmeans_by_label_optimized_no_refine(
                    X, y, k, seed=seed, n_init=n_init
                ),
                n_in,
            )
            for n_in in N_INITS
        ],
        *[
            (
                "Bisecting KMeans (R_RL)",
                lambda X,
                y,
                k,
                seed,
                n_init=n_in: precomputed_bisecting_kmeans_by_label_optimized_no_refine(
                    X, y, k, seed=seed, n_init=n_init
                ),
                n_in,
            )
            for n_in in N_INITS
        ],
        (
            "HAC Ward (NNC)",
            lambda X, y, k, seed: hac_ward_nnc_by_label(X, y, target_k=k),
            1,
        ),
    ]


def run_benchmark():
    datasets_dir = Path("data/datasets")
    dataset_files = list(datasets_dir.glob("*nodes.parquet"))
    dataset_files = [f for f in dataset_files if _is_basic_benchmark_dataset(f)]

    if not dataset_files:
        logger.error(f"No parquet files found in {datasets_dir}")
        return

    algorithms = _build_algorithms()

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

        for k_mult in K_MULTIPLIERS:
            target_k = int(n_labels * k_mult)

            if target_k > n_instances:
                logger.warning(
                    f"  Skipping k={target_k} (x{k_mult}) because it exceeds n_instances ({n_instances})"
                )
                continue

            for alg_name, alg_func, n_init in algorithms:
                logger.info(f"  Running {alg_name} | k={target_k} (x{k_mult}) | n_init={n_init}")

                start_time = time.time()

                labels = alg_func(X, y, target_k, seed=42)
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


def run_hac_strength_benchmark(cluster_multiplier: float = 1.5) -> None:
    """Run the HAC-strength benchmark.

    The target number of clusters is computed as
    ``cluster_multiplier * n_instances`` for each dataset. Since a clustering
    cannot contain more clusters than input points, values above ``1.0`` are
    capped at ``n_instances`` and recorded in metadata.
    """
    if cluster_multiplier <= 0:
        raise ValueError("cluster_multiplier must be positive")

    datasets_dir = Path("data/datasets")
    dataset_files = list(datasets_dir.glob("*nodes.parquet"))
    dataset_files = [f for f in dataset_files if _is_basic_benchmark_dataset(f)]

    if not dataset_files:
        logger.error(f"No parquet files found in {datasets_dir}")
        return

    algorithms = _build_algorithms()
    output_dir = OUTPUT_DIR / "hac_strength"
    safe_multiplier = str(cluster_multiplier).replace(".", "_")

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
            continue

        requested_target_k = int(n_instances * cluster_multiplier)
        target_k = min(requested_target_k, n_instances)

        logger.info(
            "  Instances: %s, Labels: %s, requested k=%s (x%s nodes), effective k=%s",
            n_instances,
            n_labels,
            requested_target_k,
            cluster_multiplier,
            target_k,
        )

        if requested_target_k != target_k:
            logger.warning(
                "  Requested k=%s exceeds n_instances=%s; using k=%s.",
                requested_target_k,
                n_instances,
                target_k,
            )

        if target_k < n_labels:
            logger.warning(
                "  Skipping k=%s because it is below the number of labels (%s).",
                target_k,
                n_labels,
            )
            continue

        for alg_name, alg_func, n_init in algorithms:
            safe_alg = re.sub(r"[^\w]", "_", alg_name).strip("_")
            run_name = f"nodesx{safe_multiplier}_k{target_k}_ninit{n_init}"
            meta_path = output_dir / dataset_path.stem / safe_alg / run_name / "metadata.json"
            if meta_path.exists():
                logger.info(
                    "  Skipping %s | k=%s (x%s nodes) | n_init=%s; output already exists.",
                    alg_name,
                    target_k,
                    cluster_multiplier,
                    n_init,
                )
                continue

            logger.info(
                "  Running %s | k=%s (x%s nodes) | n_init=%s",
                alg_name,
                target_k,
                cluster_multiplier,
                n_init,
            )

            start_time = time.time()
            labels = alg_func(X, y, target_k, seed=42)
            duration = time.time() - start_time

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
                    k_mult=cluster_multiplier,
                    n_init=n_init,
                    X=X,
                    y=y,
                    labels=labels,
                    duration=duration,
                    wcss=wcss,
                    output_dir=output_dir,
                    benchmark_type="hac_strength",
                    run_name=run_name,
                    extra_metadata={
                        "cluster_multiplier_basis": "nodes",
                        "requested_cluster_multiplier": cluster_multiplier,
                        "requested_n_clusters": requested_target_k,
                        "effective_n_clusters": target_k,
                        "target_k_was_capped": requested_target_k != target_k,
                    },
                )

            logger.info(
                f"    -> Time: {duration:.4f}s | WCSS: {wcss:.4f} | Clusters: {n_clusters}"
            )


def _compute_distance_metrics(X: np.ndarray, labels: np.ndarray, y: np.ndarray) -> dict:
    """Compute avg, max, and mean-max-per-label distance to cluster centroid."""
    all_dists = np.empty(len(X))
    max_per_label: dict = {}

    for c in np.unique(labels):
        mask = labels == c
        pts = X[mask]
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        all_dists[mask] = dists

    # mean-max distance per original label
    for lbl in np.unique(y):
        lbl_mask = y == lbl
        max_per_label[lbl] = float(all_dists[lbl_mask].max())

    return {
        "avg_dist_to_centroid_m": float(all_dists.mean()),
        "max_dist_to_centroid_m": float(all_dists.max()),
        "mean_max_dist_per_label_m": float(np.mean(list(max_per_label.values()))),
    }


def benchmark_com_madrid_avg_distance_to_centroid() -> None:
    """Benchmark: average distance to centroid at k=10000 for all algorithms (Community of Madrid)."""
    dataset_path = Path("data/datasets") / "com_madrid_osm_drive_nodes_split_split.parquet"
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    logger.info(f"Loading {dataset_path.name}")
    df = pd.read_parquet(dataset_path)
    X = df[["x_utm", "y_utm"]].values
    y = df["CUSEC"].values
    target_k = 10000
    logger.info(f"  Instances: {len(X)} | k={target_k}")

    algorithms = _build_algorithms()

    for alg_name, alg_func, n_init in algorithms:
        logger.info(f"  Running {alg_name} | k={target_k} | n_init={n_init}")
        start_time = time.time()
        labels = alg_func(X, y, target_k, seed=42)
        duration = time.time() - start_time

        if labels is None:
            logger.error(f"    {alg_name} failed to produce labels.")
            continue

        wcss = overall_wcss(X, labels)
        _save_run_outputs(
            dataset_name=dataset_path.stem,
            alg_name=alg_name,
            k=target_k,
            k_mult=1.0,
            n_init=n_init,
            X=X,
            y=y,
            labels=labels,
            duration=duration,
            wcss=wcss,
        )

        dist_metrics = _compute_distance_metrics(X, labels, y)

        safe_alg = re.sub(r"[^\w]", "_", alg_name).strip("_")
        meta_path = (
            OUTPUT_DIR
            / dataset_path.stem
            / safe_alg
            / f"k{target_k}_ninit{n_init}"
            / "metadata.json"
        )
        with open(meta_path) as f:
            meta = json.load(f)
        meta.update(dist_metrics)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            f"    -> avg dist: {dist_metrics['avg_dist_to_centroid_m']:.2f} m | "
            f"max dist: {dist_metrics['max_dist_to_centroid_m']:.2f} m | "
            f"mean-max/label: {dist_metrics['mean_max_dist_per_label_m']:.2f} m | "
            f"clusters: {len(np.unique(labels))} | time: {duration:.4f}s"
        )


def benchmark_castile_leon_max_response_time() -> None:
    """Benchmark: maximum response time at k=200 using province labels (Castile and León)."""
    dataset_path = Path("data/datasets") / "castile_and_leon_osm_drive_nodes.parquet"
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    logger.info(f"Loading {dataset_path.name}")
    df = pd.read_parquet(dataset_path)
    X = df[["x_utm", "y_utm"]].values
    y = df["CPRO"].values  # province labels
    target_k = 200
    logger.info(f"  Instances: {len(X)} | unique provinces: {len(np.unique(y))} | k={target_k}")

    if target_k > len(X):
        logger.error(f"k={target_k} exceeds n_instances={len(X)}. Aborting.")
        return

    algorithms = _build_algorithms()

    for alg_name, alg_func, n_init in algorithms:
        # only benchmark n_init=8 for max response time
        logger.info(f"  Running {alg_name} | k={target_k} | n_init={n_init}")
        t0 = time.time()
        labels = alg_func(X, y, target_k, seed=42)
        duration = time.time() - t0

        if labels is None:
            logger.error(f"    {alg_name} failed to produce labels.")
            continue

        wcss = overall_wcss(X, labels)
        _save_run_outputs(
            dataset_name=dataset_path.stem,
            alg_name=alg_name,
            k=target_k,
            k_mult=1.0,
            n_init=n_init,
            X=X,
            y=y,
            labels=labels,
            duration=duration,
            wcss=wcss,
        )

        dist_metrics = _compute_distance_metrics(X, labels, y)

        safe_alg = re.sub(r"[^\w]", "_", alg_name).strip("_")
        meta_path = (
            OUTPUT_DIR
            / dataset_path.stem
            / safe_alg
            / f"k{target_k}_ninit{n_init}"
            / "metadata.json"
        )
        with open(meta_path) as f:
            meta = json.load(f)
        meta["response_time_s"] = duration
        meta.update(dist_metrics)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            f"    -> time: {duration:.4f}s | "
            f"avg dist: {dist_metrics['avg_dist_to_centroid_m']:.2f} m | "
            f"max dist: {dist_metrics['max_dist_to_centroid_m']:.2f} m | "
            f"mean-max/label: {dist_metrics['mean_max_dist_per_label_m']:.2f} m"
        )


if __name__ == "__main__":
    # run_benchmark()

    run_hac_strength_benchmark(cluster_multiplier=0.75)

    # benchmark_castile_leon_max_response_time()
    # benchmark_com_madrid_avg_distance_to_centroid()
