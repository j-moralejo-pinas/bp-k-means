"""Benchmark runners and output helpers for the clustering algorithms."""

import json
import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from bp_k_means.algos.base_algo import BaseAlgo
from bp_k_means.algos.bisecting_k_means_m_rl_optimized import (
    BisectingKMeansMRLNoRefine,
)
from bp_k_means.algos.bisecting_k_means_optimized import BisectingKMeansNoRefine
from bp_k_means.algos.bp_kmeans import BPKMeans, InitAlgorithm, InitStrategy, RankingMetric
from bp_k_means.algos.cop_k_means import COPKMeansCannotLink
from bp_k_means.algos.hac import HACWardNNC
from bp_k_means.utils.logging import logger
from bp_k_means.utils.metrics import overall_wcss

REGULAR_BENCHMARK_DATASET_EXCLUDE_PATTERNS = ("com_madrid", "castile_and_leon")


# Output helpers


def _algorithm_output_dir(
    output_dir: Path,
    dataset_name: str,
    alg_name: str,
    k: int,
    n_init: int,
    run_name: str | None = None,
) -> Path:
    """
    Return the output directory for one benchmark run.

    Parameters
    ----------
    output_dir : Path
        Benchmark output root.
    dataset_name : str
        Dataset identifier.
    alg_name : str
        Algorithm display name.
    k : int
        Target cluster count.
    n_init : int
        Number of initializations.
    run_name : str | None
        Explicit run directory name.

    Returns
    -------
    Path
        Deterministic output directory path.
    """
    safe_alg = re.sub(r"[^\w]", "_", alg_name).strip("_")
    return output_dir / dataset_name / safe_alg / (run_name or f"k{k}_ninit{n_init}")


def _compute_centroids(X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    Compute one coordinate centroid row per cluster.

    Parameters
    ----------
    X : np.ndarray
        Two-dimensional coordinates.
    labels : np.ndarray
        Cluster identifier for each row of ``X``.

    Returns
    -------
    pd.DataFrame
        Columns ``cluster_id``, ``x_utm``, and ``y_utm``.
    """
    unique_labels = np.unique(labels)
    rows = []
    for c in unique_labels:
        pts = X[labels == c]
        if len(pts) > 0:
            cx, cy = pts.mean(axis=0)
            rows.append({"cluster_id": int(c), "x_utm": cx, "y_utm": cy})
    return pd.DataFrame(rows)


def _compute_wcss_per_cluster_array(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Return an array of WCSS values, one per unique cluster.

    Parameters
    ----------
    X : np.ndarray
        Input coordinates or features.
    labels : np.ndarray
        Cluster identifier for each row.

    Returns
    -------
    np.ndarray
        WCSS values ordered by sorted unique cluster identifier.
    """
    unique_clusters = np.unique(labels)
    wcss_values = np.empty(len(unique_clusters))
    for i, c in enumerate(unique_clusters):
        pts = X[labels == c]
        centroid = pts.mean(axis=0)
        diff = pts - centroid
        wcss_values[i] = np.sum(diff * diff)
    return wcss_values


def _compute_wcss_per_label_array(X: np.ndarray, y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Return an array of WCSS values, one per unique original label in y.

    Each entry is the sum of squared distances of all points belonging to that label from their
    respective (globally computed) cluster centroids.

    Parameters
    ----------
    X : np.ndarray
        Input coordinates or features.
    y : np.ndarray
        Original source labels.
    labels : np.ndarray
        Global cluster assignments.

    Returns
    -------
    np.ndarray
        WCSS values ordered by sorted unique source label.
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
    """
    Compute descriptive statistics for an array of WCSS values.

    Parameters
    ----------
    values : np.ndarray
        WCSS observations.

    Returns
    -------
    dict
        Mean, standard deviation, quartiles, minimum, and maximum.
    """
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
    output_dir: Path,
    benchmark_type: str = "regular",
    seed: int,
    extra_metadata: dict | None = None,
    run_name: str | None = None,
) -> None:
    """
    Persist labels, centroids, distributions, and metadata for one run.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier.
    alg_name : str
        Algorithm identifier.
    k : int
        Effective target cluster count.
    k_mult : float
        Requested cluster multiplier recorded in metadata.
    n_init : int
        Number of initializations.
    X : np.ndarray
        Input feature matrix.
    y : np.ndarray
        Source labels.
    labels : np.ndarray
        Fitted cluster labels.
    duration : float
        Elapsed fitting time in seconds.
    wcss : float
        Total within-cluster sum of squares.
    output_dir : Path
        Benchmark output root.
    benchmark_type : str
        Benchmark category recorded in metadata.
    seed : int
        Random seed recorded in metadata.
    extra_metadata : dict | None
        Additional metadata fields.
    run_name : str | None
        Explicit run directory suffix.
    """
    run_dir = _algorithm_output_dir(output_dir, dataset_name, alg_name, k, n_init, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = len(np.unique(labels))

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
        "n_labels": len(np.unique(y)),
        "wcss_per_cluster": _wcss_stats(wcss_cluster_arr),
        "wcss_per_label": _wcss_stats(wcss_label_arr),
    }
    metadata["seed"] = seed
    if extra_metadata:
        metadata.update(extra_metadata)
    with (run_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    # instances.parquet  — coordinates + original label + cluster assignment
    instances_df = pd.DataFrame(X, columns=pd.Index(["x_utm", "y_utm"]))
    instances_df["label"] = y
    instances_df["cluster"] = labels
    instances_df.to_parquet(run_dir / "instances.parquet", index=False)


def _is_regular_benchmark_dataset(path: Path) -> bool:
    """
    Return whether a dataset path belongs to regular benchmarks.

    Parameters
    ----------
    path : Path
        Dataset path to classify.

    Returns
    -------
    bool
        ``True`` when the filename does not contain an excluded special-dataset
        pattern.
    """
    stem = path.stem.lower()
    return not any(pattern in stem for pattern in REGULAR_BENCHMARK_DATASET_EXCLUDE_PATTERNS)


def _load_dataset(dataset_path: Path, label_column: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load coordinates and labels from a benchmark parquet file.

    Parameters
    ----------
    dataset_path : Path
        Parquet dataset path.
    label_column : str
        Column containing source labels.

    Returns
    -------
    X : np.ndarray
        Coordinate matrix.
    y : np.ndarray
        Source-label array.
    """
    df = pd.read_parquet(dataset_path)
    return df[["x_utm", "y_utm"]].to_numpy(), df[label_column].to_numpy()


def _iter_datasets(
    datasets_dir: Path,
    label_column: str,
    *,
    regular_only: bool = False,
    dataset_filename: str | None = None,
) -> Iterator[tuple[Path, np.ndarray, np.ndarray]]:
    """
    Yield benchmark datasets, skipping files that cannot be loaded.

    Parameters
    ----------
    datasets_dir : Path
        Directory searched for parquet datasets.
    label_column : str
        Source-label column to load.
    regular_only : bool
        Whether special dataset names should be excluded.
    dataset_filename : str | None
        Explicit filename to load instead of globbing.

    Yields
    ------
    (Path, np.ndarray, np.ndarray)
        Dataset path, feature matrix, and source-label array.
    """
    dataset_files = (
        [datasets_dir / dataset_filename]
        if dataset_filename is not None
        else list(datasets_dir.glob("*nodes.parquet"))
    )
    if regular_only and dataset_filename is None:
        dataset_files = [f for f in dataset_files if _is_regular_benchmark_dataset(f)]

    if not dataset_files:
        logger.error("No parquet files found in %s", datasets_dir)
        return

    if dataset_filename is not None and not dataset_files[0].exists():
        logger.error("Dataset not found: %s", dataset_files[0])
        return

    for dataset_path in dataset_files:
        logger.info("Loading dataset: %s", dataset_path.name)
        try:
            X, y = _load_dataset(dataset_path, label_column)
        except Exception:  # noqa: BLE001 - keep processing the remaining datasets
            logger.exception("Failed to process dataset %s", dataset_path.name)
            continue
        yield dataset_path, X, y


def _run_algorithm(
    algo: BaseAlgo,
    X: np.ndarray,
    y: np.ndarray,
    target_k: int,
) -> tuple[np.ndarray | None, float]:
    """
    Run one algorithm and return its labels and elapsed time.

    Parameters
    ----------
    algo : BaseAlgo
        Algorithm instance to fit.
    X : np.ndarray
        Input feature matrix.
    y : np.ndarray
        Source labels.
    target_k : int
        Requested cluster count.

    Returns
    -------
    labels : np.ndarray | None
        Fitted labels, or ``None`` when fitting raises ``RuntimeError``.
    duration : float
        Elapsed wall-clock time in seconds.
    """
    start_time = time.perf_counter()
    try:
        labels = algo.fit_predict(X, y, target_k)
    except RuntimeError:
        labels = None
    return labels, time.perf_counter() - start_time


def _update_metadata(metadata_path: Path, updates: dict) -> None:
    """
    Update a saved run's metadata in place.

    Parameters
    ----------
    metadata_path : Path
        JSON metadata file to update.
    updates : dict
        Fields merged into the existing record.
    """
    with metadata_path.open() as metadata_file:
        metadata = json.load(metadata_file)
    metadata.update(updates)
    with metadata_path.open("w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def _run_algorithms(
    *,
    algorithms: list[tuple[str, BaseAlgo]],
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    target_k: int,
    k_mult: float,
    output_dir: Path,
    seed: int,
    skip_existing: bool,
    benchmark_type: str = "regular",
    run_name_prefix: str | None = None,
    extra_metadata: dict | None = None,
    postprocess: Callable[[Path, np.ndarray, float], None] | None = None,
) -> None:
    """
    Run and persist all algorithms for one benchmark case.

    Parameters
    ----------
    algorithms : list[tuple[str, BaseAlgo]]
        Algorithm display names paired with instances.
    dataset_name : str
        Dataset identifier.
    X : np.ndarray
        Input features.
    y : np.ndarray
        Source labels.
    target_k : int
        Requested cluster count.
    k_mult : float
        Cluster multiplier recorded in metadata.
    output_dir : Path
        Output root.
    seed : int
        Random seed recorded in metadata.
    skip_existing : bool
        Whether existing metadata files are skipped.
    benchmark_type : str
        Benchmark category.
    run_name_prefix : str | None
        Prefix used for case-specific run directories.
    extra_metadata : dict | None
        Additional metadata fields.
    postprocess : Callable[[Path, np.ndarray, float], None] | None
        Callback invoked after each successful run.
    """
    for alg_name, algo in algorithms:
        run_name = f"{run_name_prefix}_ninit{algo.n_init}" if run_name_prefix else None
        meta_path = (
            _algorithm_output_dir(
                output_dir, dataset_name, alg_name, target_k, algo.n_init, run_name
            )
            / "metadata.json"
        )
        if skip_existing and meta_path.exists():
            logger.info("  Skipping %s | k=%s; output already exists.", alg_name, target_k)
            continue

        logger.info("  Running %s | k=%s | n_init=%s", alg_name, target_k, algo.n_init)
        labels, duration = _run_algorithm(algo, X, y, target_k)
        if labels is None:
            logger.error("    %s failed to produce labels.", alg_name)
            continue

        wcss = overall_wcss(X, labels)
        _save_run_outputs(
            dataset_name=dataset_name,
            alg_name=alg_name,
            k=target_k,
            k_mult=k_mult,
            n_init=algo.n_init,
            X=X,
            y=y,
            labels=labels,
            duration=duration,
            wcss=wcss,
            output_dir=output_dir,
            benchmark_type=benchmark_type,
            seed=seed,
            extra_metadata=extra_metadata,
            run_name=run_name,
        )
        logger.info(
            "    -> Time: %.4fs | WCSS: %.4f | Clusters: %s",
            duration,
            wcss,
            len(np.unique(labels)),
        )
        if postprocess is not None:
            postprocess(meta_path, labels, duration)


def _build_algorithms(  # noqa: C901
    *,
    seed: int,
    n_inits: list[int] | tuple[int, ...],
    subsample_size: int,
    bp_ranking_metrics: list[RankingMetric] | tuple[RankingMetric, ...],
    bp_init_strategies: list[InitStrategy] | tuple[InitStrategy, ...],
    bp_init_algorithms: list[InitAlgorithm] | tuple[InitAlgorithm, ...],
    include_cop_kmeans: bool,
    include_hac: bool,
    include_bisecting_kmeans: bool,
    include_bisecting_kmeans_m_rl: bool,
    include_bp_kmeans: bool = True,
) -> list[tuple[str, BaseAlgo]]:
    """
    Build the algorithms used by the benchmark suite.

    Parameters
    ----------
    seed : int
        Base random seed.
    n_inits : list[int] | tuple[int, ...]
        Initialization counts.
    subsample_size : int
        Maximum subsample size for BP-KMeans initialization.
    bp_ranking_metrics : list[RankingMetric] | tuple[RankingMetric, ...]
        BP-KMeans ranking metrics.
    bp_init_strategies : list[InitStrategy] | tuple[InitStrategy, ...]
        BP-KMeans initialization strategies.
    bp_init_algorithms : list[InitAlgorithm] | tuple[InitAlgorithm, ...]
        BP-KMeans initialization algorithms.
    include_cop_kmeans : bool
        Whether to include COP-KMeans.
    include_hac : bool
        Whether to include HAC.
    include_bisecting_kmeans : bool
        Whether to include standard bisecting K-Means.
    include_bisecting_kmeans_m_rl : bool
        Whether to include M_RL bisecting K-Means.
    include_bp_kmeans : bool
        Whether to include BP-KMeans combinations.

    Returns
    -------
    list[tuple[str, BaseAlgo]]
        Algorithm display names paired with configured instances.
    """
    algorithms: list[tuple[str, BaseAlgo]] = []

    if include_bp_kmeans:
        for ranking_metric in bp_ranking_metrics:
            for init in bp_init_strategies:
                for init_algorithm in bp_init_algorithms:
                    for n_init in n_inits:
                        name = (
                            f"BP-KMeans ({ranking_metric.name}, {init.name}, {init_algorithm.name})"
                        )
                        algorithm = BPKMeans(
                            seed=seed,
                            n_init=n_init,
                            ranking_metric=ranking_metric,
                            init_strategy=init,
                            init_algorithm=init_algorithm,
                            subsample_size=subsample_size,
                        )
                        algorithms.append((name, algorithm))

    for n_init in n_inits:
        if include_cop_kmeans:
            algorithms.append(("COP-KMeans", COPKMeansCannotLink(seed=seed, n_init=n_init)))
        if include_bisecting_kmeans:
            algorithms.append(
                ("Bisecting KMeans", BisectingKMeansNoRefine(seed=seed, n_init=n_init))
            )
        if include_bisecting_kmeans_m_rl:
            algorithms.append(
                (
                    "Bisecting KMeans M_RL",
                    BisectingKMeansMRLNoRefine(seed=seed, n_init=n_init),
                )
            )

    if include_hac:
        algorithms.append(("HAC Ward (NNC)", HACWardNNC(seed=seed)))
    return algorithms


def run_benchmark(
    datasets_dir: Path,
    output_dir: Path,
    *,
    benchmark_type: Literal["regular", "hac_strength", "special"],
    seed: int,
    n_inits: list[int] | tuple[int, ...],
    subsample_size: int,
    bp_ranking_metrics: list[RankingMetric] | tuple[RankingMetric, ...],
    bp_init_strategies: list[InitStrategy] | tuple[InitStrategy, ...],
    bp_init_algorithms: list[InitAlgorithm] | tuple[InitAlgorithm, ...],
    include_cop_kmeans: bool,
    include_hac: bool,
    skip_existing: bool,
    include_bisecting_kmeans: bool,
    include_bisecting_kmeans_m_rl: bool,
    include_bp_kmeans: bool,
    k_values: list[float] | tuple[float, ...],
    dataset_filename: str | None = None,
    label_column: str = "CUSEC",
    log_label_name: str | None = None,
) -> None:
    """
    Run a benchmark with case-specific settings selected by benchmark type.

    Parameters
    ----------
    datasets_dir : Path
        Input dataset directory.
    output_dir : Path
        Benchmark output root.
    benchmark_type : Literal['regular', 'hac_strength', 'special']
        Benchmark case to execute.
    seed : int
        Base random seed.
    n_inits : list[int] | tuple[int, ...]
        Initialization counts.
    subsample_size : int
        BP-KMeans subsample size.
    bp_ranking_metrics : list[RankingMetric] | tuple[RankingMetric, ...]
        BP-KMeans ranking metrics.
    bp_init_strategies : list[InitStrategy] | tuple[InitStrategy, ...]
        BP-KMeans initialization strategies.
    bp_init_algorithms : list[InitAlgorithm] | tuple[InitAlgorithm, ...]
        BP-KMeans initialization algorithms.
    include_cop_kmeans : bool
        Whether to include COP-KMeans.
    include_hac : bool
        Whether to include HAC.
    skip_existing : bool
        Whether completed runs are skipped.
    include_bisecting_kmeans : bool
        Whether to include standard bisecting K-Means.
    include_bisecting_kmeans_m_rl : bool
        Whether to include M_RL bisecting K-Means.
    include_bp_kmeans : bool
        Whether to include BP-KMeans.
    k_values : list[float] | tuple[float, ...]
        Requested multipliers or special target values.
    dataset_filename : str | None
        Explicit dataset filename.
    label_column : str
        Source-label column.
    log_label_name : str | None
        Human-readable label name for logging.

    Raises
    ------
    ValueError
        If ``k_values`` is empty or contains a non-positive value.
    """
    if not k_values or any(value <= 0 for value in k_values):
        msg = "k must be a non-empty list of positive values"
        raise ValueError(msg)

    algorithms = _build_algorithms(
        seed=seed,
        n_inits=n_inits,
        subsample_size=subsample_size,
        bp_ranking_metrics=bp_ranking_metrics,
        bp_init_strategies=bp_init_strategies,
        bp_init_algorithms=bp_init_algorithms,
        include_cop_kmeans=include_cop_kmeans,
        include_hac=include_hac,
        include_bisecting_kmeans=include_bisecting_kmeans,
        include_bisecting_kmeans_m_rl=include_bisecting_kmeans_m_rl,
        include_bp_kmeans=include_bp_kmeans,
    )

    for dataset_path, X, y in _iter_datasets(
        datasets_dir,
        label_column,
        regular_only=benchmark_type == "regular",
        dataset_filename=dataset_filename,
    ):
        postprocess = _distance_postprocessor(X, y) if benchmark_type == "special" else None

        n_instances = len(X)
        n_labels = len(np.unique(y))

        label_description = f"unique {log_label_name}" if log_label_name else "Labels"
        logger.info(
            "  Instances: %s, %s: %s | k=%s",
            n_instances,
            label_description,
            n_labels,
            k_values,
        )

        cases: list[tuple[int, float, Path, str, str | None, dict | None]] = []
        if benchmark_type == "regular":
            cases.extend(
                (int(n_labels * k_value), k_value, output_dir, "regular", None, None)
                for k_value in k_values
            )
        elif benchmark_type == "hac_strength":
            cases.extend(
                (
                    (
                        case_target_k := min(
                            (requested_target_k := int(n_instances * k_value)), n_instances
                        )
                    ),
                    k_value,
                    output_dir / "hac_strength",
                    "hac_strength",
                    f"nodesx{str(k_value).replace('.', '_')}_k{case_target_k}",
                    {
                        "cluster_multiplier_basis": "nodes",
                        "requested_cluster_multiplier": k_value,
                        "requested_n_clusters": requested_target_k,
                        "effective_n_clusters": case_target_k,
                        "target_k_was_capped": requested_target_k != case_target_k,
                    },
                )
                for k_value in k_values
            )
        else:
            cases.extend(
                (int(k_value), 1.0, output_dir, "special", None, None) for k_value in k_values
            )

        for case_target_k, k_mult, case_output_dir, case_type, run_name, metadata in cases:
            if not n_labels <= case_target_k <= n_instances:
                logger.warning(
                    "  Skipping k=%s because it must be between labels=%s and instances=%s.",
                    case_target_k,
                    n_labels,
                    n_instances,
                )
                continue
            if metadata and metadata.get("target_k_was_capped"):
                logger.warning(
                    "  Requested k=%s exceeds n_instances=%s; using k=%s.",
                    metadata["requested_n_clusters"],
                    n_instances,
                    case_target_k,
                )
            _run_algorithms(
                algorithms=algorithms,
                dataset_name=dataset_path.stem,
                X=X,
                y=y,
                target_k=case_target_k,
                k_mult=k_mult,
                output_dir=case_output_dir,
                seed=seed,
                skip_existing=skip_existing,
                benchmark_type=case_type,
                run_name_prefix=run_name,
                extra_metadata=metadata,
                postprocess=postprocess,
            )


def _compute_distance_metrics(X: np.ndarray, labels: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute distances to the node closest to each cluster centroid.

    Parameters
    ----------
    X : np.ndarray
        Input coordinates.
    labels : np.ndarray
        Cluster assignment for each point.
    y : np.ndarray
        Original source labels.

    Returns
    -------
    dict
        Aggregate representative-node distance metrics and compatibility aliases.
    """
    all_dists = np.empty(len(X))
    max_per_label: dict = {}
    representative_indices: list[int] = []

    for c in np.unique(labels):
        mask = labels == c
        cluster_indices = np.flatnonzero(mask)
        pts = X[mask]
        centroid = pts.mean(axis=0)
        representative_local_idx = int(np.argmin(np.sum((pts - centroid) ** 2, axis=1)))
        representative = pts[representative_local_idx]
        representative_indices.append(int(cluster_indices[representative_local_idx]))
        dists = np.linalg.norm(pts - representative, axis=1)
        all_dists[mask] = dists

    # mean-max distance per original label
    for lbl in np.unique(y):
        lbl_mask = y == lbl
        max_per_label[lbl] = float(all_dists[lbl_mask].max())

    avg_dist = float(all_dists.mean())
    max_dist = float(all_dists.max())
    mean_max_per_label = float(np.mean(list(max_per_label.values())))

    return {
        "distance_anchor": "nearest_node_to_cluster_centroid",
        "representative_node_count": len(representative_indices),
        "avg_dist_to_representative_node_m": avg_dist,
        "max_dist_to_representative_node_m": max_dist,
        "mean_max_dist_per_label_to_representative_node_m": mean_max_per_label,
        # Backwards-compatible aliases used by existing analysis scripts.
        "avg_dist_to_centroid_m": avg_dist,
        "max_dist_to_centroid_m": max_dist,
        "mean_max_dist_per_label_m": mean_max_per_label,
    }


def _distance_postprocessor(
    X: np.ndarray, y: np.ndarray
) -> Callable[[Path, np.ndarray, float], None]:
    """
    Build the postprocessor used by special distance benchmarks.

    Parameters
    ----------
    X : np.ndarray
        Input coordinates retained by the callback.
    y : np.ndarray
        Source labels retained by the callback.

    Returns
    -------
    Callable[[Path, np.ndarray, float], None]
        Callback accepting metadata path, fitted labels, and elapsed duration.
    """

    def process(meta_path: Path, labels: np.ndarray, duration: float) -> None:
        """
        Add distance metrics to one run's metadata.

        Parameters
        ----------
        meta_path : Path
            Metadata JSON file to update.
        labels : np.ndarray
            Fitted cluster labels.
        duration : float
            Elapsed fitting time used for logging.
        """
        dist_metrics = _compute_distance_metrics(X, labels, y)
        _update_metadata(meta_path, dist_metrics)
        logger.info(
            "    -> time: %.4fs | avg dist to node: %.2f m | "
            "max dist to node: %.2f m | mean-max/label: %.2f m",
            duration,
            dist_metrics["avg_dist_to_representative_node_m"],
            dist_metrics["max_dist_to_representative_node_m"],
            dist_metrics["mean_max_dist_per_label_to_representative_node_m"],
        )

    return process
