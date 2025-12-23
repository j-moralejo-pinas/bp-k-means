import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bp_k_means.experimental_bisecting_k_means import divisive_kmeans_by_label
from bp_k_means.bisecting_k_means_optimized import bisecting_kmeans_by_label_optimized
from bp_k_means.experimental_bpk_means import experimental_bp_kmeans
from bp_k_means.bp_k_means_optimized import bp_kmeans_optimized
from bp_k_means.cop_k_means import cop_kmeans_by_class
from bp_k_means.hac import hac_ward_by_label
from bp_k_means.main import overall_wcss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


def run_bisecting_kmeans(X, y, k, seed, n_init, refine_clusters=False):
    rng = np.random.default_rng(seed)
    best_wcss = float("inf")
    best_labels = None

    for i in range(n_init):
        current_seed = rng.integers(2**32)
        labels = divisive_kmeans_by_label(
            X, y, target_k=k, seed=current_seed, refine_clusters=refine_clusters
        )

        wcss = overall_wcss(X, labels)
        if wcss < best_wcss:
            best_wcss = wcss
            best_labels = labels

    return best_labels


def run_benchmark():
    datasets_dir = Path("data/datasets")
    dataset_files = list(datasets_dir.glob("*.parquet"))

    # Exclude if madrid dataset is present
    dataset_files = [f for f in dataset_files if "madrid" not in f.name.lower()]

    if not dataset_files:
        logger.error(f"No parquet files found in {datasets_dir}")
        return

    n_inits = [1, 2, 4, 8, 16, 32]
    k_multipliers = [1.5, 2, 4, 8, 16, 32]

    results = []

    algorithms = [
        # (
        #     "BP-KMeans",
        #     lambda X, y, k, seed, n_init: experimental_bp_kmeans(
        #         X, y, k, seed=seed, n_init=n_init, reuse_centroids=0
        #     ),
        # ),
        # (
        #     "BP-KMeans Reuse",
        #     lambda X, y, k, seed, n_init: experimental_bp_kmeans(
        #         X, y, k, seed=seed, n_init=n_init, reuse_centroids=1
        #     ),
        # ),
        # (
        #     "BP-KMeans Reuse Split",
        #     lambda X, y, k, seed, n_init: experimental_bp_kmeans(
        #         X, y, k, seed=seed, n_init=n_init, reuse_centroids=2
        #     ),
        # ),
        # (
        #     "BP-KMeans WCSS/Cluster",
        #     lambda X, y, k, seed, n_init: experimental_bp_kmeans(
        #         X, y, k, seed=seed, n_init=n_init, reuse_centroids=0, use_wcss_per_cluster=True
        #     ),
        # ),
        # (
        #     "BP-KMeans Reuse WCSS/Cluster",
        #     lambda X, y, k, seed, n_init: experimental_bp_kmeans(
        #         X,
        #         y,
        #         k,
        #         seed=seed,
        #         n_init=n_init,
        #         reuse_centroids=1,
        #         use_wcss_per_cluster=True,
        #     ),
        # ),
        # (
        #     "BP-KMeans Reuse Split WCSS/Cluster",
        #     lambda X, y, k, seed, n_init: experimental_bp_kmeans(
        #         X,
        #         y,
        #         k,
        #         seed=seed,
        #         n_init=n_init,
        #         reuse_centroids=2,
        #         use_wcss_per_cluster=True,
        #     ),
        # ),
        (
            "BP-KMeans Optimized",
            lambda X, y, k, seed, n_init: bp_kmeans_optimized(X, y, k, seed=seed, n_init=n_init),
        ),
        # (
        #     "COP-KMeans",
        #     lambda X, y, k, seed, n_init: run_cop_kmeans(
        #         X, y, k, seed=seed, n_init=n_init, init_ensure_class=False
        #     ),
        # ),
        # (
        #     "COP-KMeans Special",
        #     lambda X, y, k, seed, n_init: run_cop_kmeans(
        #         X, y, k, seed=seed, n_init=n_init, init_ensure_class=True
        #     ),
        # ),
        # ("HAC-Ward", lambda X, y, k, seed, n_init: hac_ward_by_label(X, y, target_k=k)),
        # (
        #     "Bisecting KMeans",
        #     lambda X, y, k, seed, n_init: run_bisecting_kmeans(
        #         X, y, k, seed, n_init, refine_clusters=False
        #     ),
        # ),
        # (
        #     "Bisecting KMeans Refined",
        #     lambda X, y, k, seed, n_init: run_bisecting_kmeans(
        #         X, y, k, seed, n_init, refine_clusters=True
        #     ),
        # ),
        (
            "Bisecting KMeans Optimized",
            lambda X, y, k, seed, n_init: bisecting_kmeans_by_label_optimized(
                X, y, k, seed=seed, n_init=n_init
            ),
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
                    if alg_name == "HAC-Ward" and n_init > 1:
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

                    logger.info(
                        f"    -> Time: {duration:.4f}s | WCSS: {wcss:.4f} | Clusters: {n_clusters}"
                    )

                    results.append(
                        {
                            "Dataset": dataset_path.name,
                            "Algorithm": alg_name,
                            "n_init": n_init,
                            "k_multiplier": k_mult,
                            "k": target_k,
                            "Time": duration,
                            "WCSS": wcss,
                            "Clusters": n_clusters,
                        }
                    )


    # Save results
    if results:
        results_df = pd.DataFrame(results)
        output_file = "benchmark_results.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"Benchmark complete. Results saved to {output_file}")
        print("\nBenchmark Summary:")
        print(results_df.to_string())
    else:
        logger.warning("No results collected.")


if __name__ == "__main__":
    run_benchmark()
