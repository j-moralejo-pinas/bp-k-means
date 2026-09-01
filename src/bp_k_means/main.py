"""Command-line entry point for running a BP-KMeans benchmark."""

import logging
import time

import pandas as pd

from bp_k_means.algos.bp_kmeans import InitStrategy, RankingStrategy, bp_kmeans
from bp_k_means.utils.metrics import overall_wcss

logger = logging.getLogger(__name__)


def main() -> None:
    """Run BP-KMeans on the configured Madrid dataset and save its labels."""
    # Load dataset
    df = pd.read_parquet("data/datasets/madrid_osm_drive_nodes.parquet")
    X = df[["x_utm", "y_utm"]].to_numpy()
    y = df["CUSEC"].to_numpy()

    # Choose number of clusters, for example 4000
    target_k = 10000
    n_init = 1

    # -------------------------------
    # BP-KMEANS
    # -------------------------------
    start_time = time.time()
    for _i in range(10):
        labels_bp = bp_kmeans(
            X,
            y,
            target_k,
            seed=42,
            n_init=n_init,
            init_strategy=InitStrategy.I_ACC,
            ranking_strategy=RankingStrategy.R_ERC,
        )
    end_time = time.time()

    wcss_bp = overall_wcss(X, labels_bp)

    logger.info("BP-KMeans clusters: %s", labels_bp.max() + 1)
    logger.info("BP-KMeans WCSS: %s", wcss_bp)
    logger.info("BP-KMeans time: %.4f seconds", end_time - start_time)

    df["bp_kmeans_cluster"] = labels_bp
    df.to_parquet("madrid_bp_kmeans_output.parquet", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
