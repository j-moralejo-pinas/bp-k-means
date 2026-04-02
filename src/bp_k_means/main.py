import logging
import time

import numpy as np
import pandas as pd

from bp_k_means.bp_kmeans import InitStrategy, RankingStrategy, bp_kmeans
from bp_k_means.bp_k_means_optimized import bp_kmeans_optimized
from bp_k_means.experimental_bisecting_k_means import divisive_kmeans_by_label
from bp_k_means.experimental_bpk_means import experimental_bp_kmeans
from bp_k_means.cop_k_means import cop_kmeans_by_class
from bp_k_means.hac import hac_ward_by_label

logger = logging.getLogger(__name__)


def overall_wcss(X, labels):
    k = labels.max() + 1
    wcss = 0.0

    for c in range(k):
        pts = X[labels == c]
        if len(pts) > 0:
            centroid = pts.mean(axis=0)
            diff = pts - centroid
            wcss += np.sum(diff * diff)

    return wcss


def main():
    rng = np.random.default_rng(42)
    # Load dataset
    df = pd.read_parquet("data/datasets/madrid_osm_drive_nodes.parquet")
    print(len(df))
    X = df[["x_utm", "y_utm"]].values
    y = df["CUSEC"].values

    # Choose number of clusters, for example 4000
    target_k = 10000
    n_init = 1

    # -------------------------------
    # BP-KMEANS
    # -------------------------------
    start_time = time.time()
    for i in range(10):
        labels_bp = bp_kmeans_optimized(X, y, target_k, seed=42, n_init=n_init)
        labels_bp = bp_kmeans(
            X,
            y,
            target_k,
            seed=42,
            n_init=n_init,
            init_strategy=InitStrategy.ADD_CENTROID_CLUSTER,
            ranking_strategy=RankingStrategy.EST_REDUCTION_CLUSTER,
        )
    end_time = time.time()

    wcss_bp = overall_wcss(X, labels_bp)

    logger.info(f"BP-KMeans clusters: {labels_bp.max() + 1}")
    logger.info(f"BP-KMeans WCSS: {wcss_bp}")
    logger.info(f"BP-KMeans time: {end_time - start_time:.4f} seconds")

    df["bp_kmeans_cluster"] = labels_bp
    df.to_parquet("madrid_bp_kmeans_output.parquet", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
