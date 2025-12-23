import numpy as np
from bp_k_means.bp_k_means_optimized import bp_kmeans_optimized


def main():
    # Generate synthetic data
    n_samples = 1000
    n_features = 2
    n_classes = 5
    target_k = 20

    rng = np.random.default_rng(42)
    X = rng.random((n_samples, n_features))
    y = rng.integers(0, n_classes, size=n_samples)

    print(f"Running bp_kmeans_optimized with target_k={target_k}")

    labels = bp_kmeans_optimized(X, y, target_k=target_k, seed=42, n_init=10)

    n_clusters = len(np.unique(labels))
    print(f"Resulting number of clusters: {n_clusters}")

    if n_clusters == target_k:
        print("Success: Target number of clusters reached.")
    else:
        print(f"Warning: Expected {target_k} clusters, got {n_clusters}")


if __name__ == "__main__":
    main()
