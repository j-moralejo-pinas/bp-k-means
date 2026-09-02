"""Tests for the unified bp_kmeans implementation."""

import numpy as np
import pytest

from bp_k_means.algos.bp_kmeans import InitStrategy, RankingStrategy, bp_kmeans


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Create three well-separated labeled point clouds."""
    rng = np.random.default_rng(42)
    # 3 classes, 30 points each, 2D
    X0 = rng.normal(loc=[0, 0], scale=0.5, size=(30, 2))
    X1 = rng.normal(loc=[5, 5], scale=0.5, size=(30, 2))
    X2 = rng.normal(loc=[10, 0], scale=0.5, size=(30, 2))
    X = np.vstack([X0, X1, X2])
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)
    return X, y


class TestBasicBehavior:
    """Test basic output and validation behavior."""

    def test_returns_correct_shape(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Return one cluster label for each input point."""
        X, y = sample_data
        labels = bp_kmeans(X, y, target_k=6, seed=42)
        assert labels.shape == (X.shape[0],)

    def test_correct_number_of_clusters(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Return the requested number of clusters."""
        X, y = sample_data
        labels = bp_kmeans(X, y, target_k=6, seed=42)
        assert len(np.unique(labels)) == 6

    def test_label_consistency_constraint(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """All points in the same cluster must share the same original label."""
        X, y = sample_data
        labels = bp_kmeans(X, y, target_k=9, seed=42)
        for cluster_id in np.unique(labels):
            cluster_labels = y[labels == cluster_id]
            assert len(np.unique(cluster_labels)) == 1

    def test_target_k_equals_n_classes(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Allow one cluster per original class."""
        X, y = sample_data
        labels = bp_kmeans(X, y, target_k=3, seed=42)
        assert len(np.unique(labels)) == 3

    def test_target_k_equals_n_samples(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Allow one cluster per input point."""
        X, y = sample_data
        labels = bp_kmeans(X, y, target_k=X.shape[0], seed=42)
        assert len(np.unique(labels)) == X.shape[0]

    def test_raises_on_invalid_target_k(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Reject target sizes outside the label-constrained feasible range."""
        X, y = sample_data
        with pytest.raises(ValueError, match="target_k cannot be larger"):
            bp_kmeans(X, y, target_k=X.shape[0] + 1, seed=42)
        with pytest.raises(ValueError, match="target_k cannot be smaller"):
            bp_kmeans(X, y, target_k=2, seed=42)  # fewer than n_classes


class TestRankingStrategies:
    """Test every supported label-ranking strategy."""

    def test_names_match_paper(self) -> None:
        """Expose the metric names used in the paper."""
        assert [ranking.name for ranking in RankingStrategy] == ["M_L", "M_C", "M_ERL", "M_RL"]

    @pytest.mark.parametrize("ranking", list(RankingStrategy))
    def test_all_rankings_produce_valid_output(
        self, sample_data: tuple[np.ndarray, np.ndarray], ranking: RankingStrategy
    ) -> None:
        """Produce valid label-constrained clusters for every ranking."""
        X, y = sample_data
        labels = bp_kmeans(X, y, target_k=9, seed=42, ranking_strategy=ranking)
        assert labels.shape == (X.shape[0],)
        assert len(np.unique(labels)) == 9
        # Label consistency
        for cluster_id in np.unique(labels):
            assert len(np.unique(y[labels == cluster_id])) == 1


class TestInitStrategies:
    """Test every supported centroid initialization strategy."""

    @pytest.mark.parametrize("init", list(InitStrategy))
    def test_all_inits_produce_valid_output(
        self, sample_data: tuple[np.ndarray, np.ndarray], init: InitStrategy
    ) -> None:
        """Produce valid label-constrained clusters for every initializer."""
        X, y = sample_data
        labels = bp_kmeans(X, y, target_k=9, seed=42, init_strategy=init)
        assert labels.shape == (X.shape[0],)
        assert len(np.unique(labels)) == 9
        for cluster_id in np.unique(labels):
            assert len(np.unique(y[labels == cluster_id])) == 1


class TestAllCombinations:
    """Test the Cartesian product of ranking and initialization strategies."""

    @pytest.mark.parametrize("ranking", list(RankingStrategy))
    @pytest.mark.parametrize("init", list(InitStrategy))
    def test_all_ranking_init_combinations(
        self,
        sample_data: tuple[np.ndarray, np.ndarray],
        ranking: RankingStrategy,
        init: InitStrategy,
    ) -> None:
        """Produce valid clusters for every strategy combination."""
        X, y = sample_data
        labels = bp_kmeans(X, y, target_k=6, seed=42, ranking_strategy=ranking, init_strategy=init)
        assert labels.shape == (X.shape[0],)
        assert len(np.unique(labels)) == 6
        for cluster_id in np.unique(labels):
            assert len(np.unique(y[labels == cluster_id])) == 1


class TestEdgeCases:
    """Test small and non-numeric label inputs."""

    def test_single_class(self) -> None:
        """Split a single class into the requested number of clusters."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 2))
        y = np.zeros(20, dtype=int)
        labels = bp_kmeans(X, y, target_k=5, seed=42)
        assert len(np.unique(labels)) == 5

    def test_two_points_per_class(self) -> None:
        """Handle classes containing exactly two points."""
        X = np.array([[0, 0], [1, 1], [5, 5], [6, 6]], dtype=float)
        y = np.array([0, 0, 1, 1])
        labels = bp_kmeans(X, y, target_k=4, seed=42)
        assert len(np.unique(labels)) == 4

    def test_string_labels(self) -> None:
        """Accept string-valued class labels."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 2))
        y = np.array(["cat"] * 10 + ["dog"] * 10 + ["bird"] * 10)
        labels = bp_kmeans(X, y, target_k=6, seed=42)
        assert len(np.unique(labels)) == 6
