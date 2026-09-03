"""Functional tests for the reproducible benchmark runner."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bp_k_means import __version__, bp_kmeans
from bp_k_means.algos.cop_k_means import COPKMeans
from bp_k_means.main import ExperimentConfig, load_config, run_experiment
from bp_k_means.tools.benchmark import _build_algorithms


def test_public_api_exports_algorithm_and_version() -> None:
    """Expose the main algorithm and package version from the package root."""
    assert callable(bp_kmeans)
    assert isinstance(__version__, str)
    assert __version__


def test_cop_kmeans_is_available_as_an_opt_in_algorithm() -> None:
    """Keep COP-KMeans available without changing the default benchmark suite."""
    default_algorithms = _build_algorithms(n_inits=(1,))
    opt_in_algorithms = _build_algorithms(n_inits=(1,), include_cop_kmeans=True)

    assert not any(name == "COP-KMeans" for name, _ in default_algorithms)
    cop_algorithms = [algo for name, algo in opt_in_algorithms if name == "COP-KMeans"]
    assert len(cop_algorithms) == 1
    assert isinstance(cop_algorithms[0], COPKMeans)


def test_hac_can_be_excluded_from_the_algorithm_suite() -> None:
    """Honor the HAC opt-out used by the benchmark configuration."""
    algorithms = _build_algorithms(n_inits=(1,), include_hac=False)

    assert not any(name == "HAC Ward (NNC)" for name, _ in algorithms)


def test_bisecting_kmeans_algorithms_are_configurable() -> None:
    """Honor independent include switches for the bisecting implementations."""
    algorithms = _build_algorithms(
        n_inits=(1,),
        include_bisecting_kmeans=False,
        include_precomputed_bisecting_kmeans=False,
    )

    assert not any(name == "Bisecting KMeans" for name, _ in algorithms)
    assert not any(name == "Bisecting KMeans (M_RL)" for name, _ in algorithms)


def test_load_config_resolves_paths_relative_to_config(tmp_path: Path) -> None:
    """Load benchmark values and resolve relative paths from the config location."""
    config_path = tmp_path / "experiment.toml"
    config_path.write_text(
        """[benchmark]
datasets_dir = "datasets"
output_dir = "results"
seed = 123
k_multipliers = [1.5]
n_inits = [1]
subsample_size = 4
run_regular = false
run_hac_strength = true
hac_strength_multiplier = 0.75
run_special = false
include_cop_kmeans = true
skip_existing = true
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.datasets_dir == (tmp_path / "datasets").resolve()
    assert config.output_dir == (tmp_path / "results").resolve()
    assert config.seed == 123
    assert config.k_multipliers == (1.5,)
    assert config.n_inits == (1,)
    assert config.run_hac_strength is True
    assert config.include_cop_kmeans is True
    assert config.skip_existing is True


def test_experiment_records_seed_and_repeats_deterministically(tmp_path: Path) -> None:
    """Run a small benchmark and verify its manifest, metadata, and stable labels."""
    datasets_dir = tmp_path / "datasets"
    output_dir = tmp_path / "results"
    datasets_dir.mkdir()
    pd.DataFrame(
        {
            "x_utm": [0.0, 1.0, 2.0, 5.0, 6.0, 7.0],
            "y_utm": [0.0, 0.0, 1.0, 5.0, 5.0, 6.0],
            "CUSEC": ["left", "left", "left", "right", "right", "right"],
        }
    ).to_parquet(datasets_dir / "toy_nodes.parquet", index=False)
    config = ExperimentConfig(
        datasets_dir=datasets_dir,
        output_dir=output_dir,
        seed=123,
        k_multipliers=(1.5,),
        n_inits=(1,),
        subsample_size=4,
        run_special=False,
        include_cop_kmeans=True,
    )

    run_experiment(config, config_name="experiment.toml")
    manifest = json.loads(
        (output_dir / "benchmark" / "experiment_config.json").read_text(encoding="utf-8")
    )
    metadata_paths = list(output_dir.rglob("metadata.json"))
    first_instances = pd.read_parquet(sorted(output_dir.rglob("instances.parquet"))[0])

    run_experiment(config, config_name="experiment.toml")
    second_instances = pd.read_parquet(sorted(output_dir.rglob("instances.parquet"))[0])

    assert manifest["config_file"] == "experiment.toml"
    assert manifest["config"]["seed"] == 123
    assert metadata_paths
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["seed"] == 123 for path in metadata_paths
    )
    assert any(
        json.loads(path.read_text(encoding="utf-8"))["algorithm"] == "COP-KMeans"
        for path in metadata_paths
    )
    np.testing.assert_array_equal(first_instances["cluster"], second_instances["cluster"])
