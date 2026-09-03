"""Run the reproducible BP-KMeans benchmark suite."""

import argparse
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from bp_k_means.tools.benchmark import (
    benchmark_castile_leon_max_response_time,
    benchmark_com_madrid_avg_distance_to_centroid,
    run_benchmark,
    run_hac_strength_benchmark,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a reproducible benchmark run."""

    datasets_dir: Path
    benchmark_output_dir: Path
    analysis_output_dir: Path
    seed: int
    k_multipliers: tuple[float, ...]
    n_inits: tuple[int, ...]
    subsample_size: int
    run_regular: bool
    run_hac_strength: bool
    hac_strength_multiplier: float
    run_special: bool
    include_cop_kmeans: bool
    include_hac: bool
    skip_existing: bool
    include_bisecting_kmeans: bool
    include_precomputed_bisecting_kmeans: bool

def _resolve_path(value: object, config_path: Path, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _read_positive_values[Number: (int, float)](
    value: object,
    field_name: str,
    value_type: type[Number],
) -> tuple[Number, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        msg = f"{field_name} must be a non-empty array"
        raise ValueError(msg)

    values = tuple(value_type(item) for item in value)
    if any(item <= 0 for item in values):
        msg = f"{field_name} must contain only positive values"
        raise ValueError(msg)
    return values


def _required_setting(settings: dict, field_name: str) -> object:
    """Return a required benchmark setting with a useful validation error."""
    if field_name not in settings:
        msg = f"missing required benchmark setting: {field_name}"
        raise ValueError(msg)
    return settings[field_name]


def load_config(config_path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from TOML."""
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    settings = raw_config.get("benchmark", raw_config)
    if not isinstance(settings, dict):
        msg = "configuration must contain a [benchmark] table"
        raise TypeError(msg)

    k_multipliers = _read_positive_values(
        _required_setting(settings, "k_multipliers"),
        "k_multipliers",
        float,
    )
    n_inits = _read_positive_values(
        _required_setting(settings, "n_inits"),
        "n_inits",
        int,
    )
    subsample_size = int(_required_setting(settings, "subsample_size"))
    if subsample_size < 1:
        msg = "subsample_size must be >= 1"
        raise ValueError(msg)

    return ExperimentConfig(
        datasets_dir=_resolve_path(
            _required_setting(settings, "datasets_dir"), config_path, "datasets_dir"
        ),
        benchmark_output_dir=_resolve_path(
            _required_setting(settings, "benchmark_output_dir"),
            config_path,
            "benchmark_output_dir",
        ),
        analysis_output_dir=_resolve_path(
            _required_setting(settings, "analysis_output_dir"),
            config_path,
            "analysis_output_dir",
        ),
        seed=int(_required_setting(settings, "seed")),
        k_multipliers=k_multipliers,
        n_inits=n_inits,
        subsample_size=subsample_size,
        run_regular=bool(_required_setting(settings, "run_regular")),
        run_hac_strength=bool(_required_setting(settings, "run_hac_strength")),
        hac_strength_multiplier=float(_required_setting(settings, "hac_strength_multiplier")),
        run_special=bool(_required_setting(settings, "run_special")),
        include_cop_kmeans=bool(_required_setting(settings, "include_cop_kmeans")),
        include_hac=bool(_required_setting(settings, "include_hac")),
        skip_existing=bool(_required_setting(settings, "skip_existing")),
        include_bisecting_kmeans=bool(_required_setting(settings, "include_bisecting_kmeans")),
        include_precomputed_bisecting_kmeans=bool(
            _required_setting(settings, "include_precomputed_bisecting_kmeans")
        ),
    )


def _config_for_json(config: ExperimentConfig) -> dict[str, object]:
    serialized = asdict(config)
    serialized["datasets_dir"] = str(config.datasets_dir)
    serialized["benchmark_output_dir"] = str(config.benchmark_output_dir)
    serialized["analysis_output_dir"] = str(config.analysis_output_dir)
    serialized["k_multipliers"] = list(config.k_multipliers)
    serialized["n_inits"] = list(config.n_inits)
    return serialized


def run_experiment(config: ExperimentConfig, *, config_name: str | None = None) -> None:
    """Run all benchmark stages selected by ``config``."""
    config.benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config_file": config_name,
        "config": _config_for_json(config),
    }
    (config.benchmark_output_dir / "experiment_config.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    if config.run_regular:
        run_benchmark(
            datasets_dir=config.datasets_dir,
            output_dir=config.benchmark_output_dir,
            seed=config.seed,
            k_multipliers=config.k_multipliers,
            n_inits=config.n_inits,
            subsample_size=config.subsample_size,
            include_cop_kmeans=config.include_cop_kmeans,
            include_hac=config.include_hac,
            skip_existing=config.skip_existing,
            include_bisecting_kmeans=config.include_bisecting_kmeans,
            include_precomputed_bisecting_kmeans=config.include_precomputed_bisecting_kmeans,
        )

    if config.run_hac_strength:
        run_hac_strength_benchmark(
            cluster_multiplier=config.hac_strength_multiplier,
            datasets_dir=config.datasets_dir,
            output_dir=config.benchmark_output_dir,
            seed=config.seed,
            n_inits=config.n_inits,
            subsample_size=config.subsample_size,
            include_cop_kmeans=config.include_cop_kmeans,
            include_hac=config.include_hac,
            skip_existing=config.skip_existing,
            include_bisecting_kmeans=config.include_bisecting_kmeans,
            include_precomputed_bisecting_kmeans=config.include_precomputed_bisecting_kmeans,
        )

    if config.run_special:
        benchmark_castile_leon_max_response_time(
            datasets_dir=config.datasets_dir,
            output_dir=config.benchmark_output_dir,
            seed=config.seed,
            n_inits=config.n_inits,
            subsample_size=config.subsample_size,
            include_cop_kmeans=config.include_cop_kmeans,
            include_hac=config.include_hac,
            skip_existing=config.skip_existing,
            include_bisecting_kmeans=config.include_bisecting_kmeans,
            include_precomputed_bisecting_kmeans=config.include_precomputed_bisecting_kmeans,
        )
        benchmark_com_madrid_avg_distance_to_centroid(
            datasets_dir=config.datasets_dir,
            output_dir=config.benchmark_output_dir,
            seed=config.seed,
            n_inits=config.n_inits,
            subsample_size=config.subsample_size,
            include_cop_kmeans=config.include_cop_kmeans,
            include_hac=config.include_hac,
            skip_existing=config.skip_existing,
            include_bisecting_kmeans=config.include_bisecting_kmeans,
            include_precomputed_bisecting_kmeans=config.include_precomputed_bisecting_kmeans,
        )


def main() -> None:
    """Run the configured experiment from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/default.toml"),
        help="TOML configuration file (default: experiments/default.toml).",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        run_experiment(config, config_name=args.config.name)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
