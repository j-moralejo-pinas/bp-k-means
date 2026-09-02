"""Run the reproducible BP-KMeans benchmark suite."""

import argparse
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from bp_k_means.tools.benchmark import (
    DEFAULT_K_MULTIPLIERS,
    DEFAULT_N_INITS,
    benchmark_castile_leon_max_response_time,
    benchmark_com_madrid_avg_distance_to_centroid,
    run_benchmark,
    run_hac_strength_benchmark,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a reproducible benchmark run."""

    datasets_dir: Path
    output_dir: Path
    seed: int = 42
    k_multipliers: tuple[float, ...] = DEFAULT_K_MULTIPLIERS
    n_inits: tuple[int, ...] = DEFAULT_N_INITS
    subsample_size: int = 10
    run_regular: bool = True
    run_hac_strength: bool = False
    hac_strength_multiplier: float = 1.5
    run_special: bool = True
    include_cop_kmeans: bool = False


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


def load_config(config_path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from TOML."""
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    settings = raw_config.get("benchmark", raw_config)
    if not isinstance(settings, dict):
        msg = "configuration must contain a [benchmark] table"
        raise TypeError(msg)

    k_multipliers = _read_positive_values(
        settings.get("k_multipliers", DEFAULT_K_MULTIPLIERS),
        "k_multipliers",
        float,
    )
    n_inits = _read_positive_values(
        settings.get("n_inits", DEFAULT_N_INITS),
        "n_inits",
        int,
    )
    subsample_size = int(settings.get("subsample_size", 10))
    if subsample_size < 1:
        msg = "subsample_size must be >= 1"
        raise ValueError(msg)

    return ExperimentConfig(
        datasets_dir=_resolve_path(
            settings.get("datasets_dir", "../data/datasets"), config_path, "datasets_dir"
        ),
        output_dir=_resolve_path(
            settings.get("output_dir", "../output"), config_path, "output_dir"
        ),
        seed=int(settings.get("seed", 42)),
        k_multipliers=k_multipliers,
        n_inits=n_inits,
        subsample_size=subsample_size,
        run_regular=bool(settings.get("run_regular", True)),
        run_hac_strength=bool(settings.get("run_hac_strength", False)),
        hac_strength_multiplier=float(settings.get("hac_strength_multiplier", 1.5)),
        run_special=bool(settings.get("run_special", True)),
        include_cop_kmeans=bool(settings.get("include_cop_kmeans", False)),
    )


def _config_for_json(config: ExperimentConfig) -> dict[str, object]:
    serialized = asdict(config)
    serialized["datasets_dir"] = str(config.datasets_dir)
    serialized["output_dir"] = str(config.output_dir)
    serialized["k_multipliers"] = list(config.k_multipliers)
    serialized["n_inits"] = list(config.n_inits)
    return serialized


def run_experiment(config: ExperimentConfig, *, config_name: str | None = None) -> None:
    """Run all benchmark stages selected by ``config``."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config_file": config_name,
        "config": _config_for_json(config),
    }
    (config.output_dir / "experiment_config.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    if config.run_regular:
        run_benchmark(
            datasets_dir=config.datasets_dir,
            output_dir=config.output_dir,
            seed=config.seed,
            k_multipliers=config.k_multipliers,
            n_inits=config.n_inits,
            subsample_size=config.subsample_size,
            include_cop_kmeans=config.include_cop_kmeans,
        )

    if config.run_hac_strength:
        run_hac_strength_benchmark(
            cluster_multiplier=config.hac_strength_multiplier,
            datasets_dir=config.datasets_dir,
            output_dir=config.output_dir,
            seed=config.seed,
            n_inits=config.n_inits,
            subsample_size=config.subsample_size,
            include_cop_kmeans=config.include_cop_kmeans,
        )

    if config.run_special:
        benchmark_castile_leon_max_response_time(
            datasets_dir=config.datasets_dir,
            output_dir=config.output_dir,
            seed=config.seed,
            n_inits=config.n_inits,
            subsample_size=config.subsample_size,
            include_cop_kmeans=config.include_cop_kmeans,
        )
        benchmark_com_madrid_avg_distance_to_centroid(
            datasets_dir=config.datasets_dir,
            output_dir=config.output_dir,
            seed=config.seed,
            n_inits=config.n_inits,
            subsample_size=config.subsample_size,
            include_cop_kmeans=config.include_cop_kmeans,
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
