"""Run the reproducible BP-KMeans benchmark suite."""

import argparse
import json
import tomllib
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from bp_k_means.algos.bp_kmeans import InitAlgorithm, InitStrategy, RankingMetric
from bp_k_means.tools.benchmark import run_benchmark

EnumValue = TypeVar("EnumValue", bound=Enum)


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuration for a reproducible benchmark run.

    Attributes
    ----------
    datasets_dir : Path
        Directory containing input parquet datasets.
    benchmark_output_dir : Path
        Directory receiving benchmark outputs.
    analysis_output_dir : Path
        Directory receiving analysis tables and figures.
    seed : int
        Base random seed.
    k : tuple[float, ...]
        Requested cluster multipliers or special target counts.
    n_inits : tuple[int, ...]
        Initialization counts to benchmark.
    subsample_size : int
        Maximum subsample size for subsampled k-means++.
    bp_ranking_metrics : tuple[RankingMetric, ...]
        BP-KMeans ranking metrics.
    bp_init_strategies : tuple[InitStrategy, ...]
        BP-KMeans initialization strategies.
    bp_init_algorithms : tuple[InitAlgorithm, ...]
        BP-KMeans initialization algorithms.
    run_regular : bool
        Whether the regular benchmark stage is enabled.
    run_hac_strength : bool
        Whether the HAC-strength benchmark stage is enabled.
    run_special : bool
        Whether the special benchmark stage is enabled.
    include_cop_kmeans : bool
        Whether COP-KMeans is included.
    include_hac : bool
        Whether HAC is included.
    skip_existing : bool
        Whether existing run outputs are skipped.
    include_bisecting_kmeans : bool
        Whether standard bisecting K-Means is included.
    include_bisecting_kmeans_m_rl : bool
        Whether M_RL bisecting K-Means is included.
    include_bp_kmeans : bool
        Whether BP-KMeans is included.
    """

    datasets_dir: Path
    benchmark_output_dir: Path
    analysis_output_dir: Path
    seed: int
    k: tuple[float, ...]
    n_inits: tuple[int, ...]
    subsample_size: int
    bp_ranking_metrics: tuple[RankingMetric, ...]
    bp_init_strategies: tuple[InitStrategy, ...]
    bp_init_algorithms: tuple[InitAlgorithm, ...]
    run_regular: bool
    run_hac_strength: bool
    run_special: bool
    include_cop_kmeans: bool
    include_hac: bool
    skip_existing: bool
    include_bisecting_kmeans: bool
    include_bisecting_kmeans_m_rl: bool
    include_bp_kmeans: bool = True


def _resolve_path(value: object, config_path: Path, field_name: str) -> Path:
    """
    Resolve a configuration path relative to its TOML file.

    Parameters
    ----------
    value : object
        Raw configuration value.
    config_path : Path
        Path to the configuration file.
    field_name : str
        Name used in validation errors.

    Returns
    -------
    Path
        Absolute resolved path.

    Raises
    ------
    ValueError
        If ``value`` is not a non-empty string.
    """
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
    """
    Read a non-empty sequence of positive typed values.

    Parameters
    ----------
    value : object
        Raw list or tuple from the configuration.
    field_name : str
        Name used in validation errors.
    value_type : type[Number]
        Callable converting each item.

    Returns
    -------
    tuple[Number, ...]
        Converted positive values.

    Raises
    ------
    ValueError
        If the value is empty or contains a non-positive item.
    """
    if not isinstance(value, (list, tuple)) or not value:
        msg = f"{field_name} must be a non-empty array"
        raise ValueError(msg)

    values = tuple(value_type(item) for item in value)
    if any(item <= 0 for item in values):
        msg = f"{field_name} must contain only positive values"
        raise ValueError(msg)
    return values


def _required_setting(settings: dict, field_name: str) -> Any:
    """
    Return a required benchmark setting with a useful validation error.

    Parameters
    ----------
    settings : dict
        Benchmark settings mapping.
    field_name : str
        Required key.

    Returns
    -------
    Any
        Setting value.

    Raises
    ------
    ValueError
        If ``field_name`` is absent.
    """
    if field_name not in settings:
        msg = f"missing required benchmark setting: {field_name}"
        raise ValueError(msg)
    return settings[field_name]


def _read_enum_values[EnumValue: Enum](
    value: object,
    field_name: str,
    enum_type: type[EnumValue],
) -> tuple[EnumValue, ...]:
    """
    Read a non-empty sequence of enum member names.

    Parameters
    ----------
    value : object
        Raw list or tuple of enum names.
    field_name : str
        Name used in validation errors.
    enum_type : type[EnumValue]
        Enum class used for conversion.

    Returns
    -------
    tuple[EnumValue, ...]
        Converted enum members.

    Raises
    ------
    ValueError
        If the value is empty or contains an unknown member name.
    """
    if not isinstance(value, (list, tuple)) or not value:
        msg = f"{field_name} must be a non-empty array of enum names"
        raise ValueError(msg)

    try:
        return tuple(enum_type[item] for item in value)
    except (KeyError, TypeError) as exc:
        valid_values = ", ".join(member.name for member in enum_type)
        msg = f"{field_name} must contain only valid {enum_type.__name__} names: {valid_values}"
        raise ValueError(msg) from exc


def load_config(config_path: Path) -> ExperimentConfig:
    """
    Load and validate an experiment configuration from TOML.

    Parameters
    ----------
    config_path : Path
        TOML configuration file.

    Returns
    -------
    ExperimentConfig
        Validated immutable benchmark configuration.

    Raises
    ------
    TypeError
        If the TOML structure is invalid.
    ValueError
        If a required setting is absent or invalid.
    """
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    settings = raw_config.get("benchmark", raw_config)
    if not isinstance(settings, dict):
        msg = "configuration must contain a [benchmark] table"
        raise TypeError(msg)

    k = _read_positive_values(
        _required_setting(settings, "k"),
        "k",
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
    bp_ranking_metrics = _read_enum_values(
        _required_setting(settings, "bp_ranking_metrics"),
        "bp_ranking_metrics",
        RankingMetric,
    )
    bp_init_strategies = _read_enum_values(
        _required_setting(settings, "bp_init_strategies"),
        "bp_init_strategies",
        InitStrategy,
    )
    bp_init_algorithms = _read_enum_values(
        _required_setting(settings, "bp_init_algorithms"),
        "bp_init_algorithms",
        InitAlgorithm,
    )

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
        k=k,
        n_inits=n_inits,
        subsample_size=subsample_size,
        bp_ranking_metrics=bp_ranking_metrics,
        bp_init_strategies=bp_init_strategies,
        bp_init_algorithms=bp_init_algorithms,
        run_regular=bool(_required_setting(settings, "run_regular")),
        run_hac_strength=bool(_required_setting(settings, "run_hac_strength")),
        run_special=bool(_required_setting(settings, "run_special")),
        include_cop_kmeans=bool(_required_setting(settings, "include_cop_kmeans")),
        include_hac=bool(_required_setting(settings, "include_hac")),
        skip_existing=bool(_required_setting(settings, "skip_existing")),
        include_bisecting_kmeans=bool(_required_setting(settings, "include_bisecting_kmeans")),
        include_bisecting_kmeans_m_rl=bool(
            _required_setting(settings, "include_bisecting_kmeans_m_rl")
        ),
        include_bp_kmeans=bool(settings.get("include_bp_kmeans", True)),
    )


def _config_for_json(config: ExperimentConfig) -> dict[str, object]:
    """
    Serialize an experiment configuration to JSON-compatible values.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration to serialize.

    Returns
    -------
    dict[str, object]
        Mapping containing strings, numbers, booleans, lists, and enum names.
    """
    serialized = asdict(config)
    serialized["datasets_dir"] = str(config.datasets_dir)
    serialized["benchmark_output_dir"] = str(config.benchmark_output_dir)
    serialized["analysis_output_dir"] = str(config.analysis_output_dir)
    serialized["k"] = list(config.k)
    serialized["n_inits"] = list(config.n_inits)
    serialized["bp_ranking_metrics"] = [metric.name for metric in config.bp_ranking_metrics]
    serialized["bp_init_strategies"] = [strategy.name for strategy in config.bp_init_strategies]
    serialized["bp_init_algorithms"] = [algorithm.name for algorithm in config.bp_init_algorithms]
    return serialized


def run_experiment(config: ExperimentConfig, *, config_name: str | None = None) -> None:
    """
    Run all benchmark stages selected by ``config``.

    Parameters
    ----------
    config : ExperimentConfig
        Benchmark configuration.
    config_name : str | None
        Name recorded in the experiment manifest.
    """
    config.benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config_file": config_name,
        "config": _config_for_json(config),
    }
    (config.benchmark_output_dir / "experiment_config.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    common_benchmark_args: dict[str, Any] = {
        "datasets_dir": config.datasets_dir,
        "output_dir": config.benchmark_output_dir,
        "seed": config.seed,
        "n_inits": config.n_inits,
        "subsample_size": config.subsample_size,
        "bp_ranking_metrics": config.bp_ranking_metrics,
        "bp_init_strategies": config.bp_init_strategies,
        "bp_init_algorithms": config.bp_init_algorithms,
        "include_cop_kmeans": config.include_cop_kmeans,
        "include_hac": config.include_hac,
        "skip_existing": config.skip_existing,
        "include_bisecting_kmeans": config.include_bisecting_kmeans,
        "include_bisecting_kmeans_m_rl": config.include_bisecting_kmeans_m_rl,
        "include_bp_kmeans": config.include_bp_kmeans,
    }

    if config.run_regular:
        run_benchmark(
            **common_benchmark_args,
            benchmark_type="regular",
            k_values=config.k,
        )

    if config.run_hac_strength:
        run_benchmark(
            **common_benchmark_args,
            benchmark_type="hac_strength",
            k_values=config.k,
        )

    if config.run_special:
        run_benchmark(
            **common_benchmark_args,
            benchmark_type="special",
            k_values=[200],
            dataset_filename="castile_and_leon_osm_drive_nodes.parquet",
            label_column="CPRO",
            log_label_name="provinces",
        )
        run_benchmark(
            **common_benchmark_args,
            benchmark_type="special",
            k_values=[10000],
            dataset_filename="com_madrid_osm_drive_nodes_split.parquet",
            label_column="CUSEC",
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
