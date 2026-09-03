"""Command-line orchestration for benchmark tables and publication figures."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd

from bp_k_means.benchmark_analysis.data import (
    SIZE_BIN_LABELS,
    add_dataset_context,
    aggregate_relative_metrics,
    compute_relative_metrics,
    load_all_metadata,
    load_hac_strength_metadata,
    select_bp_vs_bisecting_kmeans,
)
from bp_k_means.benchmark_analysis.plots import (
    plot_by_k_multiplier,
    plot_by_size_bin,
    plot_overall,
)
from bp_k_means.benchmark_analysis.plotting import (
    alg_label,
    build_color_map,
    build_fill_map,
    build_label_color_map,
    build_legend_info,
    build_marker_map,
    set_show_titles,
)
from bp_k_means.benchmark_analysis.special import (
    SPECIAL_METRICS,
    analyze_special_metric,
    bp_algorithm_from_spec,
)
from bp_k_means.main import ExperimentConfig, load_config
from bp_k_means.utils.logging import logger

# ---------------------------------------------------------------------------
# Generic analysis
# ---------------------------------------------------------------------------


def analyze_grouping(
    df: pd.DataFrame,
    group_cols: list[str],
    label_fn: Callable[[pd.Series], str],
    save_dir: Path,
    color_map: dict[str, tuple] | None = None,
    marker_map: dict[str, str] | None = None,
    fill_map: dict[str, bool] | None = None,
    legend_info: dict | None = None,
) -> None:
    """Write tables and plots for a grouping, normalized within its algorithm subset."""
    df = cast("Any", df)
    save_dir.mkdir(parents=True, exist_ok=True)

    # The baseline must reflect only the algorithms in this filtered subset.
    df = compute_relative_metrics(df)

    # Overall average
    overall = aggregate_relative_metrics(df, group_cols).sort_values("mean_relative_wcss")
    overall["label"] = overall.apply(label_fn, axis=1)
    overall.to_csv(save_dir / "overall_avg.csv", index=False)

    # By k_multiplier
    by_k_mult = aggregate_relative_metrics(df, [*group_cols, "k_multiplier"])
    by_k_mult["label"] = by_k_mult.apply(label_fn, axis=1)
    by_k_mult.to_csv(save_dir / "by_k_multiplier.csv", index=False)

    # By size bin
    by_size = aggregate_relative_metrics(df, [*group_cols, "size_bin"])
    by_size["label"] = by_size.apply(label_fn, axis=1)
    by_size["size_bin"] = pd.Categorical(
        by_size["size_bin"], categories=SIZE_BIN_LABELS, ordered=True
    )
    by_size = by_size.sort_values(["size_bin", *group_cols])
    by_size.to_csv(save_dir / "by_size_bin.csv", index=False)

    # Build local maps keyed by the simplified label (from label_fn).
    # For each group, pick a representative raw row and look up the global maps
    # via the full "algorithm | n_init=..." key so encoding is consistent.
    local_color_map: dict[str, tuple] = {}
    local_marker_map: dict[str, str] = {}
    local_fill_map: dict[str, bool] = {}
    for _grp_vals, grp_df in df.groupby(group_cols, sort=False):
        rep_row = grp_df.iloc[0]
        simplified_label = label_fn(rep_row)
        full_key = f"{rep_row['algorithm']} | n_init={rep_row['n_init']}"
        if color_map is not None:
            local_color_map[simplified_label] = color_map.get(full_key, (0.5, 0.5, 0.5))
        if marker_map is not None:
            local_marker_map[simplified_label] = marker_map.get(full_key, "o")
        if fill_map is not None:
            local_fill_map[simplified_label] = fill_map.get(full_key, True)

    _color_map = (
        local_color_map
        if color_map is not None
        else build_label_color_map(overall["label"].tolist())
    )
    _marker_map = local_marker_map if marker_map is not None else None
    _fill_map = local_fill_map if fill_map is not None else None
    _legend_info = (
        build_legend_info(df, color_map, marker_map, fill_map)
        if legend_info is not None
        and color_map is not None
        and marker_map is not None
        and fill_map is not None
        else legend_info
    )

    plot_overall(
        overall,
        color_map=_color_map,
        marker_map=_marker_map,
        fill_map=_fill_map,
        legend_info=_legend_info,
        save_dir=save_dir,
    )
    plot_by_k_multiplier(
        by_k_mult,
        color_map=_color_map,
        marker_map=_marker_map,
        fill_map=_fill_map,
        legend_info=_legend_info,
        save_dir=save_dir,
    )
    plot_by_size_bin(
        by_size,
        color_map=_color_map,
        marker_map=_marker_map,
        fill_map=_fill_map,
        legend_info=_legend_info,
        save_dir=save_dir,
    )


def analyze_hac_strength_benchmark(
    *,
    output_dir: Path,
    data_dir: Path,
    results_dir: Path,
    show_titles: bool = False,
) -> None:
    """Analyze HAC-strength rows for BP-KMeans++ and standard Bisecting KMeans."""
    set_show_titles(show_titles=show_titles)

    df = cast("Any", select_bp_vs_bisecting_kmeans(load_hac_strength_metadata(output_dir)))
    if df.empty:
        return

    save_dir = results_dir / "hac_strength"
    save_dir.mkdir(parents=True, exist_ok=True)

    rel = compute_relative_metrics(add_dataset_context(df, data_dir))

    rel_cols = [
        "dataset",
        "n_instances",
        "algorithm",
        "n_init",
        "k_multiplier",
        "requested_cluster_multiplier",
        "requested_n_clusters",
        "target_k_was_capped",
        "k",
        "n_clusters",
        "wcss",
        "time",
        "best_wcss",
        "best_time",
        "relative_wcss",
        "relative_time",
    ]
    cast("pd.DataFrame", rel[rel_cols]).sort_values(["dataset", "algorithm", "n_init"]).to_csv(
        save_dir / "relative_metrics.csv", index=False
    )

    min_lightness = 0.25
    max_lightness = 0.8
    color_map = build_color_map(rel, min_lightness=min_lightness, max_lightness=max_lightness)
    marker_map = build_marker_map(rel)
    fill_map = build_fill_map(rel)
    legend_info = build_legend_info(
        rel,
        color_map,
        marker_map,
        fill_map,
        min_lightness=min_lightness,
        max_lightness=max_lightness,
    )

    analyze_grouping(
        rel,
        group_cols=["algorithm", "n_init"],
        label_fn=alg_label,
        save_dir=save_dir,
        color_map=color_map,
        marker_map=marker_map,
        fill_map=fill_map,
        legend_info=legend_info,
    )


def parse_bp_combination_map(value: str) -> dict[int, str]:
    """Parse n_init-to-BP-KMeans-combination mapping for CLI use."""
    mapping: dict[int, str] = {}
    for raw_entry in value.split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            msg = "BP combinations must use n_init=combination entries"
            raise argparse.ArgumentTypeError(msg)
        raw_n_init, raw_spec = entry.split("=", 1)
        try:
            n_init = int(raw_n_init.strip())
        except ValueError as exc:
            msg = "BP combination n_init keys must be integers"
            raise argparse.ArgumentTypeError(msg) from exc
        if n_init < 1:
            msg = "BP combination n_init keys must be >= 1"
            raise argparse.ArgumentTypeError(msg)
        algorithm = bp_algorithm_from_spec(raw_spec)
        if not algorithm.endswith("KMEANS_PLUS_PLUS)"):
            msg = "BP combinations must use KMEANS_PLUS_PLUS initialization"
            raise argparse.ArgumentTypeError(msg)
        mapping[n_init] = algorithm

    if not mapping:
        msg = "provide at least one BP combination entry"
        raise argparse.ArgumentTypeError(msg)
    return mapping


def parse_args() -> argparse.Namespace:
    """Parse command-line options for benchmark analysis."""
    parser = argparse.ArgumentParser(description="Analyze benchmark results.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/default.toml"),
        help="TOML configuration file (default: experiments/default.toml).",
    )
    parser.add_argument(
        "--special-bp-combinations",
        type=parse_bp_combination_map,
        default="1=M_C,I_CRI,KMEANS_PLUS_PLUS;32=M_RL,I_ACL,KMEANS_PLUS_PLUS",
        help=(
            "Semicolon-separated manual BP-KMeans combination per plotted n_init; the map keys "
            "also select the n_init values, e.g. "
            "'1=M_L,I_LRI,KMEANS_PLUS_PLUS;32=M_C,I_ACC,KMEANS_PLUS_PLUS'. "
            "These combinations are plotted as BP-KMeans - globally tuned. "
            "The per-problem best BP-KMeans combination is always chosen separately "
            "for each special metric and n_init."
        ),
    )
    parser.add_argument(
        "--show-titles",
        action="store_true",
        help="Show plot titles. Titles are hidden by default.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: ExperimentConfig,
    special_bp_combinations: dict[int, str] | None = None,
    *,
    show_titles: bool = False,
) -> None:
    """Run the complete benchmark aggregation and plotting pipeline."""
    set_show_titles(show_titles=show_titles)

    output_dir = config.benchmark_output_dir
    data_dir = config.datasets_dir
    results_dir = config.analysis_output_dir
    base_results_dir = results_dir / "base"
    base_results_dir.mkdir(parents=True, exist_ok=True)

    df = cast("Any", select_bp_vs_bisecting_kmeans(load_all_metadata(output_dir)))
    if df.empty:
        logger.warning("No regular benchmark metadata found in %s", output_dir)
        return

    df = compute_relative_metrics(add_dataset_context(df, data_dir))

    # Save the relative table for the requested BP-vs-Bisecting comparison.
    rel_cols = [
        "dataset",
        "n_instances",
        "algorithm",
        "n_init",
        "k_multiplier",
        "k",
        "n_clusters",
        "wcss",
        "time",
        "best_wcss",
        "best_time",
        "relative_wcss",
        "relative_time",
    ]
    cast("pd.DataFrame", df[rel_cols]).sort_values(
        ["dataset", "k_multiplier", "algorithm", "n_init"]
    ).to_csv(base_results_dir / "relative_metrics.csv", index=False)

    min_lightness = 0.25
    max_lightness = 0.8
    color_map = build_color_map(df)
    marker_map = build_marker_map(df)
    fill_map = build_fill_map(df)
    legend_info = build_legend_info(
        df,
        color_map,
        marker_map,
        fill_map,
        min_lightness=min_lightness,
        max_lightness=max_lightness,
    )
    _ag_kwargs = {
        "color_map": color_map,
        "marker_map": marker_map,
        "fill_map": fill_map,
        "legend_info": legend_info,
    }
    analyze_grouping(
        df,
        group_cols=["algorithm", "n_init"],
        label_fn=alg_label,
        save_dir=base_results_dir,
        **_ag_kwargs,
    )

    special_benchmarks = [
        (
            "com_madrid_osm_drive_nodes_split",
            "com_madrid_distance_metrics",
            "Community of Madrid",
        ),
        (
            "castile_and_leon_osm_drive_nodes",
            "castile_leon_distance_metrics",
            "Castile and León",
        ),
    ]
    for dataset_prefix, directory, title in special_benchmarks:
        analyze_special_metric(
            dataset_prefix=dataset_prefix,
            metric_keys=SPECIAL_METRICS,
            save_dir=results_dir / directory,
            title_prefix=f"{title} (KMeans++)",
            kpp_only=True,
            color_map=color_map,
            marker_map=marker_map,
            fill_map=fill_map,
            legend_info=legend_info,
            comparison_bp_algorithms=special_bp_combinations,
            output_dir=output_dir,
        )

    analyze_hac_strength_benchmark(
        output_dir=output_dir,
        data_dir=data_dir,
        results_dir=results_dir,
        show_titles=show_titles,
    )


def cli() -> None:
    """Run the benchmark analysis command-line interface."""
    args = parse_args()
    try:
        config = load_config(args.config)
        main(
            config,
            special_bp_combinations=args.special_bp_combinations,
            show_titles=args.show_titles,
        )
    except (OSError, TypeError, ValueError) as exc:
        msg = f"error: {exc}"
        raise SystemExit(msg) from exc


if __name__ == "__main__":
    cli()
