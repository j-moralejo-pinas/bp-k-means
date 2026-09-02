"""Command-line orchestration for benchmark tables and publication figures."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd

from bp_k_means.tools.benchmark_analysis.data import (
    DATA_DIR,
    OUTPUT_DIR,
    RESULTS_DIR,
    SIZE_BIN_LABELS,
    add_dataset_context,
    aggregate_relative_metrics,
    compute_relative_metrics,
    load_all_metadata,
    load_hac_strength_metadata,
    parse_algorithm_components,
)
from bp_k_means.tools.benchmark_analysis.plots import (
    plot_by_k_multiplier,
    plot_by_size_bin,
    plot_overall,
)
from bp_k_means.tools.benchmark_analysis.plotting import (
    alg_label,
    build_color_map,
    build_fill_map,
    build_label_color_map,
    build_legend_info,
    build_marker_map,
    set_show_titles,
)
from bp_k_means.tools.benchmark_analysis.special import (
    SPECIAL_METRICS,
    analyze_special_metric,
    bp_algorithm_from_spec,
)
from bp_k_means.utils.logging import logger

SPECIAL_N_INIT = 4
SPECIAL_N_INIT_EXTENDED = 32
# Set to True to exclude HAC Ward runs from all analyses and plots
EXCLUDE_HAC = True
# ---------------------------------------------------------------------------
# Generic component-level analysis
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
    output_dir: Path = OUTPUT_DIR,
    data_dir: Path = DATA_DIR,
    results_dir: Path = RESULTS_DIR,
    show_titles: bool = False,
) -> None:
    """Run the regular relative analysis using only HAC-strength benchmark rows."""
    set_show_titles(show_titles=show_titles)

    df = cast("Any", load_hac_strength_metadata(output_dir))
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


def parse_n_init_list(value: str) -> list[int]:
    """Parse a comma-separated n_init list for CLI use."""
    try:
        n_inits = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        msg = "n_init values must be integers"
        raise argparse.ArgumentTypeError(msg) from exc
    if not n_inits:
        msg = "provide at least one n_init value"
        raise argparse.ArgumentTypeError(msg)
    if any(n < 1 for n in n_inits):
        msg = "n_init values must be >= 1"
        raise argparse.ArgumentTypeError(msg)
    return n_inits


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
        mapping[n_init] = bp_algorithm_from_spec(raw_spec)

    if not mapping:
        msg = "provide at least one BP combination entry"
        raise argparse.ArgumentTypeError(msg)
    return mapping


def parse_args() -> argparse.Namespace:
    """Parse command-line options for benchmark analysis."""
    parser = argparse.ArgumentParser(description="Analyze benchmark results.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Benchmark result directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Dataset directory (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Analysis output directory (default: {RESULTS_DIR}).",
    )
    parser.add_argument(
        "--special-n-inits",
        type=parse_n_init_list,
        default="1,32",
        help=(
            "Comma-separated n_init values for the special metric/time comparison "
            "plot. Defaults to 1 and the maximum available n_init."
        ),
    )
    parser.add_argument(
        "--special-bp-combinations",
        type=parse_bp_combination_map,
        default="1=M_C,I_CRI,KMEANS_PLUS_PLUS;32=M_RL,I_ACL,KMEANS_PLUS_PLUS",
        help=(
            "Semicolon-separated manual BP-KMeans combination per n_init, e.g. "
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


def _component_label(row: pd.Series, columns: list[str]) -> str:
    """Build a concise label for one BP-KMeans component grouping."""
    if columns == ["n_init"]:
        return f"n_init={row['n_init']}"
    return " x ".join(str(row[column]) for column in columns)


def _analyze_bp_components(
    df: pd.DataFrame,
    results_dir: Path,
    *,
    color_map: dict[str, tuple],
    marker_map: dict[str, str],
    fill_map: dict[str, bool],
    legend_info: dict,
) -> None:
    """Run the controlled BP-KMeans component comparisons used in the paper."""
    analyses = [
        (
            "vary_label_sel",
            ["ranking_metric"],
            {
                "init_strategy": "I_LRI",
                "init_algo": "KMEANS_PLUS_PLUS",
                "n_init": SPECIAL_N_INIT,
            },
        ),
        (
            "vary_reinit",
            ["init_strategy"],
            {
                "ranking_metric": "M_L",
                "init_algo": "KMEANS_PLUS_PLUS",
                "n_init": SPECIAL_N_INIT,
            },
        ),
        (
            "vary_init_algo",
            ["init_algo"],
            {
                "ranking_metric": "M_L",
                "init_strategy": "I_LRI",
                "n_init": SPECIAL_N_INIT,
            },
        ),
        (
            "vary_n_init",
            ["n_init"],
            {
                "ranking_metric": "M_L",
                "init_strategy": "I_LRI",
                "init_algo": "KMEANS_PLUS_PLUS",
            },
        ),
        (
            "vary_label_sel_x_reinit",
            ["ranking_metric", "init_strategy"],
            {"init_algo": "KMEANS_PLUS_PLUS", "n_init": SPECIAL_N_INIT},
        ),
        (
            "vary_label_sel_x_reinit_32",
            ["ranking_metric", "init_strategy"],
            {"init_algo": "KMEANS_PLUS_PLUS", "n_init": SPECIAL_N_INIT_EXTENDED},
        ),
        (
            "vary_label_sel_x_reinit_1",
            ["ranking_metric", "init_strategy"],
            {"init_algo": "KMEANS_PLUS_PLUS", "n_init": 1},
        ),
    ]
    plot_style = {
        "color_map": color_map,
        "marker_map": marker_map,
        "fill_map": fill_map,
        "legend_info": legend_info,
    }
    for directory, group_cols, fixed_values in analyses:
        selected = df
        for column, value in fixed_values.items():
            selected = selected[selected[column] == value]
        analyze_grouping(
            cast("pd.DataFrame", selected.copy()),
            group_cols=group_cols,
            label_fn=lambda row, columns=group_cols: _component_label(row, columns),
            save_dir=results_dir / directory,
            **plot_style,
        )


def main(
    special_n_inits: list[int] | None = None,
    special_bp_combinations: dict[int, str] | None = None,
    *,
    output_dir: Path = OUTPUT_DIR,
    data_dir: Path = DATA_DIR,
    results_dir: Path = RESULTS_DIR,
    show_titles: bool = False,
) -> None:
    """Run the complete benchmark aggregation and plotting pipeline."""
    set_show_titles(show_titles=show_titles)

    base_results_dir = results_dir / "base"
    base_results_dir.mkdir(parents=True, exist_ok=True)

    df = cast("Any", load_all_metadata(output_dir))
    if df.empty:
        logger.warning("No regular benchmark metadata found in %s", output_dir)
        return

    if EXCLUDE_HAC:
        df = df[~df["algorithm"].str.contains("HAC", case=False)].copy()

    df = compute_relative_metrics(add_dataset_context(df, data_dir))

    # Save full relative table
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

    df_bp = cast("Any", df[df["algorithm"].str.startswith("BP-KMeans")].copy())
    df_bp = cast("Any", parse_algorithm_components(df_bp))
    _analyze_bp_components(
        df_bp,
        results_dir,
        color_map=color_map,
        marker_map=marker_map,
        fill_map=fill_map,
        legend_info=legend_info,
    )

    # -----------------------------------------------------------------------
    # 7. Overall analysis restricted to KMEANS_PLUS_PLUS init_algo
    # -----------------------------------------------------------------------
    df_bp_parsed = cast("Any", parse_algorithm_components(df.copy()))
    df_kpp = cast(
        "Any",
        df_bp_parsed[
            (~df_bp_parsed["algorithm"].str.startswith("BP-KMeans"))
            | (df_bp_parsed["init_algo"] == "KMEANS_PLUS_PLUS")
        ].copy(),
    )
    analyze_grouping(
        cast("pd.DataFrame", df_kpp),
        group_cols=["algorithm", "n_init"],
        label_fn=alg_label,
        save_dir=results_dir / "kpp_only",
        **_ag_kwargs,
    )

    special_benchmarks = [
        (
            "com_madrid_osm_drive_nodes_split_split",
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
        for kpp_only in (False, True):
            suffix = "_kpp_only" if kpp_only else ""
            title_suffix = " (KMeans++)" if kpp_only else ""
            analyze_special_metric(
                dataset_prefix=dataset_prefix,
                metric_keys=SPECIAL_METRICS,
                save_dir=results_dir / f"{directory}{suffix}",
                title_prefix=f"{title}{title_suffix}",
                kpp_only=kpp_only,
                color_map=color_map,
                marker_map=marker_map,
                fill_map=fill_map,
                legend_info=legend_info,
                comparison_n_inits=special_n_inits,
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
    main(
        special_n_inits=args.special_n_inits,
        special_bp_combinations=args.special_bp_combinations,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        show_titles=args.show_titles,
    )


if __name__ == "__main__":
    cli()
