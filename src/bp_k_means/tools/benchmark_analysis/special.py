"""Analysis and plotting for the two special distance-metric benchmarks."""

import argparse
import re
from pathlib import Path
from typing import Any, cast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from bp_k_means.utils.logging import logger

from .data import (
    HAC_STRENGTH_BENCHMARK_TYPE,
    OUTPUT_DIR,
    parse_algorithm_components,
    read_metadata_files,
)
from .plotting import (
    add_scatter_legends,
    alg_label,
    apply_plain_tick_format,
    build_legend_info,
    draw_bar_chart,
    draw_scatter_plot,
    save_with_log_variant,
    set_title,
)

EXPECTED_COMBINATION_PARTS = 3
TIME_SHORT_LIMIT = 10
TIME_MEDIUM_LIMIT = 100
METRIC_SHORT_LIMIT = 1_000
METRIC_MEDIUM_LIMIT = 10_000
AXIS_SHORT_LIMIT = 100
AXIS_MEDIUM_LIMIT = 1_000
EXCLUDE_HAC = True


# ---------------------------------------------------------------------------
# Special-metric benchmarks (absolute, no aggregation by k_mult / size_bin)
# ---------------------------------------------------------------------------


SPECIAL_METRICS = [
    ("avg_dist_to_representative_node_m", "Mean distance to representative node (m)"),
    ("max_dist_to_representative_node_m", "Max distance to representative node (m)"),
    (
        "mean_max_dist_per_label_to_representative_node_m",
        "Mean max distance per label to representative node (m)",
    ),
]


def _load_special_metric_metadata(
    dataset_prefix: str,
    metric_keys: list[str],
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Scan output/ for special metric metadata rows containing all requested keys."""
    rows = []
    for meta in read_metadata_files(output_dir):
        if meta.get("benchmark_type") == HAC_STRENGTH_BENCHMARK_TYPE:
            continue
        if not meta.get("dataset", "").startswith(dataset_prefix):
            continue
        if EXCLUDE_HAC and "HAC" in meta.get("algorithm", "").upper():
            continue
        if not all(k in meta for k in metric_keys):
            continue
        row = {
            "dataset": meta["dataset"],
            "algorithm": meta["algorithm"],
            "n_init": int(meta["n_init"]),
            "k_multiplier": float(meta["k_multiplier"]),
            "k": int(meta["k"]),
            "n_clusters": int(meta["n_clusters"]),
            "time": float(meta["duration_seconds"]),
        }
        for k in metric_keys:
            row[k] = float(meta[k])
        rows.append(row)
    return pd.DataFrame(rows)


def _resolve_special_n_inits(df: pd.DataFrame, requested_n_inits: list[int] | None) -> list[int]:
    available = sorted(int(v) for v in df["n_init"].dropna().unique())
    if not available:
        return []
    values = [1, available[-1]] if requested_n_inits is None else requested_n_inits
    resolved = sorted({int(v) for v in values if int(v) in available})
    missing = sorted({int(v) for v in values if int(v) not in available})
    if missing:
        logger.warning("Ignoring unavailable n_init values: %s", missing)
    return resolved


def _time_comparison_number(value: float, _pos: int | None = None) -> str:
    """Format comparison time values with precision based on their magnitude."""
    if not np.isfinite(value):
        return ""
    abs_value = abs(value)
    if abs_value < TIME_SHORT_LIMIT:
        return f"{value:.2f}"
    if abs_value < TIME_MEDIUM_LIMIT:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _metric_comparison_number(value: float, _pos: int | None = None) -> str:
    """Format comparison metric values with precision based on their magnitude."""
    if not np.isfinite(value):
        return ""
    abs_value = abs(value)
    if abs_value < METRIC_SHORT_LIMIT:
        return f"{value:.2f}"
    if abs_value < METRIC_MEDIUM_LIMIT:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _special_metric_axis_step(separation: float) -> int:
    if separation < AXIS_SHORT_LIMIT:
        return 25
    if separation < AXIS_MEDIUM_LIMIT:
        return 250
    return 2_500


def _metric_axis_bounds(values: pd.Series) -> tuple[float, float] | None:
    finite_values: Any = values[np.isfinite(values)]
    if finite_values.empty:
        return None

    min_value = finite_values.min()
    max_value = finite_values.max()
    step = _special_metric_axis_step(max_value - min_value)
    low = np.floor(min_value / step) * step
    high = np.ceil(max_value / step) * step
    if low >= min_value:
        low -= step
    if high <= max_value:
        high += step
    if low == high:
        high = low + step
    return float(low), float(high)


def bp_algorithm_from_spec(spec: str) -> str:
    """Expand a compact BP-KMeans component specification."""
    spec = spec.strip()
    if spec.startswith("BP-KMeans"):
        return spec

    parts = [part.strip() for part in re.split(r"[,/]", spec) if part.strip()]
    if len(parts) != EXPECTED_COMBINATION_PARTS:
        msg = (
            "BP-KMeans combination must be either a full algorithm name or "
            "label_selection,reinit_method,init_algo"
        )
        raise argparse.ArgumentTypeError(msg)
    return f"BP-KMeans ({parts[0]}, {parts[1]}, {parts[2]})"


def _best_bp_comparison_row(
    bp_rows: pd.DataFrame,
    key: str,
) -> pd.Series:
    bp_agg = (
        bp_rows.groupby("algorithm")
        .agg(**{f"mean_{key}": (key, "mean"), "mean_time": ("time", "mean")})
        .reset_index()
        .sort_values(f"mean_{key}")
    )
    return cast("pd.Series", bp_agg.iloc[0])


def _manual_bp_comparison_row(
    bp_rows: pd.DataFrame,
    n_init: int,
    key: str,
    manual_bp_algorithms: dict[int, str] | None,
) -> pd.Series | None:
    manual_algorithm = (manual_bp_algorithms or {}).get(n_init)
    if manual_algorithm is None:
        return None

    manual_rows = bp_rows[bp_rows["algorithm"] == manual_algorithm]
    if manual_rows.empty:
        return None

    agg: Any = manual_rows[[key, "time"]].mean()
    return pd.Series(
        {
            "algorithm": manual_algorithm,
            f"mean_{key}": agg[key],
            "mean_time": agg["time"],
        }
    )


def _comparison_record(
    n_init: int,
    kind: str,
    algorithm: str,
    values: pd.Series,
    key: str,
) -> dict[str, Any]:
    return {
        "n_init": n_init,
        "kind": kind,
        "algorithm": algorithm,
        f"mean_{key}": float(values[f"mean_{key}"] if f"mean_{key}" in values else values[key]),
        "mean_time": float(values["mean_time"] if "mean_time" in values else values["time"]),
    }


def _plot_special_metric_time_comparison(
    df: pd.DataFrame,
    key: str,
    metric_label: str,
    save_dir: Path,
    title_prefix: str,
    *,
    n_inits: list[int] | None = None,
    manual_bp_algorithms: dict[int, str] | None = None,
) -> None:
    """Compare Bisecting KMeans with per-problem and globally tuned BP-KMeans rows."""
    df = cast("Any", df)
    requested_n_inits = n_inits
    if requested_n_inits is None and manual_bp_algorithms:
        requested_n_inits = sorted(manual_bp_algorithms)
    selected_n_inits = _resolve_special_n_inits(df, requested_n_inits)
    if not selected_n_inits:
        return

    rows = []
    for n_init in selected_n_inits:
        base_rows = cast(
            "pd.DataFrame",
            df[(df["algorithm"] == "Bisecting KMeans") & (df["n_init"] == n_init)],
        )
        bp_rows = cast(
            "pd.DataFrame",
            df[(df["algorithm"].str.startswith("BP-KMeans")) & (df["n_init"] == n_init)],
        )
        if base_rows.empty or bp_rows.empty:
            continue

        base_agg: Any = base_rows[[key, "time"]].mean()
        rows.append(
            _comparison_record(n_init, "Bisecting K-Means", "Bisecting KMeans", base_agg, key)
        )

        best_bp: Any = _best_bp_comparison_row(bp_rows, key)
        rows.append(
            _comparison_record(
                n_init, "BP-KMeans - per-problem best", best_bp["algorithm"], best_bp, key
            )
        )

        manual_bp: Any = _manual_bp_comparison_row(
            bp_rows,
            n_init,
            key,
            manual_bp_algorithms,
        )
        if manual_bp is not None:
            rows.append(
                _comparison_record(
                    n_init,
                    "BP-KMeans - globally tuned",
                    manual_bp["algorithm"],
                    manual_bp,
                    key,
                )
            )

    comp = pd.DataFrame(rows)
    if comp.empty:
        return

    safe_key = key.replace("/", "_")
    comp.to_csv(save_dir / f"{safe_key}_time_comparison.csv", index=False)

    plotted_n_inits = sorted(int(v) for v in comp["n_init"].unique())
    n_groups = len(plotted_n_inits)
    x = np.arange(n_groups)
    width = 0.11
    kinds = [
        "Bisecting K-Means",
        "BP-KMeans - per-problem best",
        "BP-KMeans - globally tuned",
    ]
    plotted_kinds = [kind for kind in kinds if kind in set(comp["kind"])]
    colors = {
        "Bisecting K-Means": "#4C78A8",
        "BP-KMeans - per-problem best": "#F58518",
        "BP-KMeans - globally tuned": "#54A24B",
    }
    fig, ax_metric = plt.subplots(figsize=(max(8, n_groups * 2.6), 5.5))
    ax_time = ax_metric.twinx()

    for index, kind in enumerate(plotted_kinds):
        sub = comp[comp["kind"] == kind].set_index("n_init").reindex(plotted_n_inits)
        metric_bars = ax_metric.bar(
            x + (index - 2.75) * width,
            sub[f"mean_{key}"],
            width,
            label=f"{kind} metric",
            color=colors[kind],
            edgecolor="white",
        )
        time_bars = ax_time.bar(
            x + (index + 0.75) * width,
            sub["mean_time"],
            width,
            label=f"{kind} time",
            color=colors[kind],
            edgecolor="white",
            hatch="//",
            alpha=0.45,
        )
        ax_metric.bar_label(
            metric_bars,
            labels=[_metric_comparison_number(v) for v in sub[f"mean_{key}"]],
            padding=2,
            fontsize=7,
            rotation=90,
        )
        ax_time.bar_label(
            time_bars,
            labels=[_time_comparison_number(v) for v in sub["mean_time"]],
            padding=2,
            fontsize=7,
            rotation=90,
        )

    metric_bounds = _metric_axis_bounds(cast("pd.Series", comp[f"mean_{key}"]))
    if metric_bounds is not None:
        ax_metric.set_ylim(metric_bounds)
    ax_time.set_yscale("log")
    ax_metric.set_ylabel(metric_label)
    ax_time.set_ylabel("Mean Time")
    ax_metric.set_xlabel("Number of Initializations")
    ax_metric.set_xticks(x)
    ax_metric.set_xticklabels([str(v) for v in plotted_n_inits])
    set_title(ax_metric, f"{title_prefix} - {metric_label} and time comparison")
    ax_metric.yaxis.set_major_formatter(mticker.FuncFormatter(_metric_comparison_number))
    ax_time.yaxis.set_major_formatter(mticker.FuncFormatter(_time_comparison_number))
    ax_time.yaxis.set_minor_formatter(mticker.NullFormatter())

    legend_handles = [
        *(mpatches.Patch(facecolor=colors[kind], edgecolor="white") for kind in plotted_kinds),
        mpatches.Patch(facecolor="white", edgecolor="black"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", alpha=0.45),
    ]
    legend_labels = [*plotted_kinds, metric_label, "Time"]
    ax_metric.legend(
        legend_handles,
        legend_labels,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.08, 1.0),
    )
    fig.tight_layout()
    fig.savefig(save_dir / f"{safe_key}_time_comparison.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_tradeoff(
    aggregate: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    title_prefix: str,
    save_path: Path,
    color_map: dict[str, tuple] | None,
    marker_map: dict[str, str] | None,
    fill_map: dict[str, bool] | None,
    legend_info: dict | None,
    *,
    pareto: bool,
) -> None:
    """Plot one special metric against runtime."""
    fig, ax = plt.subplots(figsize=(10, 7))
    title_suffix = " Pareto front" if pareto else " vs time"
    draw_scatter_plot(
        ax,
        aggregate,
        color_map,
        marker_map,
        fill_map,
        x_col="mean_time",
        y_col=metric_col,
        x_label="Mean time (s)",
        y_label=metric_label,
        title=f"{title_prefix} - {metric_label}{title_suffix}",
        pareto=pareto,
        floor_at_one=not pareto,
        add_reference_lines=False,
    )
    if legend_info is not None:
        add_scatter_legends(ax, legend_info, has_pareto_line=pareto)
    fig.tight_layout()
    save_with_log_variant(fig, [ax], save_path, log_x_axis=True, log_y_axis=False)


def analyze_special_metric(
    dataset_prefix: str,
    metric_keys: list[tuple[str, str]],
    save_dir: Path,
    title_prefix: str,
    *,
    kpp_only: bool = False,
    exclude_r_erc: bool = False,
    color_map: dict[str, tuple] | None = None,
    marker_map: dict[str, str] | None = None,
    fill_map: dict[str, bool] | None = None,
    legend_info: dict | None = None,
    comparison_n_inits: list[int] | None = None,
    comparison_bp_algorithms: dict[int, str] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Analyze special metrics in absolute terms per algorithm x n_init.

    No relative normalization, no breakdown by k_multiplier or size bin. *metric_keys* is a list of
    (key, label) pairs, each producing its own bar chart, scatter and Pareto plot. The primary
    metric (first entry) is also used as the Y axis of the shared scatter/Pareto.
    """
    keys = [k for k, _ in metric_keys]
    df = cast("Any", _load_special_metric_metadata(dataset_prefix, keys, output_dir))
    if df.empty:
        return

    if kpp_only:
        df_bp_parsed = parse_algorithm_components(df)
        df = cast(
            "Any",
            df_bp_parsed[
                (~df_bp_parsed["algorithm"].str.startswith("BP-KMeans"))
                | (df_bp_parsed["init_algo"] == "KMEANS_PLUS_PLUS")
            ].copy(),
        )
        if exclude_r_erc:
            df = cast(
                "Any",
                df[
                    (~df["algorithm"].str.startswith("BP-KMeans"))
                    | (df["label_selection_method"] != "R_ERC")
                ].copy(),
            )
        if df.empty:
            return

    save_dir.mkdir(parents=True, exist_ok=True)
    df["label"] = df.apply(alg_label, axis=1)
    local_legend_info = (
        build_legend_info(cast("pd.DataFrame", df), color_map, marker_map, fill_map)
        if legend_info is not None
        and color_map is not None
        and marker_map is not None
        and fill_map is not None
        else legend_info
    )

    agg_dict = {"mean_time": ("time", "mean")}
    for key in keys:
        agg_dict[f"mean_{key}"] = (key, "mean")
    agg = df.groupby(["algorithm", "n_init", "label"]).agg(**agg_dict).reset_index()

    agg.to_csv(save_dir / "overall_avg.csv", index=False)

    orig_labels = agg.sort_values(f"mean_{keys[0]}")["label"].tolist()
    for key, metric_label in metric_keys:
        metric_col = f"mean_{key}"
        sub = agg.set_index("label").loc[orig_labels].reset_index()

        # Bar chart
        n_bars = len(sub)
        fig, ax = plt.subplots(figsize=(9, max(4, n_bars * 0.35)))
        draw_bar_chart(ax, sub, metric_col, color_map)
        ax.set_xlabel(metric_label)
        set_title(ax, f"{title_prefix} - {metric_label}")
        apply_plain_tick_format(ax.xaxis)
        fig.tight_layout()
        safe_key = key.replace("/", "_")
        fig.savefig(save_dir / f"{safe_key}.png", dpi=450, bbox_inches="tight")
        plt.close(fig)

        for pareto, suffix in ((False, "scatter"), (True, "pareto")):
            _plot_metric_tradeoff(
                agg,
                metric_col,
                metric_label,
                title_prefix,
                save_dir / f"{safe_key}_{suffix}.png",
                color_map,
                marker_map,
                fill_map,
                local_legend_info,
                pareto=pareto,
            )

        _plot_special_metric_time_comparison(
            cast("pd.DataFrame", df),
            key,
            metric_label,
            save_dir,
            title_prefix,
            n_inits=comparison_n_inits,
            manual_bp_algorithms=comparison_bp_algorithms,
        )
