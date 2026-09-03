"""Complete figure suites for the relative benchmark analyses."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bp_k_means.benchmark_analysis.data import SIZE_BIN_LABELS
from bp_k_means.benchmark_analysis.plotting import (
    PLOT_OPTIONS,
    add_scatter_legends,
    alg_label,
    apply_plain_tick_format,
    bold_pareto_ticks,
    compute_pareto_front,
    create_panel_grid,
    draw_bar_chart,
    draw_scatter_plot,
    pivot_for_line,
    plot_bar_chart,
    save_with_log_variant,
    set_suptitle,
    set_title,
    to_math_label,
)


def plot_overall(
    overall: pd.DataFrame,
    save_dir: Path,
    color_map: dict[str, tuple] | None = None,
    marker_map: dict[str, str] | None = None,
    fill_map: dict[str, bool] | None = None,
    legend_info: dict | None = None,
) -> None:
    """Create aggregate bar, line, scatter, and Pareto plots."""
    overall = cast("Any", overall)
    overall = overall.copy()
    if "label" not in overall.columns:
        overall["label"] = overall.apply(alg_label, axis=1)

    pareto_mask_overall = compute_pareto_front(
        overall["mean_relative_time"].tolist(),
        overall["mean_relative_wcss"].tolist(),
    )
    pareto_labels_overall: set[str] = {
        str(row["label"]) for i, (_, row) in enumerate(overall.iterrows()) if pareto_mask_overall[i]
    }

    for metric, fname, title in [
        ("mean_relative_wcss", "overall_avg_wcss.png", "Mean Relative WCSS - overall"),
        ("mean_relative_time", "overall_avg_time.png", "Mean Relative Time - overall"),
    ]:
        plot_bar_chart(
            overall,
            metric=metric,
            title=title,
            xlabel=metric,
            save_path=save_dir / fname,
            color_map=color_map,
            pareto_labels=pareto_labels_overall,
        )

    # Combined figure
    overall_sorted_wcss = overall.sort_values("mean_relative_wcss")
    overall_sorted_time = overall.sort_values("mean_relative_time")
    n = len(overall)
    fig, axes = plt.subplots(1, 2, figsize=(18, max(5, n * 0.35)))
    for ax, sdf, metric, title in [
        (axes[0], overall_sorted_wcss, "mean_relative_wcss", "Mean Relative WCSS"),
        (axes[1], overall_sorted_time, "mean_relative_time", "Mean Relative Time"),
    ]:
        draw_bar_chart(
            ax,
            sdf,
            metric,
            color_map,
            pareto_labels_overall,
            reference_label="best=1.0",
        )
        ax.set_xlabel(metric)
        set_title(ax, title + " (overall)", force=True)
        ax.legend(fontsize=8, loc="upper right")
        apply_plain_tick_format(ax.xaxis)
    fig.tight_layout()
    bold_pareto_ticks(list(axes))
    fig.savefig(save_dir / "overall_avg.png", dpi=450, bbox_inches="tight")
    plt.close(fig)

    # Scatter: relative WCSS vs relative time
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_scatter_plot(
        ax,
        overall,
        color_map,
        marker_map,
        fill_map,
        title="WCSS vs Time trade-off (overall)",
    )
    if legend_info is not None:
        add_scatter_legends(ax, legend_info)
    fig.tight_layout()
    save_with_log_variant(fig, [ax], save_dir / "overall_scatter.png")

    # Scatter with Pareto front (overall)
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_scatter_plot(
        ax,
        overall,
        color_map,
        marker_map,
        fill_map,
        title="WCSS vs Time - Pareto front (overall)",
        pareto=True,
    )
    if legend_info is not None:
        add_scatter_legends(ax, legend_info, has_pareto_line=True)
    fig.tight_layout()
    save_with_log_variant(fig, [ax], save_dir / "overall_pareto.png")


def _group_rows(df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    """Return the rows belonging to one plot panel."""
    return cast("pd.DataFrame", df[df[column] == value].copy())


def _pareto_labels_by_group(
    df: pd.DataFrame, group_col: str, groups: list[Any]
) -> dict[Any, set[str]]:
    """Collect Pareto-optimal labels for every panel."""
    result = {}
    for group in groups:
        rows = _group_rows(df, group_col, group)
        mask = compute_pareto_front(
            rows["mean_relative_time"].tolist(), rows["mean_relative_wcss"].tolist()
        )
        result[group] = {
            str(row["label"]) for index, (_, row) in enumerate(rows.iterrows()) if mask[index]
        }
    return result


def _plot_grouped_bars(
    df: pd.DataFrame,
    group_col: str,
    groups: list[Any],
    metric: str,
    title: str,
    panel_title: Callable[[Any], str],
    save_path: Path,
    color_map: dict[str, tuple] | None,
) -> None:
    pareto_labels = _pareto_labels_by_group(df, group_col, groups)
    fig, axes = create_panel_grid(
        len(groups),
        width_per_col=7,
        height_per_row=max(5, df["label"].nunique() * 0.35),
    )
    for ax, group in zip(axes, groups, strict=False):
        rows = _group_rows(df, group_col, group).sort_values(metric)
        draw_bar_chart(
            ax,
            rows,
            metric,
            color_map,
            pareto_labels[group],
            reference_label="best=1.0",
        )
        set_title(ax, panel_title(group), force=True)
        ax.set_xlabel(metric)
        ax.legend(fontsize=8, loc="upper right")
        apply_plain_tick_format(ax.xaxis)

    set_suptitle(fig, title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95] if PLOT_OPTIONS["show_titles"] else None)
    bold_pareto_ticks(list(axes))
    fig.savefig(save_path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_line(
    df: pd.DataFrame,
    group_col: str,
    groups: list[Any],
    metric: str,
    title: str,
    x_label: str,
    save_path: Path,
    color_map: dict[str, tuple] | None,
    marker_map: dict[str, str] | None,
    *,
    set_x_ticks: bool,
) -> None:
    pivot = pivot_for_line(df, x_col=group_col, metric=metric)[groups]
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, row in pivot.iterrows():
        values = np.asarray([row.get(group, np.nan) for group in groups], dtype=float)
        label_str = str(label)
        color = color_map.get(label_str) if color_map else None
        plot_kwargs = {
            "marker": marker_map.get(label_str, "o") if marker_map else "o",
            "label": to_math_label(label_str),
            "linewidth": 1.5,
            "markersize": 5,
        }
        if color is not None:
            plot_kwargs["color"] = color
        ax.plot(groups, values, **plot_kwargs)

    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.2, label="best=1.0")
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric)
    set_title(ax, title + " (line)")
    if set_x_ticks:
        ax.set_xticks(groups)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_scatter(
    df: pd.DataFrame,
    group_col: str,
    groups: list[Any],
    title: str,
    panel_title: Callable[[Any], str],
    save_path: Path,
    color_map: dict[str, tuple] | None,
    marker_map: dict[str, str] | None,
    fill_map: dict[str, bool] | None,
    legend_info: dict | None,
    *,
    pareto: bool,
) -> None:
    fig, axes = create_panel_grid(len(groups), width_per_col=8, height_per_row=6)
    for ax, group in zip(axes, groups, strict=False):
        draw_scatter_plot(
            ax,
            _group_rows(df, group_col, group),
            color_map,
            marker_map,
            fill_map,
            title=panel_title(group),
            force_title=True,
            pareto=pareto,
        )
        if legend_info is not None:
            add_scatter_legends(
                ax,
                legend_info,
                has_pareto_line=pareto,
                loc="upper right",
                bbox_to_anchor=(0.99, 0.99),
            )
    set_suptitle(fig, title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95] if PLOT_OPTIONS["show_titles"] else None)
    save_with_log_variant(fig, list(axes), save_path)


def _plot_by_group(
    df: pd.DataFrame,
    group_col: str,
    groups: list[Any],
    file_prefix: str,
    title_suffix: str,
    line_x_label: str,
    bar_panel_title: Callable[[Any], str],
    scatter_panel_title: Callable[[Any], str],
    color_map: dict[str, tuple] | None,
    marker_map: dict[str, str] | None,
    fill_map: dict[str, bool] | None,
    legend_info: dict | None,
    save_dir: Path,
    *,
    set_line_x_ticks: bool,
) -> None:
    """Create bars, lines, scatter plots, and Pareto plots for a grouping."""
    if not groups:
        return
    df = df.copy()
    if "label" not in df.columns:
        df["label"] = df.apply(alg_label, axis=1)

    for metric, short_name, metric_title in (
        ("mean_relative_wcss", "wcss", "Mean Relative WCSS"),
        ("mean_relative_time", "time", "Mean Relative Time"),
    ):
        title = f"{metric_title} by {title_suffix}"
        path = save_dir / f"{file_prefix}_{short_name}.png"
        _plot_grouped_bars(df, group_col, groups, metric, title, bar_panel_title, path, color_map)
        _plot_grouped_line(
            df,
            group_col,
            groups,
            metric,
            title,
            line_x_label,
            path.with_name(f"{path.stem}_line.png"),
            color_map,
            marker_map,
            set_x_ticks=set_line_x_ticks,
        )

    for pareto, name, title in (
        (False, "scatter", f"WCSS vs Time trade-off by {title_suffix}"),
        (True, "pareto", f"WCSS vs Time - Pareto front by {title_suffix}"),
    ):
        _plot_grouped_scatter(
            df,
            group_col,
            groups,
            title,
            scatter_panel_title,
            save_dir / f"{file_prefix}_{name}.png",
            color_map,
            marker_map,
            fill_map,
            legend_info,
            pareto=pareto,
        )


def plot_by_k_multiplier(
    by_k_mult: pd.DataFrame,
    save_dir: Path,
    color_map: dict[str, tuple] | None = None,
    marker_map: dict[str, str] | None = None,
    fill_map: dict[str, bool] | None = None,
    legend_info: dict | None = None,
) -> None:
    """Create aggregate plots grouped by requested cluster multiplier."""
    groups = sorted(by_k_mult["k_multiplier"].unique())
    _plot_by_group(
        by_k_mult,
        "k_multiplier",
        groups,
        "by_k_multiplier",
        "k_multiplier",
        "k_multiplier",
        lambda value: f"k_multiplier = {value}",
        lambda value: f"k_multiplier = {value}",
        color_map,
        marker_map,
        fill_map,
        legend_info,
        save_dir,
        set_line_x_ticks=True,
    )


def plot_by_size_bin(
    by_size: pd.DataFrame,
    save_dir: Path,
    color_map: dict[str, tuple] | None = None,
    marker_map: dict[str, str] | None = None,
    fill_map: dict[str, bool] | None = None,
    legend_info: dict | None = None,
) -> None:
    """Create aggregate plots grouped by dataset size bin."""
    groups = [label for label in SIZE_BIN_LABELS if label in by_size["size_bin"].to_numpy()]
    _plot_by_group(
        by_size,
        "size_bin",
        groups,
        "by_size_bin",
        "dataset size",
        "Dataset size bin",
        lambda value: f"Size Bin: {value}",
        lambda value: f"size bin: {value}",
        color_map,
        marker_map,
        fill_map,
        legend_info,
        save_dir,
        set_line_x_ticks=False,
    )
