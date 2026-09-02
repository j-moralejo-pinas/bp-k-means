"""Shared visual styling and plotting primitives for benchmark analysis."""

import colorsys
from pathlib import Path
from typing import Any, cast

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .data import (
    BP_ALGORITHM_PATTERN as _BP_RE,
)
from .data import algorithm_label as alg_label

ZERO_TOLERANCE = 1e-15
ROUNDING_TOLERANCE = 1e-12
INTEGER_MAGNITUDE_LIMIT = 1e12
LOG_RANGE_LIMIT = 3.0
PLOT_OPTIONS = {"show_titles": False}


def set_show_titles(*, show_titles: bool) -> None:
    """Set title visibility for figures created during this analysis run."""
    PLOT_OPTIONS["show_titles"] = show_titles


def set_title(ax: Any, title: str | None, *, force: bool = False) -> None:
    """Set an axis title when titles are enabled or explicitly forced."""
    if (PLOT_OPTIONS["show_titles"] or force) and title:
        ax.set_title(title)


def set_suptitle(fig: Any, title: str | None, **kwargs: Any) -> None:
    """Set a figure title when titles are enabled."""
    if PLOT_OPTIONS["show_titles"] and title:
        fig.suptitle(title, **kwargs)


MATH_LABELS = {
    "M_L": r"$M_L$",
    "M_C": r"$M_C$",
    "M_ERL": r"$M_{ERL}$",
    "M_RL": r"$M_{RL}$",
    "I_LRI": r"$I_{LRI}$",
    "I_CRI": r"$I_{CRI}$",
    "I_ACL": r"$I_{ACL}$",
    "I_ACC": r"$I_{ACC}$",
}


def to_math_label(text: str) -> str:
    """Replace internal algorithm codes with display-friendly math labels."""
    rendered = text
    for plain, math in MATH_LABELS.items():
        rendered = rendered.replace(plain, math)
    return rendered


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "*", "<"]


def _plain_number(value: float, _pos: int | None = None) -> str:
    """Render numeric ticks/labels without scientific notation."""
    if not np.isfinite(value):
        return ""
    if abs(value) < ZERO_TOLERANCE:
        value = 0.0
    rounded_int = round(value)
    if abs(value - rounded_int) < ROUNDING_TOLERANCE and abs(value) < INTEGER_MAGNITUDE_LIMIT:
        return str(rounded_int)
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


def apply_plain_tick_format(axis: Any) -> None:
    """Force plain (non-scientific) tick labels on a matplotlib Axis."""
    if axis.get_scale() == "log":
        lo, hi = axis.get_view_interval()
        if hi < lo:
            lo, hi = hi, lo
        if lo > 0 and hi > 0 and np.isfinite(lo) and np.isfinite(hi):
            # In tight log ranges (common in Pareto zoom), default log locators
            # may produce zero major ticks. Use fixed geometric ticks instead.
            if hi / lo < LOG_RANGE_LIMIT:
                axis.set_major_locator(mticker.FixedLocator(np.geomspace(lo, hi, 6)))
            else:
                axis.set_major_locator(mticker.LogLocator(base=10.0, numticks=10))
        axis.set_major_formatter(mticker.FuncFormatter(_plain_number))
        axis.set_minor_formatter(mticker.NullFormatter())
        return

    fmt = mticker.ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    fmt.set_powerlimits((-10, 10))
    axis.set_major_formatter(fmt)
    axis.set_minor_formatter(mticker.NullFormatter())


def build_color_map(
    df: pd.DataFrame, min_lightness: float = 0.25, max_lightness: float = 0.80
) -> dict[str, tuple]:
    """
    Return a label-string → RGB color dict.

    Color scheme:
    - Non-BP-KMeans algorithms each get a distinct hue from tab10.
    - BP-KMeans metrics each get a distinct
      hue continuing from where non-BP hues left off.
    - Higher n_init → darker shade (lower lightness).
    """
    base_cmap = plt.get_cmap("tab10")
    unique_algs = sorted(df["algorithm"].unique())
    unique_n_inits = sorted(df["n_init"].unique())
    n_levels = len(unique_n_inits)

    non_bp_algs = [a for a in unique_algs if not a.startswith("BP-KMeans")]
    bp_algs = [a for a in unique_algs if a.startswith("BP-KMeans")]

    # Extract unique ranking methods from BP-KMeans algorithm strings
    bp_ranking_methods = sorted({m.group(1) for a in bp_algs if (m := _BP_RE.search(a))})

    # Hue index: non-BP algorithms first, then BP ranking methods
    offset = len(non_bp_algs)
    hue_idx_map = {
        **{alg: i for i, alg in enumerate(non_bp_algs)},
        **{method: offset + i for i, method in enumerate(bp_ranking_methods)},
    }

    def hue_key(alg: str) -> str:
        if alg.startswith("BP-KMeans"):
            m = _BP_RE.search(alg)
            return m.group(1) if m else alg
        return alg

    color_map: dict[str, tuple] = {}
    for alg in unique_algs:
        idx = hue_idx_map[hue_key(alg)]
        r, g, b, _ = base_cmap(idx % base_cmap.N)
        h, _l, s = colorsys.rgb_to_hls(r, g, b)
        for init_idx, n_init in enumerate(unique_n_inits):
            if n_levels == 1:
                new_l = _l
            else:
                # low n_init → max_lightness (light), high n_init → min_lightness (dark)
                new_l = max_lightness - (max_lightness - min_lightness) * (
                    init_idx / (n_levels - 1)
                )
            nr, ng, nb = colorsys.hls_to_rgb(h, new_l, min(s, 0.9))
            color_map[f"{alg} | n_init={n_init}"] = (nr, ng, nb)
    return color_map


def build_marker_map(df: pd.DataFrame) -> dict[str, str]:
    """
    Return a label-string → matplotlib marker dict based on init_strategy.

    Non-BP-KMeans algorithms get 'o'.  Each unique init_strategy in BP-KMeans gets a distinct marker
    from MARKERS.
    """
    unique_algs = sorted(df["algorithm"].unique())
    unique_n_inits = sorted(df["n_init"].unique())
    init_strategies = sorted({m.group(2) for a in unique_algs if (m := _BP_RE.search(a))})
    init_strategy_to_marker = {
        strategy: MARKERS[i % len(MARKERS)] for i, strategy in enumerate(init_strategies)
    }
    marker_map: dict[str, str] = {}
    for alg in unique_algs:
        bp_m = _BP_RE.search(alg)
        marker = init_strategy_to_marker.get(bp_m.group(2), "o") if bp_m else "o"
        for n_init in unique_n_inits:
            marker_map[f"{alg} | n_init={n_init}"] = marker
    return marker_map


def build_fill_map(df: pd.DataFrame) -> dict[str, bool]:
    """Build a fill map for KMEANS_PLUS_PLUS and other initialization algorithms."""
    unique_algs = sorted(df["algorithm"].unique())
    unique_n_inits = sorted(df["n_init"].unique())
    fill_map: dict[str, bool] = {}
    for alg in unique_algs:
        bp_m = _BP_RE.search(alg)
        filled = (not bp_m) or (bp_m.group(3) == "KMEANS_PLUS_PLUS")
        for n_init in unique_n_inits:
            fill_map[f"{alg} | n_init={n_init}"] = filled
    return fill_map


def _build_color_legend_entries(
    unique_algs: list[str], rep_n_init: int, color_map: dict[str, tuple]
) -> tuple[list[tuple[str, tuple]], list[tuple[str, tuple]]]:
    """Build baseline and ranking color legend entries."""
    baselines: dict[str, tuple] = {}
    rankings: dict[str, tuple] = {}
    for alg in unique_algs:
        bp_match = _BP_RE.search(alg)
        rep_label = f"{alg} | n_init={rep_n_init}"
        if bp_match:
            rankings.setdefault(bp_match.group(1), color_map.get(rep_label, (0.5, 0.5, 0.5)))
        else:
            baselines.setdefault(alg, color_map.get(rep_label, (0.5, 0.5, 0.5)))
    return list(baselines.items()), list(rankings.items())


def _build_component_legend_entries(
    unique_algs: list[str],
    rep_n_init: int,
    component: int,
    value_map: dict[str, Any],
    default: Any,
) -> list[tuple[str, Any]]:
    """Map one parsed BP-KMeans component to its visual encoding."""
    entries = {}
    for alg in unique_algs:
        bp_match = _BP_RE.search(alg)
        if bp_match:
            rep_label = f"{alg} | n_init={rep_n_init}"
            entries.setdefault(bp_match.group(component), value_map.get(rep_label, default))
    return sorted(entries.items())


def _build_lightness_legend_entries(
    unique_n_inits: list[int], min_lightness: float, max_lightness: float
) -> list[tuple[int, tuple]]:
    """Build the gray swatches used to explain n_init lightness."""
    n_levels = len(unique_n_inits)
    entries = []
    for init_idx, n_init in enumerate(unique_n_inits):
        lightness = (
            (min_lightness + max_lightness) / 2
            if n_levels == 1
            else max_lightness - (max_lightness - min_lightness) * init_idx / (n_levels - 1)
        )
        entries.append((n_init, colorsys.hls_to_rgb(0.0, lightness, 0.0)))
    return entries


def build_legend_info(
    df: pd.DataFrame,
    color_map: dict[str, tuple],
    marker_map: dict[str, str],
    fill_map: dict[str, bool],
    min_lightness: float = 0.25,
    max_lightness: float = 0.80,
) -> dict:
    """Build legend entries for colors, shapes, fills, and initialization count."""
    unique_algs = sorted(df["algorithm"].unique())
    unique_n_inits = sorted(df["n_init"].unique())
    rep_n_init = unique_n_inits[len(unique_n_inits) // 2]
    baseline_entries, ranking_entries = _build_color_legend_entries(
        unique_algs, rep_n_init, color_map
    )
    return {
        "baseline_color_entries": baseline_entries,
        "ranking_color_entries": ranking_entries,
        "shape_entries": _build_component_legend_entries(
            unique_algs, rep_n_init, 2, marker_map, "o"
        ),
        "fill_entries": _build_component_legend_entries(
            unique_algs, rep_n_init, 3, fill_map, default=True
        ),
        "n_init_lightness_entries": _build_lightness_legend_entries(
            unique_n_inits, min_lightness, max_lightness
        ),
    }


def _append_legend_section(
    handles: list[Any],
    labels: list[str],
    title: str,
    entries: list[tuple],
    mode: str,
) -> None:
    """Append one typed section to a scatter legend."""
    if len(entries) <= 1:
        return
    handles.append(mpatches.Patch(color="none", label=title))
    labels.append(title)
    for name, value in entries:
        if mode == "color":
            color, marker, facecolor, edgecolor = value, "o", value, None
        elif mode == "shape":
            color, marker, facecolor, edgecolor = "black", value, "black", None
        elif mode == "fill":
            color, marker, facecolor, edgecolor = (
                "black",
                "o",
                "black" if value else "none",
                "black",
            )
        else:
            color, marker, facecolor, edgecolor = value, "o", value, value
        kwargs = {
            "color": color,
            "marker": marker,
            "linestyle": "None",
            "markersize": 6,
            "markerfacecolor": facecolor,
        }
        if edgecolor is not None:
            kwargs["markeredgecolor"] = edgecolor
        handles.append(mlines.Line2D([], [], **kwargs))
        labels.append(str(name) if mode == "lightness" else to_math_label(name))


def add_scatter_legends(
    ax: Any,
    legend_info: dict,
    *,
    has_pareto_line: bool = False,
    loc: str = "upper left",
    bbox_to_anchor: tuple = (1.02, 1.0),
) -> None:
    """Add a combined color, shape, fill, and lightness legend to an axis."""
    handles: list[Any] = []
    labels: list[str] = []
    if has_pareto_line:
        handles.append(
            mlines.Line2D([], [], color="black", linewidth=1.5, linestyle="-", alpha=0.7)
        )
        labels.append("Pareto front")

    for key, title, mode in (
        ("baseline_color_entries", "── Baseline Algorithms ──", "color"),
        ("ranking_color_entries", "── Label Selection Metric ──", "color"),
        ("shape_entries", "── Reinitialization Method ──", "shape"),
        ("fill_entries", "── Initialization Algorithm ──", "fill"),
        ("n_init_lightness_entries", "── # Initializations (light→dark) ──", "lightness"),
    ):
        _append_legend_section(handles, labels, title, legend_info.get(key, []), mode)

    if not handles:
        return
    leg = ax.legend(
        handles=handles,
        labels=labels,
        fontsize=7,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        borderaxespad=0,
        frameon=True,
        handletextpad=0.5,
    )
    for text in leg.get_texts():
        if text.get_text().startswith("──"):
            text.set_fontweight("bold")


def build_label_color_map(labels: list[str]) -> dict[str, tuple]:
    """Return a label-string → RGB color dict using the tab10 palette."""
    base_cmap = plt.get_cmap("tab10")
    return {lbl: base_cmap(i % base_cmap.N) for i, lbl in enumerate(sorted(set(labels)))}


def pivot_for_line(df: pd.DataFrame, x_col: str, metric: str) -> pd.DataFrame:
    """Return a (label x x_col) pivot suitable for line plots."""
    df = df.copy()
    if "label" not in df.columns:
        df["label"] = df.apply(alg_label, axis=1)
    return df.pivot_table(
        index="label", columns=x_col, values=metric, aggfunc="mean", observed=False
    )


def compute_pareto_front(xs: list[float], ys: list[float]) -> list[bool]:
    """Return boolean mask of Pareto-optimal points (minimise both x and y)."""
    n = len(xs)
    is_pareto = [True] * n
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if xs[j] <= xs[i] and ys[j] <= ys[i] and (xs[j] < xs[i] or ys[j] < ys[i]):
                is_pareto[i] = False
                break
    return is_pareto


def draw_scatter_plot(
    ax: Any,
    sub: pd.DataFrame,
    color_map: dict[str, tuple] | None,
    marker_map: dict[str, str] | None = None,
    fill_map: dict[str, bool] | None = None,
    *,
    x_col: str = "mean_relative_time",
    y_col: str = "mean_relative_wcss",
    x_label: str = "Mean Relative Time",
    y_label: str = "Mean Relative WCSS",
    title: str | None = None,
    force_title: bool = False,
    pareto: bool = False,
    floor_at_one: bool = True,
    add_reference_lines: bool = True,
    reference_value: float = 1.0,
) -> None:
    """Draw a scatter panel with optional Pareto-front highlighting."""
    sub = cast("Any", sub)
    xs = sub[x_col].tolist()
    ys = sub[y_col].tolist()
    pareto_mask = compute_pareto_front(xs, ys) if pareto else [False] * len(xs)

    for i, (_, row) in enumerate(sub.iterrows()):
        row = cast("Any", row)
        on_front = pareto and pareto_mask[i]
        color = color_map.get(row["label"], "steelblue") if color_map else "steelblue"
        marker = marker_map.get(row["label"], "o") if marker_map else "o"
        filled = fill_map.get(row["label"], True) if fill_map else True
        ax.scatter(
            row[x_col],
            row[y_col],
            marker=marker,
            s=55 if on_front else 25 if pareto else 35,
            zorder=4 if on_front else 2 if pareto else 3,
            facecolors=color if filled else "none",
            edgecolors=color,
            linewidths=0.6 if pareto and not on_front else 0.8,
            alpha=0.25 if pareto and not on_front else 1.0,
        )

    front_sorted: list[tuple[float, float]] = []
    if pareto:
        front_sorted = sorted(
            [(xs[i], ys[i]) for i in range(len(xs)) if pareto_mask[i]],
            key=lambda p: p[0],
        )
        if front_sorted:
            x_start = min(xs) * 0.95
            xs_plot: list[float] = [x_start, front_sorted[0][0]]
            ys_plot: list[float] = [front_sorted[0][1], front_sorted[0][1]]
            for k in range(1, len(front_sorted)):
                fx, fy = front_sorted[k]
                prev_y = front_sorted[k - 1][1]
                xs_plot += [fx, fx]
                ys_plot += [prev_y, fy]
            ax.plot(
                xs_plot,
                ys_plot,
                color="black",
                linewidth=1.5,
                linestyle="-",
                zorder=5,
                alpha=0.7,
                label="Pareto front",
            )

    if add_reference_lines:
        ax.axhline(reference_value, color="crimson", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axvline(reference_value, color="crimson", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    set_title(ax, title, force=force_title)

    # Pareto zoom: lower bound at 1.0 when requested, upper bound near Pareto-front maxima.
    # Add a small top/right padding so frontier points are not clipped by the frame.
    if pareto and front_sorted:
        px = [p[0] for p in front_sorted]
        py = [p[1] for p in front_sorted]
        x_min = 1.0 if floor_at_one else min(xs)
        y_min = 1.0 if floor_at_one else min(ys)
        x_pad = max((max(px) - x_min) * 0.1, 1e-6)
        y_pad = max((max(py) - y_min) * 0.1, 1e-6)
        xlim = (x_min, max(px) + x_pad)
        ylim = (y_min, max(py) + y_pad)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.__dict__["_pareto_xlim"] = xlim
        ax.__dict__["_pareto_ylim"] = ylim

    apply_plain_tick_format(ax.xaxis)
    apply_plain_tick_format(ax.yaxis)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def draw_bar_chart(
    ax: Any,
    df: pd.DataFrame,
    metric: str,
    color_map: dict[str, tuple] | None,
    pareto_labels: set[str] | None = None,
    *,
    reference_label: str | None = None,
) -> None:
    """Draw the consistently styled horizontal bars used throughout the analysis."""
    labels = df["label"].tolist()
    display_labels = [
        f"\u2605 {to_math_label(label)}"
        if pareto_labels and label in pareto_labels
        else to_math_label(label)
        for label in labels
    ]
    colors = [color_map.get(label, "steelblue") for label in labels] if color_map else "steelblue"
    bars = ax.barh(display_labels, df[metric], color=colors, edgecolor="white")
    ax.bar_label(
        bars,
        labels=[_plain_number(value) for value in df[metric].tolist()],
        padding=3,
        fontsize=7,
    )
    if reference_label:
        ax.axvline(1.0, color="crimson", linestyle="--", linewidth=1.2, label=reference_label)


def bold_pareto_ticks(axes: list[Any]) -> None:
    """Emphasize tick labels marked as Pareto-optimal."""
    for ax in axes:
        for tick in ax.yaxis.get_ticklabels():
            if tick.get_text().startswith("\u2605"):
                tick.set_fontweight("bold")


def plot_bar_chart(
    df: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    save_path: Path,
    label_col: str = "label",
    reference_line: float | None = 1.0,
    color_map: dict[str, tuple] | None = None,
    pareto_labels: set | None = None,
) -> None:
    """Create and save a sorted horizontal bar chart."""
    sorted_df = cast("pd.DataFrame", df.sort_values(metric)).rename(columns={label_col: "label"})
    fig, ax = plt.subplots(figsize=(9, max(4, len(sorted_df) * 0.35)))
    draw_bar_chart(
        ax,
        sorted_df,
        metric,
        color_map,
        pareto_labels,
        reference_label=f"best = {reference_line:.1f}" if reference_line is not None else None,
    )
    if reference_line is not None:
        ax.legend(fontsize=8)
    ax.set_xlabel(xlabel)
    set_title(ax, title)
    apply_plain_tick_format(ax.xaxis)
    fig.tight_layout()
    bold_pareto_ticks([ax])
    fig.savefig(save_path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def create_panel_grid(
    n_panels: int, width_per_col: float, height_per_row: float
) -> tuple[Any, np.ndarray]:
    """Create subplot grid with at most 2 panels per row."""
    n_cols = 1 if n_panels <= 1 else min(2, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width_per_col * n_cols, height_per_row * n_rows),
    )
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr[n_panels:]:
        ax.set_visible(False)
    return fig, axes_arr[:n_panels]


def save_with_log_variant(
    fig: Any,
    axes_list: list[Any],
    linear_path: Path,
    *,
    log_x_axis: bool = True,
    log_y_axis: bool = False,
) -> None:
    """
    Save *linear_path* (linear scale), then a log-scale variant (*_log.png).

    Axes titles get " (log)" appended in the log version. The figure is closed after both saves.
    """
    fig.savefig(linear_path, dpi=450, bbox_inches="tight")
    for ax in axes_list:
        if log_x_axis:
            ax.set_xscale("log")
        if log_y_axis:
            ax.set_yscale("log")
        # If this is a pareto ax, reapply the stored Pareto-front zoom;
        # otherwise autoscale so all data points are visible.
        pareto_xlim = getattr(ax, "_pareto_xlim", None)
        pareto_ylim = getattr(ax, "_pareto_ylim", None)
        if pareto_xlim is not None and pareto_ylim is not None:
            ax.set_xlim(pareto_xlim)
            ax.set_ylim(pareto_ylim)
        else:
            ax.autoscale()
        # Suppress sci notation on log axes
        apply_plain_tick_format(ax.xaxis)
        apply_plain_tick_format(ax.yaxis)
        old_title = ax.get_title()
        if PLOT_OPTIONS["show_titles"] and old_title:
            if log_x_axis and log_y_axis:
                suffix = " (log)"
            elif log_x_axis:
                suffix = " (log x)"
            elif log_y_axis:
                suffix = " (log y)"
            else:
                suffix = ""
            set_title(ax, old_title + suffix)
    log_path = linear_path.parent / (linear_path.stem + "_log" + linear_path.suffix)
    fig.savefig(log_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
