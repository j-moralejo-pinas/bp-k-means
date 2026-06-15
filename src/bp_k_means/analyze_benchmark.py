"""Analyze and aggregate benchmark results from the output/ directory.

Pipeline
--------
1. Load all metadata.json files into a flat DataFrame.
2. For each (dataset, k_multiplier): compute best (min) wcss and time across
   all algorithm × n_init combinations.
3. For each row compute relative_wcss = wcss / best_wcss and
   relative_time = time / best_time.
4. Aggregate per (algorithm, n_init):
   a. Mean relative wcss / time across ALL dataset × k_multiplier.
   b. Mean relative wcss / time broken down by k_multiplier.
   c. Mean relative wcss / time broken down by dataset-size bin.
5. Save every aggregate table as CSV and produce matching plots.

Outputs (all in output/analysis/)
----------------------------------
relative_metrics.csv          – full per-run relative table
overall_avg.csv               – aggregate 4a
by_k_multiplier.csv           – aggregate 4b
by_size_bin.csv               – aggregate 4c
overall_avg.png               – bar charts for 4a
by_k_multiplier_wcss.png      – bar charts for 4b (wcss)
by_k_multiplier_time.png      – bar charts for 4b (time)
by_k_multiplier_wcss_line.png – line chart for 4b (wcss)
by_k_multiplier_time_line.png – line chart for 4b (time)
by_k_multiplier_scatter.png   – scatter (wcss vs time) per k_multiplier
by_k_multiplier_pareto.png    – scatter with Pareto front per k_multiplier
by_size_bin_wcss.png          – bar charts for 4c (wcss)
by_size_bin_time.png          – bar charts for 4c (time)
by_size_bin_scatter.png       – scatter (wcss vs time) per size bin
by_size_bin_pareto.png        – scatter with Pareto front per size bin
overall_pareto.png            – scatter with Pareto front (overall)
special/*/*_time_comparison.png
                              – metric/time bars for Bisecting KMeans vs selected BP-KMeans
"""

import argparse
import colorsys
import json
import re
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("output")
RESULTS_DIR = Path("output/analysis")
DATA_DIR = Path("data/datasets")
HAC_STRENGTH_BENCHMARK_TYPE = "hac_strength"

SIZE_BIN_LABELS = [
    ">1k labels",
    ">5k nodes <1k labels",
    "1k–5k nodes",
    "<1k nodes",
]


def assign_size_bin(n_instances: float, n_labels: int) -> str:
    if n_labels > 1_000:
        return ">1k labels"
    if n_instances > 5_000:
        return ">5k nodes <1k labels"
    if n_instances > 1000:
        return "1k–5k nodes"
    return "<1k nodes"


# Set to True to exclude HAC Ward runs from all analyses and plots
EXCLUDE_HAC = True
SHOW_TITLES = False


def _set_title(ax, title: str | None, *, force: bool = False) -> None:
    if (SHOW_TITLES or force) and title:
        ax.set_title(title)


def _set_suptitle(fig, title: str | None, **kwargs) -> None:
    if SHOW_TITLES and title:
        fig.suptitle(title, **kwargs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_metadata() -> pd.DataFrame:
    rows = []
    for meta_path in sorted(OUTPUT_DIR.rglob("metadata.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("benchmark_type") == HAC_STRENGTH_BENCHMARK_TYPE:
            continue
        rows.append(
            {
                "dataset": meta["dataset"],
                "algorithm": meta["algorithm"],
                "n_init": int(meta["n_init"]),
                "k_multiplier": float(meta["k_multiplier"]),
                "k": int(meta["k"]),
                "n_clusters": int(meta["n_clusters"]),
                "n_labels": int(meta["n_labels"]) if "n_labels" in meta else np.nan,
                "wcss": float(meta["wcss_total"]),
                "time": float(meta["duration_seconds"]),
            }
        )
    return pd.DataFrame(rows)


def load_hac_strength_metadata() -> pd.DataFrame:
    """Load only metadata rows produced by the HAC-strength benchmark."""
    rows = []
    for meta_path in sorted(OUTPUT_DIR.rglob("metadata.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("benchmark_type") != HAC_STRENGTH_BENCHMARK_TYPE:
            continue
        rows.append(
            {
                "dataset": meta["dataset"],
                "algorithm": meta["algorithm"],
                "n_init": int(meta["n_init"]),
                "k_multiplier": float(meta["k_multiplier"]),
                "k": int(meta["k"]),
                "n_clusters": int(meta["n_clusters"]),
                "n_labels": int(meta["n_labels"]) if "n_labels" in meta else np.nan,
                "wcss": float(meta["wcss_total"]),
                "time": float(meta["duration_seconds"]),
                "requested_cluster_multiplier": float(
                    meta.get("requested_cluster_multiplier", meta["k_multiplier"])
                ),
                "requested_n_clusters": int(meta.get("requested_n_clusters", meta["k"])),
                "target_k_was_capped": bool(meta.get("target_k_was_capped", False)),
            }
        )
    return pd.DataFrame(rows)


def load_dataset_sizes() -> dict[str, int]:
    import pyarrow.parquet as pq

    sizes: dict[str, int] = {}
    for path in DATA_DIR.glob("*nodes.parquet"):
        if "com" in path.stem:
            continue
        try:
            meta = pq.read_metadata(path)
            sizes[path.stem] = meta.num_rows
        except Exception as e:
            print(f"  WARNING: could not read size for {path.name}: {e}")
    return sizes


_BP_RE = re.compile(r"BP-KMeans \((\w+),\s*(\w+),\s*(\w+)\)")

MATH_LABELS = {
    "R_L": r"$R_L$",
    "R_C": r"$R_C$",
    "R_ERL": r"$R_{ERL}$",
    "R_ERC": r"$R_{ERC}$",
    "R_RL": r"$R_{RL}$",
    "I_LRI": r"$I_{LRI}$",
    "I_CRI": r"$I_{CRI}$",
    "I_ACL": r"$I_{ACL}$",
    "I_ACC": r"$I_{ACC}$",
}


def to_math_label(text: str) -> str:
    rendered = text
    for plain, math in MATH_LABELS.items():
        rendered = rendered.replace(plain, math)
    return rendered


def parse_algorithm_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add label_selection_method, reinit_method, init_algo columns for BP-KMeans rows."""
    df = df.copy()
    parsed = df["algorithm"].str.extract(_BP_RE)
    df["label_selection_method"] = parsed[0]
    df["reinit_method"] = parsed[1]
    df["init_algo"] = parsed[2]
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "*", "<"]


def _plain_number(value: float, _pos: int | None = None) -> str:
    """Render numeric ticks/labels without scientific notation."""
    if not np.isfinite(value):
        return ""
    if abs(value) < 1e-15:
        value = 0.0
    rounded_int = int(round(value))
    if abs(value - rounded_int) < 1e-12 and abs(value) < 1e12:
        return str(rounded_int)
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


def _apply_no_sci(axis) -> None:
    """Force plain (non-scientific) tick labels on a matplotlib Axis."""
    if axis.get_scale() == "log":
        lo, hi = axis.get_view_interval()
        if hi < lo:
            lo, hi = hi, lo
        if lo > 0 and hi > 0 and np.isfinite(lo) and np.isfinite(hi):
            # In tight log ranges (common in Pareto zoom), default log locators
            # may produce zero major ticks. Use fixed geometric ticks instead.
            if hi / lo < 3.0:
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


def alg_label(row: pd.Series) -> str:
    return f"{row['algorithm']} | n_init={row['n_init']}"


def build_color_map(
    df: pd.DataFrame, min_lightness: float = 0.25, max_lightness: float = 0.80
) -> dict[str, tuple]:
    """Return a label-string → RGB color dict.

    Color scheme:
    - Non-BP-KMeans algorithms each get a distinct hue from tab10.
    - BP-KMeans ranking methods (label_selection_method) each get a distinct
      hue continuing from where non-BP hues left off.
    - Higher n_init → darker shade (lower lightness).
    """
    base_cmap = plt.cm.tab10
    unique_algs = sorted(df["algorithm"].unique())
    unique_n_inits = sorted(df["n_init"].unique())
    n_levels = len(unique_n_inits)

    non_bp_algs = [a for a in unique_algs if not a.startswith("BP-KMeans")]
    bp_algs = [a for a in unique_algs if a.startswith("BP-KMeans")]

    # Extract unique ranking methods from BP-KMeans algorithm strings
    bp_ranking_methods = sorted({m.group(1) for a in bp_algs if (m := _BP_RE.search(a))})

    # Hue index: non-BP algorithms first, then BP ranking methods
    hue_idx_map: dict[str, int] = {}
    for i, alg in enumerate(non_bp_algs):
        hue_idx_map[alg] = i
    offset = len(non_bp_algs)
    for i, method in enumerate(bp_ranking_methods):
        hue_idx_map[method] = offset + i

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
    """Return a label-string → matplotlib marker dict based on reinit_method.

    Non-BP-KMeans algorithms get 'o'.  Each unique reinit_method in
    BP-KMeans gets a distinct marker from MARKERS.
    """
    unique_algs = sorted(df["algorithm"].unique())
    unique_n_inits = sorted(df["n_init"].unique())
    reinit_methods = sorted({m.group(2) for a in unique_algs if (m := _BP_RE.search(a))})
    reinit_to_marker = {rm: MARKERS[i % len(MARKERS)] for i, rm in enumerate(reinit_methods)}
    marker_map: dict[str, str] = {}
    for alg in unique_algs:
        bp_m = _BP_RE.search(alg)
        marker = reinit_to_marker.get(bp_m.group(2), "o") if bp_m else "o"
        for n_init in unique_n_inits:
            marker_map[f"{alg} | n_init={n_init}"] = marker
    return marker_map


def build_fill_map(df: pd.DataFrame) -> dict[str, bool]:
    """True → filled marker (KMEANS_PLUS_PLUS init_algo); False → unfilled."""
    unique_algs = sorted(df["algorithm"].unique())
    unique_n_inits = sorted(df["n_init"].unique())
    fill_map: dict[str, bool] = {}
    for alg in unique_algs:
        bp_m = _BP_RE.search(alg)
        filled = (not bp_m) or (bp_m.group(3) == "KMEANS_PLUS_PLUS")
        for n_init in unique_n_inits:
            fill_map[f"{alg} | n_init={n_init}"] = filled
    return fill_map


def build_legend_info(
    df: pd.DataFrame,
    color_map: dict[str, tuple],
    marker_map: dict[str, str],
    fill_map: dict[str, bool],
    min_lightness: float = 0.25,
    max_lightness: float = 0.80,
) -> dict:
    """Build legend entries for colour (ranking/alg), shape (reinit), fill (init_algo)."""
    unique_algs = sorted(df["algorithm"].unique())
    unique_n_inits = sorted(df["n_init"].unique())
    rep_n_init = unique_n_inits[len(unique_n_inits) // 2]

    # Colour entries are split in two sections:
    # - non-BP algorithms (baselines)
    # - BP ranking methods
    seen_baselines: set[str] = set()
    baseline_color_entries: list[tuple[str, tuple]] = []
    seen_rankings: set[str] = set()
    ranking_color_entries: list[tuple[str, tuple]] = []
    for alg in unique_algs:
        bp_m = _BP_RE.search(alg)
        rep_label = f"{alg} | n_init={rep_n_init}"
        if bp_m:
            ranking = bp_m.group(1)
            if ranking not in seen_rankings:
                seen_rankings.add(ranking)
                ranking_color_entries.append((ranking, color_map.get(rep_label, (0.5, 0.5, 0.5))))
        elif alg not in seen_baselines:
            seen_baselines.add(alg)
            baseline_color_entries.append((alg, color_map.get(rep_label, (0.5, 0.5, 0.5))))

    # Shape entries: unique reinit_method → marker
    seen_reinit: dict[str, str] = {}
    for alg in unique_algs:
        bp_m = _BP_RE.search(alg)
        if bp_m:
            rm = bp_m.group(2)
            if rm not in seen_reinit:
                rep_label = f"{alg} | n_init={rep_n_init}"
                seen_reinit[rm] = marker_map.get(rep_label, "o")
    shape_entries: list[tuple[str, str]] = sorted(seen_reinit.items())

    # Fill entries: unique init_algo → filled/unfilled (BP-KMeans only)
    seen_fill: dict[str, bool] = {}
    for alg in unique_algs:
        bp_m = _BP_RE.search(alg)
        if bp_m:
            ia = bp_m.group(3)
            if ia not in seen_fill:
                rep_label = f"{alg} | n_init={rep_n_init}"
                seen_fill[ia] = fill_map.get(rep_label, True)
    fill_entries: list[tuple[str, bool]] = sorted(seen_fill.items())

    # Lightness entries: n_init value → gray swatch color
    n_levels = len(unique_n_inits)
    n_init_lightness_entries: list[tuple[int, tuple]] = []
    for init_idx, n_init in enumerate(unique_n_inits):
        if n_levels == 1:
            lightness = (min_lightness + max_lightness) / 2
        else:
            lightness = max_lightness - (max_lightness - min_lightness) * (
                init_idx / (n_levels - 1)
            )
        gray = colorsys.hls_to_rgb(0.0, lightness, 0.0)
        n_init_lightness_entries.append((n_init, gray))

    return {
        "baseline_color_entries": baseline_color_entries,
        "ranking_color_entries": ranking_color_entries,
        "shape_entries": shape_entries,
        "fill_entries": fill_entries,
        "n_init_lightness_entries": n_init_lightness_entries,
    }


def _add_scatter_legends(
    ax,
    legend_info: dict,
    has_pareto_line: bool = False,
    loc: str = "upper left",
    bbox_to_anchor: tuple = (1.02, 1.0),
) -> None:
    """Add a combined colour / shape / fill legend outside the right side of ax."""
    handles: list = []
    labels: list = []

    def _section(text: str) -> None:
        handles.append(mpatches.Patch(color="none", label=text))
        labels.append(text)

    if has_pareto_line:
        # _section("── System ──")
        handles.append(
            mlines.Line2D([], [], color="black", linewidth=1.5, linestyle="-", alpha=0.7)
        )
        labels.append("Pareto front")

    if len(legend_info.get("baseline_color_entries", [])) > 1:
        _section("── Baseline algorithms ──")
        for name, color in legend_info["baseline_color_entries"]:
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=color,
                    marker="o",
                    linestyle="None",
                    markersize=6,
                    markerfacecolor=color,
                )
            )
            labels.append(to_math_label(name))

    if len(legend_info.get("ranking_color_entries", [])) > 1:
        _section("── Ranking method ──")
        for name, color in legend_info["ranking_color_entries"]:
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=color,
                    marker="o",
                    linestyle="None",
                    markersize=6,
                    markerfacecolor=color,
                )
            )
            labels.append(to_math_label(name))

    if len(legend_info.get("shape_entries", [])) > 1:
        _section("── Reinitialization Method ──")
        for name, mkr in legend_info["shape_entries"]:
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color="black",
                    marker=mkr,
                    linestyle="None",
                    markersize=6,
                    markerfacecolor="black",
                )
            )
            labels.append(to_math_label(name))

    if len(legend_info.get("fill_entries", [])) > 1:
        _section("── Initialization Algorithm ──")
        for name, filled in legend_info["fill_entries"]:
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color="black",
                    marker="o",
                    linestyle="None",
                    markersize=6,
                    markerfacecolor="black" if filled else "none",
                    markeredgecolor="black",
                )
            )
            labels.append(to_math_label(name))

    if len(legend_info.get("n_init_lightness_entries", [])) > 1:
        _section("── # Initializations (light→dark) ──")
        for n_init_val, gray in legend_info["n_init_lightness_entries"]:
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=gray,
                    marker="o",
                    linestyle="None",
                    markersize=6,
                    markerfacecolor=gray,
                    markeredgecolor=gray,
                )
            )
            labels.append(str(n_init_val))

    if handles:
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
    base_cmap = plt.cm.tab10
    return {lbl: base_cmap(i % base_cmap.N) for i, lbl in enumerate(sorted(set(labels)))}


def pivot_for_line(df: pd.DataFrame, x_col: str, metric: str) -> pd.DataFrame:
    """Return a (label × x_col) pivot suitable for line plots."""
    df = df.copy()
    if "label" not in df.columns:
        df["label"] = df.apply(alg_label, axis=1)
    return df.pivot_table(index="label", columns=x_col, values=metric, aggfunc="mean")


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


def _draw_scatter_ax(
    ax,
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
    xs = sub[x_col].tolist()
    ys = sub[y_col].tolist()
    pareto_mask = compute_pareto_front(xs, ys) if pareto else [False] * len(xs)

    # Base points (all for regular scatter; non-Pareto for Pareto view)
    for i, (_, row) in enumerate(sub.iterrows()):
        if pareto and pareto_mask[i]:
            continue
        color = color_map.get(row["label"], "steelblue") if color_map else "steelblue"
        marker = marker_map.get(row["label"], "o") if marker_map else "o"
        filled = fill_map.get(row["label"], True) if fill_map else True
        ax.scatter(
            row[x_col],
            row[y_col],
            marker=marker,
            s=25 if pareto else 35,
            zorder=2 if pareto else 3,
            facecolors=color if filled else "none",
            edgecolors=color,
            linewidths=0.6 if pareto else 0.8,
            alpha=0.25 if pareto else 1.0,
        )

    # Pareto-front points and staircase
    front_sorted: list[tuple[float, float]] = []
    if pareto:
        for i, (_, row) in enumerate(sub.iterrows()):
            if not pareto_mask[i]:
                continue
            color = color_map.get(row["label"], "steelblue") if color_map else "steelblue"
            marker = marker_map.get(row["label"], "o") if marker_map else "o"
            filled = fill_map.get(row["label"], True) if fill_map else True
            ax.scatter(
                row[x_col],
                row[y_col],
                marker=marker,
                s=55,
                zorder=4,
                facecolors=color if filled else "none",
                edgecolors=color,
                linewidths=0.8,
            )

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
    _set_title(ax, title, force=force_title)

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
        ax._pareto_xlim = xlim
        ax._pareto_ylim = ylim

    _apply_no_sci(ax.xaxis)
    _apply_no_sci(ax.yaxis)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _bar_chart(
    df: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    save_path: Path,
    label_col: str = "label",
    reference_line: float = 1.0,
    color_map: dict[str, tuple] | None = None,
    pareto_labels: set | None = None,
):
    sorted_df = df.sort_values(metric)
    n_bars = len(sorted_df)
    orig_labels = sorted_df[label_col].tolist()
    display_labels = [
        "\u2605 " + to_math_label(lbl)
        if (pareto_labels and lbl in pareto_labels)
        else to_math_label(lbl)
        for lbl in orig_labels
    ]
    colors = [color_map.get(lbl, "steelblue") for lbl in orig_labels] if color_map else "steelblue"
    fig, ax = plt.subplots(figsize=(9, max(4, n_bars * 0.35)))
    bars = ax.barh(display_labels, sorted_df[metric], color=colors, edgecolor="white")
    ax.bar_label(
        bars,
        labels=[_plain_number(v) for v in sorted_df[metric].tolist()],
        padding=3,
        fontsize=7,
    )
    if reference_line is not None:
        ax.axvline(
            reference_line,
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            label=f"best = {reference_line:.1f}",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel(xlabel)
    _set_title(ax, title)
    _apply_no_sci(ax.xaxis)
    fig.tight_layout()
    for tick in ax.yaxis.get_ticklabels():
        if tick.get_text().startswith("\u2605"):
            tick.set_fontweight("bold")
    fig.savefig(save_path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def _create_panel_grid(n_panels: int, width_per_col: float, height_per_row: float):
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


def _save_with_log_variant(
    fig,
    axes_list,
    linear_path: Path,
    *,
    log_x_axis: bool = True,
    log_y_axis: bool = False,
) -> None:
    """Save *linear_path* (linear scale), then a log-scale variant (*_log.png).

    Axes titles get " (log)" appended in the log version.
    The figure is closed after both saves.
    """
    fig.savefig(linear_path, dpi=450, bbox_inches="tight")
    for ax in axes_list:
        if log_x_axis:
            ax.set_xscale("log")
        if log_y_axis:
            ax.set_yscale("log")
        # If this is a pareto ax, reapply the stored Pareto-front zoom;
        # otherwise autoscale so all data points are visible.
        if hasattr(ax, "_pareto_xlim"):
            ax.set_xlim(ax._pareto_xlim)
            ax.set_ylim(ax._pareto_ylim)
        else:
            ax.autoscale()
        # Suppress sci notation on log axes
        _apply_no_sci(ax.xaxis)
        _apply_no_sci(ax.yaxis)
        old_title = ax.get_title()
        if SHOW_TITLES and old_title:
            if log_x_axis and log_y_axis:
                suffix = " (log)"
            elif log_x_axis:
                suffix = " (log x)"
            elif log_y_axis:
                suffix = " (log y)"
            else:
                suffix = ""
            _set_title(ax, old_title + suffix)
    log_path = linear_path.parent / (linear_path.stem + "_log" + linear_path.suffix)
    fig.savefig(log_path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def plot_overall(
    overall: pd.DataFrame,
    color_map: dict[str, tuple] | None = None,
    marker_map: dict[str, str] | None = None,
    fill_map: dict[str, bool] | None = None,
    legend_info: dict | None = None,
    save_dir: Path = RESULTS_DIR,
):
    overall = overall.copy()
    if "label" not in overall.columns:
        overall["label"] = overall.apply(alg_label, axis=1)

    pareto_mask_overall = compute_pareto_front(
        overall["mean_relative_time"].tolist(),
        overall["mean_relative_wcss"].tolist(),
    )
    pareto_labels_overall: set[str] = {
        row["label"] for i, (_, row) in enumerate(overall.iterrows()) if pareto_mask_overall[i]
    }

    for metric, fname, title in [
        ("mean_relative_wcss", "overall_avg_wcss.png", "Mean Relative WCSS – overall"),
        ("mean_relative_time", "overall_avg_time.png", "Mean Relative Time – overall"),
    ]:
        _bar_chart(
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
        orig_labels = sdf["label"].tolist()
        display_labels = [
            "\u2605 " + to_math_label(lbl) if lbl in pareto_labels_overall else to_math_label(lbl)
            for lbl in orig_labels
        ]
        colors = (
            [color_map.get(lbl, "steelblue") for lbl in orig_labels] if color_map else "steelblue"
        )
        bars = ax.barh(display_labels, sdf[metric], color=colors, edgecolor="white")
        ax.bar_label(
            bars,
            labels=[_plain_number(v) for v in sdf[metric].tolist()],
            padding=3,
            fontsize=7,
        )
        ax.axvline(1.0, color="crimson", linestyle="--", linewidth=1.2, label="best=1.0")
        ax.set_xlabel(metric)
        _set_title(ax, title + " (overall)", force=True)
        ax.legend(fontsize=8, loc="upper right")
        _apply_no_sci(ax.xaxis)
    fig.tight_layout()
    for ax in axes:
        for tick in ax.yaxis.get_ticklabels():
            if tick.get_text().startswith("\u2605"):
                tick.set_fontweight("bold")
    fig.savefig(save_dir / "overall_avg.png", dpi=450, bbox_inches="tight")
    plt.close(fig)

    # Scatter: relative WCSS vs relative time
    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_scatter_ax(
        ax,
        overall,
        color_map,
        marker_map,
        fill_map,
        title="WCSS vs Time trade-off (overall)",
    )
    if legend_info is not None:
        _add_scatter_legends(ax, legend_info)
    fig.tight_layout()
    _save_with_log_variant(fig, [ax], save_dir / "overall_scatter.png")

    # Scatter with Pareto front (overall)
    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_scatter_ax(
        ax,
        overall,
        color_map,
        marker_map,
        fill_map,
        title="WCSS vs Time – Pareto front (overall)",
        pareto=True,
    )
    if legend_info is not None:
        _add_scatter_legends(ax, legend_info, has_pareto_line=True)
    fig.tight_layout()
    _save_with_log_variant(fig, [ax], save_dir / "overall_pareto.png")


def plot_by_k_multiplier(
    by_k_mult: pd.DataFrame,
    color_map: dict[str, tuple] | None = None,
    marker_map: dict[str, str] | None = None,
    fill_map: dict[str, bool] | None = None,
    legend_info: dict | None = None,
    save_dir: Path = RESULTS_DIR,
):
    by_k_mult = by_k_mult.copy()
    if "label" not in by_k_mult.columns:
        by_k_mult["label"] = by_k_mult.apply(alg_label, axis=1)
    k_mults_all = sorted(by_k_mult["k_multiplier"].unique())
    for metric, fname, title in [
        ("mean_relative_wcss", "by_k_multiplier_wcss.png", "Mean Relative WCSS by k_multiplier"),
        ("mean_relative_time", "by_k_multiplier_time.png", "Mean Relative Time by k_multiplier"),
    ]:
        # Bar charts: one subplot per k_multiplier
        n_km = len(k_mults_all)
        pareto_by_km: dict[float, set[str]] = {}
        for _km in k_mults_all:
            _sub = by_k_mult[by_k_mult["k_multiplier"] == _km].copy()
            _mask = compute_pareto_front(
                _sub["mean_relative_time"].tolist(), _sub["mean_relative_wcss"].tolist()
            )
            pareto_by_km[_km] = {
                row["label"] for i, (_, row) in enumerate(_sub.iterrows()) if _mask[i]
            }

        fig, axes = _create_panel_grid(
            n_km,
            width_per_col=7,
            height_per_row=max(5, by_k_mult["label"].nunique() * 0.35),
        )
        for ax, km in zip(axes, k_mults_all):
            sub = by_k_mult[by_k_mult["k_multiplier"] == km].copy()
            sub = sub.sort_values(metric)
            orig_labels = sub["label"].tolist()
            pareto_set_km = pareto_by_km.get(km, set())
            display_labels = [
                "\u2605 " + to_math_label(lbl) if lbl in pareto_set_km else to_math_label(lbl)
                for lbl in orig_labels
            ]
            colors = (
                [color_map.get(lbl, "steelblue") for lbl in orig_labels]
                if color_map
                else "steelblue"
            )
            bars = ax.barh(display_labels, sub[metric], color=colors, edgecolor="white")
            ax.bar_label(
                bars,
                labels=[_plain_number(v) for v in sub[metric].tolist()],
                padding=3,
                fontsize=7,
            )
            ax.axvline(1.0, color="crimson", linestyle="--", linewidth=1.2, label="best=1.0")
            _set_title(ax, f"k_multiplier = {km}", force=True)
            ax.set_xlabel(metric)
            ax.legend(fontsize=8, loc="upper right")
            _apply_no_sci(ax.xaxis)
        _set_suptitle(fig, title, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95] if SHOW_TITLES else None)
        for ax in axes:
            for tick in ax.yaxis.get_ticklabels():
                if tick.get_text().startswith("\u2605"):
                    tick.set_fontweight("bold")
        fig.savefig(save_dir / fname, dpi=450, bbox_inches="tight")
        plt.close(fig)

        # Line chart
        pivot = pivot_for_line(by_k_mult, x_col="k_multiplier", metric=metric)
        k_mults = sorted(pivot.columns)

        fig, ax = plt.subplots(figsize=(10, 6))
        for label, row in pivot.iterrows():
            vals = [row.get(km, np.nan) for km in k_mults]
            label_str = str(label)
            color = color_map.get(label_str) if color_map else None
            mkr = marker_map.get(label_str, "o") if marker_map else "o"
            plot_kwargs = {
                "marker": mkr,
                "label": to_math_label(label_str),
                "linewidth": 1.5,
                "markersize": 5,
            }
            if color is not None:
                plot_kwargs["color"] = color
            ax.plot(k_mults, vals, **plot_kwargs)

        ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.2, label="best=1.0")
        ax.set_xlabel("k_multiplier")
        ax.set_ylabel(metric)
        _set_title(ax, title + " (line)")
        ax.set_xticks(k_mults)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        fig.tight_layout()
        line_fname = fname.replace(".png", "_line.png")
        fig.savefig(save_dir / line_fname, dpi=450, bbox_inches="tight")
        plt.close(fig)

    # Scatter: relative WCSS vs relative time, one subplot per k_multiplier
    n_km = len(k_mults_all)
    fig, axes = _create_panel_grid(n_km, width_per_col=8, height_per_row=6)
    for ax, km in zip(axes, k_mults_all):
        sub = by_k_mult[by_k_mult["k_multiplier"] == km].copy()
        _draw_scatter_ax(
            ax,
            sub,
            color_map,
            marker_map,
            fill_map,
            title=f"k_multiplier = {km}",
            force_title=True,
        )
        if legend_info is not None:
            _add_scatter_legends(ax, legend_info, loc="upper right", bbox_to_anchor=(0.99, 0.99))
    _set_suptitle(fig, "WCSS vs Time trade-off by k_multiplier", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95] if SHOW_TITLES else None)
    _save_with_log_variant(fig, list(axes), save_dir / "by_k_multiplier_scatter.png")

    # Scatter with Pareto front per k_multiplier
    fig, axes = _create_panel_grid(n_km, width_per_col=8, height_per_row=6)
    for ax, km in zip(axes, k_mults_all):
        sub = by_k_mult[by_k_mult["k_multiplier"] == km].copy()
        _draw_scatter_ax(
            ax,
            sub,
            color_map,
            marker_map,
            fill_map,
            title=f"k_multiplier = {km}",
            force_title=True,
            pareto=True,
        )
        if legend_info is not None:
            _add_scatter_legends(
                ax,
                legend_info,
                has_pareto_line=True,
                loc="upper right",
                bbox_to_anchor=(0.99, 0.99),
            )
    _set_suptitle(fig, "WCSS vs Time – Pareto front by k_multiplier", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95] if SHOW_TITLES else None)
    _save_with_log_variant(fig, list(axes), save_dir / "by_k_multiplier_pareto.png")


def plot_by_size_bin(
    by_size: pd.DataFrame,
    color_map: dict[str, tuple] | None = None,
    marker_map: dict[str, str] | None = None,
    fill_map: dict[str, bool] | None = None,
    legend_info: dict | None = None,
    save_dir: Path = RESULTS_DIR,
):
    by_size = by_size.copy()
    if "label" not in by_size.columns:
        by_size["label"] = by_size.apply(alg_label, axis=1)
    size_bins = [b for b in SIZE_BIN_LABELS if b in by_size["size_bin"].values]

    # Precompute Pareto front labels per size bin for bar chart highlighting
    pareto_by_sb: dict[str, set[str]] = {}
    for _sb in size_bins:
        _sub = by_size[by_size["size_bin"] == _sb].copy()
        _mask = compute_pareto_front(
            _sub["mean_relative_time"].tolist(), _sub["mean_relative_wcss"].tolist()
        )
        pareto_by_sb[_sb] = {row["label"] for i, (_, row) in enumerate(_sub.iterrows()) if _mask[i]}

    for metric, fname, title in [
        ("mean_relative_wcss", "by_size_bin_wcss.png", "Mean Relative WCSS by dataset size"),
        ("mean_relative_time", "by_size_bin_time.png", "Mean Relative Time by dataset size"),
    ]:
        n = len(size_bins)
        if n == 0:
            continue

        fig, axes = _create_panel_grid(
            n,
            width_per_col=7,
            height_per_row=max(5, by_size["label"].nunique() * 0.35),
        )

        for ax, sb in zip(axes, size_bins):
            sub = by_size[by_size["size_bin"] == sb].copy()
            sub = sub.sort_values(metric)
            orig_labels = sub["label"].tolist()
            pareto_set_sb = pareto_by_sb.get(sb, set())
            display_labels = [
                "\u2605 " + to_math_label(lbl) if lbl in pareto_set_sb else to_math_label(lbl)
                for lbl in orig_labels
            ]
            colors = (
                [color_map.get(lbl, "steelblue") for lbl in orig_labels]
                if color_map
                else "steelblue"
            )
            bars = ax.barh(display_labels, sub[metric], color=colors, edgecolor="white")
            ax.bar_label(
                bars,
                labels=[_plain_number(v) for v in sub[metric].tolist()],
                padding=3,
                fontsize=7,
            )
            ax.axvline(1.0, color="crimson", linestyle="--", linewidth=1.2, label="best=1.0")
            _set_title(ax, f"Size Bin: {sb}", force=True)
            ax.set_xlabel(metric)
            ax.legend(fontsize=8, loc="upper right")
            _apply_no_sci(ax.xaxis)

        _set_suptitle(fig, title, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95] if SHOW_TITLES else None)
        for ax in axes:
            for tick in ax.yaxis.get_ticklabels():
                if tick.get_text().startswith("\u2605"):
                    tick.set_fontweight("bold")
        fig.savefig(save_dir / fname, dpi=450, bbox_inches="tight")
        plt.close(fig)

        # Line chart version (trend across size bins)
        pivot = pivot_for_line(by_size, x_col="size_bin", metric=metric)
        valid_bins = [b for b in SIZE_BIN_LABELS if b in pivot.columns]
        pivot = pivot[valid_bins]

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for label, row in pivot.iterrows():
            vals = [row.get(sb, np.nan) for sb in valid_bins]
            label_str = str(label)
            color = color_map.get(label_str) if color_map else None
            mkr = marker_map.get(label_str, "o") if marker_map else "o"
            plot_kwargs = {
                "marker": mkr,
                "label": to_math_label(label_str),
                "linewidth": 1.5,
                "markersize": 5,
            }
            if color is not None:
                plot_kwargs["color"] = color
            ax2.plot(valid_bins, vals, **plot_kwargs)

        ax2.axhline(1.0, color="crimson", linestyle="--", linewidth=1.2, label="best=1.0")
        ax2.set_xlabel("Dataset size bin")
        ax2.set_ylabel(metric)
        _set_title(ax2, title + " (line)")
        ax2.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        fig2.tight_layout()
        line_fname = fname.replace(".png", "_line.png")
        fig2.savefig(save_dir / line_fname, dpi=450, bbox_inches="tight")
        plt.close(fig2)

    # Scatter: relative WCSS vs relative time, one subplot per size bin
    n_sb = len(size_bins)
    if n_sb > 0:
        fig, axes = _create_panel_grid(n_sb, width_per_col=8, height_per_row=6)
        for ax, sb in zip(axes, size_bins):
            sub = by_size[by_size["size_bin"] == sb].copy()
            _draw_scatter_ax(
                ax,
                sub,
                color_map,
                marker_map,
                fill_map,
                title=f"size bin: {sb}",
                force_title=True,
            )
            if legend_info is not None:
                _add_scatter_legends(
                    ax, legend_info, loc="upper right", bbox_to_anchor=(0.99, 0.99)
                )
        _set_suptitle(fig, "WCSS vs Time trade-off by dataset size", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95] if SHOW_TITLES else None)
        _save_with_log_variant(fig, list(axes), save_dir / "by_size_bin_scatter.png")

    # Scatter with Pareto front per size bin
    if n_sb > 0:
        fig, axes = _create_panel_grid(n_sb, width_per_col=8, height_per_row=6)
        for ax, sb in zip(axes, size_bins):
            sub = by_size[by_size["size_bin"] == sb].copy()
            _draw_scatter_ax(
                ax,
                sub,
                color_map,
                marker_map,
                fill_map,
                title=f"size bin: {sb}",
                force_title=True,
                pareto=True,
            )
            if legend_info is not None:
                _add_scatter_legends(
                    ax,
                    legend_info,
                    has_pareto_line=True,
                    loc="upper right",
                    bbox_to_anchor=(0.99, 0.99),
                )
        _set_suptitle(fig, "WCSS vs Time – Pareto front by dataset size", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95] if SHOW_TITLES else None)
        _save_with_log_variant(fig, list(axes), save_dir / "by_size_bin_pareto.png")


# ---------------------------------------------------------------------------
# Generic component-level analysis
# ---------------------------------------------------------------------------


def analyze_grouping(
    df: pd.DataFrame,
    group_cols: list[str],
    label_fn,
    save_dir: Path,
    title_suffix: str,
    color_map: dict[str, tuple] | None = None,
    marker_map: dict[str, str] | None = None,
    fill_map: dict[str, bool] | None = None,
    legend_info: dict | None = None,
) -> None:
    """Run the full analysis pipeline (CSV + plots) for an arbitrary grouping.

    Parameters
    ----------
    df:
        DataFrame with wcss, time, dataset, k_multiplier, size_bin columns.
        Relative metrics are recomputed internally so that the best baseline
        reflects only the algorithms present in this filtered subset.
    group_cols:
        Columns to group by (e.g. ["label_selection_method"]).
    label_fn:
        Callable(row) -> str producing the display label for each group.
    save_dir:
        Directory where CSVs and PNGs for this grouping are written.
    title_suffix:
        Short string used in progress messages.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Recompute best wcss/time per (dataset, k_multiplier) within this subset
    df = df.copy()
    best = (
        df.groupby(["dataset", "k_multiplier"])
        .agg(best_wcss=("wcss", "min"), best_time=("time", "min"))
        .reset_index()
    )
    df = df.drop(
        columns=["best_wcss", "best_time", "relative_wcss", "relative_time"], errors="ignore"
    )
    df = df.merge(best, on=["dataset", "k_multiplier"])
    df["relative_wcss"] = df["wcss"] / df["best_wcss"]
    df["relative_time"] = df["time"] / df["best_time"]

    # Overall average
    overall = (
        df.groupby(group_cols)
        .agg(
            mean_relative_wcss=("relative_wcss", "mean"),
            mean_relative_time=("relative_time", "mean"),
        )
        .reset_index()
        .sort_values("mean_relative_wcss")
    )
    overall["label"] = overall.apply(label_fn, axis=1)
    overall.to_csv(save_dir / "overall_avg.csv", index=False)
    print(f"  [{title_suffix}] overall_avg.csv  ({len(overall)} rows)")

    # By k_multiplier
    by_k_mult = (
        df.groupby(group_cols + ["k_multiplier"])
        .agg(
            mean_relative_wcss=("relative_wcss", "mean"),
            mean_relative_time=("relative_time", "mean"),
        )
        .reset_index()
    )
    by_k_mult["label"] = by_k_mult.apply(label_fn, axis=1)
    by_k_mult.to_csv(save_dir / "by_k_multiplier.csv", index=False)
    print(f"  [{title_suffix}] by_k_multiplier.csv  ({len(by_k_mult)} rows)")

    # By size bin
    by_size = (
        df.groupby(group_cols + ["size_bin"], observed=True)
        .agg(
            mean_relative_wcss=("relative_wcss", "mean"),
            mean_relative_time=("relative_time", "mean"),
        )
        .reset_index()
    )
    by_size["label"] = by_size.apply(label_fn, axis=1)
    by_size["size_bin"] = pd.Categorical(
        by_size["size_bin"], categories=SIZE_BIN_LABELS, ordered=True
    )
    by_size = by_size.sort_values(["size_bin"] + group_cols)
    by_size.to_csv(save_dir / "by_size_bin.csv", index=False)
    print(f"  [{title_suffix}] by_size_bin.csv  ({len(by_size)} rows)")

    # Build local maps keyed by the simplified label (from label_fn).
    # For each group, pick a representative raw row and look up the global maps
    # via the full "algorithm | n_init=..." key so encoding is consistent.
    local_color_map: dict[str, tuple] = {}
    local_marker_map: dict[str, str] = {}
    local_fill_map: dict[str, bool] = {}
    for grp_vals, grp_df in df.groupby(group_cols, sort=False):
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
    print(f"  [{title_suffix}] plots saved to {save_dir}/")


# ---------------------------------------------------------------------------
# Special-metric benchmarks (absolute, no aggregation by k_mult / size_bin)
# ---------------------------------------------------------------------------


SPECIAL_METRICS = [
    ("avg_dist_to_centroid_m", "Mean distance to centroid (m)"),
    ("max_dist_to_centroid_m", "Max distance to centroid (m)"),
    ("mean_max_dist_per_label_m", "Mean max distance per label (m)"),
]


def _load_special_metric_metadata(dataset_prefix: str, metric_keys: list[str]) -> pd.DataFrame:
    """Scan output/ for special metric metadata rows containing all requested keys."""
    rows = []
    for meta_path in sorted(OUTPUT_DIR.rglob("metadata.json")):
        with open(meta_path) as f:
            meta = json.load(f)
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
    if requested_n_inits is None:
        values = [1, available[-1]]
    else:
        values = requested_n_inits
    resolved = sorted({int(v) for v in values if int(v) in available})
    missing = sorted({int(v) for v in values if int(v) not in available})
    if missing:
        print(f"  Requested n_init values not found and skipped: {missing}")
    return resolved


def _time_comparison_number(value: float, _pos: int | None = None) -> str:
    """Format comparison time values with precision based on their magnitude."""
    if not np.isfinite(value):
        return ""
    abs_value = abs(value)
    if abs_value < 10:
        return f"{value:.2f}"
    if abs_value < 100:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _metric_comparison_number(value: float, _pos: int | None = None) -> str:
    """Format comparison metric values with precision based on their magnitude."""
    if not np.isfinite(value):
        return ""
    abs_value = abs(value)
    if abs_value < 1_000:
        return f"{value:.2f}"
    if abs_value < 10_000:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _special_metric_axis_step(separation: float) -> int:
    if separation < 100:
        return 25
    if separation < 1_000:
        return 250
    return 2_500


def _metric_axis_bounds(values: pd.Series) -> tuple[float, float] | None:
    finite_values = values[np.isfinite(values)]
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


def _bp_algorithm_from_spec(spec: str) -> str:
    spec = spec.strip()
    if spec.startswith("BP-KMeans"):
        return spec

    parts = [part.strip() for part in re.split(r"[,/]", spec) if part.strip()]
    if len(parts) != 3:
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
    return bp_agg.iloc[0]


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
        print(f"  n_init={n_init}: requested BP-KMeans combination not found: {manual_algorithm}")
        return None

    agg = manual_rows[[key, "time"]].mean()
    return pd.Series(
        {
            "algorithm": manual_algorithm,
            f"mean_{key}": agg[key],
            "mean_time": agg["time"],
        }
    )


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

    requested_n_inits = n_inits
    if requested_n_inits is None and manual_bp_algorithms:
        requested_n_inits = sorted(manual_bp_algorithms)
    selected_n_inits = _resolve_special_n_inits(df, requested_n_inits)
    if not selected_n_inits:
        print(f"  [{title_prefix}] no selected n_init values for {key}. Skipping.")
        return

    rows = []
    for n_init in selected_n_inits:
        base_rows = df[(df["algorithm"] == "Bisecting KMeans") & (df["n_init"] == n_init)]
        bp_rows = df[(df["algorithm"].str.startswith("BP-KMeans")) & (df["n_init"] == n_init)]
        if base_rows.empty or bp_rows.empty:
            print(
                f"  [{title_prefix}] n_init={n_init}: missing "
                "Bisecting KMeans or BP-KMeans data; skipped."
            )
            continue

        base_agg = base_rows[[key, "time"]].mean()
        rows.append(
            {
                "n_init": n_init,
                "kind": "Bisecting K-Means",
                "algorithm": "Bisecting KMeans",
                f"mean_{key}": float(base_agg[key]),
                "mean_time": float(base_agg["time"]),
            }
        )

        best_bp = _best_bp_comparison_row(bp_rows, key)

        rows.append(
            {
                "n_init": n_init,
                "kind": "BP-KMeans - per-problem best",
                "algorithm": best_bp["algorithm"],
                f"mean_{key}": float(best_bp[f"mean_{key}"]),
                "mean_time": float(best_bp["mean_time"]),
            }
        )

        manual_bp = _manual_bp_comparison_row(
            bp_rows,
            n_init,
            key,
            manual_bp_algorithms,
        )
        if manual_bp is not None:
            rows.append(
                {
                    "n_init": n_init,
                    "kind": "BP-KMeans - globally tuned",
                    "algorithm": manual_bp["algorithm"],
                    f"mean_{key}": float(manual_bp[f"mean_{key}"]),
                    "mean_time": float(manual_bp["mean_time"]),
                }
            )

    comp = pd.DataFrame(rows)
    if comp.empty:
        print(f"  [{title_prefix}] no complete {key}/time comparison rows. Skipping.")
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
    metric_offsets = {
        "Bisecting K-Means": -2.75 * width,
        "BP-KMeans - per-problem best": -1.75 * width,
        "BP-KMeans - globally tuned": -0.75 * width,
    }
    time_offsets = {
        "Bisecting K-Means": 0.75 * width,
        "BP-KMeans - per-problem best": 1.75 * width,
        "BP-KMeans - globally tuned": 2.75 * width,
    }

    fig, ax_metric = plt.subplots(figsize=(max(8, n_groups * 2.6), 5.5))
    ax_time = ax_metric.twinx()

    for kind in plotted_kinds:
        sub = comp[comp["kind"] == kind].set_index("n_init").reindex(plotted_n_inits)
        metric_bars = ax_metric.bar(
            x + metric_offsets[kind],
            sub[f"mean_{key}"],
            width,
            label=f"{kind} metric",
            color=colors[kind],
            edgecolor="white",
        )
        time_bars = ax_time.bar(
            x + time_offsets[kind],
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

    metric_bounds = _metric_axis_bounds(comp[f"mean_{key}"])
    if metric_bounds is not None:
        ax_metric.set_ylim(metric_bounds)
    ax_time.set_yscale("log")
    ax_metric.set_ylabel(metric_label)
    ax_time.set_ylabel("Mean Time")
    ax_metric.set_xlabel("Number of Initializations")
    ax_metric.set_xticks(x)
    ax_metric.set_xticklabels([str(v) for v in plotted_n_inits])
    _set_title(ax_metric, f"{title_prefix} - {metric_label} and time comparison")
    ax_metric.yaxis.set_major_formatter(mticker.FuncFormatter(_metric_comparison_number))
    ax_time.yaxis.set_major_formatter(mticker.FuncFormatter(_time_comparison_number))
    ax_time.yaxis.set_minor_formatter(mticker.NullFormatter())

    legend_handles: list = []
    legend_labels: list[str] = []
    for kind in plotted_kinds:
        legend_handles.append(mpatches.Patch(facecolor=colors[kind], edgecolor="white"))
        legend_labels.append(kind)
    legend_handles.extend(
        [
            mpatches.Patch(facecolor="white", edgecolor="black"),
            mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", alpha=0.45),
        ]
    )
    legend_labels.extend([metric_label, "Time"])
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
    print(f"  [{title_prefix}] {safe_key}_time_comparison.csv + .png saved")


def analyze_special_metric(
    dataset_prefix: str,
    metric_keys: list[tuple[str, str]],
    save_dir: Path,
    title_prefix: str,
    *,
    kpp_only: bool = False,
    color_map: dict[str, tuple] | None = None,
    marker_map: dict[str, str] | None = None,
    fill_map: dict[str, bool] | None = None,
    legend_info: dict | None = None,
    comparison_n_inits: list[int] | None = None,
    comparison_bp_algorithms: dict[int, str] | None = None,
) -> None:
    """Analyze special metrics in absolute terms per algorithm × n_init.

    No relative normalization, no breakdown by k_multiplier or size bin.
    *metric_keys* is a list of (key, label) pairs, each producing its own
    bar chart, scatter and Pareto plot. The primary metric (first entry)
    is also used as the Y axis of the shared scatter/Pareto.
    """
    keys = [k for k, _ in metric_keys]
    df = _load_special_metric_metadata(dataset_prefix, keys)
    if df.empty:
        print(f"  No data for dataset prefix '{dataset_prefix}'. Skipping.")
        return

    if kpp_only:
        df_bp_parsed = parse_algorithm_components(df)
        df = df_bp_parsed[
            (~df_bp_parsed["algorithm"].str.startswith("BP-KMeans"))
            | (df_bp_parsed["init_algo"] == "KMEANS_PLUS_PLUS")
        ].copy()
        if df.empty:
            print(f"  No KMeans++ data for dataset prefix '{dataset_prefix}'. Skipping.")
            return

    save_dir.mkdir(parents=True, exist_ok=True)
    df["label"] = df.apply(alg_label, axis=1)
    local_legend_info = (
        build_legend_info(df, color_map, marker_map, fill_map)
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
    print(f"  [{title_prefix}] overall_avg.csv  ({len(agg)} rows)")

    orig_labels = agg.sort_values(f"mean_{keys[0]}")["label"].tolist()
    colors = (
        [color_map.get(lbl, "steelblue") for lbl in orig_labels]
        if color_map
        else ["steelblue"] * len(orig_labels)
    )

    for key, metric_label in metric_keys:
        metric_col = f"mean_{key}"
        sub = agg.set_index("label").loc[orig_labels].reset_index()

        # Bar chart
        n_bars = len(sub)
        fig, ax = plt.subplots(figsize=(9, max(4, n_bars * 0.35)))
        bars = ax.barh(
            [to_math_label(label) for label in sub["label"]],
            sub[metric_col],
            color=colors,
            edgecolor="white",
        )
        ax.bar_label(
            bars,
            labels=[_plain_number(v) for v in sub[metric_col].tolist()],
            padding=3,
            fontsize=7,
        )
        ax.set_xlabel(metric_label)
        _set_title(ax, f"{title_prefix} – {metric_label}")
        _apply_no_sci(ax.xaxis)
        fig.tight_layout()
        safe_key = key.replace("/", "_")
        fig.savefig(save_dir / f"{safe_key}.png", dpi=450, bbox_inches="tight")
        plt.close(fig)

        # Scatter: time vs this metric
        fig, ax = plt.subplots(figsize=(10, 7))
        _draw_scatter_ax(
            ax,
            agg,
            color_map,
            marker_map,
            fill_map,
            x_col="mean_time",
            y_col=metric_col,
            x_label="Mean time (s)",
            y_label=metric_label,
            title=f"{title_prefix} – {metric_label} vs time",
            pareto=False,
            add_reference_lines=False,
        )
        if local_legend_info is not None:
            _add_scatter_legends(ax, local_legend_info)
        fig.tight_layout()
        _save_with_log_variant(
            fig,
            [ax],
            save_dir / f"{safe_key}_scatter.png",
            log_x_axis=True,
            log_y_axis=False,
        )

        # Pareto front
        fig, ax = plt.subplots(figsize=(10, 7))
        _draw_scatter_ax(
            ax,
            agg,
            color_map,
            marker_map,
            fill_map,
            x_col="mean_time",
            y_col=metric_col,
            x_label="Mean time (s)",
            y_label=metric_label,
            title=f"{title_prefix} – {metric_label} Pareto front",
            pareto=True,
            floor_at_one=False,
            add_reference_lines=False,
        )
        if local_legend_info is not None:
            _add_scatter_legends(ax, local_legend_info, has_pareto_line=True)
        fig.tight_layout()
        _save_with_log_variant(
            fig,
            [ax],
            save_dir / f"{safe_key}_pareto.png",
            log_x_axis=True,
            log_y_axis=False,
        )

        _plot_special_metric_time_comparison(
            df,
            key,
            metric_label,
            save_dir,
            title_prefix,
            n_inits=comparison_n_inits,
            manual_bp_algorithms=comparison_bp_algorithms,
        )

        print(f"  [{title_prefix}] {safe_key}: bar + scatter + pareto + comparison saved")


def analyze_hac_strength_benchmark(show_titles: bool = False) -> None:
    """Run the regular relative analysis using only HAC-strength benchmark rows."""
    global SHOW_TITLES
    SHOW_TITLES = show_titles

    df = load_hac_strength_metadata()
    if df.empty:
        print("  No HAC-strength benchmark metadata found. Skipping.")
        return

    save_dir = RESULTS_DIR / "hac_strength"
    save_dir.mkdir(parents=True, exist_ok=True)

    sizes = load_dataset_sizes()
    df["n_instances"] = df["dataset"].map(sizes)
    missing = df["n_instances"].isna().sum()
    if missing:
        print(
            f"  WARNING: {missing} HAC-strength rows could not be matched "
            "to a dataset parquet file."
        )

    df["n_labels"] = df["n_labels"].fillna(df["n_clusters"] / df["k_multiplier"])
    df["size_bin"] = [
        assign_size_bin(n, int(k)) for n, k in zip(df["n_instances"], df["n_labels"])
    ]

    best = (
        df.groupby(["dataset", "k_multiplier"])
        .agg(best_wcss=("wcss", "min"), best_time=("time", "min"))
        .reset_index()
    )
    rel = df.merge(best, on=["dataset", "k_multiplier"])
    rel["relative_wcss"] = rel["wcss"] / rel["best_wcss"]
    rel["relative_time"] = rel["time"] / rel["best_time"]

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
    rel[rel_cols].sort_values(["dataset", "algorithm", "n_init"]).to_csv(
        save_dir / "relative_metrics.csv", index=False
    )
    print(f"  [hac_strength] relative_metrics.csv  ({len(rel)} rows)")

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
        title_suffix="hac_strength",
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
        raise argparse.ArgumentTypeError("n_init values must be integers") from exc
    if not n_inits:
        raise argparse.ArgumentTypeError("provide at least one n_init value")
    if any(n < 1 for n in n_inits):
        raise argparse.ArgumentTypeError("n_init values must be >= 1")
    return n_inits


def parse_bp_combination_map(value: str) -> dict[int, str]:
    """Parse n_init-to-BP-KMeans-combination mapping for CLI use."""
    mapping: dict[int, str] = {}
    for raw_entry in value.split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise argparse.ArgumentTypeError("BP combinations must use n_init=combination entries")
        raw_n_init, raw_spec = entry.split("=", 1)
        try:
            n_init = int(raw_n_init.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError("BP combination n_init keys must be integers") from exc
        if n_init < 1:
            raise argparse.ArgumentTypeError("BP combination n_init keys must be >= 1")
        mapping[n_init] = _bp_algorithm_from_spec(raw_spec)

    if not mapping:
        raise argparse.ArgumentTypeError("provide at least one BP combination entry")
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze benchmark results.")
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
        default="1=R_C,I_CRI,KMEANS_PLUS_PLUS;32=R_RL,I_ACL,KMEANS_PLUS_PLUS",
        help=(
            "Semicolon-separated manual BP-KMeans combination per n_init, e.g. "
            "'1=R_L,I_LRI,KMEANS_PLUS_PLUS;32=R_C,I_ACC,KMEANS_PLUS_PLUS'. "
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
    special_n_inits: list[int] | None = None,
    special_bp_combinations: dict[int, str] | None = None,
    show_titles: bool = False,
):
    global SHOW_TITLES
    SHOW_TITLES = show_titles

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    print("Loading metadata files…")
    df = load_all_metadata()
    if df.empty:
        print(f"No metadata.json files found under {OUTPUT_DIR}/. Run the benchmark first.")
        return
    print(f"  Loaded {len(df)} runs across {df['dataset'].nunique()} datasets.")

    if EXCLUDE_HAC:
        before = len(df)
        df = df[~df["algorithm"].str.contains("HAC", case=False)].copy()
        print(f"  Excluded HAC Ward runs ({before - len(df)} rows removed).")

    # Add dataset sizes
    print("Loading dataset sizes…")
    sizes = load_dataset_sizes()
    df["n_instances"] = df["dataset"].map(sizes)
    missing = df["n_instances"].isna().sum()
    if missing:
        print(f"  WARNING: {missing} rows could not be matched to a dataset parquet file.")

    # 2. Best wcss and time per (dataset, k_multiplier) across all alg × n_init
    best = (
        df.groupby(["dataset", "k_multiplier"])
        .agg(best_wcss=("wcss", "min"), best_time=("time", "min"))
        .reset_index()
    )
    df = df.merge(best, on=["dataset", "k_multiplier"])

    # 3. Relative metrics
    df["relative_wcss"] = df["wcss"] / df["best_wcss"]
    df["relative_time"] = df["time"] / df["best_time"]

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
    df[rel_cols].sort_values(["dataset", "k_multiplier", "algorithm", "n_init"]).to_csv(
        RESULTS_DIR / "relative_metrics.csv", index=False
    )
    print(f"  Saved relative_metrics.csv  ({len(df)} rows)")

    alg_init = ["algorithm", "n_init"]

    df["n_labels"] = df["n_labels"].fillna(df["n_clusters"] / df["k_multiplier"])
    df["size_bin"] = [assign_size_bin(n, k) for n, k in zip(df["n_instances"], df["n_labels"])]

    # 4. Plots + CSVs (overall, by_k_multiplier, by_size_bin)
    print("Generating plots…")
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
        group_cols=alg_init,
        label_fn=alg_label,
        save_dir=RESULTS_DIR,
        title_suffix="overall",
        **_ag_kwargs,
    )

    # -----------------------------------------------------------------------
    # 6. Additional analyses: BP-KMeans component groupings
    # Base: R_L, I_LRI, KMEANS_PLUS_PLUS, n_init=4
    # Each sub-analysis varies exactly one dimension from this base.
    # -----------------------------------------------------------------------
    print("\nGenerating component-level analyses for BP-KMeans…")
    df_bp = df[df["algorithm"].str.startswith("BP-KMeans")].copy()
    df_bp = parse_algorithm_components(df_bp)

    # 6a. Vary label_selection_method (fix: I_LRI, KMEANS_PLUS_PLUS, n_init=4)
    analyze_grouping(
        df_bp[
            (df_bp["reinit_method"] == "I_LRI")
            & (df_bp["init_algo"] == "KMEANS_PLUS_PLUS")
            & (df_bp["n_init"] == 4)
        ].copy(),
        group_cols=["label_selection_method"],
        label_fn=lambda row: row["label_selection_method"],
        save_dir=RESULTS_DIR / "vary_label_sel",
        title_suffix="vary label_sel | I_LRI, kpp, n_init=4",
        **_ag_kwargs,
    )

    # 6b. Vary reinit_method (fix: R_L, KMEANS_PLUS_PLUS, n_init=4)
    analyze_grouping(
        df_bp[
            (df_bp["label_selection_method"] == "R_L")
            & (df_bp["init_algo"] == "KMEANS_PLUS_PLUS")
            & (df_bp["n_init"] == 4)
        ].copy(),
        group_cols=["reinit_method"],
        label_fn=lambda row: row["reinit_method"],
        save_dir=RESULTS_DIR / "vary_reinit",
        title_suffix="vary reinit | R_L, kpp, n_init=4",
        **_ag_kwargs,
    )

    # 6c. Vary sampling method / init_algo (fix: R_L, I_LRI, n_init=4)
    analyze_grouping(
        df_bp[
            (df_bp["label_selection_method"] == "R_L")
            & (df_bp["reinit_method"] == "I_LRI")
            & (df_bp["n_init"] == 4)
        ].copy(),
        group_cols=["init_algo"],
        label_fn=lambda row: row["init_algo"],
        save_dir=RESULTS_DIR / "vary_init_algo",
        title_suffix="vary init_algo | R_L, I_LRI, n_init=4",
        **_ag_kwargs,
    )

    # 6d. Vary n_init (fix: R_L, I_LRI, KMEANS_PLUS_PLUS)
    analyze_grouping(
        df_bp[
            (df_bp["label_selection_method"] == "R_L")
            & (df_bp["reinit_method"] == "I_LRI")
            & (df_bp["init_algo"] == "KMEANS_PLUS_PLUS")
        ].copy(),
        group_cols=["n_init"],
        label_fn=lambda row: f"n_init={row['n_init']}",
        save_dir=RESULTS_DIR / "vary_n_init",
        title_suffix="vary n_init | R_L, I_LRI, kpp",
        **_ag_kwargs,
    )

    # 6e. Vary label_selection_method × reinit_method (fix: KMEANS_PLUS_PLUS, n_init=4)
    analyze_grouping(
        df_bp[(df_bp["init_algo"] == "KMEANS_PLUS_PLUS") & (df_bp["n_init"] == 4)].copy(),
        group_cols=["label_selection_method", "reinit_method"],
        label_fn=lambda row: f"{row['label_selection_method']} × {row['reinit_method']}",
        save_dir=RESULTS_DIR / "vary_label_sel_x_reinit",
        title_suffix="vary label_sel × reinit | kpp, n_init=4",
        **_ag_kwargs,
    )

    # 6f. Vary label_selection_method × reinit_method (fix: KMEANS_PLUS_PLUS, n_init=32)
    analyze_grouping(
        df_bp[(df_bp["init_algo"] == "KMEANS_PLUS_PLUS") & (df_bp["n_init"] == 32)].copy(),
        group_cols=["label_selection_method", "reinit_method"],
        label_fn=lambda row: f"{row['label_selection_method']} × {row['reinit_method']}",
        save_dir=RESULTS_DIR / "vary_label_sel_x_reinit_32",
        title_suffix="vary label_sel × reinit | kpp, n_init=32",
        **_ag_kwargs,
    )

    # 6g. Vary label_selection_method × reinit_method (fix: KMEANS_PLUS_PLUS, n_init=1)
    analyze_grouping(
        df_bp[(df_bp["init_algo"] == "KMEANS_PLUS_PLUS") & (df_bp["n_init"] == 1)].copy(),
        group_cols=["label_selection_method", "reinit_method"],
        label_fn=lambda row: f"{row['label_selection_method']} × {row['reinit_method']}",
        save_dir=RESULTS_DIR / "vary_label_sel_x_reinit_1",
        title_suffix="vary label_sel × reinit | kpp, n_init=1",
        **_ag_kwargs,
    )

    # -----------------------------------------------------------------------
    # 7. Overall analysis restricted to KMEANS_PLUS_PLUS init_algo
    # -----------------------------------------------------------------------
    print("\nGenerating KMeans++ subset overall analysis…")
    df_bp_parsed = parse_algorithm_components(df.copy())
    df_kpp = df_bp_parsed[
        (~df_bp_parsed["algorithm"].str.startswith("BP-KMeans"))
        | (df_bp_parsed["init_algo"] == "KMEANS_PLUS_PLUS")
    ].copy()
    analyze_grouping(
        df_kpp,
        group_cols=["algorithm", "n_init"],
        label_fn=alg_label,
        save_dir=RESULTS_DIR / "kpp_only",
        title_suffix="kpp_only",
        **_ag_kwargs,
    )

    print(f"\nAll results saved to {RESULTS_DIR}/")

    # -----------------------------------------------------------------------
    # 8. Special benchmarks: absolute metric, no aggregation by k_mult/size
    # -----------------------------------------------------------------------
    print("\nGenerating special-metric analyses…")
    analyze_special_metric(
        dataset_prefix="com_madrid_osm_drive_nodes_split_split",
        metric_keys=SPECIAL_METRICS,
        save_dir=RESULTS_DIR / "com_madrid_distance_metrics",
        title_prefix="Community of Madrid",
        color_map=color_map,
        marker_map=marker_map,
        fill_map=fill_map,
        legend_info=legend_info,
        comparison_n_inits=special_n_inits,
        comparison_bp_algorithms=special_bp_combinations,
    )
    analyze_special_metric(
        dataset_prefix="com_madrid_osm_drive_nodes_split_split",
        metric_keys=SPECIAL_METRICS,
        save_dir=RESULTS_DIR / "com_madrid_distance_metrics_kpp_only",
        title_prefix="Community of Madrid (KMeans++ only)",
        kpp_only=True,
        color_map=color_map,
        marker_map=marker_map,
        fill_map=fill_map,
        legend_info=legend_info,
        comparison_n_inits=special_n_inits,
        comparison_bp_algorithms=special_bp_combinations,
    )
    analyze_special_metric(
        dataset_prefix="castile_and_leon_osm_drive_nodes",
        metric_keys=SPECIAL_METRICS,
        save_dir=RESULTS_DIR / "castile_leon_distance_metrics",
        title_prefix="Castile and León",
        color_map=color_map,
        marker_map=marker_map,
        fill_map=fill_map,
        legend_info=legend_info,
        comparison_n_inits=special_n_inits,
        comparison_bp_algorithms=special_bp_combinations,
    )
    analyze_special_metric(
        dataset_prefix="castile_and_leon_osm_drive_nodes",
        metric_keys=SPECIAL_METRICS,
        save_dir=RESULTS_DIR / "castile_leon_distance_metrics_kpp_only",
        title_prefix="Castile and León (KMeans++ only)",
        kpp_only=True,
        color_map=color_map,
        marker_map=marker_map,
        fill_map=fill_map,
        legend_info=legend_info,
        comparison_n_inits=special_n_inits,
        comparison_bp_algorithms=special_bp_combinations,
    )

    # -----------------------------------------------------------------------
    # 9. HAC-strength benchmark analysis: opt in to only the isolated outputs
    # -----------------------------------------------------------------------
    print("\nGenerating HAC-strength benchmark analysis…")
    analyze_hac_strength_benchmark(show_titles=show_titles)


if __name__ == "__main__":
    args = parse_args()
    main(
        special_n_inits=args.special_n_inits,
        special_bp_combinations=args.special_bp_combinations,
        show_titles=args.show_titles,
    )
