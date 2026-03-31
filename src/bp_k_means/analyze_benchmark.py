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
by_k_multiplier_wcss.png      – line chart for 4b (wcss)
by_k_multiplier_time.png      – line chart for 4b (time)
by_size_bin_wcss.png          – bar charts for 4c (wcss)
by_size_bin_time.png          – bar charts for 4c (time)
"""

import colorsys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("output")
RESULTS_DIR = Path("output/analysis")
DATA_DIR = Path("data/datasets")

SIZE_BIN_EDGES = [0, 5_000, 20_000, 100_000, np.inf]
SIZE_BIN_LABELS = ["<5k", "5k–20k", "20k–100k", ">100k"]

# Set to True to exclude HAC Ward runs from all analyses and plots
EXCLUDE_HAC = True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_metadata() -> pd.DataFrame:
    rows = []
    for meta_path in sorted(OUTPUT_DIR.rglob("metadata.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        rows.append(
            {
                "dataset": meta["dataset"],
                "algorithm": meta["algorithm"],
                "n_init": int(meta["n_init"]),
                "k_multiplier": float(meta["k_multiplier"]),
                "k": int(meta["k"]),
                "n_clusters": int(meta["n_clusters"]),
                "wcss": float(meta["wcss_total"]),
                "time": float(meta["duration_seconds"]),
            }
        )
    return pd.DataFrame(rows)


def load_dataset_sizes() -> dict[str, int]:
    import pyarrow.parquet as pq

    sizes: dict[str, int] = {}
    for path in DATA_DIR.glob("*.parquet"):
        try:
            meta = pq.read_metadata(path)
            sizes[path.stem] = meta.num_rows
        except Exception as e:
            print(f"  WARNING: could not read size for {path.name}: {e}")
    return sizes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def alg_label(row: pd.Series) -> str:
    return f"{row['algorithm']} | n_init={row['n_init']}"


def build_color_map(df: pd.DataFrame) -> dict[str, tuple]:
    """Return a label-string → RGB color dict.

    Each unique algorithm gets a distinct hue from tab10.
    n_init variants within the same algorithm are rendered as shades of that
    hue: darker for low n_init, lighter for high n_init.
    """
    unique_algs = sorted(df["algorithm"].unique())
    unique_n_inits = sorted(df["n_init"].unique())
    n_levels = len(unique_n_inits)
    base_cmap = plt.cm.tab10
    color_map: dict[str, tuple] = {}
    for alg_idx, alg in enumerate(unique_algs):
        r, g, b, _ = base_cmap(alg_idx % base_cmap.N)
        h, _l, s = colorsys.rgb_to_hls(r, g, b)
        for init_idx, n_init in enumerate(unique_n_inits):
            if n_levels == 1:
                new_l = _l
            else:
                # 0.25 (darkest / low n_init) → 0.70 (lightest / high n_init)
                new_l = 0.25 + 0.45 * (init_idx / (n_levels - 1))
            nr, ng, nb = colorsys.hls_to_rgb(h, new_l, min(s, 0.9))
            color_map[f"{alg} | n_init={n_init}"] = (nr, ng, nb)
    return color_map


def pivot_for_line(df: pd.DataFrame, x_col: str, metric: str) -> pd.DataFrame:
    """Return a (label × x_col) pivot suitable for line plots."""
    df = df.copy()
    df["label"] = df.apply(alg_label, axis=1)
    return df.pivot_table(index="label", columns=x_col, values=metric, aggfunc="mean")


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
):
    sorted_df = df.sort_values(metric)
    n_bars = len(sorted_df)
    colors = (
        [color_map.get(lbl, "steelblue") for lbl in sorted_df[label_col]]
        if color_map
        else "steelblue"
    )
    fig, ax = plt.subplots(figsize=(9, max(4, n_bars * 0.35)))
    bars = ax.barh(sorted_df[label_col], sorted_df[metric], color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)
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
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overall(overall: pd.DataFrame, color_map: dict[str, tuple] | None = None):
    overall = overall.copy()
    overall["label"] = overall.apply(alg_label, axis=1)

    for metric, fname, title in [
        ("mean_relative_wcss", "overall_avg_wcss.png", "Mean Relative WCSS – overall"),
        ("mean_relative_time", "overall_avg_time.png", "Mean Relative Time – overall"),
    ]:
        _bar_chart(
            overall,
            metric=metric,
            title=title,
            xlabel=metric,
            save_path=RESULTS_DIR / fname,
            color_map=color_map,
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
        colors = (
            [color_map.get(lbl, "steelblue") for lbl in sdf["label"]] if color_map else "steelblue"
        )
        bars = ax.barh(sdf["label"], sdf[metric], color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)
        ax.axvline(1.0, color="crimson", linestyle="--", linewidth=1.2, label="best=1.0")
        ax.set_xlabel(metric)
        ax.set_title(title + " (overall)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "overall_avg.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Scatter: relative WCSS vs relative time
    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in overall.iterrows():
        color = color_map.get(row["label"], "steelblue") if color_map else "steelblue"
        ax.scatter(
            row["mean_relative_time"],
            row["mean_relative_wcss"],
            color=color,
            s=80,
            zorder=3,
        )
        ax.annotate(
            row["label"],
            xy=(row["mean_relative_time"], row["mean_relative_wcss"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )
    ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axvline(1.0, color="crimson", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Mean Relative Time")
    ax.set_ylabel("Mean Relative WCSS")
    ax.set_title("WCSS vs Time trade-off (overall)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "overall_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_by_k_multiplier(by_k_mult: pd.DataFrame, color_map: dict[str, tuple] | None = None):
    for metric, fname, title in [
        ("mean_relative_wcss", "by_k_multiplier_wcss.png", "Mean Relative WCSS by k_multiplier"),
        ("mean_relative_time", "by_k_multiplier_time.png", "Mean Relative Time by k_multiplier"),
    ]:
        pivot = pivot_for_line(by_k_mult, x_col="k_multiplier", metric=metric)
        k_mults = sorted(pivot.columns)

        fig, ax = plt.subplots(figsize=(10, 6))
        for label, row in pivot.iterrows():
            vals = [row.get(km, np.nan) for km in k_mults]
            color = color_map.get(label) if color_map else None
            ax.plot(
                k_mults,
                vals,
                marker="o",
                label=label,
                linewidth=1.5,
                markersize=5,
                **({"color": color} if color is not None else {}),
            )

        ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1.2, label="best=1.0")
        ax.set_xlabel("k_multiplier")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticks(k_mults)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_by_size_bin(by_size: pd.DataFrame, color_map: dict[str, tuple] | None = None):
    size_bins = [b for b in SIZE_BIN_LABELS if b in by_size["size_bin"].values]

    for metric, fname, title in [
        ("mean_relative_wcss", "by_size_bin_wcss.png", "Mean Relative WCSS by dataset size"),
        ("mean_relative_time", "by_size_bin_time.png", "Mean Relative Time by dataset size"),
    ]:
        n = len(size_bins)
        if n == 0:
            continue

        fig, axes = plt.subplots(
            1, n, figsize=(7 * n, max(5, by_size["algorithm"].nunique() * 0.35)), sharey=False
        )
        if n == 1:
            axes = [axes]

        for ax, sb in zip(axes, size_bins):
            sub = by_size[by_size["size_bin"] == sb].copy()
            sub["label"] = sub.apply(alg_label, axis=1)
            sub = sub.sort_values(metric)
            colors = (
                [color_map.get(lbl, "steelblue") for lbl in sub["label"]]
                if color_map
                else "steelblue"
            )
            bars = ax.barh(sub["label"], sub[metric], color=colors, edgecolor="white")
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)
            ax.axvline(1.0, color="crimson", linestyle="--", linewidth=1.2)
            ax.set_title(f"size bin: {sb}")
            ax.set_xlabel(metric)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Line chart version (trend across size bins)
        pivot = pivot_for_line(by_size, x_col="size_bin", metric=metric)
        valid_bins = [b for b in SIZE_BIN_LABELS if b in pivot.columns]
        pivot = pivot[valid_bins]

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for label, row in pivot.iterrows():
            vals = [row.get(sb, np.nan) for sb in valid_bins]
            color = color_map.get(label) if color_map else None
            ax2.plot(
                valid_bins,
                vals,
                marker="o",
                label=label,
                linewidth=1.5,
                markersize=5,
                **({"color": color} if color is not None else {}),
            )

        ax2.axhline(1.0, color="crimson", linestyle="--", linewidth=1.2, label="best=1.0")
        ax2.set_xlabel("Dataset size bin")
        ax2.set_ylabel(metric)
        ax2.set_title(title + " (line)")
        ax2.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        fig2.tight_layout()
        line_fname = fname.replace(".png", "_line.png")
        fig2.savefig(RESULTS_DIR / line_fname, dpi=150, bbox_inches="tight")
        plt.close(fig2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
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

    # 4a. Overall average
    overall = (
        df.groupby(alg_init)
        .agg(
            mean_relative_wcss=("relative_wcss", "mean"),
            mean_relative_time=("relative_time", "mean"),
        )
        .reset_index()
        .sort_values("mean_relative_wcss")
    )
    overall.to_csv(RESULTS_DIR / "overall_avg.csv", index=False)
    print(f"  Saved overall_avg.csv  ({len(overall)} rows)")

    # 4b. By k_multiplier
    by_k_mult = (
        df.groupby(alg_init + ["k_multiplier"])
        .agg(
            mean_relative_wcss=("relative_wcss", "mean"),
            mean_relative_time=("relative_time", "mean"),
        )
        .reset_index()
    )
    by_k_mult.to_csv(RESULTS_DIR / "by_k_multiplier.csv", index=False)
    print(f"  Saved by_k_multiplier.csv  ({len(by_k_mult)} rows)")

    # 4c. By dataset size bin
    df["size_bin"] = pd.cut(
        df["n_instances"],
        bins=SIZE_BIN_EDGES,
        labels=SIZE_BIN_LABELS,
        right=True,
    )
    by_size = (
        df.groupby(alg_init + ["size_bin"], observed=True)
        .agg(
            mean_relative_wcss=("relative_wcss", "mean"),
            mean_relative_time=("relative_time", "mean"),
        )
        .reset_index()
    )
    # Keep ordered bin labels
    by_size["size_bin"] = pd.Categorical(
        by_size["size_bin"], categories=SIZE_BIN_LABELS, ordered=True
    )
    by_size = by_size.sort_values(["size_bin", "algorithm", "n_init"])
    by_size.to_csv(RESULTS_DIR / "by_size_bin.csv", index=False)
    print(f"  Saved by_size_bin.csv  ({len(by_size)} rows)")

    # 5. Plots
    print("Generating plots…")
    color_map = build_color_map(df)
    plot_overall(overall, color_map=color_map)
    plot_by_k_multiplier(by_k_mult, color_map=color_map)
    plot_by_size_bin(by_size, color_map=color_map)

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
