#!/usr/bin/env python3
"""Assemble cross-scenario CV comparison and generate charts.

Usage:
    python -m cgm_wgp.compare_cv --crop Beans [--crop Broccoli] [--plot]
    python -m cgm_wgp.compare_cv --crop Beans --label fullsuite --plot

Reads each scenario's combined/cv_summary.csv, assembles into a single
all_models_cv_results.csv, and optionally generates bump + grouped-bar charts.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "pipeline" / "output"
INDEX_PATH = OUTPUT_DIR / "index.json"

CV_ORDER = [
    "balanced_training_subset",
    "new_varieties",
    "individual",
    "cv00_double_novelty",
]
CV_LABELS = {
    "balanced_training_subset": "CV2\n(Gap-Filling)",
    "new_varieties": "CV1\n(New Varieties)",
    "individual": "CV0\n(New Env)",
    "cv00_double_novelty": "CV00\n(Double Novelty)",
}
# METHOD_ORDER and METHOD_COLORS come from the central registry — add or
# remove methods by editing src/cgm_wgp/methods.py.
from cgm_wgp.methods import display_order as _display_order, color_map as _color_map
METHOD_ORDER = _display_order()
METHOD_COLORS = _color_map()
SCENARIO_COLORS = {
    "balanced_training_subset": "#5B8DB8",
    "new_varieties":            "#E8985E",
    "individual":               "#4CAF50",
    "cv00_double_novelty":      "#D47B7B",
}


def _load_index():
    if not INDEX_PATH.exists():
        return []
    return json.loads(INDEX_PATH.read_text())


def _find_runs(crop, label_pattern=None):
    """Find the most recent run per cv_scheme for a crop."""
    entries = _load_index()
    entries = [e for e in entries if e.get("crop") == crop and e.get("cv_scheme")]
    if label_pattern:
        entries = [e for e in entries if label_pattern in (e.get("label") or "")]
    by_scheme = {}
    for e in sorted(entries, key=lambda x: x.get("created", "")):
        by_scheme[e["cv_scheme"]] = e
    return by_scheme


def assemble(crops, label_pattern=None):
    """Assemble cv_summary.csv from all matching runs into one DataFrame."""
    frames = []
    for crop in crops:
        runs = _find_runs(crop, label_pattern)
        for scheme in CV_ORDER:
            entry = runs.get(scheme)
            if not entry:
                continue
            run_path = OUTPUT_DIR / entry["path"]
            summary_path = run_path / "combined" / "cv_summary.csv"
            if not summary_path.exists():
                print(f"  [warn] No cv_summary.csv at {summary_path}", file=sys.stderr)
                continue
            df = pd.read_csv(summary_path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def print_table(df):
    """Print a formatted comparison table."""
    overall = df[df["planting"] == "Overall"].copy()
    if overall.empty:
        print("No overall results found.")
        return
    print(f"\n{'='*90}")
    print("CROSS-SCENARIO CV COMPARISON")
    print(f"{'='*90}")
    print(f"{'Crop':<12} {'Scenario':<28} {'Method':<14} {'N':>5} {'RMSE':>8} {'PA(r)':>8}")
    print(f"{'-'*90}")
    for crop in overall["crop"].unique():
        for scheme in CV_ORDER:
            sub = overall[(overall["crop"] == crop) & (overall["cv_scheme"] == scheme)]
            sub = sub.sort_values("method", key=lambda s: s.map(
                {m: i for i, m in enumerate(METHOD_ORDER)}))
            for _, r in sub.iterrows():
                label = CV_LABELS.get(scheme, scheme).replace("\n", " ")
                print(f"{r['crop']:<12} {label:<28} {r['method']:<14} "
                      f"{int(r['n']):>5} {r['rmse']:>8.2f} {r['pa']:>8.4f}")


def plot_grouped_bar(df, crop, metric, out_dir):
    """Grouped-bar chart: methods on x-axis, bars colored by scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = df[(df["crop"] == crop) & (df["planting"] == "Overall")].copy()
    if overall.empty:
        return

    methods = [m for m in METHOD_ORDER if m in overall["method"].values]
    scenarios = [s for s in CV_ORDER if s in overall["cv_scheme"].values]
    n_m = len(methods)
    n_s = len(scenarios)
    if n_m == 0 or n_s == 0:
        return

    fig, ax = plt.subplots(figsize=(max(7, n_m * 2.5), 5.5))
    x = np.arange(n_m)
    bar_w = 0.72 / max(n_s, 1)
    group_w = bar_w * n_s
    offset0 = -group_w / 2 + bar_w / 2

    ascending = metric == "rmse"
    ylabel = "RMSE (days)" if metric == "rmse" else "Prediction Accuracy (r)"
    fmt = (lambda v: f"{v:.1f}") if metric == "rmse" else (lambda v: f"{v:.2f}")

    for j, scen in enumerate(scenarios):
        vals = []
        for m in methods:
            row = overall[(overall["method"] == m) & (overall["cv_scheme"] == scen)]
            vals.append(float(row[metric].iloc[0]) if not row.empty else 0)
        bars = ax.bar(x + offset0 + j * bar_w, vals, width=bar_w * 0.9,
                      color=SCENARIO_COLORS.get(scen, "#999"),
                      alpha=0.85, label=CV_LABELS[scen].replace("\n", " "),
                      edgecolor="white", linewidth=0.3, zorder=2)
        for bar, v in zip(bars, vals):
            if v != 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v,
                        fmt(v), ha="center", va="bottom", fontsize=7,
                        color="#444", fontweight="medium")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(f"{crop} — {ylabel} Across CV Scenarios",
                 fontsize=13, fontweight="bold", pad=14)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors="#666", labelsize=9)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best",
              frameon=True, facecolor="white", edgecolor="#DDD")
    if metric == "pa":
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    suffix = "rmse" if metric == "rmse" else "pa"
    path = out_dir / f"{crop.lower()}_cv_scenarios_{suffix}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_bump(df, crop, metric, out_dir):
    """Ranking bump chart: scenarios on x-axis, rank on y-axis, one line per method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = df[(df["crop"] == crop) & (df["planting"] == "Overall")].copy()
    if overall.empty:
        return

    ascending = metric == "rmse"
    methods = [m for m in METHOD_ORDER if m in overall["method"].values]
    scenarios = [s for s in CV_ORDER if s in overall["cv_scheme"].values]
    n_m = len(methods)
    if n_m == 0 or not scenarios:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 2.5), 6))
    x_pos = list(range(len(scenarios)))

    fmt = (lambda v: f"{v:.1f}") if metric == "rmse" else (lambda v: f"{v:.2f}")
    ylabel = "Rank (1 = Best RMSE)" if metric == "rmse" else "Rank (1 = Best PA)"
    title_m = "RMSE" if metric == "rmse" else "Prediction Accuracy (PA)"

    for method in methods:
        ranks, vals, valid_x = [], [], []
        for xi, scen in enumerate(scenarios):
            sub = overall[overall["cv_scheme"] == scen].dropna(subset=[metric])
            if sub.empty or method not in sub["method"].values:
                continue
            sorted_sub = sub.sort_values(metric, ascending=ascending)
            rank = list(sorted_sub["method"].values).index(method) + 1
            val = float(sub[sub["method"] == method][metric].iloc[0])
            ranks.append(rank)
            vals.append(val)
            valid_x.append(xi)
        if not valid_x:
            continue
        color = METHOD_COLORS.get(method, "#999")
        ax.plot(valid_x, ranks, "-o", color=color, linewidth=2.5, markersize=10,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.5,
                zorder=5, label=method)
        for xi, rank, val in zip(valid_x, ranks, vals):
            ax.annotate(fmt(val), (xi, rank), textcoords="offset points",
                        xytext=(12, 3), fontsize=8, fontweight="bold", color=color,
                        ha="left", va="bottom")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([CV_LABELS.get(s, s) for s in scenarios], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_ylim(n_m + 0.5, 0.5)
    ax.set_yticks(range(1, n_m + 1))
    ax.set_xlim(-0.5, len(scenarios) - 0.5)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_title(f"{crop} — {title_m} Ranking Across CV Scenarios",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10,
              frameon=True, framealpha=0.9, edgecolor="gray")
    fig.tight_layout()
    suffix = "rmse" if metric == "rmse" else "pa"
    path = out_dir / f"{crop.lower()}_cv_bump_{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def _make_comparison_dir(label):
    """Create a timestamped + labeled comparison directory under pipeline/output/comparisons/.

    Avoids overwriting when the user runs compare_cv multiple times. Also
    updates a `latest` symlink for convenience.
    """
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = (label or "compare").replace("/", "_").replace(" ", "-")
    root = OUTPUT_DIR / "comparisons" / f"{ts}_{slug}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)

    latest = root.parent / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(root.name)
    except OSError:
        pass
    return root


def main():
    p = argparse.ArgumentParser(description="Assemble cross-scenario CV comparison.")
    p.add_argument("--crop", action="append", required=True,
                   help="Crop name (repeat for multiple crops).")
    p.add_argument("--label", default=None,
                   help="Filter runs whose label contains this substring. Also used "
                        "as the comparison directory slug when --plot is passed.")
    p.add_argument("--output", default=None,
                   help="Override the output directory (default: "
                        "pipeline/output/comparisons/{timestamp}_{label}/).")
    p.add_argument("--plot", action="store_true",
                   help="Generate bump and grouped-bar charts.")
    args = p.parse_args()

    df = assemble(args.crop, args.label)
    if df.empty:
        print("ERROR: No cv_summary.csv files found. Run CV scenarios first.", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out_root = Path(args.output)
        out_root.mkdir(parents=True, exist_ok=True)
        fig_dir = out_root / "figures"
        fig_dir.mkdir(exist_ok=True)
    else:
        out_root = _make_comparison_dir(args.label)
        fig_dir = out_root / "figures"

    out_csv = out_root / "all_models_cv_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Wrote: {out_csv} ({len(df)} rows)")

    print_table(df)

    if args.plot:
        for crop in args.crop:
            for metric in ("rmse", "pa"):
                plot_grouped_bar(df, crop, metric, fig_dir)
                plot_bump(df, crop, metric, fig_dir)
        print(f"\n  All figures: {fig_dir}")


if __name__ == "__main__":
    main()
