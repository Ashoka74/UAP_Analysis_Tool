"""
dedupe_all_reports.py
──────────────────────
Publication-quality EDA figures over the merged PURSUE_1_2_3_4 corpus
(SUBDATASETS_V2/PURSUE_1_2_3_4_merged.csv), extending scripts/pursue_eda.py's
printed summary into actual charts.

Follows scientific-visualization conventions: Okabe-Ito colorblind-safe
palette, sans-serif fonts, vector + raster export (PDF + PNG, 300 DPI), no
chart junk, axis labels with units. Each figure is skipped (not stubbed
blank) if its source column isn't present in the corpus.

Usage:
    uv run python dedupe_all_reports.py
    uv run python dedupe_all_reports.py --src SUBDATASETS_V2/PURSUE_1_2_3_4_merged.csv --out figures
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']

DEFAULT_SRC = "SUBDATASETS_V2/PURSUE_1_2_3_4_merged.csv"
DEFAULT_OUT = "figures"


def apply_style():
    mpl.rcParams.update({
        'figure.facecolor': 'white',
        'font.size': 9,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 0.6,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': mpl.cycler(color=OKABE_ITO),
        'axes.axisbelow': True,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5,
        'legend.frameon': False,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })


def save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir / name}.{{pdf,png}}")


def _bar_with_labels(ax, x, y, color):
    bars = ax.bar(x, y, color=color, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, y):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom", fontsize=7.5)
    return bars


def fig_source_composition(df, out_dir):
    if "_source_dataset" not in df.columns:
        return
    counts = df["_source_dataset"].value_counts()
    fig, ax = plt.subplots(figsize=(3.5, 3))
    _bar_with_labels(ax, counts.index, counts.values, OKABE_ITO[:len(counts)])
    ax.set_ylabel("Reports (n)")
    ax.set_title("Corpus composition by release batch")
    save_fig(fig, out_dir, "01_source_composition")


def fig_year_distribution(df, out_dir):
    if "date_time.year" not in df.columns:
        return
    y = pd.to_numeric(df["date_time.year"], errors="coerce")
    y = y[(y > 1400) & (y <= 2026)]
    if y.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(y, bins=60, color=OKABE_ITO[4], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Sighting year")
    ax.set_ylabel("Reports (n)")
    ax.set_title(f"Temporal distribution of sightings (n={len(y):,})")
    save_fig(fig, out_dir, "02_year_distribution")


def fig_year_eras(df, out_dir):
    if "date_time.year" not in df.columns:
        return
    y = pd.to_numeric(df["date_time.year"], errors="coerce")
    y = y[(y > 1400) & (y <= 2026)]
    if y.empty:
        return
    bins = [1400, 1945, 1975, 2000, 2026]
    labels = ["pre-1945", "1945-1975", "1976-2000", "2001-2026"]
    counts = pd.cut(y, bins=bins, labels=labels).value_counts().reindex(labels)
    fig, ax = plt.subplots(figsize=(4, 3))
    _bar_with_labels(ax, counts.index.astype(str), counts.values, OKABE_ITO[1])
    ax.set_ylabel("Reports (n)")
    ax.set_title("Sightings by era")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save_fig(fig, out_dir, "03_year_eras")


def fig_top_countries(df, out_dir, top_n=12):
    if "location.country" not in df.columns:
        return
    counts = df["location.country"].value_counts().head(top_n)
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.barh(counts.index[::-1], counts.values[::-1], color=OKABE_ITO[2],
           edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Reports (n)")
    ax.set_title(f"Top {top_n} sighting countries/regions")
    save_fig(fig, out_dir, "04_top_countries")


def fig_craft_shapes(df, out_dir, top_n=12):
    if "craft.primary_shape" not in df.columns:
        return
    counts = df["craft.primary_shape"].value_counts().head(top_n)
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.barh(counts.index[::-1], counts.values[::-1], color=OKABE_ITO[5],
           edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Reports (n)")
    ax.set_title(f"Top {top_n} reported craft shapes")
    save_fig(fig, out_dir, "05_craft_shapes")


def fig_trust_score(df, out_dir):
    if "sightingDetails.trustScore" not in df.columns:
        return
    s = pd.to_numeric(df["sightingDetails.trustScore"], errors="coerce").dropna()
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(s, bins=30, color=OKABE_ITO[0], edgecolor="white", linewidth=0.3)
    ax.axvline(s.mean(), color="black", linestyle="--", linewidth=1,
              label=f"mean = {s.mean():.1f}")
    ax.set_xlabel("Trust score (0-100)")
    ax.set_ylabel("Reports (n)")
    ax.set_title("Distribution of report trust scores")
    ax.legend()
    save_fig(fig, out_dir, "06_trust_score_distribution")


def fig_trust_bands(df, out_dir):
    if "trust_band" not in df.columns:
        return
    preferred = ["low", "mid", "high", "very_high"]
    present = df["trust_band"].dropna().unique().tolist()
    order = [b for b in preferred if b in present] or present
    counts = df["trust_band"].value_counts().reindex(order).dropna()
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(3.5, 3))
    _bar_with_labels(ax, counts.index, counts.values, OKABE_ITO[:len(counts)])
    ax.set_ylabel("Reports (n)")
    ax.set_title("Reports by trust band")
    save_fig(fig, out_dir, "07_trust_bands")


def fig_day_night(df, out_dir):
    if "date_time.day_night" not in df.columns:
        return
    label_map = {"D": "Day", "N": "Night", "U": "Unknown"}
    counts = df["date_time.day_night"].value_counts()
    if counts.empty:
        return
    labels = [label_map.get(k, k) for k in counts.index]
    fig, ax = plt.subplots(figsize=(3.5, 3))
    _bar_with_labels(ax, labels, counts.values, OKABE_ITO[:len(counts)])
    ax.set_ylabel("Reports (n)")
    ax.set_title("Sightings by time of day")
    save_fig(fig, out_dir, "08_day_night")


def fig_agencies(df, out_dir, top_n=10):
    col = next((c for c in ("source.agency", "agency") if c in df.columns), None)
    if col is None:
        return
    counts = df[col].value_counts().head(top_n)
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(4, 3))
    _bar_with_labels(ax, counts.index, counts.values, OKABE_ITO[3])
    ax.set_ylabel("Reports (n)")
    ax.set_title("Reports by source agency")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save_fig(fig, out_dir, "09_agencies")


def fig_engagement_performance(df, out_dir):
    eng_cols = [c for c in df.columns if c.startswith("engagement_flags.")]
    perf_cols = [c for c in df.columns if c.startswith("performance.")]
    cols = eng_cols + perf_cols
    if not cols:
        return
    rates = {}
    for c in cols:
        rate = (df[c].astype(str).str.upper() == "Y").mean() * 100
        label = c.split(".")[-1].replace("_", " ")
        group = "Engagement" if c.startswith("engagement_flags.") else "Performance"
        rates[f"{label} ({group})"] = rate
    s = pd.Series(rates).sort_values()
    if s.empty or s.max() == 0:
        return
    fig, ax = plt.subplots(figsize=(5, max(3, 0.28 * len(s))))
    colors = [OKABE_ITO[0] if "(Engagement)" in k else OKABE_ITO[4] for k in s.index]
    ax.barh(s.index, s.values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Reports exhibiting flag (%)")
    ax.set_title("Engagement & performance anomaly rates")
    save_fig(fig, out_dir, "10_engagement_performance_rates")


def fig_timeliness(df, out_dir):
    if "timeliness_status" not in df.columns:
        return
    counts = df["timeliness_status"].value_counts()
    if counts.empty:
        return
    labels = [str(k).replace("_", " ") for k in counts.index]
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.barh(labels[::-1], counts.values[::-1], color=OKABE_ITO[6],
           edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Reports (n)")
    ax.set_title("Investigation timeliness status")
    save_fig(fig, out_dir, "11_timeliness_status")


FIGURES = [
    fig_source_composition, fig_year_distribution, fig_year_eras,
    fig_top_countries, fig_craft_shapes, fig_trust_score, fig_trust_bands,
    fig_day_night, fig_agencies, fig_engagement_performance, fig_timeliness,
]


def main():
    ap = argparse.ArgumentParser(description="Publication-quality EDA figures for the PURSUE corpus")
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"source not found: {src} - run merge_and_dedup_pursue4.py first")
    out_dir = Path(args.out)

    print(f"Loading {src} ...")
    df = pd.read_csv(src, low_memory=False)
    print(f"  {len(df):,} rows x {len(df.columns)} columns")

    apply_style()

    print(f"\nGenerating figures -> {out_dir}/")
    for f in FIGURES:
        f(df, out_dir)

    n = len(list(out_dir.glob("*.pdf"))) if out_dir.exists() else 0
    print(f"\nDone. {n} figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
