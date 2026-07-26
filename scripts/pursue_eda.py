"""EDA for PURSUE_1_2_3_4_merged.csv"""
import json
from pathlib import Path

import pandas as pd

path = Path("SUBDATASETS_V2/PURSUE_1_2_3_4_merged.csv")
df = pd.read_csv(path, low_memory=False)

print("=== SHAPE ===")
print(f"rows={len(df)}, cols={len(df.columns)}")

if "_source_dataset" in df.columns:
    print("\n=== SOURCE DATASETS ===")
    print(df["_source_dataset"].value_counts(dropna=False).to_string())

num_cols = ["date_time.year", "sightingDetails.trustScore", "witness_count_num", "witness.count"]
for c in num_cols:
    if c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        print(f"\n=== {c} ===")
        print(s.describe().to_string())
        print(f"non-null: {s.notna().sum()} ({100 * s.notna().mean():.1f}%)")

if "trust_band" in df.columns:
    print("\n=== trust_band ===")
    print(df["trust_band"].value_counts(dropna=False).head(20).to_string())

for c in ["source.agency", "agency", "src_agency"]:
    if c in df.columns:
        print(f"\n=== {c} ===")
        print(df[c].value_counts(dropna=False).head(15).to_string())

if "location.country" in df.columns:
    print("\n=== location.country (top 20) ===")
    print(df["location.country"].value_counts(dropna=False).head(20).to_string())

if "craft.primary_shape" in df.columns:
    print("\n=== craft.primary_shape ===")
    print(df["craft.primary_shape"].value_counts(dropna=False).head(20).to_string())

if "date_time.year" in df.columns:
    y = pd.to_numeric(df["date_time.year"], errors="coerce")
    print("\n=== year bins ===")
    bins = [0, 1945, 1975, 2000, 2026]
    labels = ["pre-1945", "1945-1975", "1976-2000", "2001-2026"]
    print(pd.cut(y, bins=bins, labels=labels).value_counts().sort_index().to_string())
    print(f"year range: {y.min():.0f} - {y.max():.0f}")

scu_cols = [c for c in df.columns if "scu" in c.lower()]
print("\n=== SCU COLUMNS ===")
for c in scu_cols:
    if df[c].dtype == bool:
        print(f"{c}: {df[c].mean() * 100:.1f}% True")
    else:
        vc = df[c].value_counts(dropna=False).head(5)
        print(f"{c}: {dict(vc)}")

miss = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print("\n=== MOST COMPLETE COLUMNS ===")
print(miss.tail(15).to_string())
print("\n=== MOST SPARSE COLUMNS ===")
print(miss.head(15).to_string())

if "document_id" in df.columns:
    print(f"\nunique document_id: {df['document_id'].nunique()} / {len(df)}")
if "source.documentId" in df.columns:
    print(f"unique source.documentId: {df['source.documentId'].nunique()} / {len(df)}")

if "tier" in df.columns:
    print("\n=== tier ===")
    print(df["tier"].value_counts(dropna=False).to_string())

if "date_time.day_night" in df.columns:
    print("\n=== day_night ===")
    print(df["date_time.day_night"].value_counts(dropna=False).to_string())

for c in ["anomaly.structure", "anomaly.flight", "anomaly.occupant", "anomaly.signal"]:
    if c in df.columns:
        rate = (df[c].astype(str).str.upper() == "Y").mean()
        print(f"{c} Y-rate: {rate * 100:.1f}%")

eng_cols = [c for c in df.columns if c.startswith("engagement_flags.")]
print("\n=== ENGAGEMENT FLAGS ===")
for c in sorted(eng_cols):
    rate = (df[c].astype(str).str.upper() == "Y").mean()
    print(f"{c}: {rate * 100:.1f}%")

perf_cols = [c for c in df.columns if c.startswith("performance.")]
print("\n=== PERFORMANCE ===")
for c in sorted(perf_cols):
    rate = (df[c].astype(str).str.upper() == "Y").mean()
    print(f"{c}: {rate * 100:.1f}%")

witness_cols = [c for c in df.columns if c.startswith("witness_is_")]
print("\n=== WITNESS ROLES ===")
for c in sorted(witness_cols):
    if df[c].dtype == bool:
        print(f"{c}: {df[c].mean() * 100:.1f}%")

if "_source_dataset" in df.columns and "date_time.year" in df.columns:
    print("\n=== YEAR MEDIAN BY SOURCE ===")
    cross = df.groupby("_source_dataset")["date_time.year"].apply(
        lambda x: pd.to_numeric(x, errors="coerce").median()
    )
    print(cross.to_string())

if "timeliness_status" in df.columns:
    print("\n=== timeliness_status ===")
    print(df["timeliness_status"].value_counts(dropna=False).to_string())

if "assessment.explanationCategory" in df.columns:
    print("\n=== explanationCategory ===")
    print(df["assessment.explanationCategory"].value_counts(dropna=False).head(15).to_string())

summary = {
    "rows": len(df),
    "cols": len(df.columns),
    "source_datasets": {
        str(k): int(v) for k, v in df["_source_dataset"].value_counts().items()
    }
    if "_source_dataset" in df.columns
    else {},
    "years": {
        str(int(k)): int(v)
        for k, v in pd.to_numeric(df["date_time.year"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .items()
    }
    if "date_time.year" in df.columns
    else {},
    "countries": {
        str(k): int(v) for k, v in df["location.country"].value_counts().head(15).items()
    }
    if "location.country" in df.columns
    else {},
    "shapes": {
        str(k): int(v) for k, v in df["craft.primary_shape"].value_counts().head(15).items()
    }
    if "craft.primary_shape" in df.columns
    else {},
    "agencies": {
        str(k): int(v)
        for k, v in (
            df["agency"].value_counts().head(10).items()
            if "agency" in df.columns
            else df["source.agency"].value_counts().head(10).items()
        )
    },
    "trust_bands": {
        str(k): int(v) for k, v in df["trust_band"].value_counts().items()
    }
    if "trust_band" in df.columns
    else {},
    "day_night": {
        str(k): int(v) for k, v in df["date_time.day_night"].value_counts().items()
    }
    if "date_time.day_night" in df.columns
    else {},
    "tiers": {str(k): int(v) for k, v in df["tier"].value_counts().items()}
    if "tier" in df.columns
    else {},
    "engagement_flags": {
        c.split(".")[-1]: round((df[c].astype(str).str.upper() == "Y").mean() * 100, 1)
        for c in eng_cols
    },
    "performance": {
        c.split(".")[-1]: round((df[c].astype(str).str.upper() == "Y").mean() * 100, 1)
        for c in perf_cols
    },
    "anomaly": {
        c.split(".")[-1]: round((df[c].astype(str).str.upper() == "Y").mean() * 100, 1)
        for c in ["anomaly.structure", "anomaly.flight", "anomaly.occupant", "anomaly.signal"]
        if c in df.columns
    },
    "witness_roles": {
        c.replace("witness_is_", ""): round(df[c].mean() * 100, 1)
        for c in witness_cols
        if df[c].dtype == bool
    },
    "scu_rates": {
        c: round(df[c].mean() * 100, 1) for c in scu_cols if df[c].dtype == bool
    },
    "trust_score_stats": {
        k: float(v)
        for k, v in pd.to_numeric(df["sightingDetails.trustScore"], errors="coerce")
        .describe()
        .items()
    }
    if "sightingDetails.trustScore" in df.columns
    else {},
    "year_bins": {
        str(k): int(v)
        for k, v in pd.cut(
            pd.to_numeric(df["date_time.year"], errors="coerce"),
            bins=[0, 1945, 1975, 2000, 2026],
            labels=["pre-1945", "1945-1975", "1976-2000", "2001-2026"],
        )
        .value_counts()
        .sort_index()
        .items()
    }
    if "date_time.year" in df.columns
    else {},
    "missing_pct_avg": round(miss.mean(), 1),
    "cols_over_90pct_missing": int((miss > 90).sum()),
    "timeliness": {
        str(k): int(v) for k, v in df["timeliness_status"].value_counts().items()
    }
    if "timeliness_status" in df.columns
    else {},
}

out = Path("SUBDATASETS_V2/PURSUE_1_2_3_4_eda_summary.json")
out.write_text(json.dumps(summary, indent=2))
print(f"\nJSON saved to {out}")
