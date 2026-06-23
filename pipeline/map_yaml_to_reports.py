"""
map_yaml_to_reports.py
──────────────────────
Joins YAML records in records/ with rows in raw_reports_table.csv.

Normalization rule (both sides):
  source_file  →  strip .md  →  lowercase  →  spaces → hyphens  →  remove apostrophes
  record_id    →  already normalized (but may differ by apostrophe / trailing underscore)

Three output groups:
  A  matched        — YAML record + CSV report row joined
  B  yaml-only      — YAML record with no CSV counterpart (e.g. dow-uap-*)
  C  csv-only       — CSV row with no YAML record       (e.g. 059uap* UPDB files)

Usage
-----
    python map_yaml_to_reports.py
    python map_yaml_to_reports.py --records records --csv raw_reports_table.csv --out enriched_reports.xlsx
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_key(s: str) -> str:
    """
    Canonical join key:
      1. Strip extension (.md, .yaml, …)
      2. Lowercase
      3. Collapse runs of whitespace → single hyphen
      4. Remove apostrophes
      5. Collapse multiple consecutive hyphens or underscores into one hyphen
    """
    s = Path(s).stem          # drop extension
    s = s.lower()
    s = re.sub(r"[\s]+", "-", s)          # whitespace → hyphen
    s = s.replace("'", "")               # apostrophes gone
    s = re.sub(r"[_]+", "_", s)          # keep underscores single
    s = re.sub(r"[-]+", "-", s)          # keep hyphens single
    return s.strip("-_")


# ── YAML loading ──────────────────────────────────────────────────────────────

# Fields we want to pull out of each YAML record
_YAML_FIELDS = [
    ("record_id",        lambda y: y.get("record_id")),
    ("agency_yaml",      lambda y: y.get("csv", {}).get("agency")),
    ("report_subtype",   lambda y: y.get("csv", {}).get("report_subtype")),
    ("release_date",     lambda y: y.get("csv", {}).get("release_date")),
    ("incident_date",    lambda y: y.get("csv", {}).get("incident_date")),
    ("location_csv",     lambda y: y.get("csv", {}).get("location_csv")),
    ("uap_shape",        lambda y: y.get("observation", {}).get("morphology", {}).get("shape")),
    ("uap_maneuver",     lambda y: y.get("observation", {}).get("kinematics", {}).get("maneuver_type")),
    ("uap_speed_mph",    lambda y: y.get("observation", {}).get("kinematics", {}).get("speed_mph")),
    ("uap_altitude_ft",  lambda y: y.get("observation", {}).get("kinematics", {}).get("altitude_ft")),
    ("sensor_types",     lambda y: "; ".join(y.get("observation", {}).get("sensor_types") or []) or None),
    ("platform",         lambda y: y.get("observation", {}).get("platform")),
    ("threat_assessment",lambda y: y.get("observation", {}).get("threat_assessment")),
    ("region",           lambda y: y.get("csv", {}).get("region")),
    ("page_count",       lambda y: len(y.get("files", {}).get("pages") or [])),
    ("has_nim_report",   lambda y: bool(y.get("reports"))),
]


def load_yaml_records(records_dir: Path) -> pd.DataFrame:
    rows = []
    for yaml_path in sorted(records_dir.glob("*.yaml")):
        try:
            with open(yaml_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except Exception as exc:
            print(f"  WARN: could not parse {yaml_path.name}: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        row = {"yaml_file": yaml_path.name}
        for field_name, extractor in _YAML_FIELDS:
            try:
                row[field_name] = extractor(data)
            except Exception:
                row[field_name] = None
        rows.append(row)

    if not rows:
        sys.exit(f"ERROR: no .yaml files found in {records_dir}")

    df = pd.DataFrame(rows)
    df["join_key"] = df["record_id"].fillna("").apply(normalize_key)
    return df


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv_reports(csv_path: Path) -> pd.DataFrame:
    for sep in (",", ";", "\t", "|"):
        try:
            df = pd.read_csv(csv_path, sep=sep, dtype=str, encoding="utf-8-sig")
            if len(df.columns) > 1:
                break
        except Exception:
            pass
    else:
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")

    if "source_file" not in df.columns:
        sys.exit(
            f"ERROR: 'source_file' column not found in {csv_path}.\n"
            f"  Available: {df.columns.tolist()}"
        )

    df["join_key"] = df["source_file"].fillna("").apply(normalize_key)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Join YAML records with raw_reports_table.csv"
    )
    ap.add_argument("--records", default="records",        help="records/ directory")
    ap.add_argument("--csv",     default="raw_reports_table.csv")
    ap.add_argument("--out",     default="enriched_reports.xlsx")
    args = ap.parse_args()

    records_dir = Path(args.records)
    csv_path    = Path(args.csv)
    out_path    = Path(args.out)

    for p in (records_dir, csv_path):
        if not p.exists():
            sys.exit(f"ERROR: path not found: {p}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading YAML records from {records_dir} …")
    df_yaml = load_yaml_records(records_dir)
    print(f"  {len(df_yaml):,} YAML records  |  {df_yaml['join_key'].nunique():,} unique keys")

    print(f"\nLoading CSV from {csv_path} …")
    df_csv = load_csv_reports(csv_path)
    print(f"  {len(df_csv):,} CSV rows  |  {df_csv['join_key'].nunique():,} unique keys")

    # ── Join ──────────────────────────────────────────────────────────────────
    # Outer join so we can see all three groups
    merged = df_csv.merge(
        df_yaml,
        on="join_key",
        how="outer",
        suffixes=("_csv", "_yaml"),
        indicator=True,
    )

    # Group labels
    merged["group"] = merged["_merge"].map({
        "both":       "A_matched",
        "right_only": "B_yaml_only",
        "left_only":  "C_csv_only",
    })
    merged = merged.drop(columns=["_merge"])

    # ── Stats ─────────────────────────────────────────────────────────────────
    counts = merged["group"].value_counts().sort_index()
    total = len(merged)

    print("\n" + "─" * 60)
    print(f"{'Group':<20}  {'Count':>6}  {'%':>6}")
    print("─" * 60)
    for grp, cnt in counts.items():
        print(f"{grp:<20}  {cnt:>6,}  {cnt/total*100:>5.1f}%")
    print("─" * 60)
    print(f"{'TOTAL':<20}  {total:>6,}")

    # Breakdown of yaml-only by agency
    yaml_only = merged[merged["group"] == "B_yaml_only"]
    if len(yaml_only):
        print(f"\nYAML-only agency breakdown ({len(yaml_only):,} records):")
        for agency, cnt in yaml_only["agency_yaml"].value_counts(dropna=False).items():
            print(f"  {agency!s:<20} {cnt:>4,}")

    # Breakdown of csv-only by prefix
    csv_only = merged[merged["group"] == "C_csv_only"]
    if len(csv_only):
        print(f"\nCSV-only source_file examples (first 15):")
        for sf in csv_only["source_file"].dropna().head(15):
            print(f"  {sf}")

    # ── Duplicate key check ───────────────────────────────────────────────────
    dup_yaml = df_yaml[df_yaml.duplicated("join_key", keep=False)]
    if len(dup_yaml):
        print(f"\nWARN: {len(dup_yaml):,} YAML records share a join_key with another YAML record:")
        print(dup_yaml[["yaml_file", "record_id", "join_key"]].to_string(index=False))

    dup_csv = df_csv[df_csv.duplicated("join_key", keep=False)]
    if len(dup_csv):
        print(f"\nWARN: {len(dup_csv):,} CSV rows share a join_key with another CSV row:")
        print(dup_csv[["source_file", "join_key"]].head(10).to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    # Sort: matched first, then yaml-only, then csv-only; alpha within each group
    merged = merged.sort_values(["group", "join_key"]).reset_index(drop=True)

    if out_path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
        merged.to_excel(out_path, index=False)
    else:
        merged.to_csv(out_path, index=False)

    print(f"\nSaved → {out_path.resolve()}")
    print("Columns in output:")
    for col in merged.columns:
        print(f"  {col}")


if __name__ == "__main__":
    main()
