"""
join_reports.py
───────────────
Left-join parsed_reports_with_agency (left) onto raw_reports_table (right)
on:  left["Unnamed: 0"]  ==  right["raw_text"]

All rows from the parsed file are kept; matching columns from raw_reports_table
(source_file, chunk_count, pages, parse_errors, assessment) are appended.
The redundant raw_text column from the right side is dropped after the join.

Usage
-----
    python join_reports.py
    python join_reports.py --left parsed_reports_with_agency.xlsx \
                           --right raw_reports_table.csv \
                           --out joined_reports.xlsx
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def read_any(path: Path) -> pd.DataFrame:
    """Read CSV or Excel, sniffing format from magic bytes."""
    with open(path, "rb") as fh:
        magic = fh.read(4)
    is_excel = magic[:4] in (b"\xd0\xcf\x11\xe0", b"PK\x03\x04")
    if is_excel or path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path, dtype=str)
    for sep in (",", ";", "\t", "|"):
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig")
            if len(df.columns) > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser(description="Left-join parsed reports with raw reports table")
    ap.add_argument("--left",  default="parsed_reports_with_agency.xlsx")
    ap.add_argument("--right", default="raw_reports_table.csv")
    ap.add_argument("--out",   default="joined_reports.xlsx")
    args = ap.parse_args()

    left_path  = Path(args.left)
    right_path = Path(args.right)
    out_path   = Path(args.out)

    for p in (left_path, right_path):
        if not p.exists():
            sys.exit(f"ERROR: file not found: {p}")

    print(f"Reading left  : {left_path}")
    df_left = read_any(left_path)
    print(f"  {len(df_left):,} rows × {len(df_left.columns)} cols")
    print(f"  Columns: {df_left.columns.tolist()}\n")

    print(f"Reading right : {right_path}")
    df_right = read_any(right_path)
    print(f"  {len(df_right):,} rows × {len(df_right.columns)} cols")
    print(f"  Columns: {df_right.columns.tolist()}\n")

    # Validate join keys
    for col, df, label in [
        ("Unnamed: 0", df_left,  "left (parsed_reports_with_agency)"),
        ("raw_text",   df_right, "right (raw_reports_table)"),
    ]:
        if col not in df.columns:
            sys.exit(
                f"ERROR: column '{col}' not found in {label}.\n"
                f"  Available: {df.columns.tolist()}"
            )

    # Strip whitespace from join keys to avoid invisible mismatches
    df_left["Unnamed: 0"]  = df_left["Unnamed: 0"].str.strip()
    df_right["raw_text"]   = df_right["raw_text"].str.strip()

    # Left join
    merged = df_left.merge(
        df_right,
        left_on="Unnamed: 0",
        right_on="raw_text",
        how="left",
        suffixes=("", "_right"),
    )

    # Drop the redundant raw_text column from the right side
    if "raw_text" in merged.columns:
        merged = merged.drop(columns=["raw_text"])

    # Drop any duplicate unnamed index columns brought in from the right CSV
    unnamed_right = [c for c in merged.columns if c.startswith("Unnamed:") and c != "Unnamed: 0"]
    if unnamed_right:
        merged = merged.drop(columns=unnamed_right)

    # Rename the key column to something meaningful
    merged = merged.rename(columns={"Unnamed: 0": "raw_text"})

    # Stats
    matched   = merged["source_file"].notna().sum() if "source_file" in merged.columns else "?"
    unmatched = len(merged) - (matched if isinstance(matched, int) else 0)
    print(f"Join complete: {len(merged):,} rows total  |  matched: {matched}  |  unmatched: {unmatched}")

    if unmatched and isinstance(unmatched, int) and unmatched > 0:
        miss = merged[merged["source_file"].isna()]["raw_text"].str[:80]
        print(f"\nFirst 5 unmatched raw_text previews:")
        for s in miss.head(5):
            print(f"  {s!r}")

    # Save
    if out_path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
        merged.to_excel(out_path, index=False)
    else:
        merged.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path.resolve()}")


if __name__ == "__main__":
    main()
