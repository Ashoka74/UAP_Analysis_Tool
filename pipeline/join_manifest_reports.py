"""
join_manifest_reports.py
─────────────────────────
Joins a manifest CSV with a raw reports table on document file name.

Join key
────────
  Primary  (PDF URL stem):
    R.source_file          →  stem (drop .md), lowercase
    M."PDF | Image Link"   →  URL filename stem, lowercase
    e.g.  059uap00011.md   ↔  .../059uap00011.pdf   ✓

  Fallback (Title):
    Any R rows still unmatched after the primary join are retried
    against M.Title (strip newlines, lowercase).  This catches the
    26 NARA / series files where the Title already matches the stem.

Result: one row per extracted report, enriched with all manifest columns.
Output: joined_manifest_reports.csv  (saved in --dir, default: script folder)

Usage:
    python join_manifest_reports.py --manifest uap-data_v3.csv --reports raw_reports_table.csv
    python join_manifest_reports.py --dir D:/divided/data
"""

import re
import argparse
from pathlib import Path
import pandas as pd


def normalise(s: pd.Series) -> pd.Series:
    """Lowercase, strip whitespace and newlines."""
    return s.astype(str).str.strip().str.replace(r"[\r\n]+", " ", regex=True).str.strip().str.lower()


_CODE_RE = re.compile(r"[A-Za-z]{2,4}-UAP-[A-Za-z]?\d+")


def doc_code(s) -> str:
    """Extract a document doc-code like CIA-UAP-010 / DOW-UAP-D084 (uppercased),
    or '' if none. Lets releases ingested with a document_id but no source_file
    (e.g. PURSUE3) match the manifest even though their source_file stem is empty."""
    m = _CODE_RE.search(str(s))
    return m.group(0).upper() if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).parent))
    ap.add_argument("--manifest",  default="uap-csv-892fce75.csv")
    ap.add_argument("--reports",   default="raw_reports_table-094043e6.csv")
    ap.add_argument("--out",       default="joined_manifest_reports.csv")
    ap.add_argument("--how",       default="left",
                    choices=["left", "inner", "outer", "right"],
                    help="Join type (default: left — keep all report rows)")
    args = ap.parse_args()

    folder = Path(args.dir)

    # ── Load ─────────────────────────────────────────────────────────────────
    M = pd.read_csv(folder / args.manifest,  low_memory=False)
    R = pd.read_csv(folder / args.reports,   low_memory=False)
    print(f"Manifest  ({args.manifest}):  {M.shape[0]} rows × {M.shape[1]} cols")
    print(f"Reports   ({args.reports}): {R.shape[0]} rows × {R.shape[1]} cols")

    # ── Build join keys ───────────────────────────────────────────────────────
    # R: strip .md extension, lowercase
    R["_r_key"] = normalise(R["source_file"].apply(lambda x: Path(str(x)).stem))

    # Doc-code fallback key: source_file stem first, else document_id (PURSUE3
    # carries no source_file). Used by Pass 3 below.
    def _r_code(row):
        sf = row.get("source_file")
        c = doc_code(Path(str(sf)).stem) if pd.notna(sf) else ""
        return c or doc_code(row.get("document_id", ""))
    R["_r_code"] = R.apply(_r_code, axis=1)

    # M primary key: PDF URL filename stem (directly mirrors the .md source filename)
    pdf_col = "PDF | Image Link"
    if pdf_col in M.columns:
        M["_m_key_pdf"] = normalise(
            M[pdf_col].apply(
                lambda x: Path(str(x)).stem if pd.notna(x) and str(x).strip() not in ("", "nan") else ""
            )
        )
    else:
        print(f"⚠  Column '{pdf_col}' not found — falling back to Title only")
        M["_m_key_pdf"] = ""

    # M fallback key: Title prefix before first comma
    # e.g. "059uap00011, UFOs over Georgia…"  →  "059uap00011"
    # e.g. "DOW-UAP-PR050, \"4 UAP Formation…\"" →  "dow-uap-pr050"
    M["_m_key_title"] = normalise(M["Title"].str.split(",").str[0])

    # M doc-code key: from Title or the PDF URL stem (e.g. CIA-UAP-010).
    M["_m_code"] = M.apply(
        lambda r: doc_code(r.get("Title")) or doc_code(Path(str(r.get(pdf_col, ""))).stem),
        axis=1,
    )

    print("\nSample R keys (source_file stem):", R["_r_key"].head(5).tolist())
    print("Sample M PDF keys:               ", M["_m_key_pdf"].head(5).tolist())
    print("Sample M Title-prefix keys:      ", M["_m_key_title"].head(5).tolist())

    # ── Drop unnamed / empty trailing columns; keep helper keys in M for merging ──
    M_work = M.loc[:, ~M.columns.str.match(r"^Unnamed")].copy()
    # payload columns = everything except helper keys
    M_payload_cols = [c for c in M_work.columns if c not in ("_m_key_pdf", "_m_key_title", "_m_code")]

    # ── Pass 1: join on PDF URL stem ──────────────────────────────────────────
    # Only match non-empty PDF keys
    valid_pdf = M_work[M_work["_m_key_pdf"] != ""].drop_duplicates("_m_key_pdf")
    hit_pdf   = R["_r_key"].isin(set(valid_pdf["_m_key_pdf"]))
    print(f"\nPass 1 (PDF stem):   {hit_pdf.sum()} / {len(R)} R rows matched")

    pass1 = R[hit_pdf].merge(
        valid_pdf[M_payload_cols + ["_m_key_pdf"]],
        left_on="_r_key", right_on="_m_key_pdf", how="left"
    ).drop(columns=["_m_key_pdf"], errors="ignore")

    # ── Pass 2: fallback — unmatched rows vs Title prefix key ─────────────────
    unmatched = R[~hit_pdf].copy()
    valid_title = M_work[M_work["_m_key_title"] != ""].drop_duplicates("_m_key_title")
    hit_title   = unmatched["_r_key"].isin(set(valid_title["_m_key_title"]))
    print(f"Pass 2 (Title pfx):  {hit_title.sum()} / {len(unmatched)} remaining R rows matched")

    pass2_matched = unmatched[hit_title].merge(
        valid_title[M_payload_cols + ["_m_key_title"]],
        left_on="_r_key", right_on="_m_key_title", how="left"
    ).drop(columns=["_m_key_title"], errors="ignore")

    # ── Pass 3: doc-code fallback — catches document_id-keyed releases (PURSUE3) ──
    after_title = unmatched[~hit_title].copy()
    valid_code = M_work[M_work["_m_code"] != ""].drop_duplicates("_m_code")
    hit_code = (after_title["_r_code"] != "") & after_title["_r_code"].isin(set(valid_code["_m_code"]))
    print(f"Pass 3 (doc-code):   {hit_code.sum()} / {len(after_title)} remaining R rows matched")

    pass3_matched = after_title[hit_code].merge(
        valid_code[M_payload_cols + ["_m_code"]],
        left_on="_r_code", right_on="_m_code", how="left"
    ).drop(columns=["_m_code"], errors="ignore")

    # Unmatched rows — keep all R columns, NaN for manifest columns
    pass3_unmatched = after_title[~hit_code].copy()
    for col in M_payload_cols:
        if col not in pass3_unmatched.columns:
            pass3_unmatched[col] = pd.NA
    print(f"Still unmatched:     {len(pass3_unmatched)} R rows (no manifest entry found)")

    # ── Combine ───────────────────────────────────────────────────────────────
    result = pd.concat([pass1, pass2_matched, pass3_matched, pass3_unmatched],
                       ignore_index=True, sort=False)
    result.drop(columns=["_r_key", "_r_code", "_m_key_pdf", "_m_key_title", "_m_code"],
                inplace=True, errors="ignore")
    result.reset_index(drop=True, inplace=True)

    total_matched = len(pass1) + len(pass2_matched) + len(pass3_matched)
    print(f"\nTotal matched: {total_matched} / {len(R)} R rows  "
          f"({100 * total_matched / max(len(R), 1):.1f}%)")
    print(f"Result:  {result.shape[0]} rows × {result.shape[1]} cols")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = folder / args.out
    result.to_csv(out_path, index=False)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"✓  Saved → {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
