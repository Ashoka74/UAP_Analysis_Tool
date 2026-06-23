"""
add_source_agency.py
────────────────────
Reads an Excel file and adds two columns after 'may_release_source_file':
  • agency     — top-level agency folder  (DOD, FBI, NASA, DOS, NARA-CIA, MISC)
  • collection — sub-folder within agency (mission-reports, series-38, transcripts …)

Path/slug resolution order for both columns:
  1. Full or relative path  → parse path parts directly
  2. Slug prefix map        → e.g. "dow-uap-*" → DOD / mission-reports
  3. Substring scan         → look for known tokens anywhere in the value

Usage
-----
    python add_source_agency.py
    python add_source_agency.py --input my_file.xlsx --col file_path --out out.xlsx
"""

import argparse
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import pandas as pd

# ── Agency tokens (longer first to avoid sub-match, e.g. NARA-CIA before NASA) ─
AGENCY_TOKENS = ["NARA-CIA", "NARA_CIA", "NASA", "DOD", "FBI", "DOS", "MISC"]

# ── Fallback: map free-text agency names (from YAML) → canonical token ────────
YAML_AGENCY_FALLBACK: dict[str, str] = {
    "department of war":          "DOD",
    "department of defense":      "DOD",
    "dod":                        "DOD",
    "fbi":                        "FBI",
    "federal bureau of investigation": "FBI",
    "nasa":                       "NASA",
    "national aeronautics and space administration": "NASA",
    "dos":                        "DOS",
    "department of state":        "DOS",
    "nara-cia":                   "NARA-CIA",
    "nara_cia":                   "NARA-CIA",
    "misc":                       "MISC",
}

# ── Normalised agency set for fast membership tests ───────────────────────────
_AGENCY_SET = {"NARA-CIA", "NASA", "DOD", "FBI", "DOS", "MISC"}

# ── Known collection sub-folders per agency ────────────────────────────────────
# Values are the canonical collection name returned by the script.
COLLECTION_TOKENS = {
    # DOD
    "mission-reports": "mission-reports",
    "mission_reports": "mission-reports",
    "range-fouler-debriefs": "range-fouler-debriefs",
    "range_fouler_debriefs": "range-fouler-debriefs",
    "email-correspondence": "email-correspondence",
    "email_correspondence": "email-correspondence",
    "reports-other": "reports-other",
    "reports_other": "reports-other",
    # FBI
    "photo-collections": "photo-collections",
    "photo_collections": "photo-collections",
    # NASA
    "transcripts": "transcripts",
    "crew-debriefings": "crew-debriefings",
    "crew_debriefings": "crew-debriefings",
    # DOS
    "cables": "cables",
    # NARA-CIA series / collections
    "hs1-834228961": "hs1-834228961",
    "hs1-101634279": "hs1-101634279",
    "series-18": "series-18",
    "series-38": "series-38",
    "series-59": "series-59",
    "series-255": "series-255",
    "series-331": "series-331",
    "series-341": "series-341",
    "series-342": "series-342",
    # MISC
    "statements-redacted": "statements-redacted",
    "statements_redacted": "statements-redacted",
    "visuals": "visuals",
    "presentations": "presentations",
    "unclassified": "unclassified",
}

# ── Slug-prefix → (agency, collection) ────────────────────────────────────────
# Keyed by lowercase slug prefix; longest prefixes first.
PREFIX_MAP: list[tuple[str, str, str | None]] = [
    # DOD mission reports
    ("dow-uap-d",           "DOD",      "mission-reports"),
    ("dow-uap-",            "DOD",      "mission-reports"),
    ("dod-range-fouler",    "DOD",      "range-fouler-debriefs"),
    ("dod-email",           "DOD",      "email-correspondence"),
    ("dod-",                "DOD",      None),
    ("pr-",                 "DOD",      "mission-reports"),
    # FBI
    ("fbi-",                "FBI",      "photo-collections"),
    # NASA
    ("nasa-transcript",     "NASA",     "transcripts"),
    ("nasa-crew",           "NASA",     "crew-debriefings"),
    ("nasa-",               "NASA",     None),
    # DOS
    ("dos-",                "DOS",      "cables"),
    # NARA-CIA series (slug starts with series number)
    ("65_hs1-834228961",    "NARA-CIA", "hs1-834228961"),
    ("65_hs1-101634279",    "NARA-CIA", "hs1-101634279"),
    ("18_",                 "NARA-CIA", "series-18"),
    ("38_",                 "NARA-CIA", "series-38"),
    ("59_",                 "NARA-CIA", "series-59"),
    ("255_",                "NARA-CIA", "series-255"),
    ("331_",                "NARA-CIA", "series-331"),
    ("341_",                "NARA-CIA", "series-341"),
    ("342_",                "NARA-CIA", "series-342"),
    ("series-",             "NARA-CIA", None),
]

AGENCY_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in AGENCY_TOKENS) + r")\b",
    re.IGNORECASE,
)
COLLECTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(COLLECTION_TOKENS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _norm_agency(token: str) -> str:
    return token.upper().replace("_", "-")


def _path_parts(value: str) -> list[str]:
    """Return path parts trying both Windows and Posix interpretations."""
    for PathCls in (PureWindowsPath, PurePosixPath):
        try:
            parts = list(PathCls(value).parts)
            if len(parts) > 1:
                return parts
        except Exception:
            pass
    return [value]


def extract_both(value) -> tuple[str | None, str | None]:
    """Return (agency, collection) for a single cell value."""
    if not isinstance(value, str) or not value.strip():
        return None, None

    v = value.strip()
    lower = v.lower()

    # 1 ── Path-based: walk parts, find agency then take the next part ──────────
    parts = _path_parts(v)
    for i, part in enumerate(parts):
        norm = _norm_agency(part)
        if norm in _AGENCY_SET:
            agency = norm
            # collection = next path segment after agency, if it exists and is known
            collection = None
            if i + 1 < len(parts):
                nxt = parts[i + 1].lower().replace("_", "-")
                collection = COLLECTION_TOKENS.get(nxt) or COLLECTION_TOKENS.get(parts[i + 1])
            return agency, collection

    # 2 ── Slug prefix map ─────────────────────────────────────────────────────
    for prefix, agency, collection in PREFIX_MAP:
        if lower.startswith(prefix):
            # Try to refine collection from slug body if prefix gave None
            if collection is None:
                m = COLLECTION_RE.search(v)
                if m:
                    collection = COLLECTION_TOKENS.get(m.group(1).lower().replace("_", "-"))
            return agency, collection

    # 3 ── Substring scan for agency ──────────────────────────────────────────
    m_agency = AGENCY_RE.search(v)
    if m_agency:
        agency = _norm_agency(m_agency.group(1))
        m_coll = COLLECTION_RE.search(v)
        collection = COLLECTION_TOKENS.get(m_coll.group(1).lower().replace("_", "-")) if m_coll else None
        return agency, collection

    return None, None


def _find_path_column(df: pd.DataFrame) -> str | None:
    """Heuristic: find the column most likely to contain file paths or slugs."""
    candidates = []
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(30)
        hits = sum(
            1 for s in sample
            if extract_both(s)[0] is not None
            or any(sep in s for sep in ("/", "\\"))
        )
        if hits > 0:
            candidates.append((hits, col))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _read_file(path: Path) -> pd.DataFrame:
    """Read CSV or Excel, detecting format from content (not just extension)."""
    # Try CSV first by sniffing the first bytes
    try:
        with open(path, "rb") as fh:
            header = fh.read(8)
        is_excel = header[:4] in (b"\xd0\xcf\x11\xe0", b"PK\x03\x04")  # XLS or XLSX magic
    except OSError:
        is_excel = path.suffix.lower() in (".xlsx", ".xls", ".xlsm")

    if is_excel:
        return pd.read_excel(path)
    else:
        # CSV — try common separators
        for sep in (",", ";", "\t", "|"):
            try:
                df = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig")
                if len(df.columns) > 1:
                    return df
            except Exception:
                pass
        # Last resort: let pandas infer
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def process(input_path: Path, col_name: str | None, output_path: Path) -> None:
    print(f"Reading: {input_path}")
    df = _read_file(input_path)
    print(f"Columns found: {list(df.columns)}")

    if col_name:
        if col_name not in df.columns:
            sys.exit(f"ERROR: Column '{col_name}' not found. Available: {list(df.columns)}")
        path_col = col_name
    else:
        path_col = _find_path_column(df)
        if path_col is None:
            sys.exit(
                "ERROR: Could not auto-detect a path/slug column. "
                "Re-run with --col <column_name>."
            )
        print(f"Auto-detected path column: '{path_col}'")

    pairs = df[path_col].apply(extract_both)
    agency_values     = pairs.apply(lambda x: x[0])
    collection_values = pairs.apply(lambda x: x[1])

    # ── Fallback 1: may_release_agency_yaml column ────────────────────────────
    # Rows where path extraction failed can borrow from the pre-existing YAML
    # agency column (e.g. "Department of War" → "DOD").
    YAML_COL = "may_release_agency_yaml"
    if YAML_COL in df.columns:
        null_mask = agency_values.isna()
        if null_mask.any():
            yaml_mapped = (
                df.loc[null_mask, YAML_COL]
                .fillna("")
                .str.strip()
                .str.lower()
                .map(YAML_AGENCY_FALLBACK)
            )
            agency_values = agency_values.copy()
            agency_values[null_mask] = yaml_mapped
            fallback_filled = yaml_mapped.notna().sum()
            print(f"Fallback from '{YAML_COL}': filled {fallback_filled} additional agency values")

    # ── Fallback 2: updb_source column ───────────────────────────────────────
    # Some rows have an updb_source that contains the source document slug.
    for extra_col in ("updb_source", "overmeire_reference", "overmeire_Reference"):
        if extra_col not in df.columns:
            continue
        null_mask = agency_values.isna()
        if not null_mask.any():
            break
        extra_pairs = df.loc[null_mask, extra_col].apply(extract_both)
        extra_agency = extra_pairs.apply(lambda x: x[0])
        extra_coll   = extra_pairs.apply(lambda x: x[1])
        filled = extra_agency.notna().sum()
        if filled:
            agency_values = agency_values.copy()
            collection_values = collection_values.copy()
            agency_values[null_mask]     = extra_agency.values
            collection_values[null_mask] = extra_coll.values
            print(f"Fallback from '{extra_col}': filled {filled} additional agency values")

    # Insert both columns right after 'may_release_source_file' if it exists
    if "may_release_source_file" in df.columns:
        insert_pos = df.columns.get_loc("may_release_source_file") + 1
        df.insert(insert_pos,     "agency",     agency_values)
        df.insert(insert_pos + 1, "collection", collection_values)
    else:
        df["agency"]     = agency_values
        df["collection"] = collection_values

    found_agency = df["agency"].notna().sum()
    found_coll   = df["collection"].notna().sum()
    total        = len(df)
    print(f"agency resolved:     {found_agency}/{total}")
    print(f"collection resolved: {found_coll}/{total}")

    if found_agency < total:
        # Show a sample of what's still missing to guide further fixes
        still_missing = df[df["agency"].isna()]
        print(f"\nStill unresolved ({len(still_missing)} rows) — first 10 '{path_col}' values:")
        for v in still_missing[path_col].head(10):
            print(f"  {str(v)[:100]!r}")

    print(f"\nagency distribution:\n{df['agency'].value_counts(dropna=False).to_string()}")
    print(f"\ncollection distribution:\n{df['collection'].value_counts(dropna=False).to_string()}\n")

    if output_path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Add agency + collection columns to an Excel file"
    )
    ap.add_argument("--input", default="duplicates_UPDB_overmeire_reviewed_2.xlsx",
                    help="Input Excel file")
    ap.add_argument("--col",   default=None,
                    help="Column with file paths/slugs (auto-detected if omitted)")
    ap.add_argument("--out",   default=None,
                    help="Output path (default: <stem>_with_agency.xlsx)")
    args = ap.parse_args()

    input_path  = Path(args.input)
    if args.out:
        output_path = Path(args.out)
    else:
        # Preserve the original extension so CSV in → CSV out
        output_path = input_path.with_name(input_path.stem + "_with_agency" + input_path.suffix)
    process(input_path, args.col, output_path)


if __name__ == "__main__":
    main()
