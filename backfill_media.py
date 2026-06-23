"""Non-destructive backfill of the 7 web-card media columns for rows that are
missing them (the PURSUE3 gap: ingested document_id-keyed, so the source_file
manifest join never matched them).

Matches each media-less row to the war.gov manifest by document doc-code
(e.g. CIA-UAP-010, DOW-UAP-D084) and fills ONLY currently-empty cells. Never
overwrites existing values; writes NEW *_mediafilled files, leaving the
originals untouched.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path

MANIFEST = "uap-data.csv"
TGT = ["Title", "Description Blurb", "Agency", "PDF | Image Link",
       "Modal Image", "Image Alt Text", "Image VIRIN"]
_CODE = re.compile(r"[A-Za-z]{2,4}-UAP-[A-Za-z]?\d+")
_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def code(s):
    m = _CODE.search(str(s))
    return m.group(0).upper() if m else ""


def build_lookup(man=MANIFEST):
    M = pd.read_csv(man, low_memory=False)
    M["_code"] = M.apply(
        lambda r: code(r.get("Title")) or code(Path(str(r.get("PDF | Image Link", ""))).stem),
        axis=1)
    M = M[M["_code"] != ""].copy()
    # prefer the PDF document card over video/audio cards for the same code
    M["_ispdf"] = M["PDF | Image Link"].astype(str).str.lower().str.endswith(".pdf")
    M = M.sort_values("_ispdf", ascending=False).drop_duplicates("_code", keep="first")
    return {r["_code"]: {c: r.get(c) for c in TGT} for _, r in M.iterrows()}


def _empty(v):
    return pd.isna(v) or str(v).strip().lower() in ("", "nan", "none")


def backfill(df, lookup, keycols=("document_id", "source_file")):
    for c in TGT:
        if c not in df.columns:
            df[c] = np.nan
    filled_rows = 0
    for i in df.index:
        if not _empty(df.at[i, "Title"]):
            continue                       # row already has media — never touch it
        cd = ""
        for kc in keycols:
            if kc in df.columns:
                cd = code(df.at[i, kc])
                if cd:
                    break
        card = lookup.get(cd)
        if not card:
            continue
        touched = False
        for c in TGT:
            if _empty(df.at[i, c]) and not _empty(card.get(c)):
                df.at[i, c] = card.get(c)
                touched = True
        filled_rows += int(touched)
    return df, filled_rows


def _sanitize(df):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: _ILLEGAL.sub("", v) if isinstance(v, str) else v)
    return df


def report(df, label):
    if "release" in df.columns:
        print(f"  [{label}] non-null Title by release:",
              df.groupby("release")["Title"].apply(lambda s: int(s.notna().sum())).to_dict())


def run(in_path, out_base, lookup, read_excel_sheet=None):
    if read_excel_sheet:
        df = pd.read_excel(in_path, sheet_name=read_excel_sheet)
    else:
        df = pd.read_csv(in_path, low_memory=False)
    print(f"\n{in_path} → {df.shape}")
    report(df, "before")
    df, n = backfill(df, lookup)
    report(df, "after ")
    print(f"  rows backfilled: {n}")
    df = _sanitize(df)
    df.to_csv(out_base + ".csv", index=False)
    df.to_excel(out_base + ".xlsx", index=False)
    print(f"  wrote {out_base}.csv / .xlsx")
    return df


if __name__ == "__main__":
    lut = build_lookup()
    print(f"manifest doc-code lookup: {len(lut)} codes")
    run("SUBDATASETS_V2/PURSUE_1_2_3_normalized_full.csv",
        "SUBDATASETS_V2/PURSUE_1_2_3_normalized_full_mediafilled", lut)
    run("PURSUE_1_2_3_dedup_flagged.csv",
        "PURSUE_1_2_3_dedup_flagged_mediafilled", lut)
