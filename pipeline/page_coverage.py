"""
page_coverage.py
────────────────
For every document folder in the archive, counts:
  - Total page_XXXX subfolders  (pages that exist as folders)
  - Pages WITH a matching .md file  (OCR complete)
  - Pages WITHOUT a .md file         (gap — PDF only or missing)

Outputs:
  - Summary table to stdout
  - page_coverage.json  for downstream use / graphing
  - page_coverage.csv   for spreadsheet / manual review

Usage
-----
    python page_coverage.py
    python page_coverage.py --root D:/divided
    python page_coverage.py --root D:/divided --out-dir .
"""

import re
import os
import csv
import json
import argparse
from pathlib import Path

DEFAULT_ROOT    = os.environ.get("UAP_PIPELINE_ROOT", ".")
DEFAULT_OUT_DIR = DEFAULT_ROOT

PAGE_DIR_RE = re.compile(r"^page_(\d+)$", re.IGNORECASE)

SKIP_TRAVERSE = {
    "records", "reports_out", "pages_out", "wiki",
    "concat", "extracted", "scripts", "raw",
    ".git", "__pycache__",
}

SKIP_CANDIDATE = {
    "reorganize.py", "restructure_pages.py", "stamp_pages.py",
    "reconcile.py", "pdf_to_reports.py", "concat_pages.py",
    "find_missing_concat.py", "audit.py", "page_coverage.py",
    "uap_record_schema.yaml", "uap-csv.csv",
    "move_log.json", "page_move_log.json",
    "CLAUDE.md", "Untitled.md", "README.md",
}


def find_document_dirs(root: Path) -> list[Path]:
    doc_dirs = []
    for dirpath, dirnames, _ in os.walk(str(root)):
        dirnames[:] = [d for d in dirnames if d not in SKIP_TRAVERSE]
        p = Path(dirpath)
        if p == root or p.name in SKIP_CANDIDATE:
            continue
        try:
            page_subs = [
                sub for sub in p.iterdir()
                if sub.is_dir() and PAGE_DIR_RE.match(sub.name)
            ]
        except (PermissionError, OSError):
            continue
        if page_subs:
            doc_dirs.append(p)
    return sorted(doc_dirs)


def analyse_doc(doc_dir: Path) -> dict:
    total_pages = 0
    pages_with_md = 0
    missing_pages = []

    try:
        subs = [sub for sub in doc_dir.iterdir() if sub.is_dir()]
    except (PermissionError, OSError):
        subs = []

    for sub in subs:
        m = PAGE_DIR_RE.match(sub.name)
        if not m:
            continue
        total_pages += 1
        md = sub / f"{sub.name}.md"
        if md.exists():
            pages_with_md += 1
        else:
            missing_pages.append(sub.name)

    return {
        "doc":          doc_dir.name,
        "total_pages":  total_pages,
        "with_md":      pages_with_md,
        "missing_md":   total_pages - pages_with_md,
        "coverage_pct": round(100 * pages_with_md / total_pages, 1) if total_pages else 0,
        "missing_list": missing_pages,
    }


def main():
    ap = argparse.ArgumentParser(description="Count page coverage (.md vs total) per document")
    ap.add_argument("--root",    default=DEFAULT_ROOT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    root    = Path(args.root)
    out_dir = Path(args.out_dir)

    print(f"\n  Scanning {root} …")
    doc_dirs = find_document_dirs(root)
    print(f"  Found {len(doc_dirs)} document folders\n")

    results = [analyse_doc(d) for d in doc_dirs]

    # ── stdout summary ────────────────────────────────────────────────────────
    total_pages   = sum(r["total_pages"]  for r in results)
    total_with_md = sum(r["with_md"]      for r in results)
    total_missing = sum(r["missing_md"]   for r in results)
    full_coverage = sum(1 for r in results if r["missing_md"] == 0)
    partial       = sum(1 for r in results if 0 < r["missing_md"] < r["total_pages"])
    no_md         = sum(1 for r in results if r["with_md"] == 0)

    print(f"{'─'*72}")
    print(f"  {'Document':<55} {'Pages':>5}  {'MD':>5}  {'Gap':>5}  {'Cov%':>5}")
    print(f"{'─'*72}")
    for r in results:
        gap_flag = "  ⚠" if r["missing_md"] > 0 else ""
        print(f"  {r['doc'][:55]:<55} {r['total_pages']:>5}  {r['with_md']:>5}  {r['missing_md']:>5}{gap_flag}")
    print(f"{'─'*72}")
    print(f"  {'TOTAL':<55} {total_pages:>5}  {total_with_md:>5}  {total_missing:>5}")
    print(f"{'─'*72}")
    print(f"\n  Docs with full MD coverage : {full_coverage}")
    print(f"  Docs with partial coverage : {partial}  ← pages missing .md")
    print(f"  Docs with NO .md at all    : {no_md}")
    print()

    # ── JSON output ───────────────────────────────────────────────────────────
    json_path = out_dir / "page_coverage.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON → {json_path}")

    # ── CSV output ────────────────────────────────────────────────────────────
    csv_path = out_dir / "page_coverage.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["doc", "total_pages", "with_md", "missing_md", "coverage_pct"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV  → {csv_path}\n")


if __name__ == "__main__":
    main()
