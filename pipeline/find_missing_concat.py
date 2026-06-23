"""
find_missing_concat.py
──────────────────────
Compares document folders (those with page_XXXX subfolders) against the
files already written to concat/, and reports what's missing and why.

Usage
-----
    python find_missing_concat.py
    python find_missing_concat.py --root D:/divided
"""

import re
import os
import argparse
from pathlib import Path

DEFAULT_ROOT = "D:/divided"

PAGE_DIR_RE = re.compile(r"^page_(\d+)$", re.IGNORECASE)
SKIP_NAMES = {
    "reorganize.py", "restructure_pages.py", "stamp_pages.py",
    "reconcile.py", "pdf_to_reports.py", "concat_pages.py",
    "find_missing_concat.py", "audit.py",
    "uap_record_schema.yaml", "uap-csv.csv",
    "move_log.json", "page_move_log.json",
    "records", "reports_out", "pages_out", "wiki",
    "MISC", "DOD", "NASA", "FBI", "DOS", "NARA-CIA",
    "CLAUDE.md", "Untitled.md", "README.md",
    "scripts", "raw", "concat",
}


def find_document_dirs(root: Path) -> list[Path]:
    doc_dirs = []
    for dirpath, dirnames, _ in os.walk(str(root)):
        # Prune skip names in-place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if d not in SKIP_NAMES]
        p = Path(dirpath)
        if p == root:
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


def count_md_pages(doc_dir: Path) -> int:
    count = 0
    try:
        for sub in doc_dir.iterdir():
            if sub.is_dir() and PAGE_DIR_RE.match(sub.name):
                md = sub / f"{sub.name}.md"
                if md.exists():
                    count += 1
    except (PermissionError, OSError):
        pass
    return count


def main():
    ap = argparse.ArgumentParser(description="Find document folders missing from concat/")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()

    root = Path(args.root)
    concat_dir = root / "concat"

    print(f"\n  Scanning {root} …")
    doc_dirs = find_document_dirs(root)
    print(f"  Found {len(doc_dirs)} document folders with page_XXXX subfolders")

    existing = set()
    if concat_dir.exists():
        existing = {p.stem for p in concat_dir.glob("*.md")}
    print(f"  Files in concat/: {len(existing)}")

    # Categorise missing docs
    no_md    = []   # page dirs exist but no .md files in them
    has_md   = []   # has .md pages but not in concat/

    for d in doc_dirs:
        if d.name in existing:
            continue
        md_count = count_md_pages(d)
        if md_count == 0:
            no_md.append((d.name, str(d.relative_to(root))))
        else:
            has_md.append((d.name, md_count, str(d.relative_to(root))))

    print(f"\n{'─'*65}")
    print(f"  Missing from concat/: {len(no_md) + len(has_md)}")
    print(f"{'─'*65}")

    if has_md:
        print(f"\n  ❌  Have .md pages but NOT in concat/ ({len(has_md)}):")
        print(f"      (concat_pages.py should have caught these — investigate)")
        for name, cnt, rel in has_md:
            print(f"      [{cnt:>3d} pages]  {rel}")

    if no_md:
        print(f"\n  ⚠️   No .md files in any page subfolder ({len(no_md)}):")
        print(f"      (OCR not yet run, or pages are PDF-only)")
        for name, rel in no_md:
            print(f"                   {rel}")

    if not has_md and not no_md:
        print("\n  ✅  All document folders are represented in concat/.")

    print()


if __name__ == "__main__":
    main()
