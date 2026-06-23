"""
concat_pages.py
───────────────
Concatenates all page_XXXX.md files for each document into a single
<document-slug>.md at the document folder root AND copies a flat version
into <root>/concat/<document-slug>.md for easy bulk review.

Source (unchanged):
    <doc-slug>/page_0001/page_0001.md
    <doc-slug>/page_0002/page_0002.md
    ...

Output:
    <doc-slug>/<doc-slug>.md          ← all pages joined in order (next to source)
    concat/<doc-slug>.md              ← flat copy for bulk review / sharing

Pipeline with stamp_pages in the middle
---------------------------------------
    python stamp_pages.py --src raw --out pages_out     # stamp individual pages
    python concat_pages.py --src pages_out --execute    # concat stamped pages

Usage
-----
    python concat_pages.py                              # dry-run: list what would be written
    python concat_pages.py --execute                    # write concatenated files
    python concat_pages.py --root D:/divided            # explicit root (also controls concat/ location)
    python concat_pages.py --src pages_out --execute    # read from pages_out/, write concat/ to root
    python concat_pages.py --execute --force            # overwrite existing concatenated files
    python concat_pages.py --no-inline                  # skip writing next to source, only write concat/
"""

import os
import re
import argparse
from pathlib import Path

DEFAULT_ROOT = "D:/divided"

# Directories to never descend into during os.walk / rglob
# NOTE: agency folder names (DOD, NASA, FBI, DOS, NARA-CIA, MISC) are intentionally
# NOT here — we need to descend into them when --src points at pages_out/ or raw/.
SKIP_TRAVERSE = {
    "records", "reports_out", "wiki",
    "concat", "extracted", "scripts", "raw",
    ".git", "__pycache__",
}

# Directory/file names to skip as document-folder *candidates* (not as traversal roots)
SKIP_CANDIDATE = {
    "reorganize.py", "restructure_pages.py", "stamp_pages.py",
    "reconcile.py", "pdf_to_reports.py", "concat_pages.py",
    "find_missing_concat.py", "find_ocr_targets.py", "run_ocr.py",
    "audit.py", "page_coverage.py", "destamp_pages.py",
    "uap_record_schema.yaml", "uap-csv.csv",
    "move_log.json", "page_move_log.json",
    "CLAUDE.md", "Untitled.md", "README.md",
}

PAGE_DIR_RE = re.compile(r"^page_(\d+)$", re.IGNORECASE)

PAGE_SEP = "\n\n---\n\n"   # separator inserted between pages

CONCAT_DIR = "concat"       # flat output folder name (always at root level)


def find_document_dirs(src: Path) -> list[Path]:
    """Return all document folders (those containing page_XXXX/ subfolders)."""
    doc_dirs = []
    for dirpath, dirnames, _ in os.walk(str(src)):
        dirnames[:] = [d for d in dirnames if d not in SKIP_TRAVERSE]
        p = Path(dirpath)
        if p == src or p.name in SKIP_CANDIDATE:
            continue
        try:
            has_pages = any(
                PAGE_DIR_RE.match(sub.name)
                for sub in p.iterdir() if sub.is_dir()
            )
        except (PermissionError, OSError):
            continue
        if has_pages:
            doc_dirs.append(p)
    return sorted(doc_dirs)


def collect_pages(doc_dir: Path) -> list[Path]:
    """Return page .md files sorted by page number."""
    pages = []
    for sub in doc_dir.iterdir():
        if not sub.is_dir():
            continue
        m = PAGE_DIR_RE.match(sub.name)
        if not m:
            continue
        md = sub / f"{sub.name}.md"
        if md.exists():
            pages.append((int(m.group(1)), md))
    return [md for _, md in sorted(pages)]


def build_concat(doc_dir: Path, pages: list[Path]) -> str:
    """Build the full concatenated markdown string."""
    chunks = []
    for md in pages:
        page_num = md.parent.name          # e.g. "page_0001"
        header   = f"## {page_num}\n\n"
        body     = md.read_text(encoding="utf-8", errors="replace").strip()
        chunks.append(header + body)
    return PAGE_SEP.join(chunks)


def process(src: Path, root: Path, execute: bool, force: bool, inline: bool = True) -> None:
    doc_dirs = find_document_dirs(src)

    if not doc_dirs:
        print("  No document folders found.")
        return

    concat_dir = root / CONCAT_DIR

    if execute:
        concat_dir.mkdir(exist_ok=True)

    written = skipped = errors = 0

    for doc_dir in doc_dirs:
        pages = collect_pages(doc_dir)
        if not pages:
            continue

        inline_path = doc_dir / f"{doc_dir.name}.md"
        flat_path   = concat_dir / f"{doc_dir.name}.md"

        # In dry-run mode, report both destinations
        if not execute:
            destinations = []
            if inline and (force or not inline_path.exists()):
                try:
                    destinations.append(str(inline_path.relative_to(root)))
                except ValueError:
                    destinations.append(str(inline_path))
            if force or not flat_path.exists():
                destinations.append(f"{CONCAT_DIR}/{doc_dir.name}.md")
            if destinations:
                for dest in destinations:
                    print(f"  would write  {dest}  ({len(pages)} pages)")
                written += 1
            else:
                skipped += 1
            continue

        # Build content once, write to both destinations
        try:
            content = build_concat(doc_dir, pages)
            wrote_any = False

            if inline:
                if force or not inline_path.exists():
                    inline_path.write_text(content, encoding="utf-8")
                    print(f"  ✓  {inline_path.relative_to(root)}  ({len(pages)} pages)")
                    wrote_any = True

            if force or not flat_path.exists():
                flat_path.write_text(content, encoding="utf-8")
                print(f"  ✓  {CONCAT_DIR}/{doc_dir.name}.md  ({len(pages)} pages)")
                wrote_any = True

            if wrote_any:
                written += 1
            else:
                skipped += 1

        except Exception as exc:
            print(f"  ✗  {doc_dir.name}  ERROR: {exc}")
            errors += 1

    label = "would write" if not execute else "written"
    print(f"\n  {written} {label},  {skipped} skipped (already exist — use --force to overwrite),  {errors} errors\n")
    if execute:
        print(f"  Flat copies → {concat_dir}\n")
        print(f"  Source read → {src}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Concatenate per-page .md files into one .md per document"
    )
    ap.add_argument("--root",      default=DEFAULT_ROOT,
                    help="Root folder — controls where concat/ is written (default: D:/divided)")
    ap.add_argument("--src",       default=None,
                    help="Source folder to scan for documents (default: same as --root). "
                         "Set to pages_out/ when stamp_pages is in the pipeline.")
    ap.add_argument("--execute",   action="store_true",  help="Write files (default: dry-run)")
    ap.add_argument("--force",     action="store_true",  help="Overwrite existing concatenated files")
    ap.add_argument("--no-inline", action="store_true",  help="Skip writing next to source; only write to concat/")
    args = ap.parse_args()

    root = Path(args.root)
    src  = Path(args.src) if args.src else root

    if not root.exists():
        print(f"  ✗  Root not found: {root}")
        return
    if not src.exists():
        print(f"  ✗  Source not found: {src}")
        return

    if not args.execute:
        print(f"\n{'─'*60}")
        print(f"  DRY RUN — nothing will be written")
        print(f"  Source:      {src}")
        print(f"  Root:        {root}")
        print(f"  Flat copies: {root / CONCAT_DIR}")
        print(f"{'─'*60}\n")

    process(src, root, args.execute, args.force, inline=not args.no_inline)

    if not args.execute:
        print("  Run with --execute to apply.\n")


if __name__ == "__main__":
    main()
