"""
restructure_pages.py
────────────────────
Migrates every document folder in D:/divided from Variant B to Variant A.

Variant B (current):
    <doc-slug>/
        page_0001.pdf          ← PDF at document root
        page_0001/
            page_0001.md       ← OCR in subfolder

Variant A (target):
    <doc-slug>/
        page_0001/
            page_0001.pdf      ← PDF moved into subfolder
            page_0001.md       ← OCR already here

Usage
-----
    python restructure_pages.py                     # dry-run, print plan
    python restructure_pages.py --execute           # move PDFs
    python restructure_pages.py --undo              # reverse (reads page_move_log.json)
    python restructure_pages.py --root D:/somewhere # different root
"""

import re
import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

DEFAULT_ROOT = os.environ.get("UAP_PIPELINE_ROOT", ".")
LOG_FILENAME = "page_move_log.json"

# Names that are not document folders
SKIP_NAMES = {
    "reorganize.py",
    "restructure_pages.py",
    "reconcile.py",
    "pdf_to_reports.py",
    "uap_record_schema.yaml",
    "uap-csv.csv",
    "move_log.json",
    "page_move_log.json",
    "records",
    "reports_out",
    "MISC",
    "DOD",
    "NASA",
    "FBI",
    "DOS",
    "NARA-CIA",
    "CLAUDE.md",
    "Untitled.md",
    "README.md",
    "wiki",
    "raw",
    "scripts",
}

PAGE_PDF_RE = re.compile(r"^(page_\d+)\.pdf$", re.IGNORECASE)


def find_moves(root: Path) -> list[dict]:
    """
    Walk every document folder and find root-level page PDFs that need
    to move into their matching page subfolder.
    """
    moves = []

    for doc_dir in sorted(root.iterdir()):
        if not doc_dir.is_dir():
            continue
        if doc_dir.name in SKIP_NAMES:
            continue

        for item in sorted(doc_dir.iterdir()):
            if not item.is_file():
                continue
            m = PAGE_PDF_RE.match(item.name)
            if not m:
                continue

            page_slug = m.group(1)          # e.g. "page_0001"
            subfolder = doc_dir / page_slug  # e.g. .../page_0001/
            dst = subfolder / item.name      # e.g. .../page_0001/page_0001.pdf

            moves.append({
                "doc":  doc_dir.name,
                "src":  str(item),
                "dst":  str(dst),
                "subfolder_exists": subfolder.is_dir(),
            })

    return moves


def print_plan(moves: list[dict]) -> None:
    current_doc = None
    missing_subfolder = []

    for m in moves:
        if m["doc"] != current_doc:
            print(f"\n  📁  {m['doc']}/")
            current_doc = m["doc"]
        src_name = Path(m["src"]).name
        note = "" if m["subfolder_exists"] else "  ⚠ subfolder will be created"
        print(f"       {src_name}  →  {Path(m['dst']).parent.name}/{src_name}{note}")
        if not m["subfolder_exists"]:
            missing_subfolder.append(m["src"])

    print(f"\n  Total: {len(moves)} PDFs to move")
    if missing_subfolder:
        print(f"  ⚠  {len(missing_subfolder)} subfolder(s) will be created (no .md yet):")
        for s in missing_subfolder:
            print(f"       {Path(s).parent.name}/{Path(s).name}")
    print()


def execute_moves(moves: list[dict], log_path: Path) -> None:
    done   = []
    errors = []

    for m in moves:
        src = Path(m["src"])
        dst = Path(m["dst"])

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                print(f"  ⚠  already in place, skipping: {dst}")
                continue
            shutil.move(str(src), str(dst))
            done.append(m)
            print(f"  ✓  {src.parent.name}/{src.name}  →  {dst.parent.name}/")
        except Exception as exc:
            errors.append({"move": m, "error": str(exc)})
            print(f"  ✗  {src.name}  ERROR: {exc}")

    log = {
        "timestamp": datetime.now().isoformat(),
        "moves": done,
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\n  ✅  {len(done)} moved,  {len(errors)} errors")
    print(f"  📝  Undo log → {log_path}\n")


def undo_moves(log_path: Path) -> None:
    if not log_path.exists():
        print(f"  ✗  No undo log found at {log_path}")
        return

    log   = json.loads(log_path.read_text(encoding="utf-8"))
    moves = log.get("moves", [])
    print(f"  Reversing {len(moves)} moves from {log['timestamp']} …\n")

    errors = 0
    for m in reversed(moves):
        src = Path(m["dst"])   # where it ended up
        dst = Path(m["src"])   # where it came from
        try:
            shutil.move(str(src), str(dst))
            print(f"  ↩  {src.name}  →  {dst.parent.name}/")
        except Exception as exc:
            print(f"  ✗  {src.name}  ERROR: {exc}")
            errors += 1

    print(f"\n  ✅  Undo complete.  {len(moves) - errors} restored,  {errors} errors")
    log_path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(
        description="Move root-level page PDFs into their page subfolders (Variant B → A)"
    )
    ap.add_argument("--root",    default=DEFAULT_ROOT)
    ap.add_argument("--execute", action="store_true", help="Actually move files")
    ap.add_argument("--undo",    action="store_true", help="Reverse using page_move_log.json")
    args = ap.parse_args()

    root     = Path(args.root)
    log_path = root / LOG_FILENAME

    if args.undo:
        undo_moves(log_path)
        return

    moves = find_moves(root)

    if not moves:
        print("\n  ✅  Nothing to do — all page PDFs are already in their subfolders.\n")
        return

    if not args.execute:
        print(f"\n{'─'*60}")
        print(f"  DRY RUN — nothing will be moved")
        print(f"  Root: {root}")
        print(f"{'─'*60}")
        print_plan(moves)
        print("  Run with --execute to apply.\n")
    else:
        print(f"\n  Restructuring pages under {root} …\n")
        execute_moves(moves, log_path)


if __name__ == "__main__":
    main()
