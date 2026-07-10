"""
find_ocr_targets.py
───────────────────
Walks every document folder and finds page_XXXX subfolders where a .pdf
exists but the matching .md does not — these are your OCR targets.

Output
------
  stdout  — grouped list per document
  ocr_targets.txt  — one absolute PDF path per line (pipe-friendly)
  ocr_targets.json — structured: {doc, page, pdf_path, expected_md_path}

Usage
-----
    python find_ocr_targets.py
    python find_ocr_targets.py --root D:/divided
"""

import re
import os
import json
import argparse
from pathlib import Path

DEFAULT_ROOT = os.environ.get("UAP_PIPELINE_ROOT", ".")

PAGE_DIR_RE = re.compile(r"^page_(\d+)$", re.IGNORECASE)

# Folders to skip entirely during traversal (never descend into them)
SKIP_TRAVERSE = {
    "records", "reports_out", "pages_out", "wiki",
    "concat", "extracted", "scripts", "raw",
    ".git", "__pycache__",
}

# Folder/file names that are not document dirs (skip as candidates only)
SKIP_CANDIDATE = {
    "reorganize.py", "restructure_pages.py", "stamp_pages.py",
    "reconcile.py", "pdf_to_reports.py", "concat_pages.py",
    "find_missing_concat.py", "audit.py", "page_coverage.py",
    "find_ocr_targets.py", "extract_reports.py",
    "uap_record_schema.yaml", "uap-csv.csv",
    "move_log.json", "page_move_log.json",
    "CLAUDE.md", "Untitled.md", "README.md",
    # agency folders are valid traversal roots — don't skip them here
}


def find_document_dirs(root: Path) -> list[Path]:
    doc_dirs = []
    for dirpath, dirnames, _ in os.walk(str(root)):
        # Prune traversal — never descend into output/tool folders
        dirnames[:] = [d for d in dirnames if d not in SKIP_TRAVERSE]
        p = Path(dirpath)
        if p == root or p.name in SKIP_CANDIDATE:
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


def main():
    ap = argparse.ArgumentParser(description="Find pages with PDF but no .md (OCR targets)")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()

    root = Path(args.root)
    print(f"\n  Scanning {root} …\n")

    doc_dirs = find_document_dirs(root)

    targets = []          # {doc, page, pdf_path, expected_md_path}
    docs_with_gaps = []

    for doc_dir in doc_dirs:
        doc_targets = []
        try:
            subs = sorted(
                (sub for sub in doc_dir.iterdir()
                 if sub.is_dir() and PAGE_DIR_RE.match(sub.name)),
                key=lambda s: s.name
            )
        except (PermissionError, OSError):
            continue

        for sub in subs:
            md  = sub / f"{sub.name}.md"
            if md.exists():
                continue   # OCR already done — skip

            # .md is missing — find whatever source file is present
            # Check for PDF with matching name, then any PDF, then any file at all
            pdf_match = sub / f"{sub.name}.pdf"
            if pdf_match.exists():
                source = pdf_match
            else:
                # Look for any PDF or image in this page folder
                others = [f for f in sub.iterdir() if f.is_file()
                          and f.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}]
                source = others[0] if others else None

            doc_targets.append({
                "doc":              doc_dir.name,
                "page":             sub.name,
                "pdf_path":         str(source) if source else "NOT FOUND",
                "expected_md_path": str(md),
                "has_source_file":  source is not None,
            })

        if doc_targets:
            docs_with_gaps.append((doc_dir.name, doc_targets))
            targets.extend(doc_targets)

    # ── stdout ────────────────────────────────────────────────────────────────
    if not targets:
        print("  ✅  No gaps — every page_XXXX.pdf has a matching .md file.\n")
        return

    no_source = sum(1 for t in targets if not t["has_source_file"])

    print(f"  {'─'*65}")
    print(f"  Documents with missing .md  : {len(docs_with_gaps)}")
    print(f"  Total pages needing OCR     : {len(targets)}")
    print(f"    — have a source PDF/image : {len(targets) - no_source}")
    print(f"    — NO source file found    : {no_source}  ← page folder is empty")
    print(f"  {'─'*65}\n")

    for doc_name, doc_targets in docs_with_gaps:
        pages = [t["page"] for t in doc_targets]
        nums = sorted(int(re.search(r"\d+", p).group()) for p in pages)
        runs = []
        start = prev = nums[0]
        for n in nums[1:]:
            if n == prev + 1:
                prev = n
            else:
                runs.append(f"{start}" if start == prev else f"{start}–{prev}")
                start = prev = n
        runs.append(f"{start}" if start == prev else f"{start}–{prev}")

        missing_src = sum(1 for t in doc_targets if not t["has_source_file"])
        src_note = f"  ⚠ {missing_src} page folder(s) have no source file" if missing_src else ""
        print(f"  {doc_name}")
        print(f"    Missing {len(doc_targets)} page(s): {', '.join(runs)}{src_note}")
        for t in doc_targets:
            src_flag = "  ← NO SOURCE FILE" if not t["has_source_file"] else ""
            print(f"    → {t['pdf_path']}{src_flag}")
        print()

    # ── ocr_targets.txt ───────────────────────────────────────────────────────
    txt_path = root / "ocr_targets.txt"
    txt_path.write_text(
        "\n".join(t["pdf_path"] for t in targets) + "\n",
        encoding="utf-8"
    )
    print(f"  Flat list  → {txt_path}")

    # ── ocr_targets.json ──────────────────────────────────────────────────────
    json_path = root / "ocr_targets.json"
    json_path.write_text(
        json.dumps(targets, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  JSON list  → {json_path}\n")


if __name__ == "__main__":
    main()
