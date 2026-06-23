"""
stamp_pages.py
──────────────
Creates enriched copies of every page_XXXX.md, with a minimal YAML frontmatter
block prepended. The originals in raw/ (or D:/divided) are never modified.

Output mirrors the source folder hierarchy under --out-dir (default: pages_out/).

Example
-------
    Source:  raw/DOD/mission-reports/greece/dow-uap-d33-.../page_0001/page_0001.md
    Output:  pages_out/DOD/mission-reports/greece/dow-uap-d33-.../page_0001/page_0001.md

    Frontmatter prepended to each copy:

        ---
        document: dow-uap-d33-mission-report-greece-october-2023
        page: 1
        of: 5
        agency: DOD
        subtype: mission-reports
        region: greece
        src_path: raw/DOD/mission-reports/greece/dow-uap-d33-.../page_0001/page_0001.md
        ---

Usage
-----
    python stamp_pages.py                                # uses raw/ as source
    python stamp_pages.py --src D:/divided               # pre-reorganize flat layout
    python stamp_pages.py --src raw --out pages_out
    python stamp_pages.py --src raw --out pages_out --force   # overwrite existing copies
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

DEFAULT_SRC = "raw"
DEFAULT_OUT = "pages_out"

# Depth hints for extracting agency / subtype / region from the path
# Works for both the organised raw/ tree AND the flat D:/divided layout.
AGENCY_NAMES = {"DOD", "FBI", "NASA", "DOS", "NARA-CIA", "MISC"}

PAGE_MD_RE = re.compile(r"^page_(\d+)\.md$", re.IGNORECASE)


# ── path parsing ──────────────────────────────────────────────────────────────

def parse_path_context(doc_dir: Path, src_root: Path) -> dict:
    """
    Extract agency / subtype / region from the path relative to src_root.

    Organised tree:   src_root/DOD/mission-reports/greece/<doc-slug>/
    Flat layout:      src_root/<doc-slug>/

    Returns a dict with keys: agency, subtype, region (all may be None).
    """
    try:
        rel_parts = doc_dir.relative_to(src_root).parts
    except ValueError:
        rel_parts = (doc_dir.name,)

    agency  = None
    subtype = None
    region  = None

    if len(rel_parts) >= 4:
        # organised: agency / subtype / region / doc-slug
        agency  = rel_parts[0] if rel_parts[0] in AGENCY_NAMES else None
        subtype = rel_parts[1]
        region  = rel_parts[2]
    elif len(rel_parts) >= 2:
        agency  = rel_parts[0] if rel_parts[0] in AGENCY_NAMES else None

    return {"agency": agency, "subtype": subtype, "region": region}


# ── discovery ─────────────────────────────────────────────────────────────────

def collect_documents(src_root: Path) -> list[dict]:
    """
    Walk src_root and group page .md files by document folder.
    Returns a list of dicts, one per document.
    """
    # Map: doc_dir_path → [page paths sorted]
    doc_pages: dict[Path, list[Path]] = defaultdict(list)

    for md_file in src_root.rglob("page_*.md"):
        # The document folder is the grandparent: doc_dir/page_XXXX/page_XXXX.md
        if md_file.parent.parent == src_root:
            # flat layout: src_root/doc_dir/page_XXXX/page_XXXX.md
            doc_dir = md_file.parent.parent / md_file.parent.name
            # Actually: md_file.parent IS the page subfolder,
            # its parent IS the doc dir
        doc_dir = md_file.parent.parent
        if doc_dir == src_root:
            continue  # skip .md files sitting directly in root
        doc_pages[doc_dir].append(md_file)

    documents = []
    for doc_dir, pages in sorted(doc_pages.items()):
        pages_sorted = sorted(pages, key=lambda p: p.name)
        ctx = parse_path_context(doc_dir, src_root)
        documents.append({
            "doc_dir":   doc_dir,
            "doc_slug":  doc_dir.name,
            "page_count": len(pages_sorted),
            "pages":     pages_sorted,
            **ctx,
        })

    return documents


# ── frontmatter builder ───────────────────────────────────────────────────────

def build_frontmatter(doc: dict, md_file: Path, page_index: int, src_root: Path) -> str:
    try:
        src_path = md_file.relative_to(src_root).as_posix()
    except ValueError:
        src_path = md_file.as_posix()

    def _yaml_str(v) -> str:
        return f'"{v}"' if v is not None else "null"

    lines = [
        "---",
        f"document: {doc['doc_slug']}",
        f"page: {page_index}",
        f"of: {doc['page_count']}",
        f"agency: {_yaml_str(doc['agency'])}",
        f"subtype: {_yaml_str(doc['subtype'])}",
        f"region: {_yaml_str(doc['region'])}",
        f"src_path: {src_path}",
        "---",
        "",
    ]
    return "\n".join(lines)


# ── stamping ──────────────────────────────────────────────────────────────────

def stamp_documents(documents: list[dict], src_root: Path, out_root: Path,
                    force: bool = False) -> tuple[int, int, int]:
    written = skipped = errors = 0

    for doc in documents:
        for idx, md_file in enumerate(doc["pages"], start=1):
            try:
                rel = md_file.relative_to(src_root)
            except ValueError:
                rel = Path(doc["doc_slug"]) / md_file.parent.name / md_file.name

            dst = out_root / rel

            if dst.exists() and not force:
                skipped += 1
                continue

            try:
                original = md_file.read_text(encoding="utf-8", errors="replace")
                frontmatter = build_frontmatter(doc, md_file, idx, src_root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(frontmatter + original, encoding="utf-8")
                written += 1
            except Exception as exc:
                print(f"  ✗  {md_file}  ERROR: {exc}")
                errors += 1

    return written, skipped, errors


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Copy page .md files with YAML frontmatter into pages_out/"
    )
    ap.add_argument("--src",   default=DEFAULT_SRC,
                    help="Source root (raw/ after reorganize, or D:/divided before)")
    ap.add_argument("--out",   default=DEFAULT_OUT,
                    help="Output root for stamped copies (default: pages_out/)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output files")
    args = ap.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)

    if not src_root.exists():
        print(f"  ✗  Source not found: {src_root}")
        return

    print(f"\n  Scanning {src_root} …")
    documents = collect_documents(src_root)

    total_pages = sum(d["page_count"] for d in documents)
    print(f"  Found {len(documents)} documents, {total_pages} pages total")
    print(f"  Output → {out_root}/")
    if not args.force:
        print(f"  (existing files will be skipped — use --force to overwrite)\n")

    written, skipped, errors = stamp_documents(documents, src_root, out_root, args.force)

    print(f"\n  ✅  {written} written,  {skipped} skipped,  {errors} errors\n")


if __name__ == "__main__":
    main()
