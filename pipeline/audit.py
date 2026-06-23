"""
audit.py
────────
Produces a statistical summary of the UAP archive corpus.

Metrics
-------
  • Total documents (PDF + VID) in CSV
  • Released vs pending (CSV rows with/without matching local folder)
  • Redacted — CSV flag + OCR keyword scan per document
  • Contains embedded images / photos in .md files
  • Insufficient evidence (heavy redaction or very thin content)
  • Previously known / reviewed (keyword scan of blurb + OCR)

Output
------
  Prints a summary to stdout.
  Writes a detailed per-document markdown report to --out (default: audit_report.md).

Usage
-----
    python audit.py
    python audit.py --csv uap-csv.csv --root D:/divided --out audit_report.md
"""

import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

DEFAULT_CSV  = "uap-csv.csv"
DEFAULT_ROOT = "D:/divided"
DEFAULT_OUT  = "audit_report.md"

# ── keyword patterns ──────────────────────────────────────────────────────────

RE_REDACTED   = re.compile(r"\[?redacted\]?|\bblacked.out\b|\bcensored\b", re.I)
RE_IMAGE      = re.compile(r"!\[.*?\]\(.*?\)")
RE_INSUFF     = re.compile(
    r"insufficient\s+(?:information|evidence|data)|no\s+(?:further\s+)?detail|"
    r"unable\s+to\s+(?:determine|assess|identify)|not\s+enough\s+(?:data|evidence)|"
    r"inconclusive",
    re.I
)
RE_KNOWN      = re.compile(
    r"previously\s+(?:reported|documented|known|reviewed|identified)|"
    r"prior\s+(?:report|incident|case)|known\s+(?:case|incident)|"
    r"already\s+(?:reported|documented)|follow.up\s+to",
    re.I
)

# ── helpers ───────────────────────────────────────────────────────────────────

PAGE_DIR_RE = re.compile(r"^page_(\d+)$", re.IGNORECASE)

SKIP_NAMES = {
    "reorganize.py", "restructure_pages.py", "stamp_pages.py",
    "reconcile.py", "pdf_to_reports.py", "concat_pages.py", "audit.py",
    "uap_record_schema.yaml", "uap-csv.csv",
    "move_log.json", "page_move_log.json",
    "records", "reports_out", "pages_out", "wiki", "scripts",
    "MISC", "DOD", "NASA", "FBI", "DOS", "NARA-CIA",
    "CLAUDE.md", "Untitled.md", "README.md",
}


def _slug(title: str) -> str:
    s = title.lower().strip()
    s = re.sub(r"[\s,]+", "-", s)
    s = re.sub(r"[^\w\-]", "", s)
    return re.sub(r"-+", "-", s).strip("-")


def _index_docs(root: Path) -> dict[str, Path]:
    """Map doc-slug → Path for every folder containing page_XXXX subfolders."""
    index = {}
    for d in root.rglob("*"):
        if not d.is_dir() or d.name in SKIP_NAMES:
            continue
        if any(PAGE_DIR_RE.match(sub.name) for sub in d.iterdir() if sub.is_dir()):
            index[d.name.lower()] = d
    return index


def _collect_md_texts(doc_dir: Path) -> list[tuple[str, str]]:
    """Return [(page_name, text), ...] sorted by page number."""
    pages = []
    for sub in doc_dir.iterdir():
        if not sub.is_dir():
            continue
        m = PAGE_DIR_RE.match(sub.name)
        if not m:
            continue
        md = sub / f"{sub.name}.md"
        text = md.read_text(encoding="utf-8", errors="replace") if md.exists() else ""
        pages.append((int(m.group(1)), sub.name, text))
    return [(name, text) for _, name, text in sorted(pages)]


def _find_doc_dir(index: dict[str, Path], title: str) -> Path | None:
    slug = _slug(title)
    if slug in index:
        return index[slug]
    prefix = slug[:20]
    for key, path in index.items():
        if key.startswith(prefix):
            return path
    words = set(slug.split("-")[:5])
    for key, path in index.items():
        kwords = set(key.split("-")[:5])
        if len(words & kwords) >= min(3, len(words)):
            return path
    return None


# ── per-document analysis ─────────────────────────────────────────────────────

def analyse_document(title: str, csv_row: dict, doc_dir: Path | None) -> dict:
    result = {
        "title":           title,
        "agency":          csv_row.get("Agency", "").strip(),
        "doc_type":        csv_row.get("Type", "PDF").strip().upper(),
        "csv_redacted":    csv_row.get("Redaction", "").strip().upper() == "TRUE",
        "release_date":    csv_row.get("Release Date", "").strip(),
        "incident_date":   csv_row.get("Incident Date", "").strip(),
        "location":        csv_row.get("Incident Location", "").strip(),
        "blurb":           csv_row.get("Description Blurb", "").replace("\n", " ").strip(),
        # page-level findings
        "found_locally":   doc_dir is not None,
        "page_count":      0,
        "ocr_redacted":    False,
        "redacted_pages":  [],
        "has_images":      False,
        "image_pages":     [],
        "insuff_evidence": False,
        "previously_known":False,
        "total_chars":     0,
    }

    blurb = result["blurb"]
    if RE_INSUFF.search(blurb):
        result["insuff_evidence"] = True
    if RE_KNOWN.search(blurb):
        result["previously_known"] = True

    if doc_dir is None:
        return result

    pages = _collect_md_texts(doc_dir)
    result["page_count"] = len(pages)

    all_text = ""
    for page_name, text in pages:
        all_text += text
        if RE_REDACTED.search(text):
            result["ocr_redacted"] = True
            result["redacted_pages"].append(page_name)
        if RE_IMAGE.search(text):
            result["has_images"] = True
            result["image_pages"].append(page_name)
        if RE_INSUFF.search(text):
            result["insuff_evidence"] = True
        if RE_KNOWN.search(text):
            result["previously_known"] = True

    result["total_chars"] = len(all_text)

    # Insufficient evidence heuristic: very thin content or mostly redacted
    if result["page_count"] > 0:
        redact_ratio = len(result["redacted_pages"]) / result["page_count"]
        avg_chars    = result["total_chars"] / result["page_count"]
        if redact_ratio > 0.5 or avg_chars < 200:
            result["insuff_evidence"] = True

    return result


# ── report writer ─────────────────────────────────────────────────────────────

def write_report(docs: list[dict], out_path: Path) -> None:
    total       = len(docs)
    found       = sum(1 for d in docs if d["found_locally"])
    pending     = total - found
    pdfs        = sum(1 for d in docs if d["doc_type"] == "PDF")
    vids        = sum(1 for d in docs if d["doc_type"] == "VID")
    csv_redact  = sum(1 for d in docs if d["csv_redacted"])
    ocr_redact  = sum(1 for d in docs if d["ocr_redacted"])
    has_images  = sum(1 for d in docs if d["has_images"])
    insuff      = sum(1 for d in docs if d["insuff_evidence"])
    known       = sum(1 for d in docs if d["previously_known"])

    lines = [
        f"# UAP Archive Audit Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total documents in CSV | {total} |",
        f"| — PDFs | {pdfs} |",
        f"| — Videos (VID) | {vids} |",
        f"| Found locally (pages on disk) | {found} |",
        f"| **Pending / not yet processed** | **{pending}** |",
        f"| Redacted (CSV flag) | {csv_redact} |",
        f"| Redacted (detected in OCR text) | {ocr_redact} |",
        f"| Contains embedded photos/images | {has_images} |",
        f"| Insufficient evidence | {insuff} |",
        f"| Previously known / reviewed | {known} |",
        "",
        "---",
        "",
        "## Pending (no local folder found)",
        "",
    ]

    pending_docs = [d for d in docs if not d["found_locally"]]
    if pending_docs:
        for d in pending_docs:
            lines.append(f"- {d['title']}  _(agency: {d['agency']}, type: {d['doc_type']})_")
    else:
        lines.append("_None — all CSV rows matched a local folder._")

    lines += ["", "---", "", "## Redacted documents (OCR-detected)", ""]
    redact_docs = [d for d in docs if d["ocr_redacted"]]
    if redact_docs:
        lines.append("| Document | Pages with redactions | Total pages |")
        lines.append("|----------|-----------------------|-------------|")
        for d in redact_docs:
            pages_str = ", ".join(d["redacted_pages"][:5])
            if len(d["redacted_pages"]) > 5:
                pages_str += f" … +{len(d['redacted_pages'])-5} more"
            lines.append(f"| {d['title'][:60]} | {pages_str} | {d['page_count']} |")
    else:
        lines.append("_None detected._")

    lines += ["", "---", "", "## Documents with embedded photos/images", ""]
    img_docs = [d for d in docs if d["has_images"]]
    if img_docs:
        lines.append("| Document | Pages with images |")
        lines.append("|----------|-------------------|")
        for d in img_docs:
            lines.append(f"| {d['title'][:60]} | {', '.join(d['image_pages'])} |")
    else:
        lines.append("_None detected._")

    lines += ["", "---", "", "## Insufficient evidence", ""]
    insuff_docs = [d for d in docs if d["insuff_evidence"]]
    if insuff_docs:
        for d in insuff_docs:
            lines.append(f"- {d['title']}  _(pages: {d['page_count']}, redacted pages: {len(d['redacted_pages'])})_")
    else:
        lines.append("_None flagged._")

    lines += ["", "---", "", "## Previously known / reviewed", ""]
    known_docs = [d for d in docs if d["previously_known"]]
    if known_docs:
        for d in known_docs:
            lines.append(f"- {d['title']}")
    else:
        lines.append("_None detected._")

    lines += ["", "---", "", "## Full document inventory", ""]
    lines.append("| # | Title | Agency | Type | Local | Redacted | Images | Insuff | Known |")
    lines.append("|---|-------|--------|------|-------|----------|--------|--------|-------|")
    for i, d in enumerate(docs, 1):
        lines.append(
            f"| {i} | {d['title'][:50]} | {d['agency']} | {d['doc_type']} "
            f"| {'✓' if d['found_locally'] else '✗'} "
            f"| {'✓' if d['csv_redacted'] or d['ocr_redacted'] else '—'} "
            f"| {'✓' if d['has_images'] else '—'} "
            f"| {'✓' if d['insuff_evidence'] else '—'} "
            f"| {'✓' if d['previously_known'] else '—'} |"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Audit the UAP archive corpus")
    ap.add_argument("--csv",  default=DEFAULT_CSV)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out",  default=DEFAULT_OUT)
    args = ap.parse_args()

    root     = Path(args.root)
    out_path = Path(args.out)

    print(f"\n  Reading CSV: {args.csv}")
    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"  {len(rows)} rows found")

    print(f"  Indexing local folders under {root} …")
    doc_index = _index_docs(root)
    print(f"  {len(doc_index)} document folders found\n")

    docs = []
    for i, row in enumerate(rows):
        title = row.get("Title", "").replace("\n", " ").strip()
        if not title:
            continue
        doc_dir = _find_doc_dir(doc_index, title)
        result  = analyse_document(title, row, doc_dir)
        docs.append(result)
        status = "✓" if doc_dir else "✗"
        if (i + 1) % 10 == 0 or not doc_dir:
            print(f"  {status}  [{i+1:>3d}] {title[:60]}")

    # ── print summary ─────────────────────────────────────────────────────────
    total      = len(docs)
    found      = sum(1 for d in docs if d["found_locally"])
    print(f"\n{'─'*60}")
    print(f"  AUDIT SUMMARY")
    print(f"{'─'*60}")
    print(f"  Total in CSV          : {total}")
    print(f"  PDFs                  : {sum(1 for d in docs if d['doc_type']=='PDF')}")
    print(f"  Videos                : {sum(1 for d in docs if d['doc_type']=='VID')}")
    print(f"  Found locally         : {found}")
    print(f"  Pending (not on disk) : {total - found}")
    print(f"  Redacted (CSV)        : {sum(1 for d in docs if d['csv_redacted'])}")
    print(f"  Redacted (OCR scan)   : {sum(1 for d in docs if d['ocr_redacted'])}")
    print(f"  Has photos/images     : {sum(1 for d in docs if d['has_images'])}")
    print(f"  Insufficient evidence : {sum(1 for d in docs if d['insuff_evidence'])}")
    print(f"  Previously known      : {sum(1 for d in docs if d['previously_known'])}")
    print(f"{'─'*60}\n")

    write_report(docs, out_path)
    print(f"  ✅  Full report → {out_path}\n")


if __name__ == "__main__":
    main()
