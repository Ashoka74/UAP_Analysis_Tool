"""
split_pages.py
──────────────
Splits every multi-page PDF in a folder into single-page PDFs, creating the
standard archive structure:

    <out>/<doc-slug>/
        page_0001/
            page_0001.pdf
        page_0002/
            page_0002.pdf
        ...

By default the split folders are created *inside* the source folder, so:

    FBI/reports/ufo1.pdf  →  FBI/reports/ufo1/page_0001/page_0001.pdf

Safe to re-run — existing page folders are skipped unless --force is used.

Install dependency
------------------
    pip install pypdf

Usage
-----
    python split_pages.py --src D:/divided/FBI/reports
    python split_pages.py --src D:/divided/FBI/reports --out D:/divided/FBI/reports
    python split_pages.py --src D:/divided/FBI/reports --force   # re-split existing
    python split_pages.py --file D:/divided/FBI/reports/ufo1.pdf  # single file
"""

import argparse
from pathlib import Path

def split_pdf(pdf_path: Path, out_root: Path, force: bool = False) -> tuple[int, int]:
    """
    Split a single PDF into per-page PDFs.
    Returns (pages_written, pages_skipped).
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        raise SystemExit("  ✗  pypdf not installed. Run: pip install pypdf")

    doc_slug = pdf_path.stem                     # e.g. "ufo1"
    doc_dir  = out_root / doc_slug

    reader = PdfReader(str(pdf_path))
    n = len(reader.pages)

    written = skipped = 0

    for i, page in enumerate(reader.pages, start=1):
        page_name = f"page_{i:04d}"
        page_dir  = doc_dir / page_name
        page_file = page_dir / f"{page_name}.pdf"

        if page_file.exists() and not force:
            skipped += 1
            continue

        page_dir.mkdir(parents=True, exist_ok=True)

        writer = PdfWriter()
        writer.add_page(page)
        with open(page_file, "wb") as f:
            writer.write(f)

        written += 1

    return written, skipped


def main():
    ap = argparse.ArgumentParser(description="Split multi-page PDFs into per-page archive structure")
    ap.add_argument("--src",   default=None,
                    help="Folder containing PDFs to split")
    ap.add_argument("--out",   default=None,
                    help="Output root (default: same as --src)")
    ap.add_argument("--file",  default=None,
                    help="Split a single PDF instead of a whole folder")
    ap.add_argument("--force", action="store_true",
                    help="Re-split even if page folders already exist")
    args = ap.parse_args()

    if not args.src and not args.file:
        raise SystemExit("  ✗  Provide --src <folder> or --file <pdf>")

    if args.file:
        pdf_path = Path(args.file)
        if not pdf_path.exists():
            raise SystemExit(f"  ✗  File not found: {pdf_path}")
        out_root = Path(args.out) if args.out else pdf_path.parent
        print(f"\n  Splitting {pdf_path.name} → {out_root / pdf_path.stem}/\n")
        w, s = split_pdf(pdf_path, out_root, args.force)
        print(f"  ✓  {w} pages written,  {s} skipped\n")
        return

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"  ✗  Source not found: {src}")

    out_root = Path(args.out) if args.out else src
    pdfs = sorted(src.glob("*.pdf"))

    if not pdfs:
        print(f"  No PDFs found in {src}")
        return

    print(f"\n  Source  : {src}")
    print(f"  Output  : {out_root}")
    print(f"  PDFs    : {len(pdfs)}\n")

    total_written = total_skipped = total_errors = 0

    for pdf_path in pdfs:
        try:
            w, s = split_pdf(pdf_path, out_root, args.force)
            from pypdf import PdfReader
            n = len(PdfReader(str(pdf_path)).pages)
            label = "✓" if w > 0 else "–"
            print(f"  {label}  {pdf_path.name:<30} {n:>4} pages  ({w} written, {s} skipped)")
            total_written += w
            total_skipped += s
        except Exception as exc:
            print(f"  ✗  {pdf_path.name}  ERROR: {exc}")
            total_errors += 1

    print(f"\n  {'─'*50}")
    print(f"  Written : {total_written}")
    print(f"  Skipped : {total_skipped}")
    print(f"  Errors  : {total_errors}")
    print(f"  {'─'*50}\n")


if __name__ == "__main__":
    main()
