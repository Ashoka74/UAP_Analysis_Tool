"""
destamp_pages.py
────────────────
Strips the YAML frontmatter stamp prepended by stamp_pages.py from every
page_XXXX.md file.

The stamp always looks like this at the top of the file:
    ---
    document: ...
    page: ...
    of: ...
    agency: ...
    subtype: ...
    region: ...
    src_path: ...
    ---
    <blank line>
    <original OCR content>

Modes
-----
  --inplace   (default)  Strip stamps from --src in place.
  --out DIR              Write destamped copies to DIR, keeping the same
                         folder hierarchy. Source files are not modified.

Usage
-----
    python destamp_pages.py                         # strip pages_out/ in place
    python destamp_pages.py --src pages_out         # same, explicit
    python destamp_pages.py --out clean_pages       # write copies, don't touch originals
    python destamp_pages.py --src pages_out --dry-run  # preview without writing
"""

import re
import argparse
from pathlib import Path

DEFAULT_SRC = "pages_out"

# Matches the stamp: starts with ---, ends with --- followed by optional blank line
# Anchored to the very start of the file (re.DOTALL so . matches newlines)
STAMP_RE = re.compile(
    r"^---\r?\n(?:[^\n]+\r?\n)*?---\r?\n\r?\n?",
    re.DOTALL,
)


def strip_stamp(text: str) -> tuple[str, bool]:
    """
    Remove the leading YAML frontmatter stamp from text.
    Returns (stripped_text, was_stamped).
    """
    m = STAMP_RE.match(text)
    if m:
        return text[m.end():], True
    return text, False


def process(src_root: Path, out_root: Path | None, dry_run: bool) -> tuple[int, int, int]:
    """
    Walk src_root, strip stamps, write results.
    If out_root is None → write back to the same file (in-place).
    Returns (stripped, skipped, errors).
    """
    stripped = skipped = errors = 0

    md_files = sorted(src_root.rglob("page_*.md"))
    total = len(md_files)

    if total == 0:
        print(f"  No page_*.md files found under {src_root}")
        return 0, 0, 0

    print(f"  Found {total} page_*.md files\n")

    for md_file in md_files:
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print(f"  ✗  READ  {md_file.name}  {exc}")
            errors += 1
            continue

        clean, was_stamped = strip_stamp(text)

        if not was_stamped:
            skipped += 1
            continue

        if out_root is None:
            dst = md_file          # in-place
        else:
            rel = md_file.relative_to(src_root)
            dst = out_root / rel

        if dry_run:
            rel_display = md_file.relative_to(src_root)
            print(f"  [dry-run] would strip  {rel_display}")
            stripped += 1
            continue

        try:
            if out_root is not None:
                dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(clean, encoding="utf-8")
            stripped += 1
        except Exception as exc:
            print(f"  ✗  WRITE {dst}  {exc}")
            errors += 1

    return stripped, skipped, errors


def main():
    ap = argparse.ArgumentParser(description="Remove YAML frontmatter stamps from page .md files")
    ap.add_argument("--src",     default=DEFAULT_SRC,
                    help=f"Source folder to scan (default: {DEFAULT_SRC})")
    ap.add_argument("--out",     default=None,
                    help="Write destamped copies here instead of modifying in place")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be stripped without writing anything")
    args = ap.parse_args()

    src_root = Path(args.src)
    if not src_root.exists():
        raise SystemExit(f"  ✗  Source not found: {src_root}")

    out_root = Path(args.out) if args.out else None

    mode = "dry-run" if args.dry_run else ("in-place" if out_root is None else f"→ {out_root}")
    print(f"\n  Source : {src_root}")
    print(f"  Mode   : {mode}\n")

    stripped, skipped, errors = process(src_root, out_root, args.dry_run)

    print(f"\n  {'─'*50}")
    print(f"  Stripped : {stripped}")
    print(f"  Skipped  : {skipped}  (no stamp found — already clean or not stamped)")
    print(f"  Errors   : {errors}")
    print(f"  {'─'*50}\n")


if __name__ == "__main__":
    main()
