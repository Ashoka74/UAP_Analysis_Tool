"""
analyze_reports.py
──────────────────
Reads _all_reports.json and flags structural and content anomalies.

Usage:
    python analyze_reports.py
    python analyze_reports.py --input D:/divided/extracted/_all_reports.json
    python analyze_reports.py --input _all_reports.json --out anomalies.md
"""

import os
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

_ROOT = os.environ.get("UAP_PIPELINE_ROOT", ".")
DEFAULT_INPUT = os.path.join(_ROOT, "extracted", "_all_reports.json")
DEFAULT_OUT   = os.path.join(_ROOT, "anomaly_report.md")

# ── thresholds ─────────────────────────────────────────────────────────────────
SHORT_TEXT_CHARS    = 200    # raw_text shorter than this is suspicious
VERY_SHORT_CHARS    = 50     # almost certainly truncated / empty
LONG_TEXT_CHARS     = 60000  # single report longer than this may be two merged
ASSESS_DUP_RATIO    = 0.95   # assessment/raw_text similarity threshold (len ratio)
MIN_REPORTS_PER_CHUNK = 0.2  # if reports / chunks < this, extraction may have failed


def load_json(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def page_label_to_ints(label: str) -> list[int]:
    """Extract all page numbers from a label like 'page_0004-page_0006'."""
    return [int(m) for m in re.findall(r"(\d+)", label)]


def check_assessment_duplicate(raw: str, assess: str) -> bool:
    """Return True if assessment is essentially a copy of the end of raw_text."""
    if not assess:
        return False
    assess_stripped = assess.strip()
    raw_stripped    = raw.strip()
    # Check if assess appears verbatim in raw_text (common tail duplication)
    if assess_stripped in raw_stripped:
        return True
    # Check length ratio — if assessment is almost as long as raw_text
    if len(assess_stripped) / max(len(raw_stripped), 1) > ASSESS_DUP_RATIO:
        return True
    return False


def analyze(data: list) -> dict:
    anomalies = defaultdict(list)   # category → [items]
    stats = {
        "total_files":   len(data),
        "total_reports": 0,
        "files_with_errors": 0,
        "files_zero_reports": 0,
    }

    for entry in data:
        src   = entry.get("source_file", "UNKNOWN")
        chunks = entry.get("chunk_count", 1)
        reports = entry.get("reports", [])
        errors  = entry.get("parse_errors", [])

        stats["total_reports"] += len(reports)

        # ── A1: API / parse errors ─────────────────────────────────────────────
        if errors:
            stats["files_with_errors"] += 1
            for e in errors:
                anomalies["A1_api_parse_error"].append({
                    "file":  src,
                    "chunk": e.get("chunk"),
                    "error": e.get("error", ""),
                })

        # ── A2: Zero reports extracted ─────────────────────────────────────────
        if len(reports) == 0:
            stats["files_zero_reports"] += 1
            anomalies["A2_zero_reports"].append({
                "file":   src,
                "chunks": chunks,
                "note":   "No reports extracted — possible all-cover-page doc, "
                          "truly empty, or extraction failure",
            })
            continue  # nothing to inspect inside reports

        # ── A3: Low reports-per-chunk ratio ────────────────────────────────────
        ratio = len(reports) / chunks
        if ratio < MIN_REPORTS_PER_CHUNK and chunks > 2:
            anomalies["A3_low_report_yield"].append({
                "file":    src,
                "chunks":  chunks,
                "reports": len(reports),
                "ratio":   round(ratio, 2),
                "note":    "Very few reports relative to chunk count — "
                           "some chunks may have been silent failures",
            })

        # ── Per-report checks ──────────────────────────────────────────────────
        seen_pages = []

        for i, rep in enumerate(reports):
            raw      = rep.get("raw_text") or ""
            assess   = rep.get("assessment") or ""
            pages    = rep.get("pages")
            rep_id   = f"{src} · report {i+1}"

            # ── A4: Null or missing pages label ───────────────────────────────
            if not pages:
                anomalies["A4_null_pages"].append({
                    "file":   src,
                    "report": i + 1,
                    "note":   "pages field is null — page attribution lost",
                })

            # ── A5: Very short raw_text ────────────────────────────────────────
            if len(raw) < VERY_SHORT_CHARS:
                anomalies["A5_very_short_text"].append({
                    "file":       src,
                    "report":     i + 1,
                    "pages":      pages,
                    "char_count": len(raw),
                    "preview":    raw[:80].replace("\n", " "),
                    "note":       "raw_text < 50 chars — likely truncation or "
                                  "cover-page / separator only",
                })
            elif len(raw) < SHORT_TEXT_CHARS:
                anomalies["A6_short_text"].append({
                    "file":       src,
                    "report":     i + 1,
                    "pages":      pages,
                    "char_count": len(raw),
                    "preview":    raw[:120].replace("\n", " "),
                    "note":       "raw_text < 200 chars — possibly incomplete",
                })

            # ── A7: Suspiciously long raw_text (possible merge of 2+ reports) ─
            if len(raw) > LONG_TEXT_CHARS:
                anomalies["A7_very_long_text"].append({
                    "file":       src,
                    "report":     i + 1,
                    "pages":      pages,
                    "char_count": len(raw),
                    "note":       f"raw_text > {LONG_TEXT_CHARS} chars — "
                                  "may be two reports merged into one",
                })

            # ── A8: Assessment duplicates raw_text ─────────────────────────────
            if assess and check_assessment_duplicate(raw, assess):
                anomalies["A8_assessment_duplication"].append({
                    "file":   src,
                    "report": i + 1,
                    "pages":  pages,
                    "note":   "assessment appears to be a verbatim copy of (part of) raw_text",
                })

            # ── A9: Non-standard pages label format ───────────────────────────
            if pages and pages not in ("all-pages",):
                if not re.match(r"^page_\d{4}(-page_\d{4})?$", pages):
                    anomalies["A9_nonstandard_pages_label"].append({
                        "file":   src,
                        "report": i + 1,
                        "pages":  pages,
                        "note":   "pages label doesn't match expected pattern "
                                  "(page_XXXX or page_XXXX-page_XXXX)",
                    })

            # Accumulate page numbers for overlap check
            if pages:
                seen_pages.append((i + 1, pages, page_label_to_ints(pages)))

        # ── A10: Overlapping page ranges within same document ─────────────────
        for idx_a, (ri_a, lbl_a, nums_a) in enumerate(seen_pages):
            for ri_b, lbl_b, nums_b in seen_pages[idx_a + 1:]:
                overlap = set(nums_a) & set(nums_b)
                if overlap:
                    anomalies["A10_overlapping_pages"].append({
                        "file":      src,
                        "report_a":  ri_a,
                        "pages_a":   lbl_a,
                        "report_b":  ri_b,
                        "pages_b":   lbl_b,
                        "overlap":   sorted(overlap),
                        "note":      "Two reports claim the same page(s) — "
                                     "possible split boundary error",
                    })

    return {"stats": stats, "anomalies": dict(anomalies)}


CATEGORY_LABELS = {
    "A1_api_parse_error":      "A1 — API / parse errors",
    "A2_zero_reports":         "A2 — Zero reports extracted",
    "A3_low_report_yield":     "A3 — Low reports-per-chunk ratio",
    "A4_null_pages":           "A4 — Null pages label",
    "A5_very_short_text":      "A5 — Very short raw_text (< 50 chars)",
    "A6_short_text":           "A6 — Short raw_text (50–200 chars)",
    "A7_very_long_text":       "A7 — Very long raw_text (> 60 000 chars)",
    "A8_assessment_duplication": "A8 — Assessment duplicates raw_text",
    "A9_nonstandard_pages_label": "A9 — Non-standard pages label",
    "A10_overlapping_pages":   "A10 — Overlapping page ranges",
}


def render_markdown(result: dict, input_path: Path) -> str:
    stats     = result["stats"]
    anomalies = result["anomalies"]

    lines = [
        "# _all_reports.json — Anomaly Report",
        "",
        f"**Source:** `{input_path}`  ",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total files | {stats['total_files']} |",
        f"| Total reports extracted | {stats['total_reports']} |",
        f"| Files with API/parse errors | {stats['files_with_errors']} |",
        f"| Files with zero reports | {stats['files_zero_reports']} |",
        f"| Avg reports per file | {stats['total_reports'] / max(stats['total_files'], 1):.1f} |",
        "",
        "## Anomaly Counts",
        "",
        "| Code | Category | Count |",
        "|------|----------|-------|",
    ]

    total_issues = 0
    for cat, label in CATEGORY_LABELS.items():
        n = len(anomalies.get(cat, []))
        total_issues += n
        lines.append(f"| {cat.split('_')[0]} | {label.split(' — ')[1]} | {n} |")

    lines += [
        f"| | **Total issues** | **{total_issues}** |",
        "",
    ]

    if total_issues == 0:
        lines.append("✅ No anomalies detected.")
        return "\n".join(lines)

    lines.append("## Detailed Findings")

    for cat, label in CATEGORY_LABELS.items():
        items = anomalies.get(cat, [])
        if not items:
            continue
        lines += [
            "",
            f"### {label} ({len(items)})",
            "",
        ]
        for item in items:
            file_  = item.get("file", "")
            report = item.get("report")
            pages  = item.get("pages")
            note   = item.get("note", "")
            loc    = f"report {report}" if report else ""
            pg     = f"pages `{pages}`" if pages else ""
            detail = ", ".join(filter(None, [loc, pg]))
            lines.append(f"- **`{file_}`**{' — ' + detail if detail else ''}")
            lines.append(f"  {note}")
            # Extra fields
            for k in ("error", "preview", "char_count", "chunks", "reports",
                      "ratio", "overlap", "report_a", "pages_a", "report_b",
                      "pages_b"):
                if k in item:
                    lines.append(f"  _{k}:_ `{item[k]}`")
            lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Analyze _all_reports.json for anomalies")
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help="Path to _all_reports.json")
    ap.add_argument("--out",   default=DEFAULT_OUT,
                    help="Output Markdown report path")
    ap.add_argument("--json-out", default=None,
                    help="Also write raw anomaly data as JSON")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"✗  File not found: {input_path}")

    print(f"Loading {input_path} …")
    data = load_json(input_path)
    print(f"  {len(data)} file entries loaded")

    result = analyze(data)
    stats  = result["stats"]
    print(f"\n  Total reports  : {stats['total_reports']}")
    print(f"  Zero-report files : {stats['files_zero_reports']}")
    print(f"  Files with errors : {stats['files_with_errors']}")

    total_issues = sum(len(v) for v in result["anomalies"].values())
    print(f"  Total anomaly flags : {total_issues}")

    # Markdown report
    md = render_markdown(result, input_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"\n  ✓  Markdown report → {out_path}")

    # Optional JSON dump
    if args.json_out:
        jout = Path(args.json_out)
        jout.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"  ✓  JSON data      → {jout}")

    # Print top anomalies to console
    print("\n── Top anomalies ────────────────────────────────────────────────")
    for cat, label in CATEGORY_LABELS.items():
        items = result["anomalies"].get(cat, [])
        if items:
            print(f"  {label}: {len(items)}")


if __name__ == "__main__":
    main()
