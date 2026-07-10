"""
reconcile.py
------------
Reads the three independent layers of the UAP document archive and emits
one YAML record file per document into --out-dir.

Layers joined:
  1. CSV row         ← uap-csv.csv
  2. Page files      ← D:/divided/<slug>/page_XXXX/page_XXXX.{md,pdf}
  3. Extracted JSON  ← extracted/*.json  (extract_reports.py output)  ← primary
     NIM report .md  ← reports_out/*.md  (pdf_to_reports.py output)  ← fallback
  4. Blurb regex     ← speed / altitude / heading / count parsed inline

extract_reports.py replaces pdf_to_reports.py as the Tier 3 source.
If an extracted/*.json file exists for a record, it is used in preference
to any matching reports_out/*.md file.

Usage:
    python reconcile.py
    python reconcile.py --csv uap-csv.csv --divided D:/divided --out-dir records
    python reconcile.py --csv uap-csv.csv --divided D:/divided --extracted extracted --out-dir records
    python reconcile.py --csv uap-csv.csv --divided D:/divided --reports reports_out --out-dir records
"""

import re
import os
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime

import yaml          # pip install pyyaml --break-system-packages

# ── tuneable paths ────────────────────────────────────────────────────────────
DEFAULT_CSV        = "uap-csv.csv"
DEFAULT_DIVIDED    = os.environ.get("UAP_PIPELINE_ROOT", "raw")
DEFAULT_REPORTS    = "reports_out"
DEFAULT_EXTRACTED  = "extracted"
DEFAULT_OUT        = "records"

SCHEMA_VERSION   = "1.0"

# ── regex patterns (all blurb-parseable without NIM) ─────────────────────────
RE_SPEED_KNOTS  = re.compile(r"(\d[\d,]+)\s*kn(?:ots?)?", re.I)
RE_SPEED_MPH    = re.compile(r"(\d[\d,]*)\s*mph", re.I)
RE_ALT_FEET     = re.compile(r"([\d,]+)\s*(?:ft|feet)", re.I)
RE_HEADING_DEG  = re.compile(r"(\d{1,3})\s*degrees?", re.I)
RE_ZULU         = re.compile(r"\b(\d{4}Z)\b")
RE_OBJ_COUNT    = re.compile(
    r"(\d+)\s*(?:x\s*)?UAP|"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:separate\s+)?UAP|"
    r"a\s+(?:formation|group)\s+of\s+(\d+)|"
    r"(two|three|four|five)\s+UAP",
    re.I
)
COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
               "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

# Shape vocabulary – order matters (more specific first)
SHAPE_PATTERNS = [
    ("diamond",    re.compile(r"diamond", re.I)),
    ("triangle",   re.compile(r"triangul", re.I)),
    ("sphere",     re.compile(r"sphere|spherical|round|bouncy ball", re.I)),
    ("disc",       re.compile(r"\bdisc\b|\bsaucer\b", re.I)),
    ("balloon",    re.compile(r"balloon", re.I)),
    ("cylinder",   re.compile(r"cylindr", re.I)),
    ("orb",        re.compile(r"\borb\b", re.I)),
    ("elongated",  re.compile(r"elongated|cigar", re.I)),
    ("amorphous",  re.compile(r"ball of (?:white |bright )?light|glare|halo", re.I)),
    ("formation",  re.compile(r"formation|line of dots", re.I)),
]

MANEUVER_PATTERNS = [
    ("right_angle_turns", re.compile(r"90.degree|right.angle", re.I)),
    ("circling",          re.compile(r"circling|orbiting", re.I)),
    ("erratic",           re.compile(r"erratic|irregular", re.I)),
    ("sea_skim",          re.compile(r"sea.skim|surface.*skim", re.I)),
    ("accelerating",      re.compile(r"increased speed|accelerat", re.I)),
    ("straight",          re.compile(r"straight|consistent.*course|same.*altitude", re.I)),
    ("abrupt_turns",      re.compile(r"abrupt.*direction|directional change", re.I)),
]

THERMAL_PATTERNS = [
    ("white_hot",   re.compile(r"white.hot",   re.I)),
    ("black_hot",   re.compile(r"black.hot",   re.I)),
    ("bright_white",re.compile(r"bright white", re.I)),
    ("cold",        re.compile(r"\bcold\b",     re.I)),
]

SUBTYPE_MAP = {
    "MISREP":                       re.compile(r"Mission Report|MISREP", re.I),
    "Range_Fouler_Debrief":         re.compile(r"Range Fouler Debrief", re.I),
    "Range_Fouler_Reporting_Form":  re.compile(r"Range Fouler Reporting Form", re.I),
    "Email_Correspondence":         re.compile(r"email correspondence", re.I),
    "Mission_Briefing":             re.compile(r"mission briefing", re.I),
    "Intelligence_Report":          re.compile(r"intelligence report|air intelligence", re.I),
    "Transcript":                   re.compile(r"transcript", re.I),
    "Crew_Debriefing":              re.compile(r"crew.debriefing|technical.debriefing", re.I),
    "Case_File":                    re.compile(r"case file", re.I),
    "Launch_Summary":               re.compile(r"launch summary", re.I),
    "Cable":                        re.compile(r"\bcable\b", re.I),
    "Policy_Memo":                  re.compile(r"memorandum|memo\b", re.I),
    "Photo_Collection":             re.compile(r"photo|image|picture", re.I),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _first_int(pattern: re.Pattern, text: str) -> int | None:
    m = pattern.search(text)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return int(raw)
    except ValueError:
        return None


def _all_matches(pattern: re.Pattern, text: str) -> list[str]:
    return pattern.findall(text)


def _first_vocab(pairs: list[tuple[str, re.Pattern]], text: str) -> str | None:
    for label, pat in pairs:
        if pat.search(text):
            return label
    return None


def _parse_object_count(blurb: str) -> int | None:
    """Extract explicit UAP count from blurb."""
    m = RE_OBJ_COUNT.search(blurb)
    if not m:
        return None
    for g in m.groups():
        if g:
            word = g.lower()
            if word.isdigit():
                return int(word)
            return COUNT_WORDS.get(word)
    return None


def _parse_subtype(blurb: str) -> str:
    for label, pat in SUBTYPE_MAP.items():
        if pat.search(blurb):
            return label
    return "Other"


def _parse_date(raw: str) -> str | None:
    """Normalise messy CSV dates to ISO 8601 (best effort)."""
    if not raw or raw.strip().upper() in ("N/A", ""):
        return None
    raw = raw.strip()
    # Handle ranges like "4/10/2025-4/11/2025" → take first
    raw = raw.split("-")[0].strip()
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return raw  # fallback: return as-is


def _slug_from_title(title: str) -> str:
    """Derive a lowercase filesystem slug from a CSV title."""
    s = title.lower().strip()
    s = re.sub(r"[\s,]+", "-", s)
    s = re.sub(r"[^\w\-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _index_document_dirs(divided_root: Path) -> list[Path]:
    """
    Return every directory under divided_root that looks like a document folder
    (contains at least one page_XXXX/ subfolder).  Builds once, reused for all rows.
    """
    doc_dirs = []
    for d in divided_root.rglob("*"):
        if not d.is_dir():
            continue
        if any(re.match(r"page_\d+$", sub.name, re.I)
               for sub in d.iterdir() if sub.is_dir()):
            doc_dirs.append(d)
    return doc_dirs


def _find_document_dir(divided_root: Path, title: str,
                       doc_index: list[Path] | None = None) -> Path | None:
    """
    Match a CSV title to a document folder anywhere under divided_root.
    Pass a pre-built doc_index (from _index_document_dirs) to avoid
    re-walking the tree for every row.
    """
    slug = _slug_from_title(title)

    if doc_index is None:
        doc_index = _index_document_dirs(divided_root)

    # 1. Exact slug match
    for d in doc_index:
        if d.name.lower() == slug.lower():
            return d

    # 2. Prefix match (handles abbreviated folder names)
    prefix = slug[:20].lower()
    for d in doc_index:
        if d.name.lower().startswith(prefix):
            return d

    # 3. Word-intersection fallback
    words = set(slug.split("-")[:5])
    for d in doc_index:
        dwords = set(d.name.lower().split("-")[:5])
        if len(words & dwords) >= min(3, len(words)):
            return d

    return None


def _scan_pages(doc_dir: Path) -> list[dict]:
    """Return sorted list of page dicts from the document folder."""
    pages = []
    for sub in sorted(doc_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = re.match(r"page_(\d+)$", sub.name, re.I)
        if not m:
            continue
        idx = int(m.group(1))
        md_file  = sub / f"{sub.name}.md"
        pdf_file = sub / f"{sub.name}.pdf"

        md_text = ""
        if md_file.exists():
            md_text = md_file.read_text(encoding="utf-8", errors="replace")

        has_images = bool(re.search(r"!\[", md_text))
        image_assets = re.findall(r"!\[.*?\]\((.*?)\)", md_text)

        pages.append({
            "index": idx,
            "subdir": sub.name,
            "md":     str(md_file.relative_to(doc_dir)) if md_file.exists() else None,
            "pdf":    str(pdf_file.relative_to(doc_dir)) if pdf_file.exists() else None,
            "has_images": has_images,
            "image_assets": image_assets,
        })
    return pages


def _find_report_md(reports_root: Path, record_id: str) -> Path | None:
    """Locate the best matching .md file in reports_out for this record."""
    if not reports_root.exists():
        return None
    slug_upper = record_id.upper().replace("-", "_")
    for md in reports_root.glob("*.md"):
        name = md.stem.upper().replace("-", "_")
        if name == slug_upper or name.startswith(slug_upper[:20]):
            return md
    return None


def _parse_nim_md(md_path: Path) -> dict:
    """Extract the metadata table from a pdf_to_reports.py .md file."""
    text = md_path.read_text(encoding="utf-8", errors="replace")
    result = {k: None for k in
              ["date", "location", "agency", "classification",
               "object_description", "witnesses", "redacted_sections"]}
    result["redacted_sections"] = []

    for line in text.splitlines():
        m = re.match(r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|", line)
        if not m:
            continue
        key, val = m.group(1).strip().lower(), m.group(2).strip()
        if val in ("—", "N/A", ""):
            val = None
        if "date"        in key: result["date"]               = val
        elif "location"  in key: result["location"]           = val
        elif "agency"    in key: result["agency"]             = val
        elif "classif"   in key: result["classification"]     = val
        elif "object"    in key: result["object_description"] = val
        elif "witness"   in key: result["witnesses"]          = val
        elif "redacted"  in key and val:
            result["redacted_sections"] = [s.strip() for s in val.split(";")]

    return result


def _find_extracted_json(extracted_root: Path, record_id: str) -> Path | None:
    """
    Locate the best matching .json file in extracted/ for this record.
    Mirrors _find_report_md but for extract_reports.py JSON output.
    Skips the combined _all_reports.json file.
    """
    if not extracted_root.exists():
        return None
    slug_upper = record_id.upper().replace("-", "_")
    for jf in sorted(extracted_root.glob("*.json")):
        if jf.name.startswith("_"):          # skip _all_reports.json
            continue
        name = jf.stem.upper().replace("-", "_")
        if name == slug_upper or name.startswith(slug_upper[:20]):
            return jf
    return None


# Date patterns used by _parse_extracted_json
_RE_DATE_ISO    = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_RE_DATE_US     = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b")
_RE_DATE_LONG   = re.compile(
    r"\b((?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},?\s+\d{4})\b", re.I
)
_RE_CLASSIF     = re.compile(
    r"\b(UNCLASSIFIED|SECRET|CONFIDENTIAL|TOP\s+SECRET|FOUO|"
    r"FOR\s+OFFICIAL\s+USE\s+ONLY)\b", re.I
)
_RE_REDACTED    = re.compile(r"\[REDACTED\]|\(b\)\(\d\)", re.I)


def _parse_extracted_json(json_path: Path) -> list[dict]:
    """
    Read an extract_reports.py JSON file and return a list of nim_extracted
    dicts — one per report entry found in the file.

    Strategy:
      1. Try markdown-table parsing first (handles docs where verbatim text
         preserved a table layout from pdf_to_reports.py output).
      2. Fall back to regex patterns for date, classification, and redactions.

    Each dict has the standard nim_extracted keys plus a bonus 'pages' key
    (the page-range label from the JSON).  The caller pops 'pages' before
    storing into the YAML record.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return []                            # unreadable JSON — skip silently

    results = []
    for rep in data.get("reports", []):
        raw_text   = rep.get("raw_text") or ""
        assessment = rep.get("assessment") or ""
        combined   = raw_text + "\n" + assessment

        nim = {k: None for k in
               ["date", "location", "agency", "classification",
                "object_description", "witnesses", "redacted_sections"]}
        nim["redacted_sections"] = []

        # ── pass 1: markdown table rows (verbatim tables from NIM-style docs) ──
        for line in combined.splitlines():
            m = re.match(r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|", line)
            if not m:
                continue
            key, val = m.group(1).strip().lower(), m.group(2).strip()
            if val in ("—", "N/A", ""):
                val = None
            if "date"       in key: nim["date"]               = val
            elif "location" in key: nim["location"]           = val
            elif "agency"   in key: nim["agency"]             = val
            elif "classif"  in key: nim["classification"]     = val
            elif "object"   in key: nim["object_description"] = val
            elif "witness"  in key: nim["witnesses"]          = val
            elif "redacted" in key and val:
                nim["redacted_sections"] = [s.strip() for s in val.split(";")]

        # ── pass 2: plain-text regex fallbacks ────────────────────────────────
        if nim["date"] is None:
            for pat in (_RE_DATE_ISO, _RE_DATE_US, _RE_DATE_LONG):
                dm = pat.search(combined)
                if dm:
                    nim["date"] = dm.group(1)
                    break

        if nim["classification"] is None:
            cm = _RE_CLASSIF.search(combined)
            if cm:
                nim["classification"] = cm.group(1).upper()

        n_redacted = len(_RE_REDACTED.findall(combined))
        if n_redacted > 0 and not nim["redacted_sections"]:
            nim["redacted_sections"] = [f"{n_redacted} redaction marker(s) detected"]

        nim["pages"] = rep.get("pages")      # bonus field — popped by caller
        results.append(nim)

    return results


# ── per-row record builder ────────────────────────────────────────────────────

def build_record(row: dict, divided_root: Path, reports_root: Path,
                 doc_index: list[Path] | None = None,
                 extracted_root: Path | None = None) -> dict:
    title  = row.get("Title", "").replace("\n", " ").strip()
    blurb  = row.get("Description Blurb", "").replace("\n", " ").strip()
    agency = row.get("Agency", "").strip()
    loc    = row.get("Incident Location", "").strip()
    if loc.upper() == "N/A":
        loc = None

    record_id = _slug_from_title(title) or "unknown"

    # ── Tier 1: CSV ──────────────────────────────────────────────────────────
    csv_block = {
        "title":                title,
        "agency":               agency,
        "release_date":         _parse_date(row.get("Release Date", "")),
        "redacted":             row.get("Redaction", "").strip().upper() == "TRUE",
        "document_type":        row.get("Type", "PDF").strip(),
        "report_subtype":       _parse_subtype(blurb),
        "incident_date":        _parse_date(row.get("Incident Date", "")),
        "incident_date_precision": "day",   # refined by enrich.py
        "location_csv":         loc,
        "dvids_video_id":       row.get("DVIDS Video ID", "").strip() or None,
        "video_pairing":        row.get("Video Pairing", "").strip() or None,
        "pdf_pairing":          row.get("PDF Pairing", "").strip() or None,
        "pdf_url":              row.get("PDF | Image Link", "").strip() or None,
        "thumbnail_url":        row.get("Modal Image", "").strip() or None,
        "description_blurb":    blurb,
    }

    # ── Tier 2: Page files ───────────────────────────────────────────────────
    doc_dir = _find_document_dir(divided_root, title, doc_index)
    if doc_dir:
        pages = _scan_pages(doc_dir)
    else:
        pages = []

    files_block = {
        "document_dir": str(doc_dir) if doc_dir else None,
        "page_count":   len(pages),
        "pages":        pages,
    }

    # ── Tier 3: Extracted JSON (primary) or NIM report .md (fallback) ───────
    json_path = (
        _find_extracted_json(extracted_root, record_id)
        if extracted_root else None
    )

    if json_path:
        # ── primary: extract_reports.py JSON ──────────────────────────────────
        extracted_reports = _parse_extracted_json(json_path)

        if extracted_reports:
            reports_block = []
            for idx, nim_data in enumerate(extracted_reports, 1):
                page_label = nim_data.pop("pages", None)  # remove bonus field
                reports_block.append({
                    "report_id":  f"{record_id}_r{idx:02d}" if len(extracted_reports) > 1
                                  else record_id,
                    "json_path":  str(json_path),
                    "page_range": page_label or ([1, len(pages)] if pages else [1, None]),
                    "nim_extracted": nim_data,
                })
        else:
            # JSON found but empty / parse error
            reports_block = [{
                "report_id":  record_id,
                "json_path":  str(json_path),
                "page_range": [1, len(pages)] if pages else [1, None],
                "nim_extracted": {k: None for k in
                                  ["date", "location", "agency", "classification",
                                   "object_description", "witnesses", "redacted_sections"]},
            }]
    else:
        # ── fallback: pdf_to_reports.py .md ───────────────────────────────────
        report_md = _find_report_md(reports_root, record_id)
        nim_data  = _parse_nim_md(report_md) if report_md else {
            k: None for k in ["date", "location", "agency",
                               "classification", "object_description",
                               "witnesses", "redacted_sections"]
        }
        if "redacted_sections" not in nim_data:
            nim_data["redacted_sections"] = []

        reports_block = [{
            "report_id":     record_id,
            "md_path":       str(report_md) if report_md else None,
            "page_range":    [1, len(pages)] if pages else [1, None],
            "nim_extracted": nim_data,
        }]

    # ── Tier 4: Observation (regex from blurb) ───────────────────────────────
    observation_block = {
        "location_precise": loc,
        "morphology": {
            "shape":            _first_vocab(SHAPE_PATTERNS, blurb),
            "color":            None,
            "thermal_appearance": _first_vocab(THERMAL_PATTERNS, blurb),
            "material": "metallic" if re.search(r"metallic", blurb, re.I) else None,
            "size_estimate":    None,
        },
        "kinematics": {
            "object_count":     _parse_object_count(blurb),
            "speed_knots":      _first_int(RE_SPEED_KNOTS, blurb),
            "speed_mph":        _first_int(RE_SPEED_MPH,   blurb),
            "altitude_ft":      _first_int(RE_ALT_FEET,    blurb),
            "heading_degrees":  _first_int(RE_HEADING_DEG, blurb),
            "direction_cardinal": None,
            "duration_seconds": None,
            "maneuver_type":    _first_vocab(MANEUVER_PATTERNS, blurb),
            "flight_profile":   None,
        },
        "detection": {
            "sensor_types":         [],
            "platform":             "P8A" if re.search(r"P-8A", blurb) else None,
            "tracking_outcome":     None,
            "environmental_factors": (
                ["cloud_cover"] if re.search(r"cloud", blurb, re.I) else []
            ),
        },
        "assessment": {
            "observer_label":    None,
            "threat_assessment": "benign" if re.search(r"\bbenign\b", blurb, re.I) else None,
            "pursuit_status":    (
                "not_pursued" if re.search(r"did not pursue", blurb, re.I) else None
            ),
            "zulu_timestamps":   list(set(RE_ZULU.findall(blurb))),
        },
    }

    # ── Tier 5: Relations ────────────────────────────────────────────────────
    # Detect parent case / sibling logic for HS1 records
    case_match = re.search(r"(\d{2}-HQ-\d+|\d{2}-[A-Z]+-\d+)", title)
    relations_block = {
        "part_of_case":      case_match.group(1) if case_match else None,
        "parent_document":   None,
        "sibling_documents": [],
        "paired_video":      csv_block["video_pairing"],
        "paired_pdf":        csv_block["pdf_pairing"],
        "related_incidents": [],
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "record_id":      record_id,
        "csv":            csv_block,
        "files":          files_block,
        "reports":        reports_block,
        "observation":    observation_block,
        "relations":      relations_block,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Reconcile CSV + pages + reports → YAML records")
    ap.add_argument("--csv",       default=DEFAULT_CSV,       help="Path to uap-csv.csv")
    ap.add_argument("--divided",   default=DEFAULT_DIVIDED,   help="Root of divided/ folder")
    ap.add_argument("--extracted", default=DEFAULT_EXTRACTED,
                    help="extracted/ directory with JSON from extract_reports.py (primary Tier 3)")
    ap.add_argument("--reports",   default=DEFAULT_REPORTS,
                    help="reports_out/ directory with .md from pdf_to_reports.py (fallback Tier 3)")
    ap.add_argument("--out-dir",   default=DEFAULT_OUT,       help="Output directory for YAML files")
    ap.add_argument("--limit",     type=int, default=0,       help="Process only first N rows (0 = all)")
    args = ap.parse_args()

    divided_root   = Path(args.divided)
    extracted_root = Path(args.extracted)
    reports_root   = Path(args.reports)
    out_dir        = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Informative summary of which Tier 3 sources are present
    ext_count = len([f for f in extracted_root.glob("*.json")
                     if not f.name.startswith("_")]) if extracted_root.exists() else 0
    rpt_count = len(list(reports_root.glob("*.md"))) if reports_root.exists() else 0
    print(f"  Tier 3 — extracted JSON : {ext_count} files  ({extracted_root})")
    print(f"  Tier 3 — reports .md    : {rpt_count} files  ({reports_root})  [fallback]")

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\n  {len(rows)} CSV rows  →  {out_dir}/")
    print(f"  Indexing document folders under {divided_root} …")
    doc_index = _index_document_dirs(divided_root)
    print(f"  Found {len(doc_index)} document folders\n")

    written, skipped = 0, 0
    for i, row in enumerate(rows):
        if args.limit and i >= args.limit:
            break
        title = row.get("Title", "").replace("\n", " ").strip()
        if not title:
            skipped += 1
            continue

        record = build_record(row, divided_root, reports_root, doc_index,
                               extracted_root=extracted_root)
        slug   = record["record_id"]
        out_path = out_dir / f"{slug}.yaml"

        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(record, f, allow_unicode=True, sort_keys=False,
                      default_flow_style=False, width=100)

        written += 1
        if written <= 5 or written % 20 == 0:
            print(f"  ✓  {out_path.name}")

    print(f"\n✅  {written} records written, {skipped} skipped  →  {out_dir}/")


if __name__ == "__main__":
    main()
