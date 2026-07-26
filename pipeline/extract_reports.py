"""
extract_reports.py
──────────────────
For every concat/*.md whose filename does NOT start with a digit, sends the
full text to Gemini and asks it to extract each individual UAP/UFO sighting
report as structured JSON.

Output per file:
  {
    "source_file": "dow-uap-d33-mission-report-greece-october-2023.md",
    "document_id": "dow-uap-d33-mission-report-greece-october-2023",
    "agency":      "DOD",
    "collection":  "mission-reports",
    "region":      "greece",
    "manifest":    { ... war.gov CSV row, if --csv given ... },
    "reports": [
      {
        "document_id": "dow-uap-d33-mission-report-greece-october-2023",
        "agency":      "DOD",
        "collection":  "mission-reports",
        "pages":       "page_0004-page_0006",
        "raw_text":    "<verbatim text block for this report>",
        "assessment":  "<verbatim assessment / analysis block, or null>"
      },
      ...
    ]
  }

No summarisation — raw text only. Page numbers, agency and collection are derived
deterministically from the document path and the '## page_XXXX' markers — NOT from
the LLM, which only identifies report boundaries (raw_text + assessment).

Chunking
--------
Large documents are split into page-blocks before sending to avoid the 65,536
output-token ceiling.  Thinking is disabled (pure extraction — no reasoning
needed) to maximise the usable output budget.  Use --chunk-pages to tune block
size (default 40 pages).  Chunk results are merged into a single JSON file.

Parallelism
-----------
--workers N  runs N files concurrently (default 1 = sequential).
Gemini API handles concurrent requests; stay within your rate-limit quota.

Usage
-----
    set GEMINI_API_KEY=your_key_here          (Windows CMD)
    export GEMINI_API_KEY=your_key_here       (bash)

    python extract_reports.py
    python extract_reports.py --concat D:/divided/concat --out D:/divided/extracted
    python extract_reports.py --file dow-uap-d33-mission-report-greece-october-2023.md
    python extract_reports.py --workers 4                    # parallel files
    python extract_reports.py --chunk-pages 30               # smaller chunks
    python extract_reports.py --no-skip                      # re-process existing
"""

import os
import re
import csv
import json
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions

_ROOT = os.environ.get("UAP_PIPELINE_ROOT", ".")
DEFAULT_CONCAT       = os.path.join(_ROOT, "concat")
DEFAULT_OUT          = os.path.join(_ROOT, "extracted")
DEFAULT_CHUNK_PAGES  = 40     # pages per API call; tune down if still truncating
DEFAULT_WORKERS      = 15     # concurrent files; increase for throughput
MAX_RETRIES          = 6      # retries on 429 / 503 before giving up
RETRY_BASE_SECS      = 5      # first wait; doubles each attempt (5, 10, 20, 40, 80, 160)
MODEL                = "gemini-3.1-pro-preview"

# Thread-safe print lock
_print_lock = threading.Lock()

SYSTEM_PROMPT = """\
You are a document analyst processing declassified U.S. government UAP/UFO records.
Your job is to identify every distinct UAP/UFO sighting report within the document.
No summaries, no paraphrasing — raw verbatim text only.

Rules:
1. Each "report" is one coherent UAP/UFO sighting description (may span one or more pages).
2. "raw_text" must contain the verbatim text of that sighting report exactly as it appears
   in the source, including headers, field labels, and any partially redacted content.
   Do NOT shorten, paraphrase, or omit anything. If the report is a presented as the row of a table, include the table colnames in each report.
3. "assessment" must contain the verbatim text of any analyst assessment, conclusion,
   classification, or recommendation block associated with that report.
   Use null if none is present.
4. Return the reports in the order they appear in the document.
5. If the whole document is a single report, return one entry.
6. If this is a chunk of a larger document, extract only what appears in THIS chunk.

Do NOT output page numbers. Page ranges are assigned deterministically afterwards
from the document's '## page_XXXX' markers — that is not your job.
"""

# JSON schema for structured output — enforced at API level, no manual parsing needed.
# Google GenAI uses OpenAPI 3.0 style: nullable fields use {"type": "string", "nullable": True}
# NOT JSON Schema draft-07 union types like {"type": ["string", "null"]}.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reports": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "raw_text":   {"type": "string"},
                    "assessment": {"type": "string", "nullable": True},
                },
                "required": ["raw_text", "assessment"],
            },
        },
    },
    "required": ["reports"],
}

# ── page splitting ─────────────────────────────────────────────────────────────

PAGE_HEADER_RE = re.compile(r"^## (page_\d+)\s*$", re.MULTILINE)


def split_into_chunks(text: str, chunk_pages: int) -> list[tuple[str, str]]:
    """
    Split a concat .md (pages separated by '## page_XXXX' headers) into
    blocks of at most chunk_pages pages.

    Returns list of (label, chunk_text) where label is e.g. "page_0001-page_0040".
    If the document has no page headers (single-page or unstamped), returns one chunk.
    """
    # Find all page header positions
    headers = [(m.group(1), m.start()) for m in PAGE_HEADER_RE.finditer(text)]

    if not headers:
        return [("all-pages", text)]

    # Build page boundaries
    chunks = []
    for i in range(0, len(headers), chunk_pages):
        block_headers = headers[i : i + chunk_pages]
        start_pos     = block_headers[0][1]
        end_pos       = headers[i + chunk_pages][1] if (i + chunk_pages) < len(headers) else len(text)
        first_page    = block_headers[0][0]
        last_page     = block_headers[-1][0]
        label         = first_page if first_page == last_page else f"{first_page}-{last_page}"
        chunks.append((label, text[start_pos:end_pos]))

    return chunks


# ── deterministic metadata — page numbers, agency, collection ──────────────────
# These come from the document path and the '## page_XXXX' markers, NOT the LLM.
# The LLM only identifies report boundaries (raw_text + assessment).

_AGENCY_PREFIXES = [
    ("dow-uap-d",        ("DOD", "mission-reports")),
    ("dow-uap-",         ("DOD", "mission-reports")),
    ("dod-range-fouler", ("DOD", "range-fouler-debriefs")),
    ("dod-email",        ("DOD", "email-correspondence")),
    ("dod-",             ("DOD", None)),
    ("pr-",              ("DOD", "mission-reports")),
    ("fbi-",             ("FBI", "photo-collections")),
    ("nasa-transcript",  ("NASA", "transcripts")),
    ("nasa-crew",        ("NASA", "crew-debriefings")),
    ("nasa-",            ("NASA", None)),
    ("dos-",             ("DOS", "cables")),
    ("65_hs1-834228961", ("NARA-CIA", "hs1-834228961")),
    ("65_hs1-101634279", ("NARA-CIA", "hs1-101634279")),
    ("18_",              ("NARA-CIA", "series-18")),
    ("38_",              ("NARA-CIA", "series-38")),
    ("59_",              ("NARA-CIA", "series-59")),
    ("255_",             ("NARA-CIA", "series-255")),
    ("331_",             ("NARA-CIA", "series-331")),
    ("341_",             ("NARA-CIA", "series-341")),
    ("342_",             ("NARA-CIA", "series-342")),
    ("series-",          ("NARA-CIA", None)),
]


def _first_frontmatter(text: str) -> str:
    """Body of the first YAML frontmatter block (stamp_pages stamp), or ''."""
    m = re.search(r"(?ms)^---\r?\n(.*?)\r?\n---\r?\n", text)
    return m.group(1) if m else ""


def _fm_value(block: str, key: str):
    """Read a single `key: value` line from a frontmatter block."""
    m = re.search(rf"(?m)^{re.escape(key)}:\s*(.*)$", block)
    if not m:
        return None
    v = m.group(1).strip().strip('"').strip("'")
    return v if v and v.lower() != "null" else None


def resolve_doc_meta(text: str, document_id: str):
    """(agency, collection, region) — from the stamped frontmatter (path-derived)
    when present, else from the document slug prefix. Never from the LLM."""
    fm = _first_frontmatter(text)
    agency  = _fm_value(fm, "agency")
    subtype = _fm_value(fm, "subtype")
    region  = _fm_value(fm, "region")
    if agency:
        return agency, subtype, region
    low = document_id.lower()
    for prefix, (ag, coll) in _AGENCY_PREFIXES:
        if low.startswith(prefix):
            return ag, coll, region
    return None, None, region


def _page_spans(chunk_text: str):
    """[(page_label, start, end)] character spans for each '## page_XXXX' section."""
    hdrs = [(m.group(1), m.start()) for m in PAGE_HEADER_RE.finditer(chunk_text)]
    spans = []
    for i, (label, start) in enumerate(hdrs):
        end = hdrs[i + 1][1] if i + 1 < len(hdrs) else len(chunk_text)
        spans.append((label, start, end))
    return spans


def attach_pages(reports: list, chunk_text: str, chunk_label: str) -> list:
    """Deterministically set each report's 'pages' from the document's
    '## page_XXXX' markers.

    A multi-page report's raw_text has the page markers/separators stripped, so
    the full block won't match the chunk verbatim. Instead we anchor on the
    report's first and last non-empty *lines* — a single line never spans a
    page break — and map that [start, end] span onto the page markers it
    covers. Reports are assumed to be in document order (the prompt requires it).
    """
    spans = _page_spans(chunk_text)
    full_range = chunk_label
    if spans:
        full_range = (spans[0][0] if spans[0][0] == spans[-1][0]
                      else f"{spans[0][0]}-{spans[-1][0]}")
    cursor = 0
    for rep in reports:
        lines = [ln.strip() for ln in (rep.get("raw_text") or "").splitlines()
                 if ln.strip()]
        if not spans or not lines:
            rep["pages"] = None if chunk_label == "all-pages" else full_range
            continue
        first, last = lines[0][:120], lines[-1][:120]
        start_idx = chunk_text.find(first, cursor)
        if start_idx < 0:
            start_idx = chunk_text.find(first, 0)
        if start_idx < 0:
            rep["pages"] = full_range          # could not locate — whole chunk
            continue
        end_hit = chunk_text.find(last, start_idx)
        end_idx = (end_hit + len(last)) if end_hit >= 0 else start_idx + 1
        covered = [lbl for (lbl, s, e) in spans if s < end_idx and e > start_idx]
        rep["pages"] = (
            covered[0] if len(covered) == 1
            else f"{covered[0]}-{covered[-1]}" if covered
            else full_range
        )
        cursor = start_idx + 1
    return reports


def _slug_from_title(title: str) -> str:
    """Normalise a CSV Title to a filesystem slug — matches reconcile.py."""
    s = (title or "").lower().strip()
    s = re.sub(r"[\s,]+", "-", s)
    s = re.sub(r"[^\w\-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def load_manifest(csv_path: Path) -> dict:
    """Index the war.gov release CSV by document slug → row dict."""
    index = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            slug = _slug_from_title((row.get("Title") or "").strip())
            if slug:
                index.setdefault(slug, row)
    return index


def manifest_block(manifest_index, document_id: str):
    """Deterministic slug-join of one document to its war.gov manifest row."""
    if manifest_index is None:
        return None
    row = manifest_index.get(_slug_from_title(document_id))
    if not row:
        return {"matched": False}
    g = lambda k: (row.get(k) or "").strip()
    return {
        "matched":           True,
        "csv_title":         " ".join(g("Title").split()),
        "release_date":      g("Release Date"),
        "redacted":          g("Redaction").upper() == "TRUE",
        "csv_agency":        g("Agency"),
        "incident_date":     g("Incident Date"),
        "incident_location": g("Incident Location"),
        "pdf_link":          g("PDF | Image Link"),
        "dvids_video_id":    g("DVIDS Video ID"),
        "type":              g("Type"),
    }


# ── API call ───────────────────────────────────────────────────────────────────

def _call_gemini(client: genai.Client, model: str,
                 filename: str, chunk_label: str, chunk_text: str,
                 verbose: bool = True) -> dict:
    """
    Send one chunk to Gemini using structured JSON output (response_mime_type).
    The API guarantees valid JSON matching RESPONSE_SCHEMA — no manual parsing needed.
    Thinking is disabled (budget=0) — verbatim extraction needs no reasoning.

    Returns dict with key 'reports' (list), guaranteed by the schema.
    On API error returns {'reports': [], 'api_error': str}.
    """
    prompt = (
        f"Document filename: {filename}\n"
        f"Pages in this chunk: {chunk_label}\n\n"
        f"Document content:\n\n{chunk_text}"
    )

    # Blocking call — response is returned in full once complete.
    # Retries with exponential backoff on 429 (rate limit) and 503 (overload).
    full_response = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_json_schema=RESPONSE_SCHEMA,
                    thinking_config=types.ThinkingConfig(thinking_budget=-1),
                ),
            )
            full_response = response.text or ""

            # Success — break out of retry loop
            break

        except google_exceptions.ResourceExhausted as exc:
            # 429 — rate limited; back off and retry
            wait = RETRY_BASE_SECS * (2 ** (attempt - 1))
            with _print_lock:
                print(f"\n  ⏳ 429 rate limit ({filename} chunk {chunk_label}) "
                      f"attempt {attempt}/{MAX_RETRIES} — waiting {wait}s …")
            if attempt == MAX_RETRIES:
                return {"reports": [], "api_error": f"429 after {MAX_RETRIES} retries: {exc}"}
            time.sleep(wait)

        except google_exceptions.ServiceUnavailable as exc:
            # 503 — transient overload; same backoff
            wait = RETRY_BASE_SECS * (2 ** (attempt - 1))
            with _print_lock:
                print(f"\n  ⏳ 503 unavailable ({filename} chunk {chunk_label}) "
                      f"attempt {attempt}/{MAX_RETRIES} — waiting {wait}s …")
            if attempt == MAX_RETRIES:
                return {"reports": [], "api_error": f"503 after {MAX_RETRIES} retries: {exc}"}
            time.sleep(wait)

        except Exception as exc:
            # Non-retryable error
            with _print_lock:
                print(f"\n  ✗  API error ({filename} chunk {chunk_label}): {exc}")
            return {"reports": [], "api_error": str(exc)}

    if verbose:
        with _print_lock:
            print(f"    {filename} [{chunk_label}] ✓  {len(full_response)} chars")

    # Guaranteed valid JSON from the API; defensive parse handles edge cases
    try:
        return json.loads(full_response)
    except json.JSONDecodeError as e:
        with _print_lock:
            print(f"\n  ⚠  Unexpected JSON parse error ({filename} chunk {chunk_label}): {e}")
        return {"reports": [], "parse_error": str(e), "raw_response": full_response}


# ── per-file processing ────────────────────────────────────────────────────────

def files_to_process(concat_dir: Path, single: str | None,
                     out_dir: Path, skip_existing: bool) -> list[Path]:
    if single:
        p = concat_dir / single
        return [p] if p.exists() else []
    all_md = sorted(concat_dir.glob("*.md"))
    # Exclude only the combined output file and other underscore-prefixed files.
    # Do NOT exclude digit-prefixed files (65_hs1-..., 18_..., 255_... are valid docs).
    candidates = [f for f in all_md if not f.name.startswith("_")]
    if skip_existing:
        candidates = [f for f in candidates
                      if not (out_dir / (f.stem + ".json")).exists()]
    return candidates


def extract_file(client: genai.Client, model: str,
                 md_path: Path, out_dir: Path, chunk_pages: int,
                 file_index: int, file_total: int,
                 manifest_index: dict | None = None,
                 multi_worker: bool = False) -> dict:
    """
    Process one concat .md file: chunk it, call Gemini per chunk, merge reports.
    Writes the per-file JSON immediately and returns the result dict.

    multi_worker=True suppresses per-chunk noise and prints only one summary line
    per file — keeps output readable when many workers run concurrently.
    """
    t0 = time.time()

    # Announce start only in single-worker or low-worker mode
    if not multi_worker:
        with _print_lock:
            print(f"\n{'─'*70}")
            print(f"  [{file_index:>3}/{file_total}]  {md_path.name}")

    text   = md_path.read_text(encoding="utf-8", errors="replace")
    chunks = split_into_chunks(text, chunk_pages)
    document_id = md_path.stem
    agency, collection, region = resolve_doc_meta(text, document_id)

    if not multi_worker:
        with _print_lock:
            print(f"  Chunks: {len(chunks)}  (chunk_pages={chunk_pages})")
            print(f"{'─'*70}\n")

    all_reports  = []
    parse_errors = []

    for ci, (label, chunk_text) in enumerate(chunks, 1):
        if not multi_worker and len(chunks) > 1:
            with _print_lock:
                print(f"\n  ── chunk {ci}/{len(chunks)}  [{label}] ──\n")

        # Pass verbose=False when running multi-worker to suppress per-call noise
        inner = _call_gemini(client, model, md_path.name, label, chunk_text,
                             verbose=not multi_worker)
        # Page numbers + agency/collection are derived from the path, not the LLM.
        # document_id / agency / collection are NOT repeated here — they already
        # live in the document-level envelope.  Repeating them causes a pandas
        # column-overlap error when json_normalize flattens records + meta together.
        for _rep in attach_pages(inner.get("reports", []), chunk_text, label):
            all_reports.append({
                "pages":      _rep.get("pages"),
                "raw_text":   _rep.get("raw_text", ""),
                "assessment": _rep.get("assessment"),
            })
        if "parse_error" in inner:
            parse_errors.append({"chunk": label, "error": inner["parse_error"]})
        if "api_error" in inner:
            parse_errors.append({"chunk": label, "error": inner["api_error"]})

    result = {
        "source_file":  md_path.name,
        "document_id":  document_id,
        "agency":       agency,
        "collection":   collection,
        "region":       region,
        "chunk_count":  len(chunks),
        "manifest":     manifest_block(manifest_index, document_id),
        "reports":      all_reports,
    }
    if parse_errors:
        result["parse_errors"] = parse_errors

    out_file = out_dir / (md_path.stem + ".json")
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    with _print_lock:
        n = len(all_reports)
        errs = f"  ⚠ {len(parse_errors)} chunk error(s)" if parse_errors else ""
        print(f"\n  ✓  {n} report(s) → {out_file.name}{errs}")

    return result


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract UAP reports from concat .md files via Gemini"
    )
    ap.add_argument("--concat",       default=DEFAULT_CONCAT,
                    help="Path to concat/ folder")
    ap.add_argument("--out",          default=DEFAULT_OUT,
                    help="Output folder for JSON files")
    ap.add_argument("--file",         default=None,
                    help="Process a single file by name")
    ap.add_argument("--model",        default=MODEL,
                    help=f"Gemini model (default: {MODEL})")
    ap.add_argument("--chunk-pages",  type=int, default=DEFAULT_CHUNK_PAGES,
                    help=f"Max pages per API call (default: {DEFAULT_CHUNK_PAGES}). "
                         f"Reduce to 20-25 if still truncating.")
    ap.add_argument("--workers",      type=int, default=DEFAULT_WORKERS,
                    help=f"Concurrent files (default: {DEFAULT_WORKERS}). "
                         f"Increase for throughput; mind rate limits.")
    ap.add_argument("--no-skip",      action="store_true",
                    help="Re-process files that already have a JSON output")
    ap.add_argument("--csv",          default=None,
                    help="war.gov manifest CSV (e.g. uap-csv.csv). When given, each "
                         "document's release metadata (agency, release date, "
                         "redaction, PDF link, incident date/location) is joined in "
                         "deterministically by document slug.")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("  ✗  GEMINI_API_KEY environment variable not set.")

    concat_dir = Path(args.concat)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_index = None
    if args.csv:
        try:
            manifest_index = load_manifest(Path(args.csv))
        except Exception as exc:
            raise SystemExit(f"  ✗  Could not read manifest CSV {args.csv}: {exc}")

    files = files_to_process(concat_dir, args.file, out_dir,
                              skip_existing=not args.no_skip)
    if not files:
        already = len(list(out_dir.glob("*.json")))
        print(f"  No new files to process.  "
              f"({already} already extracted — use --no-skip to reprocess)")
        return

    print(f"\n  Files to process : {len(files)}")
    print(f"  Model            : {args.model}")
    print(f"  Chunk size       : {args.chunk_pages} pages / call")
    print(f"  Workers          : {args.workers}")
    print(f"  Thinking         : none (budget=-1)")
    if manifest_index is not None:
        print(f"  Manifest CSV     : {len(manifest_index)} documents indexed")
    print(f"  Output           : {out_dir}\n")

    client      = genai.Client(api_key=api_key)
    total       = len(files)
    all_results = [None] * total

    if args.workers <= 1:
        for i, md_path in enumerate(files, 1):
            all_results[i - 1] = extract_file(
                client, args.model, md_path, out_dir, args.chunk_pages, i, total,
                manifest_index=manifest_index,
            )
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, md_path in enumerate(files, 1):
                fut = pool.submit(
                    extract_file,
                    client, args.model, md_path, out_dir, args.chunk_pages, i, total,
                    manifest_index,
                )
                futures[fut] = i - 1
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    all_results[idx] = fut.result()
                except Exception as exc:
                    with _print_lock:
                        print(f"\n  ✗  Worker error for {files[idx].name}: {exc}")
                    all_results[idx] = {"source_file": files[idx].name,
                                        "reports": [], "error": str(exc)}

    # Combined output (merge with any pre-existing entries)
    combined_path = out_dir / "_all_reports.json"
    combined = [r for r in all_results if r is not None]
    combined_path.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    total_reports = sum(len(r.get("reports", [])) for r in combined)
    print(f"\n{'═'*70}")
    print(f"  Done. {len(combined)} files  →  {total_reports} total reports")
    print(f"  Combined → {combined_path}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
