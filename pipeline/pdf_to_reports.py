"""
pdf_to_reports.py
-----------------
Multithreaded PDF → per-report .md extractor via NVIDIA NIM.

TWO-PASS PIPELINE
  Pass 1 (parallel)  — every non-blank page is sent to NIM concurrently for text
                        clean-up and metadata extraction.
  Pass 2 (single)    — a compact "page manifest" (page number + header snippet +
                        detected signals) is sent to NIM in ONE call so it can
                        see the full document structure and authoritatively decide
                        which pages start new reports.  For PDFs > SEGMENT_CHUNK
                        pages the manifest is split into overlapping chunks and
                        the boundary lists are merged.

Usage:
    NVIDIA_API_KEY=<key> python pdf_to_reports.py path/to/file.pdf
    NVIDIA_API_KEY=<key> python pdf_to_reports.py path/to/file.pdf --out my_reports --workers 6
    NVIDIA_API_KEY=<key> python pdf_to_reports.py path/to/file.pdf --no-segment
"""

import os
import re
import time
import json
import random
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from textwrap import shorten

import pdfplumber
from openai import OpenAI, RateLimitError, APIStatusError

# ── client ────────────────────────────────────────────────────────────────────

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

MODEL            = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
MAX_RETRIES      = 6
BASE_WAIT        = 1.0    # seconds; doubles per retry attempt
MAX_CHARS_PAGE   = 8_000  # chars sent to NIM per page in pass 1
HEADER_CHARS     = 300    # chars per page sent in the pass-2 manifest
SEGMENT_CHUNK    = 80     # pages per segmentation call (fits well within context)
SEGMENT_OVERLAP  = 5      # pages of overlap between chunks to catch cross-boundary reports

# ── NIM call with exponential backoff ─────────────────────────────────────────

def call_nim(messages: list[dict], *, max_retries: int = MAX_RETRIES) -> str:
    """
    Call the NIM chat endpoint.  Retries on 429 and 5xx with
    exponential back-off + full jitter to prevent thundering herd.
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                top_p=0.95,
                max_tokens=8192,
                stream=False,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": 4096,
                },
            )
            return resp.choices[0].message.content or ""

        except RateLimitError:
            wait = BASE_WAIT * (2 ** attempt) + random.uniform(0, BASE_WAIT)
            print(f"    [rate-limit] attempt {attempt + 1}/{max_retries} — waiting {wait:.1f}s …")
            time.sleep(wait)

        except APIStatusError as exc:
            if exc.status_code >= 500:
                wait = BASE_WAIT * (2 ** attempt) + random.uniform(0, BASE_WAIT)
                print(f"    [server-error {exc.status_code}] attempt {attempt + 1}/{max_retries} — waiting {wait:.1f}s …")
                time.sleep(wait)
            else:
                raise   # 4xx other than 429 are not retryable

    raise RuntimeError(f"NIM API failed after {max_retries} attempts")


def _parse_json(raw: str) -> dict | list:
    """Strip markdown fences and parse JSON; raise on failure."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — per-page extraction (parallel)
# ══════════════════════════════════════════════════════════════════════════════

PAGE_SYSTEM = (
    "You are a precise document-analysis assistant. "
    "Output ONLY valid JSON — no prose, no markdown fences."
)

PAGE_PROMPT = """\
Analyse the page below (from a government / agency report PDF).

Return valid JSON matching this exact schema (no extra keys):
{{
  "report_id": <str|null>,
  "metadata": {{
    "date":               <str|null>,
    "location":           <str|null>,
    "agency":             <str|null>,
    "classification":     <str|null>,
    "object_description": <str|null>,
    "witnesses":          <str|null>,
    "redacted_sections":  [<str>]
  }},
  "page_text": <str>,
  "header_line": <str>   // first meaningful heading or first 120 chars of text
}}

[PAGE {page_num}]
{text}
"""


def process_page(page_num: int, raw_text: str) -> dict:
    """Send one page to NIM; return structured dict."""
    print(f"  → page {page_num:>4d}")
    messages = [
        {"role": "system", "content": PAGE_SYSTEM},
        {"role": "user",   "content": PAGE_PROMPT.format(
            page_num=page_num,
            text=raw_text[:MAX_CHARS_PAGE],
        )},
    ]
    raw = call_nim(messages)
    try:
        result = _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        result = {
            "report_id":   None,
            "metadata":    {"redacted_sections": []},
            "page_text":   raw_text,
            "header_line": raw_text[:120],
        }
    result["page_num"] = page_num
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — document-level segmentation (single / chunked call)
# ══════════════════════════════════════════════════════════════════════════════

SEGMENT_SYSTEM = (
    "You are a document-segmentation assistant. "
    "Output ONLY valid JSON — no prose, no markdown fences."
)

SEGMENT_PROMPT = """\
Below is a page manifest for a multi-report PDF.  Each entry shows the page
number, the report/case ID found on that page (if any), and the opening text.

Your task: identify which pages START a new, distinct report or case.
A new report typically begins with a new case number, incident number, new
header/title, or a clear section break.  Continuation pages (cover sheets,
attachments, exhibits that belong to the same report) are NOT new reports.

Return a JSON array — one object per page — in page order:
[
  {{"page": <int>, "starts_report": <bool>, "report_id": <str|null>}},
  ...
]

PAGE MANIFEST:
{manifest}
"""


def _build_manifest(pages: list[dict]) -> str:
    """Build a compact text manifest for the segmentation call."""
    lines = []
    for p in sorted(pages, key=lambda x: x["page_num"]):
        rid    = p.get("report_id") or "—"
        header = shorten(p.get("header_line") or p.get("page_text") or "", HEADER_CHARS, placeholder="…")
        lines.append(f"p{p['page_num']:>4d}  id={rid:<20s}  {header}")
    return "\n".join(lines)


def _segment_chunk(pages: list[dict]) -> list[dict]:
    """
    Call NIM once with a manifest for `pages` and return boundary info.
    Returns: [{"page": int, "starts_report": bool, "report_id": str|None}, ...]
    """
    manifest = _build_manifest(pages)
    messages = [
        {"role": "system", "content": SEGMENT_SYSTEM},
        {"role": "user",   "content": SEGMENT_PROMPT.format(manifest=manifest)},
    ]
    raw = call_nim(messages)
    try:
        result = _parse_json(raw)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    # fallback: first page of chunk starts a report, rest continue
    return [
        {"page": p["page_num"], "starts_report": (i == 0), "report_id": p.get("report_id")}
        for i, p in enumerate(sorted(pages, key=lambda x: x["page_num"]))
    ]


def segment_document(pages: list[dict]) -> dict[int, dict]:
    """
    Run pass-2 segmentation over the full page list, chunking if needed.

    Returns a dict keyed by page_num:
      {page_num: {"starts_report": bool, "report_id": str|None}}
    """
    sorted_pages = sorted(pages, key=lambda p: p["page_num"])
    n            = len(sorted_pages)
    boundary_map: dict[int, dict] = {}

    if n <= SEGMENT_CHUNK:
        # entire document fits in one call
        print("  → segmentation: single call")
        results = _segment_chunk(sorted_pages)
        for r in results:
            boundary_map[r["page"]] = r
    else:
        # chunk with overlap so reports that straddle chunk edges are detected
        step  = SEGMENT_CHUNK - SEGMENT_OVERLAP
        start = 0
        chunk_idx = 1
        seen: set[int] = set()

        while start < n:
            end   = min(start + SEGMENT_CHUNK, n)
            chunk = sorted_pages[start:end]
            print(f"  → segmentation chunk {chunk_idx}  (pages {chunk[0]['page_num']}–{chunk[-1]['page_num']})")
            results = _segment_chunk(chunk)
            for r in results:
                pn = r["page"]
                if pn not in seen:
                    boundary_map[pn] = r
                    seen.add(pn)
                else:
                    # overlap zone: OR the starts_report flags
                    # (if either pass said "new report", trust it)
                    boundary_map[pn]["starts_report"] = (
                        boundary_map[pn]["starts_report"] or r["starts_report"]
                    )
                    # prefer the more specific report_id
                    if not boundary_map[pn].get("report_id") and r.get("report_id"):
                        boundary_map[pn]["report_id"] = r["report_id"]
            start     += step
            chunk_idx += 1

    return boundary_map


# ══════════════════════════════════════════════════════════════════════════════
# Grouping, metadata merge, markdown output
# ══════════════════════════════════════════════════════════════════════════════

def apply_segmentation(pages: list[dict], boundary_map: dict[int, dict]) -> list[dict]:
    """
    Override each page's report_id with the authoritative value from pass 2,
    and set a 'new_report_start' flag according to the boundary map.
    """
    for p in pages:
        pn   = p["page_num"]
        info = boundary_map.get(pn, {})
        p["new_report_start"] = info.get("starts_report", False)
        if info.get("report_id"):
            p["report_id"] = info["report_id"]
    return pages


def group_into_reports(pages: list[dict]) -> list[list[dict]]:
    """Split sorted pages into report groups on new_report_start boundaries."""
    groups:  list[list[dict]] = []
    current: list[dict]       = []

    for page in sorted(pages, key=lambda p: p["page_num"]):
        if page.get("new_report_start") and current:
            groups.append(current)
            current = [page]
        else:
            current.append(page)

    if current:
        groups.append(current)

    return groups


def merge_metadata(pages: list[dict]) -> dict:
    """
    Merge metadata across all pages: first non-null scalar value wins;
    redacted_sections lists are concatenated and de-duplicated.
    """
    merged: dict = {"redacted_sections": []}
    seen_redacted: set[str] = set()

    for page in pages:
        m = page.get("metadata") or {}
        for key, val in m.items():
            if key == "redacted_sections":
                for item in val or []:
                    if item not in seen_redacted:
                        merged["redacted_sections"].append(item)
                        seen_redacted.add(item)
            elif val and key not in merged:
                merged[key] = val

    merged["report_id"] = next(
        (p["report_id"] for p in pages if p.get("report_id")),
        f"report_p{pages[0]['page_num']}",
    )
    merged["pages"] = [p["page_num"] for p in pages]
    return merged


def write_report_md(report_pages: list[dict], out_dir: Path, index: int) -> Path:
    """Write one .md file for a single logical report."""
    meta  = merge_metadata(report_pages)
    slug  = re.sub(r"[^\w\-]", "_", str(meta["report_id"]))
    path  = out_dir / f"{slug}.md"

    lines = [
        f"# {meta['report_id']}",
        "",
        "## Metadata",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Pages | {meta['pages']} |",
        f"| Date | {meta.get('date', '—')} |",
        f"| Location | {meta.get('location', '—')} |",
        f"| Agency | {meta.get('agency', '—')} |",
        f"| Classification | {meta.get('classification', '—')} |",
        f"| Object / Phenomenon | {meta.get('object_description', '—')} |",
        f"| Witnesses | {meta.get('witnesses', '—')} |",
    ]
    if meta["redacted_sections"]:
        lines.append(f"| Redacted sections | {'; '.join(meta['redacted_sections'])} |")

    lines += ["", "---", "", "## Content", ""]

    for page in sorted(report_pages, key=lambda p: p["page_num"]):
        lines.append(f"### Page {page['page_num']}")
        lines.append("")
        lines.append((page.get("page_text") or "").strip())
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def parse_pdf_to_reports(
    pdf_path:    str | Path,
    out_dir:     str | Path = "reports_out",
    max_workers: int        = 4,
    use_segment: bool       = True,
) -> list[Path]:
    """
    Full two-pass pipeline.

    Pass 1 (parallel):  extract + clean every page via NIM.
    Pass 2 (single):    segment the full document via NIM to find report
                        boundaries, chunking every SEGMENT_CHUNK pages for
                        very long PDFs and merging results across overlaps.

    Set use_segment=False to skip pass 2 and rely on per-page heuristics only.
    """
    pdf_path = Path(pdf_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 0. extract raw text ───────────────────────────────────────────────────
    print(f"\n📄  {pdf_path.name}")
    with pdfplumber.open(pdf_path) as pdf:
        pages_raw = [
            (i + 1, page.extract_text() or "")
            for i, page in enumerate(pdf.pages)
        ]
    non_blank = [(n, t) for n, t in pages_raw if t.strip()]
    print(f"    {len(pages_raw)} pages total, {len(non_blank)} non-blank")

    # ── 1. per-page NIM calls (threaded) ─────────────────────────────────────
    print(f"\n🧵  Pass 1 — page extraction  (workers={max_workers}) …")
    results: list[dict] = []
    errors:  list[int]  = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_page, num, text): num for num, text in non_blank}
        for fut in as_completed(futures):
            pn = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                print(f"  ✗ page {pn} permanently failed: {exc}")
                errors.append(pn)

    if errors:
        print(f"  ⚠  {len(errors)} page(s) failed: {errors}")

    # ── 2. document-level segmentation (single / chunked) ────────────────────
    if use_segment:
        print(f"\n🔍  Pass 2 — document segmentation …")
        boundary_map = segment_document(results)
        results = apply_segmentation(results, boundary_map)
    else:
        print("\n⏭  Skipping pass 2 (--no-segment)")

    # ── 3. group + write ──────────────────────────────────────────────────────
    groups = group_into_reports(results)
    print(f"\n📊  {len(groups)} report(s) found across {len(results)} pages\n")

    written: list[Path] = []
    for i, group in enumerate(groups, start=1):
        path = write_report_md(group, out_dir, i)
        print(f"  ✓  {path.name}  ({len(group)} page(s))")
        written.append(path)

    print(f"\n✅  Done — {len(written)} file(s) → {out_dir}/\n")
    return written


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Parse a PDF of reports into per-report .md files via NVIDIA NIM (two-pass)"
    )
    ap.add_argument("pdf",                           help="Path to input PDF")
    ap.add_argument("--out",      default="reports_out", help="Output directory  [reports_out]")
    ap.add_argument("--workers",  type=int, default=4,   help="Parallel NIM threads for pass 1  [4]")
    ap.add_argument("--no-segment", dest="segment",
                    action="store_false", default=True,
                    help="Skip pass-2 segmentation (use per-page heuristics only)")
    args = ap.parse_args()

    parse_pdf_to_reports(
        args.pdf,
        out_dir=args.out,
        max_workers=args.workers,
        use_segment=args.segment,
    )
