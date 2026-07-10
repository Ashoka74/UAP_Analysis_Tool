"""
check_new_release.py
────────────────────
Cron-friendly orchestrator: checks https://www.war.gov/UFO/ for a new UAP
document release and, if one appeared, downloads its PDFs and runs the full
preprocessing pipeline (layout → OCR → assembly → extraction).

Everything is relative to the *working directory* (or --workdir): state file,
downloads, raw/, pages_out/, concat/, extracted/. The pipeline scripts are
resolved relative to this file, so the repo can live anywhere — no absolute
paths, safe for remote/scheduled execution.

State
-----
    release_state.json   (in the workdir)
        {"last_release": 1, "seen_urls": [...], "last_checked": "...", ...}

Usage
-----
    python pipeline/check_new_release.py                # check + run if new
    python pipeline/check_new_release.py --dry-run      # check only, report
    python pipeline/check_new_release.py --force        # run pipeline regardless
    python pipeline/check_new_release.py --workdir /data/uap
    python pipeline/check_new_release.py --stop-after ocr   # partial run

Required env for a full run: MISTRAL_API_KEY (OCR), GEMINI_API_KEY (extraction).

Exit codes: 0 = no new release, or pipeline succeeded; 1 = pipeline step
failed; 2 = could not reach / parse the release page.
"""

import re
import os
import sys
import json
import time
import argparse
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

RELEASE_PAGE = "https://www.war.gov/UFO/"

# The page's JS loads this cumulative manifest — one row per document with
# Release Date / Title / Type (PDF|VID|IMG|AUD) / Agency / direct media link.
# The ?release= query param is ignored server-side (same CSV for any value);
# it exists for client-side filtering. This is the PRIMARY discovery source:
# one GET, then diff the media-link set against state. The page-scrape below
# is the fallback if this endpoint moves (note the year in the path).
MANIFEST_URL = "https://www.war.gov/Portals/1/Interactive/2026/UFO/uap-data.csv"
MANIFEST_LINK_COL = "PDF | Image Link"

# Individual PDFs (release_1-era layout; kept as fallback — the current page
# builds PDF links in JS and only exposes per-release ZIP document bundles).
PDF_LINK_RE = re.compile(
    r"""["'(](?P<url>(?:https?://[^"'()\s]+)?/?medialink/ufo/[^"'()\s]*release[_-]?0*(?P<rel>\d+)[^"'()\s]*?\.pdf)["')]""",
    re.IGNORECASE,
)
# Per-release document bundles, e.g.
#   /medialink/ufo/bundle/Release_1.zip
#   /medialink/ufo/052226/release_02/release_02_document_bundle.zip
#   /medialink/ufo/071026/release_04/release_04_documents_071026.zip
# (video ZIPs live on cloudfront, not /medialink/ufo/ — excluded by design)
ZIP_LINK_RE = re.compile(
    r"""["'(](?P<url>(?:https?://[^"'()\s]+)?/?medialink/ufo/[^"'()\s]*?\.zip)["')]""",
    re.IGNORECASE,
)
REL_NUM_RE = re.compile(r"release[_-]?0*(\d+)", re.IGNORECASE)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,application/octet-stream,*/*;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.war.gov/",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Connection": "keep-alive",
}

STATE_FILE = "release_state.json"
SCRIPTS_DIR = Path(__file__).resolve().parent

# (stage-key, script, extra args) — mirrors preprocessing.py's recommended
# sequence. {dl} is replaced with the release download folder.
PIPELINE_STEPS: list[tuple[str, str, list[str]]] = [
    ("layout",   "split_pages.py",        ["--src", "{dl}", "--out", "."]),
    ("layout",   "restructure_pages.py",  ["--root", ".", "--execute"]),
    ("layout",   "reorganize.py",         ["--root", ".", "--execute"]),
    ("ocr",      "stamp_pages.py",        ["--src", "raw", "--out", "pages_out"]),
    ("ocr",      "find_ocr_targets.py",   ["--root", "."]),
    ("ocr",      "run_ocr.py",            ["--targets", "ocr_targets.txt"]),
    ("ocr",      "destamp_pages.py",      ["--src", "pages_out"]),
    ("assembly", "page_coverage.py",      ["--root", ".", "--out-dir", "."]),
    ("assembly", "concat_pages.py",       ["--root", ".", "--src", "pages_out",
                                           "--no-inline", "--execute", "--force"]),
    ("extract",  "extract_reports.py",    ["--concat", "concat", "--out", "extracted"]),
    ("extract",  "analyze_reports.py",    ["--input", "extracted/_all_reports.json",
                                           "--out", "anomaly_report.md"]),
    # Multimodal layer — IMG/VID/AUD assets → Gemini embeddings → pgvector.
    # embed_media.py exits 0 with a notice when GEMINI_API_KEY/DATABASE_URL
    # are absent, so environments without the DB still complete the run.
    ("embed",    "embed_media.py",        ["--manifest", "uap-csv.csv",
                                           "--release", "{rel}"]),
]
STAGE_ORDER = ["layout", "ocr", "assembly", "extract", "embed"]


def make_opener():
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    try:  # prime a session cookie — war.gov hotlink protection
        opener.open(urllib.request.Request("https://www.war.gov/", headers=HEADERS), timeout=15)
    except Exception as e:
        print(f"  (homepage pre-fetch skipped: {e})")
    return opener


def _cffi_get_bytes(url: str, timeout: int = 60) -> bytes:
    """Chrome-impersonated fetch via curl_cffi.

    Akamai fingerprints TLS: plain urllib passes from residential IPs (with
    the Sec-Fetch headers) but gets 403 from datacenter IPs (Railway, Render,
    cloud runners). curl_cffi presents a real Chrome TLS fingerprint, which
    passes from both. Optional dep — ImportError propagates to the caller.
    """
    from curl_cffi import requests as cffi_requests
    r = cffi_requests.get(url, impersonate="chrome", timeout=timeout,
                          headers={"Referer": "https://www.war.gov/"})
    r.raise_for_status()
    return r.content


def _get_with_fallback(opener, url: str, timeout: int = 30) -> bytes:
    """urllib first (no extra dep), curl_cffi Chrome impersonation second."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with opener.open(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as first_err:
        try:
            return _cffi_get_bytes(url, timeout=timeout)
        except ImportError:
            raise first_err


def fetch_release_page(opener) -> str:
    return _get_with_fallback(opener, RELEASE_PAGE).decode("utf-8", errors="replace")


def fetch_manifest(opener) -> str:
    """Fetch the cumulative uap-data.csv manifest. Raises on HTTP failure."""
    return _get_with_fallback(opener, MANIFEST_URL).decode("utf-8-sig", errors="replace")


def parse_manifest(text: str) -> list[dict]:
    """Parse the manifest CSV into row dicts (blank padding columns dropped)."""
    import csv as _csv
    import io as _io
    return [
        {k: v for k, v in row.items() if k}
        for row in _csv.DictReader(_io.StringIO(text))
    ]


def discover_from_manifest(rows: list[dict]) -> dict[int, set[str]]:
    """Return {release_number: {media urls}} from manifest rows.

    Release number comes from the `release_NN` segment in the media URL;
    rows whose URL lacks one fall back to the rank of their Release Date.
    """
    def _mdy_key(d: str) -> tuple[int, int, int]:
        m, day, y = (int(x) for x in d.split("/"))  # dates are M/D/YY
        return (y, m, day)

    dates = sorted(
        {r.get("Release Date", "").strip() for r in rows if r.get("Release Date", "").strip()},
        key=_mdy_key,
    )
    date_rank = {d: i + 1 for i, d in enumerate(dates)}

    releases: dict[int, set[str]] = {}
    for r in rows:
        url = (r.get(MANIFEST_LINK_COL) or "").strip()
        if not url:
            continue
        m = REL_NUM_RE.search(url)
        rel = int(m.group(1)) if m else date_rank.get(r.get("Release Date", "").strip(), 0)
        if rel:
            releases.setdefault(rel, set()).add(_absolutize(url))
    return releases


def _absolutize(url: str) -> str:
    if not url.lower().startswith("http"):
        return urllib.parse.urljoin("https://www.war.gov/", url)
    return url


def discover_releases(html: str) -> dict[int, set[str]]:
    """Return {release_number: {absolute urls (zip bundles and/or pdfs)}}."""
    releases: dict[int, set[str]] = {}
    for m in ZIP_LINK_RE.finditer(html):
        url = _absolutize(m.group("url"))
        rel_m = REL_NUM_RE.search(os.path.basename(urllib.parse.urlparse(url).path)) \
            or REL_NUM_RE.search(url)
        if rel_m:
            releases.setdefault(int(rel_m.group(1)), set()).add(url)
    for m in PDF_LINK_RE.finditer(html):
        releases.setdefault(int(m.group("rel")), set()).add(_absolutize(m.group("url")))
    return releases


def load_state(workdir: Path) -> dict:
    p = workdir / STATE_FILE
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_release": 0, "seen_urls": []}


def save_state(workdir: Path, state: dict) -> None:
    (workdir / STATE_FILE).write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _stream_download(opener, url: str, filepath: Path) -> int:
    """Stream a URL to disk (bundles can be hundreds of MB). Returns bytes written.

    Same urllib → curl_cffi fallback as _get_with_fallback, kept streaming in
    both paths so multi-GB bundles never sit in memory.
    """
    tmp = filepath.with_suffix(filepath.suffix + ".part")
    written = 0
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with opener.open(req, timeout=120) as resp, open(tmp, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                written += len(chunk)
    except Exception as first_err:
        try:
            from curl_cffi import requests as cffi_requests
        except ImportError:
            raise first_err
        written = 0
        with cffi_requests.Session() as s, open(tmp, "wb") as f:
            r = s.get(url, impersonate="chrome", timeout=300, stream=True,
                      headers={"Referer": "https://www.war.gov/"})
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                written += len(chunk)
    if written < 100:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"response too small ({written} bytes) — likely an error page")
    tmp.replace(filepath)
    return written


def _extract_pdfs(zip_path: Path, dest: Path) -> int:
    """Extract every .pdf member of a bundle, flattened into dest."""
    import zipfile
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".pdf"):
                continue
            name = os.path.basename(member.filename)
            out = dest / name
            if out.exists() and out.stat().st_size > 1024:
                continue
            with zf.open(member) as src, open(out, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            n += 1
    return n


def download_assets(opener, urls: list[str], dest: Path) -> tuple[int, list[str]]:
    """Download release assets. ZIP bundles are extracted (PDFs only) into dest."""
    dest.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []
    for i, url in enumerate(sorted(urls), 1):
        filename = os.path.basename(urllib.parse.unquote(urllib.parse.urlparse(url).path))
        filepath = dest / filename
        try:
            if filepath.exists() and filepath.stat().st_size > 1024:
                print(f"  [{i}/{len(urls)}] SKIP (exists): {filename}")
            else:
                written = _stream_download(opener, url, filepath)
                print(f"  [{i}/{len(urls)}] OK ({written // 1024} KB): {filename}")
            if filename.lower().endswith(".zip"):
                extracted = _extract_pdfs(filepath, dest)
                print(f"       ↳ extracted {extracted} new PDFs from {filename}")
            ok += 1
        except Exception as e:
            print(f"  [{i}/{len(urls)}] FAIL: {filename} — {e}")
            failed.append(url)
        time.sleep(0.4)
    return ok, failed


def run_pipeline(workdir: Path, download_dir: Path, stop_after: str | None,
                 release: int | None = None) -> bool:
    """Run the pipeline steps as subprocesses. Returns True if all succeeded."""
    for stage, script, extra in PIPELINE_STEPS:
        args = [a.replace("{dl}", str(download_dir.relative_to(workdir)))
                 .replace("{rel}", str(release if release is not None else 0))
                for a in extra]
        cmd = [sys.executable, str(SCRIPTS_DIR / script), *args]
        print(f"\n▶ [{stage}] {script} {' '.join(args)}")
        proc = subprocess.run(cmd, cwd=str(workdir))
        if proc.returncode != 0:
            print(f"✗ {script} exited with code {proc.returncode} — aborting.")
            return False
        if stop_after and stage == stop_after and (
            PIPELINE_STEPS.index((stage, script, extra)) == max(
                i for i, s in enumerate(PIPELINE_STEPS) if s[0] == stage)
        ):
            print(f"\n■ Stopping after stage '{stop_after}' as requested.")
            return True
    return True


def main():
    ap = argparse.ArgumentParser(description="Check war.gov for a new UAP release and run the pipeline")
    ap.add_argument("--workdir", default=".", help="Pipeline data directory (default: cwd)")
    ap.add_argument("--dry-run", action="store_true", help="Check + report only, never run")
    ap.add_argument("--force", action="store_true", help="Run pipeline even if nothing new")
    ap.add_argument("--stop-after", choices=STAGE_ORDER, default=None,
                    help="Stop after this stage (layout/ocr/assembly/extract)")
    args = ap.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("UAP_PIPELINE_ROOT", str(workdir))

    print(f"Workdir : {workdir}")
    print(f"Checking: {RELEASE_PAGE}")

    opener = make_opener()

    # Primary: the CSV manifest (one GET, full per-document metadata).
    releases: dict[int, set[str]] = {}
    manifest_text = None
    try:
        manifest_text = fetch_manifest(opener)
        releases = discover_from_manifest(parse_manifest(manifest_text))
        print(f"Manifest: {MANIFEST_URL} ({len(manifest_text)//1024} KB)")
    except Exception as e:
        print(f"⚠ Manifest fetch failed ({e}) — falling back to page scrape.")

    # Fallback: scrape the landing page for release ZIP bundles / PDF links.
    if not releases:
        try:
            html = fetch_release_page(opener)
        except Exception as e:
            print(f"✗ Could not fetch release page: {e}")
            sys.exit(2)
        releases = discover_releases(html)

    if not releases:
        print("✗ No release links found via manifest or page — layout may have changed.")
        sys.exit(2)

    # Keep the manifest beside the data — reconcile.py/audit.py consume it
    # as the uap-csv.csv ground-truth (Release Date / Title / Agency columns).
    if manifest_text:
        (workdir / "uap-csv.csv").write_text(manifest_text, encoding="utf-8")

    latest = max(releases)
    state = load_state(workdir)
    seen = set(state.get("seen_urls", []))
    latest_urls = releases[latest]
    new_urls = sorted(latest_urls - seen)

    print(f"Latest release on page : release_{latest} ({len(latest_urls)} asset(s))")
    print(f"Last processed release : release_{state.get('last_release', 0)}")
    print(f"New assets             : {len(new_urls)}")

    state["last_checked"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    is_new = latest > state.get("last_release", 0) or bool(new_urls)
    if not is_new and not args.force:
        print("\n✅ Nothing new — exiting.")
        save_state(workdir, state)
        return

    if args.dry_run:
        print("\n(dry-run) New release detected — pipeline NOT run.")
        for u in new_urls[:20]:
            print(f"    {u}")
        if len(new_urls) > 20:
            print(f"    … and {len(new_urls) - 20} more")
        save_state(workdir, state)
        return

    # Fail fast on missing keys before touching anything.
    missing_keys = [k for k in ("MISTRAL_API_KEY", "GEMINI_API_KEY") if not os.environ.get(k)]
    if missing_keys:
        print(f"✗ Missing required env: {', '.join(missing_keys)}")
        sys.exit(1)

    download_dir = workdir / "downloads" / f"release_{latest}"
    to_download = new_urls if new_urls else sorted(latest_urls)
    # The text pipeline consumes PDFs (directly or out of ZIP bundles); manifest
    # rows also carry jpg/mp4 asset links — record them in state but skip the
    # download until the multimodal-embedding stage lands.
    skipped_media = [u for u in to_download
                     if not u.lower().split("?")[0].endswith((".pdf", ".zip"))]
    to_download = [u for u in to_download if u not in set(skipped_media)]
    if skipped_media:
        print(f"({len(skipped_media)} non-PDF media assets recorded but not downloaded)")
    if not to_download:
        # e.g. a video/image-only addendum — nothing for the text pipeline,
        # but mark the assets seen so next week isn't a false positive.
        print("No PDF/ZIP assets to process — recording media assets and exiting.")
        state["last_release"] = latest
        state["seen_urls"] = sorted(seen | set(skipped_media))
        save_state(workdir, state)
        return

    print(f"\nDownloading {len(to_download)} asset(s) → {download_dir}")
    ok, failed = download_assets(opener, to_download, download_dir)
    print(f"Downloaded {ok}/{len(to_download)}" + (f" ({len(failed)} failed)" if failed else ""))
    if ok == 0:
        print("✗ Nothing downloaded — aborting before pipeline.")
        sys.exit(1)

    if not run_pipeline(workdir, download_dir, args.stop_after, release=latest):
        # Keep seen_urls unchanged so the next run retries the pipeline.
        save_state(workdir, state)
        sys.exit(1)

    state["last_release"] = latest
    state["seen_urls"] = sorted((seen | set(to_download) | set(skipped_media)) - set(failed))
    state["last_run"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_state(workdir, state)
    print(f"\n✅ Pipeline complete for release_{latest}. State saved → {workdir / STATE_FILE}")


if __name__ == "__main__":
    main()
