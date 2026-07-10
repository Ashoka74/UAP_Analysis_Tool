"""
run_ocr.py
──────────
Batch OCR via Mistral with two modes:

  --mode parallel  (default)
      Sends N concurrent requests using ThreadPoolExecutor.
      Results land immediately. Use --workers to tune concurrency.

  --mode batch
      Builds a JSONL file, submits a Mistral batch job, polls until done,
      then writes all .md files. ~50% cheaper, but async (takes minutes+).

Output always goes to the correct location:
    <doc-folder>/page_XXXX/page_XXXX.md
    <doc-folder>/page_XXXX/page_XXXX_img_0.png  (if images present)

Safe to re-run — pages that already have .md are skipped.

Usage
-----
    pip install mistralai
    set MISTRAL_API_KEY=...

    python run_ocr.py                          # parallel, all targets
    python run_ocr.py --mode batch             # batch API (cheaper, async)
    python run_ocr.py --limit 20               # test on first 20
    python run_ocr.py --workers 20             # more concurrency
    python run_ocr.py --file path/to/page.pdf  # single file
    python run_ocr.py --mode batch --poll      # submit + wait for results
"""

import base64
import os
import re
import sys
import time
import json
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_TARGETS = os.path.join(os.environ.get("UAP_PIPELINE_ROOT", "."), "ocr_targets.txt")
MODEL           = "mistral-ocr-latest"
MAX_RETRIES     = 3
RETRY_WAIT      = 10
BATCH_POLL_SECS = 30   # how often to poll for batch completion


# ── helpers ───────────────────────────────────────────────────────────────────

def encode_pdf(pdf_path: Path) -> str:
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_images(page_dir: Path, page_name: str, images) -> dict:
    saved = {}
    for idx, img in enumerate(images):
        data = getattr(img, "image_base64", None)
        if not data:
            continue
        if "," in data:
            data = data.split(",", 1)[1]
        fname = f"{page_name}_img_{idx}.png"
        (page_dir / fname).write_bytes(base64.b64decode(data))
        saved[getattr(img, "id", str(idx))] = fname
    return saved


def write_md(pdf_path: Path, response) -> Path:
    """Write OCR response to the .md file next to the PDF. Returns md_path."""
    page_dir  = pdf_path.parent
    page_name = pdf_path.stem
    md_path   = page_dir / f"{page_name}.md"

    chunks = []
    for page in response.pages:
        text   = page.markdown or ""
        images = getattr(page, "images", []) or []
        if images:
            saved = save_images(page_dir, page_name, images)
            for img_id, fname in saved.items():
                text = text.replace(img_id, fname)
        chunks.append(text)

    md_path.write_text("\n\n".join(chunks).strip(), encoding="utf-8")
    return md_path


def load_targets(targets_file: Path, single: str | None, limit: int | None) -> list[Path]:
    if single:
        return [Path(single)]
    if not targets_file.exists():
        raise SystemExit(
            f"  ✗  Targets file not found: {targets_file}\n"
            f"     Run:  python find_ocr_targets.py"
        )
    lines = targets_file.read_text(encoding="utf-8").splitlines()
    paths = [
        Path(l.strip()) for l in lines
        if l.strip() and l.strip() != "NOT FOUND"
    ]
    # Skip already-done
    paths = [p for p in paths if not (p.parent / f"{p.stem}.md").exists()]
    if limit:
        paths = paths[:limit]
    return paths


# ── parallel mode ─────────────────────────────────────────────────────────────

def ocr_one(client, pdf_path: Path) -> tuple[Path, bool, str]:
    """OCR a single PDF. Returns (pdf_path, success, message)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            b64 = encode_pdf(pdf_path)
            response = client.ocr.process(
                model=MODEL,
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{b64}",
                },
                table_format=None,
                include_image_base64=True,
            )
            md_path = write_md(pdf_path, response)
            return pdf_path, True, f"{md_path.stat().st_size:,} bytes"
        except Exception as e:
            msg = str(e).lower()
            if any(x in msg for x in ("rate", "429", "503", "502")) and attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT * attempt)
                continue
            return pdf_path, False, str(e)
    return pdf_path, False, "max retries exceeded"


def run_parallel(client, targets: list[Path], workers: int) -> tuple[int, int]:
    ok = failed = 0
    total = len(targets)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ocr_one, client, p): p for p in targets}
        for i, future in enumerate(as_completed(futures), 1):
            pdf_path, success, msg = future.result()
            label = "✓" if success else "✗"
            rel   = "/".join(pdf_path.parts[-3:])
            print(f"  [{i:>4}/{total}]  {label}  {rel}  {msg}")
            if success:
                ok += 1
            else:
                failed += 1

    return ok, failed


# ── batch mode ────────────────────────────────────────────────────────────────

def build_jsonl(targets: list[Path]) -> str:
    """Build a JSONL string — one OCR request per line."""
    lines = []
    for pdf_path in targets:
        b64 = encode_pdf(pdf_path)
        record = {
            "custom_id": str(pdf_path),   # used to map results back to paths
            "body": {
                "model": MODEL,
                "document": {
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{b64}",
                },
                "table_format": None,
                "include_image_base64": True,
            },
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines)


def run_batch(client, targets: list[Path], poll: bool) -> tuple[int, int]:
    total = len(targets)
    print(f"  Building JSONL for {total} files …")
    jsonl_content = build_jsonl(targets)

    # Upload JSONL file
    print("  Uploading JSONL to Mistral Files API …")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(jsonl_content)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            uploaded = client.files.upload(
                file={"file_name": "ocr_batch.jsonl", "content": f},
                purpose="batch",
            )
        file_id = uploaded.id
        print(f"  Uploaded → file_id: {file_id}")

        # Submit batch job
        batch_job = client.batch.jobs.create(
            input_files=[file_id],
            model=MODEL,
            endpoint="/v1/ocr",
            metadata={"description": "UAP archive OCR batch"},
        )
        job_id = batch_job.id
        print(f"  Batch job submitted → job_id: {job_id}")

        # Save job ID for later retrieval if not polling
        job_log = Path(DEFAULT_TARGETS).parent / "ocr_batch_jobs.json"
        jobs = []
        if job_log.exists():
            try:
                jobs = json.loads(job_log.read_text(encoding="utf-8"))
            except Exception:
                pass
        jobs.append({
            "job_id":    job_id,
            "file_id":   file_id,
            "submitted": datetime.now().isoformat(timespec="seconds"),
            "files":     [str(p) for p in targets],
        })
        job_log.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
        print(f"  Job info saved → {job_log}")

        if not poll:
            print(
                f"\n  Job running asynchronously. To fetch results later:\n"
                f"    python run_ocr.py --mode batch-fetch --job-id {job_id}\n"
            )
            return 0, 0

        # Poll for completion
        print("\n  Polling for completion …")
        while True:
            status = client.batch.jobs.get(job_id=job_id)
            pct    = ""
            if hasattr(status, "request_counts") and status.request_counts:
                rc   = status.request_counts
                done = getattr(rc, "completed", 0) + getattr(rc, "failed", 0)
                pct  = f"  ({done}/{total})"
            print(f"  status: {status.status}{pct}", end="\r")
            if status.status in ("SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "EXPIRED"):
                print()
                break
            time.sleep(BATCH_POLL_SECS)

        if status.status != "SUCCESS":
            print(f"  ✗  Batch ended with status: {status.status}")
            return 0, total

        return fetch_batch_results(client, job_id, targets)

    finally:
        os.unlink(tmp_path)


def fetch_batch_results(client, job_id: str, targets: list[Path] | None = None) -> tuple[int, int]:
    """Download results for a completed batch job and write .md files."""
    # Build path lookup from job log if targets not provided
    path_map: dict[str, Path] = {}
    if targets:
        path_map = {str(p): p for p in targets}
    else:
        job_log = Path(DEFAULT_TARGETS).parent / "ocr_batch_jobs.json"
        if job_log.exists():
            jobs = json.loads(job_log.read_text(encoding="utf-8"))
            for job in jobs:
                if job["job_id"] == job_id:
                    path_map = {p: Path(p) for p in job.get("files", [])}
                    break

    job    = client.batch.jobs.get(job_id=job_id)
    out_id = job.output_file

    print(f"  Downloading results (output file: {out_id}) …")
    result_bytes = client.files.download(file_id=out_id)
    results_text = result_bytes.read().decode("utf-8")

    ok = failed = 0
    for line in results_text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        custom_id = record.get("custom_id", "")
        pdf_path  = path_map.get(custom_id)

        if record.get("error"):
            print(f"  ✗  {custom_id}: {record['error']}")
            failed += 1
            continue

        if pdf_path is None:
            print(f"  ⚠  Unknown custom_id: {custom_id}")
            continue

        # Reconstruct a response-like object from the JSON result
        body = record.get("response", {}).get("body", {})
        pages_data = body.get("pages", [])

        page_dir  = pdf_path.parent
        page_name = pdf_path.stem
        md_path   = page_dir / f"{page_name}.md"

        chunks = []
        for page in pages_data:
            text   = page.get("markdown", "") or ""
            images = page.get("images", []) or []
            for idx, img in enumerate(images):
                data = img.get("image_base64", "")
                if data:
                    if "," in data:
                        data = data.split(",", 1)[1]
                    fname = f"{page_name}_img_{idx}.png"
                    (page_dir / fname).write_bytes(base64.b64decode(data))
                    text = text.replace(img.get("id", str(idx)), fname)
            chunks.append(text)

        md_path.write_text("\n\n".join(chunks).strip(), encoding="utf-8")
        print(f"  ✓  {page_dir.name}/{md_path.name}  ({md_path.stat().st_size:,} bytes)")
        ok += 1

    return ok, failed


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="OCR page PDFs via Mistral")
    ap.add_argument("--mode",    default="parallel",
                    choices=["parallel", "batch", "batch-fetch"],
                    help="parallel=immediate, batch=async submit, batch-fetch=fetch existing job")
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--file",    default=None,   help="Single PDF path")
    ap.add_argument("--limit",   type=int,        help="Max files to process")
    ap.add_argument("--workers", type=int, default=10,
                    help="Concurrent workers for parallel mode (default 10)")
    ap.add_argument("--poll",    action="store_true",
                    help="In batch mode: wait and download results when done")
    ap.add_argument("--job-id",  default=None,
                    help="In batch-fetch mode: the job_id to retrieve")
    args = ap.parse_args()

    # Strip stray whitespace and surrounding quotes — a common cause of 401s
    # (e.g. Windows  set MISTRAL_API_KEY="sk-..."  keeps the quotes in the value).
    api_key = (os.environ.get("MISTRAL_API_KEY") or "").strip().strip('"').strip("'").strip()
    if not api_key:
        raise SystemExit("  ✗  MISTRAL_API_KEY not set.")

    from mistralai import Mistral
    client = Mistral(api_key=api_key)

    # Preflight — verify the key now, so a bad key fails immediately with a clear
    # message instead of after every page returns 401.
    try:
        client.models.list()
    except Exception as exc:
        if "401" in str(exc) or "unauthor" in str(exc).lower():
            raise SystemExit(
                "  ✗  MISTRAL_API_KEY rejected by Mistral — 401 Unauthorized.\n"
                "     The key is set but not valid. Check that:\n"
                "       • it is a current key from https://console.mistral.ai/api-keys\n"
                "       • it was set WITHOUT surrounding quotes or stray whitespace\n"
                "         (Windows:  set MISTRAL_API_KEY=sk-...   — no quotes)\n"
                f"       • the key in use is {len(api_key)} characters long\n"
            )
        print(f"  ⚠  Could not pre-verify the API key ({exc}) — continuing anyway.")

    # batch-fetch mode — retrieve a previously submitted job
    if args.mode == "batch-fetch":
        if not args.job_id:
            raise SystemExit("  ✗  --job-id required for batch-fetch mode.")
        ok, failed = fetch_batch_results(client, args.job_id)
        print(f"\n  Done.  ✓ {ok}  ✗ {failed}\n")
        return

    targets = load_targets(Path(args.targets), args.file, args.limit)
    if not targets:
        print("  ✅  Nothing to do — all targets already have .md files.")
        return

    print(f"\n  Mode    : {args.mode}")
    print(f"  Targets : {len(targets)} pages")
    if args.mode == "parallel":
        print(f"  Workers : {args.workers}")
    print()

    start = time.time()

    if args.mode == "parallel":
        ok, failed = run_parallel(client, targets, args.workers)
    else:
        ok, failed = run_batch(client, targets, poll=args.poll)

    elapsed = time.time() - start
    print(f"\n{'─'*60}")
    print(f"  Done in {elapsed:.1f}s   ✓ {ok}   ✗ {failed}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
