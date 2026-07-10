"""
embed_media.py
──────────────
Pipeline stage: multimodal embeddings for the non-PDF assets of a war.gov
UAP release — images (IMG), videos (VID) and audio (AUD) — stored in the
Neon pgvector `embeddings` table via embeddings_v2.py.

Sources
-------
  IMG  → direct war.gov link in the manifest ("PDF | Image Link" column);
         embedded together with the row's title + description blurb.
  VID/AUD → referenced by DVIDS id only; the bytes come from the release's
         CloudFront video ZIP (discovered on the /UFO/ page, e.g.
         uap_release04_videos_071026.zip), matched to manifest rows by the
         document code prefix (DOW-UAP-PR104_...). Videos are chunked to
         ≤90s (Gemini's 120s cap) — one pgvector row per chunk, keyed
         {doc_code}:{start_ms}-{end_ms} so hits map to a timestamp.

Requirements: GEMINI_API_KEY + DATABASE_URL (+ ffmpeg for video/audio).
When either env var is missing the stage prints a notice and exits 0, so
the weekly pipeline still succeeds without the multimodal layer.

Idempotent: rows already in the DB (has_embedding / has_video_chunks) are
skipped, so re-runs only fill gaps.

Usage
-----
    python embed_media.py                          # all unembedded assets
    python embed_media.py --release 4              # only release 4 rows
    python embed_media.py --skip-video             # images only (no zip DL)
    python embed_media.py --limit 5 --dry-run      # preview the work list
"""

import re
import os
import sys
import csv
import json
import shutil
import zipfile
import argparse
import urllib.parse
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.check_new_release import (  # noqa: E402
    HEADERS, REL_NUM_RE, MANIFEST_LINK_COL,
    make_opener, fetch_release_page, parse_manifest,
)

DEFAULT_MANIFEST = "uap-csv.csv"
# Deterministic system user for pipeline-owned rows (embeddings.user_id is
# a uuid column). Override with a real account uuid if the app needs to see
# these rows under a login.
DEFAULT_USER_ID = os.environ.get(
    "UAP_EMBED_USER_ID", "00000000-0000-0000-0000-000000000000"
)

MEDIA_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv",
              ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}

CLOUDFRONT_ZIP_RE = re.compile(
    r"""["'(](?P<url>https://[a-z0-9]+\.cloudfront\.net/[^"'()\s]+?\.zip)["')]""",
    re.IGNORECASE,
)


def doc_code(title: str) -> str:
    """'DOW-UAP-PR104, Unresolved UAP Report, …' → 'DOW-UAP-PR104'."""
    return (title or "").split(",", 1)[0].strip()


def rows_for_release(rows: list[dict], release: int | None) -> list[dict]:
    if release is None:
        return rows
    def _mdy_key(d: str) -> tuple[int, int, int]:
        m, day, y = (int(x) for x in d.split("/"))
        return (y, m, day)

    dates = sorted(
        {r.get("Release Date", "").strip() for r in rows if r.get("Release Date", "").strip()},
        key=_mdy_key,
    )
    date_rank = {d: i + 1 for i, d in enumerate(dates)}
    out = []
    for r in rows:
        url = (r.get(MANIFEST_LINK_COL) or "").strip()
        m = REL_NUM_RE.search(url) if url else None
        rel = int(m.group(1)) if m else date_rank.get(r.get("Release Date", "").strip(), 0)
        if rel == release:
            out.append(r)
    return out


def find_video_zip(opener, release: int | None) -> str | None:
    """Scrape the /UFO/ page for the release's CloudFront video ZIP."""
    try:
        html = fetch_release_page(opener)
    except Exception as e:
        print(f"  ⚠ could not scrape /UFO/ for video zips: {e}")
        return None
    zips = [m.group("url") for m in CLOUDFRONT_ZIP_RE.finditer(html)]
    if not zips:
        return None
    if release is not None:
        for u in zips:  # e.g. .../release_04/uap_release04_videos_071026.zip
            m = REL_NUM_RE.search(u)
            if m and int(m.group(1)) == release:
                return u
        return None
    return zips[-1]


def ensure_media_dir(opener, workdir: Path, release: int | None) -> Path | None:
    """Download + extract the release's video ZIP → downloads/media_release_N/."""
    tag = f"release_{release}" if release is not None else "latest"
    media_dir = workdir / "downloads" / f"media_{tag}"
    if media_dir.is_dir() and any(p.suffix.lower() in MEDIA_EXTS for p in media_dir.iterdir()):
        return media_dir  # already extracted

    zip_url = find_video_zip(opener, release)
    if not zip_url:
        print("  ⚠ no CloudFront video ZIP found for this release — VID/AUD skipped")
        return None

    zip_path = workdir / "downloads" / os.path.basename(urllib.parse.urlparse(zip_url).path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"  ↓ downloading video bundle ({zip_url}) — this can be several GB …")
        req = urllib.request.Request(zip_url, headers=HEADERS)
        tmp = zip_path.with_suffix(".part")
        with opener.open(req, timeout=300) as resp, open(tmp, "wb") as f:
            while chunk := resp.read(1 << 22):
                f.write(chunk)
        tmp.replace(zip_path)

    media_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            ext = Path(member.filename).suffix.lower()
            if member.is_dir() or ext not in MEDIA_EXTS:
                continue
            out = media_dir / os.path.basename(member.filename)
            if out.exists():
                continue
            with zf.open(member) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            n += 1
    print(f"  ↳ extracted {n} media files → {media_dir}")
    # The multi-GB zip has served its purpose; keep disk usage bounded.
    zip_path.unlink(missing_ok=True)
    return media_dir


def match_media_file(media_dir: Path, code: str) -> Path | None:
    """Find the media file whose name starts with the document code."""
    if not media_dir:
        return None
    norm = code.lower()
    for p in sorted(media_dir.iterdir()):
        if p.suffix.lower() in MEDIA_EXTS and p.name.lower().startswith(norm):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="Embed release media (IMG/VID/AUD) into pgvector")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--release", type=int, default=None,
                    help="Only rows of this release number (default: all)")
    ap.add_argument("--user-id", default=DEFAULT_USER_ID)
    ap.add_argument("--limit", type=int, default=None, help="Max assets to process")
    ap.add_argument("--skip-video", action="store_true",
                    help="Images only — do not download the video bundle")
    ap.add_argument("--dry-run", action="store_true", help="List work, embed nothing")
    args = ap.parse_args()
    if args.release == 0:  # orchestrator's "unknown release" placeholder
        args.release = None

    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise SystemExit(f"  ✗ manifest not found: {manifest} — run check_new_release.py first")
    rows = rows_for_release(parse_manifest(manifest.read_text(encoding="utf-8-sig")), args.release)

    imgs = [r for r in rows if r.get("Type", "").strip() == "IMG"
            and (r.get(MANIFEST_LINK_COL) or "").strip()]
    vids = [r for r in rows if r.get("Type", "").strip() == "VID"]
    auds = [r for r in rows if r.get("Type", "").strip() == "AUD"]
    print(f"  Manifest rows in scope: {len(rows)}  (IMG {len(imgs)} | VID {len(vids)} | AUD {len(auds)})")

    if args.dry_run:
        for r in (imgs + vids + auds)[: args.limit or 50]:
            print(f"    {r.get('Type'):>3}  {doc_code(r.get('Title'))}")
        return

    missing = [k for k in ("GEMINI_API_KEY", "DATABASE_URL") if not os.environ.get(k)]
    if missing:
        # Exit 0 on purpose: the weekly pipeline must not fail just because
        # the multimodal layer isn't configured in this environment.
        print(f"  ⚠ embed_media skipped — missing env: {', '.join(missing)}")
        return

    from embeddings_v2 import (
        format_document_text, generate_multimodal_embedding, generate_image_embedding,
        generate_audio_embedding, chunk_and_embed_video,
        store_embedding, store_video_chunk_embedding, store_audio_clip_embedding,
        has_embedding, has_video_chunks,
    )

    workdir = Path.cwd()
    opener = make_opener()
    media_dir = None
    if (vids or auds) and not args.skip_video:
        media_dir = ensure_media_dir(opener, workdir, args.release)

    done = failed = skipped = 0
    budget = args.limit or float("inf")

    def spend() -> bool:
        nonlocal budget
        budget -= 1
        return budget >= 0

    # ── images ────────────────────────────────────────────────────────────
    for r in imgs:
        code, url = doc_code(r.get("Title")), r[MANIFEST_LINK_COL].strip()
        title, blurb = r.get("Title", ""), r.get("Description Blurb", "")
        if has_embedding("asset", code):
            skipped += 1
            continue
        if not spend():
            break
        try:
            vec = (generate_multimodal_embedding(url, format_document_text(title, blurb))
                   if blurb else generate_image_embedding(url))
            store_embedding(source_type="asset", source_id=code, user_id=args.user_id,
                            embedding=vec, image_url=url, text=blurb or title)
            print(f"  ✓ IMG {code}")
            done += 1
        except Exception as e:
            print(f"  ✗ IMG {code}: {e}")
            failed += 1

    # ── videos ────────────────────────────────────────────────────────────
    for r in vids:
        code = doc_code(r.get("Title"))
        title, blurb = r.get("Title", ""), r.get("Description Blurb", "")
        if has_video_chunks(code):
            skipped += 1
            continue
        src = match_media_file(media_dir, code) if media_dir else None
        if src is None:
            print(f"  ⚠ VID {code}: no local media file (dvids {r.get('DVIDS Video ID')}) — skipped")
            continue
        if not spend():
            break
        try:
            chunks = chunk_and_embed_video(str(src), text=format_document_text(title, blurb))
            for ch in chunks:
                store_video_chunk_embedding(
                    video_id=code, start_seconds=ch.start_s, end_seconds=ch.end_s,
                    user_id=args.user_id, embedding=ch.vector,
                    video_url=f"https://www.dvidshub.net/video/{r.get('DVIDS Video ID')}",
                    text=blurb or title,
                )
            print(f"  ✓ VID {code} ({len(chunks)} chunk(s))")
            done += 1
        except Exception as e:
            print(f"  ✗ VID {code}: {e}")
            failed += 1

    # ── audio ─────────────────────────────────────────────────────────────
    for r in auds:
        code = doc_code(r.get("Title"))
        title, blurb = r.get("Title", ""), r.get("Description Blurb", "")
        if has_embedding("audio_clip", f"{code}:0-0") or has_video_chunks(code):
            skipped += 1
            continue
        src = match_media_file(media_dir, code) if media_dir else None
        if src is None or src.suffix.lower() not in AUDIO_EXTS:
            print(f"  ⚠ AUD {code}: no local audio file — skipped")
            continue
        if not spend():
            break
        try:
            vec = generate_audio_embedding(str(src), text=format_document_text(title, blurb))
            store_audio_clip_embedding(
                audio_id=code, user_id=args.user_id, embedding=vec,
                audio_url=f"https://www.dvidshub.net/audio/{r.get('DVIDS Video ID')}",
                text=blurb or title,
            )
            print(f"  ✓ AUD {code}")
            done += 1
        except Exception as e:
            print(f"  ✗ AUD {code}: {e}")
            failed += 1

    print(f"\n  embed_media done: ✓ {done}  ✗ {failed}  skipped(existing) {skipped}")
    # Non-zero exit only when nothing succeeded but work was attempted —
    # partial success shouldn't fail the weekly pipeline.
    if failed and not done:
        sys.exit(1)


if __name__ == "__main__":
    main()
