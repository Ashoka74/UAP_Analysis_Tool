"""
Multimodal embedding + pgvector storage helper.

Generates embeddings via Google Gemini (gemini-embedding-2-preview) and
persists / queries them in the `embeddings` Postgres table.

Supports text, image, audio, and video — with automatic chunking of videos
longer than the model's 120s input limit and Files API uploads for large
media. Each video chunk gets its own row keyed by (video_id, start_ms-end_ms)
so retrieval hits map back to a playable timestamp.

Setup (local / no DB — embeddings, search, clustering all in memory):
    pip install google-genai pillow requests
    export GEMINI_API_KEY=...
    # ffmpeg in PATH for video chunking

Setup (full — adds Postgres storage):
    pip install google-genai pillow requests "psycopg[binary]" pgvector
    export GEMINI_API_KEY=...
    export DATABASE_URL=postgres://...

Schema additions (run once, see SCHEMA_MIGRATION_SQL at bottom of file):
    ALTER TABLE embeddings ADD COLUMN start_seconds REAL;
    ALTER TABLE embeddings ADD COLUMN end_seconds   REAL;
    ALTER TABLE embeddings ADD COLUMN parent_id     TEXT;
    CREATE INDEX IF NOT EXISTS idx_embeddings_parent_id
        ON embeddings (parent_id) WHERE parent_id IS NOT NULL;

The DB libs are imported lazily inside _get_conn(), so importing this module
without psycopg/pgvector installed is fine — only the storage helpers will
fail with a clear error if you call them.

Embedding-only usage:
    from embeddings_v2 import (
        generate_image_embedding,
        generate_text_embedding,
        generate_multimodal_embedding,
        generate_video_embedding,
        generate_audio_embedding,
        chunk_and_embed_video,
        cosine_similarity,
    )

    vec = generate_multimodal_embedding("./chair.jpg", "modern walnut dining chair")
    chunks = chunk_and_embed_video("https://example.com/clip.mp4")
    # chunks -> [VideoChunkEmbedding(start_s, end_s, vector), ...]

Storage usage:
    from embeddings_v2 import (
        store_embedding,
        store_video_chunk_embedding,
        search_similar,
        delete_embedding,
        delete_video_chunks,
        has_embedding,
        find_clusters,
    )

    for ch in chunks:
        store_video_chunk_embedding(
            video_id="dvids_1007706",
            start_seconds=ch.start_s,
            end_seconds=ch.end_s,
            user_id="00000000-...",
            embedding=ch.vector,
            video_url="https://...",
        )
    hits = search_similar(
        generate_text_embedding("formation maneuvers over water"),
        user_id="00000000-...",
        source_type="video_chunk",
        limit=10,
    )

CLI:
    python embeddings_v2.py text "modern walnut dining chair"
    python embeddings_v2.py image ./chair.jpg
    python embeddings_v2.py video ./clip.mp4                 # single chunk (<=120s)
    python embeddings_v2.py video-chunks ./clip.mp4          # chunk + embed all
    python embeddings_v2.py audio ./speech.mp3
    python embeddings_v2.py search USER_UUID --text "query" --source-type video_chunk
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import requests
from google import genai
from google.genai import types as genai_types
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "gemini-embedding-2-preview"
# Default kept at 768 for backward compat with the existing image/text rows
# already in the DB. Override via embedding_dimensions kwarg or env var if you
# want richer video vectors (1536 is a good middle ground).
DEFAULT_EMBEDDING_DIMENSIONS = int(
    os.environ.get("GEMINI_EMBEDDING_DIMENSIONS", "768")
)

# Gemini Embedding 2 hard caps. Keep our chunks a bit under the model max so
# tiny timing differences in ffmpeg don't push us over the wire.
MAX_VIDEO_SECONDS = 120.0
DEFAULT_CHUNK_SECONDS = 90.0
DEFAULT_CHUNK_OVERLAP_SECONDS = 5.0
MAX_AUDIO_SECONDS = 80.0

# Files >= this size go through the Files API. The inline-bytes path is
# limited to ~20MB total request payload, and video files exceed that fast.
INLINE_SIZE_THRESHOLD_BYTES = 15 * 1024 * 1024  # 15 MB

# Pillow can decode these but Gemini may reject them — normalise to JPEG.
UNSUPPORTED_IMAGE_MIME = {"image/webp", "image/tiff", "image/bmp", "image/avif"}

# Source types stored in the DB. Existing app uses "asset" and "gallery";
# video/audio/pdf_page extend the union without breaking those.
SourceType = Literal["asset", "gallery", "video_chunk", "audio_clip", "pdf_page"]

# gemini-embedding-2 uses INSTRUCTION-IN-PROMPT for task signalling rather
# than the EmbedContentConfig.task_type field (which the model silently
# ignores on the consumer API). Documents go in wrapped as
#   "title: {title} | text: {body}"
# and queries as
#   "task: search result | query: {q}"
# Use the helpers below to wrap text before passing it to a generate_*
# function. generate_text_embedding() applies format_query() automatically.


def format_document_text(title: str, body: str) -> str:
    """Wrap document text in the asymmetric retrieval format. Pair with media
    or use stand-alone via generate_text_embedding() (but that one is for the
    query side; for text-only documents call _embed() directly)."""
    return f"title: {title} | text: {body}"


def format_query(query: str) -> str:
    """Wrap a search query in the asymmetric retrieval format."""
    return f"task: search result | query: {query}"

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    """Lazily-cached Gemini client. Reads GEMINI_API_KEY from environment."""
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    _client = genai.Client(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Image helpers (kept from the original module, unchanged behaviour).
# ---------------------------------------------------------------------------


@dataclass
class _MediaBytes:
    data: bytes
    mime_type: str


def _ensure_jpeg(raw: bytes, mime: str) -> _MediaBytes:
    """Convert webp/tiff/bmp/avif → jpeg."""
    if mime.lower() not in UNSUPPORTED_IMAGE_MIME:
        return _MediaBytes(raw, mime)
    img = Image.open(io.BytesIO(raw))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return _MediaBytes(buf.getvalue(), "image/jpeg")


def _load_image(source: str) -> _MediaBytes:
    """Load image bytes from a URL, local path, or `data:` URL."""
    if source.startswith("data:"):
        return _parse_data_url(source)
    if source.startswith(("http://", "https://")):
        resp = requests.get(source, timeout=30)
        resp.raise_for_status()
        mime = resp.headers.get("content-type", "image/jpeg").split(";", 1)[0]
        return _ensure_jpeg(resp.content, mime)
    p = Path(source).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {source}")
    suffix = p.suffix.lower().lstrip(".")
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        "bmp": "image/bmp",
        "avif": "image/avif",
    }.get(suffix, "image/jpeg")
    return _ensure_jpeg(p.read_bytes(), mime)


def _parse_data_url(data_url: str) -> _MediaBytes:
    """Parse `data:image/...;base64,...` into raw bytes + mime."""
    if not data_url.startswith("data:"):
        raise ValueError("Not a data URL")
    header, _, b64 = data_url.partition(",")
    if not b64:
        raise ValueError("Malformed data URL")
    mime = header[5 : header.index(";")] if ";" in header else header[5:]
    return _ensure_jpeg(base64.b64decode(b64), mime)


# ---------------------------------------------------------------------------
# Video / audio helpers: localize remote sources, probe duration, chunk.
# ---------------------------------------------------------------------------


_VIDEO_MIME_BY_EXT = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "webm": "video/webm",
    "mkv": "video/x-matroska",
    "avi": "video/x-msvideo",
}
_AUDIO_MIME_BY_EXT = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
}


def _mime_from_path(path: Path, default: str) -> str:
    ext = path.suffix.lower().lstrip(".")
    return (
        _VIDEO_MIME_BY_EXT.get(ext)
        or _AUDIO_MIME_BY_EXT.get(ext)
        or default
    )


def _localize_to_tmpfile(source: str, suffix_hint: str = ".mp4") -> Path:
    """If source is a URL, download to a tmp file and return its Path.
    If source is a local path, return Path(source). Caller cleans up tmp files.
    """
    if source.startswith(("http://", "https://")):
        suffix = suffix_hint
        # Try to keep the original extension if the URL has one.
        url_path = source.split("?", 1)[0]
        if "." in os.path.basename(url_path):
            suffix = "." + os.path.basename(url_path).rsplit(".", 1)[-1]
        fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="gemini_media_")
        os.close(fd)
        with requests.get(source, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for buf in resp.iter_content(chunk_size=1 << 20):
                    if buf:
                        f.write(buf)
        return Path(tmp)
    p = Path(source).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Media not found: {source}")
    return p


def _require_ffmpeg() -> str:
    """Return the ffmpeg binary path or raise with a helpful error."""
    binary = shutil.which("ffmpeg")
    if not binary:
        raise RuntimeError(
            "ffmpeg not found on PATH — required for video chunking. "
            "Install via your package manager (e.g. `apt install ffmpeg` or `brew install ffmpeg`)."
        )
    return binary


def _probe_duration_seconds(path: Path) -> float:
    """Use ffprobe (ships with ffmpeg) to read media duration."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found on PATH (comes with ffmpeg).")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _chunk_video_file(
    path: Path,
    *,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[tuple[float, float, Path]]:
    """Split a video into overlapping chunks via ffmpeg. Returns
    [(start_s, end_s, chunk_path), ...]. Chunks are written to a tmp dir
    the caller is responsible for cleaning up by removing the parent dir
    of the first chunk."""
    ffmpeg = _require_ffmpeg()
    duration = _probe_duration_seconds(path)
    if duration <= chunk_seconds:
        return [(0.0, duration, path)]

    out_dir = Path(tempfile.mkdtemp(prefix="gemini_chunks_"))
    chunks: list[tuple[float, float, Path]] = []
    step = max(chunk_seconds - overlap_seconds, 1.0)
    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        out_path = out_dir / f"chunk_{idx:04d}.mp4"
        # -c copy is fast but can land on non-keyframe boundaries; for embedding
        # purposes that's fine. If you need frame-accurate cuts, drop -c copy
        # and let ffmpeg re-encode (slower).
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(path),
                "-t",
                f"{end - start:.3f}",
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                str(out_path),
            ],
            check=True,
        )
        chunks.append((start, end, out_path))
        idx += 1
        if end >= duration:
            break
        start += step
    return chunks


def _upload_to_files_api(path: Path, mime_type: str):
    """Upload a local file via the Files API and wait until it's ACTIVE.

    Gemini stores the file for 48h and lets us reference it by URI in
    embedding calls — required for anything over the inline payload cap.
    """
    client = _get_client()
    uploaded = client.files.upload(
        file=str(path),
        config=genai_types.UploadFileConfig(mime_type=mime_type),
    )
    # Poll for ACTIVE — videos in particular take a few seconds to process.
    deadline = time.time() + 300  # 5 min cap
    while True:
        state = getattr(getattr(uploaded, "state", None), "name", None)
        if state == "ACTIVE":
            return uploaded
        if state == "FAILED":
            raise RuntimeError(f"Files API processing failed for {path.name}")
        if time.time() > deadline:
            raise TimeoutError(f"Files API never reached ACTIVE for {path.name}")
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)


def _media_part_for(path: Path, mime_type: str) -> genai_types.Part:
    """Build a Part for a local media file, choosing inline vs Files API
    based on size."""
    size = path.stat().st_size
    if size < INLINE_SIZE_THRESHOLD_BYTES:
        return genai_types.Part(
            inline_data=genai_types.Blob(
                mime_type=mime_type, data=path.read_bytes()
            )
        )
    uploaded = _upload_to_files_api(path, mime_type)
    return genai_types.Part.from_uri(
        file_uri=uploaded.uri, mime_type=uploaded.mime_type
    )


# ---------------------------------------------------------------------------
# Core embed call.
# ---------------------------------------------------------------------------


def _embed(
    parts: list,
    *,
    output_dimensionality: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> list[float]:
    """Single-call wrapper around client.models.embed_content.

    NOTE: no task_type config — gemini-embedding-2 expects the task to be
    expressed in the prompt via format_document_text() / format_query()."""
    client = _get_client()
    if len(parts) == 1 and isinstance(parts[0], str):
        contents = parts[0]
    else:
        contents = genai_types.Content(parts=parts)
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=contents,
        config=genai_types.EmbedContentConfig(
            output_dimensionality=output_dimensionality,
        ),
    )
    if not result.embeddings or not result.embeddings[0].values:
        raise RuntimeError("No embedding returned from model")
    return list(result.embeddings[0].values)


# ---------------------------------------------------------------------------
# Public embedding functions
# ---------------------------------------------------------------------------


def generate_image_embedding(
    source: str,
    *,
    output_dimensionality: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> list[float]:
    """Embed an image given its URL, local path, or `data:` URL. Same call
    for query- or document-side use (image-only has no text instruction)."""
    img = _load_image(source)
    part = genai_types.Part(
        inline_data=genai_types.Blob(mime_type=img.mime_type, data=img.data)
    )
    return _embed([part], output_dimensionality=output_dimensionality)


def generate_text_embedding(
    text: str,
    *,
    output_dimensionality: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> list[float]:
    """Embed a piece of text as a SEARCH QUERY.

    Auto-wraps with format_query(text) → 'task: search result | query: {text}',
    which is the documented contract for the query side of asymmetric retrieval
    against gemini-embedding-2. For embedding a text *document* (no media),
    pre-format with format_document_text(title, body) and call _embed() directly."""
    if not text:
        raise ValueError("text must be non-empty")
    return _embed([format_query(text)], output_dimensionality=output_dimensionality)


def generate_multimodal_embedding(
    image_source: str,
    text: str,
    *,
    output_dimensionality: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> list[float]:
    """Embed an image + text pair as a single vector.

    `text` is passed through verbatim — caller is responsible for wrapping with
    format_document_text(title, body) for the document side or format_query(q)
    for the query side."""
    if not text:
        raise ValueError("text must be non-empty for multimodal embedding")
    img = _load_image(image_source)
    parts = [
        genai_types.Part(
            inline_data=genai_types.Blob(mime_type=img.mime_type, data=img.data)
        ),
        genai_types.Part(text=text),
    ]
    return _embed(parts, output_dimensionality=output_dimensionality)


def generate_video_embedding(
    source: str,
    *,
    text: Optional[str] = None,
    output_dimensionality: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> list[float]:
    """Embed a video (≤120s) given a URL or local path.

    If the source is longer than the model's 120s limit, this will raise —
    use chunk_and_embed_video() instead. Optional `text` is concatenated into
    the same vector and is passed through verbatim (caller wraps with
    format_document_text() or format_query() as appropriate).
    """
    path = _localize_to_tmpfile(source, suffix_hint=".mp4")
    cleanup = path if source.startswith(("http://", "https://")) else None
    try:
        duration = _probe_duration_seconds(path)
        if duration > MAX_VIDEO_SECONDS + 0.5:
            raise ValueError(
                f"Video is {duration:.1f}s — exceeds the {MAX_VIDEO_SECONDS:.0f}s "
                "limit for a single embedding call. Use chunk_and_embed_video()."
            )
        mime = _mime_from_path(path, "video/mp4")
        parts: list = [_media_part_for(path, mime)]
        if text:
            parts.append(genai_types.Part(text=text))
        return _embed(parts, output_dimensionality=output_dimensionality)
    finally:
        if cleanup and cleanup.exists():
            cleanup.unlink(missing_ok=True)


def generate_audio_embedding(
    source: str,
    *,
    text: Optional[str] = None,
    output_dimensionality: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> list[float]:
    """Embed an audio clip (≤80s) given a URL or local path. Optional `text`
    is passed through verbatim — caller wraps."""
    path = _localize_to_tmpfile(source, suffix_hint=".mp3")
    cleanup = path if source.startswith(("http://", "https://")) else None
    try:
        duration = _probe_duration_seconds(path)
        if duration > MAX_AUDIO_SECONDS + 0.5:
            raise ValueError(
                f"Audio is {duration:.1f}s — exceeds the {MAX_AUDIO_SECONDS:.0f}s "
                "limit. Chunk it first."
            )
        mime = _mime_from_path(path, "audio/mpeg")
        parts: list = [_media_part_for(path, mime)]
        if text:
            parts.append(genai_types.Part(text=text))
        return _embed(parts, output_dimensionality=output_dimensionality)
    finally:
        if cleanup and cleanup.exists():
            cleanup.unlink(missing_ok=True)


@dataclass
class VideoChunkEmbedding:
    start_s: float
    end_s: float
    vector: list[float]


def chunk_and_embed_video(
    source: str,
    *,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
    text: Optional[str] = None,
    output_dimensionality: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> list[VideoChunkEmbedding]:
    """Split a video into ≤120s chunks and embed each separately.

    Returns one VideoChunkEmbedding per chunk with absolute (start_s, end_s)
    timestamps in the original source. Optional `text` is attached to every
    chunk's embedding (useful when you have a single caption/title for the
    whole video).

    Chunks shorter than the input video are written to a tmp dir which is
    cleaned up before returning. The source itself (if a URL) is also cleaned
    up.
    """
    if chunk_seconds > MAX_VIDEO_SECONDS:
        raise ValueError(
            f"chunk_seconds={chunk_seconds} exceeds model limit of {MAX_VIDEO_SECONDS}"
        )
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be less than chunk_seconds")

    src_path = _localize_to_tmpfile(source, suffix_hint=".mp4")
    src_is_remote = source.startswith(("http://", "https://"))
    chunk_dir: Optional[Path] = None
    out: list[VideoChunkEmbedding] = []
    try:
        chunks = _chunk_video_file(
            src_path,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
        )
        # If only one chunk and it IS the original file, no tmp dir was created.
        if len(chunks) > 1 or chunks[0][2] != src_path:
            chunk_dir = chunks[0][2].parent

        for start_s, end_s, chunk_path in chunks:
            mime = _mime_from_path(chunk_path, "video/mp4")
            parts: list = [_media_part_for(chunk_path, mime)]
            if text:
                parts.append(genai_types.Part(text=text))
            vec = _embed(
                parts,
                output_dimensionality=output_dimensionality,
            )
            out.append(VideoChunkEmbedding(start_s, end_s, vec))
        return out
    finally:
        if chunk_dir and chunk_dir.exists():
            shutil.rmtree(chunk_dir, ignore_errors=True)
        if src_is_remote and src_path.exists():
            src_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Similarity + in-memory retrieval / clustering (unchanged from original).
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 if either vector is zero."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = mag_a = mag_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        mag_a += x * x
        mag_b += y * y
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (math.sqrt(mag_a) * math.sqrt(mag_b))


def search_vectors(
    query: list[float],
    items: list[dict],
    *,
    top_k: int = 10,
    threshold: float = 0.0,
    exclude_id: Optional[str] = None,
) -> list[dict]:
    """Top-k cosine search over an in-memory list of items.

    Each item must have `id` and `vec` keys. Returned items have an added
    `similarity` key, sorted descending. Filtered by min similarity and
    optionally an id to exclude.
    """
    scored: list[tuple[float, dict]] = []
    for item in items:
        if exclude_id is not None and item.get("id") == exclude_id:
            continue
        sim = cosine_similarity(query, item["vec"])
        if sim >= threshold:
            scored.append((sim, item))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [{**item, "similarity": sim} for sim, item in scored[:top_k]]


def cluster_vectors(
    items: list[dict],
    *,
    threshold: float = 0.7,
    min_size: int = 3,
) -> list[dict]:
    """Greedy in-memory clustering."""
    if len(items) < min_size:
        return []

    assigned: set = set()
    clusters: list[dict] = []

    for seed in items:
        if seed["id"] in assigned:
            continue
        members: list[dict] = []
        for cand in items:
            if cand["id"] in assigned:
                continue
            sim = cosine_similarity(seed["vec"], cand["vec"])
            if sim >= threshold:
                members.append({**cand, "similarity": sim})
        if len(members) >= min_size:
            members.sort(key=lambda m: m["similarity"], reverse=True)
            for m in members:
                assigned.add(m["id"])
            clusters.append({"centroid_id": seed["id"], "members": members})

    clusters.sort(key=lambda c: len(c["members"]), reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Storage — Postgres + pgvector.
# ---------------------------------------------------------------------------


def _get_conn():
    """Open a fresh psycopg connection. Caller is responsible for closing it."""
    try:
        import psycopg  # type: ignore
        from pgvector.psycopg import register_vector  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            'Storage helpers require: pip install "psycopg[binary]" pgvector'
        ) from e

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    conn = psycopg.connect(dsn)
    register_vector(conn)
    return conn


def _validate_dim(embedding: list[float]) -> None:
    """Allow any of the recommended Gemini dimensions (128–3072). We don't
    pin to 768 anymore now that video may want 1536."""
    if not (128 <= len(embedding) <= 3072):
        raise ValueError(
            f"embedding has unexpected dimensionality: {len(embedding)} "
            "(expected 128–3072)"
        )


def store_embedding(
    *,
    source_type: SourceType,
    source_id: str,
    user_id: str,
    embedding: list[float],
    organization_id: Optional[str] = None,
    image_url: Optional[str] = None,
    text: Optional[str] = None,
    start_seconds: Optional[float] = None,
    end_seconds: Optional[float] = None,
    parent_id: Optional[str] = None,
) -> None:
    """Upsert an embedding into the `embeddings` table.

    For images/galleries: pass image_url and/or text, leave start/end/parent
    None. For video/audio chunks: pass start_seconds, end_seconds, parent_id
    (the source media id). `image_url` is reused as the media URL for video
    and audio rows to avoid a schema rename — see embedded_media_url alias.
    """
    _validate_dim(embedding)
    sql = """
        INSERT INTO embeddings
            (source_type, source_id, user_id, organization_id,
             embedding, embedded_image_url, embedded_text,
             start_seconds, end_seconds, parent_id)
        VALUES (%s, %s, %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s)
        ON CONFLICT ON CONSTRAINT uq_embeddings_source
        DO UPDATE SET
            embedding = EXCLUDED.embedding,
            embedded_image_url = EXCLUDED.embedded_image_url,
            embedded_text = EXCLUDED.embedded_text,
            start_seconds = EXCLUDED.start_seconds,
            end_seconds = EXCLUDED.end_seconds,
            parent_id = EXCLUDED.parent_id,
            updated_at = NOW()
    """
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            sql,
            (
                source_type,
                source_id,
                user_id,
                organization_id,
                embedding,
                image_url,
                text,
                start_seconds,
                end_seconds,
                parent_id,
            ),
        )


def store_video_chunk_embedding(
    *,
    video_id: str,
    start_seconds: float,
    end_seconds: float,
    user_id: str,
    embedding: list[float],
    video_url: Optional[str] = None,
    text: Optional[str] = None,
    organization_id: Optional[str] = None,
) -> str:
    """Convenience wrapper around store_embedding for video chunks.

    Constructs a deterministic source_id of the form
        {video_id}:{start_ms}-{end_ms}
    so repeated runs upsert into the same row.

    Returns the source_id that was used.
    """
    start_ms = int(round(start_seconds * 1000))
    end_ms = int(round(end_seconds * 1000))
    source_id = f"{video_id}:{start_ms}-{end_ms}"
    store_embedding(
        source_type="video_chunk",
        source_id=source_id,
        user_id=user_id,
        organization_id=organization_id,
        embedding=embedding,
        image_url=video_url,
        text=text,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        parent_id=video_id,
    )
    return source_id


def store_audio_clip_embedding(
    *,
    audio_id: str,
    user_id: str,
    embedding: list[float],
    start_seconds: float = 0.0,
    end_seconds: Optional[float] = None,
    audio_url: Optional[str] = None,
    text: Optional[str] = None,
    organization_id: Optional[str] = None,
) -> str:
    """Same pattern as store_video_chunk_embedding, for audio."""
    start_ms = int(round(start_seconds * 1000))
    end_ms = int(round((end_seconds or start_seconds) * 1000))
    source_id = f"{audio_id}:{start_ms}-{end_ms}"
    store_embedding(
        source_type="audio_clip",
        source_id=source_id,
        user_id=user_id,
        organization_id=organization_id,
        embedding=embedding,
        image_url=audio_url,
        text=text,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        parent_id=audio_id,
    )
    return source_id


def delete_embedding(source_type: SourceType, source_id: str) -> None:
    """Remove an embedding row by (source_type, source_id)."""
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM embeddings WHERE source_type = %s AND source_id = %s",
            (source_type, source_id),
        )


def delete_video_chunks(video_id: str) -> int:
    """Delete every chunk row for a given parent video. Returns row count."""
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM embeddings WHERE source_type = 'video_chunk' "
            "AND parent_id = %s",
            (video_id,),
        )
        return cur.rowcount


def has_embedding(source_type: SourceType, source_id: str) -> bool:
    """Return True if a row already exists for (source_type, source_id)."""
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM embeddings WHERE source_type = %s AND source_id = %s LIMIT 1",
            (source_type, source_id),
        )
        return cur.fetchone() is not None


def has_video_chunks(video_id: str) -> bool:
    """Return True if any chunk rows exist for a parent video id."""
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM embeddings WHERE source_type = 'video_chunk' "
            "AND parent_id = %s LIMIT 1",
            (video_id,),
        )
        return cur.fetchone() is not None


@dataclass
class SimilarityHit:
    source_type: str
    source_id: str
    similarity: float
    embedded_image_url: Optional[str]
    embedded_text: Optional[str]
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    parent_id: Optional[str] = None


def search_similar(
    embedding: list[float],
    *,
    user_id: str,
    organization_id: Optional[str] = None,
    source_type: Optional[SourceType] = None,
    limit: int = 20,
    exclude_source_id: Optional[str] = None,
    threshold: float = 0.30,
) -> list[SimilarityHit]:
    """Cosine-similarity search via pgvector's `<=>` operator."""
    _validate_dim(embedding)

    where = ["(e.user_id = %s::uuid"]
    if organization_id:
        where[0] += " OR e.organization_id = %s::uuid"
    where[0] += ")"

    if source_type:
        where.append("e.source_type = %s")
    if exclude_source_id:
        where.append("e.source_id != %s")

    where.append("(e.embedding <=> %s) <= %s")

    sql = f"""
        SELECT e.source_type, e.source_id, e.embedded_image_url, e.embedded_text,
               e.start_seconds, e.end_seconds, e.parent_id,
               1 - (e.embedding <=> %s) AS similarity
        FROM embeddings e
        WHERE {' AND '.join(where)}
        ORDER BY e.embedding <=> %s
        LIMIT %s
    """

    # Param ordering follows the SQL placeholder order top-to-bottom:
    # SELECT vec, then WHERE (user_id, [org], [source_type], [exclude], vec, threshold),
    # then ORDER BY vec, then LIMIT.
    ordered: list = [embedding, user_id]
    if organization_id:
        ordered.append(organization_id)
    if source_type:
        ordered.append(source_type)
    if exclude_source_id:
        ordered.append(exclude_source_id)
    ordered.extend([embedding, 1 - threshold, embedding, limit])

    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, ordered)
        return [
            SimilarityHit(
                source_type=row[0],
                source_id=row[1],
                embedded_image_url=row[2],
                embedded_text=row[3],
                start_seconds=(float(row[4]) if row[4] is not None else None),
                end_seconds=(float(row[5]) if row[5] is not None else None),
                parent_id=row[6],
                similarity=float(row[7]),
            )
            for row in cur.fetchall()
        ]


@dataclass
class Cluster:
    centroid_id: str
    members: list[SimilarityHit]


def find_clusters(
    *,
    user_id: str,
    organization_id: Optional[str] = None,
    source_type: Optional[SourceType] = None,
    similarity_threshold: float = 0.7,
    min_cluster_size: int = 3,
) -> list[Cluster]:
    """Greedy in-memory clustering pulled from DB rows."""
    where = ["(e.user_id = %s::uuid"]
    params: list = [user_id]
    if organization_id:
        where[0] += " OR e.organization_id = %s::uuid"
        params.append(organization_id)
    where[0] += ")"
    if source_type:
        where.append("e.source_type = %s")
        params.append(source_type)

    sql = f"""
        SELECT e.source_id, e.source_type, e.embedding,
               e.embedded_image_url, e.embedded_text,
               e.start_seconds, e.end_seconds, e.parent_id
        FROM embeddings e
        WHERE {' AND '.join(where)}
    """

    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if len(rows) < min_cluster_size:
        return []

    items = [
        {
            "id": r[0],
            "source_type": r[1],
            "vec": list(r[2]),
            "image_url": r[3],
            "text": r[4],
            "start_seconds": r[5],
            "end_seconds": r[6],
            "parent_id": r[7],
        }
        for r in rows
    ]

    raw = cluster_vectors(
        items, threshold=similarity_threshold, min_size=min_cluster_size
    )
    return [
        Cluster(
            centroid_id=c["centroid_id"],
            members=[
                SimilarityHit(
                    source_type=m["source_type"],
                    source_id=m["id"],
                    similarity=m["similarity"],
                    embedded_image_url=m["image_url"],
                    embedded_text=m["text"],
                    start_seconds=(
                        float(m["start_seconds"])
                        if m["start_seconds"] is not None
                        else None
                    ),
                    end_seconds=(
                        float(m["end_seconds"])
                        if m["end_seconds"] is not None
                        else None
                    ),
                    parent_id=m["parent_id"],
                )
                for m in c["members"]
            ],
        )
        for c in raw
    ]


# ---------------------------------------------------------------------------
# Schema migration text (printed by `python embeddings_v2.py schema`).
# ---------------------------------------------------------------------------

SCHEMA_MIGRATION_SQL = """\
-- Add video/audio timestamp + parent columns to the existing embeddings table.
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS start_seconds REAL;
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS end_seconds   REAL;
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS parent_id     TEXT;

-- Lookup all chunks of a given parent video efficiently.
CREATE INDEX IF NOT EXISTS idx_embeddings_parent_id
    ON embeddings (parent_id) WHERE parent_id IS NOT NULL;

-- If your source_type column has a CHECK constraint, broaden it:
-- ALTER TABLE embeddings DROP CONSTRAINT IF EXISTS embeddings_source_type_check;
-- ALTER TABLE embeddings ADD CONSTRAINT embeddings_source_type_check
--   CHECK (source_type IN ('asset', 'gallery', 'video_chunk', 'audio_clip', 'pdf_page'));
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Gemini multimodal embeddings (text/image/video/audio)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_text = sub.add_parser("text", help="Embed a text string.")
    p_text.add_argument("text")

    p_img = sub.add_parser("image", help="Embed an image (URL, path, or data URL).")
    p_img.add_argument("source")

    p_mm = sub.add_parser("multimodal", help="Embed image + text together.")
    p_mm.add_argument("source")
    p_mm.add_argument("text")

    p_vid = sub.add_parser("video", help="Embed a single video (≤120s).")
    p_vid.add_argument("source")
    p_vid.add_argument("--text", default=None)

    p_chunks = sub.add_parser(
        "video-chunks", help="Chunk a long video and embed each chunk."
    )
    p_chunks.add_argument("source")
    p_chunks.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    p_chunks.add_argument("--overlap-seconds", type=float,
                          default=DEFAULT_CHUNK_OVERLAP_SECONDS)
    p_chunks.add_argument("--text", default=None)

    p_aud = sub.add_parser("audio", help="Embed an audio clip (≤80s).")
    p_aud.add_argument("source")
    p_aud.add_argument("--text", default=None)

    p_sim = sub.add_parser(
        "similarity",
        help="Cosine similarity between two embeddings (JSON arrays).",
    )
    p_sim.add_argument("a")
    p_sim.add_argument("b")

    p_store = sub.add_parser("store", help="Embed and upsert into Postgres.")
    p_store.add_argument("source_type",
                         choices=("asset", "gallery", "video_chunk", "audio_clip"))
    p_store.add_argument("source_id")
    p_store.add_argument("user_id")
    p_store.add_argument("source", help="Media URL, path, or data URL.")
    p_store.add_argument("--text", default=None)
    p_store.add_argument("--image-url", default=None, dest="image_url")
    p_store.add_argument("--organization-id", default=None, dest="organization_id")

    p_ingest = sub.add_parser(
        "ingest-video",
        help="Chunk a video, embed every chunk, and upsert all rows.",
    )
    p_ingest.add_argument("video_id", help="Stable identifier for the source video.")
    p_ingest.add_argument("source", help="Video URL or local path.")
    p_ingest.add_argument("user_id")
    p_ingest.add_argument("--video-url", default=None,
                          help="Public URL to store alongside the embedding.")
    p_ingest.add_argument("--text", default=None,
                          help="Optional caption/title attached to every chunk.")
    p_ingest.add_argument("--organization-id", default=None, dest="organization_id")
    p_ingest.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    p_ingest.add_argument("--overlap-seconds", type=float,
                          default=DEFAULT_CHUNK_OVERLAP_SECONDS)

    p_search = sub.add_parser("search", help="Find similar items in Postgres.")
    p_search.add_argument("user_id")
    p_search.add_argument("--image", default=None)
    p_search.add_argument("--text", default=None)
    p_search.add_argument("--source-type", default=None,
                          choices=("asset", "gallery", "video_chunk", "audio_clip"))
    p_search.add_argument("--organization-id", default=None)
    p_search.add_argument("--limit", type=int, default=20)
    p_search.add_argument("--threshold", type=float, default=0.30)
    p_search.add_argument("--exclude-source-id", default=None)

    p_del = sub.add_parser("delete", help="Remove an embedding row.")
    p_del.add_argument("source_type",
                       choices=("asset", "gallery", "video_chunk", "audio_clip"))
    p_del.add_argument("source_id")

    p_del_vid = sub.add_parser(
        "delete-video", help="Remove every chunk row for a parent video_id."
    )
    p_del_vid.add_argument("video_id")

    p_has = sub.add_parser("has", help="Check if an embedding exists.")
    p_has.add_argument("source_type",
                       choices=("asset", "gallery", "video_chunk", "audio_clip"))
    p_has.add_argument("source_id")

    p_clus = sub.add_parser("clusters", help="Greedy clustering for the user/org.")
    p_clus.add_argument("user_id")
    p_clus.add_argument("--source-type", default=None,
                        choices=("asset", "gallery", "video_chunk", "audio_clip"))
    p_clus.add_argument("--organization-id", default=None)
    p_clus.add_argument("--threshold", type=float, default=0.7)
    p_clus.add_argument("--min-size", type=int, default=3)

    sub.add_parser("schema", help="Print SQL migration to extend the embeddings table.")

    args = parser.parse_args()

    def _print_vec(vec: list[float]) -> None:
        print(json.dumps(vec))

    if args.cmd == "text":
        _print_vec(generate_text_embedding(args.text))
        return 0
    if args.cmd == "image":
        _print_vec(generate_image_embedding(args.source))
        return 0
    if args.cmd == "multimodal":
        _print_vec(generate_multimodal_embedding(args.source, args.text))
        return 0
    if args.cmd == "video":
        _print_vec(generate_video_embedding(args.source, text=args.text))
        return 0
    if args.cmd == "video-chunks":
        chunks = chunk_and_embed_video(
            args.source,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            text=args.text,
        )
        print(json.dumps(
            [{"start_s": c.start_s, "end_s": c.end_s, "vector": c.vector}
             for c in chunks],
            indent=2,
        ))
        return 0
    if args.cmd == "audio":
        _print_vec(generate_audio_embedding(args.source, text=args.text))
        return 0
    if args.cmd == "similarity":
        a = json.loads(Path(args.a).read_text() if Path(args.a).is_file() else args.a)
        b = json.loads(Path(args.b).read_text() if Path(args.b).is_file() else args.b)
        print(cosine_similarity(a, b))
        return 0

    if args.cmd == "store":
        # Route to the appropriate embedder by source_type.
        if args.source_type == "video_chunk":
            vec = generate_video_embedding(args.source, text=args.text)
        elif args.source_type == "audio_clip":
            vec = generate_audio_embedding(args.source, text=args.text)
        elif args.text:
            vec = generate_multimodal_embedding(args.source, args.text)
        else:
            vec = generate_image_embedding(args.source)
        store_embedding(
            source_type=args.source_type,
            source_id=args.source_id,
            user_id=args.user_id,
            organization_id=args.organization_id,
            embedding=vec,
            image_url=args.image_url,
            text=args.text,
        )
        print(json.dumps({"stored": True, "dim": len(vec)}))
        return 0

    if args.cmd == "ingest-video":
        chunks = chunk_and_embed_video(
            args.source,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            text=args.text,
        )
        ids: list[str] = []
        for ch in chunks:
            sid = store_video_chunk_embedding(
                video_id=args.video_id,
                start_seconds=ch.start_s,
                end_seconds=ch.end_s,
                user_id=args.user_id,
                organization_id=args.organization_id,
                embedding=ch.vector,
                video_url=args.video_url,
                text=args.text,
            )
            ids.append(sid)
        print(json.dumps({"video_id": args.video_id, "chunks": len(ids), "ids": ids}))
        return 0

    if args.cmd == "search":
        if not args.image and not args.text:
            print("error: provide --image or --text", file=sys.stderr)
            return 2
        if args.image and args.text:
            vec = generate_multimodal_embedding(args.image, args.text)
        elif args.image:
            vec = generate_image_embedding(args.image)
        else:
            vec = generate_text_embedding(args.text)
        hits = search_similar(
            vec,
            user_id=args.user_id,
            organization_id=args.organization_id,
            source_type=args.source_type,
            limit=args.limit,
            exclude_source_id=args.exclude_source_id,
            threshold=args.threshold,
        )
        print(json.dumps([h.__dict__ for h in hits], indent=2))
        return 0

    if args.cmd == "delete":
        delete_embedding(args.source_type, args.source_id)
        print(json.dumps({"deleted": True}))
        return 0

    if args.cmd == "delete-video":
        n = delete_video_chunks(args.video_id)
        print(json.dumps({"deleted_chunks": n}))
        return 0

    if args.cmd == "has":
        print(json.dumps({"exists": has_embedding(args.source_type, args.source_id)}))
        return 0

    if args.cmd == "clusters":
        clusters = find_clusters(
            user_id=args.user_id,
            organization_id=args.organization_id,
            source_type=args.source_type,
            similarity_threshold=args.threshold,
            min_cluster_size=args.min_size,
        )
        out = [
            {
                "centroid_id": c.centroid_id,
                "size": len(c.members),
                "members": [m.__dict__ for m in c.members],
            }
            for c in clusters
        ]
        print(json.dumps(out, indent=2))
        return 0

    if args.cmd == "schema":
        print(SCHEMA_MIGRATION_SQL)
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(_cli())
