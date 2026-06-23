"""
Self-contained multimodal embedding helper — Python port of utils/vertex-embeddings.ts.

Generates embeddings via Google Gemini (gemini-embedding-2-preview, 768-dim).
Mirrors the image / text / multimodal entry points used by app/api/embeddings/.
Storage / pgvector search / clustering are intentionally NOT included — this
module is pure embedding generation. Drop into any Python project.

Setup:
    pip install google-genai pillow requests
    export GEMINI_API_KEY=...

Usage:
    from embeddings import (
        generate_image_embedding,
        generate_text_embedding,
        generate_multimodal_embedding,
        cosine_similarity,
    )

    vec_img = generate_image_embedding("https://example.com/chair.jpg")
    vec_txt = generate_text_embedding("modern walnut dining chair")
    vec_mm  = generate_multimodal_embedding(
        "https://example.com/chair.jpg", "modern walnut dining chair"
    )
    print(cosine_similarity(vec_img, vec_mm))

CLI:
    python embeddings.py text "modern walnut dining chair"
    python embeddings.py image https://example.com/chair.jpg
    python embeddings.py image ./local/chair.jpg
    python embeddings.py multimodal ./local/chair.jpg "modern walnut dining chair"
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import requests
from google import genai
from google.genai import types as genai_types
from PIL import Image

EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIMENSIONS = 768

# Pillow can decode these but Gemini may reject them — normalise to JPEG.
UNSUPPORTED_MIME = {"image/webp", "image/tiff", "image/bmp", "image/avif"}

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]

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


@dataclass
class _ImageBytes:
    data: bytes
    mime_type: str


def _ensure_jpeg(raw: bytes, mime: str) -> _ImageBytes:
    """Convert webp/tiff/bmp/avif → jpeg. Mirrors ensureJpeg() in the TS module."""
    if mime.lower() not in UNSUPPORTED_MIME:
        return _ImageBytes(raw, mime)
    img = Image.open(io.BytesIO(raw))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return _ImageBytes(buf.getvalue(), "image/jpeg")


def _load_image(source: str) -> _ImageBytes:
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


def _parse_data_url(data_url: str) -> _ImageBytes:
    """Parse `data:image/...;base64,...` into raw bytes + mime."""
    if not data_url.startswith("data:"):
        raise ValueError("Not a data URL")
    header, _, b64 = data_url.partition(",")
    if not b64:
        raise ValueError("Malformed data URL")
    mime = header[5 : header.index(";")] if ";" in header else header[5:]
    return _ensure_jpeg(base64.b64decode(b64), mime)


def _embed(parts: list, task_type: TaskType) -> list[float]:
    """Single-call wrapper around client.models.embed_content."""
    client = _get_client()
    if len(parts) == 1 and isinstance(parts[0], str):
        contents = parts[0]
    else:
        contents = genai_types.Content(parts=parts)
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=contents,
        config=genai_types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS,
            task_type=task_type,
        ),
    )
    if not result.embeddings or not result.embeddings[0].values:
        raise RuntimeError("No embedding returned from model")
    return list(result.embeddings[0].values)


def generate_image_embedding(
    source: str, *, task_type: TaskType = "RETRIEVAL_DOCUMENT"
) -> list[float]:
    """Embed an image given its URL, local path, or `data:` URL."""
    img = _load_image(source)
    part = genai_types.Part(
        inline_data=genai_types.Blob(mime_type=img.mime_type, data=img.data)
    )
    return _embed([part], task_type)


def generate_text_embedding(
    text: str, *, task_type: TaskType = "RETRIEVAL_QUERY"
) -> list[float]:
    """Embed a piece of text. Default task_type matches the TS search path."""
    if not text:
        raise ValueError("text must be non-empty")
    return _embed([text], task_type)


def generate_multimodal_embedding(
    image_source: str,
    text: str,
    *,
    task_type: TaskType = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    """Embed an image + text pair as a single 768-dim vector."""
    if not text:
        raise ValueError("text must be non-empty for multimodal embedding")
    img = _load_image(image_source)
    parts = [
        genai_types.Part(
            inline_data=genai_types.Blob(mime_type=img.mime_type, data=img.data)
        ),
        genai_types.Part(text=text),
    ]
    return _embed(parts, task_type)


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


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Gemini multimodal embeddings."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_text = sub.add_parser("text", help="Embed a text string.")
    p_text.add_argument("text")

    p_img = sub.add_parser("image", help="Embed an image (URL, path, or data URL).")
    p_img.add_argument("source")
    p_img.add_argument(
        "--query",
        action="store_true",
        help="Use RETRIEVAL_QUERY task type (default is RETRIEVAL_DOCUMENT).",
    )

    p_mm = sub.add_parser("multimodal", help="Embed image + text together.")
    p_mm.add_argument("source")
    p_mm.add_argument("text")

    p_sim = sub.add_parser(
        "similarity",
        help="Compute cosine similarity between two embeddings (read JSON arrays from stdin or file).",
    )
    p_sim.add_argument("a")
    p_sim.add_argument("b")

    args = parser.parse_args()

    def _print_vec(vec: list[float]) -> None:
        print(json.dumps(vec))

    if args.cmd == "text":
        _print_vec(generate_text_embedding(args.text))
        return 0
    if args.cmd == "image":
        task: TaskType = "RETRIEVAL_QUERY" if args.query else "RETRIEVAL_DOCUMENT"
        _print_vec(generate_image_embedding(args.source, task_type=task))
        return 0
    if args.cmd == "multimodal":
        _print_vec(generate_multimodal_embedding(args.source, args.text))
        return 0
    if args.cmd == "similarity":
        a = json.loads(Path(args.a).read_text() if Path(args.a).is_file() else args.a)
        b = json.loads(Path(args.b).read_text() if Path(args.b).is_file() else args.b)
        print(cosine_similarity(a, b))
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(_cli())
