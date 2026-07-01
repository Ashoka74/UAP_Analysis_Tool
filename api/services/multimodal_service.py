"""Multimodal semantic search service — Gemini-768d text/image embeddings over
the Neon + pgvector ``embeddings`` archive (UAP Release media: DoD/DVIDS video
clips, NASA audio, source-document PDF pages).

Ported from the Streamlit ``rag_search.py`` ``_mm_*`` helpers; Streamlit caching
is replaced with a module-level connection cache. The DB URL + Gemini key are
supplied per request (never stored server-side), mirroring the Cohere RAG path.
The local-media-cache rendering does NOT port (it reads server filesystem
paths) — results carry DVIDS embed URLs / war.gov source links for the browser.
"""
from __future__ import annotations

import re
import threading
from typing import Any

_MM_USER_ID = "00000000-0000-0000-0000-000000000001"   # hardcoded single tenant
_MM_PAGE_RE = re.compile(r"^(.+):p(\d+)$")
_EMBED_MODEL = "gemini-embedding-2-preview"
_EMBED_DIM = 768

# One psycopg connection per DB URL, reused across requests. Neon auto-suspends
# idle DBs, so every query gets a one-shot reconnect retry (see search_pgvector).
_CONNS: dict[str, Any] = {}
_CONN_LOCK = threading.Lock()


def _conn(db_url: str):
    import psycopg

    with _CONN_LOCK:
        c = _CONNS.get(db_url)
        if c is None or getattr(c, "closed", False):
            c = psycopg.connect(db_url, prepare_threshold=None, autocommit=True, connect_timeout=15)
            _CONNS[db_url] = c
        return c


def _reset_conn(db_url: str):
    with _CONN_LOCK:
        c = _CONNS.pop(db_url, None)
    if c is not None:
        try:
            c.close()
        except Exception:
            pass


# ── Embeddings (Gemini 768-d) ───────────────────────────────────────────────
def embed_text(text: str, gemini_key: str) -> list[float]:
    """Embed a text query. Query-side asymmetric wrapping matters for ranking."""
    from google import genai
    from google.genai import types as gt

    client = genai.Client(api_key=gemini_key)
    r = client.models.embed_content(
        model=_EMBED_MODEL,
        contents=f"task: search result | query: {text}",
        config=gt.EmbedContentConfig(output_dimensionality=_EMBED_DIM),
    )
    return list(r.embeddings[0].values)


def embed_image(image_bytes: bytes, mime: str, gemini_key: str) -> list[float]:
    from google import genai
    from google.genai import types as gt

    client = genai.Client(api_key=gemini_key)
    r = client.models.embed_content(
        model=_EMBED_MODEL,
        contents=[gt.Part.from_bytes(data=image_bytes, mime_type=mime)],
        config=gt.EmbedContentConfig(output_dimensionality=_EMBED_DIM),
    )
    return list(r.embeddings[0].values)


def _page_number(source_id: str):
    m = _MM_PAGE_RE.match(source_id or "")
    return int(m.group(2)) if m else None


# ── pgvector search ─────────────────────────────────────────────────────────
def search_pgvector(db_url, vec, *, source_type=None, release=None, limit=20, threshold=0.30):
    """Cosine search over the embeddings table. The vector is serialised to
    text form '[a,b,…]' and cast ``%s::vector`` — psycopg3 does not auto-cast
    list[float] to pgvector."""
    import psycopg

    vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
    clauses = ["user_id = %s::uuid", "(embedding <=> %s::vector) <= %s"]
    params: list = [_MM_USER_ID, vec_str, 1 - threshold]
    if source_type:
        clauses.append("source_type = %s")
        params.append(source_type)
    if release:
        clauses.append("release = %s")
        params.append(release)
    sql = f"""
        SELECT source_type, source_id, parent_id, start_seconds, end_seconds,
               embedded_image_url, embedded_text, release, release_date,
               1 - (embedding <=> %s::vector) AS similarity
        FROM embeddings
        WHERE {' AND '.join(clauses)}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    ordered = [vec_str, *params, vec_str, limit]

    # One-shot retry: a Neon connection killed while idle (auto-suspend / blip)
    # is dropped and re-dialed before giving up.
    for attempt in (1, 2):
        try:
            conn = _conn(db_url)
            with conn.cursor() as cur:
                cur.execute(sql, ordered)
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
        except (psycopg.OperationalError, psycopg.InterfaceError):
            if attempt == 1:
                _reset_conn(db_url)
                continue
            raise


def releases(db_url: str) -> list[str]:
    """Distinct release tags present in the archive (for the UI filter)."""
    try:
        conn = _conn(db_url)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT release FROM embeddings WHERE release IS NOT NULL ORDER BY release"
            )
            return [r[0] for r in cur.fetchall()]
    except Exception:
        return []


def _collapse_by_parent(rows: list[dict], keep_types=("video_chunk", "audio_clip")) -> list[dict]:
    """Collapse near-duplicate chunks of the same A/V asset to one (best-scoring)
    result. Video/audio chunks share an essentially identical title-based
    embedding (~0.997 cosine), so without this one asset floods the top-K with
    repeated titles. PDF pages are real per-page content and are kept distinct.
    The kept row records how many segments matched in ``_chunk_matches``."""
    out: list[dict] = []
    idx: dict[str, int] = {}
    for r in rows:
        stype, parent = r.get("source_type"), r.get("parent_id")
        if stype in keep_types and parent:
            if parent in idx:
                kept = out[idx[parent]]
                kept["_chunk_matches"] = kept.get("_chunk_matches", 1) + 1
                continue
            idx[parent] = len(out)
            r = {**r, "_chunk_matches": 1}
        out.append(r)
    return out


def _enrich(rows: list[dict]) -> list[dict]:
    """Shape rows for the browser: page number, DVIDS embed URL (video/audio),
    and the war.gov / source link. (The Streamlit local-media cache is skipped.)"""
    out = []
    for r in rows:
        stype = r.get("source_type") or ""
        parent = r.get("parent_id") or ""
        asset_id = parent[len("dvids_"):] if parent.startswith("dvids_") else parent
        media_url = None
        if stype == "video_chunk" and asset_id:
            media_url = f"https://www.dvidshub.net/video/embed/{asset_id}"
        elif stype == "audio_clip" and asset_id:
            media_url = f"https://www.dvidshub.net/audio/embed/{asset_id}"
        start_s, end_s = r.get("start_seconds"), r.get("end_seconds")
        out.append({
            "source_type": stype,
            "parent_id": parent,
            "source_id": r.get("source_id"),
            "similarity": round(float(r["similarity"]), 4) if r.get("similarity") is not None else None,
            "start_seconds": float(start_s) if start_s is not None else None,
            "end_seconds": float(end_s) if end_s is not None else None,
            "page": _page_number(r.get("source_id") or "") if stype == "pdf_page" else None,
            "embedded_text": r.get("embedded_text"),
            "source_url": r.get("embedded_image_url") or "",
            "media_embed_url": media_url,
            "release": r.get("release"),
            "release_date": str(r.get("release_date")) if r.get("release_date") is not None else None,
            "chunk_matches": int(r.get("_chunk_matches", 1)),
        })
    return out


def _run(db_url, vec, *, source_type, release, limit, threshold, group_by_parent):
    # Over-fetch when collapsing so we still return ~limit distinct assets.
    fetch = min(limit * 5, 200) if group_by_parent else limit
    rows = search_pgvector(db_url, vec, source_type=source_type, release=release, limit=fetch, threshold=threshold)
    if group_by_parent:
        rows = _collapse_by_parent(rows)[:limit]
    return _enrich(rows)


def search_text(db_url, gemini_key, query, *, source_type=None, release=None,
                limit=12, threshold=0.20, group_by_parent=True):
    vec = embed_text(query, gemini_key)
    return _run(db_url, vec, source_type=source_type, release=release,
                limit=limit, threshold=threshold, group_by_parent=group_by_parent)


def search_image(db_url, gemini_key, image_bytes, mime, *, source_type=None, release=None,
                 limit=12, threshold=0.20, group_by_parent=True):
    vec = embed_image(image_bytes, mime, gemini_key)
    return _run(db_url, vec, source_type=source_type, release=release,
                limit=limit, threshold=threshold, group_by_parent=group_by_parent)
