# Handoff: UAP Embeddings → Streamlit Semantic Search

Everything you need to build a Streamlit app that does semantic search over the
UAP archive embeddings currently sitting in a Neon Postgres + pgvector database.

---

## 1. Context in one paragraph

A previous session embedded all of **UAP Release 2 (5/22/26)** — 49 DoD UAP
video clips and 7 NASA Apollo/Mercury audio recordings — into a Neon Postgres
database using **Google Gemini `gemini-embedding-2-preview`** (768-dim, cosine
similarity, indexed with HNSW). The pipeline lives in `embeddings_v2.py` at the
repo root. Your job is a Streamlit UI that lets users type a query (or upload an
image), embed it with the same model, and return ranked matches with playable
media.

---

## 2. What's in the database right now

```
source_type    rows    distinct assets
video_chunk     154    49 DVIDS UAP video clips (Release 2)
pdf_page        126     5 source documents (DOW-D017 [116p], DOE-D002 [4p],
                          CIA-D001 [3p], DOE-D001 [2p], DOE-D003 [1p])
audio_clip       27     7 NASA Apollo/Mercury audio recordings (Release 2)
TOTAL           307    61 assets    all release='PURSUE_2'  release_date=2026-05-22
```

- All current rows use `user_id = '00000000-0000-0000-0000-000000000001'` (a
  placeholder UUID — the schema is multi-tenant but this archive has one tenant).
- `parent_id` is `dvids_{asset_id}` for media rows (e.g. `dvids_1007706`); doc
  slugs like `dow-uap-d017` for `pdf_page` rows.
- `source_id` is `{parent_id}:{start_ms}-{end_ms}` for media chunks and
  `{parent_id}:p{NNNN}` for PDF pages (e.g. `dow-uap-d017:p0017`).
- Vector dimension is **768**. Queries must be 768-dim too.
- Every row carries the new `release` (`'PURSUE_2'`) and `release_date`
  (`2026-05-22`) columns — filter on these in the UI when more releases land.
- One pending video (`1007708`, the 513 MB outlier) was not ingested; it can be
  added later — not a blocker for the UI.
- Nothing from earlier releases (Release 1, NARA-CIA, FBI photos, etc.) is
  embedded yet. If you build the UI to filter on `release` / `parent_id`
  patterns or future source types, leave it open.

---

## 3. Schema reference

```sql
CREATE TABLE embeddings (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_type         TEXT        NOT NULL,   -- 'video_chunk' | 'audio_clip' | 'pdf_page' (more later)
    source_id           TEXT        NOT NULL,   -- '{parent_id}:{start_ms}-{end_ms}' for chunks; '{slug}:p{NNNN}' for pages
    user_id             UUID        NOT NULL,
    organization_id     UUID,
    embedding           VECTOR(768) NOT NULL,
    embedded_image_url  TEXT,                   -- video/audio: DVIDS page URL; pdf_page: whole-PDF war.gov URL
    embedded_text       TEXT,                   -- caption used during embed (Title + Blurb; or metadata + OCR for pdf_page)
    start_seconds       REAL,                   -- chunk start (NULL for pdf_page)
    end_seconds         REAL,                   -- chunk end   (NULL for pdf_page)
    parent_id           TEXT,                   -- 'dvids_1007706' for media; doc slug like 'dow-uap-d017' for pages
    release             TEXT        NOT NULL DEFAULT 'PURSUE_2',   -- campaign tag (filter on this in the UI)
    release_date        DATE        NOT NULL DEFAULT '2026-05-22', -- when the source documents were publicly released
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_embeddings_source UNIQUE (source_type, source_id)
);

-- Already created:
CREATE INDEX idx_embeddings_embedding ON embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_embeddings_parent_id ON embeddings (parent_id) WHERE parent_id IS NOT NULL;
CREATE INDEX idx_embeddings_user_id   ON embeddings (user_id);
```

Cosine search uses pgvector's `<=>` operator (distance, lower = closer).
Convert to similarity with `1 - (embedding <=> query)`.

---

## 4. Secrets — required, not in this file

Set as env vars (or Streamlit `secrets.toml`):

```bash
DATABASE_URL    = <Neon connection string, prefer the DIRECT endpoint over -pooler>
GEMINI_API_KEY  = <Google AI Studio key, same model that produced the rows>
```

The Neon string must include `?sslmode=require`. Ask the user to paste the
values from their Neon dashboard and Google AI Studio — they're not embedded
here on purpose. The previous session ran against a Neon project owned by the
user, and the password / key from that session should be considered exposed
and rotated.

**Streamlit secrets.toml** (recommended over raw env vars):

```toml
# .streamlit/secrets.toml  -- DO NOT COMMIT
DATABASE_URL = "postgresql://USER:PASSWORD@ep-xxxx.REGION.aws.neon.tech/neondb?sslmode=require"
GEMINI_API_KEY = "AIza..."
```

Read in app with `st.secrets["DATABASE_URL"]`.

---

## 5. Dependencies

```bash
pip install streamlit google-genai pillow requests "psycopg[binary]" pgvector
```

The only file from this repo you need to copy alongside the Streamlit app is
**`embeddings_v2.py`** (it's self-contained — no project-internal imports). Or
you can inline the few functions you actually use (see §6/§7 for the bare
minimum).

---

## 6. Embedding a user query

The model and dimension must match what's already in the DB
(`gemini-embedding-2-preview`, 768-d). **The contract is asymmetric and is
expressed in the prompt, not the config**: queries get a `task: search result
| query: …` prefix; documents go in as `title: … | text: …`. The
`EmbedContentConfig.task_type` field is *silently ignored* by
gemini-embedding-2 on the consumer API — don't set it. (Helper functions in
`embeddings_v2.py` apply the wrapping for you.)

```python
import embeddings_v2 as e

# Queries — generate_text_embedding auto-wraps with format_query().
vec_text  = e.generate_text_embedding("UAP over the Aegean")
vec_image = e.generate_image_embedding("./uploaded.jpg")   # image-only: no text instruction
vec_both  = e.generate_multimodal_embedding(
    "./uploaded.jpg",
    e.format_query("what is this"),                        # pre-wrap when there IS a text part
)
```

`embeddings_v2` also exports:

- `format_document_text(title, body)` → `"title: {title} | text: {body}"` (use when storing).
- `format_query(query)` → `"task: search result | query: {query}"` (use when querying with a text part attached to media).

Minimal inline version if you don't want to import `embeddings_v2`:

```python
import os
from google import genai
from google.genai import types as gt

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def embed_text(text: str) -> list[float]:
    r = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=f"task: search result | query: {text}",   # wrap, not task_type=
        config=gt.EmbedContentConfig(output_dimensionality=768),
    )
    return list(r.embeddings[0].values)
```

---

## 7. Searching with pgvector

`embeddings_v2.search_similar()` already does this and returns a list of
`SimilarityHit` dataclasses. If you want raw SQL:

```sql
SELECT source_type, source_id, parent_id, start_seconds, end_seconds,
       embedded_image_url, embedded_text,
       1 - (embedding <=> %s) AS similarity
FROM embeddings
WHERE user_id = %s::uuid
  AND (%s::text IS NULL OR source_type = %s)
  AND (embedding <=> %s) <= %s          -- distance <= 1 - threshold
ORDER BY embedding <=> %s
LIMIT %s;
```

Params, in order: `query_vec, user_id, source_type_or_null, source_type_or_null, query_vec, (1 - threshold), query_vec, limit`.

Don't forget `register_vector(conn)` from `pgvector.psycopg` after connecting —
without it psycopg can't bind `list[float]` to the `vector` type.

---

## 8. Result interpretation (per source_type)

### `video_chunk`
- `parent_id` → e.g. `dvids_1007706`. Strip the prefix to get the DVIDS asset id.
- `embedded_image_url` → the human DVIDS page, e.g. `https://www.dvidshub.net/video/1007706`.
- `start_seconds`, `end_seconds` → the chunk's offsets within the source video
  (one video typically has multiple chunks; show the timestamp to the user).
- `embedded_text` → the caption that was attached at embed time: the
  `Video Title` + `Description Blurb` from `uap-data_v2.csv`.
- DVIDS deep-link with timestamp: append `?t={int(start_seconds)}` to the page
  URL (or use the local file with `st.video(local_path, start_time=int(start_seconds))`).

### `audio_clip`
- Same `parent_id` shape but with audio DVIDS ids (1007870–1007879 range for
  Release 2).
- `embedded_image_url` is set even though the asset is audio (it's the DVIDS
  page URL — the column was reused as the canonical media URL for any kind).
- For long recordings (>80s — the model's audio input cap), the asset is
  segmented into ≤75s pieces; one row per piece with its own start/end.

### `pdf_page`
- `parent_id` is the doc slug (e.g. `dow-uap-d017`, `cia-uap-d001`,
  `doe-uap-d001`, `doe-uap-d002`, `doe-uap-d003`).
- `source_id` is `{parent_id}:p{NNNN}` with the page number zero-padded to
  4 digits (e.g. `dow-uap-d017:p0017`). Parse with a tiny regex to surface
  the page number in the UI.
- `embedded_image_url` is the whole-PDF URL on war.gov — there's no per-page
  URL on the source site, so deep-linking to a specific page means opening
  the PDF and scrolling.
- `embedded_text` is composed at embed time as: `{Agency} - {Title}` /
  `Date: ... Location: ...` / `Page N of M.` / `Document context: {blurb}` /
  `Page OCR: {ocr}`, capped at 8000 chars. The same string was paired with
  the rendered page image in the multimodal embed call.
- `start_seconds` / `end_seconds` are NULL.
- A rendered page image lives locally at
  `D:\divided\release_2\UAP_Release_2\pages\{slug}\page_NNNN.png` (150 dpi).
  Display it directly with `st.image(local_path)`; link to
  `embedded_image_url` to open the whole PDF on war.gov.

---

## 9. Where the media files live

The previous session saved every downloaded media file under the user's local
drive (set by them as the persistence target):

```
D:\divided\release_2\UAP_Release_2\
├── videos\dvids_{id}.mp4         (49 files, normalized originals from DVIDS)
├── audio\dvids_{id}.{ext}        (7 source MP4 wrappers + extracted .m4a tracks)
└── pages\{slug}\page_NNNN.png    (PDF page renders at 150 dpi, e.g.
                                   pages\dow-uap-d017\page_0017.png)
```

The page PNGs are generated by `ingest_pdf_pages.py` and are safe to delete and
re-generate from the source `release_2\{doc}\page_NNNN\page_NNNN.pdf` files.

This matters for the Streamlit UI:

- If the app runs on the same machine, you can pass the local path straight
  into `st.video(path, start_time=...)` / `st.audio(...)` — that's the smoothest
  playback experience and supports seeking.
- If the app runs elsewhere, link out to the DVIDS page (`embedded_image_url`).
  Direct CloudFront URLs work for download but seeking via HTTP from the
  browser is hit-or-miss.
- A third option: upload the local files to S3/R2/Vercel Blob and rewrite URLs.
  Not done.

If the file isn't found locally and the URL is the DVIDS page, **don't try to
embed the CloudFront MP4 directly in `st.video()`** — DVIDS' `/download/asset/`
endpoint is 403-gated, and the CloudFront URLs aren't stored in the DB. You'd
need to re-scrape the page (see the `scrape_media_url` helper in
`retry_release_2.py` if you want that pattern).

---

## 10. Minimal working Streamlit app

Drop this at `app.py` next to `embeddings_v2.py`, set the secrets, and run
`streamlit run app.py`. It covers text query, source-type filter, threshold
slider, and inline media playback with timestamp seeking.

```python
import os
import re
from pathlib import Path

import psycopg
import streamlit as st

import embeddings_v2 as e

USER_ID = "00000000-0000-0000-0000-000000000001"
MEDIA_ROOT = Path(r"D:\divided\release_2\UAP_Release_2")  # change if elsewhere
SOURCE_TYPES = ("video_chunk", "audio_clip", "pdf_page")

st.set_page_config(page_title="UAP Archive Semantic Search", layout="wide")

# --- bootstrap ---------------------------------------------------------------
for k in ("DATABASE_URL", "GEMINI_API_KEY"):
    if k in st.secrets:
        os.environ.setdefault(k, st.secrets[k])
    if not os.environ.get(k):
        st.error(f"Missing {k} — add it to .streamlit/secrets.toml")
        st.stop()

@st.cache_resource
def get_conn():
    return psycopg.connect(os.environ["DATABASE_URL"])

@st.cache_data(ttl=3600, show_spinner=False)
def embed_query_text(text: str) -> list[float]:
    # generate_text_embedding auto-wraps with format_query() and drops task_type.
    return e.generate_text_embedding(text)

@st.cache_data(ttl=3600, show_spinner=False)
def embed_query_image(image_bytes: bytes, mime: str) -> list[float]:
    import tempfile
    suffix = "." + mime.split("/", 1)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(image_bytes)
        path = f.name
    try:
        # image-only embed: same call for query and document side.
        return e.generate_image_embedding(path)
    finally:
        os.unlink(path)

def search(vec, *, source_type=None, release=None, limit=20, threshold=0.30):
    # pgvector's psycopg adapter doesn't auto-cast list[float] to vector ---
    # serialise to the textual '[a,b,c]' form and let Postgres cast.
    vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
    clauses = ["user_id = %s::uuid", "(embedding <=> %s::vector) <= %s"]
    params  = [USER_ID, vec_str, 1 - threshold]
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
    with get_conn().cursor() as cur:
        cur.execute(sql, ordered)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

_PAGE_RE = re.compile(r"^(.+):p(\d+)$")

def local_media_path(row: dict) -> Path | None:
    st_type = row["source_type"]
    if st_type == "video_chunk":
        asset_id = row["parent_id"].removeprefix("dvids_")
        p = MEDIA_ROOT / "videos" / f"dvids_{asset_id}.mp4"
        return p if p.exists() else None
    if st_type == "audio_clip":
        asset_id = row["parent_id"].removeprefix("dvids_")
        for ext in ("m4a", "mp3", "mp4", "wav", "aac", "ogg"):
            p = MEDIA_ROOT / "audio" / f"dvids_{asset_id}.{ext}"
            if p.exists():
                return p
        return None
    if st_type == "pdf_page":
        m = _PAGE_RE.match(row["source_id"])
        if not m:
            return None
        slug, page_num = m.group(1), int(m.group(2))
        p = MEDIA_ROOT / "pages" / slug / f"page_{page_num:04d}.png"
        return p if p.exists() else None
    return None

def page_number(row: dict) -> int | None:
    if row["source_type"] != "pdf_page":
        return None
    m = _PAGE_RE.match(row["source_id"])
    return int(m.group(2)) if m else None

# --- UI ----------------------------------------------------------------------
st.title("UAP Archive — Semantic Search")
st.caption("Gemini 768-d embeddings, cosine similarity over Neon + pgvector.")

with st.sidebar:
    mode = st.radio("Query type", ["Text", "Image"], horizontal=True)
    st_filter = st.selectbox("Source type", ["all", *SOURCE_TYPES])
    release_filter = st.selectbox("Release", ["all", "PURSUE_2"])
    threshold = st.slider("Min similarity", 0.0, 0.9, 0.30, 0.05)
    limit = st.slider("Max results", 5, 50, 20)

vec = None
if mode == "Text":
    q = st.text_input("Search query", placeholder="e.g. spherical UAP over water")
    if q:
        with st.spinner("Embedding query…"):
            vec = embed_query_text(q)
else:
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if up:
        st.image(up, width=240)
        with st.spinner("Embedding image…"):
            vec = embed_query_image(up.getvalue(), up.type)

if vec is None:
    st.info("Enter a query or upload an image.")
    st.stop()

with st.spinner("Searching Neon…"):
    rows = search(
        vec,
        source_type=None if st_filter == "all" else st_filter,
        release=None if release_filter == "all" else release_filter,
        limit=limit,
        threshold=threshold,
    )

if not rows:
    st.warning("No matches above the similarity threshold. Try lowering it.")
    st.stop()

st.subheader(f"{len(rows)} result(s)")
for r in rows:
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        with c1:
            header = f"**[{r['parent_id']}]({r['embedded_image_url']})** · `{r['source_type']}` · sim **{r['similarity']:.3f}**"
            page = page_number(r)
            if page is not None:
                header += f" · page {page}"
            elif r["start_seconds"] is not None:
                header += f" · {r['start_seconds']:.1f}s → {r['end_seconds']:.1f}s"
            st.markdown(header)
            if r["embedded_text"]:
                st.write(r["embedded_text"][:600] + ("…" if len(r["embedded_text"]) > 600 else ""))
            local = local_media_path(r)
            if local and r["source_type"] == "video_chunk":
                st.video(str(local), start_time=int(r["start_seconds"] or 0))
            elif local and r["source_type"] == "audio_clip":
                st.audio(str(local), start_time=int(r["start_seconds"] or 0))
            elif local and r["source_type"] == "pdf_page":
                st.image(str(local), use_container_width=True)
                if r["embedded_image_url"]:
                    st.link_button("Open full PDF on war.gov", r["embedded_image_url"])
            elif r["embedded_image_url"]:
                st.link_button("Open source", r["embedded_image_url"])
        with c2:
            st.metric("similarity", f"{r['similarity']:.3f}")
            st.caption(f"{r['release']} · {r['release_date']}")
```

---

## 11. Gotchas / things that will trip you up

- **Pooled vs direct Neon endpoint.** The user's connection string in the
  earlier session was the `-pooler` host. For a long-lived Streamlit process
  that reuses one connection across many queries, psycopg3 will eventually
  promote a statement to a *named* prepared statement (default
  `prepare_threshold=5`), which PgBouncer in transaction-pooling mode cannot
  hold across transactions. Use the **direct** endpoint (host without
  `-pooler`) or set `prepare_threshold=None` on the connection.
- **Dimension must match.** The column is `VECTOR(768)`. Don't pass a 1536-dim
  vector — it'll fail on the cast. If you ever switch to a different
  `output_dimensionality`, you'll need to migrate the column.
- **Instruction-in-prompt, not `task_type=`.** gemini-embedding-2 silently
  ignores `EmbedContentConfig.task_type` on the consumer API and instead
  expects the task to be expressed *inside the content*. Wrap documents as
  `title: {title} | text: {body}` (via `e.format_document_text(...)`) and
  queries as `task: search result | query: {q}` (via `e.format_query(...)`,
  applied automatically by `e.generate_text_embedding`). Skipping this
  produces noticeably worse ranking — the previous version of this corpus
  ranked NASA audio narratives above DOW UAP video clips on the query
  "instantaneous acceleration" because the asymmetric format wasn't applied;
  the re-embed with proper wrapping put `dvids_1007707` at ranks 1–4.

- **Vertex-only config options.** Three `EmbedContentConfig` fields exist in
  the SDK but are rejected by the consumer Gemini API
  (`"<option> parameter is not supported in Gemini API"`):
  `document_ocr` (server-side PDF OCR), `audio_track_extraction` (pull audio
  from video for the embed), and `auto_truncate`. They're only available via
  Vertex AI. If you migrate to Vertex (`genai.Client(vertexai=True, project=...,
  location=...)`), all three become usable and would let us simplify the
  pipeline (no manual ffmpeg audio extraction, no manual OCR pre-step).
- **`<=>` is distance, not similarity.** Lower = more similar. Always do
  `1 - (embedding <=> query)` for a similarity score.
- **HNSW recall.** The HNSW index is approximate. For exact ranking on small
  result sets, you can `SET LOCAL hnsw.ef_search = 100;` before the query.
- **First Neon query after idle is slow.** Neon auto-suspends idle databases;
  expect ~500ms cold-start latency on the first request.
- **Don't ship secrets.** `secrets.toml` should be `.gitignore`d. The keys from
  the previous session are exposed in that chat transcript and should be
  rotated.
- **Streamlit `st.video` URL playback.** Local file paths work great and
  support `start_time` seeking. Remote HTTP URLs are flaky for seeking —
  prefer local files where possible.
- **Audio for the NASA recordings.** The source assets on DVIDS are MP4
  wrappers (large, ~200 MB each). The previous session extracted the audio
  track to `.m4a` (a few MB each) and embedded *that*. Use the `.m4a` for
  playback; ignore the source `.mp4` unless you want visual.
- **`pgvector` + `psycopg3`: don't pass `list[float]` bare.** The pgvector
  adapter doesn't auto-cast Python lists to the `vector` type. Either bind a
  `numpy.ndarray`, or (what the example does) serialise the vector to the
  textual form `'[a,b,c,…]'` and use `%s::vector` in the SQL. Forgetting this
  fails with `operator does not exist: vector <=> double precision[]`.
- **Text queries are biased toward text-rich modalities.** In this corpus,
  any plain text query crowds the top with `audio_clip` and `pdf_page` rows
  because their `embedded_text` is long (multi-sentence NASA narratives /
  multi-line OCR), and because video chunks' multimodal vectors are pulled
  toward visual neighborhoods that short text queries can't reach. Concrete
  example: the query "instantaneous acceleration" returns 12 NASA Apollo /
  Mercury audio rows in the top 12 — and **does not surface** the DVIDS clip
  `dvids_1007707` whose title literally contains "instant acceleration". To
  let video chunks compete: default to a `source_type` filter, present
  **faceted results** (top-N per type side by side), or steer users toward
  **image queries** (same-modality alignment with video frames).

---

## 12. Quick test: does the database actually have what this doc claims?

Run once before you start coding the UI:

```python
import os, psycopg
with psycopg.connect(os.environ["DATABASE_URL"]) as c:
    for row in c.execute(
        "SELECT source_type, COUNT(*) AS rows, COUNT(DISTINCT parent_id) AS assets "
        "FROM embeddings GROUP BY source_type ORDER BY source_type"
    ).fetchall():
        print(row)
```

Expected (as of the handoff): `('audio_clip', 27, 7)`, `('pdf_page', 126, 5)`, and `('video_chunk', 154, 49)` — total **307 rows** across **61 distinct parent_ids**, all `release='PURSUE_2'` / `release_date='2026-05-22'`.

---

## 13. Suggested next steps for the Streamlit session

1. Drop `embeddings_v2.py` and the `app.py` from §10 into a fresh folder.
2. Create `.streamlit/secrets.toml` with `DATABASE_URL` and `GEMINI_API_KEY`.
3. Run §12 to confirm DB connectivity.
4. `streamlit run app.py` and test a few queries: `"spherical UAP over water"`,
   `"high-speed maneuver"`, `"Apollo astronaut"`.
5. Polish UI: result cards, thumbnails (DVIDS pages have poster images in
   `og:image` if you want to scrape), pagination, multimodal query (already
   stubbed in the example), per-result "show all chunks of this video" drilldown.
6. Optional: add an admin tab that ingests new assets (re-uses `embeddings_v2`
   plus the `retry_release_2.py` patterns).
