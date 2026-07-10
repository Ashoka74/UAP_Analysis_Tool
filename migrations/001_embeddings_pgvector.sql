-- Multimodal embeddings table for embeddings_v2.py / pipeline/embed_media.py.
-- Run once against the Neon database (psql "$DATABASE_URL" -f migrations/001_embeddings_pgvector.sql).
--
-- Bootstraps the full table if it doesn't exist (fresh DB), then applies the
-- video/audio-chunk additions idempotently (existing DB from the Vercel app).
-- Vector dimension is 768 = embeddings_v2.DEFAULT_EMBEDDING_DIMENSIONS; if you
-- raise GEMINI_EMBEDDING_DIMENSIONS you must recreate the column/index to match.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_type         TEXT        NOT NULL,   -- asset | gallery | video_chunk | audio_clip | pdf_page
    source_id           TEXT        NOT NULL,   -- doc code, or {code}:{start_ms}-{end_ms} for chunks
    user_id             UUID        NOT NULL,
    organization_id     UUID,
    embedding           VECTOR(768) NOT NULL,
    embedded_image_url  TEXT,                   -- media URL (image/video/audio alike)
    embedded_text       TEXT,
    start_seconds       REAL,
    end_seconds         REAL,
    parent_id           TEXT,                   -- source media id for video/audio chunks
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_embeddings_source UNIQUE (source_type, source_id)
);

-- Additions for pre-existing tables (no-ops on a fresh one).
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS start_seconds REAL;
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS end_seconds   REAL;
ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS parent_id     TEXT;

-- Lookup all chunks of a given parent video efficiently.
CREATE INDEX IF NOT EXISTS idx_embeddings_parent_id
    ON embeddings (parent_id) WHERE parent_id IS NOT NULL;

-- ANN index for cosine search (search_similar uses the <=> operator).
-- Name matches the Vercel app's migration so running both is idempotent.
-- On small Neon computes, bump maintenance_work_mem cautiously for the
-- build (SET maintenance_work_mem = '512MB'); 2GB can OOM a 0.25 CU node.
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON embeddings USING hnsw (embedding vector_cosine_ops);

-- Owner/scope filters evaluated before the vector comparison.
CREATE INDEX IF NOT EXISTS idx_embeddings_user_id  ON embeddings (user_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_org_id   ON embeddings (organization_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_src_type ON embeddings (source_type);

-- If an older CHECK constraint restricts source_type, broaden it:
-- ALTER TABLE embeddings DROP CONSTRAINT IF EXISTS embeddings_source_type_check;
-- ALTER TABLE embeddings ADD CONSTRAINT embeddings_source_type_check
--   CHECK (source_type IN ('asset', 'gallery', 'video_chunk', 'audio_clip', 'pdf_page'));
