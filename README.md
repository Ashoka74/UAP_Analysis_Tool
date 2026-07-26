---
title: UFOSINT
emoji: 🛸
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.36.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: UFO/UAP AI Analyst
---

# UAP-Data-Analysis-Tool (UFOSINT)

An analysis toolkit for UFO/UAP (Unidentified Anomalous Phenomena) sighting reports: LLM-based text parsing into structured JSON, embedding/clustering, statistical and geospatial analysis, cross-source deduplication, semantic (RAG) search, and correlation against magnetic-observatory data. Built around declassified-record corpora (PURSUE, NICAP, NUFORC, NARCAP, and the DoD/DVIDS war.gov UAP releases).

The project ships **two parallel interfaces** over the same underlying pipeline:

1. **Streamlit app** (`app.py` + friends) — the original, feature-complete multi-page analyst tool.
2. **FastAPI + React web app** (`api/` + `frontend/`) — a decoupled rebuild for browser deployment; see [React feature parity](#react-vs-streamlit-feature-parity) below for what's ported so far.

## Contents
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Core analysis pipeline](#core-analysis-pipeline)
- [Repository layout](#repository-layout)
- [Data](#data)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)

## Quick start

### Streamlit (legacy/primary analyst app)

```bash
uv sync                          # or: pip install -r requirements.txt
uv run streamlit run app.py
```

Individual pages can also be run standalone for focused testing:

```bash
uv run streamlit run parsing.py
uv run streamlit run analyzing.py
uv run streamlit run rag_search.py
uv run streamlit run magnetic.py
uv run streamlit run map.py
```

Requires Python 3.12 and API keys in `.streamlit/secrets.toml` (see [Configuration](#configuration)).

### FastAPI + React (modern web stack)

```bash
# Backend — from repo root
uvicorn api.main:app --reload --port 8000

# Frontend — separate terminal
cd frontend
npm install     # first time only
npm run dev     # http://localhost:5173
```

Vite proxies `/api/*` to `http://localhost:8000` (`frontend/vite.config.ts`), so no CORS setup is needed in dev. Production build: `npm run build` (output in `frontend/dist`), served statically alongside a running `uvicorn api.main:app`.

### Gradio (alternate interface)

```bash
uv run gradio gradio_app.py
```

## Architecture

### Core analysis pipeline

```
Raw text reports (CSV/XLSX/PDF)
  └─ parsing.py / api/services/parsing_service.py  (GPT-4o-mini)   → structured JSON per FORMAT_LONG schema (config.py)
       └─ analyzing.py / api/services/analysis_service.py           → embeddings → UMAP → HDBSCAN clusters
            └─ XGBoost feature-importance / imputation / PCA-latent passes
                 └─ map.py / api/services/map_service.py             → Kepler.gl geospatial visualization
```

Alongside that primary path:

- **Deduplication** (`deduplication.py`, `api/services/dedup_service.py`) — entity-resolution (geospatial/date-window blocking + text-embedding similarity) and pure text-duplicate clustering, both single-dataset and cross-dataset (pool multiple sources, e.g. PURSUE + NICAP + NUFORC + NARCAP, and find matching/duplicate incident reports across them). See [`SUBDATASETS_V2/MASTER_DEDUPE/`](#data) for a worked output.
- **RAG search** (`rag_search.py`, `api/services/rag_service.py`) — semantic search via Cohere rerank over parsed reports, plus a separate multimodal (text+image) search path over Gemini embeddings stored in Postgres/pgvector (`api/services/multimodal_service.py`), covering DoD/DVIDS video, NASA audio, and source-document PDF pages from the war.gov UAP releases.
- **Magnetic correlation** (`magnetic.py`) — correlates sighting timestamps/locations against INTERMAGNET geomagnetic observatory data via Dynamic Time Warping (FastDTW).
- **SCU normalization** (`scu_normalizer.py`, `api/services/scu_service.py`) — maps parsed reports onto the "Scientific Coalition for UAP Studies" (SCU) criteria/phase taxonomy.
- **Gazetteer resolution** (`api/services/us_gazetteer.py`) — resolves free-text US location names to coordinates for dedup blocking and mapping.

### Key classes (`uap_analyzer.py`)

- `UAPParser` — concurrent OpenAI calls for JSON extraction, with exponential backoff.
- `UAPAnalyzer` — text embedding (`sentence-transformers/e5-large-v2`), UMAP reduction, HDBSCAN clustering, TF-IDF cluster naming, cosine-similarity cluster merging.
- `UAPVisualizer` — XGBoost prediction plots, confusion matrices, Cramér's V heatmaps, treemaps.
- `train_xgboost()` — the shared XGBoost training/eval routine used by both the Streamlit pages and `api/services/analysis_service.py` (feature importance, PCA-latent-index importance for collinear fields, and per-column missing-value imputation).

### `utils/` module (Streamlit-side helpers)

`DataProcessor` (filtering UI), `UAP_Visualizer` (Plotly views), `SessionStateManager`, `EmbeddingCacheManager` (caches sentence-transformer embeddings), `MemoryManager` (CUDA memory), `UAP_Pipeline` (end-to-end orchestration). See `utils/__init__.py` for the full export list.

### FastAPI backend (`api/`)

Everything lives in one router file, `api/main.py` (~2000 lines) — there is no separate `api/routes/`; endpoints are grouped by prefix:

| Prefix | Covers |
|---|---|
| `/api/data/*` | load/upload/filter datasets, column introspection |
| `/api/parse/*` | schema management, batch LLM parsing, export |
| `/api/analyze/*`, `/api/analysis/*` | clustering, XGBoost importance/PCA/imputation, Cramér's V, contingency |
| `/api/dedup/*` | simple pairwise similarity, advanced/cross-DB entity dedup, apply/export |
| `/api/rag/*` | text search + multimodal (image/video/audio) search |
| `/api/scu/*` | SCU criteria normalization/remap/export |
| `/api/map/*` | Kepler.gl HTML generation |
| `/api/magnetic/*` | magnetic-correlation runs |
| `/api/query/*` | Gemini free-form querying |
| `/api/dashboard/*` | summary stats |

Business logic lives in `api/services/*_service.py` (one per domain, mirroring the prefixes above); `api/utils/data_utils.py` holds shared helpers.

### React frontend (`frontend/`)

Vite + React 19 + TypeScript + Tailwind 4 + Zustand + React Router. `frontend/src/components/` mirrors the API domains: `analysis/`, `data/`, `dedup/`, `magnetic/`, `map/`, `parsing/`, `query/`, `rag/`, `scu/`, plus `dashboard/`, `layout/`, and `common/`. API client in `frontend/src/api/`, shared types in `frontend/src/types/`, Zustand stores in `frontend/src/store/`.

`frontend2/` and `frontend_backup/` are earlier/abandoned frontend attempts kept for reference — not part of the active build.

#### React vs. Streamlit feature parity

The React app is a rebuild-in-progress, not a full mirror of the Streamlit app yet:

- **Ported:** parsing (schema select/merge/coverage-diff, cost estimate, batch run), SCU normalization, Cohere RAG text search, multimodal (image/video/audio) RAG search over the Neon pgvector `embeddings` table, and analysis (UMAP+HDBSCAN clustering, Cramér's V explorer, XGBoost feature importance with stratified k-fold CV).
- **Deferred:** the cross-DB cosine-similarity + Gemini "judge" threshold workflow from `rag_search.py`, OpenAI server-batch parsing, SCU xlsx export, embeddings→HDF5 export.
- **Known rough edge:** `App.tsx`'s standalone `clusters` page (`ClusterView.tsx`) iframes a static pre-generated HTML file that isn't in the repo and nothing generates — it 404s. Don't confuse it with the real, live cluster scatter under Analysis → Clusters (`ClusterVisualization.tsx`), fed by `/api/analyze/run`.

## Repository layout

```
app.py, app2.py, app3.py     Streamlit entry points (app.py is the active multi-page app)
parsing.py, analyzing.py,    Streamlit pages — one per pipeline stage
rag_search.py, magnetic.py,
map.py
gradio_app.py                 Alternate Gradio interface
uap_analyzer.py                Core parser/analyzer/visualizer classes
deduplication.py, geo_dedup.py Standalone dedup pipelines (pre-API)
scu_normalizer.py              SCU criteria normalization
config.py                      API keys (legacy) + FORMAT_LONG parsing schema
embeddings.py, embeddings_v2.py, preprocessing.py
                                Embedding generation + pgvector storage (Neon-backed)
api/                            FastAPI backend (see above)
frontend/                       Active React web app
utils/                          Streamlit-side helper modules
pipeline/                       war.gov UAP-release ingestion: PDF→OCR→page extraction→
                                report/manifest joins→embeddings; also the weekly
                                release-watcher (check_new_release.py)
pipeline_data/                  Per-document extracted/normalized records from the
                                official releases (one folder per document code)
SUBDATASETS_V2/                 PURSUE + NICAP/NUFORC/NARCAP corpora, merges, and the
                                dedup pipeline output (MASTER_DEDUPE/) — see below
scripts/                        One-off analysis scripts (currently: pursue_eda.py)
migrations/                     SQL migrations (pgvector embeddings table, etc.)
tests/                          pytest suite (parser, analysis, cluster-merge, SCU, API)
autoresearch/                   Automated dedup-threshold tuning loop artifacts
kepler.gl/                      Vendored Kepler.gl source (map rendering)
```

### `SUBDATASETS_V2/` and the dedup pipeline

This is the working corpus for PURSUE-focused analysis:

- `PURSUE_1_2_3_normalized_full.csv`, `PURSUE_4_normalized.csv` → merged into `PURSUE_1_2_3_4_merged.csv` (1873 rows, 363 cols; `_source_dataset` distinguishes PURSUE_1_2_3 vs PURSUE_4).
- `PURSUE_1_2_3_4_eda_summary.json` (generated by `scripts/pursue_eda.py`) — row/column stats, year/country/shape/agency distributions, trust-score stats, and dedup summary.
- `NARCAP/`, `NUFORC/`, `NICAP_SAMPLE.csv`, `UAV_DATABASE/` — the other source corpora pooled for cross-source dedup.
- `MASTER_DEDUPE/` — output of the cluster-blocked dedup pipeline (`dedup_service.run_advanced_dedup`) pooling PURSUE + REGEX-filtered + NARCAP + combined NICAP/NUFORC (21,226 rows total): `deduped_dataset.csv` (per-row cluster/canonical flags, both entity-resolution and pure-text clustering), `cluster_edges.csv` (pairwise entity-match edges with distance/date-gap/similarity), `dedup_run_metadata.json` (run parameters/counts). Within PURSUE alone, ~35% of rows are text-level near-duplicates of other PURSUE rows (see the `dedup` block in the EDA summary above).
- Root-level `PURSUE_1_2_3_4_dedup_flagged.xlsx` — full-schema PURSUE export carrying an independent, earlier dedup pass (`dup_cluster_id`/`is_duplicate` columns; different run/thresholds than `MASTER_DEDUPE`, joinable to it via `case_text.text`).

### `pipeline/` — war.gov UAP release ingestion

Scrapes and processes the DoD/DVIDS official UAP release pages: PDF download → OCR (`run_ocr.py`) → page extraction/concatenation → report/manifest joins → multimodal embeddings (`embed_media.py`, images/video/audio into the Neon pgvector `embeddings` table). `check_new_release.py` is the release-watcher entry point (see project memory: runs on a scheduled cloud cron, not GitHub Actions).

## Data

Full per-file row/column inventory: **[`DATA_FILE_INVENTORY.md`](DATA_FILE_INVENTORY.md)**.

Notable datasets (not exhaustive — most large binaries are `.gitignore`d or Git-LFS):
- `final_ufoseti_dataset.h5`, `parsed_files_distance_embeds.h5` — pre-parsed datasets with embeddings.
- `global_power_plant_database.csv`, `secret_bases.csv` — reference layers for nuclear/military-proximity analysis in `map.py`.
- `*.kgl` — Kepler.gl map configs (`uap_config.kgl`, `military_config.kgl`).
- `uap_master_schema_v2.json` — the canonical UAP report schema (superset of `config.py`'s `FORMAT_LONG`).

## Configuration

### API keys
Legacy Streamlit path: `.streamlit/secrets.toml` (not committed) — `OPENAI_KEY`, `GEMINI_KEY`, `COHERE_KEY`.
FastAPI path: keys are supplied per-request (never stored server-side) for the RAG/multimodal endpoints; `DATABASE_URL` (Neon Postgres) is required for pgvector-backed embedding storage/search.

### GPU
The embedding model (`sentence-transformers/e5-large-v2`) and XGBoost use CUDA when available (`torch.cuda.empty_cache()` calls throughout for memory management). CPU fallback works but is slower for large corpora.

## Testing

```bash
uv run pytest
```

Covers the parser (`test_parser.py`), analysis/XGBoost paths (`test_analysis.py`), cluster merging (`test_cluster_merge.py`), SCU normalization (`test_scu.py`), and the FastAPI routes (`test_api.py`).

## Deployment

Full instructions: **[`DEPLOY.md`](DEPLOY.md)**.

- **Backend** → Render (`render.yaml`, service `ufosint-api`) or Railway (`railway.json`) — both plain container/PaaS hosting, not serverless/edge.
- **Frontend** → Vercel (static `frontend/dist` build).
- **Streamlit app** → HuggingFace Spaces (frontmatter at the top of this file) or `docker-compose.yml` / `Dockerfile.api(.gpu)` for self-hosting.
- Large data files (`.h5`, etc.) need external hosting (R2/S3/HF dataset repo) with a presigned/direct URL set as an env var on the Render service — see `DEPLOY.md` for the exact variable names.
