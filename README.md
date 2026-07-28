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

A data analysis platform for UFO/UAP (Unidentified Aerial Phenomena) sighting
reports: LLM-based feature extraction, semantic search, statistical
clustering, magnetic-anomaly correlation, and geospatial visualization — over
both a large multi-source catalog of historical sighting reports and a
purpose-built pipeline that ingests the U.S. government's (`war.gov`) UAP
document releases end to end, from raw PDF to a structured, deduplicated,
query-ready table.

Three interchangeable front ends sit on the same analysis core:

| Front end | Entry point | Use case |
|---|---|---|
| **Streamlit** (primary) | `app.py` | Full multi-page app — the reference implementation of every feature below |
| **React + FastAPI** | `frontend/` + `api/main.py` | Decoupled web stack for a production-grade deploy (Vercel + Render/Railway) |
| **Gradio** | `gradio_app.py` | Lightweight alternative UI |

## Data sources

Two distinct kinds of input feed this tool, handled by different subsystems:

### 1. The PURSUE government-document corpus (`pipeline/`)

Raw PDF releases from `war.gov/UFO` are scraped, OCR'd, and structured into a
report table entirely in-repo (no external scripts or interpreter):

```
Scrape (war.gov ZIPs) → Layout (split/restructure pages) → OCR (Mistral) →
Assembly (concat pages) → Extraction (Gemini/OpenAI/NVIDIA NIM) →
Table (join + reconcile + source-agency mapping) → embed_media.py (Gemini
multimodal embeddings → Neon pgvector)
```

See [`pipeline/README.md`](pipeline/README.md) for the full stage-by-stage
script inventory and [`preprocessing.py`](preprocessing.py) for the Streamlit
page that drives it. `check_new_release.py` is a cron entry point that
detects and ingests new `war.gov` releases automatically. This corpus is the
one [`DEDUP_METHODOLOGY.md`](DEDUP_METHODOLOGY.md) documents deduplication
against — 1,791 rows after extraction, spanning 1833–2025.

### 2. External sighting-report catalogs (`config.py`)

`config.py` defines per-source JSON extraction/merge schemas
(`FORMAT_NUFORC`, `FORMAT_BLUE_BOOK`, `FORMAT_UK_NATIONAL_ARCHIVES`,
`FORMAT_COBEPS_NOTIFICATIONS_PAN`, `FORMAT_GEP`, `FORMAT_UPDB_NICAP`,
`FORMAT_OVNIBASE`, `FORMAT_UFOSETI`, `FORMAT_UAP_SIGHTINGS_GITHUB`,
`FORMAT_WEINSTEIN_PILOT_CATALOG`, `FORMAT_PETROWITSCH_LATAM`,
`FORMAT_UFOCAT`, `FORMAT_CISU_CISUCAT`, `FORMAT_UAPCHECK`) so each catalog's
native fields normalize into one `FORMAT_MERGED` shape. `FORMAT_LONG` is the
general-purpose GPT-4o-mini extraction schema (narrative, trust score,
location, UAP characteristics, witness details, evidence, etc.) used by
`parsing.py` for free-text reports that don't come from a known catalog.

### Bulk dataset artifacts

`final_ufoseti_dataset.h5` and `parsed_files_distance_embeds.h5` (pre-parsed
+ embedded datasets) are too large for git — `.h5`/`.csv` are gitignored.
`data_fetch.py` resolves them at runtime: local path → `$UAP_DATASET_DIR` →
repo root → download from the `UFOSINT/uap-datasets` Hugging Face dataset
repo (zstd parquet, cached under `~/.cache/huggingface`). Any deploy target
(HF Space, Docker, Railway, cron) gets the data without bundling it into the
image.

## Core analysis pipeline

```
Raw Text Reports (CSV/XLSX) or PURSUE PDFs
    └── parsing.py (GPT-4o-mini) → Structured JSON (config.FORMAT_LONG)
            └── analyzing.py → sentence-transformer embeddings → UMAP → HDBSCAN clusters
                    └── XGBoost predictions + Cramer's V correlations
                            └── map.py (Kepler.gl) / magnetic.py (FastDTW vs InterMagnet) / rag_search.py (Cohere rerank)
```

| Stage | File | What it does |
|---|---|---|
| Parsing | `parsing.py` | Concurrent GPT-4o-mini calls (`uap_analyzer.UAPParser`) extracting structured JSON per `config.FORMAT_LONG`, with exponential backoff |
| SCU normalization (optional) | `scu_normalizer.py` | Canonicalizes parsed output (countries, states, witness roles, craft fields) and derives the SCU five-criterion eligibility gate |
| Analysis | `analyzing.py` | `sentence-transformers/e5-large-v2` embeddings → UMAP reduction → HDBSCAN clustering → TF-IDF/LLM cluster naming → XGBoost classifiers for feature correlation → `dedupe_semantic` feature-column redundancy control (distinct from sighting-level dedup, see below) |
| RAG search | `rag_search.py` | Cohere rerank semantic search over natural-language queries; also multimodal (image/video/audio) search over the pgvector embeddings table |
| Magnetic analysis | `magnetic.py` | Correlates sightings with InterMagnet station data via Dynamic Time Warping (`fastdtw`) |
| Map | `map.py` | Kepler.gl visualization with proximity analysis to military bases (`secret_bases.csv`) and nuclear facilities (`global_power_plant_database.csv`) |
| PDF OCR | `pdf_ocr.py` | Standalone LightOnOCR-based table extractor page |
| Timeline / dedup | `make_timeline.py`, `geo_dedup.py`, `apply_dedup.py`, `eval_pairs.py` | See [`DEDUP_METHODOLOGY.md`](DEDUP_METHODOLOGY.md) |

### Key classes (`uap_analyzer.py`)

- **`UAPParser`** — concurrent OpenAI calls for JSON extraction with exponential backoff
- **`UAPAnalyzer`** — embeddings, UMAP, HDBSCAN, TF-IDF cluster naming, cosine-similarity cluster merging
- **`UAPVisualizer`** — XGBoost prediction plots, confusion matrices, Cramer's V heatmaps, treemaps

## Repository layout

```
app.py, app2.py, app3.py   Streamlit entry points (app.py = current multi-page nav;
                           app2.py = legacy single-file dashboard; app3.py = st_paywall-gated variant)
gradio_app.py              Gradio alternative UI
parsing.py, analyzing.py, rag_search.py, magnetic.py, map.py, preprocessing.py, pdf_ocr.py
                           Streamlit pages (see table above)
uap_analyzer.py            Core parser/analyzer/visualizer classes
config.py                  Per-source extraction schemas (FORMAT_*), facility types
scu_normalizer.py          SCU methodology normalization + eligibility gate
make_timeline.py, geo_dedup.py, apply_dedup.py, eval_pairs.py
                           Sighting-record deduplication + timeline rendering (see DEDUP_METHODOLOGY.md)
color_dedup_xlsx.py        Rebuilds the dedup workbook with cluster-shaded formatting
backfill_media.py          Non-destructive media-column backfill via war.gov manifest join
embeddings.py, embeddings_v2.py, embed_csv.py, embed_narratives.py
                           Embedding generation (sentence-transformers, Gemini multimodal, pgvector storage)
data_fetch.py              Resolves/downloads the bulk .h5 dataset artifacts from anywhere
sankey_updb.py, make_loop_chart.py, uap_xlsx_filler.py
                           Reporting/plotting utilities (Sankey diagrams, autoresearch-loop staircase charts, xlsx export)
pipeline/                  Vendored PDF→report-table scripts (scrape, OCR, extraction, assembly) — see pipeline/README.md
utils/                     Shared Streamlit utilities (DataProcessor, UAP_Visualizer, SessionStateManager,
                           EmbeddingCacheManager, MemoryManager, UAP_Pipeline) — see utils/README.md
api/                       FastAPI backend (main.py + services/) mirroring every Streamlit page as a REST endpoint
frontend/                  Vite + React + TypeScript + Tailwind + Zustand SPA
tests/                     pytest suite (analysis stats, API, cluster merge, parser, SCU)
autoresearch/              Tuning-loop run artifacts (dedup calibration, see DEDUP_METHODOLOGY.md)
paper/                     LaTeX methodology write-up (redundancy control, clustering, XGBoost design)
migrations/                SQL migrations for the Postgres/pgvector embeddings store
```

## Modern Front End (React + FastAPI)

The repo ships a decoupled web stack alongside the Streamlit app:

- **Backend** — FastAPI service under `./api` (routes inline in `api/main.py`,
  business logic in `api/services/`: `analysis_service.py`, `parsing_service.py`,
  `rag_service.py`, `scu_service.py`, `map_service.py`, `multimodal_service.py`).
- **Frontend** — Vite + React + TypeScript + Tailwind + Zustand under
  `./frontend`, talking to the backend through a Vite proxy.

### Prerequisites
- Python 3.11+ with `fastapi` and `uvicorn` installed (covered by
  `requirements.txt` / `pyproject.toml`).
- Node 20+ for the frontend.

### 1. Start the backend (FastAPI)

```bash
uvicorn api.main:app --reload --port 8000
```

### 2. Start the frontend (Vite + React)

```bash
cd frontend
npm install      # first time only
npm run dev
```

Vite starts on port **5173** and proxies every `/api/*` request to
`http://localhost:8000` (see `frontend/vite.config.ts`) — no CORS
configuration needed for local dev.

### 3. Open the app

http://localhost:5173

### Production build

```bash
cd frontend
npm run build       # output in frontend/dist
```

Serve the `dist/` artifacts behind any static host and keep
`uvicorn api.main:app` running for the API.

### API reference

Every endpoint lives in `api/main.py`, grouped by feature:

| Group | Endpoints |
|---|---|
| Data | `GET /api/data/load`, `POST /api/data/upload`, `POST /api/data/filter`, `GET /api/data/columns`, `GET /api/data/column-values` |
| Analysis | `POST /api/analyze/run`, `GET /api/analyze/results`, `GET /api/analysis/clusters`, `POST /api/analysis/column-groups`, `POST /api/analysis/xgboost(-pca\|-impute)`, `POST /api/analysis/interpret`, `POST /api/analysis/cramers-v`, `POST /api/analysis/contingency`, `POST /api/analysis/conditional` |
| Parsing | `GET /api/parse/schemas`, `POST /api/parse/schema-merge`, `POST /api/parse/schema-coverage`, `POST /api/parse/upload(-batch)`, `POST /api/parse/use-loaded`, `POST /api/parse/estimate`, `POST /api/parse/run`, `GET /api/parse/result`, `GET /api/parse/export` |
| SCU | `GET /api/scu/criteria`, `POST /api/scu/normalize(-upload)`, `POST /api/scu/remap`, `GET /api/scu/export`, `POST /api/scu/filter` |
| Search | `POST /api/rag/search`, `POST /api/rag/multimodal/releases`, `POST /api/rag/multimodal/search(-image)`, `POST /api/query/gemini` |
| Map / magnetic | `GET /api/map/html`, `POST /api/magnetic/run` |
| Misc | `GET /api/dashboard/summary` |

Sessions are isolated server-side by an `X-Session-ID` header (24h TTL,
in-memory `SessionState`), so concurrent users don't share uploaded data or
analysis results.

## Running the Streamlit app

```bash
# Python 3.12+, using uv (recommended)
uv sync
uv run streamlit run app.py

# Or using pip
pip install -r requirements.txt
streamlit run app.py

# Gradio alternative
uv run gradio gradio_app.py
```

Each page can also be run standalone for isolated testing:

```bash
uv run streamlit run parsing.py
uv run streamlit run analyzing.py
uv run streamlit run rag_search.py
uv run streamlit run magnetic.py
uv run streamlit run map.py
uv run streamlit run preprocessing.py
```

## Configuration

### API keys

Secrets live in `.streamlit/secrets.toml` (not committed):

| Key | Used by |
|---|---|
| `OPENAI_KEY` | GPT-4o-mini parsing (`parsing.py`) |
| `GEMINI_KEY` | Gemini Pro summarization, multimodal embeddings, LLM dedup judge (`eval_pairs.py`) |
| `COHERE_KEY` | Rerank search (`rag_search.py`) |
| `MISTRAL_API_KEY` | OCR in the document preprocessing pipeline |
| `NVIDIA_API_KEY` | Optional NIM-based extraction |

### Environment variables (API/deploy)

| Variable | Purpose |
|---|---|
| `UAP_API_CORS_ORIGINS` | Comma-separated frontend origin allowlist |
| `UAP_ANALYSIS_MODE` | `production` (real pipeline) vs `mock` (fast demo backend) |
| `UAP_DATASET_WEST` / `UAP_DATASET_EAST` | Override paths for the two bulk `.h5` datasets |
| `UAP_DATASET_DIR` | Local search directory `data_fetch.py` checks before hitting HF Hub |
| `UAP_HF_DATASET_REPO` | Hugging Face dataset repo for the bulk artifacts (default `UFOSINT/uap-datasets`) |
| `HF_TOKEN` | Auth for private HF dataset/model access |
| `DATASET_URL_WEST` / `DATASET_URL_EAST` | Presigned URLs `render-start.sh` downloads to the persistent disk on first boot |
| `UAP_PIPELINE_ROOT` | Working directory root for the `pipeline/` scripts |

### Data files

- `final_ufoseti_dataset.h5`, `parsed_files_distance_embeds.h5` — bulk pre-parsed/embedded datasets (fetched via `data_fetch.py`, see above)
- `global_power_plant_database.csv` — nuclear facility locations
- `secret_bases.csv` — military base locations
- `*.kgl` — Kepler.gl map configuration files (`uap_config.kgl`, `military_config.kgl`)
- `PURSUE_1_2_3_dedup_flagged.xlsx` — deduplicated PURSUE corpus workbook (see `DEDUP_METHODOLOGY.md`)
- `uap_datasets_schema_complete.json`, `uap_master_schema.json` — full schema references for the merged dataset

## GPU requirements

The embedding model (`sentence-transformers/e5-large-v2`) and XGBoost run on
CUDA by default. The code includes `torch.cuda.empty_cache()` calls for
memory management; `MemoryManager` in `utils/` offers CPU-only fallbacks and
chunked processing for large files.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

| File | Covers |
|---|---|
| `test_analysis.py` | `analysis_service` statistics — CI, FDR, sparsity, conditional independence, feature-column dedup |
| `test_api.py` | FastAPI endpoint behavior |
| `test_cluster_merge.py` | HDBSCAN cluster-merging logic |
| `test_parser.py` | `UAPParser` JSON extraction |
| `test_scu.py` | SCU normalizer + eligibility gate |
| `conftest.py` | Shared fixtures |

## Deployment

| Target | Config | Notes |
|---|---|---|
| Hugging Face Spaces | YAML frontmatter above (`README.md`) | `sdk: streamlit`, entry `app.py` |
| Render | `render.yaml`, `Dockerfile.api`, `render-start.sh` | Deploys the FastAPI backend as a Docker web service on a persistent disk; datasets fetched at boot from `DATASET_URL_WEST/EAST` when set |
| Railway | `railway.json`, `Dockerfile.api` | Same Docker image/start script as Render |
| Vercel | `frontend/` | Static build of the React SPA; point it at the deployed API's origin |
| Docker Compose | `docker-compose.yml`, `docker-compose.gpu.yml` | Local multi-container dev (CPU/GPU variants) |

See `DEPLOY.md` for the full step-by-step deploy walkthrough.

## Further documentation

- [`DEDUP_METHODOLOGY.md`](DEDUP_METHODOLOGY.md) — sighting-record deduplication pipeline, calibration results, file inventory
- [`pipeline/README.md`](pipeline/README.md) — PDF-to-report-table preprocessing stages
- [`utils/README.md`](utils/README.md) — shared utilities package API
- [`STREAMLIT_HANDOFF.md`](STREAMLIT_HANDOFF.md) — embeddings/pgvector semantic search handoff notes
- [`ENHANCED_PIPELINE_README.md`](ENHANCED_PIPELINE_README.md) — data-processing/visualization pipeline enhancements
- [`PLAN.md`](PLAN.md) — React + FastAPI migration architecture plan
- [`DEPLOY.md`](DEPLOY.md) — deployment walkthrough
- [`paper/methodology.tex`](paper/methodology.tex) — academic write-up of the redundancy-control, clustering, and XGBoost methodology
