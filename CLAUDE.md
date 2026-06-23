# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UAP-Data-Analysis-Tool (also known as UFOSINT) is a Streamlit-based data analysis application for UFO/UAP (Unidentified Aerial Phenomena) sighting reports. It provides text parsing, clustering, statistical analysis, geospatial visualization, and magnetic anomaly correlation capabilities.

## Commands

### Running the Application

```bash
# Streamlit app (main interface)
uv run streamlit run app.py

# Gradio app (alternative interface)
uv run gradio gradio_app.py
```

### Dependency Management

```bash
# Install dependencies using uv (Python 3.12+)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Testing/Running Individual Pages

Each Streamlit page can be tested individually:
```bash
uv run streamlit run parsing.py
uv run streamlit run analyzing.py
uv run streamlit run rag_search.py
uv run streamlit run magnetic.py
uv run streamlit run map.py
```

## Architecture

### Core Analysis Pipeline

The application follows a multi-step NLP analysis pipeline:

1. **Parsing (`parsing.py`)** - Uses OpenAI GPT-4o-mini to extract structured JSON features from raw UAP report text. The JSON schema is defined in `config.py` (`FORMAT_LONG`).

2. **Analysis (`analyzing.py`)** - Performs dimensionality reduction (UMAP) and clustering (HDBSCAN) on text embeddings, then trains XGBoost classifiers to identify feature correlations.

3. **RAG Search (`rag_search.py`)** - Implements semantic search using Cohere's rerank API to find relevant reports based on natural language queries.

4. **Magnetic Analysis (`magnetic.py`)** - Correlates UAP sightings with InterMagnet station data using Dynamic Time Warping (FastDTW).

5. **Map Visualization (`map.py`)** - Interactive Kepler.gl maps showing sighting locations with proximity analysis to military bases and nuclear facilities.

### Key Classes (`uap_analyzer.py`)

- **`UAPParser`** - Concurrent OpenAI API calls for JSON extraction with exponential backoff
- **`UAPAnalyzer`** - Text embedding (sentence-transformers/e5-large-v2), UMAP reduction, HDBSCAN clustering, TF-IDF cluster naming, cluster merging via cosine similarity
- **`UAPVisualizer`** - XGBoost prediction plots, confusion matrices, Cramer's V heatmaps, treemaps

### Utils Module (`utils/`)

Enhanced utilities providing:
- `DataProcessor` - DataFrame filtering with Streamlit UI
- `UAP_Visualizer` - Interactive Plotly visualizations
- `SessionStateManager` - Streamlit session state handling
- `EmbeddingCacheManager` - Caches sentence-transformer embeddings to avoid recomputation
- `MemoryManager` - GPU memory management for CUDA operations
- `UAP_Pipeline` - End-to-end analysis pipeline orchestration

### Data Flow

```
Raw Text Reports (CSV/XLSX)
    └── parsing.py (GPT-4o-mini) → Structured JSON
            └── analyzing.py → Embeddings → UMAP → HDBSCAN clusters
                    └── XGBoost predictions + Cramer's V correlations
                            └── map.py (Kepler.gl visualization)
```

### Session State Keys

The app uses Streamlit session state extensively. Key variables:
- `parsed_responses` / `parsed_responses_df` - Parsed JSON data
- `analyzers` / `clusters` / `col_names` - Analysis results
- `stage` - UI workflow state
- `api_key_valid` - OpenAI key validation status

## Configuration

### API Keys

Secrets are stored in `.streamlit/secrets.toml` (not committed). Required keys:
- `OPENAI_KEY` - For GPT-4o-mini parsing
- `GEMINI_KEY` - For Gemini Pro summarization
- `COHERE_KEY` - For rerank search

### Data Files

- `final_ufoseti_dataset.h5` - Pre-parsed UAP dataset
- `parsed_files_distance_embeds.h5` - Dataset with embeddings
- `global_power_plant_database.csv` - Nuclear facility locations
- `secret_bases.csv` - Military base locations
- `*.kgl` - Kepler.gl map configuration files

## GPU Requirements

The embedding model (`embaas/sentence-transformers-e5-large-v2`) and XGBoost run on CUDA by default. The code includes `torch.cuda.empty_cache()` calls for memory management.

## HuggingFace Deployment

The app is configured for HuggingFace Spaces deployment (see `README.md` metadata). The `sdk_version: 1.36.0` specifies the Streamlit version.
