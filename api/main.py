import sys
import os
import io
import time
import logging
import traceback
from datetime import datetime, timezone
import uuid

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for importing uap_analyzer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Suppress expected outputs to only have clean API routes
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="UAP Analysis API", version="1.0.0")

# CORS Improvement (Codex #6): Explicit origin allowlist, credentials disabled unless required
origins = os.getenv("UAP_API_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
allow_origins = [origin.strip() for origin in origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# State Management (Codex #1): Session isolation
# ---------------------------------------------------------------------------
class SessionState:
    def __init__(self):
        self.dataset = None
        self.filtered_data = None
        self.analyzers = []
        self.col_names = []
        self.new_data = None
        self.data_processed = False
        self.analysis_results = {}
        self.cluster_viz = {}
        self.cramers_v = None
        self.analysis_runs = 0
        self.last_analysis_at = None
        self.analysis_mode = os.getenv("UAP_ANALYSIS_MODE", "production")  # demo vs production
        # Parsing / SCU state (mirrors the Streamlit parsing.py session keys)
        self.parse_source_df = None       # raw uploaded reports awaiting extraction
        self.parsed_responses = None      # {description: parsed JSON dict}
        self.parsed_df = None             # flat parsed-responses DataFrame
        self.scu_normalized_df = None     # scu_normalizer.normalize() output
        self.last_accessed = time.time()

sessions = {}
SESSION_TTL = 3600 * 24  # 24 hours

def get_session(session_id: str = Header(None, alias="X-Session-ID")) -> SessionState:
    if not session_id:
        session_id = "default"
    
    # Cleanup expired sessions
    current_time = time.time()
    expired = [k for k, v in sessions.items() if current_time - v.last_accessed > SESSION_TTL]
    for k in expired:
        del sessions[k]
        
    if session_id not in sessions:
        sessions[session_id] = SessionState()
        
    sessions[session_id].last_accessed = current_time
    return sessions[session_id]

# Dataset paths. Default to the repo-root .h5 files (local / docker-compose
# mount), but allow an absolute override via env so a hosted deploy (e.g. a
# Render persistent disk at /data) can keep the 1.8 GB dataset off the image.
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH_WEST = os.getenv("UAP_DATASET_WEST", os.path.join(_REPO_ROOT, "parsed_files_distance_embeds.h5"))
DATA_PATH_EAST = os.getenv("UAP_DATASET_EAST", os.path.join(_REPO_ROOT, "final_ufoseti_dataset.h5"))
# Default for backward compatibility
DATA_PATH = DATA_PATH_WEST
MAX_QUERY_CONTEXT_ROWS = int(os.getenv("UAP_QUERY_MAX_ROWS", "250"))
MAX_QUERY_CONTEXT_CHARS = int(os.getenv("UAP_QUERY_MAX_CHARS", "24000"))

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class FilterSpec(BaseModel):
    column: str
    type: str  # "categorical", "numeric", "text"
    values: list = None
    min_val: float = None
    max_val: float = None
    pattern: str = None


class AnalysisRequest(BaseModel):
    columns: list[str]
    # Cluster pipeline tuning (mirrors analyzing.py controls). When enable_tfidf
    # is False (the analyzing.py default), clusters keep numeric "Cluster N"
    # names and raw HDBSCAN labels are passed straight to XGBoost.
    enable_tfidf: bool = False
    min_cluster_size: int = 15
    n_neighbors: int = 15
    min_dist: float = 0.1
    top_n: int = 32


class QueryRequest(BaseModel):
    question: str
    columns: list[str]
    gemini_key: str


# ── Parsing ────────────────────────────────────────────────────────────────
class SchemaMergeRequest(BaseModel):
    labels: list[str]
    custom_fields: dict | None = None


class SchemaCoverageRequest(BaseModel):
    labels: list[str]
    custom_fields: dict | None = None
    # Dataset columns to diff against; defaults to the uploaded parse source.
    columns: list[str] | None = None


class ParseEstimateRequest(BaseModel):
    columns: list[str]
    format_json: str                   # merged schema template (JSON string)
    model: str = "gpt-4o-mini"
    use_cache: bool = True
    use_batch: bool = False


class ParseRunRequest(BaseModel):
    columns: list[str]
    format_json: str                   # merged schema template (JSON string)
    provider: str = "openai"           # "openai" | "deepseek"
    model: str = "gpt-4o-mini"
    api_key: str
    max_workers: int = 10
    keep_columns: list[str] = []       # carry-through source columns


# ── SCU ────────────────────────────────────────────────────────────────────
class ScuFilterRequest(BaseModel):
    criterion_keys: list[str]


# ── RAG ────────────────────────────────────────────────────────────────────
class RagSearchRequest(BaseModel):
    columns: list[str]
    question: str
    cohere_key: str
    top_n: int = 50


# ── Cramér's V explorer ────────────────────────────────────────────────────
class CramersVRequest(BaseModel):
    columns: list[str] | None = None
    drop_missing: bool = False
    exclude_trivial: bool = True
    strong_threshold: float = 0.30
    high_threshold: int = 30
    source: str = "dataset"            # "dataset" | "parsed"


class ContingencyRequest(BaseModel):
    col1: str
    col2: str
    drop_missing: bool = False
    source: str = "dataset"


class ColumnGroupsRequest(BaseModel):
    source: str = "dataset"            # "dataset" | "parsed"
    high_threshold: int = 30


class XgboostRequest(BaseModel):
    columns: list[str]
    source: str = "dataset"            # "dataset" | "parsed"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def df_to_json(df: pd.DataFrame, max_rows: int = 50000, total_rows: int | None = None) -> dict:
    df_subset = df.head(max_rows).copy()
    for col in df_subset.columns:
        # Convert categories and datetimes to strings
        if df_subset[col].dtype.name == "category" or pd.api.types.is_datetime64_any_dtype(df_subset[col]):
            df_subset[col] = df_subset[col].astype(str)
        else:
            # For object columns, we need to be careful with fillna if they contain dicts
            # We use a custom fillna for objects
            if df_subset[col].dtype == object:
                # Fill missing values without triggering hash checks on the content
                mask = df_subset[col].isna()
                df_subset.loc[mask, col] = ""
            else:
                df_subset[col] = df_subset[col].fillna("")
    return {
        "columns": list(df_subset.columns),
        "rows": df_subset.to_dict(orient="records"),
        # total_rows reflects the full dataset; pass it explicitly when df has
        # already been truncated upstream so the count is not misreported.
        "total_rows": total_rows if total_rows is not None else len(df),
        "returned_rows": len(df_subset),
    }

def get_column_stats(df: pd.DataFrame) -> list[dict]:
    stats = []
    for col in df.columns:
        # Standard info
        info = {
            "name": col, 
            "dtype": str(df[col].dtype), 
            "non_null": int(df[col].notna().sum())
        }
        
        # Safe nunique
        try:
            info["unique"] = int(df[col].nunique())
        except (TypeError, ValueError):
            # For unhashable types (dicts/lists), we count unique by converting to string first
            try:
                info["unique"] = int(df[col].astype(str).nunique())
            except Exception:
                info["unique"] = 0
            
        # Numerical stats
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                if df[col].notna().any():
                    info["min"] = float(df[col].min())
                    info["max"] = float(df[col].max())
                    info["mean"] = float(df[col].mean())
                else:
                    info["min"] = info["max"] = info["mean"] = None
            except Exception:
                pass
        # Categorical / Object stats
        elif pd.api.types.is_object_dtype(df[col]) or df[col].dtype.name == "category":
            try:
                top = df[col].value_counts().head(10)
                info["top_values"] = [{"value": str(k), "count": int(v)} for k, v in top.items()]
            except (TypeError, ValueError):
                # Handle unhashable types in value_counts by converting to string
                try:
                    top = df[col].astype(str).value_counts().head(10)
                    info["top_values"] = [{"value": str(k), "count": int(v)} for k, v in top.items()]
                except Exception:
                    info["top_values"] = []
            except Exception:
                info["top_values"] = []
                
        stats.append(info)
    return stats

def truncate_context(items: list[str], max_chars: int) -> tuple[str, int]:
    context_parts = []
    used = 0
    chars = 0
    for item in items:
        to_add = item if not context_parts else f"\\n{item}"
        if chars + len(to_add) > max_chars:
            break
        context_parts.append(item)
        chars += len(to_add)
        used += 1
    return "\\n".join(context_parts), used

# ---------------------------------------------------------------------------
# Data endpoints
# ---------------------------------------------------------------------------
@app.get("/api/data/load")
def load_data(
    rows: int = Query(default=15000, le=50000),
    type: str = Query(default="west"),
    state: SessionState = Depends(get_session)
):
    path = DATA_PATH_EAST if type == "east" else DATA_PATH_WEST

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {os.path.basename(path)}")
    try:
        df = pd.read_hdf(path, key="df")
        if "embeddings" in df.columns:
            df = df.drop(columns=["embeddings"])
        full_row_count = len(df)
        df = df.head(rows)
        state.dataset = df
        state.filtered_data = df
        state.data_processed = False
        state.col_names = []
        return {
            "status": "ok",
            "data": df_to_json(df, total_rows=full_row_count),
            "column_stats": get_column_stats(df),
        }
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error loading {type} dataset: {e}")

@app.get("/api/analysis/clusters")
def get_clusters_viz():
    from fastapi.responses import FileResponse
    path = os.path.join(os.path.dirname(__file__), "..", "frontend", "uap_clusters_llm.html")
    if not os.path.exists(path):
        # Fallback to current dir or project root
        path = os.path.join(os.path.dirname(__file__), "..", "uap_clusters_llm.html")
        
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Cluster visualization not found")
    return FileResponse(path)

@app.post("/api/data/upload")
async def upload_data(file: UploadFile = File(...), state: SessionState = Depends(get_session)):
    try:
        contents = await file.read()
        name = (file.filename or "").lower()
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        elif name.endswith(".json"):
            import json as _json
            obj = _json.loads(contents.decode("utf-8"))
            df = pd.json_normalize(obj if isinstance(obj, list) else [obj])
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type (CSV / XLSX / JSON).")
        state.dataset = df
        state.filtered_data = df
        state.data_processed = False
        return {"status": "ok", "filename": file.filename, "data": df_to_json(df), "column_stats": get_column_stats(df)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/filter")
def filter_data(filters: list[FilterSpec], state: SessionState = Depends(get_session)):
    if state.dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    df = state.dataset.copy()
    try:
        for f in filters:
            if f.column not in df.columns:
                continue
            if f.type == "categorical" and f.values:
                df = df[df[f.column].astype(str).isin([str(v) for v in f.values])]
            elif f.type == "numeric":
                if f.min_val is not None and f.max_val is not None:
                    if f.min_val > f.max_val:
                        raise ValueError(f"Invalid range: min {f.min_val} > max {f.max_val}")
                    df = df[df[f.column].between(f.min_val, f.max_val)]
                elif f.min_val is not None:
                    df = df[df[f.column] >= f.min_val]
                elif f.max_val is not None:
                    df = df[df[f.column] <= f.max_val]
            elif f.type == "text" and f.pattern:
                import re
                df = df[df[f.column].astype(str).str.contains(re.escape(f.pattern), case=False, na=False)]
        state.filtered_data = df
        return {"status": "ok", "data": df_to_json(df), "column_stats": get_column_stats(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Filtering failed: {e}")

@app.get("/api/data/columns")
def get_columns(state: SessionState = Depends(get_session)):
    if state.dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    df = state.dataset
    columns = []
    for col in df.columns:
        columns.append({
            "name": col, "dtype": str(df[col].dtype),
            "unique": int(df[col].nunique()), "non_null": int(df[col].notna().sum()),
        })
    return {"columns": columns}

@app.get("/api/data/column-values")
def get_column_values(
    column: str = Query(...),
    search: str = Query(default=""),
    limit: int = Query(default=50, le=500),
    state: SessionState = Depends(get_session),
):
    """Return unique values of a column, optionally filtered by a search term.

    Backs the searchable categorical filter so any value can be selected,
    not just the precomputed top-N from column statistics.
    """
    if state.dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    if column not in state.dataset.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found")

    import re
    series = state.dataset[column].astype(str)
    counts = series.value_counts()
    if search:
        mask = counts.index.str.contains(re.escape(search), case=False, na=False)
        counts = counts[mask]
    total_matches = int(len(counts))
    counts = counts.head(limit)
    return {
        "column": column,
        "values": [{"value": str(k), "count": int(v)} for k, v in counts.items()],
        "total_matches": total_matches,
    }

@app.post("/api/analyze/run")
def run_analysis(req: AnalysisRequest, state: SessionState = Depends(get_session)):
    if state.filtered_data is None or state.filtered_data.empty:
        raise HTTPException(status_code=400, detail="No data available")
    
    df = state.filtered_data
    valid_columns = [c for c in req.columns if c in df.columns]
    if not valid_columns:
        raise HTTPException(status_code=400, detail="No valid columns selected")
        
    results, cluster_viz, xgboost_results = {}, {}, {}
    state.analyzers = []
    new_data = pd.DataFrame()

    if state.analysis_mode == "production":
        # Codex #7: Run real embedding, UMAP, HDBSCAN cluster pipeline instead of generic mock
        from sklearn.model_selection import train_test_split
        from uap_analyzer import UAPAnalyzer, train_xgboost
        
        for col in valid_columns:
            analyzer = UAPAnalyzer(df, column=col)
            try:
                analyzer.preprocess_data(top_n=req.top_n)
                analyzer.reduce_dimensionality(
                    method="UMAP", n_components=2,
                    n_neighbors=req.n_neighbors, min_dist=req.min_dist,
                )
                analyzer.cluster_data(method="HDBSCAN", min_cluster_size=req.min_cluster_size)
                _labels = analyzer.__dict__["cluster_labels"]
                row_terms = None
                if req.enable_tfidf:
                    # TF-IDF naming + near-duplicate cluster merging. The merge
                    # path re-encodes on GPU and is best-effort; fall back to
                    # numeric labels if anything in it fails.
                    try:
                        analyzer.get_tf_idf_clusters(top_n=3)
                        row_terms = analyzer.merge_similar_clusters(
                            cluster_terms=analyzer.__dict__["cluster_terms"],
                            cluster_labels=analyzer.__dict__["cluster_labels"],
                        )
                    except Exception as merge_err:
                        logger.warning(f"TF-IDF naming failed for {col}, using raw labels: {merge_err}")
                        row_terms = None
                if row_terms is None:
                    # Numeric placeholder names; raw HDBSCAN labels flow to XGBoost.
                    row_terms = [f"Cluster {cid}" for cid in _labels]
                # cluster_terms is per-row here (length == len(df)) so the
                # cluster_viz masks below index reduced_embeddings correctly.
                analyzer.cluster_terms = row_terms

                new_data[f"Analyzer_{col}"] = analyzer.cluster_terms
                state.analyzers.append(analyzer)
                
                traces = []
                unique_terms = np.unique(analyzer.cluster_terms)
                for term in unique_terms:
                    mask = (np.array(analyzer.cluster_terms) == term)
                    if not mask.any(): continue
                    traces.append({
                        "name": term,
                        "x": analyzer.reduced_embeddings[mask, 0].tolist(),
                        "y": analyzer.reduced_embeddings[mask, 1].tolist(),
                        "text": df[col].iloc[mask].astype(str).tolist(),
                        "count": int(mask.sum()),
                    })
                cluster_viz[col] = {"traces": traces, "title": f"{col} Clusters (HDBSCAN)"}
                _dist = pd.Series(analyzer.cluster_terms).astype(str).value_counts().head(20)
                results[col] = {
                    "cluster_count": len(unique_terms),
                    "total_points": len(df),
                    "distribution": [{"label": str(k), "count": int(v)} for k, v in _dist.items()],
                }
            except Exception as e:
                logger.error(f"Error analyzing {col}: {e}")
                
        if len(state.analyzers) >= 2:
            new_data_cat = new_data.fillna('null').astype('category')
            data_nums = new_data_cat.apply(lambda x: x.cat.codes)
            for col in data_nums.columns:
                try:
                    categories = new_data_cat[col].cat.categories
                    x_train, x_test, y_train, y_test = train_test_split(data_nums.drop(columns=[col]), data_nums[col], test_size=0.2, random_state=42)
                    bst, accuracy, preds = train_xgboost(x_train, y_train, x_test, y_test, len(categories))
                    
                    importances_dict = bst.get_score(importance_type="gain")
                    importances = {k.replace("Analyzer_", ""): float(v) for k, v in importances_dict.items()}
                    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
                    xgboost_results[col.replace("Analyzer_", "")] = {
                        "feature_importance": importances,
                        "accuracy": round(accuracy, 3),
                    }
                except Exception as e:
                    logger.error(f"Error in xgboost for {col}: {e}")
                    
    else:
        # Mock mode
        for col in valid_columns:
            col_data = df[col].fillna("").astype(str)
            vc = col_data.value_counts()
            top_labels = vc.head(32).index.tolist()
            cmap = {l: i for i, l in enumerate(top_labels)}
            clabels = col_data.apply(lambda x: cmap.get(x, -1)).values
            
            n = len(col_data)
            np.random.seed(42)
            red_embeds = np.random.randn(n, 2)
            
            cluster_terms = []
            for i, label in enumerate(top_labels[:20]):
                mask = col_data == label
                red_embeds[mask.values] = (np.random.randn(2) * 3) + np.random.randn(mask.sum(), 2) * 0.5
                
            new_data[f"Analyzer_{col}"] = [top_labels[c] if 0 <= c < len(top_labels) else "Other" for c in clabels]
            
            traces = []
            for i, term in enumerate(top_labels[:20]):
                mask = (clabels == i)
                if mask.sum() == 0: continue
                traces.append({
                    "name": term,
                    "x": red_embeds[mask, 0].tolist(),
                    "y": red_embeds[mask, 1].tolist(),
                    "text": col_data[mask].tolist(),
                    "count": int(mask.sum()),
                })
            cluster_viz[col] = {"traces": traces, "title": f"{col} Clusters (Mock)"}
            results[col] = {"cluster_count": len(top_labels), "distribution": [{"label": str(k), "count": int(v)} for k, v in vc.head(20).items()], "total_points": n}
        
    cramers_data = None
    if len(valid_columns) >= 2:
        new_data_cat = new_data.fillna("null").astype("category")
        cols = list(new_data_cat.columns)
        from scipy.stats import chi2_contingency
        matrix = []
        for c1 in cols:
            row = []
            for c2 in cols:
                if c1 == c2:
                    row.append(1.0)
                else:
                    try:
                        ct = pd.crosstab(new_data_cat[c1], new_data_cat[c2])
                        chi2 = chi2_contingency(ct)[0]
                        n_obs = ct.sum().sum()
                        phi2 = chi2 / n_obs
                        r, k = ct.shape
                        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n_obs-1))
                        rcorr, kcorr = r - ((r-1)**2)/(n_obs-1), k - ((k-1)**2)/(n_obs-1)
                        denom = min(kcorr-1, rcorr-1)
                        v = np.sqrt(phi2corr / denom) if denom > 0 else 0.0
                        row.append(round(float(v), 3))
                    except Exception: row.append(0.0)
            matrix.append(row)
        cramers_data = {"labels": [c.replace("Analyzer_", "") for c in cols], "matrix": matrix}

    state.new_data = new_data
    state.data_processed = True
    state.analysis_results = results
    state.cluster_viz = cluster_viz
    state.cramers_v = cramers_data
    state.col_names = valid_columns
    state.analysis_runs += 1
    state.last_analysis_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    return {
        "status": "ok",
        "analysis_mode": state.analysis_mode,
        "mock_mode": state.analysis_mode == "mock",
        "warnings": ["Results generated by mock pipeline for demo purposes."] if state.analysis_mode == "mock" else [],
        "results": results,
        "cluster_viz": cluster_viz,
        "cramers_v": cramers_data,
        "xgboost": xgboost_results if xgboost_results else {},
        "processed_data": df_to_json(new_data),
    }

@app.get("/api/analyze/results")
def get_analysis_results(state: SessionState = Depends(get_session)):
    if not state.data_processed:
        raise HTTPException(status_code=400, detail="No analysis run yet")
    return {"results": state.analysis_results, "cluster_viz": state.cluster_viz, "cramers_v": state.cramers_v}

@app.post("/api/query/gemini")
def query_gemini(req: QueryRequest, state: SessionState = Depends(get_session)):
    if state.filtered_data is None:
        raise HTTPException(status_code=400, detail="No data")
    valid_cols = [c for c in req.columns if c in state.filtered_data.columns]
    if not valid_cols:
        raise HTTPException(status_code=400, detail="No valid columns selected")
    try:
        import google.generativeai as genai
        # Combine the selected columns per row with " - " (mirrors rag_search.py
        # and the parsing _build_text_series), then drop empty rows.
        text_series = _build_text_series(state.filtered_data, valid_cols)
        filtered = [t for t in text_series.dropna().tolist() if t.strip()]
        context_seed = filtered[:MAX_QUERY_CONTEXT_ROWS]
        # Token aware chunking
        context, rows_used = truncate_context(context_seed, MAX_QUERY_CONTEXT_CHARS)
        if not context:
            raise HTTPException(status_code=400, detail="No text available for querying")

        genai.configure(api_key=req.gemini_key)
        model = genai.GenerativeModel("models/gemini-3.1-pro-preview")
        response = model.generate_content([f"{req.question or 'Summarize'}\\nContext: {context}\\n\\n"])
        return {"status": "ok", "response": response.text,
                "context_rows_used": rows_used, "columns_used": valid_cols}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/summary")
def get_dashboard_summary(state: SessionState = Depends(get_session)):
    if state.dataset is None:
        return {"loaded": False, "analysis_runs": state.analysis_runs, "last_analysis_at": state.last_analysis_at}
    df = state.dataset
    return {
        "loaded": True,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "analyzed": state.data_processed,
        "analyzed_columns": len(state.col_names),
        "analysis_runs": state.analysis_runs,
        "last_analysis_at": state.last_analysis_at,
        "analysis_mode": state.analysis_mode,
        "null_counts": {col: int(df[col].isna().sum()) for col in df.columns}
    }

# ---------------------------------------------------------------------------
# Parsing — LLM feature extraction (parsing.py parity, core tier)
# ---------------------------------------------------------------------------
from api.services import parsing_service, scu_service, rag_service, analysis_service


def _build_text_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Concatenate the selected raw-text columns per row with ' - ' (mirrors
    parsing.py's _build_text), dropping empty cells."""
    def _row(row):
        parts = [
            str(row[c]).strip() for c in columns
            if c in row and pd.notna(row[c]) and str(row[c]).strip()
        ]
        return " - ".join(parts) if parts else None
    return df.apply(_row, axis=1)


@app.get("/api/parse/schemas")
def parse_schemas():
    try:
        return {**parsing_service.list_schemas(), "models": parsing_service.available_models()}
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Could not load schemas: {e}")


@app.post("/api/parse/schema-merge")
def parse_schema_merge(req: SchemaMergeRequest):
    try:
        return parsing_service.merge_schema(req.labels, req.custom_fields)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/parse/schema-coverage")
def parse_schema_coverage(req: SchemaCoverageRequest, state: SessionState = Depends(get_session)):
    """Diff the merged schema's leaf fields against the dataset columns (🟢/🔴
    coverage) and return per-mode extraction schemas (all / missing / database)."""
    cols = req.columns
    if cols is None:
        if state.parse_source_df is None:
            raise HTTPException(
                status_code=400,
                detail="No raw dataset uploaded and no columns provided.",
            )
        cols = list(state.parse_source_df.columns)
    try:
        return parsing_service.schema_coverage_report(req.labels, cols, req.custom_fields)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Coverage diff failed: {e}")


@app.post("/api/parse/upload")
async def parse_upload(file: UploadFile = File(...), state: SessionState = Depends(get_session)):
    """Upload a raw report dataset to be fed through the LLM extractor."""
    try:
        contents = await file.read()
        name = (file.filename or "").lower()
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        elif name.endswith(".json"):
            import json as _json
            raw = contents.decode("utf-8")
            obj = _json.loads(raw)
            df = pd.json_normalize(obj if isinstance(obj, list) else [obj])
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type (CSV / XLSX / JSON).")
        state.parse_source_df = df
        return {
            "status": "ok",
            "filename": file.filename,
            "data": df_to_json(df, max_rows=200),
            "columns": list(df.columns),
            "total_rows": len(df),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/parse/estimate")
def parse_estimate(req: ParseEstimateRequest, state: SessionState = Depends(get_session)):
    if state.parse_source_df is None:
        raise HTTPException(status_code=400, detail="No raw dataset uploaded. Upload reports first.")
    valid = [c for c in req.columns if c in state.parse_source_df.columns]
    if not valid:
        raise HTTPException(status_code=400, detail="No valid text columns selected.")
    texts = _build_text_series(state.parse_source_df, valid).dropna().tolist()
    if not texts:
        raise HTTPException(status_code=400, detail="Selected columns produced no text.")
    try:
        return parsing_service.estimate(texts, req.format_json, req.model,
                                        use_cache=req.use_cache, use_batch=req.use_batch)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not estimate cost: {e}")


@app.post("/api/parse/run")
def parse_run(req: ParseRunRequest, state: SessionState = Depends(get_session)):
    if state.parse_source_df is None:
        raise HTTPException(status_code=400, detail="No raw dataset uploaded. Upload reports first.")
    if not req.api_key:
        raise HTTPException(status_code=400, detail="An API key is required to run extraction.")
    src = state.parse_source_df
    valid = [c for c in req.columns if c in src.columns]
    if not valid:
        raise HTTPException(status_code=400, detail="No valid text columns selected.")

    text_series = _build_text_series(src, valid)
    texts = text_series.dropna().tolist()
    if not texts:
        raise HTTPException(status_code=400, detail="Selected columns produced no text to parse.")

    try:
        result = parsing_service.run_parse(
            texts, req.format_json, provider=req.provider, model=req.model,
            api_key=req.api_key, max_workers=req.max_workers,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Parsing failed: {e}")

    df = result["df"]
    # Carry-through columns: align on the raw-text input, like parsing.py.
    if req.keep_columns and not df.empty:
        keep = [c for c in req.keep_columns if c in src.columns]
        if keep:
            keep_src = src[keep].copy()
            keep_src.index = text_series.values
            keep_src = keep_src[keep_src.index.notna()]
            keep_src = keep_src[~keep_src.index.duplicated(keep="first")]
            df = df.join(keep_src, how="left")

    state.parsed_responses = result["parsed_responses"]
    state.parsed_df = df
    return {
        "status": "ok",
        "n_ok": result["n_ok"],
        "n_total": result["n_total"],
        "n_failed": len(result["errors"]),
        "errors": result["errors"][:10],
        "data": df_to_json(df, max_rows=2000),
    }


@app.get("/api/parse/result")
def parse_result(state: SessionState = Depends(get_session)):
    if state.parsed_df is None:
        raise HTTPException(status_code=400, detail="No parsed data in session.")
    return {"status": "ok", "data": df_to_json(state.parsed_df, max_rows=2000),
            "n_records": len(state.parsed_df)}


# ---------------------------------------------------------------------------
# SCU normalization (scu_normalizer parity)
# ---------------------------------------------------------------------------
@app.get("/api/scu/criteria")
def scu_criteria():
    try:
        return scu_service.criteria_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scu/normalize")
def scu_normalize(state: SessionState = Depends(get_session)):
    if not state.parsed_responses:
        raise HTTPException(
            status_code=400,
            detail="No parsed responses in session. Run the Parsing step first.",
        )
    try:
        result = scu_service.normalize(state.parsed_responses)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Normalization failed: {e}")
    state.scu_normalized_df = result["df"]
    return {
        "status": "ok",
        "metrics": result["metrics"],
        "audit_markdown": result["audit_markdown"],
        "data": df_to_json(result["df"], max_rows=5000),
    }


@app.post("/api/scu/filter")
def scu_filter(req: ScuFilterRequest, state: SessionState = Depends(get_session)):
    if state.scu_normalized_df is None:
        raise HTTPException(status_code=400, detail="No normalized data. Run SCU normalization first.")
    try:
        result = scu_service.filter_eligibility(state.scu_normalized_df, req.criterion_keys)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Filter failed: {e}")
    return {
        "status": "ok",
        "funnel": result["funnel"],
        "n_passed": result["n_passed"],
        "data": df_to_json(result["df"], max_rows=5000),
    }


# ---------------------------------------------------------------------------
# RAG search — Cohere rerank (rag_search.py Dataset RAG parity)
# ---------------------------------------------------------------------------
@app.post("/api/rag/search")
def rag_search(req: RagSearchRequest, state: SessionState = Depends(get_session)):
    df = state.filtered_data if state.filtered_data is not None else state.dataset
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No dataset loaded. Load data first.")
    if not req.cohere_key:
        raise HTTPException(status_code=400, detail="A Cohere API key is required.")
    try:
        result = rag_service.rerank(df, req.columns, req.question, req.cohere_key, top_n=req.top_n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail=f"Cohere rerank failed: {e}")
    return {
        "status": "ok",
        "n_results": result["n_results"],
        "searched_columns": result["searched_columns"],
        "data": df_to_json(result["df"], max_rows=req.top_n),
    }


# ---------------------------------------------------------------------------
# Cramér's V Categorical Association Explorer (analyzing.py parity)
# ---------------------------------------------------------------------------
def _assoc_source(state: SessionState, source: str) -> pd.DataFrame:
    if source == "parsed":
        if state.parsed_df is None:
            raise HTTPException(status_code=400, detail="No parsed data in session.")
        return state.parsed_df
    df = state.filtered_data if state.filtered_data is not None else state.dataset
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    return df


@app.post("/api/analysis/column-groups")
def analysis_column_groups(req: ColumnGroupsRequest, state: SessionState = Depends(get_session)):
    """Eligible categorical columns grouped by dotted parent, for the explorer's
    group selector. Cheap (cardinality only) so it loads before any matrix compute."""
    df = _assoc_source(state, req.source)
    try:
        return analysis_service.column_groups(df, high_threshold=req.high_threshold)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Column grouping failed: {e}")


@app.post("/api/analysis/xgboost")
def analysis_xgboost(req: XgboostRequest, state: SessionState = Depends(get_session)):
    """XGBoost feature importance computed directly on the selected categorical
    columns (no cluster pipeline) — fed by the Cramér's V explorer selection."""
    df = _assoc_source(state, req.source)
    try:
        return analysis_service.xgboost_importance(df, req.columns)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"XGBoost feature importance failed: {e}")


@app.post("/api/analysis/cramers-v")
def analysis_cramers_v(req: CramersVRequest, state: SessionState = Depends(get_session)):
    df = _assoc_source(state, req.source)
    try:
        return analysis_service.cramers_v_report(
            df, req.columns, drop_missing=req.drop_missing,
            exclude_trivial=req.exclude_trivial, strong_threshold=req.strong_threshold,
            high_threshold=req.high_threshold,
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Cramér's V failed: {e}")


@app.post("/api/analysis/contingency")
def analysis_contingency(req: ContingencyRequest, state: SessionState = Depends(get_session)):
    df = _assoc_source(state, req.source)
    try:
        return analysis_service.contingency(df, req.col1, req.col2, drop_missing=req.drop_missing)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contingency failed: {e}")


# ---------------------------------------------------------------------------
# Map and Magnetic Features
# ---------------------------------------------------------------------------
from fastapi.responses import HTMLResponse
from api.services.map_service import MapService

@app.get("/api/map/html")
def get_map_html(state: SessionState = Depends(get_session)):
    if state.filtered_data is None or state.filtered_data.empty:
        return HTMLResponse(
            "<html><body style='background:#121212;color:white;display:flex;"
            "justify-content:center;align-items:center;height:100vh;font-family:sans-serif;'>"
            "<h3>No data to display. Please load and filter your dataset first.</h3>"
            "</body></html>"
        )
    try:
        html, from_cache = MapService.get_or_generate(state.filtered_data)
        headers = {"X-Map-Cache": "HIT" if from_cache else "MISS"}
        return HTMLResponse(content=html, headers=headers)
    except Exception as e:
        logger.error(f"Map Error: {traceback.format_exc()}")
        return HTMLResponse(
            f"<html><body style='background:#121212;color:red;'>"
            f"<h3>Map generation failed: {e}</h3></body></html>"
        )


class MagneticRequest(BaseModel):
    lat_col: str
    lon_col: str
    date_col: str
    distance: int = 100

def _fig_to_data_uri(fig, dpi: int = 80) -> str:
    """Render a matplotlib figure to a base64-encoded PNG data URI on a dark background."""
    import io, base64
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# Max events scanned per request (each match is a live BGS API call). Synchronous
# endpoint, so this is kept small to bound request duration.
MAGNETIC_MAX_EVENTS = 25
MAGNETIC_TIME_BUDGET_S = 240


@app.post("/api/magnetic/run")
def run_magnetic(req: MagneticRequest, state: SessionState = Depends(get_session)):
    """Cross-reference UAP events with geomagnetic observatory data.

    For each event with valid coordinates and a post-1995 date, find the nearest
    INTERMAGNET/BGS observatory within ``distance`` km, fetch its X/Y/Z/S minute
    data around the event, and render a time-series graph. Also produces a
    FastDTW-aligned aggregate across every matched event.
    """
    if state.filtered_data is None or state.filtered_data.empty:
        raise HTTPException(status_code=400, detail="No data loaded. Load a dataset in the Data Explorer first.")

    df = state.filtered_data
    for label, col in (("Latitude", req.lat_col), ("Longitude", req.lon_col), ("Date", req.date_col)):
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"{label} column '{col}' not found in dataset.")

    import sys
    proj_root = os.path.dirname(os.path.dirname(__file__))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass
    import magnetic

    started = time.time()

    # 1. Observatory list from the BGS API
    try:
        stations = magnetic.get_stations()
        st_lat = pd.to_numeric(stations["Latitude"], errors="coerce").to_numpy()
        st_lon = pd.to_numeric(stations["Longitude"], errors="coerce").to_numpy()
    except Exception as e:
        logger.error(f"Magnetic: station fetch failed: {traceback.format_exc()}")
        raise HTTPException(status_code=502, detail=f"Could not reach the BGS observatory API: {e}")

    # 2. Candidate events: numeric coordinates + parseable, post-1995 dates
    work = pd.DataFrame({
        "lat": pd.to_numeric(df[req.lat_col], errors="coerce"),
        "lon": pd.to_numeric(df[req.lon_col], errors="coerce"),
        "date": df[req.date_col].apply(magnetic.parse_uap_date),
    }).dropna(subset=["lat", "lon", "date"])
    work = work[work["lat"].between(-90, 90) & work["lon"].between(-180, 180)]
    work = work[work["date"].dt.year >= 1995]
    candidates = len(work)
    if candidates == 0:
        raise HTTPException(
            status_code=400,
            detail="No events with valid coordinates and a post-1995 date. Check the selected columns.",
        )

    # 3. Per-event: nearest observatory -> fetch data -> render graph
    graphs, agg_data, agg_times = [], [], []
    scanned = skipped_no_station = skipped_no_data = 0

    for lat_, lon_, date_ in work[["lat", "lon", "date"]].itertuples(index=False):
        if scanned >= MAGNETIC_MAX_EVENTS or (time.time() - started) > MAGNETIC_TIME_BUDGET_S:
            break
        scanned += 1

        dists = np.array([
            magnetic.get_haversine_distance(lat_, lon_, a, b)
            for a, b in zip(st_lat, st_lon)
        ])
        nearest = int(np.nanargmin(dists))
        if not np.isfinite(dists[nearest]) or dists[nearest] > req.distance:
            skipped_no_station += 1
            continue
        station = stations.iloc[nearest]
        iaga = str(station["IagaCode"])

        d = pd.Timestamp(date_)
        if d.tz is not None:
            d = d.tz_localize(None)
        start = (d - pd.Timedelta(hours=12)).strftime("%Y-%m-%d")
        end = (d + pd.Timedelta(hours=48)).strftime("%Y-%m-%d")

        try:
            res = magnetic.get_data(iaga, start, end)
        except Exception:
            res = None
        dt = (res or {}).get("datetime") or []
        if not dt:
            skipped_no_data += 1
            continue

        n = len(dt)
        comp = {}
        for k in ("X", "Y", "Z", "S"):
            vals = res.get(k) or []
            comp[k] = (
                pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy()
                if len(vals) == n else np.full(n, np.nan)
            )
        if not any(np.isfinite(v).any() for v in comp.values()):
            skipped_no_data += 1
            continue

        dt_naive = pd.to_datetime(pd.Series(dt), utc=True, errors="coerce").dt.tz_localize(None)
        plotted = pd.DataFrame({"datetime": dt_naive, **comp})

        subtitle = (
            f"{d.date()}  |  {lat_:.3f}, {lon_:.3f}  |  "
            f"{station['Name']} ({iaga})  |  {dists[nearest]:.0f} km"
        )
        try:
            fig = magnetic.plot_data_custom(plotted.copy(), date=d, save_path=None, subtitle=subtitle)
            graphs.append({
                "title": f"{d.date()} — {station['Name']}",
                "station": str(station["Name"]),
                "iaga": iaga,
                "distance_km": round(float(dists[nearest]), 1),
                "event_date": str(d),
                "image": _fig_to_data_uri(fig),
            })
        except Exception:
            logger.error(f"Magnetic: plot failed for {iaga} {d}: {traceback.format_exc()}")
            skipped_no_data += 1
            continue

        agg = plotted.copy()
        agg["datetime"] = agg["datetime"].dt.tz_localize("UTC")
        agg_data.append(agg)
        agg_times.append(d.tz_localize("UTC"))

    # 4. FastDTW-aligned aggregate across every matched event
    aggregate = None
    if len(agg_data) >= 2:
        try:
            fig = magnetic.plot_average_timeseries_with_dtw(agg_data, agg_times, window_hours=12, save_path=None)
            aggregate = {
                "title": f"FastDTW-aligned average across {len(agg_data)} events",
                "image": _fig_to_data_uri(fig),
            }
        except Exception:
            logger.error(f"Magnetic: DTW aggregate failed: {traceback.format_exc()}")

    return {
        "status": "ok",
        "candidates": candidates,
        "scanned": scanned,
        "matched": len(graphs),
        "skipped_no_station": skipped_no_station,
        "skipped_no_data": skipped_no_data,
        "distance_km": req.distance,
        "elapsed_s": round(time.time() - started, 1),
        "graphs": graphs,
        "aggregate": aggregate,
        "events": len(graphs),
        "message": (
            f"Scanned {scanned} of {candidates} candidate events within {req.distance} km — "
            f"{len(graphs)} produced graphs, {skipped_no_station} had no observatory in range, "
            f"{skipped_no_data} returned no usable data."
        ),
    }
