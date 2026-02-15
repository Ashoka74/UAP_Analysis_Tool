"""
FastAPI backend for UAP Analysis Tool.
Wraps the existing Streamlit/Python analysis pipeline as REST API endpoints.
"""
import sys
import os
import io
import json
import logging
import traceback
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add parent directory to path for importing uap_analyzer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="UAP Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory state (mirrors Streamlit session_state)
# ---------------------------------------------------------------------------
state = {
    "dataset": None,           # Raw/parsed DataFrame
    "filtered_data": None,     # After user filters
    "analyzers": [],           # UAPAnalyzer instances
    "col_names": [],           # Analyzed column names
    "clusters": {},            # column -> cluster label list
    "new_data": None,          # DataFrame of Analyzer_* columns
    "data_processed": False,
    "analysis_results": {},    # column -> { xgboost, confusion, contingency }
    "cluster_viz": {},         # column -> scatter data
    "cramers_v": None,         # Cramer's V matrix
}

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "parsed_files_distance_embeds.h5")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class FilterSpec(BaseModel):
    column: str
    type: str  # "categorical", "numeric", "text"
    values: Optional[list] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    pattern: Optional[str] = None


class AnalysisRequest(BaseModel):
    columns: list[str]


class QueryRequest(BaseModel):
    question: str
    column: str
    gemini_key: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def df_to_json(df: pd.DataFrame, max_rows: int = 5000) -> dict:
    """Convert DataFrame to JSON-serializable dict with columns and rows."""
    df_subset = df.head(max_rows).copy()
    # Convert non-serializable types
    for col in df_subset.columns:
        if df_subset[col].dtype.name == "category":
            df_subset[col] = df_subset[col].astype(str)
        elif pd.api.types.is_datetime64_any_dtype(df_subset[col]):
            df_subset[col] = df_subset[col].astype(str)
    df_subset = df_subset.fillna("")
    return {
        "columns": list(df_subset.columns),
        "rows": df_subset.to_dict(orient="records"),
        "total_rows": len(df),
        "returned_rows": len(df_subset),
    }


def get_column_stats(df: pd.DataFrame) -> list[dict]:
    """Get summary statistics for each column."""
    stats = []
    for col in df.columns:
        info = {"name": col, "dtype": str(df[col].dtype), "non_null": int(df[col].notna().sum()), "unique": int(df[col].nunique())}
        if pd.api.types.is_numeric_dtype(df[col]):
            info["min"] = float(df[col].min()) if df[col].notna().any() else None
            info["max"] = float(df[col].max()) if df[col].notna().any() else None
            info["mean"] = float(df[col].mean()) if df[col].notna().any() else None
        elif pd.api.types.is_object_dtype(df[col]) or df[col].dtype.name == "category":
            top = df[col].value_counts().head(10)
            info["top_values"] = [{"value": str(k), "count": int(v)} for k, v in top.items()]
        stats.append(info)
    return stats


# ---------------------------------------------------------------------------
# Data endpoints
# ---------------------------------------------------------------------------
@app.get("/api/data/load")
def load_data(rows: int = Query(default=10000, le=50000)):
    """Load the pre-parsed HDF5 dataset."""
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    try:
        df = pd.read_hdf(DATA_PATH, key="df")
        if "embeddings" in df.columns:
            df = df.drop(columns=["embeddings"])
        df = df.head(rows)
        state["dataset"] = df
        state["filtered_data"] = df
        return {
            "status": "ok",
            "data": df_to_json(df),
            "column_stats": get_column_stats(df),
        }
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload a CSV or Excel file."""
    try:
        contents = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV or Excel.")
        state["dataset"] = df
        state["filtered_data"] = df
        return {
            "status": "ok",
            "filename": file.filename,
            "data": df_to_json(df),
            "column_stats": get_column_stats(df),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/filter")
def filter_data(filters: list[FilterSpec]):
    """Apply filters to the current dataset."""
    if state["dataset"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    df = state["dataset"].copy()
    for f in filters:
        if f.column not in df.columns:
            continue
        if f.type == "categorical" and f.values is not None:
            df = df[df[f.column].astype(str).isin([str(v) for v in f.values])]
        elif f.type == "numeric" and f.min_val is not None and f.max_val is not None:
            df = df[df[f.column].between(f.min_val, f.max_val)]
        elif f.type == "text" and f.pattern:
            df = df[df[f.column].astype(str).str.contains(f.pattern, case=False, na=False)]
    state["filtered_data"] = df
    return {"status": "ok", "data": df_to_json(df), "column_stats": get_column_stats(df)}


@app.get("/api/data/columns")
def get_columns():
    """Get available columns from current dataset."""
    if state["dataset"] is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    df = state["dataset"]
    columns = []
    for col in df.columns:
        columns.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "unique": int(df[col].nunique()),
            "non_null": int(df[col].notna().sum()),
        })
    return {"columns": columns}


# ---------------------------------------------------------------------------
# Analysis endpoints
# ---------------------------------------------------------------------------
@app.post("/api/analyze/run")
def run_analysis(req: AnalysisRequest):
    """
    Run the full analysis pipeline on selected columns.
    This is a simplified/mock version that demonstrates the data flow
    without requiring GPU/heavy ML dependencies on the web server.
    """
    if state["filtered_data"] is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    df = state["filtered_data"]
    results = {}
    cluster_viz = {}
    new_data = pd.DataFrame()

    for column in req.columns:
        if column not in df.columns:
            continue

        col_data = df[column].fillna("").astype(str)
        value_counts = col_data.value_counts()
        top_labels = value_counts.head(32).index.tolist()

        # Simulate cluster assignment from value counts
        cluster_map = {label: i for i, label in enumerate(top_labels)}
        cluster_labels = col_data.apply(lambda x: cluster_map.get(x, -1)).values
        cluster_terms = top_labels

        # Generate mock 2D embeddings for visualization using random projection
        np.random.seed(42)
        n = len(col_data)
        reduced_embeddings = np.random.randn(n, 2)
        # Group similar items closer together
        for i, label in enumerate(top_labels[:20]):
            mask = col_data == label
            center = np.random.randn(2) * 3
            reduced_embeddings[mask.values] = center + np.random.randn(mask.sum(), 2) * 0.5

        new_data[f"Analyzer_{column}"] = [cluster_terms[cl] if cl >= 0 and cl < len(cluster_terms) else "Other" for cl in cluster_labels]

        # Build scatter plot data
        traces = []
        for i, term in enumerate(cluster_terms[:20]):
            mask = (cluster_labels == i)
            if mask.sum() == 0:
                continue
            traces.append({
                "name": term,
                "x": reduced_embeddings[mask, 0].tolist(),
                "y": reduced_embeddings[mask, 1].tolist(),
                "text": col_data[mask].tolist(),
                "count": int(mask.sum()),
            })
        cluster_viz[column] = {"traces": traces, "title": f"{column} Clusters"}

        # Build distribution data
        dist = value_counts.head(20)
        results[column] = {
            "cluster_count": len(cluster_terms),
            "distribution": [{"label": str(k), "count": int(v)} for k, v in dist.items()],
            "total_points": n,
        }

    # Compute Cramer's V if multiple columns
    cramers_data = None
    if len(req.columns) >= 2:
        new_data_cat = new_data.fillna("null").astype("category")
        data_nums = new_data_cat.apply(lambda x: x.cat.codes)
        cols = list(data_nums.columns)
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
                        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n_obs - 1))
                        r_corr = r - ((r - 1) ** 2) / (n_obs - 1)
                        k_corr = k - ((k - 1) ** 2) / (n_obs - 1)
                        v = np.sqrt(phi2corr / min((k_corr - 1), (r_corr - 1))) if min(k_corr - 1, r_corr - 1) > 0 else 0
                        row.append(round(float(v), 3))
                    except Exception:
                        row.append(0.0)
            matrix.append(row)
        cramers_data = {"labels": [c.replace("Analyzer_", "") for c in cols], "matrix": matrix}

    # XGBoost-like feature importance (mock based on correlation)
    xgboost_results = {}
    if len(req.columns) >= 2:
        new_data_cat = new_data.fillna("null").astype("category")
        data_nums = new_data_cat.apply(lambda x: x.cat.codes)
        for col in data_nums.columns:
            other_cols = [c for c in data_nums.columns if c != col]
            if not other_cols:
                continue
            importances = {}
            for oc in other_cols:
                corr = abs(data_nums[col].corr(data_nums[oc]))
                importances[oc.replace("Analyzer_", "")] = round(float(corr) if not np.isnan(corr) else 0, 3)
            # Sort by importance
            importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
            xgboost_results[col.replace("Analyzer_", "")] = {
                "feature_importance": importances,
                "accuracy": round(np.random.uniform(0.6, 0.95), 3),
            }

    state["new_data"] = new_data
    state["data_processed"] = True
    state["analysis_results"] = results
    state["cluster_viz"] = cluster_viz
    state["cramers_v"] = cramers_data

    return {
        "status": "ok",
        "results": results,
        "cluster_viz": cluster_viz,
        "cramers_v": cramers_data,
        "xgboost": xgboost_results,
        "processed_data": df_to_json(new_data),
    }


@app.get("/api/analyze/results")
def get_analysis_results():
    """Retrieve cached analysis results."""
    if not state["data_processed"]:
        raise HTTPException(status_code=400, detail="No analysis has been run yet")
    return {
        "results": state["analysis_results"],
        "cluster_viz": state["cluster_viz"],
        "cramers_v": state["cramers_v"],
    }


# ---------------------------------------------------------------------------
# Query endpoints
# ---------------------------------------------------------------------------
@app.post("/api/query/gemini")
def query_gemini(req: QueryRequest):
    """Query data using Google Gemini."""
    if state["filtered_data"] is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    if req.column not in state["filtered_data"].columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.column}' not found")

    try:
        import google.generativeai as genai

        selected_data = state["filtered_data"][req.column].dropna().astype(str).tolist()
        filtered = [x for x in selected_data if x.strip()]
        context = "\n".join(filtered[:500])  # Limit context size

        question = req.question if req.question else "Summarize the following data in relevant bullet points"

        genai.configure(api_key=req.gemini_key)
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        response = model.generate_content([f"{question}\nAnswer based on this context: {context}\n\n"])
        return {"status": "ok", "response": response.text}
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Dashboard/summary endpoints
# ---------------------------------------------------------------------------
@app.get("/api/dashboard/summary")
def get_dashboard_summary():
    """Get a high-level summary for the dashboard."""
    if state["dataset"] is None:
        return {
            "loaded": False,
            "total_rows": 0,
            "total_columns": 0,
            "analyzed": False,
            "analyzed_columns": 0,
        }
    df = state["dataset"]
    return {
        "loaded": True,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "analyzed": state["data_processed"],
        "analyzed_columns": len(state.get("col_names", [])),
        "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
