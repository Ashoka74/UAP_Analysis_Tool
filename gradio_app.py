import os
import io
import json
import time
import math
import concurrent.futures
from datetime import timedelta
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import plotly.express as px
import plotly.graph_objects as go

import gradio as gr


def load_dataframe(uploaded_file: Optional[gr.File]) -> Tuple[pd.DataFrame, List[str]]:
    """Load CSV/XLSX into a DataFrame and return available column names."""
    if uploaded_file is None:
        return pd.DataFrame(), []

    # Gradio's File object provides name and path
    name = getattr(uploaded_file, "name", None) or ""
    path = getattr(uploaded_file, "name", None) or ""
    # If running from temp file handle
    file_like = getattr(uploaded_file, "file", None)

    try:
        if name.lower().endswith(".csv"):
            if file_like and not isinstance(file_like, (str, bytes)):
                df = pd.read_csv(file_like)
            else:
                df = pd.read_csv(path)
        elif name.lower().endswith((".xlsx", ".xls")):
            if file_like and not isinstance(file_like, (str, bytes)):
                df = pd.read_excel(file_like)
            else:
                df = pd.read_excel(path)
        else:
            # Attempt CSV first, then Excel
            try:
                df = pd.read_csv(path)
            except Exception:
                df = pd.read_excel(path)
    except Exception as e:
        raise gr.Error(f"Failed to read file: {e}")

    return df, df.columns.tolist()


def cohere_rerank(
    df: pd.DataFrame,
    columns_to_query: List[str],
    question: str,
    cohere_api_key: str,
    top_n: int = 5,
):
    """Rerank rows using Cohere's rerank API based on selected columns as documents."""
    if df is None or df.empty:
        raise gr.Error("Please upload a dataset first.")
    if not columns_to_query:
        raise gr.Error("Please select at least one column.")
    if not question:
        raise gr.Error("Please enter a question.")
    if not cohere_api_key:
        raise gr.Error("Please enter your Cohere API key.")

    import cohere  # Imported lazily to avoid dependency if not used

    documents = df[columns_to_query].to_dict("records")
    json_documents = [json.dumps(doc, ensure_ascii=False) for doc in documents]

    co = cohere.Client(api_key=cohere_api_key)
    try:
        results = co.rerank(
            model="rerank-english-v3.0",
            query=question,
            documents=json_documents,
            top_n=int(top_n),
            return_documents=True,
        )
    except Exception as e:
        raise gr.Error(f"Cohere rerank failed: {e}")

    reranked_indices = [res.index for res in results.results]
    reranked_scores = [res.relevance_score for res in results.results]

    out_df = df.iloc[reranked_indices].copy()
    out_df.insert(0, "rank", range(1, len(reranked_indices) + 1))
    out_df.insert(1, "relevance_score", reranked_scores)
    if "embeddings" in out_df.columns:
        out_df = out_df.drop(columns=["embeddings"])
    return out_df


def parse_with_openai(
    texts: List[str],
    json_schema: str,
    openai_api_key: str,
    model: str = "gpt-4o-mini",
    max_workers: int = 16,
    progress: Optional[gr.Progress] = None,
) -> dict:
    """Parse a list of texts into structured JSON using OpenAI responses."""
    if not openai_api_key:
        raise gr.Error("Please provide an OpenAI API key.")
    if not texts:
        return {}

    try:
        schema_obj = json.loads(json_schema)
    except Exception as e:
        raise gr.Error(f"Invalid JSON schema: {e}")

    os.environ["OPENAI_API_KEY"] = openai_api_key
    from openai import OpenAI

    client = OpenAI()

    def _call(text: str) -> Optional[str]:
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a helpful assistant which is tasked to help parse data."},
                    {
                        "role": "user",
                        "content": (
                            f"Input report: {text}\n\n"
                            f"Parse data following this json structure; leave missing data empty: {json_schema}  Output:"
                        ),
                    },
                ],
            )
            return resp.choices[0].message.content
        except Exception:
            return None

    results = {}
    total = len(texts)
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_call, t): t for t in texts}
        for fut in concurrent.futures.as_completed(futures):
            text = futures[fut]
            out = fut.result()
            if out:
                results[text] = out
            completed += 1
            if progress is not None:
                progress(completed / max(1, total))

    return results


def normalize_parsed_json(responses: dict, nested_field: Optional[str] = None) -> pd.DataFrame:
    """Turn dict of raw JSON strings into a normalized DataFrame."""
    parsed = {}
    for k, v in (responses or {}).items():
        try:
            obj = json.loads(v)
        except Exception:
            try:
                obj = json.loads(v.replace("'", '"'))
            except Exception:
                continue
        parsed[k] = obj

    if not parsed:
        return pd.DataFrame()

    df = pd.DataFrame(parsed).T
    if nested_field and nested_field in df.columns:
        flat = pd.json_normalize(df[nested_field])
        flat.index = df.index
        return flat
    return pd.json_normalize(df)


# ===== Magnetic functions (adapted from magnetic.py) =====
def get_stations() -> pd.DataFrame:
    base_url = (
        "https://imag-data.bgs.ac.uk:/GIN_V1/GINServices?Request=GetCapabilities&format=json"
    )
    response = requests.get(base_url, timeout=60)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame.from_dict(data.get("ObservatoryList", []))


def get_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def compare_stations(test_lat_lon, data_table, distance=1000, closest=False):
    table_updated = pd.DataFrame()
    distances = {}
    for lat, lon, names in data_table[["Latitude", "Longitude", "Name"]].values:
        harv_distance = get_haversine_distance(
            test_lat_lon[0], test_lat_lon[1], lat, lon
        )
        if harv_distance < distance:
            table_updated = pd.concat(
                [table_updated, data_table[data_table["Name"] == names]]
            )
            distances[names] = harv_distance
    if closest and distances:
        closest_station = min(distances, key=distances.get)
        table_updated = data_table[data_table["Name"] == closest_station].copy()
        table_updated["Distance"] = distances[closest_station]
    return table_updated


def get_data(IagaCode: str, start_date: str, end_date: str):
    # validate dates
    pd.to_datetime(start_date)
    pd.to_datetime(end_date)
    duration = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    params = {
        "Request": "GetData",
        "format": "PNG",
        "testObsys": "0",
        "observatoryIagaCode": IagaCode,
        "samplesPerDay": "minute",
        "publicationState": "Best available",
        "dataStartDate": start_date,
        "dataDuration": duration.days,
        "traceList": "1234",
        "colourTraces": "true",
        "pictureSize": "Automatic",
        "dataScale": "Automatic",
        "pdfSize": "21,29.7",
    }
    base_url_json = (
        "https://imag-data.bgs.ac.uk:/GIN_V1/GINServices?Request=GetData&format=json"
    )
    response = requests.get(base_url_json, params=params, timeout=120)
    response.raise_for_status()
    return response.json()


def clean_uap_data(dataset: pd.DataFrame, lat: str, lon: str, date: str):
    processed = dataset[dataset[[lat, lon, date]].notnull().all(axis=1)].copy()
    processed[lat] = pd.to_numeric(processed[lat], errors="coerce")
    processed[lon] = pd.to_numeric(processed[lon], errors="coerce")
    processed = processed.dropna(subset=[lat, lon])
    return processed


def plot_overlapped_timeseries(data_list, event_times, window_hours=12):
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    components = ["X", "Y", "Z", "S"]
    colors = ["red", "green", "blue", "black"]
    for i, component in enumerate(components):
        axs[i].set_ylabel(component)
        axs[i].grid(True, alpha=0.3)
        axs[i].set_title(f"{component}")
        axs[i].set_xlabel("Time Difference from Event (hours)")
        for (df, event_time) in zip(data_list, event_times):
            df = df.copy()
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
            event_time = pd.to_datetime(event_time).tz_localize(None)
            df["time_diff"] = (
                (df["datetime"] - event_time).dt.total_seconds() / 3600
            )
            df_window = df[(df["time_diff"] >= -window_hours) & (df["time_diff"] <= window_hours)]
            std = df_window[component].std() or 1.0
            df_window[component] = (df_window[component] - df_window[component].mean()) / std
            axs[i].plot(
                df_window["time_diff"],
                df_window[component],
                color=colors[i],
                alpha=0.7,
                linewidth=1,
            )
        axs[i].axvline(x=0, color="red", linewidth=2, linestyle="--")
        axs[i].set_xlim(-window_hours, window_hours)
    axs[-1].set_xlabel("Hours from Event")
    fig.suptitle("Overlapped Time Series of Components", fontsize=16)
    fig.tight_layout()
    return fig


def align_series(reference: np.ndarray, series: np.ndarray) -> np.ndarray:
    reference = reference.flatten()
    series = series.flatten()
    _, path = fastdtw(reference, series, dist=euclidean)
    aligned = np.zeros(len(reference))
    for ref_idx, series_idx in path:
        aligned[ref_idx] = series[series_idx]
    return aligned


def plot_average_timeseries_with_dtw(data_list, event_times, window_hours=12):
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    components = ["X", "Y", "Z", "S"]
    colors = ["red", "green", "blue", "black"]
    for i, component in enumerate(components):
        axs[i].set_ylabel(component)
        axs[i].grid(True, alpha=0.3)
        all_aligned_data = []
        reference_df = None
        for (df, event_time) in zip(data_list, event_times):
            df = df.copy()
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
            event_time = pd.to_datetime(event_time).tz_localize(None)
            df["time_diff"] = (
                (df["datetime"] - event_time).dt.total_seconds() / 3600
            )
            df_window = df[(df["time_diff"] >= -window_hours) & (df["time_diff"] <= window_hours)]
            df_window[component] = df_window[component] - df_window[component].mean()
            if reference_df is None:
                reference_df = df_window
                all_aligned_data.append(reference_df[component].values)
            else:
                try:
                    aligned = align_series(reference_df[component].values, df_window[component].values)
                    all_aligned_data.append(aligned)
                except Exception:
                    pass
        all_aligned_data = np.array(all_aligned_data)
        if all_aligned_data.size == 0:
            continue
        avg_data = np.mean(all_aligned_data, axis=0)
        std_data = np.std(all_aligned_data, axis=0)
        axs[i].plot(reference_df["time_diff"], avg_data, color=colors[i], label="Average")
        try:
            axs[i].fill_between(
                reference_df["time_diff"], avg_data - std_data, avg_data + std_data, color=colors[i], alpha=0.2
            )
        except Exception:
            pass
        axs[i].axvline(x=0, color="red", linewidth=2, linestyle="--")
        axs[i].set_xlim(-window_hours, window_hours)
    axs[-1].set_xlabel("Hours from Event")
    fig.suptitle("Average Time Series of Components (FastDTW Aligned)", fontsize=16)
    fig.tight_layout()
    return fig


def run_magnetic(df: pd.DataFrame, lon_col: str, lat_col: str, date_col: str, distance_km: int):
    if df is None or df.empty:
        raise gr.Error("Please upload a dataset first.")
    stations = get_stations()
    processed = clean_uap_data(df, lat_col, lon_col, date_col)
    all_data, all_event_times = [], []
    for lon_, lat_, date_ in processed[[lon_col, lat_col, date_col]].values:
        test_lat_lon = (lat_, lon_)
        try:
            str_date = pd.to_datetime(date_).strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            str_date = str(date_)
        twelve_hours = pd.Timedelta(hours=12)
        forty_eight_hours = pd.Timedelta(hours=48)
        try:
            str_date_start = (pd.to_datetime(str_date) - twelve_hours).strftime("%Y-%m-%dT%H:%M:%S")
            str_date_end = (pd.to_datetime(str_date) + forty_eight_hours).strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            continue
        try:
            new_dataset = compare_stations(test_lat_lon, stations, distance=distance_km, closest=True)
            if new_dataset.empty:
                continue
            test_ = get_data(new_dataset.iloc[0]["IagaCode"], str_date_start, str_date_end)
            if test_ and any(test_.get(k) for k in ["X", "Y", "Z", "S"]):
                plotted = pd.DataFrame({
                    "datetime": test_["datetime"],
                    "X": test_.get("X", []),
                    "Y": test_.get("Y", []),
                    "Z": test_.get("Z", []),
                    "S": test_.get("S", []),
                })
                if plotted[["X", "Y", "Z", "S"]].any().any():
                    all_data.append(plotted)
                    all_event_times.append(pd.to_datetime(date_))
        except Exception:
            continue
    fig_over, fig_dtw = None, None
    if all_data:
        fig_over = plot_overlapped_timeseries(all_data, all_event_times)
        fig_dtw = plot_average_timeseries_with_dtw(all_data, all_event_times)
    return processed, fig_over, fig_dtw


# ===== Analyzing (lightweight wrapper around UAPAnalyzer) =====
def run_analyzing(
    df: pd.DataFrame,
    text_column: str,
    dim_method: str = "PCA",
    cluster_method: str = "HDBSCAN",
    kmeans_k: int = 10,
):
    if df is None or df.empty:
        raise gr.Error("Please upload a dataset first.")
    if text_column not in df.columns:
        raise gr.Error("Select a valid text column.")
    from uap_analyzer import UAPAnalyzer
    analyzer = UAPAnalyzer(df[[text_column]].copy(), text_column)
    analyzer.preprocess_data(trim=False, has_embeddings=False)
    try:
        analyzer.reduce_dimensionality(method=dim_method, n_components=2)
    except Exception:
        analyzer.reduce_dimensionality(method="PCA", n_components=2)
    try:
        if cluster_method == "KMeans":
            analyzer.cluster_data(method="KMeans", n_clusters=int(kmeans_k))
        else:
            analyzer.cluster_data(method="HDBSCAN", min_cluster_size=10)
    except Exception:
        analyzer.cluster_data(method="KMeans", n_clusters=int(kmeans_k))
    labels = analyzer.cluster_labels
    emb = analyzer.reduced_embeddings
    texts = analyzer.data[text_column].astype(str).tolist()
    fig = go.Figure()
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        fig.add_trace(go.Scatter(
            x=emb[idx, 0], y=emb[idx, 1], mode="markers", name=f"Cluster {int(cid)}",
            text=[texts[i] for i in idx], hoverinfo="text", marker=dict(size=6, opacity=0.8)
        ))
    fig.update_layout(title="Embedding clusters", showlegend=True)
    out_df = df.copy()
    out_df["cluster_label"] = labels
    return fig, out_df


# ===== Map helpers =====
def run_map(df: pd.DataFrame, lat_col: str, lon_col: str, color_col: Optional[str] = None):
    if df is None or df.empty:
        raise gr.Error("Please upload a dataset first.")
    if lat_col not in df.columns or lon_col not in df.columns:
        raise gr.Error("Please select valid latitude and longitude columns.")
    sdf = df.copy()
    sdf[lat_col] = pd.to_numeric(sdf[lat_col], errors="coerce")
    sdf[lon_col] = pd.to_numeric(sdf[lon_col], errors="coerce")
    sdf = sdf.dropna(subset=[lat_col, lon_col])
    fig = px.scatter_mapbox(
        sdf,
        lat=lat_col,
        lon=lon_col,
        color=color_col if color_col and color_col in sdf.columns else None,
        hover_data=sdf.columns,
        zoom=1,
        height=700,
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


with gr.Blocks(title="UAP Analytics - Gradio Demo") as demo:
    gr.Markdown("## UAP Analytics - Gradio Demo")
    gr.Markdown(
        """
        This demo mirrors the core flows from the Streamlit app:
        - Smart-Search (Reranking with Cohere)
        - UAP Feature Extraction (Parsing with OpenAI)
        - Magnetic Anomaly Detection
        - Statistical Analysis
        - Interactive Map
        """
    )

    # Default dataset loader (final_ufoseti_dataset.h5)
    with gr.Accordion("Default Dataset", open=False):
        use_default = gr.Checkbox(
            label="Use default dataset (final_ufoseti_dataset.h5)", value=False
        )
        default_h5_path = gr.Textbox(
            label="HDF5 path", value="final_ufoseti_dataset.h5", interactive=True
        )
        default_h5_key = gr.Textbox(label="HDF5 key", value="df", interactive=True)
        load_default_btn = gr.Button("Load Default Dataset")
        default_df_state = gr.State(pd.DataFrame())
        default_df_view = gr.Dataframe(label="Default dataset preview", interactive=False)

        def _load_default(h5_path: str, key: str):
            try:
                df = pd.read_hdf(h5_path, key=key)
            except Exception as e:
                raise gr.Error(f"Failed to read HDF5: {e}")
            return df, df.head(200)

        load_default_btn.click(
            _load_default,
            inputs=[default_h5_path, default_h5_key],
            outputs=[default_df_state, default_df_view],
        )

    with gr.TabItem("Smart-Search (RAG)"):
        with gr.Row():
            file_uploader = gr.File(label="Upload CSV or XLSX", file_count="single")
            cohere_key = gr.Textbox(label="Cohere API Key", type="password")
        with gr.Row():
            columns_dropdown = gr.CheckboxGroup(label="Columns to include as documents", choices=[])
            question_box = gr.Textbox(label="Question", placeholder="Ask a question to rerank relevant rows")
            topn = gr.Slider(1, 20, value=5, step=1, label="Top N")
        with gr.Row():
            load_btn = gr.Button("Load Dataset")
            rerank_btn = gr.Button("Rerank")

        df_state = gr.State(pd.DataFrame())
        df_view = gr.Dataframe(label="Dataset", interactive=False, wrap=True)
        df_reranked = gr.Dataframe(label="Reranked Results", interactive=False, wrap=True)

        def _on_load(file: gr.File, use_def: bool, def_df: pd.DataFrame):
            if use_def and def_df is not None and not def_df.empty:
                df = def_df
            else:
                df, _ = load_dataframe(file)
            cols = df.columns.tolist()
            return df, gr.update(choices=cols, value=[]), df

        load_btn.click(
            _on_load,
            inputs=[file_uploader, use_default, default_df_state],
            outputs=[df_state, columns_dropdown, df_view],
        )

        def _on_rerank(df: pd.DataFrame, cols: List[str], q: str, key: str, n: int):
            return cohere_rerank(df, cols, q, key, n)

        rerank_btn.click(
            _on_rerank,
            inputs=[df_state, columns_dropdown, question_box, cohere_key, topn],
            outputs=[df_reranked],
        )

    with gr.TabItem("UAP Feature Extraction (Parsing)"):
        with gr.Row():
            parse_file = gr.File(label="Upload CSV or XLSX (raw narratives)")
        with gr.Row():
            col_select = gr.Dropdown(choices=[], label="Text column", interactive=True)
            openai_key = gr.Textbox(label="OpenAI API Key", type="password")
        with gr.Row():
            schema_text = gr.Textbox(
                label="JSON Schema (as JSON)",
                lines=14,
                value='{"sightingDetails": {"shape": "", "color": "", "speed": "", "other": ""}}',
            )
        with gr.Row():
            max_workers = gr.Slider(1, 32, value=16, step=1, label="Max parallel requests")
            nested_field = gr.Textbox(label="Nested field to normalize (optional)", value="sightingDetails")
            parse_btn = gr.Button("Parse Dataset")
            use_default_for_parsing = gr.Button("Use Default Dataset Here")

        parsed_state = gr.State({})
        parse_df_state = gr.State(pd.DataFrame())
        parsed_df_view = gr.Dataframe(label="Parsed DataFrame", interactive=False, wrap=True)
        download_btn = gr.DownloadButton(label="Download JSON", visible=False)

        def _load_parse_file(file: gr.File, use_def: bool, def_df: pd.DataFrame):
            if use_def and def_df is not None and not def_df.empty:
                df = def_df
                cols = df.columns.tolist()
            else:
                df, cols = load_dataframe(file)
            # filter object-like columns for choices
            text_like = [c for c in cols if str(df[c].dtype) in ("object", "string")]
            return df, gr.update(choices=text_like, value=(text_like[0] if text_like else None))

        parse_file.change(
            _load_parse_file,
            inputs=[parse_file, use_default, default_df_state],
            outputs=[parse_df_state, col_select],
        )

        def _use_default_for_parsing(use_def: bool, def_df: pd.DataFrame):
            if use_def and def_df is not None and not def_df.empty:
                df = def_df
                cols = df.columns.tolist()
                text_like = [c for c in cols if str(df[c].dtype) in ("object", "string")]
                return df, gr.update(choices=text_like, value=(text_like[0] if text_like else None))
            raise gr.Error("Default dataset not loaded. Toggle and load it in the Default Dataset accordion or upload a file.")

        use_default_for_parsing.click(
            _use_default_for_parsing,
            inputs=[use_default, default_df_state],
            outputs=[parse_df_state, col_select],
        )

        def _parse(df: pd.DataFrame, col: str, schema: str, key: str, workers: int, nested: str, progress=gr.Progress(track_tqdm=False)):
            if df is None or df.empty:
                raise gr.Error("Please load data first (upload a file or use the default dataset).")
            if not col:
                raise gr.Error("Please select the text column.")
            texts = [str(x) for x in df[col].tolist() if str(x) and str(x) != "nan"]
            raw = parse_with_openai(texts, schema, key, max_workers=int(workers), progress=progress)
            df_parsed = normalize_parsed_json(raw, nested_field=nested or None)
            json_bytes = json.dumps(raw, indent=2).encode("utf-8")
            return raw, df_parsed, gr.update(value=json_bytes, visible=True)

        parse_btn.click(
            _parse,
            inputs=[parse_df_state, col_select, schema_text, openai_key, max_workers, nested_field],
            outputs=[parsed_state, parsed_df_view, download_btn],
        )

    with gr.TabItem("Magnetic Anomaly Detection"):
        mag_file = gr.File(label="Upload CSV/XLSX (with lat/lon/date)")
        with gr.Row():
            mag_lat = gr.Dropdown(choices=[], label="Latitude column")
            mag_lon = gr.Dropdown(choices=[], label="Longitude column")
            mag_date = gr.Dropdown(choices=[], label="Date column")
            mag_dist = gr.Slider(10, 2000, value=100, step=10, label="Max station distance (km)")
        mag_load = gr.Button("Load for Magnetic")
        mag_df_state = gr.State(pd.DataFrame())
        mag_df_view = gr.Dataframe(label="Filtered dataset preview", interactive=False)
        mag_run = gr.Button("Run Magnetic Analysis")
        mag_plot1 = gr.Plot(label="Overlapped Timeseries")
        mag_plot2 = gr.Plot(label="DTW Average Timeseries")

        def _mag_load(file: gr.File, use_def: bool, def_df: pd.DataFrame):
            if use_def and def_df is not None and not def_df.empty:
                df = def_df
                cols = df.columns.tolist()
            else:
                df, cols = load_dataframe(file)
            # Guess lat/lon/date columns
            lat_guess = next((c for c in cols if "lat" in c.lower()), None)
            lon_guess = next((c for c in cols if ("lon" in c.lower() or "lng" in c.lower())), None)
            date_guess = next((c for c in cols if "date" in c.lower()), None)
            return (
                df,
                gr.update(choices=cols, value=lat_guess),
                gr.update(choices=cols, value=lon_guess),
                gr.update(choices=cols, value=date_guess),
                df.head(100),
            )

        mag_load.click(
            _mag_load,
            inputs=[mag_file, use_default, default_df_state],
            outputs=[mag_df_state, mag_lat, mag_lon, mag_date, mag_df_view],
        )

        def _mag_run(df: pd.DataFrame, lon: str, lat: str, date: str, dist: int):
            processed, fig_over, fig_dtw = run_magnetic(df, lon, lat, date, int(dist))
            return processed.head(200), fig_over, fig_dtw

        mag_run.click(
            _mag_run,
            inputs=[mag_df_state, mag_lon, mag_lat, mag_date, mag_dist],
            outputs=[mag_df_view, mag_plot1, mag_plot2],
        )

    with gr.TabItem("Statistical Analysis"):
        an_file = gr.File(label="Upload CSV/XLSX (text column)")
        an_load = gr.Button("Load for Analysis")
        an_df_state = gr.State(pd.DataFrame())
        an_text_col = gr.Dropdown(choices=[], label="Text column")
        dim_method = gr.Radio(["PCA", "UMAP"], value="PCA", label="Dimensionality reduction")
        cluster_method = gr.Radio(["HDBSCAN", "KMeans"], value="HDBSCAN", label="Clustering")
        kmeans_k = gr.Slider(2, 50, value=10, step=1, label="KMeans k")
        an_run = gr.Button("Run Analysis")
        an_plot = gr.Plot(label="Embedding clusters")
        an_out_df = gr.Dataframe(label="Clustered data (preview)")

        def _an_load(file: gr.File, use_def: bool, def_df: pd.DataFrame):
            if use_def and def_df is not None and not def_df.empty:
                df = def_df
                cols = df.columns.tolist()
            else:
                df, cols = load_dataframe(file)
            text_like = [c for c in cols if str(df[c].dtype) in ("object", "string")]
            return df, gr.update(choices=text_like, value=(text_like[0] if text_like else None))

        an_load.click(_an_load, inputs=[an_file, use_default, default_df_state], outputs=[an_df_state, an_text_col])

        def _an_run(df: pd.DataFrame, col: str, dm: str, cm: str, k: int):
            fig, out_df = run_analyzing(df, col, dm, cm, int(k))
            return fig, out_df.head(200)

        an_run.click(
            _an_run,
            inputs=[an_df_state, an_text_col, dim_method, cluster_method, kmeans_k],
            outputs=[an_plot, an_out_df],
        )

    with gr.TabItem("Interactive Map"):
        mp_file = gr.File(label="Upload CSV/XLSX (with lat/lon)")
        mp_load = gr.Button("Load for Map")
        mp_df_state = gr.State(pd.DataFrame())
        mp_lat = gr.Dropdown(choices=[], label="Latitude column")
        mp_lon = gr.Dropdown(choices=[], label="Longitude column")
        mp_color = gr.Dropdown(choices=[], label="Color by (optional)")
        mp_run = gr.Button("Generate Map")
        mp_plot = gr.Plot(label="Map")

        def _mp_load(file: gr.File, use_def: bool, def_df: pd.DataFrame):
            if use_def and def_df is not None and not def_df.empty:
                df = def_df
                cols = df.columns.tolist()
            else:
                df, cols = load_dataframe(file)
            lat_guess = next((c for c in cols if "lat" in c.lower()), None)
            lon_guess = next((c for c in cols if ("lon" in c.lower() or "lng" in c.lower())), None)
            return (
                df,
                gr.update(choices=cols, value=lat_guess),
                gr.update(choices=cols, value=lon_guess),
                gr.update(choices=[""] + cols, value=""),
            )

        mp_load.click(_mp_load, inputs=[mp_file, use_default, default_df_state], outputs=[mp_df_state, mp_lat, mp_lon, mp_color])

        def _mp_run(df: pd.DataFrame, lat: str, lon: str, color: Optional[str]):
            return run_map(df, lat, lon, color)

        mp_run.click(_mp_run, inputs=[mp_df_state, mp_lat, mp_lon, mp_color], outputs=[mp_plot])

        # (Parsing is already wired in its own tab above)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


