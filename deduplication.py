import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# Add repo root and api folder to sys.path so we can import dedup_service directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from api.services import dedup_service
except ImportError:
    import api.services.dedup_service as dedup_service

st.title("🧬 Deduplication Studio")
st.markdown("""
Identify near-duplicate reports, cross-database matching, and spatial/temporal entity resolution using **`microsoft/harrier-oss-v1-270m`**.
""")

# ---------------------------------------------------------------------------
# 1. Dataset Loading (Similar to RAG search / cross-db)
# ---------------------------------------------------------------------------
st.subheader("1. Active Sighting Dataset (DB1)")
datasett = st.file_uploader("Upload Sighting Dataset A (CSV or Excel)", type=["csv", "xlsx"], key="dedup_file_uploader")

# State Management for DataFrame
if datasett is not None:
    try:
        df = pd.read_csv(datasett) if datasett.type == "text/csv" else pd.read_excel(datasett)
        st.session_state['parsed_responses'] = df
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Fallback to default UAP database if none uploaded/active
if 'parsed_responses' not in st.session_state or st.session_state['parsed_responses'] is None:
    default_csv = "SUBDATASETS_V2/PURSUE_1_2_3_normalized_full.csv"
    if os.path.exists(default_csv):
        st.session_state['parsed_responses'] = pd.read_csv(default_csv)
    else:
        st.session_state['parsed_responses'] = pd.DataFrame({
            "id": ["MOCK-1", "MOCK-2"],
            "witness.notes": ["A bright sphere hovered over the lake.", "Metallic orb seen near the water."],
            "date_time": ["2026-07-11 21:00:00", "2026-07-11 21:05:00"],
            "latitude": [34.0522, 34.0530],
            "longitude": [-118.2437, -118.2440]
        })

df_active = st.session_state['parsed_responses']
st.success(f"Active Dataset DB1 Loaded: `{df_active.shape[0]:,}` rows × `{df_active.shape[1]}` columns")
with st.expander("Preview DB1 Data", expanded=False):
    st.dataframe(df_active.head(5), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 2. Embedding Column Selection (Similar to RAG search / cross-db)
# ---------------------------------------------------------------------------
st.subheader("2. Semantic Representation Config")
default_cols = [c for c in ["witness.notes", "description", "case_text.text", "sightingDetails.observerDetails.observerBackground.occupation"] if c in df_active.columns]
if not default_cols:
    default_cols = [df_active.columns[0]]

ec1, ec2 = st.columns([2, 3])
with ec1:
    embed_model_name = st.selectbox(
        "Embedding Model",
        options=dedup_service.EMBEDDING_MODEL_OPTIONS,
        help="Harrier is asymmetric (encode_document/encode_query) and matches "
             "the rest of the app's clustering; the MiniLM options are lighter "
             "and multilingual-capable.",
        key="dedup_model_name",
    )
with ec2:
    embed_cols = st.multiselect(
        "Embedding Columns for DB1 (Concatenated to build narrative vector)",
        options=list(df_active.columns),
        default=default_cols,
        help="Select one or more text columns that describe the sighting narrative."
    )

# ---------------------------------------------------------------------------
# 3. Field Mapping (Date & Location) — the date/distance gates used to look
# for hardcoded column names (date_time/date, latitude/lat, longitude/lon)
# and silently never fire if a dataset used different names. These selectors
# let you point at the real columns; auto-detected defaults below keep
# existing datasets working with no changes.
# ---------------------------------------------------------------------------
st.subheader("3. Field Mapping (Date & Location)")
st.caption("Auto-detected from common column names — override if your columns are named differently.")

_DATE_COL_HINTS = ["date_time", "date", "sighting_date", "sightingDetails.date", "date_time.year"]
_LAT_COL_HINTS = ["latitude", "lat", "sightingDetails.location.latitude"]
_LON_COL_HINTS = ["longitude", "lon", "lng", "sightingDetails.location.longitude"]
_LOCATION_NAME_HINTS = ["location.name", "sightingDetails.location.name", "location_name", "city", "city_name", "location.city"]
_STATE_COL_HINTS = ["location.state", "location_state_norm", "state", "sightingDetails.location.state"]
_CITY_COL_HINTS = ["location.city", "city", "city_name", "sightingDetails.location.city"]

_auto_date_col = next((c for c in _DATE_COL_HINTS if c in df_active.columns), None)
_auto_lat_col = next((c for c in _LAT_COL_HINTS if c in df_active.columns), None)
_auto_lon_col = next((c for c in _LON_COL_HINTS if c in df_active.columns), None)
_auto_loc_name_col = next((c for c in _LOCATION_NAME_HINTS if c in df_active.columns), None)
_auto_state_col = next((c for c in _STATE_COL_HINTS if c in df_active.columns), None)
_auto_city_col = next((c for c in _CITY_COL_HINTS if c in df_active.columns), None)

fm1, fm2 = st.columns([1, 2])
with fm1:
    _date_options = ["(none — no date gating)"] + list(df_active.columns)
    _date_idx = _date_options.index(_auto_date_col) if _auto_date_col in _date_options else 0
    _date_sel = st.selectbox("Date Column", options=_date_options, index=_date_idx, key="dedup_date_col")
    date_col = None if _date_sel.startswith("(none") else _date_sel

with fm2:
    location_mode = st.radio(
        "Location Field Type",
        ["Lat/Lon Columns (precise distance)", "Location Name Column (text match — no coordinates needed)"],
        horizontal=True,
        index=0 if (_auto_lat_col and _auto_lon_col) else (1 if _auto_loc_name_col else 0),
        key="dedup_location_mode",
    )
    lat_col = lon_col = location_col = state_col = city_col = None
    use_gazetteer = False
    if location_mode.startswith("Lat/Lon"):
        lc1, lc2 = st.columns(2)
        _coord_options = ["(none)"] + list(df_active.columns)
        with lc1:
            _lat_sel = st.selectbox("Latitude Column", options=_coord_options,
                                     index=_coord_options.index(_auto_lat_col) if _auto_lat_col in _coord_options else 0,
                                     key="dedup_lat_col")
            lat_col = None if _lat_sel == "(none)" else _lat_sel
        with lc2:
            _lon_sel = st.selectbox("Longitude Column", options=_coord_options,
                                     index=_coord_options.index(_auto_lon_col) if _auto_lon_col in _coord_options else 0,
                                     key="dedup_lon_col")
            lon_col = None if _lon_sel == "(none)" else _lon_sel
    else:
        _loc_options = ["(none)"] + list(df_active.columns)
        loc_c1, loc_c2, loc_c3 = st.columns(3)
        with loc_c1:
            _loc_sel = st.selectbox(
                "Location Name Column", options=_loc_options,
                index=_loc_options.index(_auto_loc_name_col) if _auto_loc_name_col in _loc_options else 0,
                key="dedup_location_col",
                help="Free-text place description. Matched via fuzzy text by default, or parsed "
                     "for the gazetteer lookup below when no City Column is set.",
            )
            location_col = None if _loc_sel == "(none)" else _loc_sel
        with loc_c2:
            _city_sel = st.selectbox(
                "City Column (optional, preferred)", options=_loc_options,
                index=_loc_options.index(_auto_city_col) if _auto_city_col in _loc_options else 0,
                key="dedup_city_col",
                help="A clean, already-separated city name is a far more reliable gazetteer key "
                     "than one parsed out of a free-text Location Name sentence (e.g. 'Parking "
                     "lot back of police station, Portland, Oregon'). Falls back to Location "
                     "Name when left as (none).",
            )
            city_col = None if _city_sel == "(none)" else _city_sel
        with loc_c3:
            _state_sel = st.selectbox(
                "State Column (optional)", options=_loc_options,
                index=_loc_options.index(_auto_state_col) if _auto_state_col in _loc_options else 0,
                key="dedup_state_col",
                help="Pairs with City Column (preferred) or Location Name Column for gazetteer "
                     "matching (many city names repeat across states). If left as (none) and "
                     "Location Name contains a combined 'City, ST' string, the state is parsed "
                     "out of that instead.",
            )
            state_col = None if _state_sel == "(none)" else _state_sel
        use_gazetteer = st.toggle(
            "Use offline US Census Gazetteer lookup (city/state → real lat/lon centroid)",
            value=False, key="dedup_use_gazetteer",
            help="Opt-in: it's US-only and a bare place name can collide across states/"
                 "countries (~32k places, no API calls, no rate limit), so it doesn't "
                 "silently activate for every location-name dataset. Off falls back to "
                 "fuzzy text matching on the Location Name Column only.",
        )

# ---------------------------------------------------------------------------
# Cluster-Blocked Comparison — the pairwise search in tabs 2/3 normally
# builds a dense len(rows) x len(rows) similarity matrix, which stops being
# feasible somewhere in the tens of thousands of rows (100k x 100k floats is
# 40GB before the matmul even runs). Turning this on pre-groups rows with
# UMAP+HDBSCAN (the same technique the Analysis tab's cluster pipeline uses)
# and only compares rows within the same group — trading a small recall
# risk near group boundaries for comparisons that scale with group size
# instead of dataset size. Off by default so nothing changes silently.
# ---------------------------------------------------------------------------
with st.expander("⚡ Cluster-Blocked Comparison (for large datasets)"):
    use_cluster_blocking = st.toggle(
        "Pre-group rows by embedding similarity before pairwise comparison",
        value=False, key="dedup_use_blocking",
        help="Recommended past roughly 20-30k rows, where the full comparison matrix gets "
             "expensive. Leave off for smaller datasets — a true duplicate pair split across "
             "two groups near a boundary would be missed.",
    )
    block_min_cluster_size = st.number_input(
        "Min group size", min_value=2, max_value=500, value=15, step=1,
        key="dedup_block_min_size", disabled=not use_cluster_blocking,
    )

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "🔍 Simple Pair Checker (`is_similar` / `is_duplicate`)",
    "⚙️ Batch Cluster Pipeline",
    "🧬 Cross-DB Pipeline (`Easy Path` vs `Hard Path`)"
])

# Helper to build text representation for a row
def get_concatenated_row_text(row, cols):
    parts = []
    for col in cols:
        val = str(row.get(col, "")).strip()
        if val and val.lower() != "nan" and val.lower() != "none":
            parts.append(val)
    return " - ".join(parts) if parts else ""

# ---------------------------------------------------------------------------
# TAB 1: SIMPLE PAIR CHECKER
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Pairwise Sighting Screening")
    st.markdown("Compare two reports to calculate semantic similarity and spatial/temporal gates.")
    
    input_mode = st.radio("Input Mode", ["Compare Two Rows from Active Dataset", "Input Texts Manually"], horizontal=True, key="simple_mode_sel")
    
    narr_a, date_a, lat_a, lon_a = "", "", "", ""
    narr_b, date_b, lat_b, lon_b = "", "", "", ""
    
    if input_mode == "Compare Two Rows from Active Dataset":
        col_sel_a, col_sel_b = st.columns(2)
        with col_sel_a:
            idx_a = st.number_input("Row A Index", min_value=0, max_value=len(df_active)-1, value=0, step=1, key="idx_a")
            row_a = df_active.iloc[idx_a]
            
            st.markdown("**Record A Preview:**")
            narr_a = get_concatenated_row_text(row_a, embed_cols)
            st.info(f"**Narrative (concatenated):** {narr_a[:200]}...")
            
            date_col = next((c for c in ["date_time", "date", "date_time.year"] if c in df_active.columns), "")
            lat_col = next((c for c in ["latitude", "lat", "sightingDetails.location.latitude"] if c in df_active.columns), "")
            lon_col = next((c for c in ["longitude", "lon", "sightingDetails.location.longitude"] if c in df_active.columns), "")
            
            date_a = str(row_a.get(date_col, "")) if date_col else ""
            lat_a = str(row_a.get(lat_col, "")) if lat_col else ""
            lon_a = str(row_a.get(lon_col, "")) if lon_col else ""
            st.caption(f"Date: `{date_a}` | Lat: `{lat_a}` | Lon: `{lon_a}`")

        with col_sel_b:
            idx_b = st.number_input("Row B Index", min_value=0, max_value=len(df_active)-1, value=min(1, len(df_active)-1), step=1, key="idx_b")
            row_b = df_active.iloc[idx_b]
            
            st.markdown("**Record B Preview:**")
            narr_b = get_concatenated_row_text(row_b, embed_cols)
            st.info(f"**Narrative (concatenated):** {narr_b[:200]}...")
            
            date_b = str(row_b.get(date_col, "")) if date_col else ""
            lat_b = str(row_b.get(lat_col, "")) if lat_col else ""
            lon_b = str(row_b.get(lon_col, "")) if lon_col else ""
            st.caption(f"Date: `{date_b}` | Lat: `{lat_b}` | Lon: `{lon_b}`")
            
    else:
        colA, colB = st.columns(2)
        with colA:
            st.markdown("### 🛸 Record A")
            narr_a = st.text_area("Narrative Description A", value="A large triangular craft hovered silently over Phoenix.", height=120, key="manual_narr_a")
            c1, c2, c3 = st.columns(3)
            with c1: date_a = st.text_input("Date A", value="1997-03-13", key="manual_date_a")
            with c2: lat_a = st.text_input("Lat A", value="33.4484", key="manual_lat_a")
            with c3: lon_a = st.text_input("Lon A", value="-112.0740", key="manual_lon_a")

        with colB:
            st.markdown("### 🛸 Record B")
            narr_b = st.text_area("Narrative Description B", value="Silent triangular UFO with lights near Phoenix on same evening.", height=120, key="manual_narr_b")
            c1, c2, c3 = st.columns(3)
            with c1: date_b = st.text_input("Date B", value="1997-03-13", key="manual_date_b")
            with c2: lat_b = st.text_input("Lat B", value="33.4500", key="manual_lat_b")
            with c3: lon_b = st.text_input("Lon B", value="-112.0710", key="manual_lon_b")

    st.markdown("---")
    btn_sim, btn_dup = st.columns([1, 1])
    
    with btn_sim:
        if st.button("✨ Check Similarity (is_similar)", type="secondary", use_container_width=True, key="btn_sim_checker"):
            with st.spinner(f"Computing {embed_model_name.rsplit('/', 1)[-1]} vector cosine similarity..."):
                res = dedup_service.check_similarity(narr_a, narr_b, model_name=embed_model_name)
                score = res.get("similarity_score", 0.0)
                is_sim = res.get("is_similar", False)
                st.markdown("#### Semantic Verdict")
                if is_sim:
                    st.success(f"**HIGH SIMILARITY (`is_similar = True`)** — Cosine Score: `{score:.4f}`")
                else:
                    st.warning(f"**BELOW THRESHOLD (`is_similar = False`)** — Cosine Score: `{score:.4f}`")
                st.progress(min(1.0, max(0.0, score)))
                st.caption(res.get("reason", ""))

    with btn_dup:
        if st.button("🧬 Screen Sighting Pair (is_duplicate)", type="primary", use_container_width=True, key="btn_dup_checker"):
            with st.spinner("Evaluating semantic, spatial, and temporal convergence..."):
                rec_a = {"narrative": narr_a, "date": date_a, "lat": lat_a, "lon": lon_a}
                rec_b = {"narrative": narr_b, "date": date_b, "lat": lat_b, "lon": lon_b}
                res = dedup_service.check_duplicate(rec_a, rec_b, model_name=embed_model_name)
                
                is_dup = res.get("is_duplicate", False)
                conf = res.get("confidence", "LOW")
                metrics = res.get("metrics", {})
                
                st.markdown("#### Multi-Gate Verdict")
                if is_dup:
                    st.success(f"**TRUE DUPLICATE (`is_duplicate = True`)** | Confidence: **{conf}**")
                else:
                    st.error(f"**DISTINCT EVENT (`is_duplicate = False`)** | Confidence: **{conf}**")
                    
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Similarity Score", f"{metrics.get('similarity_score', 0.0)*100:.1f}%")
                mc2.metric("Spatial Separation", f"{metrics.get('haversine_km')} km" if metrics.get('haversine_km') is not None else "Unknown")
                mc3.metric("Temporal Diff", f"{metrics.get('date_diff_days')} days" if metrics.get('date_diff_days') is not None else "Unknown")
                st.markdown("##### Decision Audit Trail")
                for r in res.get("reasons", []):
                    st.markdown(f"- ✅ {r}")

# ---------------------------------------------------------------------------
# TAB 2: ADVANCED DEDUPE STUDIO
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Multi-Gate Batch Clustering Pipeline")
    st.markdown(
        "Runs live over the active dataset (DB1) using the embedding columns selected above — two "
        "chained but separately-reported stages: **Stage A** screens every candidate pair through the "
        "same multi-gate engine as the Cross-DB pipeline; **Stage B** unions only the full-convergence "
        "pairs (text + date + location all agree) into clusters via Union-Find and picks one canonical "
        "row per cluster. Both stages stay visible below so each is independently auditable."
    )

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1: thresh = st.slider("Cosine Similarity Threshold", min_value=0.60, max_value=0.95, value=0.80, step=0.01, key="advanced_thresh")
    with sc2: days = st.slider("Max Temporal Diff (Days)", min_value=0, max_value=30, value=3, step=1, key="advanced_days")
    with sc3: km = st.slider("Max Spatial Radius (Haversine km)", min_value=5.0, max_value=200.0, value=50.0, step=5.0, key="advanced_km")
    with sc4:
        st.markdown("<br>", unsafe_allow_html=True)
        use_llm = st.toggle(
            "Enable LLM Tier-3 Judge", value=False,
            help="Reserved for a future Gemini confirmation pass on borderline pairs — not yet wired up; has no effect on results.",
            key="advanced_use_llm",
        )

    st.markdown("---")
    if st.button("🚀 Run Advanced Deduplication & Clustering", type="primary", key="btn_run_advanced"):
        with st.spinner(f"Screening candidates via {embed_model_name.rsplit('/', 1)[-1]} matrix embeddings, "
                       "then union-find graph clustering on full-convergence pairs..."):
            st.session_state['advanced_pipeline_results'] = dedup_service.run_advanced_dedup(
                records=df_active.to_dict("records"),
                cols=embed_cols,
                threshold=thresh,
                date_diff_days=days,
                max_km=km,
                model_name=embed_model_name,
                use_llm_judge=use_llm,
                date_col=date_col,
                lat_col=lat_col,
                lon_col=lon_col,
                location_col=location_col,
                state_col=state_col,
                city_col=city_col,
                use_gazetteer=use_gazetteer,
                use_cluster_blocking=use_cluster_blocking,
                block_min_cluster_size=block_min_cluster_size,
            )

    if 'advanced_pipeline_results' in st.session_state:
        adv_res = st.session_state['advanced_pipeline_results']
        if adv_res.get("status") != "success":
            st.error(adv_res.get("message", "Advanced deduplication failed."))
        else:
            summary = adv_res.get("summary", {})
            pairs = adv_res.get("pairs", [])
            clusters = adv_res.get("clusters", [])

            kc1, kc2, kc3, kc4 = st.columns(4)
            kc1.metric("Total Clusters Formed", summary.get("total_clusters", 0))
            kc2.metric("Rows in Clusters", summary.get("rows_in_clusters", 0))
            kc3.metric("Redundant Rows Saved", summary.get("redundant_rows_saved", 0))
            kc4.metric("Candidate Pairs Screened", summary.get("total_pairs_evaluated", len(pairs)))

            _bstats = summary.get("block_stats")
            if _bstats:
                st.caption(
                    f"⚡ Cluster-blocked: {_bstats['block_count']} groups cut the comparison matrix "
                    f"from {_bstats['full_matrix_cells']:,} to {_bstats['compared_cells']:,} cells"
                    + (f" ({_bstats['noise_block_size']} rows ungrouped)." if _bstats['noise_block_size'] > 0 else ".")
                )

            st.markdown("#### 🔍 Stage A — Multi-Gate Pairwise Audit Trail")
            st.caption("Every candidate pair the screening engine evaluated, with each gate's individual verdict.")
            if pairs:
                df_pairs = pd.DataFrame([{
                    "ID A": p["id_a"], "ID B": p["id_b"],
                    "Similarity": f"{p['similarity']*100:.1f}%",
                    "Bin": p["bin"],
                    "Distance (km)": p["haversine_km"],
                    "Date Diff (days)": p["date_diff_days"],
                    "Similar Text": p["flags"]["is_similar_text"],
                    "Similar Date": p["flags"]["is_similar_date"],
                    "Similar Location": p["flags"]["is_similar_location"],
                    "Similar All (Full Convergence)": p["flags"]["is_similar_all"],
                } for p in pairs])
                st.dataframe(df_pairs, use_container_width=True)
            else:
                st.info("No candidate pairs matched the criteria.")

            st.markdown("#### 🧩 Stage B — Canonical Clusters")
            st.caption("Connected components over the full-convergence pairs above, one canonical (most-complete) row picked per cluster.")
            if clusters:
                df_clusters = pd.DataFrame([{
                    "Cluster ID": c["cluster_id"],
                    "Size": c["size"],
                    "Canonical Row ID": c["canonical_id"],
                    "Member IDs": ", ".join(c["member_ids"]),
                } for c in clusters])
                st.dataframe(df_clusters, use_container_width=True)
                with st.expander("Preview a cluster's members (selected columns)"):
                    pick = st.selectbox("Cluster", options=[c["cluster_id"] for c in clusters], key="advanced_cluster_pick")
                    match = next((c for c in clusters if c["cluster_id"] == pick), None)
                    if match:
                        member_previews = match.get("member_previews", [])
                        cols = match.get("preview_cols") or sorted({
                            k for m in member_previews for k in m["values"].keys()
                        })
                        if member_previews and cols:
                            df_preview = pd.DataFrame(
                                {f"{m['id']}{' (canonical)' if m['id'] == match['canonical_id'] else ''}": [m["values"].get(col) for col in cols]
                                 for m in member_previews},
                                index=cols,
                            )
                            st.dataframe(df_preview, use_container_width=True)
                        else:
                            st.info("No preview columns available for this cluster.")
            else:
                st.info("No clusters formed — no pairs reached full convergence (text + date + location) at the current settings.")

# ---------------------------------------------------------------------------
# TAB 3: CROSS-DB PIPELINE (Easy Path vs Hard Path)
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Cross-DB Deduplication & Semantic Search Pipeline")
    st.markdown("Choose your workflow path to analyze semantic similarity within or between databases:")
    
    path_choice = st.radio(
        "Workflow Mode:",
        [
            "⚡ a. Easy Path: Automated Semantic Similarity Bins (Flag duplicates by score intervals)",
            "🎛️ b. Hard Path: Interactive Multi-Gate Batch Overview (Filter with Similar Date/Location/Text buttons)"
        ],
        horizontal=False,
        key="cross_path_choice"
    )
    
    cmp_mode = st.radio("Comparison Scope:", ["Within single dataset (DB1 vs DB1)", "Between two datasets (DB1 vs DB2)"], horizontal=True, key="cross_cmp_mode")
    
    df_b = df_active
    embed_cols_b = embed_cols
    if cmp_mode == "Between two datasets (DB1 vs DB2)":
        up_b = st.file_uploader("Upload Target Dataset B (CSV or Excel)", type=["csv", "xlsx"], key="cross_up_b")
        if up_b is not None:
            try:
                df_b = pd.read_csv(up_b) if up_b.type == "text/csv" else pd.read_excel(up_b)
                st.success(f"Dataset B Loaded: `{len(df_b):,}` rows × `{len(df_b.columns)}` columns")
            except Exception as e:
                st.error(f"Error loading Dataset B: {e}")
        else:
            st.info("Using DB1 as fallback for DB2 until uploaded.")
        embed_cols_b = st.multiselect(
            "Embedding Columns for DB2",
            options=list(df_b.columns),
            default=[c for c in embed_cols if c in df_b.columns] or [df_b.columns[0]],
            key="cross_cols_b"
        )

    cc1, cc2, cc3 = st.columns(3)
    with cc1: cross_thresh = st.slider("Cosine Similarity Threshold", min_value=0.60, max_value=0.95, value=0.80, step=0.01, key="cross_thresh")
    with cc2: cross_days = st.slider("Max Temporal Diff (Days)", min_value=0, max_value=30, value=3, step=1, key="cross_days")
    with cc3: cross_km = st.slider("Max Spatial Radius (Haversine km)", min_value=5.0, max_value=200.0, value=100.0, step=5.0, key="cross_km")

    st.markdown("---")

    if st.button("🚀 Execute Cross-DB Similarity Matrix", type="primary", key="btn_run_cross_pipeline"):
        with st.spinner(f"Computing cross-database {embed_model_name.rsplit('/', 1)[-1]} "
                       "matrix embeddings and gate criteria (no row cap — every "
                       "row in both datasets is encoded)..."):
            records_a = df_active.to_dict("records")
            records_b = df_b.to_dict("records") if cmp_mode == "Between two datasets (DB1 vs DB2)" else None

            res = dedup_service.run_cross_db_pipeline(
                records_a=records_a,
                records_b=records_b,
                cols_a=embed_cols,
                cols_b=embed_cols_b,
                threshold=cross_thresh,
                max_days=cross_days,
                max_km=cross_km,
                top_k=5,
                model_name=embed_model_name,
                date_col=date_col,
                lat_col=lat_col,
                lon_col=lon_col,
                location_col=location_col,
                state_col=state_col,
                city_col=city_col,
                use_gazetteer=use_gazetteer,
                use_cluster_blocking=use_cluster_blocking,
                block_min_cluster_size=block_min_cluster_size,
            )
            st.session_state['cross_db_pipeline_results'] = res

    if 'cross_db_pipeline_results' in st.session_state:
        c_res = st.session_state['cross_db_pipeline_results']
        pairs = c_res.get("pairs", [])
        summary = c_res.get("summary", {})
        bins = summary.get("bins", {})
        gates = summary.get("gate_counts", {})
        
        st.success(f"Evaluated `{summary.get('total_pairs_evaluated', 0)}` top candidate pairs across databases!")

        _bstats = summary.get("block_stats")
        if _bstats:
            st.caption(
                f"⚡ Cluster-blocked: {_bstats['block_count']} groups cut the comparison matrix "
                f"from {_bstats['full_matrix_cells']:,} to {_bstats['compared_cells']:,} cells"
                + (f" ({_bstats['noise_block_size']} rows ungrouped)." if _bstats['noise_block_size'] > 0 else ".")
            )

        if path_choice.startswith("⚡ a. Easy Path"):
            st.markdown("### 📊 Semantic Similarity Bins Overview")
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("🔴 Exact Duplicates (≥ 0.88)", bins.get("exact_duplicate", 0))
            bc2.metric("🟠 Strong Similar (0.80 - 0.88)", bins.get("strong_similar", 0))
            bc3.metric("🟡 Moderate Similar (0.70 - 0.80)", bins.get("moderate_similar", 0))
            bc4.metric("🟢 Distinct (< 0.70)", bins.get("distinct", 0))
            
            bin_filter = st.selectbox("Filter Table by Semantic Bin:", ["All Bins", "exact_duplicate", "strong_similar", "moderate_similar", "distinct"], key="easy_bin_filter")
            filtered_pairs = pairs if bin_filter == "All Bins" else [p for p in pairs if p["bin"] == bin_filter]
            
            if filtered_pairs:
                df_show = pd.DataFrame(filtered_pairs)
                df_show["similarity"] = df_show["similarity"].apply(lambda x: f"{x*100:.1f}%")
                df_show = df_show[["id_a", "id_b", "similarity", "bin", "haversine_km", "date_diff_days", "text_a_preview", "text_b_preview"]]
                df_show.columns = ["ID A", "ID B", "Score", "Bin Category", "Distance (km)", "Date Diff (days)", "Report A Preview", "Report B Preview"]
                st.dataframe(df_show, use_container_width=True)
            else:
                st.info("No pairs in the selected semantic bin.")
                
        else:
            st.markdown("### 🎛️ Interactive Multi-Gate Batch Overview")
            st.caption("Click any filter button below to dynamically slice the batch similarity matrix:")
            
            # Interactive Filter Buttons
            gate_sel = st.radio(
                "Quick-Action Multi-Gate Filter:",
                [
                    f"📝 Similar Text Only (`sim ≥ 0.80`) — [{gates.get('similar_text', 0)} pairs]",
                    f"📅 Similar Date Only (`± 3 days`) — [{gates.get('similar_date', 0)} pairs]",
                    f"📍 Similar Location Only (`≤ 50 km`) — [{gates.get('similar_location', 0)} pairs]",
                    f"🧩 Text + Date (`sim ≥ 0.80` and `± 3 days`) — [{gates.get('similar_text_date', 0)} pairs]",
                    f"⚡ Similar Both (Spatial + Temporal) — [{gates.get('similar_both', 0)} pairs]",
                    f"🎯 Similar All / Flagged Duplicates (`Full Convergence`) — [{gates.get('similar_all', 0)} pairs]",
                    "🌐 Show All Evaluated Pairs"
                ],
                horizontal=False,
                key="hard_gate_radio"
            )

            filtered_pairs = pairs
            if gate_sel.startswith("📝"):
                filtered_pairs = [p for p in pairs if p["flags"]["is_similar_text"]]
            elif gate_sel.startswith("📅"):
                filtered_pairs = [p for p in pairs if p["flags"]["is_similar_date"]]
            elif gate_sel.startswith("📍"):
                filtered_pairs = [p for p in pairs if p["flags"]["is_similar_location"]]
            elif gate_sel.startswith("🧩"):
                filtered_pairs = [p for p in pairs if p["flags"]["is_similar_text_date"]]
            elif gate_sel.startswith("⚡"):
                filtered_pairs = [p for p in pairs if p["flags"]["is_similar_both"]]
            elif gate_sel.startswith("🎯"):
                filtered_pairs = [p for p in pairs if p["flags"]["is_similar_all"]]
                
            if filtered_pairs:
                df_show = pd.DataFrame(filtered_pairs)
                df_show["similarity"] = df_show["similarity"].apply(lambda x: f"{x*100:.1f}%")
                df_show = df_show[["id_a", "id_b", "similarity", "haversine_km", "date_diff_days", "text_a_preview", "text_b_preview"]]
                df_show.columns = ["ID A", "ID B", "Similarity", "Distance (km)", "Date Diff (days)", "Report A Preview", "Report B Preview"]
                st.dataframe(df_show, use_container_width=True)
            else:
                st.warning("No candidate pairs satisfied the selected multi-gate criteria.")
