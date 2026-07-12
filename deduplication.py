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

embed_cols = st.multiselect(
    "Embedding Columns for DB1 (Concatenated to build narrative vector)",
    options=list(df_active.columns),
    default=default_cols,
    help="Select one or more text columns that describe the sighting narrative."
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
            with st.spinner("Computing Harrier vector cosine similarity..."):
                res = dedup_service.check_similarity(narr_a, narr_b)
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
                res = dedup_service.check_duplicate(rec_a, rec_b)
                
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
    st.markdown("Run automated deduplication across candidate datasets with customizable semantic thresholds, geographic radii, and temporal gating.")
    
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1: thresh = st.slider("Cosine Similarity Threshold", min_value=0.60, max_value=0.95, value=0.80, step=0.01, key="advanced_thresh")
    with sc2: days = st.slider("Max Temporal Diff (Days)", min_value=0, max_value=30, value=3, step=1, key="advanced_days")
    with sc3: km = st.slider("Max Spatial Radius (Haversine km)", min_value=5.0, max_value=200.0, value=50.0, step=5.0, key="advanced_km")
    with sc4:
        st.markdown("<br>", unsafe_allow_html=True)
        use_llm = st.toggle("Enable LLM Tier-3 Judge", value=False, help="Use Gemini confirmation on borderline candidate pairs", key="advanced_use_llm")

    st.markdown("---")
    if st.button("🚀 Run Advanced Deduplication & Clustering", type="primary", key="btn_run_advanced"):
        with st.spinner("Screening candidates via GPU matrix multiplication and Union-Find graph clustering..."):
            adv_res = dedup_service.run_advanced_dedup(
                threshold=thresh, date_diff_days=days, max_km=km, use_llm_judge=use_llm
            )
            summary = adv_res.get("summary", {})
            flagged = adv_res.get("flagged_pairs", [])
            
            kc1, kc2, kc3, kc4 = st.columns(4)
            kc1.metric("Total Clusters Formed", summary.get("total_clusters", 0))
            kc2.metric("Rows in Clusters", summary.get("rows_in_clusters", 0))
            kc3.metric("Redundant Rows Saved", summary.get("redundant_rows_saved", 0))
            kc4.metric("Flagged Pairs Displayed", summary.get("flagged_pairs_count", len(flagged)))
            
            st.markdown("#### High-Confidence Flagged Sighting Pairs")
            if flagged:
                df_flagged = pd.DataFrame(flagged)
                df_flagged["similarity"] = df_flagged["similarity"].apply(lambda x: f"{x*100:.1f}%")
                df_flagged.columns = ["Record A ID", "Record B ID", "Similarity", "LLM Same Event", "Justification"]
                st.dataframe(df_flagged, use_container_width=True)
            else:
                st.info("No flagged pairs matched the criteria.")

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

    st.markdown("---")
    
    if st.button("🚀 Execute Cross-DB Similarity Matrix", type="primary", key="btn_run_cross_pipeline"):
        with st.spinner("Computing cross-database Harrier matrix embeddings and gate criteria..."):
            records_a = df_active.to_dict("records")
            records_b = df_b.to_dict("records") if cmp_mode == "Between two datasets (DB1 vs DB2)" else None
            
            res = dedup_service.run_cross_db_pipeline(
                records_a=records_a,
                records_b=records_b,
                cols_a=embed_cols,
                cols_b=embed_cols_b,
                threshold=0.80,
                max_days=3,
                max_km=50.0,
                top_k=5,
                max_pairs=300
            )
            st.session_state['cross_db_pipeline_results'] = res

    if 'cross_db_pipeline_results' in st.session_state:
        c_res = st.session_state['cross_db_pipeline_results']
        pairs = c_res.get("pairs", [])
        summary = c_res.get("summary", {})
        bins = summary.get("bins", {})
        gates = summary.get("gate_counts", {})
        
        st.success(f"Evaluated `{summary.get('total_pairs_evaluated', 0)}` top candidate pairs across databases!")
        
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
