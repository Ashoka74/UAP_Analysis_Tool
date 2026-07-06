import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from uap_analyzer import (UAPParser, UAPAnalyzer, UAPVisualizer,
                          OPENAI_MODELS, DEEPSEEK_MODELS, estimate_cost, _extract_json)
# import ChartGen
# from ChartGen import ChartGPT
from Levenshtein import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import streamlit.components.v1 as components
from dateutil import parser
from sentence_transformers import SentenceTransformer
import torch
import squarify
import matplotlib.colors as mcolors
import textwrap
import datamapplot
import openai
from openai import OpenAI
import os
import json
from utils.data_processing import DataProcessor
import plotly.graph_objects as go


def read_uploaded_sheet(uploaded, *, key, **read_kwargs):
    """Read an uploaded tabular file into a DataFrame.

    For multi-sheet .xlsx/.xls/.xlsm workbooks, render a selectbox so the user
    picks which sheet to load. Single-sheet workbooks and CSVs load directly
    (a CSV has no sheets). Returns a DataFrame, or None if nothing is uploaded.
    Exceptions propagate to the caller's existing try/except.
    """
    if uploaded is None:
        return None
    name = (getattr(uploaded, "name", "") or "").lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        xls = pd.ExcelFile(uploaded)
        sheets = list(xls.sheet_names)
        sheet = sheets[0]
        if len(sheets) > 1:
            sheet = st.selectbox(
                f"📑 {len(sheets)} sheets in “{getattr(uploaded, 'name', 'workbook')}” "
                "— select the sheet to load",
                sheets, key=key,
            )
        return pd.read_excel(xls, sheet_name=sheet, **read_kwargs)
    return pd.read_csv(uploaded, **read_kwargs)

# st.set_option('deprecation.showPyplotGlobalUse', False)

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)



def load_data(file_path, key='df'):
    return pd.read_hdf(file_path, key=key)


def gemini_query(question, selected_data, gemini_key):

    if question == "":
        question = "Summarize the following data in relevant bullet points"

    import pathlib
    import textwrap

    import google.generativeai as genai

    from IPython.display import display
    from IPython.display import Markdown


    def to_markdown(text):
        text = text.replace('â€¢', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
    
    # selected_data is a list
    # remove empty

    filtered = [str(x) for x in selected_data if str(x) != '' and x is not None]
    # make a string
    context = '\n'.join(filtered)

    genai.configure(api_key=gemini_key)
    query_model = genai.GenerativeModel('models/gemini-3.1-pro-preview')
    response = query_model.generate_content([f"{question}\n Answer based on this context: {context}\n\n"])
    return(response.text)

def plot_treemap(df, column, top_n=32):
        # Get the value counts and the top N labels
        value_counts = df[column].value_counts()
        top_labels = value_counts.iloc[:top_n].index
        
        # Use np.where to replace all values not in the top N with 'Other'
        revised_column = f'{column}_revised'
        df[revised_column] = np.where(df[column].isin(top_labels), df[column], 'Other')

        # Get the value counts including the 'Other' category
        sizes = df[revised_column].value_counts().values
        labels = df[revised_column].value_counts().index

        # Get a gradient of colors
        # colors = list(mcolors.TABLEAU_COLORS.values())

        n_colors = len(sizes)
        colors = plt.cm.Oranges(np.linspace(0.3, 0.9, n_colors))[::-1]


        # Get % of each category
        percents = sizes / sizes.sum()

        # Prepare labels with percentages
        labels = [f'{label}\n {percent:.1%}' for label, percent in zip(labels, percents)]

        fig, ax = plt.subplots(figsize=(20, 12))

        # Plot the treemap
        squarify.plot(sizes=sizes, label=labels, alpha=0.7, pad=True, color=colors, text_kwargs={'fontsize': 10})

        ax = plt.gca()
        for text, rect in zip(ax.texts, ax.patches):
            background_color = rect.get_facecolor()
            r, g, b, _ = mcolors.to_rgba(background_color)
            brightness = np.average([r, g, b])
            text.set_color('white' if brightness < 0.5 else 'black')

        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        plt.tight_layout()
        return fig

# st.set_option('deprecation.showPyplotGlobalUse', False)

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


class CachedUAPParser(UAPParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'parsed_responses' not in st.session_state:
            st.session_state['parsed_responses'] = {}

    def parse_responses(self):
        parsed_responses = {}
        not_parsed = 0
        try:
            for k, v in self.responses.items():
                if k in {"__batch_ids__", "__batch_descs__"}:
                    continue
                result = _extract_json(v)
                if result is not None:
                    parsed_responses[k] = result
                else:
                    not_parsed += 1

            st.session_state['parsed_responses'] = parsed_responses
        except Exception as e:
            st.error(f"Error parsing responses: {e}")

        st.write(f"Parsed: {len(parsed_responses)}  |  Failed: {not_parsed}")
        return st.session_state['parsed_responses']

    def responses_to_df(self, col, parsed_responses):
        try:
            parsed_df = pd.DataFrame(parsed_responses).T
            # If col exists as a column key (e.g. 'sightingDetails' for Default UAP format)
            if col is not None and col in parsed_df.columns:
                parsed_df2 = pd.json_normalize(parsed_df[col].tolist())
                parsed_df2.index = parsed_df.index
            else:
                # SCU / multi-section format: each value is a dict of sections —
                # json_normalize the raw response dicts directly so nested keys
                # become flat columns like source.name, date_time.year, etc.
                parsed_df2 = pd.json_normalize(list(parsed_responses.values()))
                parsed_df2.index = parsed_df.index

            for column in parsed_df2.columns:
                if parsed_df2[column].dtype == 'object':
                    parsed_df2[column] = parsed_df2[column].astype(str)

            return parsed_df2
        except Exception as e:
            st.error(f"Error converting responses to DataFrame: {e}")
            return pd.DataFrame()


def load_data(file_path, key='df'):
    return pd.read_hdf(file_path, key=key)


def gemini_query(question, selected_data, gemini_key):

    if question == "":
        question = "Summarize the following data in relevant bullet points"

    import pathlib
    import textwrap

    import google.generativeai as genai

    from IPython.display import display
    from IPython.display import Markdown


    def to_markdown(text):
        text = text.replace('â€¢', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
    
    # selected_data is a list
    # remove empty

    filtered = [str(x) for x in selected_data if str(x) != '' and x is not None]
    # make a string
    context = '\n'.join(filtered)

    genai.configure(api_key=gemini_key)
    query_model = genai.GenerativeModel('models/gemini-3.1-pro-preview')
    response = query_model.generate_content([f"{question}\n Answer based on this context: {context}\n\n"])
    return(response.text)


def plot_hist(df, column, bins=10, kde=True, figsize=(12, 6), color='orange', title=None):
    """
    Create a histogram with improved styling and error handling.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        column (str): Column name to plot
        bins (int): Number of bins for histogram
        kde (bool): Whether to show kernel density estimation
        figsize (tuple): Figure size
        color (str): Color for the plot
        title (str): Custom title for the plot
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    try:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        
        # Check if column exists and has data
        if column not in df.columns:
            ax.text(0.5, 0.5, f'Column "{column}" not found', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
            
        data_series = df[column].dropna()
        if len(data_series) == 0:
            ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        # Create histogram with improved bins calculation
        if bins == 'auto':
            bins = min(50, max(10, int(np.sqrt(len(data_series)))))
        
        sns.histplot(data=df, x=column, kde=kde, bins=bins, color=color, ax=ax, alpha=0.7)
        
        # Set title
        ax.set_title(title or f'Distribution of {column}', color=color, fontweight='bold', fontsize=14)
        ax.set_xlabel(column, color=color, fontsize=12)
        ax.set_ylabel('Count', color=color, fontsize=12)
        
        # Style the plot
        for spine in ax.spines.values():
            spine.set_color(color)
        
        ax.tick_params(axis='both', colors=color)
        ax.grid(True, alpha=0.3, color=color)
        
        # Set transparent background
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Error creating histogram:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
        ax.set_title('Histogram Error', color='red')
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        return fig



def is_api_key_valid(api_key, model='gpt-4o-mini'):
    try:
        os.environ['OPENAI_API_KEY'] = api_key
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": 'Say Hello World!'}])
        text = response.choices[0].message.content
        if len(text) >= 0:
            return True
    except Exception as e:
       st.error(f'Error with the API key :{e}')
       return False

def download_json(data):
    json_str = json.dumps(data, indent=2)
    return json_str


# ── Custom-schema-field editor helpers ─────────────────────────────────────────
# The "Custom / Additional Fields" editor keeps two views (a key→value field
# builder and a raw-JSON box) in sync over one source of truth. Fields are
# addressed by dotted path (e.g. "location.city") so nested schema levels can be
# reached without recursive widgets — matching how scu_normalizer already reads
# dotted columns. These helpers convert between the nested dict and flat rows.

def _flatten_dotted(d, prefix=""):
    """Flatten a nested dict to {dotted_key: leaf_value}; non-dict values
    (strings, lists, numbers) are treated as leaves."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            flat.update(_flatten_dotted(v, key))
        else:
            flat[key] = v
    return flat


def _unflatten_dotted(flat):
    """Inverse of _flatten_dotted: expand dotted keys back into a nested dict."""
    out = {}
    for key, v in flat.items():
        parts = [p for p in str(key).split(".") if p != ""]
        if not parts:
            continue
        node = out
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = v
    return out


def _value_to_text(v):
    """Render a leaf value for the editor cell (strings verbatim, else JSON)."""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _coerce_custom_value(s):
    """Parse an editor cell back to a typed value. Bare strings stay strings;
    JSON literals (lists, objects, numbers, booleans, null) are decoded so they
    round-trip cleanly and don't trigger a sync loop."""
    if not isinstance(s, str):
        return s
    t = s.strip()
    if t == "":
        return ""
    if t in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[t]
    if t[:1] in ("[", "{", "-") or t[:1].isdigit():
        try:
            return json.loads(t)
        except (ValueError, TypeError):
            return s
    return s


def _rows_to_dict(edited_df):
    """Build a nested dict from the field-builder rows (skipping blank keys)."""
    flat = {}
    for _, row in edited_df.iterrows():
        key = str(row.get("Field (dotted path)", "")).strip()
        if not key:
            continue
        flat[key] = _coerce_custom_value(str(row.get("Description / value", "")))
    return _unflatten_dotted(flat)


def attach_kept_columns(df):
    """Append the user-selected source columns onto a parsed-output DataFrame.

    The parsed DataFrame is indexed by the LLM input text (the concatenated
    raw-text columns); `st.session_state['source_keep_df']` is indexed the same
    way, so a reindex aligns each kept source column to its parsed row. A kept
    column whose name collides with a parsed column is prefixed `src_`. Returns
    the DataFrame unchanged when no carry-through columns were selected.
    """
    keep_df = st.session_state.get('source_keep_df')
    if keep_df is None or df is None or df.empty or keep_df.empty:
        return df
    aligned = keep_df.reindex(df.index)
    out = df.copy()
    for c in aligned.columns:
        name = c if c not in out.columns else f"src_{c}"
        out[name] = aligned[c].values
    return out


def _parsed_responses_dict():
    """The parsing dict ({description: parsed_json}) from session state, or None.

    Other pages (analyzing.py, map.py, …) store a *DataFrame* under the shared
    'parsed_responses' session key. Returning None for any non-dict value keeps
    parsing.py's dict-only paths from raising "The truth value of a DataFrame is
    ambiguous" when the user navigates back here after using another page.
    """
    pr = st.session_state.get('parsed_responses')
    return pr if isinstance(pr, dict) and pr else None


def convert_cached_data_to_df(parsed_responses):
    if not isinstance(parsed_responses, dict) or not parsed_responses:
        st.warning("No cached data available. Please parse the dataset first.")
        return
    try:
        parsed_df_raw = pd.DataFrame(parsed_responses).T
        # Default UAP format has ONLY a 'sightingDetails' key; multi-section
        # schemas (SCU Spreadsheet, SCU_v2, FORMAT_MERGED) carry it alongside
        # other top-level blocks and must be normalised from the top level.
        if set(parsed_df_raw.columns) == {'sightingDetails'}:
            responses_df = pd.json_normalize(parsed_df_raw['sightingDetails'].tolist())
            responses_df.index = parsed_df_raw.index
        else:
            responses_df = pd.json_normalize(list(parsed_responses.values()))
            responses_df.index = parsed_df_raw.index

        for column in responses_df.columns:
            if responses_df[column].dtype == 'object':
                responses_df[column] = responses_df[column].astype(str)

        if not responses_df.empty:
            responses_df = attach_kept_columns(responses_df)
            st.dataframe(responses_df)
            st.session_state['parsed_responses_df'] = responses_df.copy()
            st.success("Successfully converted cached data to DataFrame.")
        else:
            st.error("Failed to create DataFrame from cached responses.")
    except Exception as e:
        st.error(f"Error converting responses to DataFrame: {e}")


def _extract_year_series(series):
    """Best-effort 4-digit year (float; NaN when unknown) from a column that may
    hold datetimes, numeric years, or free-text dates. Lets a separate /
    carried-through date column drive the SCU window when the parsed
    ``date_time.year`` is missing (e.g. parsing ran on the description only)."""
    import re
    s = series
    # 1. Native datetime.
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.year.astype("float")
    # 2. Numeric values that already look like calendar years (1976, 1980.0).
    num = pd.to_numeric(s, errors="coerce")
    yr_like = num.where((num >= 1000) & (num <= 2100))
    if yr_like.notna().mean() > 0.3:
        return yr_like.astype("float")
    # 3. Parseable date strings / date objects.
    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().mean() > 0.3:
        return dt.dt.year.astype("float")
    # 4. Regex 4-digit-year fallback (handles "c. 1980", "May 1976", …).
    rx = re.compile(r"(1[89]\d{2}|20\d{2})")
    def _yr(v):
        m = rx.search(str(v))
        return float(m.group(1)) if m else np.nan
    return s.map(_yr)


def _date_parts(series):
    """Return (year, month, day) float Series from `series`, mirroring map.py's
    robust date handling (auto_create_date_column / detect_and_combine_date_columns).

    A bare calendar-year column yields NaN month/day, so a year-only date is
    correctly treated as an *incomplete* core date by SCU Criterion 2 (which
    requires a full year+month+day)."""
    s = series
    if pd.api.types.is_datetime64_any_dtype(s):
        return (s.dt.year.astype("float"), s.dt.month.astype("float"),
                s.dt.day.astype("float"))
    # Bare numeric calendar years (1976, 1980.0): year known, month/day unknown.
    num = pd.to_numeric(s, errors="coerce")
    yr_like = num.where((num >= 1000) & (num <= 2100))
    if yr_like.notna().mean() > 0.3:
        nan = pd.Series(np.nan, index=s.index)
        return yr_like.astype("float"), nan, nan
    # Full date strings / objects → parse and split into components.
    dt = pd.to_datetime(s, errors="coerce")
    return (dt.dt.year.astype("float"), dt.dt.month.astype("float"),
            dt.dt.day.astype("float"))


# Parsed-schema field prefixes — a carried-through *source* date column has a
# plain name, so it sorts ahead of these and lands first in the picker.
_SCU_SCHEMA_PREFIXES = ("date_time.", "source.", "sightingDetails.",
                        "investigation.", "manifest.", "location.", "craft.")

def _candidate_date_columns(df):
    """Column names in `df` that look like dates (by dtype, name, or a parse
    test), ordered so non-schema (carried-through source) columns come first."""
    import re
    name_rx = re.compile(r"date|year|\btime\b|\byr\b|\bdt\b", re.I)
    cands = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s) or name_rx.search(str(c)):
            cands.append(c)
            continue
        sample = s.dropna().head(50)
        if len(sample) and _extract_year_series(sample).notna().mean() > 0.6:
            cands.append(c)
    cands.sort(key=lambda c: (str(c).startswith(_SCU_SCHEMA_PREFIXES), str(c)))
    return cands


def _render_scu_column_mapping_editor(audit, raw_df):
    """Review / override how input columns were auto-mapped (exact → suffix →
    leaf-name → fuzzy) onto the canonical fields the SCU gate reads.

    Schemas that nest fields differently (e.g. MasterSCU_v1's ``object.*`` /
    ``engagement.engagement_type.*`` / ``behavior.performance.*``) resolve
    automatically; wrong or missing matches can be corrected per-field here.
    Overrides live in ``st.session_state['scu_column_map']`` ({canonical:
    actual}); submitting the form reruns the normalizer with them applied.
    """
    mapping = audit.get("column_mapping") or {}
    methods = audit.get("column_mapping_methods") or {}
    unmatched = audit.get("column_mapping_unmatched") or []
    expected = audit.get("expected_input_columns") or []
    if not expected:
        return   # mapping disabled / nothing to show

    manual = st.session_state.get("scu_column_map") or {}
    n_heur = sum(1 for m in methods.values() if m not in ("exact", "manual"))
    label = (
        f"🔀 Input column mapping — {len(unmatched)} unmatched · "
        f"{n_heur} heuristic match(es)" + (f" · {len(manual)} manual" if manual else "")
    )
    with st.expander(label, expanded=False):
        st.caption(
            "Fields the SCU gate reads, and which dataset column feeds each one. "
            "`suffix` / `leaf` / `fuzzy` matches were resolved automatically — "
            "review them and override any wrong match. Exact matches are hidden "
            "unless toggled."
        )
        show_exact = st.toggle("Show exact matches", value=False, key="scu_map_show_exact")
        rows = [
            c for c in expected
            if show_exact or methods.get(c) != "exact" or c in manual
        ]
        # unmatched first, then heuristic, then the rest
        rows.sort(key=lambda c: (0 if c in unmatched else 1 if methods.get(c) not in ("exact", "manual") else 2, c))
        if not rows:
            st.info("Every gate input matched its column exactly — nothing to review.")
            return

        options = ["(auto)"] + list(raw_df.columns)
        with st.form("scu_column_map_form", border=False):
            new_map: dict[str, str] = {}
            for canon in rows:
                auto_actual = mapping.get(canon)
                method = "manual" if canon in manual else methods.get(canon, "unmatched")
                cur = manual.get(canon)
                idx = options.index(cur) if cur in options else 0
                c1, c2 = st.columns([2, 3])
                c1.markdown(
                    f"<div style='padding-top:0.4rem'><code>{canon}</code><br/>"
                    f"<small>{'⚠️ unmatched' if canon in unmatched else f'{method} ← <code>{auto_actual}</code>'}</small></div>",
                    unsafe_allow_html=True,
                )
                pick = c2.selectbox(
                    canon, options, index=idx, key=f"scu_map_{canon}",
                    label_visibility="collapsed",
                )
                if pick != "(auto)":
                    new_map[canon] = pick
            if st.form_submit_button("Apply mapping & re-normalize"):
                st.session_state["scu_column_map"] = new_map
                st.rerun()
        if manual:
            if st.button("Reset manual mapping", key="scu_map_reset"):
                st.session_state["scu_column_map"] = {}
                st.rerun()


def render_scu_normalization(parsed_responses, show_advanced_filters=False):
    """Run the SCU v2 normalizer on parsed responses; show audit + downloads.

    Builds a DataFrame straight from the raw parsed-response dicts (so real
    NaNs and list-valued fields are preserved), runs scu_normalizer.normalize(),
    then surfaces the SCU five-criterion eligibility funnel and download buttons.
    Toggled on/off via the global "Apply SCU normalization" switch in app.py.

    When ``show_advanced_filters`` is True (set for restored sessions, which
    never pass through the raw-upload filter_dataframe call), the categorical
    distribution treemaps and the numeric/date/text filters are rendered on the
    normalized data, directly below the audit report.
    """
    if not isinstance(parsed_responses, dict) or not parsed_responses:
        st.info("No parsed data to normalize yet.")
        return
    try:
        import scu_normalizer
    except Exception as e:
        st.error(f"Could not import scu_normalizer: {e}")
        return
    try:
        raw_df = pd.json_normalize(list(parsed_responses.values()))
        # Re-attach carried-through source columns (e.g. a separate date column
        # kept when parsing ran on the description), aligning on the parsed-dict
        # keys (the LLM input text) so they can drive the SCU window filter below.
        _keys = list(parsed_responses.keys())
        if len(raw_df) == len(_keys):
            raw_df.index = pd.Index(_keys)
            raw_df = attach_kept_columns(raw_df).reset_index(drop=True)
    except Exception as e:
        st.error(f"Could not build a DataFrame from parsed responses: {e}")
        return
    if raw_df.empty:
        st.warning("Parsed responses produced an empty DataFrame.")
        return
    # Manual column-map overrides ({canonical: actual}) from the mapping editor
    # below — applied on top of the automatic exact/suffix/leaf/fuzzy matching.
    _manual_map = st.session_state.get("scu_column_map") or {}
    _manual_map = {k: v for k, v in _manual_map.items() if v in raw_df.columns}
    try:
        norm_df, audit = scu_normalizer.normalize(raw_df, column_map=_manual_map or None)
    except Exception as e:
        st.error(f"Normalization failed: {e}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{audit.get('output_rows', len(norm_df)):,}")
    c2.metric("SCU-eligible", f"{audit.get('scu_eligible_count', 0):,}")
    c3.metric("In 1945-1975 window", f"{audit.get('in_scu_window_count', 0):,}")
    c4.metric("Credible witness", f"{audit.get('has_credible_witness_count', 0):,}")

    _render_scu_column_mapping_editor(audit, raw_df)

    audit_md = scu_normalizer.audit_to_markdown(audit)
    with st.expander("Normalization audit report", expanded=False):
        st.markdown(audit_md)

    # ── Window date column ───────────────────────────────────────────────────
    # The post-1975 window is normally derived from the parsed `date_time.year`.
    # When parsing ran on the description and the real date lives in a separate
    # (carried-through) column, that field is null and the window drops every
    # row. Let the user pick the actual date column and a start year (default
    # 1976) — this recomputes `post_1975_window` used by the presets/funnel below.
    with st.expander("📅 Window date column & start year", expanded=False):
        st.caption(
            "By default the SCU post-1975 window uses the parsed `date_time.year`. "
            "If the real date is in a separate column (e.g. parsing ran on the "
            "description and the date was carried through), pick that column here. "
            "Rows with no parseable year are excluded from the window."
        )
        _DEFAULT_SRC = "Parsed date_time.year"
        _cands = [c for c in _candidate_date_columns(norm_df) if c != "date_time.year"]
        _src = st.selectbox(
            "Date column for the window", [_DEFAULT_SRC] + _cands,
            key="scu_window_date_col",
            help="Carried-through source columns are listed first; parsed schema "
                 "date fields follow.",
        )
        _win_start = int(st.number_input(
            "Window start year (inclusive)", min_value=1800, max_value=2100,
            value=1976, step=1, key="scu_window_start_year",
        ))
        if _src == _DEFAULT_SRC:
            _src_col = (norm_df["date_time.year"] if "date_time.year" in norm_df.columns
                        else pd.Series([np.nan] * len(norm_df), index=norm_df.index))
            _yr = pd.to_numeric(_src_col, errors="coerce")
            _src_label = "date_time.year"
        else:
            _yr = _extract_year_series(norm_df[_src])
            _src_label = _src
        norm_df["post_1975_window"] = (_yr >= _win_start).fillna(False)
        st.session_state["scu_window_label"] = f"Window: {_src_label} ≥ {_win_start}"
        _n_pass = int(norm_df["post_1975_window"].sum())
        _n_bad = int(_yr.isna().sum())
        st.success(
            f"Window = **{_src_label} ≥ {_win_start}** → {_n_pass:,} of "
            f"{len(norm_df):,} rows in window"
            + (f"; {_n_bad:,} row(s) have no parseable year (excluded)." if _n_bad else ".")
        )

        # Criterion 2 (has_core_fields) normally needs date_time.year/month/day +
        # country. When the date comes from a separate column those are null, so
        # derive the date part from the selected column too (map.py-style parse).
        if _src != _DEFAULT_SRC:
            _use_for_core = st.checkbox(
                "Also derive Criterion 2 (core fields) date from this column",
                value=True, key="scu_window_core_fields",
                help="has_core_fields requires a full date (year + month + day) "
                     "plus country. This swaps the date part to the selected "
                     "column; a year-only column leaves month/day missing.",
            )
            if _use_for_core:
                _y, _m, _d = _date_parts(norm_df[_src])
                _country = (norm_df["location_country_iso"].notna()
                            if "location_country_iso" in norm_df.columns
                            else pd.Series(False, index=norm_df.index))
                norm_df["has_core_fields"] = (
                    _y.notna() & _m.notna() & _d.notna() & _country
                )
                _n_core = int(norm_df["has_core_fields"].sum())
                _yr_no_md = int((_y.notna() & ~(_m.notna() & _d.notna())).sum())
                st.success(
                    f"Core fields (Criterion 2) now use **{_src}** for the date → "
                    f"{_n_core:,} of {len(norm_df):,} rows have full date + country."
                    + (f" {_yr_no_md:,} row(s) have a year but no month/day "
                       "(incomplete)." if _yr_no_md else "")
                )

    # Restored data never passes through the raw-upload Advanced Filters block
    # (filter_dataframe() at the top of the parse flow), so the categorical
    # distribution treemaps — and the numeric/date/text filters alongside them —
    # are otherwise unavailable for a restored session. Surface them here, on the
    # normalized data, directly below the audit report.
    # The Advanced Filters return value flows into the table, download, and the
    # SCU Eligibility Filter below, so narrowing the data here updates everything
    # downstream. Without a filter applied this is just norm_df unchanged.
    norm_df_view = norm_df
    if show_advanced_filters:
        st.markdown("#### 🔬 Advanced Filters")
        st.caption(
            "Explore the normalized data — select categorical columns to see "
            "their distribution treemaps, plus numeric / date / text filters. "
            "Selections here narrow the table, download, and SCU Eligibility "
            "Filter below."
        )
        norm_df_view = filter_dataframe(norm_df)

    _disp = norm_df_view.copy()
    for _c in _disp.columns:
        if _disp[_c].dtype == 'object':
            _disp[_c] = _disp[_c].astype(str)
    st.dataframe(_disp)
    # Canonical normalizer output stays the full frame; the filtered view drives
    # only the display, download, and SCU filter.
    st.session_state['scu_normalized_df'] = norm_df

    col_a, col_b = st.columns(2)
    col_a.download_button(
        "Download normalized CSV",
        data=convert_df(_disp),
        file_name="parsed_reports_normalized.csv",
        mime="text/csv",
        key="dl_scu_norm_csv",
    )
    col_b.download_button(
        "Download audit report (.md)",
        data=audit_md,
        file_name="normalization_audit.md",
        mime="text/markdown",
        key="dl_scu_norm_audit",
    )

    st.divider()
    render_scu_filter(norm_df_view)


def render_scu_filter(norm_df):
    """SCU eligibility filter section: presets + incremental-filter funnel.

    Operates on the normalized DataFrame from scu_normalizer.normalize(), which
    carries one boolean column per SCU criterion. Selecting a preset ANDs the
    chosen criteria together (in cascade order) and draws a funnel showing how
    each incremental filter shrinks the dataset, plus the filtered rows.
    """
    import scu_normalizer

    st.markdown("### 🔎 SCU Eligibility Filter")

    criteria = scu_normalizer.SCU_CRITERIA          # list of (key, column, label)
    extra_criteria = scu_normalizer.SCU_EXTRA_CRITERIA
    all_keys = [k for k, _c, _l in criteria]
    label_map = {k: lbl for k, _c, lbl in (criteria + extra_criteria)}
    # Reflect the user's chosen window date column / start year (set in the
    # normalization section above) in the criterion label.
    _win_lbl = st.session_state.get("scu_window_label")
    if _win_lbl:
        label_map["post_1975_window"] = _win_lbl

    # The post-1975 gate swaps in_scu_window for the post-1975 companion window.
    post_1975_keys = ["post_1975_window"] + [k for k in all_keys if k != "in_scu_window"]

    # Presets — "Full five-criterion gate" is the corrected scu_eligible.
    presets = {
        "Full five-criterion gate (1945-1975, recommended)": all_keys,
        "Post-1975 five-criterion gate (1975 onwards)": post_1975_keys,
        "Phase-3 analog — no window / anomaly / credibility gates": [
            "has_core_fields", "has_investigation_channel",
            "has_engagement_signal", "day_night_resolved", "military_public_known",
        ],
        "SCU window only (1945-1975)": ["in_scu_window"],
        "Post-1975 window only (1975 onwards)": ["post_1975_window"],
        "Data quality only — Criterion 2": ["has_core_fields"],
        "Custom": None,
    }

    preset = st.selectbox("Preset", list(presets.keys()), key="scu_filter_preset")

    if preset == "Custom":
        st.caption("Tick the criteria to AND together (cascade order):")
        selected = []
        _cols = st.columns(2)
        for _i, (key, _col, label) in enumerate(criteria):
            with _cols[_i % 2]:
                if st.checkbox(label, value=True, key=f"scu_crit_{key}"):
                    selected.append(key)
    else:
        selected = list(presets[preset])
        st.caption("Criteria applied (cascade order):")
        st.markdown(
            "\n".join(f"- ✅ {label_map.get(k, k)}" for k in selected)
        )

    col_x1, col_x2, col_x3 = st.columns(3)
    with col_x1:
        require_1yr = st.checkbox(
            "Also require report within 1 year of sighting",
            value=False,
            key="scu_filter_within_1yr",
            help="SCU Criterion 1 — uses investigation.reports_within_1_year_of_sighting.",
        )
    with col_x2:
        require_1mo = st.checkbox(
            "Also require report within 1 month of sighting",
            value=False,
            key="scu_filter_within_1mo",
            help="SCU_v2 — uses investigation.reports_within_1_month_of_sighting.",
        )
    with col_x3:
        exclude_contradict = st.checkbox(
            "Also exclude source-denied UAPs (assessment.contradictsUap = true)",
            value=False,
            key="scu_filter_exclude_contradict",
            help="SCU_v2 only — drops rows the source itself says were not a UAP.",
        )

    if require_1yr and "reports_within_1_year" in norm_df.columns:
        selected = list(selected) + ["reports_within_1_year"]
    if require_1mo and "reports_within_1_month" in norm_df.columns:
        selected = list(selected) + ["reports_within_1_month"]

    if not selected and not exclude_contradict:
        st.info("Select at least one criterion to apply a filter.")
        return

    stages, values, mask = scu_normalizer.incremental_funnel(norm_df, selected)
    if exclude_contradict and "contradicts_uap" in norm_df.columns:
        mask = mask & ~norm_df["contradicts_uap"].fillna(False).astype(bool)
        stages = stages + ["Exclude source-denied UAPs"]
        values = values + [int(mask.sum())]

    total = values[0] if values else 0

    # ── Incremental-filter funnel ───────────────────────────────────────────
    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textposition="inside",
        textinfo="value+percent initial",
        textfont={"color": "white"},
        marker={
            "color": "rgba(255,140,0,0.85)",
            "line": {"color": "orange", "width": 1.5},
        },
        connector={"line": {"color": "orange", "width": 1, "dash": "dot"}},
    ))
    fig.update_layout(
        title={"text": "Incremental SCU filter cascade", "font": {"color": "orange"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "orange"},
        height=80 * len(stages) + 100,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-step drop table ─────────────────────────────────────────────────
    drop_rows = []
    for _i in range(1, len(stages)):
        drop_rows.append({
            "Step": stages[_i],
            "Remaining": values[_i],
            "Dropped here": values[_i - 1] - values[_i],
            "% of parsed": f"{values[_i] / total * 100:.1f}%" if total else "—",
        })
    if drop_rows:
        st.dataframe(
            pd.DataFrame(drop_rows), hide_index=True, use_container_width=True
        )

    # ── Filtered dataset ────────────────────────────────────────────────────
    filtered = norm_df[mask]
    m1, m2, m3 = st.columns(3)
    m1.metric("Parsed rows", f"{total:,}")
    m2.metric("Rows passing filter", f"{len(filtered):,}")
    m3.metric(
        "% retained", f"{len(filtered) / total * 100:.1f}%" if total else "—"
    )

    st.session_state['scu_filtered_df'] = filtered
    _fdisp = filtered.copy()
    for _c in _fdisp.columns:
        if _fdisp[_c].dtype == 'object':
            _fdisp[_c] = _fdisp[_c].astype(str)
    st.dataframe(_fdisp)
    st.download_button(
        "Download filtered CSV",
        data=convert_df(_fdisp),
        file_name="scu_filtered.csv",
        mime="text/csv",
        key="dl_scu_filtered_csv",
    )


def plot_line(df, x_column, y_columns, figsize=(12, 10), color='orange', title=None, rolling_mean_value=2):
    import matplotlib.cm as cm
    # Sort the dataframe by the date column
    df = df.sort_values(by=x_column)

    # Calculate rolling mean for each y_column
    if rolling_mean_value:
        df[y_columns] = df[y_columns].rolling(len(df) // rolling_mean_value).mean()

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = cm.Oranges(np.linspace(0.2, 1, len(y_columns)))

    # Plot each y_column as a separate line with a different color
    for i, y_column in enumerate(y_columns):
        df.plot(x=x_column, y=y_column, ax=ax, color=colors[i], label=y_column, linewidth=.5)

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

    # Format x_column as date if it is
    if np.issubdtype(df[x_column].dtype, np.datetime64) or np.issubdtype(df[x_column].dtype, np.timedelta64):
        df[x_column] = pd.to_datetime(df[x_column]).dt.date

    # Set title, labels, and legend
    ax.set_title(title or f'{", ".join(y_columns)} over {x_column}', color=color, fontweight='bold')
    ax.set_xlabel(x_column, color=color)
    ax.set_ylabel(', '.join(y_columns), color=color)
    ax.spines['bottom'].set_color('orange')
    ax.spines['top'].set_color('orange')
    ax.spines['right'].set_color('orange')
    ax.spines['left'].set_color('orange')
    ax.xaxis.label.set_color('orange')
    ax.yaxis.label.set_color('orange')
    ax.tick_params(axis='x', colors='orange')
    ax.tick_params(axis='y', colors='orange')
    ax.title.set_color('orange')

    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), facecolor='black', framealpha=.4, labelcolor='orange', edgecolor='orange')

    # Remove background
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    return fig


def plot_bar(df, x_column, y_column, figsize=(12, 10), color='orange', title=None):
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(data=df, x=x_column, y=y_column, color=color, ax=ax)

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    ax.set_title(title if title else f'{y_column} by {x_column}', color=color, fontweight='bold')
    ax.set_xlabel(x_column, color=color)
    ax.set_ylabel(y_column, color=color)

    ax.tick_params(axis='x', colors=color)
    ax.tick_params(axis='y', colors=color)

    # Remove background
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.spines['bottom'].set_color('orange')
    ax.spines['top'].set_color('orange')
    ax.spines['right'].set_color('orange')
    ax.spines['left'].set_color('orange')
    ax.xaxis.label.set_color('orange')
    ax.yaxis.label.set_color('orange')
    ax.tick_params(axis='x', colors='orange')
    ax.tick_params(axis='y', colors='orange')
    ax.title.set_color('orange')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), facecolor='black', framealpha=.4, labelcolor='orange', edgecolor='orange')
    return fig

def plot_grouped_bar(df, x_columns, y_column, figsize=(12, 10), colors=None, title=None):
    fig, ax = plt.subplots(figsize=figsize)

    width = 0.8 / len(x_columns)  # the width of the bars
    x = np.arange(len(df))  # the label locations

    for i, x_column in enumerate(x_columns):
        sns.barplot(data=df, x=x, y=y_column, color=colors[i] if colors else None, ax=ax, width=width, label=x_column)
        x += width  # add the width of the bar to the x position for the next bar

    ax.set_title(title if title else f'{y_column} by {", ".join(x_columns)}', color='orange', fontweight='bold')
    ax.set_xlabel('Groups', color='orange')
    ax.set_ylabel(y_column, color='orange')

    ax.set_xticks(x - width * len(x_columns) / 2)
    ax.set_xticklabels(df.index)

    ax.tick_params(axis='x', colors='orange')
    ax.tick_params(axis='y', colors='orange')

    # Remove background
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.spines['bottom'].set_color('orange')
    ax.spines['top'].set_color('orange')
    ax.spines['right'].set_color('orange')
    ax.spines['left'].set_color('orange')
    ax.xaxis.label.set_color('orange')
    ax.yaxis.label.set_color('orange')
    ax.title.set_color('orange')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), facecolor='black', framealpha=.4, labelcolor='orange', edgecolor='orange')

    return fig

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    try:
        csv = df.to_csv().encode("utf-8")
    except:
        csv = df.to_csv().encode("utf-8-sig")
    return csv


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Shared filtering UI — delegates to the DataProcessor so the parsing and
    analysis pages use one and the same filter system (mirrors
    ``analyzing.py``'s ``filter_dataframe``). Binary 0/1 columns get a value
    picker; other numerics get range/percentile/std-dev modes.
    """
    return DataProcessor.filter_dataframe_enhanced(
        df, enable_quick_filters=False, enable_advanced_filters=True
    )


def filter_dataframe_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """

    title_font = "Arial"
    body_font = "Arial"
    title_size = 32
    colors = ["red", "green", "blue"]
    interpretation = False
    extract_docx = False
    title = "My Chart"
    regex = ".*"
    img_path = 'default_image.png'


    #try:
    #    modify = st.checkbox("Add filters on raw data")
    #except:
    #    try:
    #        modify = st.checkbox("Add filters on processed data")
    #    except:
    #        try:
    #            modify = st.checkbox("Add filters on parsed data")
    #        except:
    #            pass

    #if not modify:
    #    return df

    df_ = df.copy()
    # Try to convert datetimes into a standard format (datetime, no timezone)

#modification_container = st.container()

#with modification_container:
    to_filter_columns = st.multiselect("Filter dataframe on", df_.columns)

    date_column = None
    filtered_columns = []

    for column in to_filter_columns:
        left, right = st.columns((1, 20))
        # Treat columns with < 200 unique values as categorical if not date or numeric
        if is_categorical_dtype(df_[column]) or (df_[column].nunique() < 120 and not is_datetime64_any_dtype(df_[column]) and not is_numeric_dtype(df_[column])):
            user_cat_input = right.multiselect(
                f"Values for {column}",
                df_[column].value_counts().index.tolist(),
                default=list(df_[column].value_counts().index)
            )
            df_ = df_[df_[column].isin(user_cat_input)]
            filtered_columns.append(column)

            with st.status(f"Category Distribution: {column}", expanded=False) as stat:
                st.pyplot(plot_treemap(df_, column))

        elif is_numeric_dtype(df_[column]):
            _nonnull = df_[column].dropna()
            _uniq = set(_nonnull.unique())
            # Binary boolean columns (values ⊆ {0, 1}) — a slider is meaningless
            # here, so offer a value picker (0/1) instead of a min–max range.
            if _uniq and _uniq.issubset({0, 1}):
                _opts = sorted(_uniq)
                user_bin_input = right.multiselect(
                    f"Values for {column}",
                    _opts,
                    default=_opts,
                    format_func=lambda v: f"{int(v)} — {'true' if int(v) == 1 else 'false'}",
                    key=f"filt_bin_{column}",
                )
                df_ = df_[df_[column].isin(user_bin_input)]
                filtered_columns.append(column)

                with st.status(f"Binary Distribution: {column}", expanded=False):
                    _vc = df_[column].value_counts().sort_index()
                    _vc.index = _vc.index.map(
                        lambda v: f"{int(v)} ({'true' if int(v) == 1 else 'false'})"
                    )
                    st.bar_chart(_vc)
            else:
                _min = float(df_[column].min())
                _max = float(df_[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df_ = df_[df_[column].between(*user_num_input)]
                filtered_columns.append(column)

                # Chart_GPT = ChartGPT(df_, title_font, body_font, title_size,
                #      colors, interpretation, extract_docx, img_path)

                with st.status(f"Numerical Distribution: {column}", expanded=False) as stat_:
                    st.pyplot(plot_hist(df_, column, bins=int(round(len(df_[column].unique())-1)/2)))

        elif is_object_dtype(df_[column]):
            _orig_col = df_[column].copy()
            _is_valid_date = False
            try:
                df_[column] = pd.to_datetime(df_[column], errors='coerce').astype('datetime64[ms]')
            except Exception:
                pass

            if is_datetime64_any_dtype(df_[column]):
                try:
                    df_[column] = df_[column].dt.tz_localize(None)
                except TypeError:
                    df_[column] = df_[column].dt.tz_convert(None)
                valid = df_[column].dropna()
                if not valid.empty:
                    min_date = valid.min().date()
                    max_date = valid.max().date()
                    if min_date != max_date:
                        _is_valid_date = True
                        user_date_input = right.date_input(
                            f"Values for {column}",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                        )
                        if len(user_date_input) == 2:
                            user_date_input = tuple(map(pd.to_datetime, user_date_input))
                            start_date, end_date = user_date_input

                            time_units = {
                                'year': df_[column].dt.year,
                                'month': df_[column].dt.to_period('M'),
                                'day': df_[column].dt.date
                            }
                            unique_counts = {unit: col.nunique() for unit, col in time_units.items()}
                            closest_to_36 = min(unique_counts, key=lambda k: abs(unique_counts[k] - 36))

                            grouped = df_.groupby(time_units[closest_to_36]).size().reset_index(name='count')
                            grouped.columns = [column, 'count']

                            if closest_to_36 == 'year':
                                date_range = pd.date_range(start=f"{start_date.year}-01-01", end=f"{end_date.year}-12-31", freq='YS')
                            elif closest_to_36 == 'month':
                                date_range = pd.date_range(start=start_date.replace(day=1), end=end_date + pd.offsets.MonthEnd(0), freq='MS')
                            else:
                                date_range = pd.date_range(start=start_date, end=end_date, freq='D')

                            complete_range = pd.DataFrame({column: date_range})
                            if closest_to_36 == 'year':
                                complete_range[column] = complete_range[column].dt.year
                            elif closest_to_36 == 'month':
                                complete_range[column] = complete_range[column].dt.to_period('M')

                            final_data = pd.merge(complete_range, grouped, on=column, how='left').fillna(0)
                            with st.status(f"Date Distributions: {column}", expanded=False) as stat:
                                try:
                                    st.pyplot(plot_bar(final_data, column, 'count'))
                                except Exception as e:
                                    st.error(f"Error plotting bar chart: {e}")
                            df_ = df_.loc[df_[column].between(start_date, end_date)]

                        date_column = column
                        if date_column and filtered_columns:
                            numeric_columns = [col for col in filtered_columns if is_numeric_dtype(df_[col])]
                            categorical_columns = [col for col in filtered_columns if is_categorical_dtype(df_[col])]
                            with st.status(f"Date Distribution: {column}", expanded=False) as stat:
                                if numeric_columns:
                                    try:
                                        st.pyplot(plot_line(df_, date_column, numeric_columns))
                                    except Exception as e:
                                        st.error(f"Error plotting line chart: {e}")
                                if categorical_columns:
                                    try:
                                        st.pyplot(plot_bar(df_, date_column, categorical_columns[0]))
                                    except Exception as e:
                                        st.error(f"Error plotting bar chart: {e}")

            if not _is_valid_date:
                df_[column] = _orig_col
                _txt_col, _btn_col = right.columns([5, 1])
                user_text_input = _txt_col.text_input(
                    f"Substring or regex in {column}",
                    key=f"regex_{column}",
                )
                _drop_null = _btn_col.checkbox("Drop nulls", key=f"dropnull_{column}")
                if user_text_input:
                    try:
                        mask = df_[column].astype(str).str.contains(user_text_input, case=False, na=False, regex=True)
                    except Exception:
                        mask = df_[column].astype(str).str.contains(user_text_input, case=False, na=False, regex=False)
                    df_ = df_.loc[mask]
                    filtered_columns.append(column)
                if _drop_null:
                    df_ = df_.loc[df_[column].notna() & (df_[column].astype(str).str.strip() != '')]
                    if column not in filtered_columns:
                        filtered_columns.append(column)
    # write len of df after filtering with % of original
    st.write(f"{len(df_)} rows ({len(df_) / len(df) * 100:.2f}%)")
    return df_


from config import (
    FORMAT_MASTER_SCU_V1,
    FORMAT_LONG, FORMAT_LONG_XLSX, FORMAT_SCU_V1, FORMAT_SCU_V2, FORMAT_SCU_V3, FORMAT_MERGED, FORMAT_UFOSETI_RU,
    FORMAT_NUFORC, FORMAT_BLUE_BOOK, FORMAT_UK_NATIONAL_ARCHIVES,
    FORMAT_COBEPS_NOTIFICATIONS_PAN, FORMAT_COBEPS_COB_2021,
    FORMAT_GEP, FORMAT_UPDB_NICAP, FORMAT_OVNIBASE, FORMAT_UFOSETI,
    FORMAT_UAP_SIGHTINGS_GITHUB, FORMAT_WEINSTEIN_PILOT_CATALOG,
    FORMAT_PETROWITSCH_LATAM, FORMAT_UFOCAT,
    FORMAT_CISU_CISUCAT, FORMAT_CUFOC_1977_ITALY_FRANCE, FORMAT_UAPCHECK,
)

OPENAI_KEY   = st.secrets["OPENAI_KEY"]
GEMINI_KEY   = st.secrets["GEMINI_KEY"]
DEEPSEEK_KEY = st.secrets.get("DEEPSEEK_KEY", "")


# ── Schema registry (single source of truth for the format pickers) ────────────
# Maps a human label to its schema (a dict, or a JSON string for FORMAT_LONG).
# Both the format selector and the schema↔dataset comparison read from here so
# the two never drift apart.
SCHEMA_FORMATS = {
    "MasterSCU_v1":                    FORMAT_MASTER_SCU_V1,
    "SCU_v1":                          FORMAT_SCU_V1,
    "Default UAP Format":              FORMAT_LONG,
    "SCU Spreadsheet":                 FORMAT_LONG_XLSX,
    "SCU_v2":                          FORMAT_SCU_V2,
    "SCU_v3":                          FORMAT_SCU_V3,
    "UFOSETI (RU)":                    FORMAT_UFOSETI_RU,
    "NUFORC":                          FORMAT_NUFORC,
    "Blue Book (USAF)":                FORMAT_BLUE_BOOK,
    "UK National Archives":            FORMAT_UK_NATIONAL_ARCHIVES,
    "COBEPS — PAN Notifications (BE)": FORMAT_COBEPS_NOTIFICATIONS_PAN,
    "COBEPS — COB 2021 (BE)":          FORMAT_COBEPS_COB_2021,
    "GEP (DE)":                        FORMAT_GEP,
    "UPDB / NICAP":                    FORMAT_UPDB_NICAP,
    "OVNIBASE (FR)":                   FORMAT_OVNIBASE,
    "UFOSETI (raw)":                   FORMAT_UFOSETI,
    "UAP Sightings GitHub":            FORMAT_UAP_SIGHTINGS_GITHUB,
    "Weinstein Pilot Catalog":         FORMAT_WEINSTEIN_PILOT_CATALOG,
    "Petrowitsch LATAM":               FORMAT_PETROWITSCH_LATAM,
    "UFOCAT 2023":                     FORMAT_UFOCAT,
    "CISU / CISUCAT (IT)":             FORMAT_CISU_CISUCAT,
    "CUFOC 1977 Italy-France":         FORMAT_CUFOC_1977_ITALY_FRANCE,
    "UAPCHECK Registry":               FORMAT_UAPCHECK,
}


# Logical grouping of the schema formats — this is the interoperability layer:
# every curated dataset is compared against one of these standard schemas, and
# the schema picker presents them grouped so the right standard is easy to find.
SCHEMA_FORMAT_GROUPS = {
    "Canonical & SCU": [
        "MasterSCU_v1", "SCU_v1", "Default UAP Format", "SCU Spreadsheet", "SCU_v2", "SCU_v3",
    ],
    "Government & official archives": [
        "Blue Book (USAF)", "UK National Archives",
    ],
    "European national databases": [
        "COBEPS — PAN Notifications (BE)", "COBEPS — COB 2021 (BE)",
        "GEP (DE)", "OVNIBASE (FR)", "CISU / CISUCAT (IT)",
        "CUFOC 1977 Italy-France",
    ],
    "Research catalogs": [
        "UPDB / NICAP", "UFOCAT 2023", "Weinstein Pilot Catalog",
        "Petrowitsch LATAM",
    ],
    "Public & crowd-sourced": [
        "NUFORC", "UAP Sightings GitHub", "UAPCHECK Registry",
    ],
    "UFOSETI / SETI": [
        "UFOSETI (RU)", "UFOSETI (raw)",
    ],
}


# Short provenance blurb per schema (drawn from the config.py header comments)
# so the picker explains where each dataset/standard comes from at a glance.
SCHEMA_FORMAT_ORIGINS = {
    "MasterSCU_v1":
        "The default extraction schema — the full canonical SCU master schema "
        "(uap_master_schema.json). The most complete field set in the app: "
        "record/study provenance, source, date_time, location, witness, "
        "detection, object, behavior (+performance), anomaly, engagement "
        "(types/flags), military, effects, entities (+morphology/tools), "
        "contact, environment, evidence, classification, assessment, scenarios "
        "(ICD-203 intention scoring), investigation, narrative and context.",
    "SCU_v1":
        "A compact SCU schema — the full SCU_v3 field set minus the "
        "verbose `sightingDetails` narrative block and the all-null `manifest` "
        "join placeholders. Keeps source, assessment, anomaly, location, "
        "witness, craft, performance, military, effects and engagement fields.",
    "Default UAP Format":
        "The project's canonical nested narrative schema (`sightingDetails`). "
        "General-purpose GPT extraction template covering location, craft, "
        "observer, evidence and effects.",
    "SCU Spreadsheet":
        "Mirrors the *UAP Activities Data 1975 Onwards* spreadsheet column "
        "groups; used by `uap_xlsx_filler.py` to populate the SCU spreadsheet.",
    "SCU_v2":
        "A redlined evolution of the SCU spreadsheet schema — adds "
        "source/assessment/anomaly blocks, tri-state (Y/N/U) flags, structured "
        "witness roles and an evidence array for the SCU five-criterion gate.",
    "SCU_v3":
        "Extends SCU_v2 with UFOSETI assessment frameworks: UFOCAT-style "
        "explanation taxonomy, Hynek–Vallée recovery + biological-entity flags, "
        "and a human-injury medical-effect extension.",
    "UFOSETI (RU)":
        "LLM extraction template mirroring the Russian UFOSETI database export; "
        "anomaly codes follow the UFOCAT Codebook 2023.",
    "NUFORC":
        "US National UFO Reporting Center — the large public US crowd-sourced "
        "report database (~70,000 rows, EN).",
    "Blue Book (USAF)":
        "US Air Force Project Blue Book official case files, 1947–1969 "
        "(~15,000 rows, EN).",
    "UK National Archives":
        "UK Ministry of Defence / National Archives UAP catalogue, 1978–2000 "
        "(~2,827 rows, EN).",
    "COBEPS — PAN Notifications (BE)":
        "Belgium's COBEPS official PAN (UAP) notifications up to 31 Dec 2023 "
        "(~1,308 rows, FR).",
    "COBEPS — COB 2021 (BE)":
        "COBEPS *COB Online* historical Belgian catalogue, pre-2021 "
        "(~2,111 rows, FR).",
    "GEP (DE)":
        "German GEP UFO/UAP case data, 1972–2023 (~5,577 rows, DE).",
    "UPDB / NICAP":
        "UPDB / NICAP *NSID* database of US cases (~5,885 rows, EN).",
    "OVNIBASE (FR)":
        "French OVNIBASE UFO database (~3,145 rows, FR).",
    "UFOSETI (raw)":
        "Raw column schema of the UFOSETI.org Russian/international UAP dataset "
        "(~3,069 rows, EN+RU). The RU variant above is its extraction template.",
    "UAP Sightings GitHub":
        "MiChaelinzo/UAP_SIGTHINGS multi-source sighting JSON aggregation "
        "(~1,000-row sample, EN).",
    "Weinstein Pilot Catalog":
        "Dominique Weinstein's catalogue of aircraft-crew UAP encounters, "
        "1916–1999 (~1,300 rows, EN).",
    "Petrowitsch LATAM":
        "Petrowitsch catalogue of Chilean & Latin American UAP cases "
        "(~3,529+ rows, ES).",
    "UFOCAT 2023":
        "UFOCAT 2023 from the J. Allen Hynek Center for UFO Studies / Sun River "
        "Research Institute — a very large research catalogue (~146,000 rows, EN).",
    "CISU / CISUCAT (IT)":
        "The Italian CISU national UAP catalogue (CISUCAT) "
        "(~50,000+ rows est., IT).",
    "CUFOC 1977 Italy-France":
        "CUFOC study of the 1977 Italy–France wave (Bourdon/Delaval "
        "digitisation), FR/IT.",
    "UAPCHECK Registry":
        "UAPCHECK.com international data-exchange API registry of 22 "
        "organisations — an organisational registry, not a sighting catalogue.",
}


def schema_format_groups() -> dict:
    """SCHEMA_FORMAT_GROUPS, but with any label not yet assigned to a group
    swept into an 'Other' bucket — so a newly added schema never disappears
    from the picker if someone forgets to slot it into a group."""
    grouped = {g: [lbl for lbl in labels if lbl in SCHEMA_FORMATS]
               for g, labels in SCHEMA_FORMAT_GROUPS.items()}
    assigned = {lbl for labels in grouped.values() for lbl in labels}
    leftover = [lbl for lbl in SCHEMA_FORMATS if lbl not in assigned]
    if leftover:
        grouped["Other"] = leftover
    return {g: labels for g, labels in grouped.items() if labels}


def _fmt_as_dict(v) -> dict:
    """Coerce a schema entry (dict or JSON string) to a dict; {} on failure."""
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v)
    except Exception:
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` onto `base` without mutating either."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _current_dataset_columns() -> pd.DataFrame:
    """Best-available view of the curated dataset's columns for comparison.

    Prefers the flattened parsed-table DataFrame; otherwise normalizes the
    parsed_responses record dict. Returns an empty frame when nothing's loaded.
    """
    df = st.session_state.get('parsed_responses_df')
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    pr = _parsed_responses_dict()
    if pr:
        try:
            return pd.json_normalize(list(pr.values()))
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def render_schema_dataset_diff(schema_dict: dict, df: pd.DataFrame,
                               ignore_parent: bool = True) -> dict:
    """Field-level coverage diff between a JSON schema and a dataset.

    Renders a red/green table:
      🔴 missing — a schema field with no matching dataset column
      🟢 added   — a dataset column the schema doesn't define
      ✓  matched — present in both

    `ignore_parent` matches on the leaf field name (so `location.city` ≡ `city`),
    which is what you want when the curated dataset flattened or renamed parents.
    Returns the three name-lists for downstream use (e.g. upsert planning).
    """
    norm = (lambda s: s.split(".")[-1]) if ignore_parent else (lambda s: s)

    schema_leaves = list(_flatten_dotted(schema_dict).keys())
    data_cols = [str(c) for c in df.columns]

    # First occurrence wins so we can show a representative original name.
    s_map: dict = {}
    for s in schema_leaves:
        s_map.setdefault(norm(s), s)
    d_map: dict = {}
    for c in data_cols:
        d_map.setdefault(norm(c), c)

    s_keys, d_keys = set(s_map), set(d_map)
    matched_k = sorted(s_keys & d_keys)
    missing_k = sorted(s_keys - d_keys)
    extra_k   = sorted(d_keys - s_keys)

    n_schema = len(s_keys)
    coverage = (len(matched_k) / n_schema * 100) if n_schema else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Schema fields", n_schema)
    m2.metric("✓ Matched", len(matched_k))
    m3.metric("🔴 Missing", len(missing_k),
              help="Defined in the schema but absent from the dataset.")
    m4.metric("🟢 Added", len(extra_k),
              help="Present in the dataset but not defined by the schema.")
    st.progress(min(1.0, coverage / 100),
                text=f"Schema coverage: {coverage:.0f}% of schema fields present in the dataset")

    rows = (
        [{"Field": s_map[k], "Status": "✓ In both"} for k in matched_k]
        + [{"Field": s_map[k], "Status": "🔴 Missing from dataset"} for k in missing_k]
        + [{"Field": d_map[k], "Status": "🟢 Added (not in schema)"} for k in extra_k]
    )
    diff_df = pd.DataFrame(rows, columns=["Field", "Status"])

    def _row_style(r):
        status = r["Status"]
        if status.startswith("🔴"):
            color = "rgba(255, 75, 75, 0.22)"
        elif status.startswith("🟢"):
            color = "rgba(60, 200, 90, 0.22)"
        else:
            color = ""
        return [f"background-color: {color}"] * len(r)

    st.dataframe(
        diff_df.style.apply(_row_style, axis=1),
        hide_index=True, use_container_width=True,
    )
    st.download_button(
        "⬇️ Download schema↔dataset diff (CSV)",
        diff_df.to_csv(index=False),
        "schema_dataset_diff.csv", "text/csv", key="dl_schema_dataset_diff",
    )

    # ── Send a field subset to the extraction JSON editor ─────────────────────
    # Each button builds a pruned JSON schema from one side of the diff and
    # stashes it in session_state; the "UAP Feature Extraction" JSON editor
    # (rendered later in the same page) picks it up and fills its text area.
    flat_schema = _flatten_dotted(schema_dict)  # {dotted schema path: description}
    src_payload  = dict(flat_schema)                                                   # every schema field
    both_payload = {s_map[k]: flat_schema.get(s_map[k], "string") for k in matched_k}  # ✓ in both
    miss_payload = {s_map[k]: flat_schema.get(s_map[k], "string") for k in missing_k}  # 🔴 missing
    dest_payload = {d_map[k]: flat_schema.get(s_map[k], "string") for k in matched_k}  # dataset columns
    dest_payload.update({d_map[k]: "string — in dataset, not defined by the schema"
                         for k in extra_k})

    st.caption(
        "**Send to extract** — fill the JSON Format editor on the **UAP Feature "
        "Extraction** step with one side of this diff:"
    )

    def _send_to_extract(note: str, payload: dict) -> None:
        n = len(payload)
        st.session_state["_extract_seed_json"] = json.dumps(
            _unflatten_dotted(payload), indent=2
        )
        st.session_state["_extract_seed_note"] = f"{note} ({n} field{'' if n == 1 else 's'})"
        st.toast(f"📤 Sent {n} field{'' if n == 1 else 's'} to the extraction JSON editor.")

    sb1, sb2, sb3, sb4 = st.columns(4)
    if sb1.button(f"📤 Source · {len(src_payload)}", key="send_extract_source",
                  use_container_width=True,
                  help="All fields defined by the selected schema."):
        _send_to_extract("Source / schema fields", src_payload)
    if sb2.button(f"📤 Destination · {len(dest_payload)}", key="send_extract_dest",
                  use_container_width=True,
                  help="Every column in the uploaded dataset (matched + 🟢 extra)."):
        _send_to_extract("Destination / dataset columns", dest_payload)
    if sb3.button(f"📤 Missing · {len(miss_payload)}", key="send_extract_missing",
                  use_container_width=True, disabled=not miss_payload,
                  help="🔴 Schema fields the dataset lacks — extract these to fill the gaps."):
        _send_to_extract("Missing fields", miss_payload)
    if sb4.button(f"📤 Both · {len(both_payload)}", key="send_extract_both",
                  use_container_width=True, disabled=not both_payload,
                  help="✓ Fields present in both the schema and the dataset."):
        _send_to_extract("Matched fields", both_payload)

    return {
        "matched": [s_map[k] for k in matched_k],
        "missing": [s_map[k] for k in missing_k],
        "added":   [d_map[k] for k in extra_k],
    }


def render_schema_dataset_diff_ui(df: pd.DataFrame) -> None:
    """Grouped schema picker → explicit Compare → coverage diff.

    The schema formats act as the interoperability layer: you upload a curated
    dataset, pick the standard schema to map it onto (organised by logical
    group), then press **Compare**. Results persist across reruns so changing
    other widgets doesn't wipe them.
    """
    st.caption(
        "Pick the standard schema to map your curated dataset onto, then press "
        "**Compare**. **🔴 red** = in the schema but missing from the dataset; "
        "**🟢 green** = extra columns in the dataset the schema doesn't define."
    )

    groups = schema_format_groups()
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        group = st.selectbox(
            "Schema group", list(groups.keys()), key="schema_diff_group"
        )
    with c2:
        # Reset the schema choice when the group changes so it stays valid.
        _opts = groups.get(group, [])
        if st.session_state.get("schema_diff_schema") not in _opts:
            st.session_state["schema_diff_schema"] = _opts[0] if _opts else None
        schema_label = st.selectbox(
            "Schema", _opts, key="schema_diff_schema"
        )
    with c3:
        ignore_parent = st.checkbox(
            "Match on field name only",
            value=True, key="schema_diff_ignore_parent",
            help="Treat `location.city` and `city` as the same field — useful "
                 "when the curated dataset flattens or renames parent paths.",
        )

    if schema_label and schema_label in SCHEMA_FORMAT_ORIGINS:
        st.caption(f"ℹ️ *Origin —* {SCHEMA_FORMAT_ORIGINS[schema_label]}")

    st.caption(f"Dataset: **{len(df):,}** rows × **{len(df.columns)}** columns.")

    if st.button("🔍 Compare", type="primary", key="schema_diff_run",
                 disabled=not schema_label):
        st.session_state["schema_diff_result"] = {
            "schema_label": schema_label,
            "ignore_parent": ignore_parent,
        }

    res = st.session_state.get("schema_diff_result")
    if not res:
        st.info("Choose a schema and press **Compare** to see the coverage diff.")
        return
    if res["schema_label"] not in SCHEMA_FORMATS:
        st.session_state.pop("schema_diff_result", None)
        return

    st.markdown(f"**Comparing dataset against schema:** `{res['schema_label']}`")
    merged = _fmt_as_dict(SCHEMA_FORMATS[res["schema_label"]])
    render_schema_dataset_diff(merged, df, ignore_parent=res["ignore_parent"])

with torch.no_grad():
    torch.cuda.empty_cache()

#st.set_page_config(
#    page_title="UAP ANALYSIS",
#    page_icon=":alien:",
#    layout="wide",
#    initial_sidebar_state="expanded",
#)

st.title('UAP Feature Extraction')

# Initialize session state
if 'analyzers' not in st.session_state:
    st.session_state['analyzers'] = []
if 'col_names' not in st.session_state:
    st.session_state['col_names'] = []
if 'clusters' not in st.session_state:
    st.session_state['clusters'] = {}
if 'new_data' not in st.session_state:
    st.session_state['new_data'] = pd.DataFrame()
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = pd.DataFrame()
if 'data_processed' not in st.session_state:
    st.session_state['data_processed'] = False
if 'stage' not in st.session_state:
    st.session_state['stage'] = 0
if 'filtered_data' not in st.session_state:
    st.session_state['filtered_data'] = None
if 'gemini_answer' not in st.session_state:
    st.session_state['gemini_answer'] = None
if 'parsed_responses' not in st.session_state:
    st.session_state['parsed_responses'] = None
if 'parsed_responses_df' not in st.session_state:
    st.session_state['parsed_responses_df'] = None
if 'json_format' not in st.session_state:
    st.session_state['json_format'] = None
if 'api_key_valid' not in st.session_state:
    st.session_state['api_key_valid'] = False
if 'previous_api_key' not in st.session_state:
    st.session_state['previous_api_key'] = None

    
# Unparsed data
#unparsed_tickbox = st.checkbox('Data Parsing')
#if unparsed_tickbox:
col_up1, col_up2 = st.columns(2)
unparsed = col_up1.file_uploader("Upload Raw DataFrame (CSV / XLSX / JSON)", type=["csv", "xlsx", "json"])
restore_file = col_up2.file_uploader(
    "Or restore parsed data (JSON / CSV / XLSX)",
    type=["json", "csv", "xlsx"],
    help=(
        "Skip re-parsing by restoring a previous session — the loader is "
        "chosen by file type:\n\n"
        "• **parsed_responses.json** — the exported records dict.\n\n"
        "• **a saved flat parsed table (CSV / XLSX, e.g. uap_data.csv)** — each "
        "row is re-nested into the parsed_responses dict so the SCU normalizer "
        "can run; extra/custom schema columns pass through unchanged."
    ),
)
# One uploader, two loaders: route by extension so the JSON-restore and
# parsed-table-restore paths below keep working unchanged.
_is_json_restore = (
    restore_file is not None and restore_file.name.lower().endswith(".json")
)
json_restore = restore_file if _is_json_restore else None
parsed_table_restore = (
    restore_file if (restore_file is not None and not _is_json_restore) else None
)

if json_restore is not None:
    try:
        restored = json.loads(json_restore.read())
        st.session_state['parsed_responses'] = restored
        st.success(f"Loaded {len(restored)} records from JSON.")
    except Exception as e:
        st.error(f"Could not read JSON: {e}")

if parsed_table_restore is not None:
    try:
        import ast as _ast

        _flat_df = read_uploaded_sheet(parsed_table_restore, key="restore_sheet")
        # Strip the unnamed index column pandas writes by default on to_csv()
        _flat_df = _flat_df.loc[
            :, ~_flat_df.columns.astype(str).str.match(r"^Unnamed: \d+$")
        ]

        # CSV round-trip stringifies list / dict cells (e.g. witness.roles
        # becomes the literal string "['Civilian']"), which scu_normalizer's
        # _roles_from_value can't parse — it isinstance-checks for list and
        # then falls through to a comma-split that fails on the brackets.
        # ast.literal_eval rehydrates any cell that looks like a Python
        # literal collection so list-valued normalizer paths see real lists.
        def _maybe_literal(v):
            if not isinstance(v, str):
                return v
            s = v.strip()
            if len(s) >= 2 and ((s[0] == "[" and s[-1] == "]") or
                                (s[0] == "{" and s[-1] == "}") or
                                (s[0] == "(" and s[-1] == ")")):
                try:
                    return _ast.literal_eval(s)
                except (ValueError, SyntaxError):
                    return v
            return v

        # Re-nest each row into a flat dotted-key dict; scu_normalizer reads
        # dotted columns (location.country, source.name, …) verbatim, and
        # _strip_or_none() in the normalizer already drops "nan"/"none"/"null"
        # string sentinels, so no extra cleaning is needed here.
        restored_table = {
            str(idx): {
                k: _maybe_literal(v)
                for k, v in row.items() if pd.notna(v)
            }
            for idx, row in _flat_df.iterrows()
        }
        st.session_state['parsed_responses'] = restored_table
        st.success(
            f"Loaded {len(restored_table)} rows × {len(_flat_df.columns)} columns. "
            "Enable 'Apply SCU normalization' in the sidebar to normalize."
        )
    except Exception as e:
        st.error(f"Could not read CSV/XLSX: {e}")

# ── JSON-only export controls (shown when JSON loaded but no CSV uploaded) ─────
if json_restore is not None and unparsed is None and _parsed_responses_dict():
    st.divider()
    st.markdown("**Loaded from JSON — export options**")

    json_str = download_json(st.session_state['parsed_responses'])
    st.download_button(
        label="Download Parsed Data as JSON",
        data=json_str,
        file_name="parsed_responses.json",
        mime="application/json",
        key="dl_json_restore",
    )
    if st.button("Convert to DataFrame", key="btn_convert_json"):
        convert_cached_data_to_df(st.session_state['parsed_responses'])

    if st.session_state.get('parsed_responses_df') is not None:
        st.download_button(
            label="Save CSV",
            data=convert_df(st.session_state['parsed_responses_df']),
            file_name="uap_data.csv",
            mime="text/csv",
            key="dl_csv_from_json",
        )

    st.divider()
    st.markdown("**Export to SCU Spreadsheet**")
    xlsx_template_j = st.file_uploader(
        "Upload template spreadsheet (.xlsx)",
        type=["xlsx"],
        key="scu_template_json",
        help="Upload 'UAP Activities Data 1975 Onwards.xlsx'.",
    )
    start_num_j = st.number_input(
        "Starting Internal Number (0 = leave blank)", min_value=0, value=0, step=1,
        key="scu_startnum_json",
    )
    if st.button("Generate SCU xlsx", key="btn_scu_json"):
        if xlsx_template_j is None:
            st.warning("Please upload the template .xlsx first.")
        else:
            try:
                import tempfile
                from uap_xlsx_filler import build_dataframe, append_to_xlsx

                _proxy = CachedUAPParser.__new__(CachedUAPParser)
                _proxy.responses = {}
                import threading as _t2
                _proxy._responses_lock = _t2.Lock()

                new_df = build_dataframe(_proxy, st.session_state['parsed_responses'])
                st.write(f"Built DataFrame: {len(new_df)} rows × {len(new_df.columns)} columns")

                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                    tmp.write(xlsx_template_j.read())
                    out_path = tmp.name
                append_to_xlsx(
                    new_df,
                    xlsx_path=out_path,
                    output_path=out_path,
                    start_internal_number=int(start_num_j) if start_num_j > 0 else None,
                )
                with open(out_path, "rb") as f:
                    xlsx_bytes = f.read()
                st.download_button(
                    label="Download SCU xlsx",
                    data=xlsx_bytes,
                    file_name="UAP_Activities_filled.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_scu_from_json",
                )
                st.success(f"{len(new_df)} rows written to spreadsheet.")
            except Exception as e:
                st.error(f"xlsx export failed: {e}")

# ── Schema ↔ Dataset alignment (re-ingestion coverage diff) ────────────────────
# Available whenever a curated dataset is loaded into the session (via the JSON
# or parsed-table uploaders above). Compares the chosen schema's fields against
# the dataset's columns and colours the gap red/green so you can see, at a
# glance, what the schema expects that the dataset lacks (🔴) and what extra
# curated columns the schema doesn't yet cover (🟢).
_cmp_df = _current_dataset_columns()
if not _cmp_df.empty:
    with st.expander("🟢🔴 Compare schema to dataset (coverage diff)", expanded=False):
        render_schema_dataset_diff_ui(_cmp_df)

# ── Markdown Folder Ingestion ──────────────────────────────────────────────────
# Hidden from the UI for now; flip the gate to re-enable. Block body is
# unchanged so swapping `if False` back to `with st.expander(...)` restores it.
if False:
    st.markdown(
        "Upload a batch of `.md` files (e.g. case notes, reports). "
        "Each file is sent to the LLM and extracted into **FORMAT_MERGED** — "
        "a combined schema covering both UAP and SCU fields — "
        "then concatenated into a single DataFrame stored in the session."
    )

    md_files = st.file_uploader(
        "Upload .md files",
        type=["md"],
        accept_multiple_files=True,
        key="md_folder_uploader",
    )

    if md_files:
        st.caption(f"{len(md_files)} file(s) selected.")

        col_md1, col_md2 = st.columns(2)
        with col_md1:
            md_provider = st.radio(
                "Provider", ["OpenAI", "DeepSeek"], horizontal=True, key="md_provider"
            )
        with col_md2:
            md_model = st.selectbox(
                "Model",
                OPENAI_MODELS if md_provider == "OpenAI" else DEEPSEEK_MODELS,
                index=0,
                key="md_model",
            )

        md_api_key = st.text_input(
            f"{md_provider} API Key",
            OPENAI_KEY if md_provider == "OpenAI" else DEEPSEEK_KEY,
            type="password",
            key="md_api_key_input",
        )
        md_workers = st.slider(
            "Max parallel workers", 1, 32, 5, key="md_workers",
            help="Number of concurrent LLM calls. Start low for rate-limited providers.",
        )

        with st.expander("Preview FORMAT_MERGED schema", expanded=False):
            st.json(FORMAT_MERGED)

        if st.button("Process Markdown Files", key="btn_process_md"):
            if not md_api_key:
                st.warning("Please enter an API key.")
            else:
                import concurrent.futures as _cf

                md_contents: dict[str, str] = {}
                for _f in md_files:
                    try:
                        md_contents[_f.name] = _f.read().decode("utf-8")
                    except Exception as _e:
                        st.warning(f"Could not read {_f.name}: {_e}")

                _md_parser = CachedUAPParser(
                    api_key=md_api_key,
                    model=md_model,
                    use_batch=False,
                    col=[],
                )

                _merged_fmt = json.dumps(FORMAT_MERGED, indent=2)
                md_results: dict = {}
                md_errors: list[str] = []

                with st.status(
                    f"Processing {len(md_contents)} markdown file(s)…", expanded=True
                ) as _md_stat:
                    _progress = st.progress(0)
                    with _cf.ThreadPoolExecutor(max_workers=md_workers) as _executor:
                        _future_to_name = {
                            _executor.submit(
                                _md_parser.fetch_response, content, _merged_fmt
                            ): name
                            for name, content in md_contents.items()
                        }
                        _done = 0
                        for _future in _cf.as_completed(_future_to_name):
                            _name = _future_to_name[_future]
                            _done += 1
                            _progress.progress(_done / len(md_contents), text=_name)
                            try:
                                _resp = _future.result()
                                _text = (
                                    _resp.choices[0].message.content if _resp else None
                                )
                                if _text:
                                    _parsed = _extract_json(_text)
                                    if _parsed is not None:
                                        md_results[_name] = _parsed
                                    else:
                                        md_errors.append(
                                            f"{_name}: JSON extraction failed"
                                        )
                                else:
                                    md_errors.append(f"{_name}: empty response")
                            except Exception as _exc:
                                md_errors.append(f"{_name}: {_exc}")

                    if md_errors:
                        st.warning(f"{len(md_errors)} file(s) had errors.")
                        with st.expander("Error details"):
                            for _err in md_errors:
                                st.code(_err)

                    if md_results:
                        st.session_state['parsed_responses'] = md_results
                        st.success(
                            f"{len(md_results)} file(s) processed. "
                            "Results stored — use the export controls below."
                        )
                        _md_stat.update(
                            label="Processing complete", state="complete", expanded=False
                        )

                        # Build flat preview DataFrame keyed by filename
                        try:
                            _records = []
                            for _fname, _parsed in md_results.items():
                                _flat: dict = {"_filename": _fname}
                                for _group, _fields in _parsed.items():
                                    if isinstance(_fields, dict):
                                        for _k, _v in _fields.items():
                                            if isinstance(_v, dict):
                                                for _k2, _v2 in _v.items():
                                                    _flat[f"{_group}.{_k}.{_k2}"] = _v2
                                            else:
                                                _flat[f"{_group}.{_k}"] = _v
                                    else:
                                        _flat[_group] = _fields
                                _records.append(_flat)

                            _md_df = pd.DataFrame(_records).fillna("")
                            for _c in _md_df.columns:
                                if _md_df[_c].dtype == "object":
                                    _md_df[_c] = _md_df[_c].astype(str)
                            st.dataframe(_md_df)
                            st.session_state['parsed_responses_df'] = _md_df
                        except Exception as _df_err:
                            st.error(f"Could not build preview DataFrame: {_df_err}")
                    else:
                        st.error("No files were processed successfully.")

    # ── Inline export controls (available after processing) ───────────────
    if _parsed_responses_dict() and md_files:
        _pr = st.session_state['parsed_responses']
        _json_dl = json.dumps(_pr, indent=2)
        st.download_button(
            "Download parsed JSON",
            data=_json_dl,
            file_name="md_parsed_responses.json",
            mime="application/json",
            key="dl_md_json",
        )
        if st.session_state.get('parsed_responses_df') is not None:
            st.download_button(
                "Download CSV",
                data=convert_df(st.session_state['parsed_responses_df']),
                file_name="md_parsed_data.csv",
                mime="text/csv",
                key="dl_md_csv",
            )

st.divider()

if unparsed is not None:
    filtered_data = None
    try:
        if unparsed.type == "application/json" or unparsed.name.endswith(".json"):
            raw = unparsed.read().decode("utf-8").strip()
            # Support both JSONL (one object per line) and a single JSON array/object
            try:
                parsed_json = json.loads(raw)
                if isinstance(parsed_json, list):
                    data = pd.json_normalize(parsed_json)
                else:
                    data = pd.json_normalize([parsed_json])
            except json.JSONDecodeError:
                # Try repairing a truncated JSON array: find the last complete "}" and
                # close the array there so partial trailing records are dropped cleanly.
                repaired = None
                if raw.lstrip().startswith("["):
                    last_brace = raw.rfind("}")
                    if last_brace != -1:
                        candidate = raw[: last_brace + 1].rstrip().rstrip(",") + "\n]"
                        try:
                            repaired = json.loads(candidate)
                        except json.JSONDecodeError:
                            pass
                if repaired is not None:
                    n_dropped = raw.count('"pages"') - len(repaired)
                    if n_dropped > 0:
                        st.warning(
                            f"JSON was truncated — loaded {len(repaired)} complete records "
                            f"and dropped {n_dropped} incomplete trailing record(s)."
                        )
                    data = pd.json_normalize(repaired)
                else:
                    # Last resort: JSONL (newline-delimited JSON)
                    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
                    data = pd.json_normalize(records)

            # Detect columns that contain lists-of-dicts and offer to explode one.
            nested_cols = [
                c for c in data.columns
                if data[c].dropna().apply(lambda v: isinstance(v, list)).any()
            ]
            if nested_cols:
                explode_col = st.selectbox(
                    "Nested array detected — explode column into rows:",
                    options=["(keep as-is)"] + nested_cols,
                    index=1 if len(nested_cols) == 1 else 0,
                    key="json_explode_col",
                    help="Select the column whose list-of-objects should become individual rows. "
                         "Parent fields (e.g. source_file) are carried forward on every row.",
                )
                if explode_col != "(keep as-is)":
                    # Carry parent scalar columns forward, explode the nested array,
                    # then json_normalize the dicts inside it.
                    parent_cols = [c for c in data.columns if c != explode_col]
                    exploded = data.explode(explode_col).reset_index(drop=True)
                    nested_df = pd.json_normalize(
                        exploded[explode_col].dropna().tolist()
                    )
                    nested_df.index = exploded[explode_col].dropna().index
                    # Some datasets repeat envelope-level fields (e.g. document_id,
                    # agency, collection) inside each nested record; others keep
                    # them only on the document. Drop the redundant nested copies
                    # so the join works for both shapes instead of raising
                    # "columns overlap but no suffix specified" — the
                    # envelope-level value is authoritative.
                    overlap = [c for c in nested_df.columns if c in parent_cols]
                    if overlap:
                        nested_df = nested_df.drop(columns=overlap)
                    data = exploded[parent_cols].join(nested_df).reset_index(drop=True)
                    if overlap:
                        st.caption(
                            f"De-duplicated {len(overlap)} field(s) present at both "
                            f"the document and report level: {', '.join(overlap)}."
                        )
                    st.caption(
                        f"Exploded **{explode_col}** → {len(data):,} rows "
                        f"({', '.join(nested_df.columns.tolist()[:6])}{'…' if len(nested_df.columns) > 6 else ''})"
                    )
        else:
            data = read_uploaded_sheet(unparsed, key="unparsed_sheet")
        filtered_data = filter_dataframe(data)
        st.dataframe(filtered_data)
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")

    if filtered_data is None:
        st.stop()

    # ── Format selector ────────────────────────────────────────────────────────
    _FORMAT_MAP = SCHEMA_FORMATS  # single source of truth (see module top)

    fmt_choices = st.multiselect(
        "Output JSON Format (select multiple to merge fields)",
        list(_FORMAT_MAP.keys()),
        default=[list(_FORMAT_MAP.keys())[0]],
    )
    if not fmt_choices:
        st.warning("Select at least one format.")
        st.stop()

    # _deep_merge / _fmt_as_dict are defined at module top (single source).
    _merged_dict: dict = {}
    for _lbl in fmt_choices:
        _merged_dict = _deep_merge(_merged_dict, _fmt_as_dict(_FORMAT_MAP[_lbl]))

    active_format = json.dumps(_merged_dict, indent=2)

    with st.expander("➕ Custom / Additional Fields (merged last)", expanded=False):
        st.caption(
            "Add or override schema fields. The **field builder** and the **raw "
            "JSON** stay in sync — edit either side. Use dotted paths "
            "(e.g. `location.city`) to reach nested levels."
        )

        if 'custom_fields_dict' not in st.session_state:
            st.session_state['custom_fields_dict'] = {}
        _src = st.session_state['custom_fields_dict']

        _col_form, _col_raw = st.columns(2)

        # ── Field builder (form view) — dynamic rows, no widget key so it always
        #    re-renders from the shared dict after a sync. ─────────────────────
        with _col_form:
            st.markdown("**Field builder**")
            _flat = _flatten_dotted(_src)
            _rows_df = pd.DataFrame(
                [{"Field (dotted path)": k, "Description / value": _value_to_text(v)}
                 for k, v in _flat.items()],
                columns=["Field (dotted path)", "Description / value"],
            )
            _edited = st.data_editor(
                _rows_df,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
            )
            _form_dict = _rows_to_dict(_edited)

        # ── Raw JSON (text view) — also keyless for the same reason. ──────────
        with _col_raw:
            st.markdown("**Raw JSON**")
            _raw_text = st.text_area(
                "JSON object — fields added or overriding the selected schema(s)",
                value=json.dumps(_src, indent=2) if _src else "{}",
                height=260,
                label_visibility="collapsed",
            )
            _raw_ok = True
            try:
                _raw_dict = json.loads(_raw_text) if _raw_text.strip() else {}
                if not isinstance(_raw_dict, dict):
                    _raw_ok = False
                    st.warning("Top-level custom JSON must be an object.")
                    _raw_dict = _src
            except json.JSONDecodeError as _je:
                _raw_ok = False
                st.warning(f"Custom fields JSON is invalid: {_je}")
                _raw_dict = _src

        # ── Reconcile: whichever view changed becomes the new source of truth.
        #    A rerun refreshes the untouched view from it. ─────────────────────
        if _raw_ok and _raw_dict != _src:
            st.session_state['custom_fields_dict'] = _raw_dict
            st.rerun()
        elif _form_dict != _src:
            st.session_state['custom_fields_dict'] = _form_dict
            st.rerun()

        _custom_extra = st.session_state['custom_fields_dict']
        if _custom_extra:
            _merged_dict = _deep_merge(_merged_dict, _custom_extra)
            active_format = json.dumps(_merged_dict, indent=2)

    # A "Send to extract" action from the schema↔dataset diff (above) seeds this
    # editor with a chosen field subset. Drop the persisted widget value so the
    # new payload becomes the text area's content (setting the keyed value
    # directly would clash with the value= arg passed to the widget below).
    _seed = st.session_state.pop("_extract_seed_json", None)
    if _seed is not None:
        active_format = _seed
        st.session_state.pop("preview_json_text", None)
        _seed_note = st.session_state.pop("_extract_seed_note", "")
        st.success(
            f"📥 Loaded {_seed_note} from the schema diff into the editor."
            if _seed_note else
            "📥 Loaded fields from the schema diff into the editor."
        )

    st.markdown("### Preview / Edit JSON Format")
    _tab_tree, _tab_fields, _tab_raw = st.tabs(["🌳 Tree", "📋 Fields", "✏️ Raw"])

    # ── Editable source of truth — lives in the Raw tab but executes every
    #    rerun (all tab bodies run), so the read-only views always reflect the
    #    latest text. ───────────────────────────────────────────────────────
    with _tab_raw:
        active_format = st.text_area(
            "JSON Format", active_format, height=500, key="preview_json_text"
        )
        st.download_button("Save Format", active_format)

    # Parse once for the read-only views (text may be mid-edit / invalid).
    try:
        _preview_obj = json.loads(active_format)
    except json.JSONDecodeError:
        _preview_obj = None

    with _tab_tree:
        if _preview_obj is None:
            st.info("Fix the JSON in the **Raw** tab to see the tree view.")
        else:
            # Top-level keys visible; nested objects start collapsed.
            try:
                st.json(_preview_obj, expanded=1)
            except TypeError:  # older Streamlit: no integer-depth support
                st.json(_preview_obj, expanded=False)

    with _tab_fields:
        if not isinstance(_preview_obj, dict):
            st.info("Fix the JSON in the **Raw** tab to see the field table.")
        else:
            _pflat = _flatten_dotted(_preview_obj)
            _fields_df = pd.DataFrame(
                [{"Field (dotted path)": k, "Description / type": _value_to_text(v)}
                 for k, v in _pflat.items()],
                columns=["Field (dotted path)", "Description / type"],
            )
            st.caption(f"{len(_fields_df)} leaf field(s) across the merged schema.")
            st.dataframe(
                _fields_df, hide_index=True, use_container_width=True
            )

    try:
        _fmt_parsed = json.loads(active_format)
        _top_keys = set(_fmt_parsed.keys()) if isinstance(_fmt_parsed, dict) else set()
        # Use the nested key only when FORMAT_LONG is the sole schema selected;
        # any merge with flat schemas must normalise from the top level instead.
        _extract_key = 'sightingDetails' if _top_keys == {'sightingDetails'} else None
        st.session_state['json_format'] = True
        st.session_state['_json_format_error'] = None
    except json.JSONDecodeError as e:
        st.session_state['json_format'] = False
        st.session_state['_json_format_error'] = str(e)

        # ── Show error with surrounding context ───────────────────────────────
        _lines = active_format.splitlines()
        _el = e.lineno - 1          # 0-indexed line
        _ec = e.colno               # 1-indexed column
        _ctx_start = max(0, _el - 4)
        _ctx_end   = min(len(_lines), _el + 5)
        _ctx = []
        for _i in range(_ctx_start, _ctx_end):
            _pfx = f"{_i + 1:5d} | "
            if _i == _el:
                _ctx.append(f">>> {_pfx}{_lines[_i]}")
                _ctx.append(f"    {' ' * len(_pfx)}{'-' * max(1, _ec - 1)}^ {e.msg}")
            else:
                _ctx.append(f"    {_pfx}{_lines[_i]}")

        st.error(
            f"**Invalid JSON** — line {e.lineno}, column {e.colno}: `{e.msg}`\n\n"
            f"Fix it in the **Preview / Edit** expander above, then re-run.\n\n"
            f"```\n{chr(10).join(_ctx)}\n```"
        )
        st.stop()

    # ── Provider / Model / API key ─────────────────────────────────────────────
    st.divider()
    col_prov, col_model = st.columns(2)
    with col_prov:
        provider = st.radio("Provider", ["OpenAI", "DeepSeek"], horizontal=True)
    with col_model:
        model = st.selectbox("Model",
                             OPENAI_MODELS if provider == "OpenAI" else DEEPSEEK_MODELS,
                             index=0)

    if provider == "OpenAI":
        API_KEY = st.text_input("OpenAI API Key", OPENAI_KEY, type="password")
        ds_key  = None
    else:
        API_KEY = None
        ds_key  = st.text_input("DeepSeek API Key", DEEPSEEK_KEY, type="password")

    # ── Column selector ────────────────────────────────────────────────────────
    cols_unparsed = st.multiselect(
        "Select columns containing raw text (concatenated)",
        data.columns,
        default=[data.columns[0]],
    )
    if not cols_unparsed:
        st.warning("Select at least one column.")
        st.stop()

    def _build_text(row):
        parts = [
            str(row[c]).strip()
            for c in cols_unparsed
            if pd.notna(row[c]) and str(row[c]).strip()
        ]
        return " - ".join(parts) if parts else None

    # ── Carry-through columns ──────────────────────────────────────────────────
    cols_keep = st.multiselect(
        "Columns to keep from the source dataset",
        list(data.columns),
        help="Carried straight through and appended to the parsed output — no "
             "manual merge afterwards. Rows are aligned on the raw-text input "
             "(the columns selected above).",
    )
    if cols_keep:
        _keep_texts = filtered_data.apply(_build_text, axis=1)
        _keep_src = filtered_data[cols_keep].copy()
        _keep_src.index = pd.Index(_keep_texts.values)
        _keep_src = _keep_src[_keep_src.index.notna()]
        _keep_src = _keep_src[~_keep_src.index.duplicated(keep="first")]
        st.session_state['source_keep_df'] = _keep_src
        st.caption(
            f"{len(cols_keep)} source column(s) will be appended to the parsed output."
        )
    else:
        st.session_state['source_keep_df'] = None

    # ── Execution mode ─────────────────────────────────────────────────────────
    st.divider()
    with st.expander("⚡ Execution Mode", expanded=True):
        exec_mode = st.radio(
            "Mode",
            ["🔄 Client parallel (live results)", "📦 Server Batch — OpenAI only (async, 50% off)"],
            horizontal=True,
        )

        if exec_mode.startswith("🔄"):
            use_batch   = False
            max_workers = st.slider(
                "Max parallel workers",
                min_value=1, max_value=64, value=10, step=1,
                help=(
                    "Number of concurrent HTTP requests sent to the API at the same time. "
                    "Higher = faster but you may hit the provider's rate limits. "
                    "OpenAI Tier 1 typically supports ~60 RPM for gpt-4o-mini. "
                    "DeepSeek has dynamic limits; start at 5-10 and raise if no 429 errors."
                ),
            )
            ckpt_path = st.text_input(
                "Checkpoint file (optional)",
                value=st.session_state.get("ckpt_path_live", ""),
                placeholder="e.g. checkpoints/my_run.json",
                help="If set, completed responses are saved immediately. Resume an interrupted run by pointing to the same file.",
                key="ckpt_path_live",
            )
            st.caption(
                "Requests are processed in a thread-pool while this browser tab is open. "
                "A progress bar is shown live. If you close the tab mid-run, completed "
                "responses in the checkpoint file are preserved and will be skipped on the next run."
            )

        else:  # Server Batch
            use_batch   = True
            max_workers = 1  # not used

            if provider != "OpenAI":
                st.warning("Server Batch requires OpenAI. Switch provider or choose Client parallel.")
                use_batch = False

            import uuid as _uuid
            if "server_batch_ckpt" not in st.session_state:
                st.session_state["server_batch_ckpt"] = (
                    f"checkpoints/batch_{_uuid.uuid4().hex[:10]}.json"
                )

            ckpt_path = st.text_input(
                "Checkpoint file path",
                value=st.session_state["server_batch_ckpt"],
                help="The batch ID is stored here. Save this path — you will need it to fetch results after reconnecting.",
                key="server_batch_ckpt_input",
            )
            st.session_state["server_batch_ckpt"] = ckpt_path

            st.info(
                "**How server batch works**\n\n"
                "1. **Submit** — all requests are bundled into a JSONL file, uploaded to "
                "OpenAI, and a batch job is created. The **batch ID** is written to the "
                "checkpoint file on this server. Returns immediately — no waiting.\n\n"
                "2. **Disconnect** — OpenAI processes everything on their side. "
                "You can close this tab. The job runs for up to 24 h.\n\n"
                "3. **Reconnect** — open this page, scroll here, paste the same "
                "checkpoint path, and click **Fetch / Check Results**. "
                "The app polls OpenAI once and writes any completed responses into "
                "the checkpoint file.\n\n"
                "4. **Multiple users** — every user gets a unique checkpoint filename "
                "(UUID-based). Checkpoint files are independent: user A's results never "
                "mix with user B's. If you want a colleague to fetch your results, "
                "share your checkpoint path with them — they enter it and click Fetch."
            )

            # ── Fetch results panel (shown even without a fresh file upload) ──
            st.divider()
            st.markdown("**Fetch results from a previous batch**")
            fetch_ckpt = st.text_input(
                "Checkpoint path to check",
                value=ckpt_path,
                placeholder="checkpoints/batch_abc123def.json",
                key="fetch_ckpt_input",
            )
            fetch_api_key = st.text_input(
                "OpenAI API Key (for fetching)", OPENAI_KEY, type="password",
                key="fetch_api_key",
            )
            if st.button("Fetch / Check Results"):
                if not fetch_api_key:
                    st.warning("Enter your OpenAI API key to check batch status.")
                elif not fetch_ckpt:
                    st.warning("Enter the checkpoint path from your original submission.")
                else:
                    _fetcher = CachedUAPParser(
                        api_key=fetch_api_key,
                        model=model,
                        checkpoint_path=fetch_ckpt,
                    )
                    result = _fetcher.fetch_batch_results()
                    if result["status"] == "no_batch":
                        st.info(f"No active batch found in `{fetch_ckpt}`.")
                    elif result["status"] == "in_progress":
                        st.warning("Batch still in progress.")
                        for b in result["batches"]:
                            prog = b["completed"] / max(b["total"], 1)
                            st.progress(prog, text=(
                                f"{b['batch_id']}: {b['status']} — "
                                f"{b['completed']}/{b['total']} done, "
                                f"{b['failed']} failed"
                            ))
                        st.caption("Come back later and click Fetch again.")
                    else:  # complete
                        st.success(
                            f"Batch complete! {result['n_saved']} new responses collected."
                        )
                        st.session_state['parsed_responses'] = _fetcher.parse_responses()
                        responses_df = _fetcher.responses_to_df(
                            _extract_key, st.session_state['parsed_responses']
                        )
                        responses_df = attach_kept_columns(responses_df)
                        if not responses_df.empty:
                            st.dataframe(responses_df)
                            st.session_state['parsed_responses_df'] = responses_df.copy()

            # ── Import a manually-downloaded batch output (.jsonl) ────────────
            # The OpenAI dashboard serves batch results as .jsonl; import it here
            # to get the exact same session shape as a live parallel parse
            # (strict prune + schema-complete applied when a schema is selected).
            st.markdown("**Or import a downloaded batch output file**")
            _batch_up = st.file_uploader(
                "OpenAI Batch output (.jsonl / .json) downloaded from the dashboard",
                type=["jsonl", "json"], key="batch_jsonl_upload",
            )
            if _batch_up is not None and st.button("Import batch output", key="batch_jsonl_go"):
                from uap_analyzer import batch_jsonl_to_parsed
                try:
                    _fmt = active_format
                except NameError:
                    _fmt = None
                _descs = st.session_state.get("result")
                _parsed_b, _errs_b = batch_jsonl_to_parsed(
                    _batch_up.getvalue().decode("utf-8", errors="replace"),
                    format_long=_fmt,
                    descriptions=_descs if isinstance(_descs, list) else None,
                )
                if not _parsed_b:
                    st.error(
                        "No parseable responses found — is this a Batch API output "
                        f".jsonl? {('First error: ' + _errs_b[0]) if _errs_b else ''}"
                    )
                else:
                    st.session_state['parsed_responses'] = _parsed_b
                    responses_df = pd.json_normalize(list(_parsed_b.values()))
                    responses_df.index = pd.Index(list(_parsed_b.keys()))
                    responses_df = attach_kept_columns(responses_df)
                    st.session_state['parsed_responses_df'] = responses_df.copy()
                    st.success(
                        f"Imported {len(_parsed_b)} parsed response(s)"
                        + (f" — {len(_errs_b)} line(s) failed" if _errs_b else "")
                        + (". Keys re-mapped to the source texts."
                           if isinstance(_descs, list) else
                           ". Keyed by batch custom_id (load the source dataset "
                           "first to re-key by input text and enable carry-through).")
                    )
                    if _errs_b:
                        with st.expander("Import errors (first 10)"):
                            for _e in _errs_b[:10]:
                                st.code(_e)
                    st.dataframe(responses_df.head(50))

    # ── Cost estimate ──────────────────────────────────────────────────────────
    descriptions_preview = filtered_data.apply(_build_text, axis=1).dropna().tolist()
    if descriptions_preview:
        try:
            est = estimate_cost(
                descriptions_preview,
                active_format,
                model=model,
                use_cache=True,
                use_batch=use_batch,
            )
            with st.expander("💰 Estimated Cost", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total", f"${est['total_usd']:.4f}")
                c2.metric("Queries", f"{est['n_queries']:,}")
                c3.metric("Input tokens", f"{est['total_input_tokens']:,}")
                c4.metric("Output tokens", f"{est['output_tokens']:,}")
                c1b, c2b, c3b, c4b = st.columns(4)
                c1b.metric("Cached input", f"${est['cost_cached_usd']:.4f}")
                c2b.metric("Uncached input", f"${est['cost_input_usd']:.4f}")
                c3b.metric("Output cost", f"${est['cost_output_usd']:.4f}")
                c4b.metric("Batch discount", f"{est['batch_discount_pct']*100:.0f}%")
                st.caption(
                    f"Model: **{est['model']}** | Provider: **{est['provider']}** | "
                    f"Avg tokens/query: **{est['avg_desc_tokens']}** | "
                    f"Prefix (cached): **{est['prefix_tokens']}**"
                )
        except Exception as e:
            st.warning(f"Could not estimate cost: {e}")

    # ── Parse / Submit button ──────────────────────────────────────────────────
    parse_label = "Submit Batch" if use_batch else "Parse Dataset"
    if use_batch:
        st.info("📦 **Server Batch mode** — clicking Submit will enqueue the job with OpenAI and return immediately. No results until you fetch later.")
    else:
        st.info(f"🔄 **Client parallel mode** — {len(filtered_data.apply(_build_text, axis=1).dropna())} descriptions will be sent concurrently to **{provider} / {model}** while this tab stays open.")
    if st.button(parse_label) and st.session_state['json_format']:
        active_api_key = API_KEY if provider == "OpenAI" else ds_key
        if not active_api_key:
            st.warning("Please enter your API key to proceed.")
            st.stop()

        selected_column_data = filtered_data.apply(_build_text, axis=1).tolist()
        st.session_state.result = selected_column_data

        with st.status(
            "Submitting batch…" if use_batch else "Parsing…", expanded=True
        ) as stat:
            try:
                _parser = CachedUAPParser(
                    api_key=active_api_key,
                    model=model,
                    use_batch=False,   # process_descriptions handles threading; batch via submit_batch_only
                    col=st.session_state.result,
                    checkpoint_path=ckpt_path if ckpt_path else None,
                )

                if use_batch:
                    # Non-blocking: submit only, return immediately
                    saved_ckpt = _parser.submit_batch_only(
                        st.session_state.result, active_format
                    )
                    st.success(
                        f"Batch submitted to OpenAI. **Checkpoint file:** `{saved_ckpt}`\n\n"
                        "Save this path. Return to this page within 24 h, paste the path "
                        "into **Fetch / Check Results** above, and click the button to "
                        "collect your results."
                    )
                    stat.update(label="Batch submitted", state="complete", expanded=False)

                else:
                    n_desc = len(st.session_state.result)
                    st.write(f"Sending **{n_desc}** descriptions to **{provider} / {model}**…")
                    if n_desc == 0:
                        st.error("No descriptions found in the selected column. Check your column selection.")
                        st.stop()

                    _prog_bar  = st.progress(0.0)
                    _prog_text = st.empty()

                    def _on_progress(n_ok, n_fail, total):
                        done = n_ok + n_fail
                        pct  = done / total if total else 1.0
                        _prog_bar.progress(min(pct, 1.0))
                        _prog_text.caption(
                            f"✓ {n_ok} ok · ✗ {n_fail} failed · "
                            f"{done}/{total} ({pct*100:.1f}%)"
                        )

                    _parser.process_descriptions(
                        st.session_state.result, active_format,
                        max_workers=max_workers,
                        progress_callback=_on_progress,
                    )
                    _prog_bar.progress(1.0)
                    _prog_text.empty()

                    # Surface any API errors before attempting to build the DataFrame
                    if _parser.last_errors:
                        st.warning(
                            f"{len(_parser.last_errors)} of {n_desc} API call(s) failed."
                        )
                        with st.expander("Error details (first 10)"):
                            for err in _parser.last_errors[:10]:
                                st.code(err)

                    if not _parser.responses:
                        st.error(
                            "No responses were stored. Likely causes: invalid API key, "
                            "wrong model name, or network error. See error details above."
                        )
                        st.stop()

                    st.session_state['parsed_responses'] = _parser.parse_responses()
                    responses_df = _parser.responses_to_df(
                        _extract_key, st.session_state['parsed_responses']
                    )
                    responses_df = attach_kept_columns(responses_df)
                    if not responses_df.empty:
                        st.dataframe(responses_df)
                        st.session_state['parsed_responses_df'] = responses_df.copy()
                        stat.update(label="Parsing complete", state="complete", expanded=False)
                    else:
                        st.error("Failed to create DataFrame from parsed responses.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # ── Download / export ──────────────────────────────────────────────────────
    if _parsed_responses_dict() is not None:
        json_str = download_json(st.session_state['parsed_responses'])
        st.download_button(
            label="Download Parsed Data as JSON",
            data=json_str,
            file_name="parsed_responses.json",
            mime="application/json",
        )
        if st.button("Convert Cached Data to DataFrame"):
            convert_cached_data_to_df(st.session_state['parsed_responses'])

    if st.session_state['parsed_responses_df'] is not None:
        st.download_button(
            label="Save CSV",
            data=convert_df(st.session_state['parsed_responses_df']),
            file_name="uap_data.csv",
            mime="text/csv",
        )

    # ── Export to SCU xlsx ─────────────────────────────────────────────────────
    if "SCU Spreadsheet" in fmt_choices:
        st.divider()
        st.markdown("**Export to SCU Spreadsheet**")

        # Allow loading a previously saved parsed_responses.json when there is
        # no live session (e.g. user reloaded the page after parsing).
        if _parsed_responses_dict() is None:
            st.info(
                "No parsed data in session. Upload a previously saved "
                "`parsed_responses.json` **or** an SCU-format `.csv` to continue."
            )
            col_restore1, col_restore2 = st.columns(2)
            json_upload = col_restore1.file_uploader(
                "Load parsed_responses.json",
                type=["json"],
                key="scu_json_restore",
            )
            csv_upload = col_restore2.file_uploader(
                "Load SCU CSV (group.field columns)",
                type=["csv"],
                key="scu_csv_restore",
            )
            if json_upload is not None:
                try:
                    st.session_state['parsed_responses'] = json.loads(json_upload.read())
                    st.success(
                        f"Loaded {len(st.session_state['parsed_responses'])} records from JSON."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load JSON: {e}")
            if csv_upload is not None:
                st.session_state['_scu_csv_df'] = pd.read_csv(csv_upload)
                st.success(f"Loaded CSV: {len(st.session_state['_scu_csv_df'])} rows.")

        if _parsed_responses_dict() is not None or '_scu_csv_df' in st.session_state:
            col_t, col_n = st.columns(2)
            xlsx_template = col_t.file_uploader(
                "Upload template spreadsheet (.xlsx)",
                type=["xlsx"],
                help="Upload 'UAP Activities Data 1975 Onwards.xlsx'. "
                     "The output will be a filled copy — your original is never modified.",
                key="scu_template",
            )
            start_num = col_n.number_input(
                "Starting Internal Number (0 = leave blank)", min_value=0, value=0, step=1
            )
            if st.button("Generate SCU xlsx"):
                if xlsx_template is None:
                    st.warning("Please upload the template .xlsx first.")
                else:
                    try:
                        import tempfile
                        from uap_xlsx_filler import (
                            build_dataframe, build_dataframe_from_csv, append_to_xlsx
                        )

                        # CSV path — no JSON/session needed
                        if '_scu_csv_df' in st.session_state:
                            new_df = build_dataframe_from_csv(st.session_state['_scu_csv_df'])
                        else:
                            _proxy = CachedUAPParser.__new__(CachedUAPParser)
                            _proxy.responses = {}
                            import threading as _t
                            _proxy._responses_lock = _t.Lock()
                            new_df = build_dataframe(_proxy, st.session_state['parsed_responses'])

                        st.write(f"Built DataFrame: {len(new_df)} rows × {len(new_df.columns)} columns")

                        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                            tmp.write(xlsx_template.read())
                            out_path = tmp.name
                        append_to_xlsx(
                            new_df,
                            xlsx_path=out_path,
                            output_path=out_path,
                            start_internal_number=int(start_num) if start_num > 0 else None,
                        )
                        with open(out_path, "rb") as f:
                            xlsx_bytes = f.read()
                        st.download_button(
                            label="Download SCU xlsx",
                            data=xlsx_bytes,
                            file_name="UAP_Activities_filled.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                        st.success(f"{len(new_df)} rows written to spreadsheet.")
                    except Exception as e:
                        st.error(f"xlsx export failed: {e}")


#except Exception as e:
#    stat.update(label=f"Parsing failed: {e}", state="error")
# st.write("Parsing descriptions...")
# st.update_status("Parsing descriptions...")


# ── SCU Normalization (optional — toggle in the app.py sidebar) ───────────────
# When the global "Apply SCU normalization" toggle is on, normalize whatever
# parsed responses are currently in the session (live parse, restored JSON, or
# Markdown ingestion) and expose the SCU v2 normalized CSV + audit report.
if st.session_state.get('scu_normalize_enabled'):
    st.divider()
    if _parsed_responses_dict():
        # Titled section, not an expander: render_scu_normalization opens its own
        # "Normalization audit report" expander, and Streamlit forbids nesting
        # expanders inside expanders.
        st.markdown("#### 🧹 SCU Normalization (SCU v2)")
        st.markdown(
            "Canonicalises the parsed data — country ISO codes, US-state "
            "codes, witness roles, craft shape/size bands — and derives the "
            "**SCU five-criterion eligibility gate** (`scu_eligible` plus the "
            "per-criterion columns). Turn this off in the app sidebar to hide."
        )
        render_scu_normalization(
            st.session_state['parsed_responses'],
            show_advanced_filters=restore_file is not None,
        )
    else:
        st.caption(
            "🧹 SCU Normalization is enabled — parse a dataset or load a "
            "`parsed_responses.json`, and the normalized output will appear here."
        )

