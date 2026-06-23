import streamlit as st
#import cudf.pandas
#cudf.pandas.install()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from uap_analyzer import UAPParser, UAPAnalyzer, UAPVisualizer, cramers_v
# import ChartGen
# from ChartGen import ChartGPT
from Levenshtein import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency
import plotly.graph_objects as go
from tqdm import tqdm
import streamlit.components.v1 as components
from dateutil import parser
from sentence_transformers import SentenceTransformer
import torch
import squarify
import matplotlib.colors as mcolors
import textwrap
import datamapplot

# Import enhanced utilities
from utils.data_processing import DataProcessor
from utils.visualization import UAP_Visualizer as Enhanced_Visualizer
from utils.session_manager import SessionStateManager

# st.set_option('deprecation.showPyplotGlobalUse', False)

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)



def load_data(file_path, key='df'):
    return pd.read_hdf(file_path, key=key)


# Gemini Q&A moved to rag_search.py (the RAG search page) — see gemini_query there.

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
        # Iterate over text elements and rectangles (patches) in the axes for color adjustment
        for text, rect in zip(ax.texts, ax.patches):
            background_color = rect.get_facecolor()
            r, g, b, _ = mcolors.to_rgba(background_color)
            brightness = np.average([r, g, b])
            text.set_color('white' if brightness < 0.5 else 'black')

            # Adjust font size based on rectangle's area and wrap long text
            coef = 0.8
            font_size = np.sqrt(rect.get_width() * rect.get_height()) * coef
            text.set_fontsize(font_size)
            wrapped_text = textwrap.fill(text.get_text(), width=20)
            text.set_text(wrapped_text)

        plt.axis('off')
        plt.gca().invert_yaxis()
        plt.gcf().set_size_inches(20, 12)

        fig.patch.set_alpha(0)

        ax.patch.set_alpha(0)
        return fig

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


def plot_bar(df, x_column, y_column, figsize=(12, 10), color='orange', title=None, rotation=45):
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(data=df, x=x_column, y=y_column, color=color, ax=ax)

    ax.set_title(title if title else f'{y_column} by {x_column}', color=color, fontweight='bold')
    ax.set_xlabel(x_column, color=color)
    ax.set_ylabel(y_column, color=color)

    ax.tick_params(axis='x', colors=color)
    ax.tick_params(axis='y', colors=color)

    plt.xticks(rotation=rotation)

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


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced filtering interface using the improved DataProcessor
    """
    return DataProcessor.filter_dataframe_enhanced(df, enable_quick_filters=False, enable_advanced_filters=True)

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
    try:
        to_filter_columns = st.multiselect("Filter dataframe on", df_.columns)
    except:
        try:
            to_filter_columns = st.multiselect("Filter dataframe", df_.columns)
        except:
            try:
                to_filter_columns = st.multiselect("Filter the dataframe on", df_.columns)
            except:
                pass

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
                df_[column] = pd.to_datetime(df_[column], errors='coerce')
            except Exception:
                try:
                    df_[column] = df_[column].apply(lambda x: parser.parse(str(x)) if pd.notna(x) else pd.NaT)
                except Exception:
                    pass

            if is_datetime64_any_dtype(df_[column]):
                try:
                    df_[column] = df_[column].dt.tz_localize(None)
                except TypeError:
                    df_[column] = df_[column].dt.tz_convert(None)
                _valid = df_[column].dropna()
                if not _valid.empty:
                    min_date = _valid.min().date()
                    max_date = _valid.max().date()
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

def merge_clusters(df, column):
    cluster_terms_ = df.__dict__['cluster_terms']
    cluster_labels_ = df.__dict__['cluster_labels']
    label_name_map = {label: cluster_terms_[label] for label in set(cluster_labels_)}
    merge_map = {}
    # Iterate over term pairs and decide on merging based on the distance
    for idx, term1 in enumerate(cluster_terms_):
        for jdx, term2 in enumerate(cluster_terms_):
            if idx < jdx and distance(term1, term2) <= 3:  # Adjust threshold as needed
                # Decide to merge labels corresponding to jdx into labels corresponding to idx
                # Find labels corresponding to jdx and idx
                labels_to_merge = [label for label, term_index in enumerate(cluster_labels_) if term_index == jdx]
                for label in labels_to_merge:
                    merge_map[label] = idx  # Map the label to use the term index of term1

    # Update the analyzer with the merged numeric labels 
    updated_cluster_labels_ = [merge_map[label] if label in merge_map else label for label in cluster_labels_]

    df.__dict__['cluster_labels'] = updated_cluster_labels_
    # Optional: Update string labels to reflect merged labels
    updated_string_labels = [cluster_terms_[label] for label in updated_cluster_labels_]
    df.__dict__['string_labels'] = updated_string_labels
    return updated_string_labels

# ── Categorical Association Explorer (Cramér's V) ───────────────────────────
# Bands columns by unique-value count so genuinely categorical columns are
# auto-selected for a fast first-pass Cramér's V, while high-cardinality
# columns (free-text, IDs, numerics) are excluded by default.

def _safe_nunique(series):
    """nunique that tolerates unhashable cells (lists/dicts) by stringifying."""
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.astype(str).nunique(dropna=True))


def band_columns(df, high_threshold=30):
    """Bucket columns into categorical bands by cardinality.

    Returns (bands, nunique_map):
      bands = {"binary": [...], "low": [...], "medium": [...], "high": [...],
               "constant": [...]}
        - binary    : exactly 2 distinct values
        - low       : 3 .. 9
        - medium    : 10 .. high_threshold-1
        - high       : >= high_threshold (treated as non-categorical, excluded)
        - constant  : <= 1 distinct value (useless for association)
    """
    bands = {"binary": [], "low": [], "medium": [], "high": [], "constant": []}
    nunique_map = {}
    for c in df.columns:
        nu = _safe_nunique(df[c])
        nunique_map[c] = nu
        if nu <= 1:
            bands["constant"].append(c)
        elif nu == 2:
            bands["binary"].append(c)
        elif nu <= 9:
            bands["low"].append(c)
        elif nu < high_threshold:
            bands["medium"].append(c)
        else:
            bands["high"].append(c)
    return bands, nunique_map


# Single canonical label for every flavour of "absent" so missingness isn't
# fragmented into nan / None / <NA> / NaT / "" as separate categories.
_MISSING_LABEL = "(missing)"
_NULL_STR_TOKENS = {"nan", "none", "null", "<na>", "nat", ""}


def _coalesce(series):
    """Stringify a column and fold all null representations into one
    `(missing)` category (fixes fragmentation). Never mutates the source."""
    s = series.astype(str).str.strip()
    return s.mask(s.str.lower().isin(_NULL_STR_TOKENS), _MISSING_LABEL)


def _pair_series(df, c1, c2, drop_missing):
    """Coalesced (and optionally complete-case) aligned pair of columns."""
    a, b = _coalesce(df[c1]), _coalesce(df[c2])
    if drop_missing:
        keep = (a != _MISSING_LABEL) & (b != _MISSING_LABEL)
        a, b = a[keep], b[keep]
    return a, b


def compute_cramers_v_df(df, cols, drop_missing=False):
    """Symmetric Cramér's V matrix over `cols` (diagonal = 1.0). Each cell is
    computed once and mirrored. Nulls are coalesced to a single `(missing)`
    category; when `drop_missing` is set, each pair is reduced to its
    complete cases (note: per-cell N then varies across the matrix)."""
    cv = pd.DataFrame(index=cols, columns=cols, data=np.nan, dtype=float)
    cache = {c: _coalesce(df[c]) for c in cols}
    for i, c1 in enumerate(cols):
        cv.at[c1, c1] = 1.0
        for c2 in cols[i + 1:]:
            a, b = cache[c1], cache[c2]
            if drop_missing:
                keep = (a != _MISSING_LABEL) & (b != _MISSING_LABEL)
                a, b = a[keep], b[keep]
            v = 0.0 if len(a) == 0 else cramers_v(pd.crosstab(a, b))
            cv.at[c1, c2] = v
            cv.at[c2, c1] = v
    return cv


_CV_TOL = 1e-6  # treat values within this of 0 / 1 as trivial


def _is_trivial_v(v, tol=_CV_TOL):
    """True for a Cramér's V that's effectively 0 (no association — likely
    null/constant) or 1 (perfect association — likely a duplicate column)."""
    return (v <= tol) or (v >= 1.0 - tol)


def high_correlation_columns(cv_df, strong_threshold=0.30, exclude_trivial=True):
    """Columns reaching Cramér's V ≥ strong_threshold with ≥1 other column.

    Mirrors the pass-2 pre-fill logic in render_cramers_v_explorer so the
    analysis form can reuse whatever the explorer last computed. Trivial pairs
    (V≈0 null/constant, V≈1 duplicate) are skipped when exclude_trivial is set.
    Returns [] if cv_df is None or empty.
    """
    if cv_df is None or getattr(cv_df, "empty", True):
        return []
    out = []
    for col in cv_df.columns:
        others = cv_df[col].drop(labels=[col], errors="ignore")
        for v in others:
            if pd.isna(v):
                continue
            v = float(v)
            if exclude_trivial and _is_trivial_v(v):
                continue
            if v >= strong_threshold:
                out.append(col)
                break
    return out


def _cramers_pairs_table(cv_df, exclude_trivial=True):
    """Ranked long-form table of unique off-diagonal pairs, strongest first.

    When `exclude_trivial` is set, pairs with Cramér's V ≈ 0 (no association,
    likely null/constant) or ≈ 1 (perfect association, likely duplicate
    columns) are dropped. Returns (table, n_excluded)."""
    rows, n_excluded = [], 0
    cols = list(cv_df.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            v = cv_df.at[c1, c2]
            if pd.isna(v):
                continue
            v = float(v)
            if exclude_trivial and _is_trivial_v(v):
                n_excluded += 1
                continue
            rows.append({"Variable A": c1, "Variable B": c2,
                         "Cramér's V": round(v, 3)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Cramér's V", ascending=False).reset_index(drop=True)
    return out, n_excluded


def _interactive_cv_heatmap(cv_df, title, key):
    """Clickable lower-triangle Cramér's V heatmap. Returns the (row, col)
    column names of the most recently clicked cell, or None.

    Clicking a cell drives the contingency drill-down below the chart.
    """
    cols = list(cv_df.columns)
    n = len(cols)
    z = cv_df.astype(float).values.copy()
    # Mask the strict upper triangle so each pair shows once (lower triangle).
    z[np.triu(np.ones_like(z, dtype=bool), k=1)] = np.nan
    annotate = n <= 25
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=cols, colorscale="RdBu_r", zmin=0, zmax=1,
        colorbar=dict(title="Cramér's V"),
        hovertemplate="A (row): %{y}<br>B (col): %{x}<br>V: %{z:.3f}"
                      "<extra></extra>",
        texttemplate="%{z:.2f}" if annotate else None,
        textfont={"size": 8},
    ))
    fig.update_layout(
        title=f"{title} — click a cell (or pick a pair below) to inspect categories",
        height=max(450, 24 * n + 160),
        yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    try:
        event = st.plotly_chart(fig, use_container_width=True,
                                on_select="rerun", key=key)
    except TypeError:
        # Older Streamlit without on_select — render non-interactive.
        st.plotly_chart(fig, use_container_width=True, key=key)
        return None
    # Selection payload shape varies across Streamlit versions; dig defensively.
    sel = {}
    if event is not None:
        sel = (event.get("selection") if hasattr(event, "get") else None) or {}
    pts = sel.get("points", []) if hasattr(sel, "get") else []
    if pts:
        p = pts[-1]
        c_row, c_col = p.get("y"), p.get("x")
        if c_row in cv_df.columns and c_col in cv_df.columns and c_row != c_col:
            return c_row, c_col
    return None


def _pair_drilldown(df, cv_df, pairs_df, key_prefix, clicked, drop_missing=False):
    """Render Variable-A / Variable-B selectors (defaulting to the strongest
    pair, or to a clicked heatmap cell) and the contingency drill-down. Works
    even when heatmap click events don't fire — the selectors are the reliable
    path; a click just pre-sets them."""
    cols = list(cv_df.columns)
    if len(cols) < 2:
        return
    ka, kb = f"{key_prefix}_a", f"{key_prefix}_b"

    # Strongest pair from the (already filtered) ranked table → sensible default.
    if pairs_df is not None and not pairs_df.empty:
        default_a = pairs_df.iloc[0]["Variable A"]
        default_b = pairs_df.iloc[0]["Variable B"]
    else:
        default_a, default_b = cols[0], cols[1]

    # Seed defaults once; reset if a stale value isn't in the current matrix.
    if ka not in st.session_state or st.session_state[ka] not in cols:
        st.session_state[ka] = default_a
    if kb not in st.session_state or st.session_state[kb] not in cols:
        st.session_state[kb] = default_b
    # A heatmap click overrides the current selection.
    if clicked:
        st.session_state[ka], st.session_state[kb] = clicked[0], clicked[1]

    cA, cB = st.columns(2)
    with cA:
        a = st.selectbox("Variable A (row)", cols, key=ka)
    with cB:
        b = st.selectbox("Variable B (col)", cols, key=kb)
    if a == b:
        st.info("Pick two different columns to see their co-occurring categories.")
        return
    _render_contingency(df, a, b, float(cv_df.at[a, b]), key_prefix, drop_missing)


def _render_contingency(df, c1, c2, v, key_prefix, drop_missing=False):
    """Show which category VALUES co-vary for the chosen column pair: a
    crosstab the user can view as counts, row/column %, or standardized
    residuals (the cells that drive the association). Nulls are coalesced to a
    single `(missing)` category, or dropped (complete cases) when requested."""
    st.markdown(f"#### 🔬 `{c1}`  ✕  `{c2}` — Cramér's V = **{v:.3f}**")
    a, b = _pair_series(df, c1, c2, drop_missing)
    n_used = len(a)
    n_total = len(df)
    if drop_missing:
        st.caption(f"Complete cases: **{n_used:,}** of {n_total:,} rows "
                   f"({n_used / n_total * 100:.1f}%) — rows missing either "
                   "column dropped.")
    else:
        st.caption(f"All **{n_total:,}** rows; nulls folded into one "
                   "`(missing)` category.")
    if n_used == 0:
        st.info("No rows left for this pair after dropping missing values.")
        return
    ct = pd.crosstab(a, b)
    if ct.size == 0:
        st.info("Empty crosstab for this pair.")
        return

    # Cap very large crosstabs to the most frequent values per axis.
    MAXC = 30
    note = ""
    r, k = ct.shape
    if r > MAXC or k > MAXC:
        top_r = a.value_counts().nlargest(MAXC).index
        top_k = b.value_counts().nlargest(MAXC).index
        ct = ct.loc[ct.index.isin(top_r), ct.columns.isin(top_k)]
        note = f" · showing top {MAXC} values per axis"

    # χ²-expected-count health check (the residual approximation wants ≥5).
    if min(ct.shape) >= 2:
        try:
            _, _, _, _exp = chi2_contingency(ct)
            _low = float((_exp < 5).mean()) * 100
            if _low > 20:
                st.warning(
                    f"⚠️ {_low:.0f}% of cells have an expected count < 5 — the "
                    "χ² / standardized-residual approximation is unstable here. "
                    "Treat residuals as indicative, not exact."
                )
        except ValueError:
            pass

    view = st.radio(
        "Cell values", ["Counts", "Row %", "Column %", "Std. residuals"],
        horizontal=True, key=f"{key_prefix}_ctview",
        help="Std. residual = (observed − expected) / √expected. |residual| > 2 "
             "marks a value-pair that co-occurs far more (blue) or less (red) "
             "than chance — these are what make the two columns covariable.",
    )

    if view == "Counts":
        mat, fmt, cs, zmid = ct, ".0f", "Blues", None
    elif view == "Row %":
        mat = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100
        fmt, cs, zmid = ".0f", "Blues", None
    elif view == "Column %":
        mat = ct.div(ct.sum(axis=0).replace(0, np.nan), axis=1) * 100
        fmt, cs, zmid = ".0f", "Blues", None
    else:  # Std. residuals
        chi2, p, dof, expected = chi2_contingency(ct)
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = (ct.values - expected) / np.sqrt(expected)
        mat = pd.DataFrame(np.nan_to_num(resid), index=ct.index, columns=ct.columns)
        fmt, cs, zmid = ".1f", "RdBu", 0

    rows, kcols = mat.shape
    show_text = rows <= 25 and kcols <= 25
    fig = go.Figure(go.Heatmap(
        z=mat.values,
        x=[str(c) for c in mat.columns], y=[str(i) for i in mat.index],
        colorscale=cs, zmid=zmid,
        texttemplate=f"%{{z:{fmt}}}" if show_text else None,
        textfont={"size": 8},
        hovertemplate=f"{c1}=%{{y}}<br>{c2}=%{{x}}<br>%{{z:{fmt}}}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{c1} ✕ {c2} — {view}{note}",
        height=max(380, 22 * rows + 160),
        xaxis_title=c2, yaxis_title=c1,
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_ctfig")

    if view == "Std. residuals":
        # Surface the strongest covariable value-pairs as a ranked table.
        rl = [
            {f"{c1}": i, f"{c2}": j, "std_residual": round(float(mat.at[i, j]), 2),
             "count": int(ct.at[i, j])}
            for i in mat.index for j in mat.columns
            if abs(mat.at[i, j]) >= 2
        ]
        if rl:
            rl_df = pd.DataFrame(rl)
            rl_df = (rl_df.assign(_abs=rl_df["std_residual"].abs())
                          .sort_values("_abs", ascending=False)
                          .drop(columns="_abs")
                          .reset_index(drop=True))
            st.caption("Value-pairs that co-occur more/less than chance "
                       "(|std. residual| ≥ 2):")
            st.dataframe(rl_df.head(30), hide_index=True, use_container_width=True)
        else:
            st.caption("No value-pair exceeds |std. residual| ≥ 2 — the "
                       "association is spread evenly rather than driven by "
                       "specific value combinations.")


def render_cramers_v_explorer(df):
    """Cardinality-banded, two-pass Cramér's V association explorer.

    Pass 1 auto-runs over every column the cardinality heuristic flags as
    categorical (binary + low + medium bands). Pass 2 lets the user refine the
    set — pre-filled with whichever columns carried a strong association.
    """
    st.markdown("### 🎲 Categorical Association Explorer (Cramér's V)")
    st.caption(
        "Columns are banded by their number of distinct values. Genuinely "
        "categorical columns are auto-selected for a first-pass association "
        "heatmap; high-cardinality columns (free-text, IDs, numerics) are "
        "treated as non-categorical and excluded by default."
    )

    cc1, cc2 = st.columns(2)
    with cc1:
        high_threshold = int(st.number_input(
            "Non-categorical threshold (distinct values)",
            min_value=3, max_value=1000, value=30, step=1,
            key="cv_high_threshold",
            help="Columns with this many or more distinct values are treated "
                 "as non-categorical and land in the (opt-in) High band.",
        ))
    with cc2:
        strong_threshold = float(st.slider(
            "Strong-association threshold (for pass 2 pre-fill)",
            0.0, 1.0, 0.30, 0.05, key="cv_strong_threshold",
            help="In pass 2, columns are pre-selected if they reach at least "
                 "this Cramér's V with any other column in pass 1.",
        ))

    fc1, fc2 = st.columns([2, 3])
    with fc1:
        exclude_trivial = st.checkbox(
            "Filter out V≈0 and V≈1 pairs (likely null / duplicate)",
            value=True, key="cv_exclude_trivial",
            help="Drops pairs with no association (V≈0 — often constant/null "
                 "columns) and perfect association (V≈1 — often duplicate "
                 "columns) from the ranked tables and the pass-2 pre-fill.",
        )
    with fc2:
        missing_mode = st.radio(
            "Missing values",
            ["Treat as “(missing)” category", "Drop missing (complete cases)"],
            horizontal=True, key="cv_missing_mode",
            help="Coalesce: all nulls (nan/None/<NA>) become one “(missing)” "
                 "level — keeps every row, lets you study co-missingness. "
                 "Drop: each pair uses only rows where both are present "
                 "(unbiased only if data is missing-completely-at-random).",
        )
    drop_missing = missing_mode.startswith("Drop")

    bands, nunique_map = band_columns(df, high_threshold=high_threshold)

    band_meta = [
        ("binary", "Binary (2 values)", True),
        ("low", "Low (3–9 values)", True),
        ("medium", f"Medium (10–{high_threshold - 1} values)", True),
        ("high", f"High (≥{high_threshold} — non-categorical)", False),
    ]

    selected = []
    for key, label, prefill in band_meta:
        opts = bands[key]
        if not opts:
            continue
        # Sort options by cardinality so the most-categorical appear first.
        opts = sorted(opts, key=lambda c: nunique_map[c])
        default = opts if prefill else []
        picked = st.multiselect(
            f"{label} — {len(opts)} column(s)",
            options=opts,
            default=default,
            key=f"cv_band_{key}",
            help="Pre-filled and included in pass 1." if prefill
                 else "Not pre-filled — tick to opt in (crosstabs can be large).",
        )
        selected.extend(picked)

    if bands["constant"]:
        st.caption(
            f":grey[Skipped {len(bands['constant'])} constant/empty column(s): "
            f"{', '.join(bands['constant'][:8])}"
            f"{'…' if len(bands['constant']) > 8 else ''}]"
        )

    # De-dup while preserving order.
    selected = list(dict.fromkeys(selected))
    n_sel = len(selected)
    n_pairs = n_sel * (n_sel - 1) // 2
    st.caption(f"**{n_sel}** column(s) selected → **{n_pairs:,}** pairwise "
               "Cramér's V computations.")
    if n_sel > 40:
        st.warning(
            f"{n_sel} columns selected — pass 1 computes {n_pairs:,} crosstabs "
            "and may be slow. Consider trimming the Medium band."
        )

    if st.button("▶️ Run Cramér's V — Pass 1", type="primary",
                 disabled=n_sel < 2, key="cv_run_pass1"):
        with st.spinner(f"Computing Cramér's V over {n_sel} columns…"):
            cv_df = compute_cramers_v_df(df, selected, drop_missing=drop_missing)
        st.session_state["cv_pass1_df"] = cv_df
        st.session_state.pop("cv_pass2_df", None)

    cv1 = st.session_state.get("cv_pass1_df")
    if cv1 is not None:
        st.markdown("#### Pass 1 — auto-selected categoricals")
        clicked1 = _interactive_cv_heatmap(cv1, "Cramér's V — Pass 1", "cv_hm1")
        pairs1, n_excl1 = _cramers_pairs_table(cv1, exclude_trivial=exclude_trivial)
        # Keep only the download button in the main flow; tuck the pair
        # comparison drill-down and the ranked table behind a dropdown.
        if not pairs1.empty:
            st.download_button(
                "⬇️ Download pass-1 pairs (CSV)", pairs1.to_csv(index=False),
                "cramers_v_pass1.csv", "text/csv", key="cv_dl_pass1",
            )
        # Popover, not an expander: this whole explorer is rendered inside the
        # "Categorical Association Explorer" expander, and Streamlit forbids
        # nesting an expander inside another expander.
        with st.popover("🔎 Pair comparison & ranked table", use_container_width=True):
            _pair_drilldown(df, cv1, pairs1, "cv_dd1", clicked1, drop_missing)
            if n_excl1:
                st.caption(f":grey[Filtered {n_excl1} trivial pair(s) "
                           "(V≈0 null/constant or V≈1 duplicate).]")
            if not pairs1.empty:
                st.caption("Strongest associations (pass 1):")
                st.dataframe(pairs1.head(25), hide_index=True, use_container_width=True)
            else:
                st.info("No non-trivial associations found in pass 1.")

        # ── Pass 2 — refine ────────────────────────────────────────────────
        st.divider()
        st.markdown("#### Pass 2 — refine")
        # A column qualifies if it reaches the strong threshold with another
        # column — but trivial values (≈1 duplicates, ≈0) are skipped when the
        # filter is on, so duplicate-driven columns don't auto-fill pass 2.
        def _qualifies(col):
            others = cv1[col].drop(labels=[col])
            for v in others:
                if pd.isna(v):
                    continue
                v = float(v)
                if exclude_trivial and _is_trivial_v(v):
                    continue
                if v >= strong_threshold:
                    return True
            return False
        strong_cols = [c for c in cv1.columns if _qualifies(c)]
        st.caption(
            f"Pre-filled with the **{len(strong_cols)}** column(s) reaching "
            f"Cramér's V ≥ {strong_threshold:.2f} with any other column in "
            "pass 1. Edit the set and re-run for a focused heatmap."
        )
        refine_sel = st.multiselect(
            "Columns for pass 2",
            options=list(cv1.columns),
            default=strong_cols or list(cv1.columns),
            key="cv_refine_sel",
        )
        refine_sel = list(dict.fromkeys(refine_sel))
        if st.button("🔬 Run Cramér's V — Pass 2 (refined)",
                     disabled=len(refine_sel) < 2, key="cv_run_pass2"):
            with st.spinner(f"Re-computing over {len(refine_sel)} columns…"):
                # Sub-select from the pass-1 matrix when possible; recompute
                # only if the user somehow added columns not in pass 1.
                if set(refine_sel).issubset(set(cv1.columns)):
                    cv2 = cv1.loc[refine_sel, refine_sel]
                else:
                    cv2 = compute_cramers_v_df(df, refine_sel, drop_missing=drop_missing)
            st.session_state["cv_pass2_df"] = cv2

        cv2 = st.session_state.get("cv_pass2_df")
        if cv2 is not None:
            clicked2 = _interactive_cv_heatmap(cv2, "Cramér's V — Pass 2 (refined)", "cv_hm2")
            pairs2, n_excl2 = _cramers_pairs_table(cv2, exclude_trivial=exclude_trivial)
            # Same as pass 1: only the download button stays in the main flow.
            if not pairs2.empty:
                st.download_button(
                    "⬇️ Download pass-2 pairs (CSV)", pairs2.to_csv(index=False),
                    "cramers_v_pass2.csv", "text/csv", key="cv_dl_pass2",
                )
            with st.popover("🔎 Pair comparison & ranked table (refined)", use_container_width=True):
                _pair_drilldown(df, cv2, pairs2, "cv_dd2", clicked2, drop_missing)
                if n_excl2:
                    st.caption(f":grey[Filtered {n_excl2} trivial pair(s) "
                               "(V≈0 null/constant or V≈1 duplicate).]")
                if not pairs2.empty:
                    st.dataframe(pairs2.head(40), hide_index=True,
                                 use_container_width=True)
                else:
                    st.info("No non-trivial associations in the refined set.")


def render_categorical_flow_sankey(df):
    """All-with-all categorical flow Sankey for the Statistical Analysis section.

    Unlike the cross-DB matcher in rag_search.py (which pairs records by
    similarity), this links every value of each chosen column to every
    co-occurring value of the next column, with link width = the number of rows
    sharing that combination (a chained crosstab). No pairwise matching — just
    all-with-all co-occurrence counts.
    """
    st.markdown("### 🌊 Categorical Flow (Sankey)")
    st.caption(
        "Pick two or more categorical columns as ordered levels (left → right). "
        "Every value is linked to every co-occurring value of the next level, "
        "with link width = the number of rows sharing that combination — no "
        "pairwise matching, just all-with-all co-occurrence counts."
    )

    n = len(df)
    if n == 0:
        st.info("No rows to chart.")
        return

    def _safe_nunique(s):
        # Columns of dicts/lists are unhashable; fall back to string form.
        try:
            return s.nunique(dropna=False)
        except TypeError:
            return s.astype(str).nunique(dropna=False)

    cat_like = [
        c for c in df.columns
        if 1 < _safe_nunique(df[c]) <= max(50, int(n * 0.5))
    ]
    if len(cat_like) < 2:
        st.info("Need at least two categorical-like columns for a flow diagram.")
        return

    sc1, sc2 = st.columns([3, 1])
    with sc1:
        cols = st.multiselect(
            "Levels (left → right, in order)",
            options=list(df.columns),
            default=cat_like[:3],
            key="sankey_flow_cols",
        )
    with sc2:
        top_n = int(st.number_input(
            "Top-N / level", min_value=2, max_value=50, value=12, step=1,
            key="sankey_flow_topn",
            help="Keep only the most frequent N values per level; the rest are "
                 "lumped into “Other” so the diagram stays readable.",
        ))

    if len(cols) < 2:
        st.info("Select at least two columns.")
        return

    # Normalise: string-cast, fill missing, cap to top-N per level (+ “Other”).
    work = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c].astype(str).where(df[c].notna(), "(missing)")
        keep = s.value_counts().head(top_n).index
        work[c] = s.where(s.isin(keep), "Other")

    # Namespace nodes by (level, value) so identical values in different levels
    # never merge into one node.
    PALETTE = ["#f97316", "#22c55e", "#3b82f6", "#a855f7", "#ec4899",
               "#14b8a6", "#eab308", "#ef4444"]
    index, node_label, node_color = {}, [], []
    L = len(cols)
    max_per_level = 1
    for level, c in enumerate(cols):
        vc = work[c].value_counts()
        max_per_level = max(max_per_level, len(vc))
        for v in vc.index:
            key = (level, v)
            if key not in index:
                index[key] = len(node_label)
                node_label.append(str(v))
                node_color.append(PALETTE[level % len(PALETTE)])

    # Links: co-occurrence counts between consecutive levels (the "all with all").
    srcs, tgts, vals = [], [], []
    for level in range(L - 1):
        a, b = cols[level], cols[level + 1]
        counts = work.groupby([a, b]).size().reset_index(name="value")
        for _, row in counts.iterrows():
            srcs.append(index[(level, row[a])])
            tgts.append(index[(level + 1, row[b])])
            vals.append(int(row["value"]))

    def _rgba(hex_color, alpha=0.35):
        h = hex_color.lstrip("#")
        return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=node_label, color=node_color, pad=14, thickness=18),
        link=dict(source=srcs, target=tgts, value=vals,
                  color=[_rgba(node_color[s]) for s in srcs]),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_size=12, height=max(420, 60 + 24 * max_per_level),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{' → '.join(cols)} · {len(node_label)} nodes · {len(srcs)} links "
        "(all-with-all co-occurrence, no matching)."
    )


def analyze_and_predict(data, analyzers, col_names, clusters):
    visualizer = UAPVisualizer()
    new_data = pd.DataFrame()
    for i, column  in enumerate(col_names):
        #new_data[f'Analyzer_{column}'] = analyzers[column].__dict__['cluster_labels']
        new_data[f'Analyzer_{column}'] = clusters[column]
        data[f'Analyzer_{column}'] = clusters[column]
        #data[f'Analyzer_{column}'] = analyzer.__dict__['cluster_labels']

        print(f"Cluster terms extracted for {column}")

    for col in data.columns:
        if 'Analyzer' in col:
            data[col] = data[col].astype('category')

    new_data = new_data.fillna('null').astype('category')
    data_nums = new_data.apply(lambda x: x.cat.codes)

    for col in data_nums.columns:
        try:
            categories = new_data[col].cat.categories
            x_train, x_test, y_train, y_test = train_test_split(data_nums.drop(columns=[col]), data_nums[col], test_size=0.2, random_state=42)
            bst, accuracy, preds = visualizer.train_xgboost(x_train, y_train, x_test, y_test, len(categories))
            fig = visualizer.plot_results(new_data, bst, x_test, y_test, preds, categories, accuracy, col)
            with st.status(f"Charts Analyses: {col}", expanded=True) as status:
                st.pyplot(fig)
                status.update(label=f"Chart Processed: {col}", expanded=False)   
        except Exception as e:
            print(f"Error processing {col}: {e}")
            continue
    return new_data, data

from config import API_KEY, GEMINI_KEY, FORMAT_LONG

with torch.no_grad():
    torch.cuda.empty_cache()

#st.set_page_config(
#    page_title="UAP ANALYSIS",
#    page_icon=":alien:",
#    layout="wide",
#    initial_sidebar_state="expanded",
#)

st.title('UAP Analysis Dashboard')

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

# Load dataset
data_path = 'parsed_files_distance_embeds.h5'

my_dataset = st.file_uploader("Upload Parsed DataFrame", type=["csv", "xlsx"])
if my_dataset is not None:
    try:
        if my_dataset.type == "text/csv":
            data = pd.read_csv(my_dataset)
        elif my_dataset.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            data = pd.read_excel(my_dataset)
        else:
            st.error("Unsupported file type. Please upload a CSV, Excel or HD5 file.")
            st.stop()
        filtered = filter_dataframe(data)
        st.session_state['parsed_responses'] = filtered
        st.dataframe(filtered)
        st.success(f"Successfully loaded and displayed data from {my_dataset.name}")
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
else:
    parsed = load_data(data_path).drop(columns=['embeddings'])
    parsed_responses = filter_dataframe(parsed)
    st.session_state['parsed_responses'] = parsed_responses
    st.dataframe(parsed_responses)
# Gemini Q&A over a column moved to the RAG search page (rag_search.py).
st.session_state['stage'] = 1

# Add enhanced visualization toggle
with st.expander("🚀 Enhanced Visualization Options", expanded=False):
    use_interactive_viz = st.checkbox("Use Interactive Visualizations", value=True, help="Enable modern interactive charts with better performance")
    show_data_profile = st.checkbox("Show Data Profile", value=True, help="Display intelligent data analysis before filtering")
    enable_performance_mode = st.checkbox("Performance Mode", value=True, help="Enable smart sampling for large datasets")

# TF-IDF cluster naming — default OFF. When ON, clusters get human-readable
# names derived from their top TF-IDF terms and near-duplicate names are
# merged. When OFF, clusters keep their numeric HDBSCAN ids and XGBoost
# downstream runs against the raw labels.
enable_tfidf_clusters = st.toggle(
    "Enable TF-IDF cluster naming + merging",
    value=False,
    key="enable_tfidf_clusters",
    help="When off, clusters are labeled 'Cluster N' and HDBSCAN labels are "
         "passed straight to XGBoost. When on, top TF-IDF terms name each "
         "cluster and near-duplicate names are merged (slower).",
)

# Categorical association explorer — runs directly on the parsed DataFrame,
# independent of the embedding/cluster pipeline below.
if st.session_state['stage'] > 0 and st.session_state.get('parsed_responses') is not None:
    with st.expander("🎲 Categorical Association Explorer (Cramér's V)", expanded=False):
        render_cramers_v_explorer(st.session_state['parsed_responses'])
    with st.expander("🌊 Categorical Flow (Sankey)", expanded=False):
        render_categorical_flow_sankey(st.session_state['parsed_responses'])

if st.session_state['stage'] > 0 :
    # ── High-correlation shortcut ──────────────────────────────────────────────
    # Pre-fill the column selector with whatever the Cramér's V explorer above
    # last flagged as strongly associated. Kept OUTSIDE the form so toggling it
    # reruns immediately and updates the (keyless) multiselect default; a
    # checkbox inside the form would only take effect on submit.
    _valid_cols = list(st.session_state['parsed_responses'].columns)
    _cv_df = st.session_state.get("cv_pass2_df")
    if _cv_df is None:
        _cv_df = st.session_state.get("cv_pass1_df")
    _cv_strong = float(st.session_state.get("cv_strong_threshold", 0.30))
    _cv_excl = bool(st.session_state.get("cv_exclude_trivial", True))
    _high_corr_cols = [
        c for c in high_correlation_columns(_cv_df, _cv_strong, _cv_excl)
        if c in _valid_cols
    ]

    pass_high_corr = st.checkbox(
        "Pass all high-correlated columns from Cramér's V",
        value=False,
        key="analyze_pass_high_corr",
        disabled=not _high_corr_cols,
        help=(
            f"Pre-select every column reaching Cramér's V ≥ {_cv_strong:.2f} with "
            "another column in the latest Cramér's V pass. Run the Categorical "
            "Association Explorer above first to populate this."
        ),
    )
    if not _high_corr_cols:
        st.caption(
            ":grey[No Cramér's V results yet — run the Categorical Association "
            "Explorer above to enable the high-correlation shortcut.]"
        )
    elif pass_high_corr:
        st.caption(
            f":green[{len(_high_corr_cols)} high-correlation column(s) pre-selected:] "
            f"{', '.join(_high_corr_cols[:12])}"
            f"{'…' if len(_high_corr_cols) > 12 else ''}"
        )

    _default_cols = _high_corr_cols if (pass_high_corr and _high_corr_cols) else []

    with st.form(border=True, key='Select Columns for Analysis'):
        columns_to_analyze = st.multiselect(
            label='Select columns to analyze',
            options=st.session_state['parsed_responses'].columns,
            default=_default_cols,
        )
        if st.form_submit_button("Process Data"):
            if columns_to_analyze:
                analyzers = []
                col_names = []
                clusters = {}
                for column in columns_to_analyze:
                    with torch.no_grad():    
                        with st.status(f"Processing {column}", expanded=True) as status:
                            analyzer = UAPAnalyzer(st.session_state['parsed_responses'], column)
                            st.write(f"Processing {column}...")
                            analyzer.preprocess_data(top_n=32)
                            st.write("Reducing dimensionality...")
                            analyzer.reduce_dimensionality(method='UMAP', n_components=2, n_neighbors=15, min_dist=0.1)
                            st.write("Clustering data...")
                            analyzer.cluster_data(method='HDBSCAN', min_cluster_size=15)
                            if enable_tfidf_clusters:
                                analyzer.get_tf_idf_clusters(top_n=3)
                                st.write("Naming clusters...")
                                clusters[column] = analyzer.merge_similar_clusters(
                                    cluster_terms=analyzer.__dict__['cluster_terms'],
                                    cluster_labels=analyzer.__dict__['cluster_labels'],
                                )
                            else:
                                # TF-IDF disabled: numeric placeholder names + raw HDBSCAN labels.
                                _labels = analyzer.__dict__['cluster_labels']
                                analyzer.cluster_terms = pd.Categorical(
                                    [f"Cluster {cid}" for cid in np.unique(_labels) if cid != -1]
                                )
                                clusters[column] = list(_labels)
                            analyzers.append(analyzer)
                            col_names.append(column)
                            
                            # Run the visualization
                            # fig = datamapplot.create_plot(
                            #     analyzer.__dict__['reduced_embeddings'],
                            #     analyzer.__dict__['cluster_labels'].astype(str),
                            #     #label_font_size=11,
                            #     label_wrap_width=20,
                            #     use_medoids=True,
                            # )#.to_html(full_html=False, include_plotlyjs='cdn')
                            # st.pyplot(fig.savefig())
                            status.update(label=f"Processing {column} complete", expanded=False)
                st.session_state['analyzers'] = analyzers
                st.session_state['col_names'] = col_names
                st.session_state['clusters'] = clusters
                
                # save space
                parsed = None
                analyzers = None
                col_names = None
                clusters = None
        
                if st.session_state['clusters'] is not None:
                    try:
                        new_data, parsed_responses = analyze_and_predict(st.session_state['parsed_responses'], st.session_state['analyzers'], st.session_state['col_names'], st.session_state['clusters'])       
                        st.session_state['dataset'] = parsed_responses
                        st.session_state['new_data'] = new_data
                        st.session_state['data_processed'] = True
                    except Exception as e:
                        st.write(f"Error processing data: {e}")
            
                if st.session_state['data_processed']:
                    try:
                        visualizer = UAPVisualizer(data=st.session_state['new_data'])
                        #new_data = pd.DataFrame()  # Assuming new_data is prepared earlier in the code
                        fig2 = visualizer.plot_cramers_v_heatmap(data=st.session_state['new_data'], significance_level=0.05)
                        with st.status(f"Cramer's V Chart", expanded=True) as statuss:
                            st.pyplot(fig2)
                            statuss.update(label="Cramer's V chart plotted", expanded=False)   
                    except Exception as e:
                        st.write(f"Error plotting Cramers V: {e}")
        
                    for i, column in enumerate(st.session_state['col_names']):
                        #if stateful_button(f"Show {column} clusters {i}", key=f"show_{column}_clusters"):
                        # if st.session_state['data_processed']:
                        #     with st.status(f"Show clusters {column}", expanded=True) as stats:
                        #         fig3 = st.session_state['analyzers'][i].plot_embeddings4(title=f"{column} clusters", cluster_terms=st.session_state['analyzers'][i].__dict__['cluster_terms'], cluster_labels=st.session_state['analyzers'][i].__dict__['cluster_labels'], reduced_embeddings=st.session_state['analyzers'][i].__dict__['reduced_embeddings'], column=f'Analyzer_{column}', data=st.session_state['new_data'])
                        #         stats.update(label=f"Show clusters {column} complete", expanded=False)
                        if st.session_state['data_processed']:
                            with st.status(f"Show clusters {column}", expanded=True) as stats:
                                fig3 = st.session_state['analyzers'][i].plot_embeddings4(
                                    title=f"{column} clusters", 
                                    cluster_terms=st.session_state['analyzers'][i].__dict__['cluster_terms'], 
                                    cluster_labels=st.session_state['analyzers'][i].__dict__['cluster_labels'], 
                                    reduced_embeddings=st.session_state['analyzers'][i].__dict__['reduced_embeddings'], 
                                    column=column,  # Use the original column name here
                                    data=st.session_state['parsed_responses']  # Use the original dataset here
                                )
                                stats.update(label=f"Show clusters {column} complete", expanded=False)
                        st.session_state['analysis_complete'] = True


# Gemini Q&A over processed/analyzed data moved to the RAG search page (rag_search.py).

# Enhanced visualization section
if 'analysis_complete' in st.session_state and st.session_state['analysis_complete']:
    st.header("📊 Enhanced Interactive Visualizations")
    
    # Check if enhanced visualization is enabled
    use_interactive_viz = st.session_state.get('use_interactive_viz', True)
    show_data_profile = st.session_state.get('show_data_profile', True)
    enable_performance_mode = st.session_state.get('enable_performance_mode', True)
    
    if use_interactive_viz and 'parsed_responses' in st.session_state:
        
        # Get the parsed responses for visualization
        df_for_viz = st.session_state['parsed_responses']
        
        # Interactive visualization options
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📈 Interactive Charts", "🎯 Analysis Results", "📋 Data Explorer"])
        
        with viz_tab1:
            st.subheader("Interactive Data Exploration")
            
            # Get numeric and categorical columns
            numeric_cols = df_for_viz.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df_for_viz.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols, key="enhanced_x")
                with col2:
                    y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col], key="enhanced_y")
                with col3:
                    color_col = st.selectbox("Color by", ["None"] + categorical_cols, key="enhanced_color")
                    color_col = None if color_col == "None" else color_col
                
                if st.button("Generate Interactive Scatter Plot", key="enhanced_scatter"):
                    fig = Enhanced_Visualizer.plot_interactive_scatter(
                        df_for_viz, x_col, y_col, color_col=color_col, 
                        max_points=10000 if enable_performance_mode else len(df_for_viz)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Interactive correlation matrix
            if len(numeric_cols) >= 3:
                st.subheader("Correlation Analysis")
                selected_cols = st.multiselect("Select columns for correlation", numeric_cols, default=numeric_cols[:8])
                
                if selected_cols and len(selected_cols) >= 2:
                    if st.button("Generate Correlation Matrix", key="enhanced_corr"):
                        fig = Enhanced_Visualizer.plot_correlation_matrix(df_for_viz[selected_cols])
                        st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            st.subheader("Analysis Results Visualization")
            
            # Show cluster analysis results if available
            if st.session_state.get('clusters'):
                for column, cluster_info in st.session_state['clusters'].items():
                    st.write(f"### Cluster Analysis: {column}")
                    
                    # Create interactive treemap for clusters
                    if len(cluster_info) > 0:
                        cluster_df = pd.DataFrame({'cluster': cluster_info})
                        fig = Enhanced_Visualizer.plot_interactive_treemap(cluster_df, 'cluster', top_n=15)
                        st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab3:
            st.subheader("Enhanced Data Explorer")
            
            # Show data profile if enabled
            if show_data_profile:
                with st.expander("📊 Intelligent Data Profile", expanded=False):
                    profile = DataProcessor.profile_data(df_for_viz)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", f"{len(df_for_viz):,}")
                    with col2:
                        st.metric("Categorical Cols", len(profile['categorical_columns']))
                    with col3:
                        st.metric("Numeric Cols", len(profile['numeric_columns']))
                    with col4:
                        st.metric("Memory Usage", f"{profile['memory_usage'] / 1024**2:.1f} MB")
            
            # Enhanced filtering
            st.write("### Advanced Data Filtering")
            filtered_df = DataProcessor.filter_dataframe_enhanced(df_for_viz, enable_quick_filters=False, enable_advanced_filters=True)
            
            if len(filtered_df) != len(df_for_viz):
                st.success(f"✅ Filtered data: {len(filtered_df):,} rows (from {len(df_for_viz):,})")
                
                # Option to re-run analysis on filtered data
                if st.button("🔄 Re-run Analysis on Filtered Data"):
                    st.session_state['parsed_responses'] = filtered_df
                    st.session_state['analysis_complete'] = False
                    st.rerun()
