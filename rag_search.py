
import streamlit as st
import pandas as pd
import cohere
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from uap_analyzer import UAPParser, UAPAnalyzer, UAPVisualizer, get_embed_model
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
import json
from utils.data_processing import DataProcessor

# st.set_option('deprecation.showPyplotGlobalUse', False)

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


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

def plot_date_distribution(df, date_column, figsize=(14, 8), color='orange', title=None):
    """
    Create a year-based bar chart showing counts of data points per year/month.
    
    Args:
        df (pd.DataFrame): DataFrame with date data
        date_column (str): Name of the date column
        figsize (tuple): Figure size
        color (str): Color for the bars
        title (str): Chart title
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=150)
        fig.patch.set_facecolor('white')
        
        # Validate input
        if date_column not in df.columns:
            ax1.text(0.5, 0.5, f'Date column "{date_column}" not found', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12, color='red')
            ax2.text(0.5, 0.5, f'Date column "{date_column}" not found', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12, color='red')
            return fig
        
        # Ensure we have a datetime column
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            date_series = pd.to_datetime(df[date_column], errors='coerce')
        else:
            date_series = df[date_column]
        
        # Remove NaT values
        date_series = date_series.dropna()
        
        if len(date_series) == 0:
            ax1.text(0.5, 0.5, 'No valid dates found', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No valid dates found', ha='center', va='center', transform=ax2.transAxes)
            return fig
        
        # Plot 1: Year-based counts
        year_counts = date_series.dt.year.value_counts().sort_index()
        
        if len(year_counts) > 0:
            bars1 = ax1.bar(year_counts.index, year_counts.values, color=color, alpha=0.7, 
                           edgecolor='white', linewidth=0.5)
            ax1.set_title('Events by Year', color=color, fontweight='bold', fontsize=12)
            ax1.set_xlabel('Year', color=color)
            ax1.set_ylabel('Count', color=color)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8, color='black')
            
            # Style the axes
            for spine in ax1.spines.values():
                spine.set_color(color)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.tick_params(axis='both', colors=color)
            ax1.grid(True, alpha=0.3, axis='y', color=color)
        
        # Plot 2: Month-based counts
        month_counts = date_series.dt.month.value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        if len(month_counts) > 0:
            month_labels = [month_names[i-1] for i in month_counts.index]
            bars2 = ax2.bar(month_labels, month_counts.values, color=color, alpha=0.7, 
                           edgecolor='white', linewidth=0.5)
            ax2.set_title('Events by Month', color=color, fontweight='bold', fontsize=12)
            ax2.set_xlabel('Month', color=color)
            ax2.set_ylabel('Count', color=color)
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8, color='black')
            
            # Style the axes
            for spine in ax2.spines.values():
                spine.set_color(color)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.tick_params(axis='both', colors=color)
            ax2.grid(True, alpha=0.3, axis='y', color=color)
        
        # Overall title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', color=color)
        else:
            fig.suptitle(f'Temporal Distribution: {date_column}', fontsize=14, fontweight='bold', color=color)
        
        # Set transparent background
        fig.patch.set_alpha(0)
        ax1.patch.set_alpha(0)
        ax2.patch.set_alpha(0)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Error creating date visualization:\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
        ax.set_title('Date Visualization Error', color='red')
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        plt.tight_layout()
        return fig


def gemini_query(question, selected_data, gemini_key):
    """Ask Gemini a question over a column's values (blank question → summarize).

    Moved here from analyzing.py so all natural-language search/Q&A over a
    dataset lives on the RAG search page.
    """
    import google.generativeai as genai

    if not question:
        question = "Summarize the following data in relevant bullet points"

    filtered = [str(x) for x in selected_data if str(x) != '' and x is not None]
    context = '\n'.join(filtered)

    genai.configure(api_key=gemini_key)
    query_model = genai.GenerativeModel('models/gemini-3.1-pro-preview')
    response = query_model.generate_content(
        [f"{question}\n Answer based on this context: {context}\n\n"]
    )
    return response.text


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Shared filtering UI — delegates to the DataProcessor so the RAG search
    page uses the same filter system as the parsing and analysis pages. Binary
    0/1 columns get a value picker; other numerics get range/percentile/std-dev.
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

    df_ = df.copy()

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
                df_[column] = pd.to_datetime(df_[column], infer_datetime_format=True, errors='coerce')
            except Exception:
                try:
                    df_[column] = df_[column].apply(
                        lambda x: parser.parse(str(x)) if pd.notna(x) else pd.NaT
                    )
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
                            if numeric_columns:
                                fig = plot_line(df_, date_column, numeric_columns)
                                with st.status(f"Date Numerical Distributions: {column}", expanded=False) as stat:
                                    try:
                                        st.pyplot(fig)
                                    except Exception as e:
                                        st.error(f"Error plotting line chart: {e}")
                            categorical_columns = [col for col in filtered_columns if is_categorical_dtype(df_[col])]
                            if categorical_columns:
                                fig2 = plot_grouped_bar(df_, categorical_columns, date_column)
                                with st.status(f"Date Categorical Distributions: {column}", expanded=False) as sta:
                                    try:
                                        st.pyplot(fig2)
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

        else:
            user_text_input = right.text_input(
                f"Substring or regex in {column}",
            )
            if user_text_input:
                df_ = df_[df_[column].astype(str).str.contains(user_text_input)]
    # write len of df after filtering with % of original
    st.write(f"{len(df_)} rows ({len(df_) / len(df) * 100:.2f}%)")
    return df_


# ──────────────────────────────────────────────────────────────────────────
# Cross-DB threshold suggestion — sample matched pairs across similarity
# bands and have an LLM judge which are real duplicates.
# ──────────────────────────────────────────────────────────────────────────
def _make_bands(lo_pct, hi_pct, step_pct):
    """Build [(lo, hi), …] cosine-similarity bands (0–1 scale) from percent edges."""
    lo_pct, hi_pct, step_pct = int(lo_pct), int(hi_pct), int(step_pct)
    edges = list(range(lo_pct, hi_pct, step_pct)) + [hi_pct]
    return [(edges[k] / 100.0, edges[k + 1] / 100.0) for k in range(len(edges) - 1)]


def _collect_pairs(top_scores, top_idx):
    """Flatten the (rows × K) match grid into parallel (row, col, score) arrays,
    dropping masked (-inf) self-similarity diagonal entries."""
    na, K = top_scores.shape
    rows = np.repeat(np.arange(na), K)
    cols = np.asarray(top_idx).reshape(-1)
    scores = np.asarray(top_scores, dtype=np.float64).reshape(-1)
    keep = np.isfinite(scores)
    return rows[keep], cols[keep], scores[keep]


def _bin_pairs(rows, cols, scores, bands, n_per_bin, seed=0):
    """Sample up to n_per_bin (row, col, score) triples within each band.
    Passing n_per_bin=None disables subsampling — every pair in the band is kept."""
    rng = np.random.default_rng(seed)
    out = []
    for k, (lo, hi) in enumerate(bands):
        is_last = k == len(bands) - 1
        mask = (scores >= lo) & ((scores <= hi) if is_last else (scores < hi))
        idx = np.where(mask)[0]
        n_avail = len(idx)
        if n_per_bin is not None and n_avail > n_per_bin:
            idx = rng.choice(idx, n_per_bin, replace=False)
        sampled = [(int(rows[t]), int(cols[t]), float(scores[t])) for t in idx]
        out.append({'lo': lo, 'hi': hi, 'pairs': sampled, 'n_available': n_avail})
    return out


def _row_to_text(df, pos, cols, max_chars=400):
    """Render one DataFrame row (by positional index) as 'col: value' lines."""
    try:
        row = df.iloc[pos]
    except (IndexError, KeyError):
        return "(row unavailable)"
    parts = []
    for c in cols:
        if c not in row.index:
            continue
        val = row[c]
        if pd.isna(val):
            continue
        s = str(val).strip()
        if not s:
            continue
        if len(s) > max_chars:
            s = s[:max_chars] + "…"
        parts.append(f"{c}: {s}")
    return "\n".join(parts) if parts else "(empty)"


# Trace columns appended to every duplicates-download row so each flagged pair
# can be traced back to its original record on either side, even when those
# columns weren't part of the embedding/judge text. Case-insensitive lookup.
TRACE_COL_CANDIDATES = ['source', 'denseNarrative']


def _resolve_trace_cols(df_columns, already_selected):
    """Return the trace columns present in `df_columns` (case-insensitive)
    that aren't already in `already_selected`. Preserves candidate order."""
    lookup = {str(c).lower(): c for c in df_columns}
    exclude = {str(c).lower() for c in already_selected}
    found = []
    for tc in TRACE_COL_CANDIDATES:
        key = tc.lower()
        if key in lookup and key not in exclude:
            found.append(lookup[key])
    return found


def _safe_json(raw):
    """Parse a JSON array from an LLM response, tolerating code fences / prose."""
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


# Gemini judging budget — shared between actual chunking and upfront estimates.
GEMINI_INPUT_TOK_CAP = 900_000      # ~10% headroom below the 1M input window
GEMINI_OUTPUT_TOK_CAP = 60_000      # ~10% headroom below the 65k output cap
GEMINI_PROMPT_OVERHEAD_TOK = 600    # rubric + JSON instructions
GEMINI_OUT_TOK_PER_PAIR = 80        # one verdict object


def _est_pair_in_tokens(item):
    """Rough input-token estimate for one judge item (chars / 4)."""
    return (len(item['text_a']) + len(item['text_b']) + 60) // 4


def _chunk_judge_items(items,
                       max_in=GEMINI_INPUT_TOK_CAP,
                       max_out=GEMINI_OUTPUT_TOK_CAP):
    """Greedy-chunk items so each Gemini call respects both budgets."""
    chunks = []
    cur, cur_in, cur_out = [], GEMINI_PROMPT_OVERHEAD_TOK, 0
    for it in items:
        ti = _est_pair_in_tokens(it)
        if cur and (cur_in + ti > max_in
                    or cur_out + GEMINI_OUT_TOK_PER_PAIR > max_out):
            chunks.append(cur)
            cur, cur_in, cur_out = [], GEMINI_PROMPT_OVERHEAD_TOK, 0
        cur.append(it)
        cur_in += ti
        cur_out += GEMINI_OUT_TOK_PER_PAIR
    if cur:
        chunks.append(cur)
    return chunks


def _estimate_judge_cost(all_pairs, cols_by_pair, bands, n_per_bin):
    """Estimate (total_pairs, input_tokens, output_tokens, n_calls) for a pass.
    Uses a 5-pair per-DB sample of text length to extrapolate, so this stays
    cheap even when n_per_bin is None (ALL mode) over very large DataFrames."""
    total_pairs = 0
    total_in_tokens = GEMINI_PROMPT_OVERHEAD_TOK
    for pid, res in all_pairs.items():
        cols_a, cols_b = cols_by_pair.get(pid, (res['db_a']['cols'], res['db_b']['cols']))
        rows, cols, scores = _collect_pairs(res['top_scores'], res['top_idx'])

        db_total = 0
        for k, (lo, hi) in enumerate(bands):
            is_last = k == len(bands) - 1
            mask = (scores >= lo) & ((scores <= hi) if is_last else (scores < hi))
            n_avail = int(mask.sum())
            n_use = n_avail if n_per_bin is None else min(n_per_bin, n_avail)
            db_total += n_use
        total_pairs += db_total

        if db_total > 0 and len(rows) > 0:
            sample_n = min(5, len(rows))
            sample_in = 0
            for ti in range(sample_n):
                ta = _row_to_text(res['db_a']['df'], int(rows[ti]), cols_a)
                tb = _row_to_text(res['db_b']['df'], int(cols[ti]), cols_b)
                sample_in += (len(ta) + len(tb) + 60) // 4
            avg_in_per_pair = sample_in / sample_n
            total_in_tokens += int(db_total * avg_in_per_pair)

    total_out_tokens = total_pairs * GEMINI_OUT_TOK_PER_PAIR
    n_calls_in = max(1, (total_in_tokens + GEMINI_INPUT_TOK_CAP - 1)
                     // GEMINI_INPUT_TOK_CAP)
    n_calls_out = max(1, (total_out_tokens + GEMINI_OUTPUT_TOK_CAP - 1)
                      // GEMINI_OUTPUT_TOK_CAP)
    return total_pairs, total_in_tokens, total_out_tokens, max(n_calls_in, n_calls_out)


def _format_cost_line(label, n_pairs, in_tok, out_tok, n_calls):
    """One-line cost summary used by the threshold tab UIs."""
    calls_part = ("1 Gemini call" if n_calls == 1
                  else f"**{n_calls} chunked calls**")
    return (f"📊 {label}: **{n_pairs:,} pairs** · ~{in_tok:,} input · "
            f"~{out_tok:,} output tokens · {calls_part}")


def _gemini_judge_pairs(items, gemini_key, model_name="models/gemini-3.1-pro-preview"):
    """Ask Gemini whether each pair is a true duplicate. Auto-splits into
    multiple calls whenever the input or output would exceed Gemini's window
    (input cap and output cap defined at the top of this module).

    items: list of {'id': int, 'text_a': str, 'text_b': str}.
    Returns {id: {'duplicate': bool, 'reason': str}}.
    """
    if not items:
        return {}
    import google.generativeai as genai
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(
        model_name,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.0,
            "max_output_tokens": 65536,
        },
    )

    chunks = _chunk_judge_items(items)
    n_chunks = len(chunks)
    pbar = (st.progress(0.0, text=f"Gemini judging — call 1/{n_chunks}")
            if n_chunks > 1 else None)

    out = {}
    for ci, chunk in enumerate(chunks):
        blocks = [
            f"=== PAIR {it['id']} ===\nRecord A:\n{it['text_a']}\n\nRecord B:\n{it['text_b']}"
            for it in chunk
        ]
        prompt = (
            "You are a record-deduplication judge. Each numbered pair below holds two "
            "records, A and B, drawn from two databases. For EACH pair decide whether "
            "A and B describe the SAME real-world report/event — a true duplicate — "
            "allowing for differences in wording, formatting, units, ordering, missing "
            "fields or partial detail. Judge the underlying entity, not the surface text.\n\n"
            f"Return ONLY a JSON array of exactly {len(chunk)} objects, one per pair:\n"
            '{"id": <pair id>, "duplicate": <true|false>, "reason": "<short justification>"}\n\n'
            + "\n\n".join(blocks)
        )
        resp = model.generate_content(prompt)
        for v in _safe_json(resp.text):
            try:
                out[int(v['id'])] = {
                    'duplicate': bool(v.get('duplicate', False)),
                    'reason': str(v.get('reason', '')),
                }
            except (KeyError, TypeError, ValueError):
                continue
        if pbar:
            pbar.progress((ci + 1) / n_chunks,
                          text=f"Gemini call {ci + 1}/{n_chunks} — "
                               f"{len(out):,} verdicts so far")
    if pbar:
        pbar.empty()
    return out


def _run_threshold_pass(bands, top_scores, top_idx, prim, corp, cols_a, cols_b,
                        n_per_bin, gemini_key, target, db_a_name, db_b_name):
    """Sample pairs per band, judge them with Gemini, aggregate per-band ratios."""
    rows, cols, scores = _collect_pairs(top_scores, top_idx)
    binned = _bin_pairs(rows, cols, scores, bands, n_per_bin)

    # Trace columns (source, denseNarrative…) extend the snapshot so the
    # duplicates CSV is traceable back to the source records; they are NOT
    # added to the LLM-facing text (cols_a / cols_b) to keep prompts compact.
    trace_a = _resolve_trace_cols(prim.columns, cols_a)
    trace_b = _resolve_trace_cols(corp.columns, cols_b)
    snap_cols_a = list(cols_a) + trace_a
    snap_cols_b = list(cols_b) + trace_b

    items, meta, gid = [], [], 0
    for bi, b in enumerate(binned):
        for (ai, bj, sc) in b['pairs']:
            ta = _row_to_text(prim, ai, cols_a)
            tb = _row_to_text(corp, bj, cols_b)
            row_a = {c: prim.iloc[ai].get(c, '') for c in snap_cols_a}
            row_b = {c: corp.iloc[bj].get(c, '') for c in snap_cols_b}
            items.append({'id': gid, 'text_a': ta, 'text_b': tb})
            meta.append({'id': gid, 'band_i': bi, 'ai': ai, 'bi': bj,
                         'score': sc, 'text_a': ta, 'text_b': tb,
                         'row_a': row_a, 'row_b': row_b})
            gid += 1

    verdicts = _gemini_judge_pairs(items, gemini_key)

    band_results = []
    for bi, b in enumerate(binned):
        rws = []
        for m in meta:
            if m['band_i'] != bi:
                continue
            v = verdicts.get(m['id'], {'duplicate': False,
                                       'reason': '(no verdict returned)'})
            rws.append({**m, 'duplicate': bool(v['duplicate']), 'reason': v['reason']})
        n = len(rws)
        n_dup = sum(1 for r in rws if r['duplicate'])
        band_results.append({
            'lo': b['lo'], 'hi': b['hi'],
            'label': f"{b['lo'] * 100:.0f}–{min(b['hi'], 1.0) * 100:.0f}%",
            'n_sampled': n, 'n_dup': n_dup,
            'ratio': (n_dup / n) if n else 0.0,
            'n_available': b['n_available'], 'rows': rws,
        })
    return {
        'bands': band_results, 'target': target,
        'db_a_name': db_a_name, 'db_b_name': db_b_name,
        'cols_a': snap_cols_a, 'cols_b': snap_cols_b,
        'n_judged': len(items), 'n_verdicts': len(verdicts),
    }


def _suggested_threshold(band_results, target):
    """Lowest band (percent) whose duplicate ratio reaches the target, else None."""
    for b in band_results:
        if b['n_sampled'] > 0 and b['ratio'] >= target:
            return int(round(b['lo'] * 100))
    return None


def _transition_band(band_results, target):
    """Percent window (lo, hi) to refine in pass 2 — the band where duplicates
    take over, or the highest-ratio populated band as a fallback."""
    for b in band_results:
        if b['n_sampled'] > 0 and b['ratio'] >= target:
            return int(round(b['lo'] * 100)), int(round(b['hi'] * 100))
    populated = [b for b in band_results if b['n_sampled'] > 0]
    if populated:
        best = max(populated, key=lambda b: b['ratio'])
        return int(round(best['lo'] * 100)), int(round(best['hi'] * 100))
    return 85, 95


def _render_threshold_pass(pass_data, title, key_prefix, prim=None):
    """Draw the per-band duplicate-ratio bar chart + an inspector for one pass.
    Returns the suggested threshold percent (or None)."""
    import plotly.graph_objects as go

    band_results = pass_data['bands']
    target = pass_data['target']
    da, db = pass_data['db_a_name'], pass_data['db_b_name']
    suggested = _suggested_threshold(band_results, target)

    labels = [b['label'] for b in band_results]
    ratios = [round(b['ratio'] * 100, 1) for b in band_results]
    bar_text = [f"{b['n_dup']}/{b['n_sampled']}" if b['n_sampled'] else "n/a"
                for b in band_results]
    colors = ["#22c55e" if (b['n_sampled'] and b['ratio'] >= target) else "#f97316"
              for b in band_results]
    fig = go.Figure(go.Bar(
        x=labels, y=ratios, text=bar_text, textposition="outside",
        marker_color=colors,
        customdata=[[b['n_dup'], b['n_sampled'], b['n_available']]
                    for b in band_results],
        hovertemplate=("Band %{x}<br>%{customdata[0]} of %{customdata[1]} "
                       "judged duplicate<br>%{customdata[2]:,} pairs available"
                       "<extra></extra>"),
    ))
    fig.add_hline(y=target * 100, line_dash="dash", line_color="red",
                  annotation_text=f"target {target:.0%}",
                  annotation_position="top left")
    fig.update_layout(
        title=title, xaxis_title="Cosine-similarity band",
        yaxis_title="Real-duplicate ratio (%)", yaxis_range=[0, 112],
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    if pass_data.get('n_verdicts', 0) < pass_data.get('n_judged', 0):
        st.warning(
            f"Gemini returned {pass_data['n_verdicts']} of {pass_data['n_judged']} "
            "verdicts — missing pairs counted as non-duplicates. Re-run if needed."
        )
    if suggested is not None:
        st.success(
            f"**Suggested threshold ≈ {suggested}%** — the lowest band whose "
            f"Gemini-judged duplicate ratio reaches {target:.0%}."
        )
    else:
        st.warning(
            "No band reached the target duplicate ratio — try raising the sample "
            "size, lowering the target, or inspecting the judged pairs below."
        )

    with st.expander(f"🔍 Inspect the {sum(b['n_sampled'] for b in band_results)} "
                     "judged pairs"):
        insp = [
            {
                'band': b['label'],
                'similarity_pct': round(r['score'] * 100, 1),
                'is_duplicate': r['duplicate'],
                da: r['text_a'].replace("\n", " | "),
                db: r['text_b'].replace("\n", " | "),
                'gemini_reason': r['reason'],
            }
            for b in band_results for r in b['rows']
        ]
        if insp:
            insp_df = pd.DataFrame(insp)
            st.dataframe(insp_df, use_container_width=True, hide_index=True)
            # All-rows flagged dataset: copy of the primary DB with
            # is_duplicate / similarity / reason filled for each primary row
            # that had at least one Gemini-confirmed match (best match wins
            # on ties). Empty fields elsewhere. Same shape as the flag UI
            # download so per-primary counts agree.
            best_per_a = {}  # ai -> (score, bi, reason)
            for b in band_results:
                for r in b['rows']:
                    if not r['duplicate']:
                        continue
                    a = int(r['ai'])
                    s = float(r['score'])
                    if a not in best_per_a or s > best_per_a[a][0]:
                        best_per_a[a] = (s, int(r['bi']),
                                         str(r.get('reason', '')))
            n_flagged = len(best_per_a)
            _jc, _dc = st.columns(2)
            with _jc:
                st.download_button(
                    "⬇️ Download judged pairs (CSV)", insp_df.to_csv(index=False),
                    f"{key_prefix}_judged_pairs.csv", "text/csv",
                    key=f"dl_{key_prefix}",
                )
            with _dc:
                if prim is None:
                    st.caption(
                        "Primary DataFrame not available — flagged-rows "
                        "download disabled."
                    )
                elif n_flagged:
                    n_total = len(prim)
                    is_dup  = np.zeros(n_total, dtype=bool)
                    dup_sim = np.full(n_total, np.nan, dtype=float)
                    dup_row = np.full(n_total, np.nan, dtype=float)
                    gem_rsn = np.array([''] * n_total, dtype=object)
                    for a, (s, b_idx, reason) in best_per_a.items():
                        if 0 <= a < n_total:
                            is_dup[a]  = True
                            dup_sim[a] = round(s * 100, 1)
                            dup_row[a] = b_idx
                            gem_rsn[a] = reason
                    out_df = prim.copy()
                    out_df['is_duplicate'] = is_dup
                    out_df['duplicate_similarity_pct'] = dup_sim
                    out_df['duplicate_source_db'] = np.where(is_dup, db, '')
                    out_df['duplicate_match_row'] = dup_row
                    out_df['gemini_reason'] = gem_rsn
                    st.download_button(
                        f"⬇️ Download all rows w/ duplicates flagged "
                        f"({n_flagged:,}/{n_total:,}) (CSV)",
                        out_df.to_csv(index=False),
                        f"{key_prefix}_flagged_dataset.csv", "text/csv",
                        key=f"dl_{key_prefix}_dups",
                    )
                else:
                    st.caption("No Gemini-confirmed duplicates to flag.")
        else:
            st.caption("No pairs were sampled.")
    return suggested


def _threshold_tab(res, sel_key, gemini_key, pct=70):
    """Render the 'Suggest Threshold' UI for one cross-DB pair result.

    Widget and session-state keys are suffixed by `sel_key` so the tab can be
    reused across the All-view (multiple pairs selectable) and the single-pair
    view without state collisions when switching pairs.

    `pct` is the outer 'Similarity threshold' slider value (percent). Pass 1
    bands start from it — clamped to 0–95 so at least one 5% band exists. The
    inline 'Flag duplicates' UI uses the raw `pct` so flagging matches the
    main view.
    """
    pct = int(pct)
    min_pct = max(0, min(pct, 95))
    prim = res['db_a']['df']
    corp = res['db_b']['df']
    top_scores = res['top_scores']
    top_idx = res['top_idx']
    da_name = res['db_a']['name']
    db_name = res['db_b']['name']
    kp = "_".join(str(x) for x in (sel_key if isinstance(sel_key, tuple) else (sel_key,)))

    st.caption(
        "Samples matched pairs across cosine-similarity bands and asks Gemini to "
        "judge which are *real* duplicates — the duplicate ratio per band reveals "
        "where a meaningful similarity cutoff lies. A 2nd pass refines it at 1%."
    )

    with st.expander("⚙️ Configuration", expanded=False):
        _tc1, _tc2 = st.columns(2)
        with _tc1:
            st.caption(f"**{da_name}** fields shown to the judge")
            thr_cols_a = st.multiselect(
                "Columns A", list(prim.columns),
                default=res['db_a']['cols'], key=f"xdb_thr_cols_a_{kp}",
            )
        with _tc2:
            st.caption(f"**{db_name}** fields shown to the judge")
            thr_cols_b = st.multiselect(
                "Columns B", list(corp.columns),
                default=res['db_b']['cols'], key=f"xdb_thr_cols_b_{kp}",
            )
        thr_gemini_key = gemini_key  # set at the top of the cross-DB section
        _ta, _ts, _tt = st.columns([1, 1, 1])
        with _ta:
            sample_all = st.toggle(
                "Sample ALL", value=False, key=f"xdb_thr_all_{kp}",
                help="Judge every pair in each band (no subsampling). Cost "
                     "scales with the data; if it exceeds Gemini's 1M-token "
                     "window the request is split into multiple calls.",
            )
        with _ts:
            thr_n = st.slider("Samples per band", 3, 30, 10,
                              key=f"xdb_thr_spb_{kp}", disabled=sample_all)
        with _tt:
            thr_target = st.slider(
                "Target duplicate ratio", 0.0, 1.0, 0.5, 0.01,
                key=f"xdb_thr_target_{kp}",
                help="The lowest band whose real-duplicate ratio reaches this "
                     "value sets the suggested threshold.",
            )

    _thr_ready = bool(thr_gemini_key and thr_cols_a and thr_cols_b)
    if not _thr_ready:
        st.info("Set a Gemini API key and at least one column per database in "
                "⚙️ Configuration to enable suggestion.")

    n_per_bin_eff = None if sample_all else thr_n
    _cols_by_pair = {sel_key: (thr_cols_a, thr_cols_b)}
    _p1_bands = _make_bands(min_pct, 100, 5)
    _n_pairs, _in_tok, _out_tok, _n_calls = _estimate_judge_cost(
        {sel_key: res}, _cols_by_pair, _p1_bands, n_per_bin_eff,
    )
    _cost = _format_cost_line("Pass 1", _n_pairs, _in_tok, _out_tok, _n_calls)
    if _n_calls > 1:
        st.warning(_cost + " — exceeds Gemini's 1M-token window, will be split.")
    else:
        st.caption(_cost)

    p1_key, p2_key = f'xdb_thr_pass1_{kp}', f'xdb_thr_pass2_{kp}'

    if st.button(
        f"🎯 Suggest threshold — pass 1 ({min_pct}→100%, 5% bands)",
        type="primary", disabled=not _thr_ready, key=f"xdb_thr_run1_{kp}",
    ):
        if top_scores.size == 0:
            st.warning("No similarity scores available for this pair.")
        else:
            try:
                _spin_msg = (
                    "Sampling ALL pairs/band and asking Gemini to judge…"
                    if sample_all else
                    f"Sampling {thr_n} pairs/band and asking Gemini to judge…"
                )
                with st.spinner(_spin_msg):
                    _p1 = _run_threshold_pass(
                        _p1_bands,
                        top_scores, top_idx, prim, corp,
                        thr_cols_a, thr_cols_b, n_per_bin_eff,
                        thr_gemini_key, thr_target, da_name, db_name,
                    )
                st.session_state[p1_key] = _p1
                st.session_state.pop(p2_key, None)
            except Exception as _e:
                st.error(f"Threshold suggestion failed: {_e}")

    _p1 = st.session_state.get(p1_key)
    if _p1:
        st.markdown(f"#### Pass 1 — 5% bands ({min_pct}→100%)")
        _render_threshold_pass(_p1, "Real-duplicate ratio per 5% band",
                               f"thr_pass1_{kp}", prim=prim)

        st.divider()
        st.markdown("#### Pass 2 — refine at 1% resolution")
        _t_lo, _t_hi = _transition_band(_p1['bands'], _p1['target'])
        st.caption(
            f"Proposed refinement window **{_t_lo}–{_t_hi}%** — the band where "
            "duplicates take over. Adjust if needed."
        )
        _rc1, _rc2, _rc3 = st.columns([1, 1, 2])
        with _rc1:
            _r_lo = int(st.number_input("From %", 1, 99, _t_lo,
                                        key=f"xdb_thr_rlo_{kp}"))
        with _rc2:
            _r_hi = int(st.number_input("To %", _r_lo + 1, 100,
                                        max(_t_hi, _r_lo + 1),
                                        key=f"xdb_thr_rhi_{kp}"))
        with _rc3:
            _samples_label = "ALL" if sample_all else str(thr_n)
            st.caption(f"Samples {_samples_label} pairs in each 1% band from "
                       f"{_r_lo}% to {_r_hi}%.")

        _p2_bands = _make_bands(_r_lo, _r_hi, 1)
        _p2_n, _p2_in, _p2_out, _p2_calls = _estimate_judge_cost(
            {sel_key: res}, _cols_by_pair, _p2_bands, n_per_bin_eff,
        )
        _p2_cost = _format_cost_line("Pass 2", _p2_n, _p2_in, _p2_out, _p2_calls)
        if _p2_calls > 1:
            st.warning(_p2_cost + " — exceeds Gemini's 1M-token window, will be split.")
        else:
            st.caption(_p2_cost)

        if st.button(
            f"🔬 Run refined pass ({_r_lo}–{_r_hi}%, 1% bands)",
            disabled=not _thr_ready, key=f"xdb_thr_run2_{kp}",
        ):
            try:
                with st.spinner("Re-judging refined bands with Gemini…"):
                    _p2 = _run_threshold_pass(
                        _p2_bands,
                        top_scores, top_idx, prim, corp,
                        thr_cols_a, thr_cols_b, n_per_bin_eff,
                        thr_gemini_key, thr_target, da_name, db_name,
                    )
                st.session_state[p2_key] = _p2
            except Exception as _e:
                st.error(f"Refined pass failed: {_e}")

        _p2 = st.session_state.get(p2_key)
        if _p2:
            _sug2 = _render_threshold_pass(_p2, "Real-duplicate ratio per 1% band",
                                           f"thr_pass2_{kp}", prim=prim)
            if _sug2 is not None:
                st.info(
                    f"🎯 Refined suggested threshold **{_sug2}%** — set the "
                    "**Similarity threshold** slider above to this value to apply it."
                )

    st.divider()
    st.markdown("**Flag duplicates**")
    _has_gemini = bool(st.session_state.get(p2_key)
                       or st.session_state.get(p1_key))
    fc1, fc2 = st.columns([3, 1])
    with fc1:
        flag_scope = st.radio(
            "Scope",
            [f"All matches above threshold ({pct}%)", "100% matches only"],
            key=f"xdb_thr_flag_scope_{kp}",
            horizontal=True,
            label_visibility="collapsed",
        )
    with fc2:
        do_flag = st.button(
            "🚩 Flag as duplicates", key=f"xdb_thr_flag_btn_{kp}",
            use_container_width=True,
        )
    if _has_gemini and flag_scope != "100% matches only":
        st.caption(
            f"🤖 Gemini verdicts available — flagging will use **only the "
            f"Gemini-confirmed duplicates** from the latest pass, not every "
            f"pair above {pct}%."
        )
    else:
        st.caption(
            f"Marks matched **{da_name}** rows with `is_duplicate`, "
            f"`duplicate_similarity_pct`, `duplicate_source_db`, and the matched "
            f"**{db_name}** columns as `dup__{db_name}__<col>`. "
            "Each flagged row keeps its single best (highest-similarity) match."
        )
    if do_flag:
        _exact = flag_scope == "100% matches only"
        _thr = 1.0 if _exact else pct / 100.0
        # Pull Gemini verdicts from the latest pass that ran (pass 2 wins if
        # both exist). Reasons are keyed by (ai, bi) so they line up with
        # whichever match survives the best-per-row dedupe below.
        _reason_lookup = {}
        for _pk in (p2_key, p1_key):
            _pass = st.session_state.get(_pk)
            if not _pass:
                continue
            for _band in _pass['bands']:
                for _r in _band['rows']:
                    _reason_lookup[(int(_r['ai']), int(_r['bi']))] = {
                        'duplicate': bool(_r['duplicate']),
                        'reason': str(_r.get('reason', '')),
                        'score': float(_r['score']),
                    }
            if _reason_lookup:
                break

        # When Gemini judgments exist (and the user isn't restricting to
        # 100% matches), trust them: only flag pairs Gemini confirmed as
        # duplicates, regardless of how many other pairs sit above pct.
        _use_gemini = bool(_reason_lookup) and not _exact
        if _use_gemini:
            _scope = [
                (a, b, v['score']) for (a, b), v in _reason_lookup.items()
                if v['duplicate']
            ]
        else:
            _scope = [
                (int(i), int(top_idx[i, j]), float(top_scores[i, j]))
                for i in range(len(top_scores))
                for j in range(top_scores.shape[1])
                if np.isfinite(top_scores[i, j]) and top_scores[i, j] >= _thr
            ]

        _n = len(prim)
        _is_dup  = np.zeros(_n, dtype=bool)
        _dup_sim = np.full(_n, np.nan, dtype=float)
        _dup_row = np.full(_n, np.nan, dtype=float)
        _bcols = list(dict.fromkeys(res['db_b']['cols']))
        _dup_extra = {c: np.array([None] * _n, dtype=object) for c in _bcols}
        _gem_verdict = np.array([''] * _n, dtype=object)
        _gem_reason  = np.array([''] * _n, dtype=object)
        _best = {}
        for _a, _b, _s in _scope:
            if _a not in _best or _s > _best[_a][0]:
                _best[_a] = (_s, _b)
        for _a, (_s, _b) in _best.items():
            _is_dup[_a]  = True
            _dup_sim[_a] = round(_s * 100, 1)
            _dup_row[_a] = _b
            for _c in _bcols:
                _dup_extra[_c][_a] = corp.iloc[_b].get(_c, None)
            _v = _reason_lookup.get((_a, _b))
            if _v is not None:
                _gem_verdict[_a] = 'duplicate' if _v['duplicate'] else 'not_duplicate'
                _gem_reason[_a]  = _v['reason']
        _flagged_df = prim.copy()
        _flagged_df['is_duplicate'] = _is_dup
        _flagged_df['duplicate_similarity_pct'] = _dup_sim
        _flagged_df['duplicate_source_db'] = np.where(_is_dup, db_name, '')
        _flagged_df['duplicate_match_row'] = _dup_row
        if _reason_lookup:
            _flagged_df['gemini_verdict'] = _gem_verdict
            _flagged_df['gemini_reason']  = _gem_reason
        for _c in _bcols:
            _flagged_df[f"dup__{db_name}__{_c}"] = _dup_extra[_c]
        if _use_gemini:
            _scope_desc = "Gemini-confirmed duplicates"
        elif _exact:
            _scope_desc = "100% matches"
        else:
            _scope_desc = f"≥{pct}% matches"
        st.session_state[f'xdb_thr_flagged_{kp}'] = {
            'pair': sel_key,
            'df': _flagged_df,
            'caption': (
                f"Flagged **{int(_is_dup.sum()):,}** of {_n:,} "
                f"{da_name} rows as duplicates of {db_name} "
                f"({_scope_desc})."
            ),
        }
    _flagged = st.session_state.get(f'xdb_thr_flagged_{kp}')
    if _flagged and _flagged.get('pair') == sel_key:
        st.success(_flagged['caption'])
        _fdf = _flagged['df']
        _dups_only = _fdf[_fdf['is_duplicate']]
        st.caption(f"Preview — {len(_dups_only):,} flagged row(s):")
        st.dataframe(_dups_only, use_container_width=True)
        st.download_button(
            "⬇️ Download flagged dataset (CSV)",
            _fdf.to_csv(index=False),
            "flagged_duplicates.csv", "text/csv",
            key=f"dl_thr_flagged_{kp}",
        )


def _run_threshold_pass_multi(all_pairs, bands, n_per_bin, cols_by_pair,
                              gemini_key, target):
    """Multi-DB version: sample n_per_bin pairs per band PER comparison, send
    every pair to Gemini in a single call, then aggregate per-band per-pair
    plus an overall (pooled) duplicate ratio."""
    items, meta, gid = [], [], 0
    per_pair_binned = {}
    snap_cols_by_pair = {}  # per-pair snapshot columns (LLM cols + trace cols)
    for pid, res in all_pairs.items():
        prim = res['db_a']['df']
        corp = res['db_b']['df']
        cols_a, cols_b = cols_by_pair.get(pid, (res['db_a']['cols'], res['db_b']['cols']))
        trace_a = _resolve_trace_cols(prim.columns, cols_a)
        trace_b = _resolve_trace_cols(corp.columns, cols_b)
        snap_a = list(cols_a) + trace_a
        snap_b = list(cols_b) + trace_b
        snap_cols_by_pair[pid] = (snap_a, snap_b)
        rows, cols, scores = _collect_pairs(res['top_scores'], res['top_idx'])
        binned = _bin_pairs(rows, cols, scores, bands, n_per_bin,
                            seed=abs(hash(pid)) & 0xFFFF)
        per_pair_binned[pid] = binned
        for bi, b in enumerate(binned):
            for (ai, bj, sc) in b['pairs']:
                ta = _row_to_text(prim, ai, cols_a)
                tb = _row_to_text(corp, bj, cols_b)
                row_a = {c: prim.iloc[ai].get(c, '') for c in snap_a}
                row_b = {c: corp.iloc[bj].get(c, '') for c in snap_b}
                items.append({'id': gid, 'text_a': ta, 'text_b': tb})
                meta.append({
                    'id': gid, 'pid': pid, 'band_i': bi, 'ai': ai, 'bi': bj,
                    'score': sc, 'text_a': ta, 'text_b': tb,
                    'row_a': row_a, 'row_b': row_b,
                })
                gid += 1

    verdicts = _gemini_judge_pairs(items, gemini_key)

    band_results = []
    for bi, (lo, hi) in enumerate(bands):
        per_pair = {}
        for pid in all_pairs:
            rws = []
            for m in meta:
                if m['pid'] != pid or m['band_i'] != bi:
                    continue
                v = verdicts.get(m['id'], {'duplicate': False,
                                           'reason': '(no verdict returned)'})
                rws.append({**m, 'duplicate': bool(v['duplicate']),
                            'reason': v['reason']})
            n = len(rws)
            n_dup = sum(1 for r in rws if r['duplicate'])
            per_pair[pid] = {
                'n': n, 'n_dup': n_dup,
                'ratio': (n_dup / n) if n else 0.0,
                'n_available': per_pair_binned[pid][bi]['n_available'],
                'rows': rws,
            }
        all_rows = [r for p in per_pair.values() for r in p['rows']]
        n_total = sum(p['n'] for p in per_pair.values())
        n_dup_total = sum(p['n_dup'] for p in per_pair.values())
        band_results.append({
            'lo': lo, 'hi': hi,
            'label': f"{lo * 100:.0f}–{min(hi, 1.0) * 100:.0f}%",
            'per_pair': per_pair,
            'overall_n': n_total, 'overall_n_dup': n_dup_total,
            'overall_ratio': (n_dup_total / n_total) if n_total else 0.0,
            'rows': all_rows,
        })

    return {
        'bands': band_results, 'target': target,
        'pair_order': list(all_pairs.keys()),
        'pair_labels': {pid: f"{r['db_a']['name']} ↔ {r['db_b']['name']}"
                        for pid, r in all_pairs.items()},
        'cols_by_pair': snap_cols_by_pair,
        'db_names': {pid: (r['db_a']['name'], r['db_b']['name'])
                     for pid, r in all_pairs.items()},
        'n_judged': len(items), 'n_verdicts': len(verdicts),
    }


def _overall_transition_band(band_results, target):
    """Percent window (lo, hi) where overall (pooled) duplicates take over."""
    for b in band_results:
        if b['overall_n'] > 0 and b['overall_ratio'] >= target:
            return int(round(b['lo'] * 100)), int(round(b['hi'] * 100))
    populated = [b for b in band_results if b['overall_n'] > 0]
    if populated:
        best = max(populated, key=lambda b: b['overall_ratio'])
        return int(round(best['lo'] * 100)), int(round(best['hi'] * 100))
    return 85, 95


def _render_threshold_pass_multi(pass_data, title, key_prefix, all_pairs=None):
    """Grouped bars per band — one bar per DB comparison (PALETTE-colored,
    Sankey-style) plus a dotted Overall line. Returns the suggested cross-DB
    threshold percent (None if no band reaches target)."""
    import plotly.graph_objects as go
    PALETTE = ["#f97316", "#3b82f6", "#22c55e", "#a855f7", "#ec4899", "#14b8a6"]

    bands = pass_data['bands']
    target = pass_data['target']
    pair_order = pass_data['pair_order']
    pair_labels = pass_data['pair_labels']

    labels = [b['label'] for b in bands]
    fig = go.Figure()
    for idx, pid in enumerate(pair_order):
        ys = [round(b['per_pair'].get(pid, {}).get('ratio', 0.0) * 100, 1)
              for b in bands]
        texts = [
            (f"{b['per_pair'][pid]['n_dup']}/{b['per_pair'][pid]['n']}"
             if b['per_pair'].get(pid, {}).get('n', 0) else "")
            for b in bands
        ]
        fig.add_trace(go.Bar(
            x=labels, y=ys, name=pair_labels[pid],
            marker_color=PALETTE[idx % len(PALETTE)],
            text=texts, textposition="outside", textfont=dict(size=9),
        ))
    overall_y = [round(b['overall_ratio'] * 100, 1) for b in bands]
    overall_text = [f"{b['overall_n_dup']}/{b['overall_n']}" for b in bands]
    fig.add_trace(go.Scatter(
        x=labels, y=overall_y, name="Overall (pooled)",
        mode="lines+markers+text",
        line=dict(color="white", width=3, dash="dot"),
        marker=dict(size=11, color="white", line=dict(color="black", width=1)),
        text=overall_text, textposition="top center",
        textfont=dict(color="white", size=10),
    ))
    fig.add_hline(y=target * 100, line_dash="dash", line_color="red",
                  annotation_text=f"target {target:.0%}",
                  annotation_position="top left")
    fig.update_layout(
        title=title, barmode="group",
        xaxis_title="Cosine-similarity band",
        yaxis_title="Real-duplicate ratio (%)", yaxis_range=[0, 125],
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    if pass_data.get('n_verdicts', 0) < pass_data.get('n_judged', 0):
        st.warning(
            f"Gemini returned {pass_data['n_verdicts']} of "
            f"{pass_data['n_judged']} verdicts — missing pairs counted as "
            "non-duplicates. Re-run if needed."
        )

    overall_suggested = None
    for b in bands:
        if b['overall_n'] > 0 and b['overall_ratio'] >= target:
            overall_suggested = int(round(b['lo'] * 100))
            break
    if overall_suggested is not None:
        st.success(
            f"**Suggested cross-DB threshold ≈ {overall_suggested}%** — "
            f"first band whose pooled duplicate ratio reaches {target:.0%}."
        )
    else:
        st.warning(
            "No band's overall ratio reached the target — try lowering the "
            "target, raising the sample size, or inspecting the pairs below."
        )

    per_pair_lines = []
    for pid in pair_order:
        this_sug = None
        for b in bands:
            pp = b['per_pair'].get(pid, {})
            if pp.get('n', 0) > 0 and pp['ratio'] >= target:
                this_sug = int(round(b['lo'] * 100))
                break
        per_pair_lines.append(
            f"- **{pair_labels[pid]}**: "
            + (f"{this_sug}%" if this_sug is not None
               else "no band reached target")
        )
    if per_pair_lines:
        st.caption("Per-pair thresholds at the same target:")
        st.markdown("\n".join(per_pair_lines))

    with st.expander(f"🔍 Inspect the {sum(b['overall_n'] for b in bands)} "
                     "judged pairs"):
        insp = [
            {
                'pair': pair_labels[r['pid']],
                'band': b['label'],
                'similarity_pct': round(r['score'] * 100, 1),
                'is_duplicate': r['duplicate'],
                'A': r['text_a'].replace("\n", " | "),
                'B': r['text_b'].replace("\n", " | "),
                'gemini_reason': r['reason'],
            }
            for b in bands for r in b['rows']
        ]
        if insp:
            insp_df = pd.DataFrame(insp)
            st.dataframe(insp_df, use_container_width=True, hide_index=True)
            # All-rows flagged dataset (cross-DB): copy of the shared primary
            # DB with is_duplicate / similarity / source_db / reason filled
            # for each primary row that had at least one Gemini-confirmed
            # match against ANY secondary DB (best match across all DBs
            # wins on ties). Empty fields elsewhere.
            db_names = pass_data.get('db_names', {})
            best_per_a = {}  # ai -> (score, bi, pid, reason)
            for b in bands:
                for r in b['rows']:
                    if not r['duplicate']:
                        continue
                    a = int(r['ai'])
                    s = float(r['score'])
                    if a not in best_per_a or s > best_per_a[a][0]:
                        best_per_a[a] = (s, int(r['bi']), r['pid'],
                                         str(r.get('reason', '')))
            n_flagged = len(best_per_a)
            _jc, _dc = st.columns(2)
            with _jc:
                st.download_button(
                    "⬇️ Download judged pairs (CSV)", insp_df.to_csv(index=False),
                    f"{key_prefix}_judged_pairs.csv", "text/csv",
                    key=f"dl_{key_prefix}",
                )
            with _dc:
                if all_pairs is None:
                    st.caption(
                        "Primary DataFrame not available — flagged-rows "
                        "download disabled."
                    )
                elif n_flagged:
                    prim = next(iter(all_pairs.values()))['db_a']['df']
                    n_total = len(prim)
                    is_dup  = np.zeros(n_total, dtype=bool)
                    dup_sim = np.full(n_total, np.nan, dtype=float)
                    dup_row = np.full(n_total, np.nan, dtype=float)
                    dup_src = np.array([''] * n_total, dtype=object)
                    gem_rsn = np.array([''] * n_total, dtype=object)
                    for a, (s, b_idx, pid, reason) in best_per_a.items():
                        if 0 <= a < n_total:
                            is_dup[a]  = True
                            dup_sim[a] = round(s * 100, 1)
                            dup_row[a] = b_idx
                            dup_src[a] = db_names.get(pid, ('', ''))[1]
                            gem_rsn[a] = reason
                    out_df = prim.copy()
                    out_df['is_duplicate'] = is_dup
                    out_df['duplicate_similarity_pct'] = dup_sim
                    out_df['duplicate_source_db'] = dup_src
                    out_df['duplicate_match_row'] = dup_row
                    out_df['gemini_reason'] = gem_rsn
                    st.download_button(
                        f"⬇️ Download all rows w/ duplicates flagged "
                        f"({n_flagged:,}/{n_total:,}) (CSV)",
                        out_df.to_csv(index=False),
                        f"{key_prefix}_flagged_dataset.csv", "text/csv",
                        key=f"dl_{key_prefix}_dups",
                    )
                else:
                    st.caption("No `is_duplicate=True` rows to download.")
        else:
            st.caption("No pairs were sampled.")
    return overall_suggested


def _threshold_tab_all(all_pairs, gemini_key, pct=70):
    """Cross-DB threshold suggestion: samples each pair, judges every pair in
    one Gemini call, reports per-pair + overall (Sankey-style aggregation).

    `pct` is the outer 'Similarity threshold' slider value (percent). Pass 1
    bands start from it (clamped to 0–95 so at least one 5% band exists). The
    inline 'Flag duplicates' UI uses the raw `pct` so flagging matches the
    main view.
    """
    if not all_pairs:
        st.info("No pair results available.")
        return
    pct = int(pct)
    min_pct = max(0, min(pct, 95))
    pair_labels = {pid: f"{r['db_a']['name']} ↔ {r['db_b']['name']}"
                   for pid, r in all_pairs.items()}

    st.caption(
        "Samples matched pairs from **every** database comparison, asks Gemini "
        "to judge real duplicates in one call, then reports per-pair and "
        "overall duplicate ratios per band — pick a single threshold that "
        "works across all DBs."
    )

    cols_by_pair = {}
    with st.expander("⚙️ Configuration", expanded=False):
        st.caption("Pick which columns each side shows to the judge "
                   "(defaults to embedding columns).")
        for pid, res in all_pairs.items():
            kp = "_".join(str(x) for x in pid)
            st.markdown(f"**{pair_labels[pid]}**")
            cc1, cc2 = st.columns(2)
            with cc1:
                ca = st.multiselect(
                    f"A — {res['db_a']['name']}",
                    list(res['db_a']['df'].columns),
                    default=res['db_a']['cols'],
                    key=f"xdb_thrall_cols_a_{kp}",
                )
            with cc2:
                cb = st.multiselect(
                    f"B — {res['db_b']['name']}",
                    list(res['db_b']['df'].columns),
                    default=res['db_b']['cols'],
                    key=f"xdb_thrall_cols_b_{kp}",
                )
            cols_by_pair[pid] = (ca, cb)
        thr_gemini_key = gemini_key  # set at the top of the cross-DB section
        _ga, _gn, _gt = st.columns([1, 1, 1])
        with _ga:
            sample_all = st.toggle(
                "Sample ALL", value=False, key="xdb_thrall_all",
                help="Judge every pair in each band across all comparisons. "
                     "Cost scales with the data; if it exceeds Gemini's "
                     "1M-token window the request is split into multiple calls.",
            )
        with _gn:
            thr_n = st.slider("Samples per band, per pair", 3, 30, 10,
                              key="xdb_thrall_spb", disabled=sample_all)
        with _gt:
            thr_target = st.slider(
                "Target duplicate ratio", 0.0, 1.0, 0.5, 0.01,
                key="xdb_thrall_target",
                help="The lowest band whose OVERALL (pooled) duplicate ratio "
                     "reaches this value sets the suggested cross-DB threshold.",
            )

    _ready = bool(thr_gemini_key
                  and all(ca and cb for ca, cb in cols_by_pair.values()))
    if not _ready:
        st.info("Set a Gemini API key and at least one column per side in "
                "⚙️ Configuration to enable suggestion.")

    n_pairs = len(all_pairs)
    n_per_bin_eff = None if sample_all else thr_n
    _p1_bands = _make_bands(min_pct, 100, 5)
    _n_total, _in_tok, _out_tok, _n_calls = _estimate_judge_cost(
        all_pairs, cols_by_pair, _p1_bands, n_per_bin_eff,
    )
    _cost = _format_cost_line("Pass 1", _n_total, _in_tok, _out_tok, _n_calls)
    if _n_calls > 1:
        st.warning(_cost + " — exceeds Gemini's 1M-token window, will be split.")
    else:
        st.caption(_cost)

    if st.button(
        f"🎯 Suggest threshold across all DBs — pass 1 ({min_pct}→100%, 5% bands)",
        type="primary", disabled=not _ready, key="xdb_thrall_run1",
    ):
        try:
            _spin_msg = (
                f"Sampling ALL pairs/band across {n_pairs} comparison"
                f"{'s' if n_pairs != 1 else ''} and asking Gemini to judge…"
                if sample_all else
                f"Sampling {thr_n}/band per pair across {n_pairs} comparison"
                f"{'s' if n_pairs != 1 else ''} and asking Gemini to judge…"
            )
            with st.spinner(_spin_msg):
                _p1 = _run_threshold_pass_multi(
                    all_pairs, _p1_bands, n_per_bin_eff,
                    cols_by_pair, thr_gemini_key, thr_target,
                )
            st.session_state['xdb_thrall_pass1'] = _p1
            st.session_state.pop('xdb_thrall_pass2', None)
        except Exception as _e:
            st.error(f"Threshold suggestion failed: {_e}")

    _p1 = st.session_state.get('xdb_thrall_pass1')
    if _p1:
        st.markdown(f"#### Pass 1 — 5% bands ({min_pct}→100%), all databases")
        _render_threshold_pass_multi(
            _p1, "Real-duplicate ratio per 5% band — per pair + overall",
            "thrall_pass1", all_pairs=all_pairs,
        )

        st.divider()
        st.markdown("#### Pass 2 — refine at 1% resolution")
        _t_lo, _t_hi = _overall_transition_band(_p1['bands'], _p1['target'])
        st.caption(
            f"Proposed refinement window **{_t_lo}–{_t_hi}%** — the band where "
            "the *overall* duplicate ratio takes over."
        )
        _rc1, _rc2, _rc3 = st.columns([1, 1, 2])
        with _rc1:
            _r_lo = int(st.number_input("From %", 1, 99, _t_lo,
                                        key="xdb_thrall_rlo"))
        with _rc2:
            _r_hi = int(st.number_input("To %", _r_lo + 1, 100,
                                        max(_t_hi, _r_lo + 1),
                                        key="xdb_thrall_rhi"))
        with _rc3:
            _samples_label = "ALL" if sample_all else str(thr_n)
            st.caption(
                f"Samples {_samples_label} pairs per pair in each 1% band "
                f"from {_r_lo}% to {_r_hi}%."
            )

        _p2_bands = _make_bands(_r_lo, _r_hi, 1)
        _p2_n, _p2_in, _p2_out, _p2_calls = _estimate_judge_cost(
            all_pairs, cols_by_pair, _p2_bands, n_per_bin_eff,
        )
        _p2_cost = _format_cost_line("Pass 2", _p2_n, _p2_in, _p2_out, _p2_calls)
        if _p2_calls > 1:
            st.warning(_p2_cost + " — exceeds Gemini's 1M-token window, will be split.")
        else:
            st.caption(_p2_cost)

        if st.button(
            f"🔬 Run refined pass ({_r_lo}–{_r_hi}%, 1% bands, all DBs)",
            disabled=not _ready, key="xdb_thrall_run2",
        ):
            try:
                with st.spinner("Re-judging refined bands with Gemini…"):
                    _p2 = _run_threshold_pass_multi(
                        all_pairs, _p2_bands, n_per_bin_eff,
                        cols_by_pair, thr_gemini_key, thr_target,
                    )
                st.session_state['xdb_thrall_pass2'] = _p2
            except Exception as _e:
                st.error(f"Refined pass failed: {_e}")

        _p2 = st.session_state.get('xdb_thrall_pass2')
        if _p2:
            _sug2 = _render_threshold_pass_multi(
                _p2, "Real-duplicate ratio per 1% band — per pair + overall",
                "thrall_pass2", all_pairs=all_pairs,
            )
            if _sug2 is not None:
                st.info(
                    f"🎯 Refined cross-DB threshold **{_sug2}%** — set the "
                    "**Similarity threshold** slider above to this value to "
                    "apply it across all databases."
                )

    st.divider()
    st.markdown("**Flag duplicates across all databases**")
    _has_gemini = bool(st.session_state.get('xdb_thrall_pass2')
                       or st.session_state.get('xdb_thrall_pass1'))
    _fa1, _fa2 = st.columns([3, 1])
    with _fa1:
        flag_scope = st.radio(
            "Scope",
            [f"All matches above threshold ({pct}%)", "100% matches only"],
            key="xdb_thrall_flag_scope",
            horizontal=True,
            label_visibility="collapsed",
        )
    with _fa2:
        do_flag = st.button(
            "🚩 Flag as duplicates", key="xdb_thrall_flag_btn",
            use_container_width=True,
        )
    _prim_ref = next(iter(all_pairs.values()))['db_a']
    _prim_name = _prim_ref['name']
    if _has_gemini and flag_scope != "100% matches only":
        st.caption(
            f"🤖 Gemini verdicts available — flagging will use **only the "
            f"Gemini-confirmed duplicates** from the latest pass, not every "
            f"pair above {pct}%."
        )
    else:
        st.caption(
            f"Marks **{_prim_name}** rows that match any of "
            f"{len(all_pairs)} secondary DB(s). Each flagged row keeps its single "
            "best match across all comparisons; `duplicate_source_db` records "
            "which DB it came from."
        )
    if do_flag:
        _exact = flag_scope == "100% matches only"
        _thr = 1.0 if _exact else pct / 100.0
        # Pull Gemini verdicts from the latest multi-DB pass (pass 2 wins if
        # both exist). Keyed by (pid, ai, bi) since the same primary row can
        # match different DBs.
        _reason_lookup = {}
        for _pk in ('xdb_thrall_pass2', 'xdb_thrall_pass1'):
            _pass = st.session_state.get(_pk)
            if not _pass:
                continue
            for _band in _pass['bands']:
                for _r in _band['rows']:
                    _reason_lookup[(_r['pid'], int(_r['ai']), int(_r['bi']))] = {
                        'duplicate': bool(_r['duplicate']),
                        'reason': str(_r.get('reason', '')),
                        'score': float(_r['score']),
                    }
            if _reason_lookup:
                break

        _use_gemini = bool(_reason_lookup) and not _exact

        _n = len(_prim_ref['df'])
        _is_dup  = np.zeros(_n, dtype=bool)
        _dup_sim = np.full(_n, np.nan, dtype=float)
        _dup_row = np.full(_n, np.nan, dtype=float)
        _dup_src = np.array([''] * _n, dtype=object)
        _gem_verdict = np.array([''] * _n, dtype=object)
        _gem_reason  = np.array([''] * _n, dtype=object)
        _best = {}  # a_idx -> (score, b_idx, pid)
        if _use_gemini:
            # Trust Gemini: only consider judged pairs it confirmed as
            # duplicates, regardless of similarity threshold.
            for (pid, a, b), v in _reason_lookup.items():
                if not v['duplicate']:
                    continue
                s = v['score']
                if a not in _best or s > _best[a][0]:
                    _best[a] = (s, b, pid)
        else:
            for pid, res in all_pairs.items():
                _ts, _ti = res['top_scores'], res['top_idx']
                for i in range(len(_ts)):
                    for j in range(_ts.shape[1]):
                        s = float(_ts[i, j])
                        if not np.isfinite(s) or s < _thr:
                            continue
                        if i not in _best or s > _best[i][0]:
                            _best[i] = (s, int(_ti[i, j]), pid)
        for a, (s, b, pid) in _best.items():
            _is_dup[a]  = True
            _dup_sim[a] = round(s * 100, 1)
            _dup_row[a] = b
            _dup_src[a] = all_pairs[pid]['db_b']['name']
            _v = _reason_lookup.get((pid, a, b))
            if _v is not None:
                _gem_verdict[a] = 'duplicate' if _v['duplicate'] else 'not_duplicate'
                _gem_reason[a]  = _v['reason']
        _flagged_df = _prim_ref['df'].copy()
        _flagged_df['is_duplicate'] = _is_dup
        _flagged_df['duplicate_similarity_pct'] = _dup_sim
        _flagged_df['duplicate_source_db'] = _dup_src
        _flagged_df['duplicate_match_row'] = _dup_row
        if _reason_lookup:
            _flagged_df['gemini_verdict'] = _gem_verdict
            _flagged_df['gemini_reason']  = _gem_reason
        if _use_gemini:
            _scope_desc = "Gemini-confirmed duplicates"
        elif _exact:
            _scope_desc = "100% matches"
        else:
            _scope_desc = f"≥{pct}% matches"
        st.session_state['xdb_thrall_flagged'] = {
            'df': _flagged_df,
            'caption': (
                f"Flagged **{int(_is_dup.sum()):,}** of {_n:,} "
                f"{_prim_name} rows as duplicates across "
                f"{len(all_pairs)} DB(s) ({_scope_desc})."
            ),
        }
    _flagged = st.session_state.get('xdb_thrall_flagged')
    if _flagged:
        st.success(_flagged['caption'])
        _fdf = _flagged['df']
        _dups_only = _fdf[_fdf['is_duplicate']]
        st.caption(f"Preview — {len(_dups_only):,} flagged row(s):")
        st.dataframe(_dups_only, use_container_width=True)
        st.download_button(
            "⬇️ Download flagged dataset (CSV)",
            _fdf.to_csv(index=False),
            "flagged_duplicates_all.csv", "text/csv",
            key="dl_thrall_flagged",
        )


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
if 'json_format' not in st.session_state:
    st.session_state['json_format'] = None
if 'api_key_valid' not in st.session_state:
    st.session_state['api_key_valid'] = False
if 'previous_api_key' not in st.session_state:
    st.session_state['previous_api_key'] = None

OPENAI_KEY = st.secrets["OPENAI_KEY"]
GEMINI_KEY = st.secrets["GEMINI_KEY"]
COHERE_KEY = st.secrets["COHERE_KEY"]

def load_data(file_path, key='df'):
    return pd.read_hdf(file_path, key=key)


@st.cache_resource(show_spinner="Loading local embedding model...")
def load_local_embedding_model(model_name: str):
    """Load a sentence-transformers model once and cache it across reruns."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


HARRIER_MODEL = "microsoft/harrier-oss-v1-270m"


def _hash_corpus(json_documents, model_name: str) -> str:
    """Build a stable hash key for the corpus + model so we cache embeddings correctly."""
    import hashlib
    h = hashlib.md5()
    h.update(model_name.encode("utf-8"))
    for doc in json_documents:
        h.update(doc.encode("utf-8"))
    return h.hexdigest()


def _encode_with_progress(texts: list[str], encode_fn, batch_size: int = 256, label: str = "Encoding") -> np.ndarray:
    """Encode texts in batches while updating a Streamlit progress bar."""
    n = len(texts)
    bar = st.progress(0.0, text=f"{label} — 0 / {n}")
    all_embs = []
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        embs = np.array(encode_fn(batch), dtype=np.float32)
        all_embs.append(embs)
        done = min(start + batch_size, n)
        bar.progress(done / n, text=f"{label} — {done} / {n}")
    bar.empty()
    return np.vstack(all_embs)


# ──────────────────────────────────────────────────────────────────────────
# Multi-Modal Search — semantic search over the UAP Release 2 archive
# (DoD UAP video clips, NASA Apollo/Mercury audio, source-document PDF pages)
# embedded with Gemini 768-d into Neon + pgvector.
# See STREAMLIT_HANDOFF.md for the full ingest pipeline.
# ──────────────────────────────────────────────────────────────────────────
_MM_USER_ID = "00000000-0000-0000-0000-000000000001"
_MM_SOURCE_TYPES = ("video_chunk", "audio_clip", "pdf_page")
_MM_PAGE_RE = __import__("re").compile(r"^(.+):p(\d+)$")


@st.cache_resource(show_spinner=False)
def _mm_neon_conn(db_url: str):
    """One pooled Neon connection per process — handed back to every query.
    `prepare_threshold=None` keeps PgBouncer transaction-pool mode happy;
    `autocommit=True` prevents idle-in-transaction timeouts on Neon — psycopg3
    otherwise auto-opens a transaction on the first SELECT and never closes it
    between Streamlit reruns, which Neon kills after a few idle minutes."""
    import psycopg
    return psycopg.connect(db_url, prepare_threshold=None, autocommit=True)


def _mm_reset_neon_conn(db_url: str):
    """Drop the cached connection (e.g. after a stale-connection error) so the
    next _mm_neon_conn() call dials a fresh one."""
    try:
        _mm_neon_conn.clear()
    except Exception:
        pass


@st.cache_data(ttl=3600, show_spinner=False)
def _mm_embed_text_gemini(text: str, gemini_key: str) -> list[float]:
    """Embed a text query with gemini-embedding-2-preview (768-d).
    Query-side asymmetric wrapping per STREAMLIT_HANDOFF.md §6."""
    from google import genai
    from google.genai import types as gt
    client = genai.Client(api_key=gemini_key)
    r = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=f"task: search result | query: {text}",
        config=gt.EmbedContentConfig(output_dimensionality=768),
    )
    return list(r.embeddings[0].values)


@st.cache_data(ttl=3600, show_spinner=False)
def _mm_embed_image_gemini(image_bytes: bytes, mime: str, gemini_key: str) -> list[float]:
    """Embed an image query with gemini-embedding-2-preview (768-d)."""
    from google import genai
    from google.genai import types as gt
    client = genai.Client(api_key=gemini_key)
    r = client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=[gt.Part.from_bytes(data=image_bytes, mime_type=mime)],
        config=gt.EmbedContentConfig(output_dimensionality=768),
    )
    return list(r.embeddings[0].values)


def _mm_search_pgvector(db_url, vec, *, user_id, source_type=None,
                        release=None, limit=20, threshold=0.30):
    """Cosine-similarity search over the embeddings table. Serialises the
    vector as text-form '[a,b,…]' and casts in SQL — psycopg3's adapter does
    not auto-cast list[float] to pgvector. See handoff §11."""
    vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
    clauses = ["user_id = %s::uuid", "(embedding <=> %s::vector) <= %s"]
    params: list = [user_id, vec_str, 1 - threshold]
    if source_type:
        clauses.append("source_type = %s")
        params.append(source_type)
    if release:
        clauses.append("release = %s")
        params.append(release)
    sql = f"""
        SELECT source_type, source_id, parent_id, start_seconds, end_seconds,
               embedded_image_url, embedded_text, release, release_date,
               1 - (embedding <=> %s::vector) AS similarity
        FROM embeddings
        WHERE {' AND '.join(clauses)}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    ordered = [vec_str, *params, vec_str, limit]

    import psycopg
    # One-shot retry: if the cached Neon connection was killed while idle
    # (idle-in-transaction timeout, network blip, Neon auto-suspend), drop it
    # and dial a fresh one before giving up.
    for attempt in (1, 2):
        try:
            conn = _mm_neon_conn(db_url)
            with conn.cursor() as cur:
                cur.execute(sql, ordered)
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
        except (psycopg.OperationalError, psycopg.InterfaceError):
            if attempt == 1:
                _mm_reset_neon_conn(db_url)
                continue
            raise


def _mm_page_number(source_id: str):
    """Extract the page index from a pdf_page source_id like 'slug:p0017'."""
    m = _MM_PAGE_RE.match(source_id or "")
    return int(m.group(2)) if m else None


def _mm_resolve_local(media_root: str, *parts: str):
    """Return the first existing file under `media_root / *parts`, trying both
    the path as-given AND a WSL-translated variant of any 'D:\\…' Windows path.
    Returns None if nothing resolves."""
    from pathlib import Path
    if not media_root:
        return None
    raw = media_root.strip().strip('"').strip("'")
    candidates: list[str] = [str(Path(raw, *parts))]
    if len(raw) >= 2 and raw[1] == ":":
        drive, rest = raw[0].lower(), raw[2:].replace("\\", "/").lstrip("/")
        candidates.append(str(Path(f"/mnt/{drive}", rest, *parts)))
    for p in candidates:
        try:
            if Path(p).is_file():
                return p
        except OSError:
            continue
    return None


def _mm_local_video_mp4(media_root: str, asset_id: str):
    """Resolve the local 49-file DVIDS video cache (handoff §9):
    `{media_root}/videos/dvids_{id}.mp4`."""
    if not asset_id:
        return None
    return _mm_resolve_local(media_root, "videos", f"dvids_{asset_id}.mp4")


def _mm_local_audio(media_root: str, asset_id: str):
    """Resolve the local DVIDS audio cache (handoff §9). Prefers the small
    extracted `.m4a` over the multi-hundred-MB source `.mp4` wrapper."""
    if not asset_id:
        return None
    for ext in ("m4a", "mp3", "mp4", "wav", "aac", "ogg"):
        p = _mm_resolve_local(media_root, "audio", f"dvids_{asset_id}.{ext}")
        if p:
            return p
    return None


def _mm_local_page_png(media_root: str, slug: str, page_num: int):
    """Resolve a per-page PNG render (handoff §9):
    `{media_root}/pages/{slug}/page_NNNN.png` (150 dpi)."""
    if not slug or page_num is None:
        return None
    return _mm_resolve_local(
        media_root, "pages", slug, f"page_{int(page_num):04d}.png"
    )


def _render_multimodal_search():
    """Multi-Modal Search tab: query the Neon + pgvector archive with text or
    an image, render DVIDS iframes for video/audio results and link out to
    war.gov for PDF pages."""
    st.markdown("### 🎬 Multi-Modal Semantic Search")
    st.caption(
        "Searches the **UAP Release 2 (2026-05-22)** archive — 49 DoD UAP "
        "video clips, 7 NASA Apollo/Mercury audio recordings, and 5 source "
        "documents — embedded with Gemini 768-d into Neon + pgvector. "
        "Backed by the pipeline documented in `STREAMLIT_HANDOFF.md`."
    )

    # ── Dependency check ─────────────────────────────────────────────────
    try:
        import psycopg  # noqa: F401
    except ImportError:
        st.error(
            "`psycopg` is not installed. Run "
            "`pip install \"psycopg[binary]\" pgvector google-genai` "
            "and restart Streamlit."
        )
        return
    try:
        from google import genai  # noqa: F401
    except ImportError:
        st.error(
            "`google-genai` is not installed. Run `pip install google-genai`."
        )
        return

    # ── Connection / API key ────────────────────────────────────────────
    _sec_db = st.secrets.get("DATABASE_URL", "") if hasattr(st, "secrets") else ""
    _sec_gem = (
        st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    ) or GEMINI_KEY
    _sec_root = (
        st.secrets.get("UAP_MEDIA_ROOT", "") if hasattr(st, "secrets") else ""
    ) or r"D:\divided\release_2\UAP_Release_2"
    with st.expander("🔑 Connection & API key", expanded=not (_sec_db and _sec_gem)):
        c_db, c_key = st.columns(2)
        with c_db:
            db_url = st.text_input(
                "Neon DATABASE_URL",
                value=_sec_db,
                type="password",
                placeholder="postgresql://USER:PASS@ep-xxxx.REGION.aws.neon.tech/neondb?sslmode=require",
                key="mm_db_url",
                help="Use the DIRECT Neon endpoint (no `-pooler`) for long-lived Streamlit sessions.",
            )
        with c_key:
            gemini_key = st.text_input(
                "Gemini API key",
                value=_sec_gem,
                type="password",
                key="mm_gem_key",
                help="Must be the same model family used during embedding (gemini-embedding-2-preview).",
            )
        media_root = st.text_input(
            "Media root (for PDF page thumbnails)",
            value=_sec_root,
            key="mm_media_root",
            help=(
                "Local directory containing `pages/{slug}/page_NNNN.png` from "
                "the ingest pipeline (handoff §9). Used for PDF-page "
                "thumbnails. Windows paths like `D:\\divided\\…` are also "
                "tried as WSL-translated `/mnt/d/divided/…` so the same "
                "setting works on either side. Leave blank to disable "
                "thumbnails and use a war.gov link button instead."
            ),
        )
    if not db_url or not gemini_key:
        st.info(
            "Paste a `DATABASE_URL` and a Gemini API key above to enable "
            "search. Both can also be set in `.streamlit/secrets.toml` "
            "(`DATABASE_URL` and `GEMINI_API_KEY`)."
        )
        return

    # ── Search options ──────────────────────────────────────────────────
    with st.expander("⚙️ Search options", expanded=True):
        opt1, opt2, opt3, opt4 = st.columns(4)
        with opt1:
            mode = st.radio("Query type", ["Text", "Image"], horizontal=True, key="mm_mode")
        with opt2:
            stype_filter = st.selectbox("Source type", ["all", *_MM_SOURCE_TYPES], key="mm_stype")
        with opt3:
            release_filter = st.selectbox("Release", ["all", "PURSUE_2"], key="mm_rel")
        with opt4:
            limit = st.slider("Max results", 5, 50, 12, key="mm_limit")
        threshold = st.slider("Min similarity", 0.0, 0.9, 0.20, 0.05, key="mm_thr")

    # ── Query input ─────────────────────────────────────────────────────
    vec = None
    if mode == "Text":
        q = st.text_input(
            "Search query",
            placeholder="e.g. spherical UAP over water, high-speed manoeuvre, Apollo astronaut",
            key="mm_q",
        )
        if q and st.button("🔍 Search", type="primary", key="mm_btn_text"):
            try:
                with st.spinner("Embedding query with Gemini…"):
                    vec = _mm_embed_text_gemini(q, gemini_key)
            except Exception as e:
                st.error(f"Embedding failed: {e}")
                return
    else:
        up = st.file_uploader(
            "Upload an image to search by",
            type=["jpg", "jpeg", "png", "webp"],
            key="mm_img",
        )
        if up is not None:
            st.image(up, width=240)
            if st.button("🔍 Search", type="primary", key="mm_btn_img"):
                try:
                    with st.spinner("Embedding image with Gemini…"):
                        vec = _mm_embed_image_gemini(up.getvalue(), up.type, gemini_key)
                except Exception as e:
                    st.error(f"Embedding failed: {e}")
                    return

    if vec is None:
        return

    # ── Search Neon ─────────────────────────────────────────────────────
    try:
        with st.spinner("Searching Neon…"):
            rows = _mm_search_pgvector(
                db_url, vec,
                user_id=_MM_USER_ID,
                source_type=None if stype_filter == "all" else stype_filter,
                release=None if release_filter == "all" else release_filter,
                limit=limit, threshold=threshold,
            )
    except Exception as e:
        st.error(
            f"Search failed: {e}\n\n"
            "Common causes: wrong Neon URL, IP not allow-listed, "
            "or schema missing (see `STREAMLIT_HANDOFF.md` §3)."
        )
        return

    if not rows:
        st.warning(
            "No matches above the similarity threshold. Try lowering it "
            "(text queries can need 0.05–0.20 to surface video chunks)."
        )
        return

    # ── Result rendering ────────────────────────────────────────────────
    st.subheader(f"{len(rows)} result(s)")
    for r in rows:
        with st.container(border=True):
            head_l, head_r = st.columns([4, 1])
            with head_l:
                parent = r.get("parent_id") or ""
                stype = r.get("source_type") or ""
                page = _mm_page_number(r.get("source_id") or "") if stype == "pdf_page" else None
                src_url = r.get("embedded_image_url") or ""
                title = f"**`{parent}`** · `{stype}` · sim **{r['similarity']:.3f}**"
                if page is not None:
                    title += f" · page {page}"
                elif r.get("start_seconds") is not None:
                    title += f" · {float(r['start_seconds']):.1f}s → {float(r['end_seconds']):.1f}s"
                st.markdown(title)
                if r.get("embedded_text"):
                    txt = r["embedded_text"]
                    st.caption(txt[:600] + ("…" if len(txt) > 600 else ""))

                # ── Embedded media ──
                # Prefer the local DVIDS cache (handoff §9): st.video/st.audio
                # supports precise start/end seeking. Fall back to the DVIDS
                # iframe when the local file isn't reachable — DVIDS' JW Player
                # ignores `#t=` and `?t=` on the embed URL, so the iframe
                # always plays from 0 and the chunk range stays visible in the
                # header so users can seek manually.
                start_s = r.get("start_seconds")
                end_s = r.get("end_seconds")
                start_i = int(start_s) if start_s is not None else 0
                end_i = int(end_s) if end_s is not None else None
                page_t = f"?t={start_i}" if start_s is not None else ""
                if stype == "video_chunk":
                    asset_id = parent.removeprefix("dvids_")
                    local = _mm_local_video_mp4(media_root, asset_id)
                    if local:
                        st.video(local, start_time=start_i, end_time=end_i)
                    else:
                        components.iframe(
                            f"https://www.dvidshub.net/video/embed/{asset_id}",
                            height=400, scrolling=False,
                        )
                        if start_s is not None:
                            st.caption(
                                f"_(DVIDS embed ignores timestamps — chunk "
                                f"is **{start_s:.1f}s → {float(end_s):.1f}s**; "
                                "set **Media root** to the local DVIDS cache "
                                "for precise seek.)_"
                            )
                elif stype == "audio_clip":
                    asset_id = parent.removeprefix("dvids_")
                    local = _mm_local_audio(media_root, asset_id)
                    if local:
                        st.audio(local, start_time=start_i, end_time=end_i)
                    else:
                        components.iframe(
                            f"https://www.dvidshub.net/audio/embed/{asset_id}",
                            height=180, scrolling=False,
                        )
                        if start_s is not None:
                            st.caption(
                                f"_(DVIDS embed ignores timestamps — chunk "
                                f"is **{start_s:.1f}s → {float(end_s):.1f}s**; "
                                "set **Media root** to the local DVIDS cache "
                                "for precise seek.)_"
                            )
                elif stype == "pdf_page":
                    # Try to show the local 150-dpi PNG render (handoff §9);
                    # fall back to a link button if it can't be reached.
                    thumb = _mm_local_page_png(media_root, parent, page) if page is not None else None
                    if thumb:
                        st.image(
                            thumb,
                            caption=f"{parent} · page {page}",
                            use_container_width=True,
                        )
                        if src_url:
                            st.link_button(
                                f"📄 Open full PDF on war.gov (page {page})",
                                src_url,
                            )
                    elif src_url:
                        st.link_button(
                            f"📄 Open full PDF on war.gov (page {page if page is not None else '?'})",
                            src_url,
                        )
                        st.caption(
                            "_(thumbnail unavailable — set **Media root** in "
                            "the 🔑 expander to the folder containing "
                            "`pages/{slug}/page_NNNN.png`)_"
                        )

                if src_url:
                    deep = src_url + page_t if stype in ("video_chunk", "audio_clip") else src_url
                    st.markdown(f"[Open source page]({deep})")
            with head_r:
                st.metric("similarity", f"{r['similarity']:.3f}")
                st.caption(
                    f"{r.get('release', '') or ''} · {r.get('release_date', '') or ''}"
                )


# ──────────────────────────────────────────────────────────────────────────
# Top-level tabs: the existing Cohere/Cross-DB flow + the new Multi-Modal
# Search. Multi-Modal Search is independent of any local CSV upload — it
# queries the Neon + pgvector archive directly.
# ──────────────────────────────────────────────────────────────────────────
tab_rag, tab_mm = st.tabs([
    "📊 Dataset RAG (Cohere + Cross-DB)",
    "🎬 Multi-Modal Search (Neon + Gemini)",
])

with tab_mm:
    _render_multimodal_search()

with tab_rag:
    datasett = st.file_uploader("Upload Raw DataFrame", type=["csv", "xlsx"])
    if datasett is not None:
        try:
            data = pd.read_csv(datasett) if datasett.type == "text/csv" else pd.read_excel(datasett)
            filtered_data = filter_dataframe(data)
            st.session_state['parsed_responses'] = filtered_data
            st.dataframe(filtered_data)
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

        st.divider()

        cross_db = st.toggle(
            "Cross-DB Similarity Search",
            value=False,
            help="OFF — RAG search within this dataset via Cohere rerank. ON — cosine similarity against a second dataset using local embeddings.",
        )

        if not cross_db:
            # ── RAG mode: Cohere rerank, same dataset ──────────────────────────────
            col1, col2 = st.columns(2)
            with col1:
                columns_to_query = st.multiselect(
                    "Select columns to search",
                    options=list(st.session_state['parsed_responses'].columns),
                )
            with col2:
                COHERE_KEY = st.text_input("Cohere API Key", COHERE_KEY, type="password")

            question = st.text_input("Ask a question")

            if columns_to_query and question and COHERE_KEY:
                try:
                    co = cohere.Client(api_key=COHERE_KEY)
                    json_documents = [
                        json.dumps(doc, default=str)
                        for doc in st.session_state['parsed_responses'][columns_to_query].to_dict("records")
                    ]
                    results = co.rerank(
                        model="rerank-english-v3.0",
                        query=question,
                        documents=json_documents,
                        top_n=50,
                        return_documents=True,
                    )
                    reranked_indices = [r.index for r in results.results]
                    reranked_scores  = [r.relevance_score for r in results.results]

                    st.subheader("Reranked Results:")
                    reranked_df = st.session_state['parsed_responses'].iloc[reranked_indices].copy()
                    reranked_df['relevance_score'] = reranked_scores
                    reranked_df['rank'] = range(1, len(reranked_indices) + 1)
                    reranked_df.set_index('rank', inplace=True)
                    st.dataframe(reranked_df)
                except Exception as e:
                    st.error(f"An error occurred during reranking: {e}")

            # ── Gemini Q&A (moved from analyzing.py) ───────────────────────────────
            st.divider()
            st.markdown("#### 🔮 Gemini Q&A")
            st.caption(
                "Ask Gemini a question over one column's values, or leave the "
                "question empty to get a bullet-point summary."
            )
            gq1, gq2 = st.columns(2)
            with gq1:
                gemini_col = st.selectbox(
                    "Which column do you want to query?",
                    list(st.session_state['parsed_responses'].columns),
                    key="gemini_qa_col",
                )
            with gq2:
                gemini_key_input = st.text_input(
                    "Gemini API Key", value=GEMINI_KEY, type="password",
                    help="Defaults to GEMINI_KEY in secrets.", key="gemini_qa_key",
                )
            if gemini_col and gemini_key_input:
                gemini_question = st.text_input(
                    "Ask a question or leave empty for summarization",
                    key="gemini_qa_question",
                )
                selected_column_data = st.session_state['parsed_responses'][gemini_col].tolist()
                if st.button("Generate Query", key="gemini_qa_run") and selected_column_data:
                    with st.status("Generating answer with Gemini…", expanded=True):
                        st.write(gemini_query(gemini_question, selected_column_data, gemini_key_input))

        else:
            # ── Multi-DB Cross Similarity ─────────────────────────────────────────
            import plotly.graph_objects as go
            from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
            from collections import Counter
            import itertools, uuid as _uuid

            # ── Model + Gemini key ────────────────────────────────────────────────
            _mc1, _mc2 = st.columns([2, 3])
            with _mc1:
                local_model_name = st.selectbox(
                    "Embedding model",
                    options=[
                        "microsoft/harrier-oss-v1-270m",
                        "sentence-transformers/all-MiniLM-L6-v2",
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    ],
                )
            with _mc2:
                GEMINI_KEY = st.text_input(
                    "Gemini API key",
                    GEMINI_KEY,
                    type="password",
                    help="Used by the 🎯 Suggest Threshold tab to judge real "
                         "duplicates. Defaults to `GEMINI_KEY` in "
                         "`.streamlit/secrets.toml`; override here to use your own.",
                    key="xdb_gemini_key",
                )

            # ── Database list ─────────────────────────────────────────────────────
            primary_df = st.session_state['parsed_responses']

            if 'xdb_extra_ids' not in st.session_state:
                st.session_state['xdb_extra_ids'] = []
            if 'xdb_slots' not in st.session_state:
                st.session_state['xdb_slots'] = []

            # Primary DB row
            with st.container(border=True):
                pc1, pc2 = st.columns([1, 3])
                with pc1:
                    db1_name = st.text_input("Name", "DB1", key="xdb_name_primary")
                    st.caption(f"{primary_df.shape[0]:,} rows × {primary_df.shape[1]} cols")
                with pc2:
                    db1_cols = st.multiselect(
                        "Embedding columns",
                        options=list(primary_df.columns),
                        key="xdb_cols_primary",
                    )
                st.dataframe(primary_df.head(5), use_container_width=True, hide_index=True)

            # Extra DB rows (dynamically added)
            _remove_id = None
            extra_db_configs = []
            for db_id in st.session_state['xdb_extra_ids']:
                with st.container(border=True):
                    hdr, rm_col = st.columns([11, 1])
                    with rm_col:
                        if st.button("✕", key=f"xdb_rm_{db_id}"):
                            _remove_id = db_id
                    ec1, ec2, ec3 = st.columns([1, 2, 3])
                    with ec1:
                        db_name = st.text_input("Name", f"DB{len(extra_db_configs)+2}", key=f"xdb_name_{db_id}")
                    with ec2:
                        _src = st.radio(
                            "Source",
                            ["Upload file", "Loaded dataset (DB1)"],
                            key=f"xdb_src_{db_id}",
                            horizontal=True,
                            help="'Loaded dataset (DB1)' reuses the dataset loaded at the top of "
                                 "the page — e.g. to find near-duplicate rows within it. Its "
                                 "embeddings are reused, not recomputed.",
                        )
                        if _src == "Loaded dataset (DB1)":
                            _edf = primary_df
                            st.caption(
                                f"{primary_df.shape[0]:,} rows × {primary_df.shape[1]} cols "
                                "— same as DB1"
                            )
                        else:
                            up = st.file_uploader("File", type=["csv", "xlsx"], key=f"xdb_file_{db_id}")
                            if up is not None:
                                try:
                                    _df = pd.read_csv(up) if up.type == "text/csv" else pd.read_excel(up)
                                    st.session_state[f'xdb_df_{db_id}'] = _df
                                except Exception as _e:
                                    st.error(f"Error: {_e}")
                            _edf = st.session_state.get(f'xdb_df_{db_id}')
                            if _edf is not None:
                                st.caption(f"{_edf.shape[0]:,} rows × {_edf.shape[1]} cols")
                    with ec3:
                        _ecols = st.multiselect(
                            "Embedding columns",
                            options=list(_edf.columns) if _edf is not None else [],
                            key=f"xdb_cols_{db_id}",
                            disabled=_edf is None,
                        )
                    if _edf is not None:
                        st.dataframe(_edf.head(5), use_container_width=True, hide_index=True)
                    extra_db_configs.append({'id': db_id, 'name': db_name, 'df': _edf, 'cols': _ecols})

            if _remove_id:
                st.session_state['xdb_extra_ids'].remove(_remove_id)
                st.session_state.pop(f'xdb_df_{_remove_id}', None)
                st.rerun()

            if st.button("＋ Add database"):
                st.session_state['xdb_extra_ids'].append(str(_uuid.uuid4())[:8])
                st.rerun()

            # Collect all fully loaded DBs
            all_dbs = [{'id': 'primary', 'name': db1_name, 'df': primary_df, 'cols': db1_cols}] + \
                      [db for db in extra_db_configs if db['df'] is not None]
            ready_dbs = [db for db in all_dbs if db['cols']]

            if len(ready_dbs) < 2:
                st.info("Configure embedding columns for at least 2 databases above.")
            else:
                # ── Column mapping (slots) ─────────────────────────────────────────
                with st.expander("⚙️ Embedding column mapping", expanded=False):
                    st.caption(
                        "Define named slots to align heterogeneous schemas across databases. "
                        "Each slot maps one column per database to a shared semantic field. "
                        "Leave a cell as **—** if that database has no equivalent column. "
                        "If no slots are defined, each database embeds its own selected columns independently."
                    )
                    st.divider()

                    # Header row
                    n_dbs = len(ready_dbs)
                    hcols = st.columns([2] + [3] * n_dbs + [1])
                    hcols[0].markdown("**Slot name**")
                    for _j, _db in enumerate(ready_dbs):
                        hcols[_j + 1].markdown(f"**{_db['name']}**")

                    _rm_slots = []
                    for _si, slot in enumerate(st.session_state['xdb_slots']):
                        scols = st.columns([2] + [3] * n_dbs + [1])
                        with scols[0]:
                            slot['name'] = st.text_input(
                                "", slot['name'], key=f"sln_{_si}",
                                label_visibility="collapsed",
                            )
                        for _j, _db in enumerate(ready_dbs):
                            with scols[_j + 1]:
                                _opts = ['—'] + list(_db['df'].columns)
                                _cur  = slot['mapping'].get(_db['id'], '—')
                                _idx  = _opts.index(_cur) if _cur in _opts else 0
                                _sel  = st.selectbox(
                                    "", _opts, index=_idx,
                                    key=f"slm_{_si}_{_db['id']}",
                                    label_visibility="collapsed",
                                )
                                slot['mapping'][_db['id']] = _sel if _sel != '—' else None
                        with scols[-1]:
                            if st.button("✕", key=f"sl_rm_{_si}"):
                                _rm_slots.append(_si)

                    for _i in sorted(_rm_slots, reverse=True):
                        st.session_state['xdb_slots'].pop(_i)
                    if _rm_slots:
                        st.rerun()

                    if st.button("＋ Add slot", key="add_slot"):
                        st.session_state['xdb_slots'].append(
                            {'name': f'slot_{len(st.session_state["xdb_slots"])}', 'mapping': {}}
                        )
                        st.rerun()

                # ── Compute ───────────────────────────────────────────────────────
                primary_ready = ready_dbs[0]
                extra_ready   = ready_dbs[1:]
                n_pairs = len(extra_ready)
                compute_btn = st.button(
                    f"⚡ Compare {primary_ready['name']} against {n_pairs} database{'s' if n_pairs != 1 else ''}",
                    type="primary",
                )

                def _make_texts(db, slots):
                    if slots:
                        rows = []
                        for _, row in db['df'].iterrows():
                            parts = [
                                f"{s['name']}: {row[col]}"
                                for s in slots
                                for col in [s['mapping'].get(db['id'])]
                                if col and col in row.index and pd.notna(row[col])
                            ]
                            rows.append(" | ".join(parts) if parts else "")
                        return rows
                    return [json.dumps(r, default=str) for r in db['df'][db['cols']].to_dict("records")]

                if compute_btn:
                    is_harrier = (local_model_name == HARRIER_MODEL)
                    if is_harrier:
                        _m = get_embed_model()
                        _enc = lambda t: _m.encode_document(t, batch_size=256, show_progress_bar=False)
                    else:
                        _m = load_local_embedding_model(local_model_name)
                        _kw = dict(convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                        _enc = lambda t: _m.encode(t, **_kw)

                    slots = st.session_state['xdb_slots']
                    # Encode each DB once. Databases whose embedding text is identical
                    # — e.g. the loaded dataset reused as another DB with the same
                    # columns — share embeddings instead of being re-encoded.
                    dbs_to_encode = [primary_ready] + extra_ready
                    db_embs = {}
                    db_text_key = {}
                    _text_cache = {}
                    for db in dbs_to_encode:
                        texts = _make_texts(db, slots)
                        _key = hash(tuple(texts))
                        db_text_key[db['id']] = _key
                        if _key in _text_cache:
                            db_embs[db['id']] = _text_cache[_key]
                            st.caption(f"↳ {db['name']}: reused embeddings (identical text to an already-encoded database)")
                        else:
                            emb = _encode_with_progress(texts, _enc, label=f"Encoding {db['name']}")
                            _text_cache[_key] = emb
                            db_embs[db['id']] = emb

                    K, CHUNK = 20, 500
                    pair_results = {}
                    ea = db_embs[primary_ready['id']]
                    na = len(ea)
                    primary_key = db_text_key[primary_ready['id']]
                    for db_b in extra_ready:
                        eb = db_embs[db_b['id']]
                        # Self-comparison: the extra DB is the loaded dataset itself
                        # (same DataFrame object — any column choice), or has
                        # byte-identical embedding text. Either way row i lines up
                        # with row i, so mask the diagonal — a row should never be
                        # returned as its own ~100% match.
                        is_self = (
                            db_b['df'] is primary_ready['df']
                            or db_text_key[db_b['id']] == primary_key
                        )
                        _self_txt = " (self-similarity)" if is_self else ""
                        tidx    = np.zeros((na, K), dtype=np.int32)
                        tscores = np.zeros((na, K), dtype=np.float32)
                        pbar = st.progress(0.0, text=f"Similarity: {primary_ready['name']} ↔ {db_b['name']}{_self_txt}…")
                        for start in range(0, na, CHUNK):
                            end  = min(start + CHUNK, na)
                            sims = _cos_sim(ea[start:end], eb)
                            if is_self:
                                for _r in range(end - start):
                                    sims[_r, start + _r] = -np.inf
                            ka   = min(K, sims.shape[1])
                            idx  = np.argsort(sims, axis=1)[:, -ka:][:, ::-1]
                            tidx[start:end,    :ka] = idx
                            tscores[start:end, :ka] = np.take_along_axis(sims, idx, axis=1)
                            pbar.progress(end / na, text=f"{primary_ready['name']} ↔ {db_b['name']}{_self_txt}: {end:,}/{na:,}")
                        pbar.empty()
                        pair_results[(primary_ready['id'], db_b['id'])] = {
                            'top_idx': tidx, 'top_scores': tscores,
                            'db_a': primary_ready, 'db_b': db_b, 'is_self': is_self,
                        }

                    st.session_state['cross_db_multi_results'] = {'pairs': pair_results}
                    st.success(f"Done — {primary_ready['name']} compared against {n_pairs} database{'s' if n_pairs != 1 else ''}.")

                # ── Results ───────────────────────────────────────────────────────
                if 'cross_db_multi_results' in st.session_state:
                    all_pairs = st.session_state['cross_db_multi_results']['pairs']
                    if not all_pairs:
                        st.info("No results yet.")
                    else:
                        PALETTE = ["#f97316", "#3b82f6", "#22c55e", "#a855f7", "#ec4899", "#14b8a6"]

                        def _hex_rgba(hex_color, alpha=0.3):
                            h = hex_color.lstrip("#")
                            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                            return f"rgba({r},{g},{b},{alpha})"

                        def _render_dual_path_dedup(pair_res_dict, current_thr, kp):
                            st.markdown("### 🧬 Cross-DB Deduplication Pipeline (`Easy Path` vs `Hard Path`)")
                            st.caption("Inspect similarity candidates using either interval-based semantic bins or multi-gate spatial/temporal/text boolean filters.")
                            
                            path_sel = st.radio(
                                "Select Workflow Path:",
                                [
                                    "⚡ a. Easy Path: Automated Semantic Similarity Bins",
                                    "🎛️ b. Hard Path: Interactive Multi-Gate Batch Overview"
                                ],
                                key=f"dp_path_sel_{kp}"
                            )
                            
                            extracted_pairs = []
                            for pk, pres in pair_res_dict.items():
                                ts, ti = pres['top_scores'], pres['top_idx']
                                dfa, dfb = pres['db_a']['df'], pres['db_b']['df']
                                name_a, name_b = pres['db_a']['name'], pres['db_b']['name']
                                
                                # Find lat/lon/date columns once per dataframe
                                lca = next((c for c in ["latitude", "lat"] if c in dfa.columns), "")
                                oca = next((c for c in ["longitude", "lon"] if c in dfa.columns), "")
                                dca = next((c for c in ["date_time", "date"] if c in dfa.columns), "")
                                lcb = next((c for c in ["latitude", "lat"] if c in dfb.columns), "")
                                ocb = next((c for c in ["longitude", "lon"] if c in dfb.columns), "")
                                dcb = next((c for c in ["date_time", "date"] if c in dfb.columns), "")
                                
                                for i in range(len(ts)):
                                    for j in range(ts.shape[1]):
                                        sim = float(ts[i, j])
                                        if sim >= 0.60:
                                            row_b_idx = int(ti[i, j])
                                            ra, rb = dfa.iloc[i], dfb.iloc[row_b_idx]
                                            
                                            km_val = None
                                            if lca and oca and lcb and ocb:
                                                try:
                                                    l1, o1 = float(ra[lca]), float(ra[oca])
                                                    l2, o2 = float(rb[lcb]), float(rb[ocb])
                                                    km_val = round(_haversine_km(l1, o1, l2, o2), 2)
                                                except Exception:
                                                    pass
                                                    
                                            day_val = None
                                            if dca and dcb:
                                                da = _parse_date(ra[dca])
                                                db = _parse_date(rb[dcb])
                                                if da and db:
                                                    day_val = abs((da - db).days)
                                                    
                                            b_lbl = "distinct"
                                            if sim >= 0.88: b_lbl = "exact_duplicate"
                                            elif sim >= 0.80: b_lbl = "strong_similar"
                                            elif sim >= 0.70: b_lbl = "moderate_similar"
                                            
                                            is_txt = bool(sim >= current_thr)
                                            is_dt = bool(day_val is not None and day_val <= 3)
                                            is_loc = bool(km_val is not None and km_val <= 50.0)
                                            is_both = bool(is_dt and is_loc)
                                            is_all = bool(is_txt and is_both)
                                            
                                            extracted_pairs.append({
                                                "db_pair": f"{name_a} ↔ {name_b}",
                                                "row_a": int(i),
                                                "row_b": int(row_b_idx),
                                                "similarity": round(sim, 4),
                                                "bin": b_lbl,
                                                "km": km_val,
                                                "days": day_val,
                                                "is_similar_text": is_txt,
                                                "is_similar_date": is_dt,
                                                "is_similar_location": is_loc,
                                                "is_similar_both": is_both,
                                                "is_similar_all": is_all
                                            })
                            if not extracted_pairs:
                                st.info("No candidates evaluated above 0.60 similarity.")
                                return
                                
                            bin_counts = {"exact_duplicate": 0, "strong_similar": 0, "moderate_similar": 0, "distinct": 0}
                            gate_counts = {"similar_text": 0, "similar_date": 0, "similar_location": 0, "similar_both": 0, "similar_all": 0}
                            for ep in extracted_pairs:
                                if ep["bin"] in bin_counts: bin_counts[ep["bin"]] += 1
                                if ep["is_similar_text"]: gate_counts["similar_text"] += 1
                                if ep["is_similar_date"]: gate_counts["similar_date"] += 1
                                if ep["is_similar_location"]: gate_counts["similar_location"] += 1
                                if ep["is_similar_both"]: gate_counts["similar_both"] += 1
                                if ep["is_similar_all"]: gate_counts["similar_all"] += 1
                                
                            if path_sel.startswith("⚡ a. Easy Path"):
                                st.markdown("#### 📊 Semantic Similarity Bins Overview")
                                bc1, bc2, bc3, bc4 = st.columns(4)
                                bc1.metric("🔴 Exact Duplicates (≥ 0.88)", bin_counts["exact_duplicate"])
                                bc2.metric("🟠 Strong Similar (0.80 - 0.88)", bin_counts["strong_similar"])
                                bc3.metric("🟡 Moderate Similar (0.70 - 0.80)", bin_counts["moderate_similar"])
                                bc4.metric("🟢 Distinct (< 0.70)", bin_counts["distinct"])
                                
                                b_filt = st.selectbox("Filter Table by Bin Category:", ["All", "exact_duplicate", "strong_similar", "moderate_similar", "distinct"], key=f"dp_bfilt_{kp}")
                                f_eps = extracted_pairs if b_filt == "All" else [e for e in extracted_pairs if e["bin"] == b_filt]
                                if f_eps:
                                    df_disp = pd.DataFrame(f_eps)[["db_pair", "row_a", "row_b", "similarity", "bin", "km", "days"]]
                                    df_disp["similarity"] = df_disp["similarity"].apply(lambda x: f"{x*100:.1f}%")
                                    df_disp.columns = ["Databases", "Row A Index", "Row B Index", "Score", "Bin", "Distance (km)", "Date Diff (days)"]
                                    st.dataframe(df_disp, use_container_width=True)
                                else:
                                    st.info("No candidates inside this bin.")
                            else:
                                st.markdown("#### 🎛️ Interactive Multi-Gate Batch Overview")
                                g_filt = st.radio(
                                    "Filter Batch by Multi-Gate Action:",
                                    [
                                        f"📝 Similar Text Only (`sim ≥ {int(current_thr*100)}%`) — [{gate_counts['similar_text']} pairs]",
                                        f"📅 Similar Date Only (`± 3 days`) — [{gate_counts['similar_date']} pairs]",
                                        f"📍 Similar Location Only (`≤ 50 km`) — [{gate_counts['similar_location']} pairs]",
                                        f"⚡ Similar Both (`Spatial + Temporal`) — [{gate_counts['similar_both']} pairs]",
                                        f"🎯 Similar All / Flagged Duplicates (`Full Convergence`) — [{gate_counts['similar_all']} pairs]",
                                        "🌐 Show All Candidates"
                                    ],
                                    key=f"dp_gfilt_{kp}"
                                )
                                f_eps = extracted_pairs
                                if g_filt.startswith("📝"): f_eps = [e for e in extracted_pairs if e["is_similar_text"]]
                                elif g_filt.startswith("📅"): f_eps = [e for e in extracted_pairs if e["is_similar_date"]]
                                elif g_filt.startswith("📍"): f_eps = [e for e in extracted_pairs if e["is_similar_location"]]
                                elif g_filt.startswith("⚡"): f_eps = [e for e in extracted_pairs if e["is_similar_both"]]
                                elif g_filt.startswith("🎯"): f_eps = [e for e in extracted_pairs if e["is_similar_all"]]
                                
                                if f_eps:
                                    df_disp = pd.DataFrame(f_eps)[["db_pair", "row_a", "row_b", "similarity", "km", "days"]]
                                    df_disp["similarity"] = df_disp["similarity"].apply(lambda x: f"{x*100:.1f}%")
                                    df_disp.columns = ["Databases", "Row A Index", "Row B Index", "Similarity", "Distance (km)", "Date Diff (days)"]
                                    st.dataframe(df_disp, use_container_width=True)
                                else:
                                    st.warning("No candidate pairs satisfied this multi-gate button selection.")

                        pair_labels = {k: f"{v['db_a']['name']} ↔ {v['db_b']['name']}" for k, v in all_pairs.items()}
                        view_options = (["All"] + list(pair_labels.values())) if len(all_pairs) > 1 else list(pair_labels.values())
                        sel_label = st.selectbox("View", view_options, key="xdb_pair_sel")
                        is_all = sel_label == "All"

                        pct = st.slider("Similarity threshold", 0, 100, 70, format="%d%%", key="xdb_thr")
                        thr = pct / 100.0

                        if is_all:
                            # ── Aggregate coverage metrics ─────────────────────────────
                            prim_ref = list(all_pairs.values())[0]['db_a']
                            primary_matched_rows = set()
                            per_db_cov = {}
                            for res in all_pairs.values():
                                ts, ti = res['top_scores'], res['top_idx']
                                mask = ts[:, 0] >= thr
                                primary_matched_rows.update(np.where(mask)[0].tolist())
                                matched_b = {int(ti[i, j]) for i in np.where(mask)[0]
                                             for j in range(ts.shape[1]) if ts[i, j] >= thr}
                                per_db_cov[res['db_b']['name']] = (len(matched_b), len(res['db_b']['df']))

                            metric_cols = st.columns(1 + len(all_pairs))
                            metric_cols[0].metric(
                                f"{prim_ref['name']} coverage",
                                f"{len(primary_matched_rows)/len(prim_ref['df'])*100:.1f}%",
                                f"{len(primary_matched_rows)}/{len(prim_ref['df'])}",
                            )
                            for ci, (db_name, (matched_b, total_b)) in enumerate(per_db_cov.items()):
                                metric_cols[ci + 1].metric(
                                    f"{db_name} coverage",
                                    f"{matched_b/total_b*100:.1f}%",
                                    f"{matched_b}/{total_b}",
                                )

                            tab_hist, tab_pairs, tab_sankey, tab_thr, tab_dedup = st.tabs(
                                ["📊 Score Distribution", "🔀 Matched Pairs",
                                 "🌊 Sankey", "🎯 Suggest Threshold", "🧬 Dedupe & Similarity Bins (Easy vs Hard)"]
                            )

                            with tab_hist:
                                fig_h = go.Figure()
                                for idx, (k, res) in enumerate(all_pairs.items()):
                                    fig_h.add_trace(go.Histogram(
                                        x=res['top_scores'][:, 0], nbinsx=50,
                                        name=pair_labels[k],
                                        marker_color=PALETTE[idx % len(PALETTE)], opacity=0.65,
                                    ))
                                fig_h.add_vline(x=thr, line_dash="dash", line_color="red",
                                                annotation_text=f"Threshold {pct}%", annotation_position="top right")
                                fig_h.update_layout(
                                    barmode="overlay",
                                    title="Best-match score distribution — all pairs",
                                    xaxis_title="Cosine similarity", yaxis_title="Rows",
                                    template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                                )
                                st.plotly_chart(fig_h, use_container_width=True)

                            with tab_pairs:
                                with st.expander("⚙️ Column configuration", expanded=False):
                                    _cfg_cols = st.columns(1 + len(all_pairs))
                                    with _cfg_cols[0]:
                                        st.caption(f"**{prim_ref['name']}** (left) context")
                                        ctx_a_all = st.multiselect(
                                            "Extra columns",
                                            [c for c in prim_ref['df'].columns if c not in prim_ref['cols']],
                                            key="xdb_ctx_a_all",
                                        )
                                    ctx_b_all = {}
                                    for ci, (k, res) in enumerate(all_pairs.items()):
                                        with _cfg_cols[ci + 1]:
                                            st.caption(f"**{res['db_b']['name']}** (right) context")
                                            ctx_b_all[k] = st.multiselect(
                                                "Extra columns",
                                                [c for c in res['db_b']['df'].columns if c not in res['db_b']['cols']],
                                                key=f"xdb_ctx_b_all_{k[1]}",
                                            )

                                combined = []
                                for k, res in all_pairs.items():
                                    ts, ti = res['top_scores'], res['top_idx']
                                    for i in range(len(ts)):
                                        for j in range(ts.shape[1]):
                                            if ts[i, j] >= thr:
                                                combined.append((i, int(ti[i, j]), float(ts[i, j]), k, res))
                                combined.sort(key=lambda x: -x[2])

                                if not combined:
                                    st.info("No matches above threshold — lower the slider.")
                                else:
                                    PAGE = 15
                                    n_pages = max(1, (len(combined) + PAGE - 1) // PAGE)
                                    page = st.number_input("Page", 1, n_pages, 1, key="xdb_page_all") - 1
                                    d1c = prim_ref['cols'] + ctx_a_all
                                    _cap_col, _dl_col = st.columns([4, 1])
                                    _cap_col.caption(f"Showing {page*PAGE+1}–{min((page+1)*PAGE, len(combined))} of {len(combined):,}")
                                    _dl_rows = []
                                    for _ai, _bi, _sc, _k, _r in combined:
                                        _row = {'source_db': _r['db_b']['name'], 'similarity_pct': round(_sc * 100, 1)}
                                        for _c in d1c:
                                            _row[f"{prim_ref['name']}_{_c}"] = _r['db_a']['df'].iloc[_ai].get(_c, '')
                                        for _c in (_r['db_b']['cols'] + ctx_b_all.get(_k, [])):
                                            _row[f"{_r['db_b']['name']}_{_c}"] = _r['db_b']['df'].iloc[_bi].get(_c, '')
                                        _dl_rows.append(_row)
                                    _dl_col.download_button(
                                        "⬇️ CSV", pd.DataFrame(_dl_rows).to_csv(index=False),
                                        "matched_pairs.csv", "text/csv", key="dl_pairs_all",
                                    )
                                    for ai, bi, score, k, res in combined[page*PAGE : (page+1)*PAGE]:
                                        lc, mc, rc = st.columns([5, 1, 5])
                                        with lc:
                                            st.dataframe(res['db_a']['df'].iloc[[ai]][d1c],
                                                         use_container_width=True, hide_index=True)
                                        with mc:
                                            st.metric("", f"{score*100:.0f}%")
                                            st.caption(f"→ {res['db_b']['name']}")
                                        with rc:
                                            d2c = res['db_b']['cols'] + ctx_b_all.get(k, [])
                                            st.dataframe(res['db_b']['df'].iloc[[bi]][d2c],
                                                         use_container_width=True, hide_index=True)
                                        st.divider()

                            with tab_sankey:
                                with st.expander("⚙️ Column configuration", expanded=True):
                                    sk_grid = st.columns(1 + len(all_pairs))
                                    with sk_grid[0]:
                                        st.caption(f"**{prim_ref['name']}** (left)")
                                        sk_col_primary = st.selectbox(
                                            "Group by", list(prim_ref['df'].columns), key="xdb_sk_all_primary"
                                        )
                                        if prim_ref['df'][sk_col_primary].nunique() > 50:
                                            st.caption(f":orange[{prim_ref['df'][sk_col_primary].nunique()} unique]")
                                    sk_cols_all = {}
                                    for ci, (k, res) in enumerate(all_pairs.items()):
                                        with sk_grid[ci + 1]:
                                            st.caption(f"**{res['db_b']['name']}** (right)")
                                            sc = st.selectbox(
                                                "Group by", list(res['db_b']['df'].columns),
                                                key=f"xdb_sk_all_{k[1]}",
                                            )
                                            sk_cols_all[k] = sc
                                            if res['db_b']['df'][sc].nunique() > 50:
                                                st.caption(f":orange[{res['db_b']['df'][sc].nunique()} unique]")
                                    top_n_grp = st.slider("Max nodes per side", 3, 50, 15, key="xdb_sk_topn_all")

                                # Right-side labels include the DB name so nodes from different DBs are distinct
                                link_counts_all: Counter = Counter()
                                for k, res in all_pairs.items():
                                    g1s = res['db_a']['df'][sk_col_primary].fillna("Unknown").astype(str)
                                    g2s = res['db_b']['df'][sk_cols_all[k]].fillna("Unknown").astype(str)
                                    for i in range(len(res['top_scores'])):
                                        for j in range(res['top_scores'].shape[1]):
                                            if res['top_scores'][i, j] >= thr:
                                                rnode = f"{res['db_b']['name']} · {g2s.iloc[int(res['top_idx'][i, j])]}"
                                                link_counts_all[(g1s.iloc[i], rnode)] += 1

                                if not link_counts_all:
                                    st.info("No matches above threshold for Sankey.")
                                else:
                                    top_g1 = {g for g, _ in Counter(
                                        {k[0]: v for k, v in link_counts_all.items()}
                                    ).most_common(top_n_grp)}
                                    # Top nodes per extra DB to respect per-DB limit
                                    top_g2 = set()
                                    for ci, (k, res) in enumerate(all_pairs.items()):
                                        prefix = res['db_b']['name'] + " · "
                                        db_counts = {lbl: cnt for (_, lbl), cnt in link_counts_all.items()
                                                     if lbl.startswith(prefix)}
                                        top_g2.update(g for g, _ in Counter(db_counts).most_common(top_n_grp))

                                    lg1 = sorted(top_g1)
                                    lg2 = sorted(top_g2)
                                    node_colors = ["#f97316"] * len(lg1)
                                    for lbl in lg2:
                                        db_idx = next(
                                            (ci for ci, (k, res) in enumerate(all_pairs.items())
                                             if lbl.startswith(res['db_b']['name'] + " · ")), 0
                                        )
                                        node_colors.append(PALETTE[(db_idx + 1) % len(PALETTE)])

                                    all_labels = [f"{prim_ref['name']} · {g}" for g in lg1] + lg2
                                    map1 = {g: i for i, g in enumerate(lg1)}
                                    map2 = {g: len(lg1) + i for i, g in enumerate(lg2)}
                                    src, tgt, val, lnk_colors = [], [], [], []
                                    for (g1, g2), cnt in link_counts_all.items():
                                        if g1 in map1 and g2 in map2:
                                            db_idx = next(
                                                (ci for ci, (k, res) in enumerate(all_pairs.items())
                                                 if g2.startswith(res['db_b']['name'] + " · ")), 0
                                            )
                                            src.append(map1[g1]); tgt.append(map2[g2]); val.append(cnt)
                                            lnk_colors.append(_hex_rgba(PALETTE[(db_idx + 1) % len(PALETTE)]))
                                    st.caption(f"{sum(val):,} matched pairs above {pct}% across all databases")
                                    fig_sk = go.Figure(data=[go.Sankey(
                                        arrangement="snap",
                                        node=dict(label=all_labels, pad=15, thickness=20, color=node_colors),
                                        link=dict(source=src, target=tgt, value=val, color=lnk_colors),
                                    )])
                                    fig_sk.update_layout(
                                        title=f"{prim_ref['name']} ({sk_col_primary}) → all databases at ≥{pct}%",
                                        template="plotly_dark",
                                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                        height=max(500, top_n_grp * 30),
                                    )
                                    st.plotly_chart(fig_sk, use_container_width=True)

                            with tab_thr:
                                _threshold_tab_all(all_pairs, GEMINI_KEY, pct)

                            with tab_dedup:
                                _render_dual_path_dedup(all_pairs, thr, "all")

                        else:
                            # ── Single pair view ───────────────────────────────────────
                            sel_key    = next(k for k, v in pair_labels.items() if v == sel_label)
                            res        = all_pairs[sel_key]
                            top_idx    = res['top_idx']
                            top_scores = res['top_scores']
                            prim       = res['db_a']['df']
                            corp       = res['db_b']['df']
                            max_per_row = top_scores[:, 0]

                            mask     = max_per_row >= thr
                            n_m1     = int(mask.sum())
                            matched2 = {int(top_idx[i, j]) for i in np.where(mask)[0]
                                        for j in range(top_scores.shape[1]) if top_scores[i, j] >= thr}
                            m1c, m2c, m3c = st.columns(3)
                            m1c.metric("Matched pairs", f"{n_m1:,}")
                            m2c.metric(f"{res['db_a']['name']} coverage",
                                       f"{n_m1/len(prim)*100:.1f}%", f"{n_m1}/{len(prim)}")
                            m3c.metric(f"{res['db_b']['name']} coverage",
                                       f"{len(matched2)/len(corp)*100:.1f}%", f"{len(matched2)}/{len(corp)}")

                            tab_hist, tab_pairs, tab_sankey, tab_thr, tab_dedup = st.tabs(
                                ["📊 Score Distribution", "🔀 Matched Pairs",
                                 "🌊 Sankey", "🎯 Suggest Threshold", "🧬 Dedupe & Similarity Bins (Easy vs Hard)"]
                            )

                            with tab_hist:
                                fig_h = go.Figure()
                                fig_h.add_trace(go.Histogram(x=max_per_row, nbinsx=50,
                                                             marker_color="orange", opacity=0.8))
                                fig_h.add_vline(x=thr, line_dash="dash", line_color="red",
                                                annotation_text=f"Threshold {pct}%", annotation_position="top right")
                                fig_h.update_layout(
                                    title=f"Best-match score distribution — {sel_label}",
                                    xaxis_title="Cosine similarity", yaxis_title="Rows",
                                    template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                )
                                st.plotly_chart(fig_h, use_container_width=True)

                            with tab_pairs:
                                with st.expander("⚙️ Column configuration", expanded=False):
                                    xc1, xc2 = st.columns(2)
                                    with xc1:
                                        st.caption(f"**{res['db_a']['name']}** context")
                                        ctx_a = st.multiselect("Extra columns",
                                            [c for c in prim.columns if c not in res['db_a']['cols']],
                                            key="xdb_ctx_a")
                                    with xc2:
                                        st.caption(f"**{res['db_b']['name']}** context")
                                        ctx_b = st.multiselect("Extra columns",
                                            [c for c in corp.columns if c not in res['db_b']['cols']],
                                            key="xdb_ctx_b")

                                pair_list = [
                                    (i, int(top_idx[i, j]), float(top_scores[i, j]))
                                    for i in range(len(top_scores))
                                    for j in range(top_scores.shape[1])
                                    if top_scores[i, j] >= thr
                                ]
                                # Order every matched pair by decreasing cosine similarity.
                                pair_list.sort(key=lambda x: -x[2])
                                if not pair_list:
                                    st.info("No matches above threshold — lower the slider.")
                                else:
                                    PAGE    = 15
                                    n_pages = max(1, (len(pair_list) + PAGE - 1) // PAGE)
                                    page    = st.number_input("Page", 1, n_pages, 1, key="xdb_page") - 1
                                    d1c = res['db_a']['cols'] + ctx_a
                                    d2c = res['db_b']['cols'] + ctx_b
                                    _cap_col, _dl_col = st.columns([4, 1])
                                    _cap_col.caption(f"Showing {page*PAGE+1}–{min((page+1)*PAGE, len(pair_list))} of {len(pair_list):,}")
                                    _dl_rows = []
                                    for _ai, _bi, _sc in pair_list:
                                        _row = {'similarity_pct': round(_sc * 100, 1)}
                                        for _c in d1c:
                                            _row[f"{res['db_a']['name']}_{_c}"] = prim.iloc[_ai].get(_c, '')
                                        for _c in d2c:
                                            _row[f"{res['db_b']['name']}_{_c}"] = corp.iloc[_bi].get(_c, '')
                                        _dl_rows.append(_row)
                                    _dl_col.download_button(
                                        "⬇️ CSV", pd.DataFrame(_dl_rows).to_csv(index=False),
                                        "matched_pairs.csv", "text/csv", key="dl_pairs_single",
                                    )

                                    # ── Flag duplicates ───────────────────────────
                                    st.divider()
                                    st.markdown("**Flag duplicates**")
                                    fc1, fc2 = st.columns([3, 1])
                                    with fc1:
                                        flag_scope = st.radio(
                                            "Scope",
                                            [f"All matches above threshold ({pct}%)", "100% matches only"],
                                            key="xdb_flag_scope",
                                            horizontal=True,
                                            label_visibility="collapsed",
                                        )
                                    with fc2:
                                        do_flag = st.button(
                                            "🚩 Flag as duplicates", key="xdb_flag_btn",
                                            use_container_width=True,
                                        )
                                    st.caption(
                                        f"Marks matched **{res['db_a']['name']}** rows with `is_duplicate`, "
                                        f"`duplicate_similarity_pct`, `duplicate_source_db`, and the matched "
                                        f"**{res['db_b']['name']}** columns as `dup__{res['db_b']['name']}__<col>`. "
                                        "Each flagged row keeps its single best (highest-similarity) match."
                                    )
                                    if do_flag:
                                        _exact = flag_scope == "100% matches only"
                                        _scope = [
                                            (a, b, s) for (a, b, s) in pair_list
                                            if (round(s * 100, 1) >= 100.0 if _exact else True)
                                        ]
                                        _n = len(prim)
                                        _is_dup  = np.zeros(_n, dtype=bool)
                                        _dup_sim = np.full(_n, np.nan, dtype=float)
                                        _dup_row = np.full(_n, np.nan, dtype=float)
                                        # All configured 2nd-dataset columns (embedding + extra context).
                                        _bcols = list(dict.fromkeys(res['db_b']['cols'] + ctx_b))
                                        _dup_extra = {c: np.array([None] * _n, dtype=object) for c in _bcols}
                                        _best = {}
                                        for _a, _b, _s in _scope:
                                            if _a not in _best or _s > _best[_a][0]:
                                                _best[_a] = (_s, _b)
                                        for _a, (_s, _b) in _best.items():
                                            _is_dup[_a]  = True
                                            _dup_sim[_a] = round(_s * 100, 1)
                                            _dup_row[_a] = _b
                                            for _c in _bcols:
                                                _dup_extra[_c][_a] = corp.iloc[_b].get(_c, None)
                                        _flagged_df = prim.copy()
                                        _flagged_df['is_duplicate'] = _is_dup
                                        _flagged_df['duplicate_similarity_pct'] = _dup_sim
                                        _flagged_df['duplicate_source_db'] = np.where(
                                            _is_dup, res['db_b']['name'], ''
                                        )
                                        _flagged_df['duplicate_match_row'] = _dup_row
                                        for _c in _bcols:
                                            _flagged_df[f"dup__{res['db_b']['name']}__{_c}"] = _dup_extra[_c]
                                        st.session_state['xdb_flagged'] = {
                                            'pair': sel_key,
                                            'df': _flagged_df,
                                            'caption': (
                                                f"Flagged **{int(_is_dup.sum()):,}** of {_n:,} "
                                                f"{res['db_a']['name']} rows as duplicates of "
                                                f"{res['db_b']['name']} "
                                                f"({'100% matches' if _exact else f'≥{pct}% matches'})."
                                            ),
                                        }
                                    _flagged = st.session_state.get('xdb_flagged')
                                    if _flagged and _flagged.get('pair') == sel_key:
                                        st.success(_flagged['caption'])
                                        _fdf = _flagged['df']
                                        _dups_only = _fdf[_fdf['is_duplicate']]
                                        st.caption(f"Preview — {len(_dups_only):,} flagged row(s):")
                                        st.dataframe(_dups_only, use_container_width=True)
                                        st.download_button(
                                            "⬇️ Download flagged dataset (CSV)",
                                            _fdf.to_csv(index=False),
                                            "flagged_duplicates.csv", "text/csv",
                                            key="dl_flagged_dups",
                                        )
                                    st.divider()

                                    for ai, bi, score in pair_list[page*PAGE : (page+1)*PAGE]:
                                        lc, mc, rc = st.columns([5, 1, 5])
                                        with lc:
                                            st.dataframe(prim.iloc[[ai]][d1c], use_container_width=True, hide_index=True)
                                        with mc:
                                            st.metric("", f"{score*100:.0f}%")
                                        with rc:
                                            st.dataframe(corp.iloc[[bi]][d2c], use_container_width=True, hide_index=True)
                                        st.divider()

                            with tab_sankey:
                                with st.expander("⚙️ Column configuration", expanded=True):
                                    sk1, sk2 = st.columns(2)
                                    with sk1:
                                        st.caption(f"**{res['db_a']['name']}** — left nodes")
                                        sk_col1 = st.selectbox("Group by", list(prim.columns), key="xdb_sk1")
                                        if prim[sk_col1].nunique() > 50:
                                            st.caption(f":orange[{prim[sk_col1].nunique()} unique values]")
                                    with sk2:
                                        st.caption(f"**{res['db_b']['name']}** — right nodes")
                                        sk_col2 = st.selectbox("Group by", list(corp.columns), key="xdb_sk2")
                                        if corp[sk_col2].nunique() > 50:
                                            st.caption(f":orange[{corp[sk_col2].nunique()} unique values]")
                                    top_n_grp = st.slider("Max nodes per side", 3, 50, 15, key="xdb_sk_topn")

                                g1s = prim[sk_col1].fillna("Unknown").astype(str)
                                g2s = corp[sk_col2].fillna("Unknown").astype(str)
                                link_counts: Counter = Counter()
                                for i in range(len(top_scores)):
                                    for j in range(top_scores.shape[1]):
                                        if top_scores[i, j] >= thr:
                                            link_counts[(g1s.iloc[i], g2s.iloc[int(top_idx[i, j])])] += 1

                                if not link_counts:
                                    st.info("No matches above threshold for Sankey.")
                                else:
                                    top_g1 = {g for g, _ in Counter({k[0]: v for k, v in link_counts.items()}).most_common(top_n_grp)}
                                    top_g2 = {g for g, _ in Counter({k[1]: v for k, v in link_counts.items()}).most_common(top_n_grp)}
                                    lg1, lg2 = sorted(top_g1), sorted(top_g2)
                                    all_labels = [f"{res['db_a']['name']} · {g}" for g in lg1] + \
                                                 [f"{res['db_b']['name']} · {g}" for g in lg2]
                                    map1 = {g: i for i, g in enumerate(lg1)}
                                    map2 = {g: len(lg1) + i for i, g in enumerate(lg2)}
                                    src, tgt, val = [], [], []
                                    for (g1, g2), cnt in link_counts.items():
                                        if g1 in map1 and g2 in map2:
                                            src.append(map1[g1]); tgt.append(map2[g2]); val.append(cnt)
                                    st.caption(f"{sum(val):,} matched pairs above {pct}% — {sk_col1} → {sk_col2}")
                                    fig_sk = go.Figure(data=[go.Sankey(
                                        arrangement="snap",
                                        node=dict(label=all_labels, pad=15, thickness=20,
                                                  color=["#f97316"]*len(lg1) + ["#3b82f6"]*len(lg2)),
                                        link=dict(source=src, target=tgt, value=val, color="rgba(249,115,22,0.25)"),
                                    )])
                                    fig_sk.update_layout(
                                        title=f"{res['db_a']['name']} ({sk_col1}) → {res['db_b']['name']} ({sk_col2}) at ≥{pct}%",
                                        template="plotly_dark",
                                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                        height=max(500, top_n_grp * 25),
                                    )
                                    st.plotly_chart(fig_sk, use_container_width=True)

                            with tab_thr:
                                _threshold_tab(res, sel_key, GEMINI_KEY, pct)

                            with tab_dedup:
                                _render_dual_path_dedup({sel_key: res}, thr, f"sel_{sel_key[0]}_{sel_key[1]}")

