import json
import datetime
import streamlit as st
#import geopandas as gpd
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from uap_analyzer import UAPParser, UAPAnalyzer, UAPVisualizer
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
from streamlit_extras.stateful_button import button as stateful_button
from streamlit_keplergl import keplergl_static
from keplergl import KeplerGl

# Import enhanced utilities
from utils.data_processing import DataProcessor
from utils.visualization import UAP_Visualizer as Enhanced_Visualizer
from utils.session_manager import SessionStateManager



from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
    is_period_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.title('Interactive Map')

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
if 'map_generated' not in st.session_state:
    st.session_state['map_generated'] = False
if 'date_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False


if "datasets" not in st.session_state:
    st.session_state.datasets = []
# Filter UI selection (use one or both)
col_f1, col_f2 = st.columns(2)
with col_f1:
    use_enhanced_filter = st.checkbox("Enhanced Filtering", value=True)
with col_f2:
    use_legacy_filter = st.checkbox("Date Merging", value=False)

def apply_selected_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply selected filters in sequence: Enhanced then Legacy if both enabled."""
    df_out = df
    if use_enhanced_filter:
        df_out = filter_dataframe(df_out)
    if use_legacy_filter:
        df_out = filter_dataframe_legacy(df_out)
    return df_out


# sf_zip_geo_gdf = gpd.read_file("sf_zip_geo.geojson")
# sf_zip_geo_gdf.label = "SF Zip Geo"
# sf_zip_geo_gdf.id = "sf-zip-geo"
# st.session_state.datasets.append(sf_zip_geo_gdf)

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

def plot_hist(df, column, bins=10, kde=True):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data=df, x=column, kde=True, bins=bins, color='orange', ax=ax)
        # set the ticks and frame in orange
        ax.spines['bottom'].set_color('orange')
        ax.spines['top'].set_color('orange')
        ax.spines['right'].set_color('orange')
        ax.spines['left'].set_color('orange')
        ax.xaxis.label.set_color('orange')
        ax.yaxis.label.set_color('orange')
        ax.tick_params(axis='x', colors='orange')
        ax.tick_params(axis='y', colors='orange')
        ax.title.set_color('orange')

        # Set transparent background
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.patch.set_facecolor('white')
        
        # Ensure we have a datetime column
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            date_series = pd.to_datetime(df[date_column], errors='coerce')
        else:
            date_series = df[date_column]
        
        # Remove NaT values
        date_series = date_series.dropna()
        
        if len(date_series) == 0:
            # No valid dates
            ax1.text(0.5, 0.5, 'No valid dates found', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No valid dates found', ha='center', va='center', transform=ax2.transAxes)
            return fig
        
        # Plot 1: Year-based counts
        year_counts = date_series.dt.year.value_counts().sort_index()
        
        if len(year_counts) > 0:
            bars1 = ax1.bar(year_counts.index, year_counts.values, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
            ax1.set_title('Sightings by Year', color=color, fontweight='bold', fontsize=12)
            ax1.set_xlabel('Year', color=color)
            ax1.set_ylabel('Count', color=color)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8, color='black')
            
            # Style the axes
            ax1.spines['bottom'].set_color(color)
            ax1.spines['left'].set_color(color)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.tick_params(axis='x', colors=color, rotation=45)
            ax1.tick_params(axis='y', colors=color)
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Month-based counts (aggregated across all years)
        month_counts = date_series.dt.month.value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        if len(month_counts) > 0:
            month_labels = [month_names[i-1] for i in month_counts.index]
            bars2 = ax2.bar(month_labels, month_counts.values, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
            ax2.set_title('Sightings by Month', color=color, fontweight='bold', fontsize=12)
            ax2.set_xlabel('Month', color=color)
            ax2.set_ylabel('Count', color=color)
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8, color='black')
            
            # Style the axes
            ax2.spines['bottom'].set_color(color)
            ax2.spines['left'].set_color(color)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.tick_params(axis='x', colors=color, rotation=45)
            ax2.tick_params(axis='y', colors=color)
            ax2.grid(True, alpha=0.3, axis='y')
        
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
        # Create error figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Error creating date visualization:\n{str(e)}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Date Visualization Error', color='red')
        plt.tight_layout()
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


def sanitize_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """Convert non-JSON-serializable values (datetime, date, time, period, timedelta) to strings."""
    df_safe = df.copy()

    # Ensure simple integer index and string column names
    df_safe = df_safe.reset_index(drop=True)
    df_safe.columns = df_safe.columns.map(str)

    # Datetime columns: export as ISO 8601 strings with millisecond precision, null-safe
    for col in df_safe.select_dtypes(include=["datetime64[ns]"]).columns:
        try:
            dt = pd.to_datetime(df_safe[col], errors='coerce')
            # format to ISO with milliseconds (slice microseconds to 3 digits)
            iso = dt.dt.strftime('%Y-%m-%dT%H:%M:%S.%f').str.slice(0, 23)
            # set None where NaT
            iso = iso.where(dt.notna(), None)
            df_safe[col] = iso
        except Exception:
            # Fallback to plain string
            df_safe[col] = df_safe[col].astype(object).where(~df_safe[col].isna(), None)

    # Period dtype
    for col in list(df_safe.columns):
        try:
            if is_period_dtype(df_safe[col]):
                df_safe[col] = df_safe[col].astype(str)
        except Exception:
            pass

    # Timedelta dtype
    for col in list(df_safe.columns):
        try:
            if is_timedelta64_dtype(df_safe[col]):
                df_safe[col] = df_safe[col].astype(str)
        except Exception:
            pass

    # Object columns: convert date/time-like objects to strings
    def _to_serializable(x):
        # Handle missing values first
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        if isinstance(x, pd.Timestamp):
            try:
                return x.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                return None
        if isinstance(x, (datetime.datetime, datetime.date)):
            return x.isoformat()
        if isinstance(x, datetime.time):
            return x.strftime('%H:%M:%S')
        return x

    for col in df_safe.select_dtypes(include=['object']).columns:
        try:
            df_safe[col] = df_safe[col].map(_to_serializable)
        except Exception:
            df_safe[col] = df_safe[col].apply(_to_serializable)

    # Final sweep: any remaining datetime-like dtypes (any unit) -> string
    for col in list(df_safe.columns):
        try:
            if np.issubdtype(df_safe[col].dtype, np.datetime64):
                df_safe[col] = pd.to_datetime(df_safe[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            if 'datetime' in str(df_safe[col].dtype).lower():
                try:
                    df_safe[col] = df_safe[col].astype(str)
                except Exception:
                    pass

    return df_safe

def detect_and_combine_date_columns(df):
    """
    Detect separate year, month, day, hour columns and combine them into datetime columns
    """
    df_combined = df.copy()
    
    # Look for common date component column patterns
    year_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in ['year', 'yr', 'yyyy'])]
    month_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in ['month', 'mon', 'mm'])]
    day_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in ['day', 'dd', 'date'])]
    hour_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in ['hour', 'hr', 'hh', 'time'])]
    
    # Try to find matching sets of date components
    date_combinations = []
    
    for year_col in year_cols:
        for month_col in month_cols:
            for day_col in day_cols:
                # Check if these columns have compatible data
                try:
                    # Test if we can create valid dates from a sample
                    sample_size = min(10, len(df))
                    test_sample = df.iloc[:sample_size]
                    
                    # Check if values are reasonable for date components
                    years = pd.to_numeric(test_sample[year_col], errors='coerce')
                    months = pd.to_numeric(test_sample[month_col], errors='coerce')
                    days = pd.to_numeric(test_sample[day_col], errors='coerce')
                    
                    # Validate ranges
                    valid_years = years.between(1900, 2100).all()
                    valid_months = months.between(1, 12).all()
                    valid_days = days.between(1, 31).all()
                    
                    if valid_years and valid_months and valid_days:
                        combination = {
                            'year': year_col,
                            'month': month_col,
                            'day': day_col,
                            'hour': None
                        }
                        
                        # Check if there's a compatible hour column
                        for hour_col in hour_cols:
                            try:
                                hours = pd.to_numeric(test_sample[hour_col], errors='coerce')
                                if hours.between(0, 23).all():
                                    combination['hour'] = hour_col
                                    break
                            except:
                                continue
                        
                        date_combinations.append(combination)
                        
                except Exception:
                    continue
    
    # Process the detected date combinations
    for i, combo in enumerate(date_combinations):
        try:
            # Create datetime column name
            datetime_col_name = f"datetime_combined_{i+1}"
            
            # Convert components to numeric
            year = pd.to_numeric(df_combined[combo['year']], errors='coerce')
            month = pd.to_numeric(df_combined[combo['month']], errors='coerce')
            day = pd.to_numeric(df_combined[combo['day']], errors='coerce')
            
            if combo['hour']:
                hour = pd.to_numeric(df_combined[combo['hour']], errors='coerce')
                # Create datetime with hour
                df_combined[datetime_col_name] = pd.to_datetime({
                    'year': year,
                    'month': month,
                    'day': day,
                    'hour': hour
                }, errors='coerce')
            else:
                # Create datetime without hour (defaults to midnight)
                df_combined[datetime_col_name] = pd.to_datetime({
                    'year': year,
                    'month': month,
                    'day': day
                }, errors='coerce')
            
            # Show info about the combined column
            valid_dates = df_combined[datetime_col_name].notna().sum()
            total_rows = len(df_combined)
            
            if valid_dates > 0:
                st.info(f"✅ Created datetime column '{datetime_col_name}' from {combo['year']}, {combo['month']}, {combo['day']}" + 
                       (f", {combo['hour']}" if combo['hour'] else "") + 
                       f" ({valid_dates}/{total_rows} valid dates)")
            
        except Exception as e:
            st.warning(f"Could not combine date columns {combo}: {e}")
    
    return df_combined

def auto_create_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Automatically create a unified datetime column `date_x` for mapping.
    Strategy:
      1) Prefer existing datetime-like columns (by dtype or strong parse signal)
      2) Otherwise, combine legacy year/month/day/hour columns
      3) Pick the candidate with the most valid timestamps
    """
    df_auto = df.copy()

    # If date_x already exists and is usable, keep it
    if 'date_x' in df_auto.columns:
        try:
            parsed = pd.to_datetime(df_auto['date_x'], errors='coerce')
            if parsed.notna().sum() > 0:
                df_auto['date_x'] = parsed
                return df_auto
        except Exception:
            pass

    # 1) Scan candidate columns that look like dates
    candidate_cols = [c for c in df_auto.columns if any(k in c.lower() for k in ['date', 'datetime', 'timestamp'])]
    # Include dtype-based candidates
    candidate_cols.extend([c for c in df_auto.select_dtypes(include=["datetime64[ns]"]).columns if c not in candidate_cols])
    best_col = None
    best_valid = -1
    for col in candidate_cols:
        try:
            parsed = pd.to_datetime(df_auto[col], errors='coerce')
            valid = parsed.notna().sum()
            if valid > best_valid:
                best_valid = valid
                best_col = col
        except Exception:
            continue

    if best_col is not None and best_valid > 0:
        try:
            df_auto['date_x'] = pd.to_datetime(df_auto[best_col], errors='coerce')
            # ensure tz-naive and floor to ms to align with kepler expectations
            try:
                df_auto['date_x'] = df_auto['date_x'].dt.tz_localize(None)
            except Exception:
                pass
            df_auto['date_x'] = df_auto['date_x'].dt.floor('ms')
            return df_auto
        except Exception:
            pass

    # 2) Try legacy combination
    df_combined = detect_and_combine_date_columns(df_auto)
    combined_cols = [c for c in df_combined.columns if c.startswith('datetime_combined_')]
    best_combined = None
    best_combined_valid = -1
    for col in combined_cols:
        try:
            valid = df_combined[col].notna().sum()
            if valid > best_combined_valid:
                best_combined_valid = valid
                best_combined = col
        except Exception:
            continue

    if best_combined is not None and best_combined_valid > 0:
        df_combined['date_x'] = pd.to_datetime(df_combined[best_combined], errors='coerce')
        try:
            df_combined['date_x'] = df_combined['date_x'].dt.tz_localize(None)
        except Exception:
            pass
        df_combined['date_x'] = df_combined['date_x'].dt.floor('ms')
        return df_combined

    # 3) Nothing found; return original
    return df_auto

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced filtering interface for map data"""
    return DataProcessor.filter_dataframe_enhanced(df, for_map=True, enable_quick_filters=False)

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

    # First, try to detect and combine separate date columns
    df_ = detect_and_combine_date_columns(df)
    
    # Add manual date combination option
    with st.expander("Combine separate year/month/day/hour columns manually", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            year_col = st.selectbox("Year Column", ["None"] + list(df_.columns), key="year_select")
        with col2:
            month_col = st.selectbox("Month Column", ["None"] + list(df_.columns), key="month_select")
        with col3:
            day_col = st.selectbox("Day Column", ["None"] + list(df_.columns), key="day_select")
        with col4:
            hour_col = st.selectbox("Hour Column (Optional)", ["None"] + list(df_.columns), key="hour_select")
        
        if st.button("Combine Selected Columns", key="manual_combine"):
            if year_col != "None" and month_col != "None" and day_col != "None":
                try:
                    # Convert to numeric
                    year = pd.to_numeric(df_[year_col], errors='coerce')
                    month = pd.to_numeric(df_[month_col], errors='coerce')
                    day = pd.to_numeric(df_[day_col], errors='coerce')
                    
                    datetime_dict = {'year': year, 'month': month, 'day': day}
                    
                    if hour_col != "None":
                        hour = pd.to_numeric(df_[hour_col], errors='coerce')
                        datetime_dict['hour'] = hour
                    
                    # Create the combined datetime column
                    df_['date_x'] = pd.to_datetime(datetime_dict, errors='coerce')
                    
                    valid_dates = df_['date_x'].notna().sum()
                    st.success(f"✅ Created 'date_x' column with {valid_dates}/{len(df_)} valid dates")
                    
                except Exception as e:
                    st.error(f"Error combining columns: {e}")
            else:
                st.warning("Please select at least Year, Month, and Day columns")
    
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

        elif is_object_dtype(df_[column]) or is_datetime64_any_dtype(df_[column]):
            # Check if this is already a datetime column that should get date visualization
            if is_datetime64_any_dtype(df_[column]):
                with st.status(f"Date Distribution: {column}", expanded=False) as stat:
                    try:
                        date_fig = plot_date_distribution(df_, column)
                        st.pyplot(date_fig)
                    except Exception as e:
                        st.warning(f"Could not create date visualization: {e}")
                filtered_columns.append(column)
                continue
            # First, check if this column actually contains date-like data
            is_likely_date = False
            
            # Check if column already has datetime dtype
            if is_datetime64_any_dtype(df_[column]):
                is_likely_date = True
            else:
                # For object columns, sample a few values to check if they're date-like
                sample_values = df_[column].dropna().head(10)
                if not sample_values.empty:
                    date_count = 0
                    for val in sample_values:
                        try:
                            # Try to parse as date
                            pd.to_datetime(val, errors='raise')
                            date_count += 1
                        except (ValueError, TypeError, pd.errors.ParserError):
                            continue
                    
                    # If more than 50% of samples are valid dates, treat as date column
                    is_likely_date = (date_count / len(sample_values)) > 0.5
            
            if is_likely_date:
                # Try to convert to datetime
                try:
                    original_column = df_[column].copy()
                    df_[column] = pd.to_datetime(df_[column], errors='coerce', infer_datetime_format=True)
                    
                    # Safely remove timezone only if tz-aware
                    try:
                        if getattr(df_[column].dt, "tz", None) is not None:
                            df_[column] = df_[column].dt.tz_localize(None)
                    except (AttributeError, TypeError):
                        pass

                    # Check if we have valid dates after conversion
                    valid = df_[column].dropna()
                    if len(valid) == 0 or len(valid) < len(df_[column]) * 0.1:  # Less than 10% valid dates
                        # Revert to original and treat as text
                        df_[column] = original_column
                        user_text_input = right.text_input(
                            f"Substring or regex in {column}",
                        )
                        if user_text_input:
                            try:
                                df_ = df_[df_[column].astype(str).str.contains(user_text_input, na=False, case=False)]
                                filtered_columns.append(column)
                            except Exception as e:
                                st.warning(f"Error filtering column {column}: {e}")
                        continue

                    # Clip to safe pandas Timestamp bounds to avoid ns overflow
                    SAFE_MIN = pd.Timestamp(1677, 9, 22)
                    SAFE_MAX = pd.Timestamp(2262, 4, 10)

                    try:
                        min_ts = max(valid.min(), SAFE_MIN)
                        max_ts = min(valid.max(), SAFE_MAX)
                        if min_ts > max_ts:
                            min_ts, max_ts = max_ts, min_ts

                        min_date = min_ts.date()
                        max_date = max_ts.date()

                        user_date_input = right.date_input(
                            f"Values for {column}",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                        )

                        # Improved tuple handling for date input
                        if user_date_input is not None:
                            if isinstance(user_date_input, (list, tuple)) and len(user_date_input) == 2:
                                try:
                                    start_date, end_date = user_date_input
                                    if start_date is not None and end_date is not None:
                                        start_ts = pd.Timestamp(start_date)
                                        end_ts = pd.Timestamp(end_date)
                                        start_ts = max(start_ts, SAFE_MIN)
                                        end_ts = min(end_ts, SAFE_MAX)
                                        df_ = df_.loc[df_[column].between(start_ts, end_ts)]
                                        filtered_columns.append(column)
                                except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
                                    st.warning(f"Error processing date range for {column}: {e}")
                                    continue
                            elif hasattr(user_date_input, 'date'):  # Single date object
                                try:
                                    single_date = pd.Timestamp(user_date_input)
                                    single_date = max(min(single_date, SAFE_MAX), SAFE_MIN)
                                    df_ = df_.loc[df_[column].dt.date == single_date.date()]
                                    filtered_columns.append(column)
                                except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
                                    st.warning(f"Error processing single date for {column}: {e}")
                                    continue

                        date_column = column

                        # Create date distribution visualization
                        with st.status(f"Date Distribution: {column}", expanded=False) as stat:
                            try:
                                # Create the main date distribution chart (year/month counts)
                                date_fig = plot_date_distribution(df_, column)
                                st.pyplot(date_fig)
                                
                                # Additional plots if there are other filtered columns
                                if filtered_columns:
                                    fig = None
                                    fig2 = None
                                    
                                    numeric_columns = [col for col in filtered_columns if col != column and is_numeric_dtype(df_[col])]
                                    if numeric_columns:
                                        try:
                                            fig = plot_line(df_, column, numeric_columns)
                                            st.pyplot(fig)
                                        except Exception as e:
                                            st.warning(f"Could not create line chart: {e}")
                                    
                                    categorical_columns = [col for col in filtered_columns if col != column and is_categorical_dtype(df_[col])]
                                    if categorical_columns:
                                        try:
                                            fig2 = plot_bar(df_, column, categorical_columns[0])
                                            st.pyplot(fig2)
                                        except Exception as e:
                                            st.warning(f"Could not create bar chart: {e}")
                                            
                            except Exception as e:
                                st.error(f"Could not create date visualizations: {e}")
                        
                        # Set this as the date column for later use
                        date_column = column

                        # Finally convert back to str for the map layer
                        try:
                            df_[column] = df_[column].dt.strftime('%Y-%m-%d %H:%M:%S')
                        except Exception:
                            # If strftime fails, convert to string another way
                            df_[column] = df_[column].astype(str)
                            
                    except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
                        st.warning(f"Error processing dates in column {column}: {e}")
                        # Revert to original and treat as text
                        df_[column] = original_column
                        user_text_input = right.text_input(
                            f"Substring or regex in {column}",
                        )
                        if user_text_input:
                            try:
                                df_ = df_[df_[column].astype(str).str.contains(user_text_input, na=False, case=False)]
                                filtered_columns.append(column)
                            except Exception as e:
                                st.warning(f"Error filtering column {column}: {e}")
                        
                except Exception as e:
                    st.warning(f"Could not process column {column} as dates: {e}")
                    # Treat as text input
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df_ = df_[df_[column].astype(str).str.contains(user_text_input, na=False)]
            else:
                # Not a date column, treat as text
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df_ = df_[df_[column].astype(str).str.contains(user_text_input, na=False)]
        else:
            user_text_input = right.text_input(
                f"Substring or regex in {column}",
            )
            if user_text_input:
                try:
                    df_ = df_[df_[column].astype(str).str.contains(user_text_input, na=False, case=False)]
                    filtered_columns.append(column)
                except Exception as e:
                    st.warning(f"Error filtering column {column} with pattern '{user_text_input}': {e}")
                    # Try exact match as fallback
                    try:
                        df_ = df_[df_[column].astype(str).str.lower().str.contains(user_text_input.lower(), na=False)]
                        filtered_columns.append(column)
                    except Exception:
                        st.error(f"Could not filter column {column}. Please check your input.")
    # write len of df after filtering with % of original
    st.write(f"{len(df_)} rows ({len(df_) / len(df) * 100:.2f}%)")
    return df_

def find_lat_lon_columns(df):
    lat_columns = df.columns[df.columns.str.lower().str.contains('lat')]
    lon_columns = df.columns[df.columns.str.lower().str.contains('lon|lng')]
    
    if len(lat_columns) > 0 and len(lon_columns) > 0:
        return lat_columns[0], lon_columns[0]
    else:
        return None, None

def load_data(file_path, key='df'):
    from data_fetch import load_uap_dataset
    return load_uap_dataset(file_path, key=key)


# Load dataset
data_path = 'parsed_files_distance_embeds.h5'



my_dataset = st.file_uploader("Upload Parsed DataFrame", type=["csv", "xlsx"])

if my_dataset is not None:    
    # optional: reset any previously loaded dataset to save memory
    try:
        if my_dataset.type == "text/csv":
            data = pd.read_csv(my_dataset)
        elif my_dataset.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            data = pd.read_excel(my_dataset)
        else:
            st.error("Unsupported file type. Please upload a CSV, Excel or HD5 file.")
            st.stop()
        # Auto-create date_x before filtering/mapping
        data = auto_create_date_column(data)
        filtered_df = apply_selected_filters(data)
        st.session_state['parsed_responses'] = filtered_df
        st.dataframe(filtered_df)
        # st.success(f"Successfully loaded and displayed data from {my_dataset.name}")
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
else:
    parsed = load_data(data_path).drop(columns=['embeddings'])
    parsed = auto_create_date_column(parsed)
    parsed_responses = apply_selected_filters(parsed)
    st.session_state['parsed_responses'] = parsed_responses
    # st.dataframe(parsed_responses)


# Create map regardless of data source
map_1 = KeplerGl(height=1200)
powerplant = pd.read_csv('global_power_plant_database.csv')
secret_bases = pd.read_csv('secret_bases.csv')

map_1.add_data(data=secret_bases, name="secret_bases")
map_1.add_data(data=powerplant, name='nuclear_powerplants')

# Get the already filtered data from session state for the map
filtered_df = st.session_state.get('parsed_responses', pd.DataFrame())
if my_dataset is not None:
    data_source = "uploaded"
else:
    data_source = "preloaded" if not filtered_df.empty else None

# Generate map if we have data
if not filtered_df.empty and data_source:
    try:
        st.session_state['data_loaded'] = True
        
        # Load the base config
        with open('military_config.kgl', 'r') as f:
            base_config = json.load(f)

        with open('uap_config.kgl', 'r') as f:
            uap_config = json.load(f)

        if filtered_df.columns.str.contains('date').any():
            # Get the date column name
            date_column = filtered_df.columns[filtered_df.columns.str.contains('date')].values[0]

            # Create a new filter
            new_filter = {
                "dataId": "uap_sightings",
                "name": date_column
            }

            # Append the new filter to the existing filters
            base_config['config']['visState']['filters'].append(new_filter)

            # Update the map config
            map_1.config = base_config

        # Find the latitude and longitude columns in the dataframe
        lat_col, lon_col = find_lat_lon_columns(filtered_df)
        if lat_col and lon_col:
            # Update the layer configurations
            for layer in uap_config['config']['visState']['layers']:
                if 'config' in layer and 'columns' in layer['config']:
                    if 'lat' in layer['config']['columns']:
                        layer['config']['columns']['lat'] = lat_col
                    if 'lng' in layer['config']['columns']:
                        layer['config']['columns']['lng'] = lon_col

            # Now extend the base_config with the updated uap_config layers
            base_config['config']['visState']['layers'].extend(uap_config['config']['visState']['layers'])
            map_1.config = base_config
        else:
            base_config['config']['visState']['layers'].extend([layer for layer in uap_config['config']['visState']['layers']])
            map_1.config = base_config

        # Ensure JSON-serializable dataframe for Kepler ingestion
        safe_map_df = sanitize_dataframe_for_json(filtered_df)
        map_1.add_data(data=safe_map_df, name="uap_sightings")
        
        keplergl_static(map_1, center_map=True)
        st.session_state['map_generated'] = True
        
        with st.container(border=True):
            st.write("Military Base coordinates approximated from: https://www.dpiarchive.com/ (Archive / UFO Related Secret Facilities / Top Priority Documents / Facilities Map and List.pdf)\n\nNuclear Powerplants from: https://datasets.wri.org/dataset/globalpowerplantdatabase")
    except Exception as e:
        st.error(f"An error occurred while generating the map: {e}")
else:
    if data_source is None:
        st.warning("Please upload a file to get started.")
    else:
        st.warning("No data available for map generation.")