
import math
import pandas as pd
import numpy as np
import json
import requests
import datetime
from datetime import timedelta
from PIL import Image
# alternative to PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import matplotlib.dates as mdates
import seaborn as sns
from IPython.display import Image as image_display
path = os.getcwd()
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from IPython.display import display
from dateutil import parser
from Levenshtein import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
# tqdm.pandas() is skipped as we are mostly using standard iterators here.
import streamlit.components.v1 as components
from dateutil import parser
from sentence_transformers import SentenceTransformer
import torch
import squarify
import matplotlib.colors as mcolors
import textwrap
import datamapplot
import streamlit as st


if 'form_submitted' not in st.session_state:
    st.session_state['form_submitted'] = False


st.title('Magnetic Correlations Dashboard')

# Configure matplotlib and streamlit for better chart display
plt.style.use('default')  # Ensure consistent plotting style


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
    """
    Create a line plot with improved styling and error handling.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        x_column (str): Column name for x-axis
        y_columns (list): List of column names for y-axis
        figsize (tuple): Figure size
        color (str): Base color for styling
        title (str): Custom title for the plot
        rolling_mean_value (int): Window size for rolling mean (0 to disable)
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    try:
        import matplotlib.cm as cm
        
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        
        # Validate inputs
        if x_column not in df.columns:
            ax.text(0.5, 0.5, f'X column "{x_column}" not found', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        missing_cols = [col for col in y_columns if col not in df.columns]
        if missing_cols:
            ax.text(0.5, 0.5, f'Y columns not found: {missing_cols}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        # Work with a copy to avoid modifying original data
        df_plot = df[[x_column] + y_columns].copy().dropna()
        
        if len(df_plot) == 0:
            ax.text(0.5, 0.5, 'No data to plot after removing NaN values', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        # Sort by x_column
        df_plot = df_plot.sort_values(by=x_column)

        # Calculate rolling mean if specified
        if rolling_mean_value and rolling_mean_value > 1:
            window = max(2, len(df_plot) // rolling_mean_value)
            for col in y_columns:
                df_plot[f'{col}_smooth'] = df_plot[col].rolling(window=window, center=True).mean()
                y_columns_to_plot = [f'{col}_smooth' for col in y_columns]
        else:
            y_columns_to_plot = y_columns

        # Generate colors
        colors = cm.Oranges(np.linspace(0.3, 0.9, len(y_columns_to_plot)))

        # Plot each y_column as a separate line
        for i, y_column in enumerate(y_columns_to_plot):
            original_col = y_column.replace('_smooth', '') if '_smooth' in y_column else y_column
            if y_column in df_plot.columns:
                ax.plot(df_plot[x_column], df_plot[y_column], 
                       color=colors[i], label=original_col, linewidth=2, alpha=0.8)

        # Format x-axis labels
        ax.tick_params(axis='x', rotation=30, colors=color)
        
        # Handle date formatting
        if pd.api.types.is_datetime64_any_dtype(df_plot[x_column]):
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()

        # Set title and labels
        ax.set_title(title or f'{", ".join(y_columns)} over {x_column}', 
                    color=color, fontweight='bold', fontsize=14)
        ax.set_xlabel(x_column, color=color, fontsize=12)
        ax.set_ylabel(', '.join(y_columns), color=color, fontsize=12)
        
        # Style the plot
        for spine in ax.spines.values():
            spine.set_color(color)
        
        ax.tick_params(axis='both', colors=color)
        ax.grid(True, alpha=0.3, color=color)
        
        # Legend
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1), 
                 facecolor='black', framealpha=0.4, labelcolor=color, edgecolor=color)

        # Set transparent background
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Error creating line plot:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
        ax.set_title('Line Plot Error', color='red')
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        return fig

def plot_bar(df, x_column, y_column, figsize=(12, 10), color='orange', title=None, rotation=45):
    """
    Create a bar plot with improved styling and error handling.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        x_column (str): Column name for x-axis (categories)
        y_column (str): Column name for y-axis (values)
        figsize (tuple): Figure size
        color (str): Color for the bars
        title (str): Custom title for the plot
        rotation (int): Rotation angle for x-axis labels
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    try:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        
        # Validate inputs
        if x_column not in df.columns:
            ax.text(0.5, 0.5, f'X column "{x_column}" not found', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
            
        if y_column not in df.columns:
            ax.text(0.5, 0.5, f'Y column "{y_column}" not found', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        # Remove NaN values and check for data
        df_plot = df[[x_column, y_column]].dropna()
        if len(df_plot) == 0:
            ax.text(0.5, 0.5, 'No data to plot after removing NaN values', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        # Create bar plot
        sns.barplot(data=df_plot, x=x_column, y=y_column, color=color, ax=ax, alpha=0.8)
        
        # Set title and labels
        ax.set_title(title or f'{y_column} by {x_column}', color=color, fontweight='bold', fontsize=14)
        ax.set_xlabel(x_column, color=color, fontsize=12)
        ax.set_ylabel(y_column, color=color, fontsize=12)

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')
        
        # Style the plot
        for spine in ax.spines.values():
            spine.set_color(color)
        
        ax.tick_params(axis='both', colors=color)
        ax.grid(True, alpha=0.3, color=color, axis='y')
        
        # Add value labels on bars if not too many
        if len(df_plot[x_column].unique()) <= 20:
            for i, (idx, row) in enumerate(df_plot.iterrows()):
                if pd.notna(row[y_column]):
                    ax.text(i, row[y_column], f'{row[y_column]:.1f}', 
                           ha='center', va='bottom', fontsize=8, color='black')

        # Set transparent background
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Error creating bar plot:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
        ax.set_title('Bar Plot Error', color='red')
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        return fig

def plot_grouped_bar(df, x_columns, y_column, figsize=(12, 10), colors=None, title=None):
    """
    Create a grouped bar plot with improved styling and error handling.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        x_columns (list): List of column names for grouping
        y_column (str): Column name for y-axis values
        figsize (tuple): Figure size
        colors (list): List of colors for different groups
        title (str): Custom title for the plot
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    try:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        
        # Validate inputs
        missing_cols = [col for col in x_columns if col not in df.columns]
        if missing_cols:
            ax.text(0.5, 0.5, f'X columns not found: {missing_cols}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
            
        if y_column not in df.columns:
            ax.text(0.5, 0.5, f'Y column "{y_column}" not found', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            return fig
        
        # Set default colors if not provided
        if colors is None:
            colors = plt.cm.Set3(np.linspace(0, 1, len(x_columns)))
        
        width = 0.8 / len(x_columns)  # the width of the bars
        x = np.arange(len(df))  # the label locations

        for i, x_column in enumerate(x_columns):
            try:
                ax.bar(x + i * width, df[y_column], width, 
                      color=colors[i] if i < len(colors) else 'orange', 
                      label=x_column, alpha=0.8)
            except Exception as e:
                continue

        # Set title and labels
        ax.set_title(title or f'{y_column} by {", ".join(x_columns)}', 
                    color='orange', fontweight='bold', fontsize=14)
        ax.set_xlabel('Groups', color='orange', fontsize=12)
        ax.set_ylabel(y_column, color='orange', fontsize=12)

        ax.set_xticks(x + width * (len(x_columns) - 1) / 2)
        ax.set_xticklabels(df.index)

        # Style the plot
        for spine in ax.spines.values():
            spine.set_color('orange')
        
        ax.tick_params(axis='both', colors='orange')
        ax.grid(True, alpha=0.3, color='orange', axis='y')
        
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1), 
                 facecolor='black', framealpha=0.4, labelcolor='orange', edgecolor='orange')

        # Set transparent background
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Error creating grouped bar plot:\n{str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
        ax.set_title('Grouped Bar Plot Error', color='red')
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
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


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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


def get_stations():
    base_url = 'https://imag-data.bgs.ac.uk:/GIN_V1/GINServices?Request=GetCapabilities&format=json' 
    response = requests.get(base_url)
    data = response.json()
    dataframe_stations = pd.DataFrame.from_dict(data['ObservatoryList'])
    return dataframe_stations

def get_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d

def compare_stations(test_lat_lon, data_table, distance=1000, closest=False):
    table_updated = pd.DataFrame()
    distances = dict()
    for lat,lon,names in data_table[['Latitude', 'Longitude', 'Name']].values:
        harv_distance = get_haversine_distance(test_lat_lon[0], test_lat_lon[1], lat, lon)
        if harv_distance < distance:
            #print(f"Station {names} is at {round(harv_distance,2)} km from the test point")
            table_updated = pd.concat([table_updated, data_table[data_table['Name'] == names]])
            distances[names] = harv_distance
    if closest:
        closest_station = min(distances, key=distances.get)
        #print(f"The closest station is {closest_station} at {round(distances[closest_station],2)} km")
        table_updated = data_table[data_table['Name'] == closest_station].copy()  # Use .copy() to avoid SettingWithCopyWarning
        table_updated.loc[:, 'Distance'] = distances[closest_station]  # Use .loc for proper assignment
    return table_updated

def get_data(IagaCode, start_date, end_date):
    """
    Get magnetic data from BGS API with improved date handling.
    
    Args:
        IagaCode (str): Station code
        start_date (str): Start date in ISO format
        end_date (str): End date in ISO format
    
    Returns:
        dict: Magnetic data or None if request fails
    """
    try:
        # Handle various date formats more robustly
        def parse_api_date(date_str):
            """Parse date for API call, handling ISO format with time"""
            if isinstance(date_str, str):
                # Remove time component if present for API compatibility
                if 'T' in date_str:
                    date_str = date_str.split('T')[0]
                return datetime.datetime.strptime(date_str, '%Y-%m-%d')
            return pd.to_datetime(date_str).date()
        
        start_date_ = parse_api_date(start_date)
        end_date_ = parse_api_date(end_date)
        
    except Exception as e:
        st.warning(f"Date parsing error for {IagaCode}: {e}")
        return None

    duration = end_date_ - start_date_
    # Define the parameters for the request
    params = {
        'Request': 'GetData',
        'format': 'PNG',
        'testObsys': '0',
        'observatoryIagaCode': IagaCode,
        'samplesPerDay': 'minute',
        'publicationState': 'Best available',
        'dataStartDate': start_date,
        # make substraction
        'dataDuration': duration.days,
        'traceList': '1234',
        'colourTraces': 'true',
        'pictureSize': 'Automatic',
        'dataScale': 'Automatic',
        'pdfSize': '21,29.7',
    }

    base_url_json = 'https://imag-data.bgs.ac.uk:/GIN_V1/GINServices?Request=GetData&format=json'
    #base_url_img = 'https://imag-data.bgs.ac.uk:/GIN_V1/GINServices?Request=GetData&format=png'

    answer = None
    for base_url in [base_url_json]:#, base_url_img]:
        try:
            response = requests.get(base_url, params=params, timeout=30)
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                
                if 'application/json' in content_type:
                    # Expected JSON response
                    try:
                        answer = response.json()
                        # Validate that we have magnetic data
                        if not answer or not any(key in answer for key in ['X', 'Y', 'Z', 'S', 'datetime']):
                            st.warning(f"No magnetic data available for {IagaCode} on {start_date}")
                            return None
                    except ValueError as e:
                        st.warning(f"JSON parsing error for {IagaCode}: {e}")
                        return None
                        
                elif 'image' in content_type:
                    # Handle image response (not currently used)
                    st.info(f"Received image response for {IagaCode}")
                    return None
                    
                else:
                    # Unexpected content type - try to parse as JSON anyway
                    try:
                        answer = response.json()
                        if answer and any(key in answer for key in ['X', 'Y', 'Z', 'S', 'datetime']):
                            # Valid data despite unexpected content type
                            pass
                        else:
                            st.warning(f"No magnetic data in response for {IagaCode}")
                            return None
                    except:
                        st.warning(f"Unexpected content type '{content_type}' for {IagaCode}, cannot parse response")
                        return None
                        
            else:
                st.warning(f"API request failed for {IagaCode}: HTTP {response.status_code}")
                return None
                
        except requests.Timeout:
            st.warning(f"Request timeout for {IagaCode}")
            return None
        except requests.RequestException as e:
            st.warning(f"Request error for {IagaCode}: {e}")
            return None
        except Exception as e:
            st.warning(f"Unexpected error for {IagaCode}: {e}")
            return None
            
    return answer


# def get_data(IagaCode, start_date, end_date):
#     # Convert dates to datetime
#     try:
#         start_date_ = pd.to_datetime(start_date)
#         end_date_ = pd.to_datetime(end_date)
#     except ValueError as e:
#         print(f"Error: {e}")
#         return None, None

#     duration = (end_date_ - start_date_).days

#     # Define the parameters for the request
#     params = {
#         'Request': 'GetData',
#         'format': 'json',
#         'testObsys': '0',
#         'observatoryIagaCode': IagaCode,
#         'samplesPerDay': 'minute',
#         'publicationState': 'Best available',
#         'dataStartDate': start_date_.strftime('%Y-%m-%d'),
#         'dataDuration': duration,
#         'traceList': '1234',
#         'colourTraces': 'true',
#         'pictureSize': 'Automatic',
#         'dataScale': 'Automatic',
#         'pdfSize': '21,29.7',
#     }

#     base_url_json = 'https://imag-data.bgs.ac.uk:/GIN_V1/GINServices?Request=GetData&format=json'
#     base_url_img = 'https://imag-data.bgs.ac.uk:/GIN_V1/GINServices?Request=GetData&format=png'

#     try:
#         # Request JSON data
#         response_json = requests.get(base_url_json, params=params)
#         response_json.raise_for_status()  # Raises an error for bad status codes
#         data = response_json.json()

#         # Request Image
#         params['format'] = 'png'
#         response_img = requests.get(base_url_img, params=params)
#         response_img.raise_for_status()

#         # Save and display image if response is successful
#         if 'image' in response_img.headers.get('Content-Type'):
#             output_image_path = "plot_image.png"
#             with open(output_image_path, 'wb') as file:
#                 file.write(response_img.content)
#             print(f"Image successfully saved as {output_image_path}")

#             img = mpimg.imread(output_image_path)
#             plt.imshow(img)
#             plt.axis('off')
#             plt.show()
#             img_answer = Image.open(output_image_path)
#         else:
#             img_answer = None

#         return data, img_answer

#     except requests.RequestException as e:
#         print(f"Request failed: {e}")
#         return None, None
#     except ValueError as e:
#         print(f"JSON decode error: {e}")
#         return None, None

def parse_uap_date(date_value):
    """
    Comprehensive date parser for UAP data that handles various formats.
    
    Args:
        date_value: Date value in various formats (string, int, float, datetime)
        
    Returns:
        pd.Timestamp or pd.NaT: Parsed datetime or NaT if parsing fails
    """
    if pd.isna(date_value):
        return pd.NaT
    
    # If already a datetime, return as pandas Timestamp
    if isinstance(date_value, (pd.Timestamp, datetime.datetime)):
        return pd.to_datetime(date_value)
    
    # Convert to string for parsing
    date_str = str(date_value).strip()
    
    # Handle empty or invalid strings
    if not date_str or date_str.lower() in ['nan', 'none', 'null', '', 'unknown']:
        return pd.NaT
    
    # Try different parsing approaches
    try:
        # Method 1: Standard pandas to_datetime (handles ISO, common formats)
        parsed = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(parsed):
            return parsed
    except:
        pass
    
    try:
        # Method 2: Handle numeric dates (timestamps, Excel dates, etc.)
        if date_str.replace('.', '').replace('-', '').isdigit():
            numeric_val = float(date_str)
            
            # Handle Unix timestamps (seconds)
            if 1000000000 <= numeric_val <= 2000000000:  # Roughly 2001-2033
                return pd.to_datetime(numeric_val, unit='s')
            
            # Handle Unix timestamps (milliseconds)
            if 1000000000000 <= numeric_val <= 2000000000000:
                return pd.to_datetime(numeric_val, unit='ms')
            
            # Handle Excel date serial numbers
            if 1 <= numeric_val <= 100000:  # Excel date range
                return pd.to_datetime('1899-12-30') + pd.Timedelta(days=numeric_val)
            
            # Handle year-only dates
            if 1900 <= numeric_val <= 2100:
                return pd.to_datetime(f"{int(numeric_val)}-01-01")
                
    except:
        pass
    
    try:
        # Method 3: dateutil parser with fuzzy matching
        from dateutil import parser as date_parser
        parsed = date_parser.parse(date_str, fuzzy=True)
        return pd.to_datetime(parsed)
    except:
        pass
    
    try:
        # Method 4: Handle common UAP date formats
        common_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y', 
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M',
            '%d-%m-%Y',
            '%Y%m%d',
            '%m-%d-%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%Y-%m-%d %H:%M:%S.%f'
        ]
        
        for fmt in common_formats:
            try:
                parsed = pd.to_datetime(date_str, format=fmt)
                return parsed
            except:
                continue
    except:
        pass
    
    # If all methods fail, return NaT
    return pd.NaT

def clean_uap_data(dataset, lat, lon, date):
    """
    Clean UAP data by validating coordinates and dates.
    
    Args:
        dataset (pd.DataFrame): Input dataset
        lat (str): Latitude column name
        lon (str): Longitude column name  
        date (str): Date column name
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    try:
        # Create a copy to avoid modifying the original dataset
        processed = dataset.copy()
        
        # Filter rows where lat, lon, and date are not null
        processed = processed[processed[[lat, lon, date]].notnull().all(axis=1)]
        
        if len(processed) == 0:
            st.warning("No valid rows found after filtering for non-null lat/lon/date values")
            return processed
        
        # Converting 'Lat' and 'Long' columns to floats, handling errors
        processed[lat] = pd.to_numeric(processed[lat], errors='coerce')
        processed[lon] = pd.to_numeric(processed[lon], errors='coerce')
        
        st.info(f"Parsing {len(processed)} date values...")
        
        # Show sample of original date formats before parsing
        if len(processed) > 0:
            sample_dates = processed[date].dropna().head(5).tolist()
            if sample_dates:
                st.info(f"Sample original date formats: {sample_dates}")
        
        # Apply comprehensive date parsing using the specialized parser
        processed[date] = processed[date].apply(parse_uap_date)
        
        # Count successful parsing
        valid_dates = processed[date].notna().sum()
        total_dates = len(processed)
        st.info(f"Successfully parsed {valid_dates}/{total_dates} dates ({valid_dates/total_dates*100:.1f}%)")
        
        # Show sample of parsed dates
        if valid_dates > 0:
            sample_parsed = processed[date].dropna().head(5).tolist()
            st.info(f"Sample parsed dates: {sample_parsed}")
        
        # Filter out rows with invalid dates
        processed = processed[processed[date].notna()]
        
        if len(processed) == 0:
            st.warning("No valid dates found after parsing")
            return processed
        
        # Define safe date bounds for pandas (avoid timestamp overflow)
        min_safe_date = pd.Timestamp('1677-09-22')  # Pandas minimum safe date
        max_safe_date = pd.Timestamp('2262-04-11')  # Pandas maximum safe date
        
        # Filter dates within safe bounds
        date_mask = (processed[date] >= min_safe_date) & (processed[date] <= max_safe_date)
        before_filter = len(processed)
        processed = processed[date_mask]
        after_filter = len(processed)
        
        if before_filter > after_filter:
            st.info(f"Filtered {before_filter - after_filter} rows with dates outside safe range (1677-2262)")
        
        # Drop rows where lat/lon conversion failed (became NaN)
        before_coords = len(processed)
        processed = processed.dropna(subset=[lat, lon])
        after_coords = len(processed)
        
        if before_coords > after_coords:
            st.info(f"Removed {before_coords - after_coords} rows with invalid coordinates")
        
        # Final validation - ensure we have reasonable coordinate ranges
        processed = processed[
            (processed[lat].between(-90, 90)) & 
            (processed[lon].between(-180, 180))
        ]
        
        st.success(f"Data cleaning complete. Final dataset: {len(processed)} rows")
        
        return processed
        
    except Exception as e:
        st.error(f"Error in clean_uap_data: {str(e)}")
        st.info("Returning original dataset without cleaning")
        return dataset


def plot_overlapped_timeseries(data_list, event_times, window_hours=12, save_path=None):
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    fig.patch.set_alpha(0)  # Make figure background transparent

    components = ['X', 'Y', 'Z', 'S']
    colors = ['red', 'green', 'blue', 'black']

    for i, component in enumerate(components):
        axs[i].patch.set_alpha(0)  # Make subplot background transparent
        axs[i].set_ylabel(component, color='orange')
        axs[i].grid(True, color='orange', alpha=0.3)
        
        for spine in axs[i].spines.values():
            spine.set_color('orange')
        
        axs[i].tick_params(axis='both', colors='orange')  # Change tick color
        axs[i].set_title(f'{component}', color='orange')
        axs[i].set_xlabel('Time Difference from Event (hours)', color='orange')

        for j, (df, event_time) in enumerate(zip(data_list, event_times)):
            # Convert datetime column to UTC if it has timezone info, otherwise assume it's UTC
            df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
            
            # Convert event_time to UTC if it has timezone info, otherwise assume it's UTC
            event_time = pd.to_datetime(event_time).tz_localize(None)
            
            # Calculate time difference from event
            df['time_diff'] = (df['datetime'] - event_time).dt.total_seconds() / 3600  # Convert to hours
            
            # Filter data within the specified window
            df_window = df[(df['time_diff'] >= -window_hours) & (df['time_diff'] <= window_hours)]

            # normalize component data
            df_window[component] = (df_window[component] - df_window[component].mean()) / df_window[component].std()
            
            axs[i].plot(df_window['time_diff'], df_window[component], color=colors[i], alpha=0.7, label=f'Event {j+1}', linewidth=1)
        
        axs[i].axvline(x=0, color='red', linewidth=2, linestyle='--', label='Event Time')
        axs[i].set_xlim(-window_hours, window_hours)
        #axs[i].legend(loc='upper left', bbox_to_anchor=(1, 1))

    axs[-1].set_xlabel('Hours from Event', color='orange')
    fig.suptitle('Overlapped Time Series of Components', fontsize=16, color='orange')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, right=0.85)

    if save_path:
        fig.savefig(save_path, transparent=True, bbox_inches='tight')
        plt.close(fig)
        return save_path
    else:
        return fig
    
def plot_average_timeseries(data_list, event_times, window_hours=12, save_path=None):
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    fig.patch.set_alpha(0)  # Make figure background transparent

    components = ['X', 'Y', 'Z', 'S']
    colors = ['red', 'green', 'blue', 'black']

    for i, component in enumerate(components):
        axs[i].patch.set_alpha(0)
        axs[i].set_ylabel(component, color='orange')
        axs[i].grid(True, color='orange', alpha=0.3)

        for spine in axs[i].spines.values():
                spine.set_color('orange')
        
        axs[i].tick_params(axis='both', colors='orange')

        all_data = []
        time_diffs = []

        for j, (df, event_time) in enumerate(zip(data_list, event_times)):
            # Convert datetime column to UTC if it has timezone info, otherwise assume it's UTC
            df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)

            # Convert event_time to UTC if it has timezone info, otherwise assume it's UTC
            event_time = pd.to_datetime(event_time).tz_localize(None)

            # Calculate time difference from event
            df['time_diff'] = (df['datetime'] - event_time).dt.total_seconds() / 3600  # Convert to hours

            # Filter data within the specified window
            df_window = df[(df['time_diff'] >= -window_hours) & (df['time_diff'] <= window_hours)]

            # Normalize component data
            df_window[component] = (df_window[component] - df_window[component].mean())# / df_window[component].std()

            all_data.append(df_window[component].values)
            time_diffs.append(df_window['time_diff'].values)

        # Calculate average and standard deviation
        try:
            avg_data = np.mean(all_data, axis=0)
        except:
            avg_data = np.zeros_like(all_data[0])
        try:
            std_data = np.std(all_data, axis=0)
        except:
            std_data = np.zeros_like(avg_data)

        axs[-1].set_xlabel('Hours from Event', color='orange')
        fig.suptitle('Average Time Series of Components', fontsize=16, color='orange')

        # Plot average line
        axs[i].plot(time_diffs[0], avg_data, color=colors[i], label='Average')

        # Plot standard deviation as shaded region
        try:
            axs[i].fill_between(time_diffs[0], avg_data - std_data, avg_data + std_data, color=colors[i], alpha=0.2)
        except:
            pass

        axs[i].axvline(x=0, color='red', linewidth=2, linestyle='--', label='Event Time')
        axs[i].set_xlim(-window_hours, window_hours)
        # orange frame, orange label legend
        axs[i].legend(loc='upper right', bbox_to_anchor=(1, 1), facecolor='black', framealpha=.4, labelcolor='orange', edgecolor='orange')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, right=0.85)

    if save_path:
        fig.savefig(save_path, transparent=True, bbox_inches='tight')
        plt.close(fig)
        return save_path
    else:
        return fig
    
def align_series(reference, series):
    reference = reference.flatten()
    series = series.flatten()
    _, path = fastdtw(reference, series, dist=euclidean)
    aligned = np.zeros(len(reference))
    for ref_idx, series_idx in path:
        aligned[ref_idx] = series[series_idx]
    return aligned

def plot_average_timeseries_with_dtw(data_list, event_times, window_hours=12, save_path=None):
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    fig.patch.set_alpha(0)  # Make figure background transparent

    components = ['X', 'Y', 'Z', 'S']
    colors = ['red', 'green', 'blue', 'black']
    fig.text(0.02, 0.5, 'Geomagnetic Variation (nT)', va='center', rotation='vertical', color='orange')


    for i, component in enumerate(components):
        axs[i].patch.set_alpha(0)
        axs[i].set_ylabel(component, color='orange', rotation=90)
        axs[i].grid(True, color='orange', alpha=0.3)
        
        for spine in axs[i].spines.values():
            spine.set_color('orange')
        
        axs[i].tick_params(axis='both', colors='orange')

        all_aligned_data = []
        reference_df = None

        for j, (df, event_time) in enumerate(zip(data_list, event_times)):
            df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
            event_time = pd.to_datetime(event_time).tz_localize(None)
            df['time_diff'] = (df['datetime'] - event_time).dt.total_seconds() / 3600
            df_window = df[(df['time_diff'] >= -window_hours) & (df['time_diff'] <= window_hours)]
            df_window[component] = (df_window[component] - df_window[component].mean())# / df_window[component].std()
            
            if reference_df is None:
                reference_df = df_window
                all_aligned_data.append(reference_df[component].values)
            else:
                try:
                    aligned_series = align_series(reference_df[component].values, df_window[component].values)
                    all_aligned_data.append(aligned_series)
                except:
                    pass

        # Calculate average and standard deviation of aligned data
        all_aligned_data = np.array(all_aligned_data)
        avg_data = np.mean(all_aligned_data, axis=0)

        # round float to avoid sqrt errors
        def calculate_std(data):
            if data is not None and len(data) > 0:
                data = np.array(data)
                std_data = np.std(data)
                return std_data
            else:
                return "Data is empty or not a list"
            
        std_data = calculate_std(all_aligned_data)

        # Plot average line
        axs[i].plot(reference_df['time_diff'], avg_data, color=colors[i], label='Average')

        # Plot standard deviation as shaded region
        try:
            axs[i].fill_between(reference_df['time_diff'], avg_data - std_data, avg_data + std_data, color=colors[i], alpha=0.2)
        except TypeError as e:
            #print(f"Error: {e}")
            pass
            

        axs[i].axvline(x=0, color='red', linewidth=2, linestyle='--', label='Event Time')
        axs[i].set_xlim(-window_hours, window_hours)
        axs[i].legend(loc='upper right', bbox_to_anchor=(1, 1), facecolor='black', framealpha=.2, labelcolor='orange', edgecolor='orange')


    axs[-1].set_xlabel('Hours from Event', color='orange')
    fig.suptitle('Average Time Series of Components (FastDTW Aligned)', fontsize=16, color='orange')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, right=0.85, left=0.1)

    if save_path:
        fig.savefig(save_path, transparent=True, bbox_inches='tight')
        plt.close(fig)
        return save_path
    else:
        return fig

def plot_data_custom(df, date, save_path=None, subtitle=None):
    df['datetime'] = pd.to_datetime(df['datetime'])
    event = pd.to_datetime(date)
    window = timedelta(hours=12)
    x_min = event - window
    x_max = event + window

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.patch.set_alpha(0)  # Make figure background transparent

    components = ['X', 'Y', 'Z', 'S']
    colors = ['red', 'green', 'blue', 'black']

    fig.text(0.02, 0.5, 'Geomagnetic Variation (nT)', va='center', rotation='vertical', color='orange')

    # if df[component].isnull().all().all():
    #     return None
            
    for i, component in enumerate(components):
        axs[i].plot(df['datetime'], df[component], label=component, color=colors[i])
        axs[i].axvline(x=event, color='red', linewidth=2, label='Event', linestyle='--')
        axs[i].set_ylabel(component, color='orange', rotation=90)
        axs[i].set_xlim(x_min, x_max)
        axs[i].legend(loc='upper right', bbox_to_anchor=(1, 1), facecolor='black', framealpha=.2, labelcolor='orange', edgecolor='orange')
        axs[i].grid(True, color='orange', alpha=0.3)
        axs[i].patch.set_alpha(0)  # Make subplot background transparent
        
        for spine in axs[i].spines.values():
            spine.set_color('orange')
        
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axs[i].xaxis.set_major_locator(mdates.HourLocator(interval=1))
        axs[i].tick_params(axis='both', colors='orange')

    plt.setp(axs[-1].xaxis.get_majorticklabels(), rotation=45)
    axs[-1].set_xlabel('Hours', color='orange')
    fig.suptitle(f'Time Series of Components with Event Marks\n{subtitle}', fontsize=12, color='orange')
    
    plt.tight_layout()
    #plt.subplots_adjust(top=0.85)
    plt.subplots_adjust(top=0.85, right=0.85, left=0.1)


    if save_path:
        fig.savefig(save_path, transparent=True)
        plt.close(fig)
        return save_path
    else:
        return fig


def batch_requests(stations, dataset, lon, lat, date, distance=100):
    results = {"station": [], "data": [], "image": [], "custom_image": []}
    all_data = []
    all_event_times = []

    for lon_, lat_, date_ in dataset[[lon, lat, date]].values:
        test_lat_lon = (lat_, lon_)
        
        # Skip processing very old dates that likely don't have magnetic data
        try:
            parsed_date = pd.to_datetime(date_)
            if parsed_date.year < 1950:  # Magnetic data typically starts around 1950s
                st.info(f"Skipping very old date: {parsed_date.date()} (before 1950)")
                continue
        except:
            st.warning(f"Could not parse date: {date_}")
            continue
        
        try:
            # Format date properly for API (date only, no time)
            str_date = pd.to_datetime(date_).strftime('%Y-%m-%d')
        except:
            str_date = str(date_)
            
        twelve_hours = pd.Timedelta(hours=12)
        forty_eight_hours = pd.Timedelta(hours=48)
        
        try:
            # Calculate date range for magnetic data request
            base_date = pd.to_datetime(str_date)
            str_date_start = (base_date - twelve_hours).strftime('%Y-%m-%d')
            str_date_end = (base_date + forty_eight_hours).strftime('%Y-%m-%d')
        except Exception as e:
            st.warning(f"Date calculation error for {date_}: {e}")
            continue
        
        try:
            new_dataset = compare_stations(test_lat_lon, stations, distance=distance, closest=True)
            station_name = new_dataset['Name']
            station_distance = new_dataset['Distance']
            test_ = get_data(new_dataset.iloc[0]['IagaCode'], str_date_start, str_date_end)

            if test_ and any(test_.get(key) for key in ['X', 'Y', 'Z', 'S']):
                plotted = pd.DataFrame({
                    'datetime': test_['datetime'],
                    'X': test_.get('X', []),
                    'Y': test_.get('Y', []),
                    'Z': test_.get('Z', []),
                    'S': test_.get('S', []),
                })
                
                if plotted[['X', 'Y', 'Z', 'S']].any().any():
                    all_data.append(plotted)
                    all_event_times.append(pd.to_datetime(date_))
                    additional_data = f"Date: {date_}\nLat/Lon: {lat_}, {lon_}\nClosest station: {station_name.values[0]}\nDistance: {round(station_distance.values[0], 2)} km"
                    fig = plot_data_custom(plotted, date=pd.to_datetime(date_), save_path=None, subtitle=additional_data)
                    with st.status(f'Magnetic Data: {date_}', expanded=False) as status:
                        st.pyplot(fig)
                        status.update(f'Magnetic Data: {date_} - Finished!')
                else:
                    print(f"No data for X, Y, Z, or S for date: {date_}")
        except Exception as e:
            #print(f"An error occurred: {e}")
            pass

            # if test_:
            #     results["station"].append(new_dataset.iloc[0]['IagaCode'])
            #     results["data"].append(test_)
            #     plotted = pd.DataFrame({
            #         'datetime': test_['datetime'],
            #         'X': test_['X'],
            #         'Y': test_['Y'],
            #         'Z': test_['Z'],
            #         'S': test_['S'],
            #     })
        #         all_data.append(plotted)
        #         all_event_times.append(pd.to_datetime(date_))
        #         # print(date_)
        #         additional_data = f"Date: {date_}\nLat/Lon: {lat_}, {lon_}\nClosest station: {station_name.values[0]}\n Distance:{round(station_distance.values[0],2)} km"
        #         fig = plot_data_custom(plotted, date=pd.to_datetime(date_), save_path=None, subtitle =additional_data)
        #         with st.status(f'Magnetic Data: {date_}', expanded=False) as status:
        #             st.pyplot(fig)
        #             status.update(f'Magnetic Data: {date_} - Finished!')
        # except Exception as e:
        #     #print(f"An error occurred: {e}")
        #     pass

    if all_data:
        fig_overlapped = plot_overlapped_timeseries(all_data, all_event_times)
        display(fig_overlapped)
        plt.close(fig_overlapped)
        # fig_average = plot_average_timeseries(all_data, all_event_times)
        # st.pyplot(fig_average)
        fig_average_aligned = plot_average_timeseries_with_dtw(all_data, all_event_times)
        with st.status(f'Dynamic Time Warping Data', expanded=False) as stts:
            st.pyplot(fig_average_aligned)
    return results


df = pd.DataFrame()


# Upload dataset
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    stations = get_stations()
    st.write("Dataset Loaded:")
    df = filter_dataframe(df)
    st.dataframe(df)

    # Select columns
    with st.form(border=True, key='Select Columns for Analysis'):
        lon_col = st.selectbox("Select Longitude Column", df.columns)
        lat_col = st.selectbox("Select Latitude Column", df.columns)
        date_col = st.selectbox("Select Date Column", df.columns)
        distance = st.number_input("Enter Distance", min_value=0, value=100)
        if st.form_submit_button("Process Data"):
            cases = clean_uap_data(df, lat_col, lon_col, date_col)
            results = batch_requests(stations, cases, lon_col, lat_col, date_col, distance=distance)
