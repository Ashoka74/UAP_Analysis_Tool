"""
Enhanced centralized visualization utilities for UAP Data Analysis Tool
Consolidates all plotting functions with advanced caching, progressive rendering, and performance optimization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# import squarify
import matplotlib.colors as mcolors
import textwrap
import streamlit as st
from functools import lru_cache
import hashlib
import time
import logging
from typing import Optional, Union, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UAP_Visualizer:
    """Centralized visualization class for all UAP data plots"""
    
    @staticmethod
    @st.cache_data
    def plot_treemap(df, column, top_n=32):
        """Generate a treemap visualization with caching"""
        # Create a hash of the dataframe for cache key
        df_hash = hashlib.md5(pd.util.hash_pandas_object(df[column]).values).hexdigest()
        
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
        n_colors = len(sizes)
        colors = plt.cm.Oranges(np.linspace(0.3, 0.9, n_colors))[::-1]

        # Get % of each category
        percents = sizes / sizes.sum()

        # Prepare labels with percentages
        labels = [f'{label}\n {percent:.1%}' for label, percent in zip(labels, percents)]

        fig, ax = plt.subplots(figsize=(20, 12))

        # Plot the treemap
        # squarify.plot(sizes=sizes, label=labels, alpha=0.7, pad=True, color=colors, text_kwargs={'fontsize': 10})

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

        # Set transparent background
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        return fig

    @staticmethod
    @st.cache_data
    def plot_hist(df, column, bins=10, kde=True):
        """Generate histogram with caching"""
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data=df, x=column, kde=kde, bins=bins, color='orange')
        
        # Apply orange theme
        UAP_Visualizer._apply_orange_theme(ax, fig)
        
        return fig

    @staticmethod
    @st.cache_data
    def plot_line(df, x_column, y_columns, figsize=(12, 10), color='orange', title=None, rolling_mean_value=2):
        """Generate line plot with caching"""
        import matplotlib.cm as cm
        
        # Sort the dataframe by the date column
        df = df.sort_values(by=x_column).copy()

        # Calculate rolling mean for each y_column
        if rolling_mean_value and len(df) > rolling_mean_value:
            df[y_columns] = df[y_columns].rolling(len(df) // rolling_mean_value).mean()

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        colors = cm.Oranges(np.linspace(0.2, 1, len(y_columns)))

        # Plot each y_column as a separate line with a different color
        for i, y_column in enumerate(y_columns):
            df.plot(x=x_column, y=y_column, ax=ax, color=colors[i], label=y_column, linewidth=.5)

        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

        # Format x_column as date if it is
        if pd.api.types.is_datetime64_any_dtype(df[x_column]):
            df[x_column] = pd.to_datetime(df[x_column]).dt.date

        # Set title, labels, and legend
        ax.set_title(title or f'{", ".join(y_columns)} over {x_column}', color=color, fontweight='bold')
        ax.set_xlabel(x_column, color=color)
        ax.set_ylabel(', '.join(y_columns), color=color)

        # Apply orange theme
        UAP_Visualizer._apply_orange_theme(ax, fig)
        
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1), facecolor='black', 
                 framealpha=.4, labelcolor='orange', edgecolor='orange')

        return fig

    @staticmethod
    @st.cache_data  
    def plot_bar(df, x_column, y_column, figsize=(12, 10), color='orange', title=None, rotation=45):
        """Generate bar plot with caching"""
        fig, ax = plt.subplots(figsize=figsize)

        sns.barplot(data=df, x=x_column, y=y_column, color=color, ax=ax)

        ax.set_title(title if title else f'{y_column} by {x_column}', color=color, fontweight='bold')
        ax.set_xlabel(x_column, color=color)
        ax.set_ylabel(y_column, color=color)

        plt.xticks(rotation=rotation)

        # Apply orange theme
        UAP_Visualizer._apply_orange_theme(ax, fig)

        return fig

    @staticmethod
    @st.cache_data
    def plot_grouped_bar(df, x_columns, y_column, figsize=(12, 10), colors=None, title=None):
        """Generate grouped bar plot with caching"""
        fig, ax = plt.subplots(figsize=figsize)

        width = 0.8 / len(x_columns)  # the width of the bars
        x = np.arange(len(df))  # the label locations

        for i, x_column in enumerate(x_columns):
            offset = i * width - (len(x_columns) - 1) * width / 2
            ax.bar(x + offset, df[y_column], width, 
                  color=colors[i] if colors else None, label=x_column)

        ax.set_title(title if title else f'{y_column} by {", ".join(x_columns)}', 
                    color='orange', fontweight='bold')
        ax.set_xlabel('Groups', color='orange')
        ax.set_ylabel(y_column, color='orange')

        ax.set_xticks(x)
        ax.set_xticklabels(df.index)

        # Apply orange theme
        UAP_Visualizer._apply_orange_theme(ax, fig)
        
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1), facecolor='black', 
                 framealpha=.4, labelcolor='orange', edgecolor='orange')

        return fig

    @staticmethod
    def _apply_orange_theme(ax, fig):
        """Apply consistent orange theme to plots"""
        ax.spines['bottom'].set_color('orange')
        ax.spines['top'].set_color('orange')
        ax.spines['right'].set_color('orange')
        ax.spines['left'].set_color('orange')
        ax.xaxis.label.set_color('orange')
        ax.yaxis.label.set_color('orange')
        ax.tick_params(axis='x', colors='orange')
        ax.tick_params(axis='y', colors='orange')
        if hasattr(ax, 'title'):
            ax.title.set_color('orange')
        
        # Set transparent background
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    
    @staticmethod
    def _get_optimal_sample_size(df_length: int, max_points: int = 10000) -> int:
        """Calculate optimal sample size for large datasets to maintain performance"""
        if df_length <= max_points:
            return df_length
        
        # Use logarithmic scaling for very large datasets
        if df_length > 100000:
            return min(max_points, int(max_points * 0.8))
        else:
            return min(max_points, int(df_length * 0.5))
    
    @staticmethod
    def _smart_sampling(df: pd.DataFrame, max_points: int = 10000) -> pd.DataFrame:
        """Intelligent sampling that preserves data distribution"""
        if len(df) <= max_points:
            return df
        
        # For large datasets, use stratified sampling if possible
        try:
            # Try to identify categorical columns for stratified sampling
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) > 0:
                # Use the first categorical column for stratification
                strat_col = categorical_cols[0]
                sample_ratio = max_points / len(df)
                
                sampled_dfs = []
                for category in df[strat_col].unique():
                    category_df = df[df[strat_col] == category]
                    category_sample_size = max(1, int(len(category_df) * sample_ratio))
                    sampled_dfs.append(category_df.sample(n=category_sample_size, random_state=42))
                
                return pd.concat(sampled_dfs, ignore_index=True)
            else:
                # Random sampling for datasets without clear categories
                return df.sample(n=max_points, random_state=42)
                
        except Exception as e:
            logger.warning(f"Stratified sampling failed: {e}. Using random sampling.")
            return df.sample(n=max_points, random_state=42)
    
    @staticmethod
    @st.cache_data
    def plot_interactive_scatter(df: pd.DataFrame, x_col: str, y_col: str, 
                                color_col: Optional[str] = None, size_col: Optional[str] = None,
                                title: Optional[str] = None, max_points: int = 5000) -> go.Figure:
        """Create interactive scatter plot with intelligent sampling for large datasets"""
        
        # Apply smart sampling for performance
        df_sample = UAP_Visualizer._smart_sampling(df, max_points)
        
        # Create the scatter plot
        fig = px.scatter(
            df_sample, 
            x=x_col, 
            y=y_col,
            color=color_col,
            size=size_col,
            title=title or f"{y_col} vs {x_col}",
            hover_data=df_sample.columns.tolist()[:5],  # Limit hover data for performance
            template="plotly_dark"
        )
        
        # Update layout for better appearance
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='orange'),
            title_font=dict(color='orange', size=16),
            height=600
        )
        
        # Add sampling info if data was sampled
        if len(df_sample) < len(df):
            fig.add_annotation(
                text=f"Showing {len(df_sample):,} of {len(df):,} points",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(color="orange", size=10),
                bgcolor="rgba(0,0,0,0.5)"
            )
        
        return fig
    
    @staticmethod
    @st.cache_data
    def plot_interactive_histogram(df: pd.DataFrame, column: str, bins: int = 50, 
                                  title: Optional[str] = None) -> go.Figure:
        """Create interactive histogram with enhanced features"""
        
        fig = px.histogram(
            df, 
            x=column,
            nbins=bins,
            title=title or f"Distribution of {column}",
            template="plotly_dark",
            marginal="box"  # Add box plot on top
        )
        
        # Update layout
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='orange'),
            title_font=dict(color='orange', size=16),
            height=500
        )
        
        # Update traces for orange theme
        fig.update_traces(marker_color='orange', marker_line_color='darkorange', marker_line_width=1)
        
        return fig
    
    @staticmethod
    @st.cache_data
    def plot_interactive_treemap(df: pd.DataFrame, column: str, top_n: int = 20) -> go.Figure:
        """Create interactive treemap with drill-down capabilities"""
        
        # Get value counts
        value_counts = df[column].value_counts().head(top_n)
        
        # Create treemap
        fig = go.Figure(go.Treemap(
            labels=value_counts.index,
            values=value_counts.values,
            parents=[""] * len(value_counts),
            textinfo="label+value+percent entry",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentEntry}<extra></extra>",
            marker=dict(
                colorscale="Oranges",
                colorbar=dict(title="Count")
            )
        ))
        
        fig.update_layout(
            title=f"Distribution of {column} (Top {top_n})",
            font=dict(color='orange'),
            title_font=dict(color='orange', size=16),
            paper_bgcolor='rgba(0,0,0,0)',
            height=600
        )
        
        return fig
    
    @staticmethod
    @st.cache_data
    def plot_correlation_matrix(df: pd.DataFrame, method: str = 'pearson', 
                               figsize: Tuple[int, int] = (12, 10)) -> go.Figure:
        """Create interactive correlation matrix heatmap"""
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation analysis")
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=method)
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Correlation Matrix ({method.title()})",
            font=dict(color='orange'),
            title_font=dict(color='orange', size=16),
            paper_bgcolor='rgba(0,0,0,0)',
            height=600,
            width=800
        )
        
        return fig
    
    @staticmethod
    @st.cache_data
    def plot_time_series(df: pd.DataFrame, date_col: str, value_cols: List[str],
                        title: Optional[str] = None, resample_freq: Optional[str] = None) -> go.Figure:
        """Create interactive time series plot with resampling options"""
        
        # Ensure date column is datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Resample if specified
        if resample_freq:
            df_resampled = df.set_index(date_col)[value_cols].resample(resample_freq).mean().reset_index()
            df = df_resampled
        
        # Create subplots if multiple value columns
        if len(value_cols) > 1:
            fig = make_subplots(
                rows=len(value_cols), cols=1,
                shared_xaxes=True,
                subplot_titles=value_cols,
                vertical_spacing=0.05
            )
            
            colors = px.colors.qualitative.Set1[:len(value_cols)]
            
            for i, col in enumerate(value_cols):
                fig.add_trace(
                    go.Scatter(
                        x=df[date_col],
                        y=df[col],
                        name=col,
                        line=dict(color=colors[i]),
                        hovertemplate=f"<b>{col}</b><br>Date: %{{x}}<br>Value: %{{y}}<extra></extra>"
                    ),
                    row=i+1, col=1
                )
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df[date_col],
                    y=df[value_cols[0]],
                    name=value_cols[0],
                    line=dict(color='orange'),
                    hovertemplate=f"<b>{value_cols[0]}</b><br>Date: %{{x}}<br>Value: %{{y}}<extra></extra>"
                )
            )
        
        fig.update_layout(
            title=title or f"Time Series: {', '.join(value_cols)}",
            font=dict(color='orange'),
            title_font=dict(color='orange', size=16),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400 * len(value_cols) if len(value_cols) > 1 else 500,
            showlegend=len(value_cols) > 1
        )
        
        # Add range selector
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        return fig
    
    @staticmethod
    def create_dashboard_layout(charts: List[go.Figure], layout: str = "2x2") -> go.Figure:
        """Create a dashboard layout with multiple charts"""
        
        if layout == "2x2" and len(charts) <= 4:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[f"Chart {i+1}" for i in range(len(charts))],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            positions = [(1,1), (1,2), (2,1), (2,2)]
            
            for i, chart in enumerate(charts):
                if i < 4:
                    row, col = positions[i]
                    for trace in chart.data:
                        fig.add_trace(trace, row=row, col=col)
                        
        elif layout == "vertical":
            fig = make_subplots(
                rows=len(charts), cols=1,
                subplot_titles=[f"Chart {i+1}" for i in range(len(charts))],
                vertical_spacing=0.1
            )
            
            for i, chart in enumerate(charts):
                for trace in chart.data:
                    fig.add_trace(trace, row=i+1, col=1)
        
        fig.update_layout(
            height=800,
            title="UAP Data Analysis Dashboard",
            font=dict(color='orange'),
            title_font=dict(color='orange', size=20),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        return fig