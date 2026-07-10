"""
Enhanced Data processing utilities for UAP Data Analysis Tool
Centralizes filtering, transformation, and data handling logic with intelligent caching and optimization
"""

import pandas as pd
import numpy as np
import streamlit as st
from dateutil import parser
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import json
import logging
import hashlib
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from functools import wraps, lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class DataProcessor:
    """Enhanced centralized data processing functionality with intelligent filtering and caching"""
    
    # Class-level cache for filter states
    _filter_cache = {}
    _data_profile_cache = {}
    
    @staticmethod
    def _get_dataframe_hash(df: pd.DataFrame) -> str:
        """Generate a robust hash for a dataframe to use as cache key.
        Handles unhashable cell types (lists/dicts/sets) gracefully.
        """
        try:
            # Fast path: hash pandas object bytes
            hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
            return hashlib.md5(hashed).hexdigest()[:16]
        except Exception:
            # Fallback: serialize to JSON with safe default handler
            try:
                serialized = df.to_json(orient='split', default_handler=str)
            except Exception:
                # Last resort: stringify cells
                serialized = df.astype(str).to_csv(index=True)
            return hashlib.md5(serialized.encode('utf-8')).hexdigest()[:16]
    
    @staticmethod
    @performance_monitor
    def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Profile dataframe to optimize filtering UI"""
        df_hash = DataProcessor._get_dataframe_hash(df)
        
        if df_hash in DataProcessor._data_profile_cache:
            return DataProcessor._data_profile_cache[df_hash]
        
        profile = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'categorical_columns': [],
            'numeric_columns': [],
            'datetime_columns': [],
            'text_columns': [],
            'high_cardinality_columns': []
        }
        
        for col in df.columns:
            # Safe nunique for unhashable values
            try:
                unique_count = df[col].nunique()
            except Exception:
                unique_count = None
            total_count = len(df)
            
            if is_categorical_dtype(df[col]) or (
                unique_count is not None and unique_count < 120 and 
                not is_datetime64_any_dtype(df[col]) and not is_numeric_dtype(df[col])
            ):
                profile['categorical_columns'].append({
                    'name': col,
                    'unique_count': unique_count if unique_count is not None else -1,
                    'top_values': (df[col].value_counts().head(10).to_dict() if unique_count is not None else
                                   df[col].astype(str).value_counts().head(10).to_dict())
                })
            elif is_numeric_dtype(df[col]):
                profile['numeric_columns'].append({
                    'name': col,
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()) if df[col].std() is not None else 0
                })
            elif is_datetime64_any_dtype(df[col]):
                # Already a datetime dtype
                try:
                    mn = df[col].dropna().min()
                    mx = df[col].dropna().max()
                    profile['datetime_columns'].append({
                        'name': col,
                        'min_date': str(mn) if pd.notna(mn) else '',
                        'max_date': str(mx) if pd.notna(mx) else '',
                    })
                except Exception:
                    profile['text_columns'].append(col)
            elif is_object_dtype(df[col]):
                # Treat as text by default; only classify as datetime if a strong sample converts cleanly
                try:
                    sample = df[col].dropna().astype(str).head(50)
                    # Require a strong signal (>70%) of parsable strings to consider datetime
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().mean() > 0.7:
                        converted = pd.to_datetime(df[col], errors='coerce')
                        mn = converted.dropna().min()
                        mx = converted.dropna().max()
                        profile['datetime_columns'].append({
                            'name': col,
                            'min_date': str(mn) if pd.notna(mn) else '',
                            'max_date': str(mx) if pd.notna(mx) else '',
                        })
                    else:
                        profile['text_columns'].append(col)
                except Exception:
                    profile['text_columns'].append(col)
            
            # Flag high cardinality columns that might need special handling
            if unique_count is not None and unique_count > total_count * 0.8:
                profile['high_cardinality_columns'].append(col)
        
        DataProcessor._data_profile_cache[df_hash] = profile
        return profile
    
    @staticmethod
    def filter_dataframe_enhanced(df: pd.DataFrame, for_map: bool = False, enable_quick_filters: bool = False, 
                                  enable_advanced_filters: bool = True) -> pd.DataFrame:
        """Enhanced filtering interface that builds on top of helper methods.
        Shows a compact data profile, optional quick filters, and an advanced filter builder.
        """
        from utils.session_manager import SessionStateManager
        
        SessionStateManager.initialize()
        profile = DataProcessor.profile_data(df)
        df_ = df.copy()
        
        # Top-level summary
        try:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Rows", f"{len(df_):,}")
            with c2:
                st.metric("Columns", len(df_.columns))
            with c3:
                st.metric("Memory", f"{profile['memory_usage'] / 1024**2:.1f} MB")
            with c4:
                st.metric("Nulls", sum(profile['null_counts'].values()))
        except Exception:
            pass
        
        # Quick filters
        if enable_quick_filters:
            with st.expander("Quick Filters", expanded=False):
                df_ = DataProcessor._render_quick_filters(df_, profile)
        
        # Advanced filters
        if enable_advanced_filters:
            with st.expander("Advanced Filters", expanded=False):
                categories = {
                    "📊 Categorical": [c['name'] for c in profile['categorical_columns']],
                    "🔢 Numeric": [c['name'] for c in profile['numeric_columns']],
                    "📅 DateTime": [c['name'] for c in profile['datetime_columns']],
                    "📝 Text": profile['text_columns']
                }
                options = []
                for k, cols in categories.items():
                    options.extend([f"{k}: {col}" for col in cols])
                try:
                    selected = st.multiselect("Select columns to filter", options)
                except:
                    try:
                        selected = st.multiselect("Select columns to filter on", options)
                    except:
                        pass
                active = {}
                for item in selected:
                    if ": " in item:
                        k, col = item.split(": ", 1)
                        active[col] = k
                if active:
                    df_ = DataProcessor._apply_intelligent_filters(df_, active, profile)
        
        # Results summary
        try:
            st.write(f"{len(df_)} rows ({(len(df_) / max(len(df),1)) * 100:.2f}%)")
        except Exception:
            pass
        
        if for_map:
            df_ = DataProcessor._prepare_for_mapping(df_)
        
        return df_

    @staticmethod
    @st.cache_data(show_spinner="Applying intelligent filters...")
    def filter_dataframe(df: pd.DataFrame, for_map: bool = True, enable_quick_filters: bool = False) -> pd.DataFrame:
        """
        Adds a UI on top of a dataframe to let viewers filter columns
        
        Args:
            df (pd.DataFrame): Original dataframe
            for_map (bool): If True, convert datetime columns to string at the end for mapping layers
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        from utils.visualization import UAP_Visualizer
        
        df_ = df.copy()
        
        # Get columns to filter
        to_filter_columns = st.multiselect("Filter dataframe on", df_.columns)
        
        date_column = None
        filtered_columns = []
        
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            
            # Handle categorical columns
            if is_categorical_dtype(df_[column]) or (df_[column].nunique() < 120 and 
                                                     not is_datetime64_any_dtype(df_[column]) and 
                                                     not is_numeric_dtype(df_[column])):
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df_[column].value_counts().index.tolist(),
                    default=list(df_[column].value_counts().index)
                )
                df_ = df_[df_[column].isin(user_cat_input)]
                filtered_columns.append(column)
                
                with st.status(f"Category Distribution: {column}", expanded=False) as stat:
                    st.pyplot(UAP_Visualizer.plot_treemap(df_, column))
                    
            # Handle numeric columns
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
                
                with st.status(f"Numerical Distribution: {column}", expanded=False) as stat_:
                    bins = int(round(len(df_[column].unique())-1)/2) if len(df_[column].unique()) > 2 else 10
                    st.pyplot(UAP_Visualizer.plot_hist(df_, column, bins=bins))
                    
            # Handle date columns
            elif is_object_dtype(df_[column]):
                # Only handle as date if strong evidence; otherwise treat as text
                df_ = DataProcessor._handle_date_column(df_, column, right, filtered_columns)
                if pd.api.types.is_datetime64_any_dtype(df_[column]):
                    date_column = column
                    
            # Handle text columns
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df_ = df_[df_[column].astype(str).str.contains(user_text_input, regex=True, na=False)]
                    
        # Display filter results
        st.write(f"{len(df_)} rows ({len(df_) / len(df) * 100:.2f}%)")

        # Optional: convert datetime columns to string for mapping libraries
        if for_map:
            for col in df_.columns:
                if is_datetime64_any_dtype(df_[col]):
                    try:
                        df_[col] = df_[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        # If conversion fails, leave column as-is
                        pass
        
        return df_
    
    @staticmethod
    def _handle_date_column(df_, column, right_col, filtered_columns):
        """Handle date column filtering and visualization"""
        from utils.visualization import UAP_Visualizer
        
        # Try to convert to datetime
        try:
            df_[column] = pd.to_datetime(df_[column], infer_datetime_format=True, errors='coerce')
        except Exception:
            try:
                df_[column] = df_[column].apply(parser.parse)
            except Exception:
                pass
                
        if is_datetime64_any_dtype(df_[column]):
            df_[column] = df_[column].dt.tz_localize(None)
            valid = df_[column].dropna()
            if valid.empty:
                return df_
            try:
                min_date = valid.min().date()
                max_date = valid.max().date()
            except (OverflowError, ValueError, OSError):
                return df_

            user_date_input = right_col.date_input(
                f"Values for {column}",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            
            if len(user_date_input) == 2:
                user_date_input = tuple(map(pd.to_datetime, user_date_input))
                start_date, end_date = user_date_input
                
                # Create date distribution visualization
                DataProcessor._visualize_date_distribution(df_, column, start_date, end_date)
                
                df_ = df_.loc[df_[column].between(start_date, end_date)]
                
        return df_
    
    @staticmethod
    def _visualize_date_distribution(df_, column, start_date, end_date):
        """Create date distribution visualizations"""
        from utils.visualization import UAP_Visualizer
        
        # Determine the most appropriate time unit for plot
        time_units = {
            'year': df_[column].dt.year,
            'month': df_[column].dt.to_period('M'),
            'day': df_[column].dt.date
        }
        unique_counts = {unit: col.nunique() for unit, col in time_units.items()}
        closest_to_36 = min(unique_counts, key=lambda k: abs(unique_counts[k] - 36))
        
        # Group by the most appropriate time unit and count occurrences
        grouped = df_.groupby(time_units[closest_to_36]).size().reset_index(name='count')
        grouped.columns = [column, 'count']
        
        # Create a complete date range
        if closest_to_36 == 'year':
            date_range = pd.date_range(start=f"{start_date.year}-01-01", 
                                     end=f"{end_date.year}-12-31", freq='YS')
        elif closest_to_36 == 'month':
            date_range = pd.date_range(start=start_date.replace(day=1), 
                                     end=end_date + pd.offsets.MonthEnd(0), freq='MS')
        else:  # day
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
        # Create a DataFrame with the complete date range
        complete_range = pd.DataFrame({column: date_range})
        
        # Convert the date column to the appropriate format
        if closest_to_36 == 'year':
            complete_range[column] = complete_range[column].dt.year
        elif closest_to_36 == 'month':
            complete_range[column] = complete_range[column].dt.to_period('M')
            
        # Merge the complete range with the grouped data
        final_data = pd.merge(complete_range, grouped, on=column, how='left').fillna(0)
        
        with st.status(f"Date Distributions: {column}", expanded=False) as stat:
            try:
                st.pyplot(UAP_Visualizer.plot_bar(final_data, column, 'count'))
            except Exception as e:
                st.error(f"Error plotting bar chart: {e}")
    
    @staticmethod
    @st.cache_data
    def load_data(file_path: str, key: str = 'df') -> pd.DataFrame:
        """Load data by name or path (local h5/parquet, else HF Hub) with caching"""
        try:
            from data_fetch import load_uap_dataset
            return load_uap_dataset(file_path, key=key)
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    @staticmethod
    def parse_responses_parallel(responses: Dict[str, str], max_workers: int = 4) -> Dict[str, Any]:
        """Parse responses in parallel for better performance"""
        
        def parse_single_response(key: str, value: str) -> Tuple[str, Any]:
            """Parse a single response with proper error handling"""
            try:
                return key, json.loads(value)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error for key {key}: {e}")
                try:
                    # Try with single quotes replaced
                    return key, json.loads(value.replace("'", '"'))
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse response for key {key}")
                    return key, None
        
        results = {}
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(parse_single_response, k, v): k 
                      for k, v in responses.items()}
            
            # Process completed tasks
            for future in as_completed(futures):
                key = futures[future]
                try:
                    k, parsed_value = future.result()
                    if parsed_value is not None:
                        results[k] = parsed_value
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Unexpected error parsing key {key}: {e}")
                    failed_count += 1
        
        logger.info(f"Successfully parsed {len(results)} responses, {failed_count} failed")
        return results
    
    @staticmethod
    def find_lat_lon_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Find latitude and longitude columns in dataframe"""
        lat_columns = df.columns[df.columns.str.lower().str.contains('lat')]
        lon_columns = df.columns[df.columns.str.lower().str.contains('lon|lng')]
        
        if len(lat_columns) > 0 and len(lon_columns) > 0:
            return lat_columns[0], lon_columns[0]
        else:
            return None, None
    
    @staticmethod
    def merge_clusters(df, column, distance_threshold: int = 3):
        """Merge similar clusters based on Levenshtein distance"""
        from Levenshtein import distance
        
        cluster_terms_ = df.__dict__.get('cluster_terms', [])
        cluster_labels_ = df.__dict__.get('cluster_labels', [])
        
        if not cluster_terms_ or not cluster_labels_:
            logger.warning("No cluster information found")
            return []
        
        merge_map = {}
        
        # Iterate over term pairs and decide on merging based on the distance
        for idx, term1 in enumerate(cluster_terms_):
            for jdx, term2 in enumerate(cluster_terms_):
                if idx < jdx and distance(term1, term2) <= distance_threshold:
                    # Find labels corresponding to jdx and map them to idx
                    labels_to_merge = [label for label, term_index in enumerate(cluster_labels_) 
                                     if term_index == jdx]
                    for label in labels_to_merge:
                        merge_map[label] = idx
        
        # Update the analyzer with the merged numeric labels 
        updated_cluster_labels_ = [merge_map.get(label, label) for label in cluster_labels_]
        
        df.__dict__['cluster_labels'] = updated_cluster_labels_
        
        # Update string labels to reflect merged labels
        updated_string_labels = [cluster_terms_[label] for label in updated_cluster_labels_]
        df.__dict__['string_labels'] = updated_string_labels
        
        return updated_string_labels
    
    @staticmethod
    def _render_quick_filters(df: pd.DataFrame, profile: Dict[str, Any]) -> pd.DataFrame:
        """Render quick filter presets for common filtering scenarios"""
        quick_filter_options = []
        
        # Add quick filters based on data characteristics
        if profile['categorical_columns']:
            quick_filter_options.extend([
                "🏆 Top Categories Only",
                "🔍 Remove Rare Categories",
                "📊 Balanced Sample"
            ])
        
        if profile['numeric_columns']:
            quick_filter_options.extend([
                "📈 Remove Outliers",
                "🎯 Focus on Normal Range"
            ])
        
        if profile['datetime_columns']:
            quick_filter_options.extend([
                "📅 Recent Data (Last Year)",
                "🕐 Recent Data (Last Month)"
            ])
        
        if quick_filter_options:
            selected_quick_filter = st.selectbox(
                "Apply Quick Filter",
                options=["None"] + quick_filter_options,
                help="Pre-configured filters for common analysis scenarios"
            )
            
            if selected_quick_filter != "None":
                return DataProcessor._apply_quick_filter(df, selected_quick_filter, profile)
        
        return df
    
    @staticmethod
    def _apply_quick_filter(df: pd.DataFrame, filter_type: str, profile: Dict[str, Any]) -> pd.DataFrame:
        """Apply predefined quick filters"""
        df_filtered = df.copy()
        
        if filter_type == "🏆 Top Categories Only":
            # Keep only top 5 categories for each categorical column
            for col_info in profile['categorical_columns']:
                col = col_info['name']
                top_categories = list(col_info['top_values'].keys())[:5]
                df_filtered = df_filtered[df_filtered[col].isin(top_categories)]
                
        elif filter_type == "🔍 Remove Rare Categories":
            # Remove categories that appear less than 1% of the time
            min_count = len(df) * 0.01
            for col_info in profile['categorical_columns']:
                col = col_info['name']
                value_counts = df[col].value_counts()
                frequent_values = value_counts[value_counts >= min_count].index
                df_filtered = df_filtered[df_filtered[col].isin(frequent_values)]
                
        elif filter_type == "📈 Remove Outliers":
            # Remove statistical outliers using IQR method
            for col_info in profile['numeric_columns']:
                col = col_info['name']
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)]
        
        return df_filtered
    
    @staticmethod
    def _apply_intelligent_filters(df: pd.DataFrame, active_filters: Dict[str, str], profile: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters with intelligent optimization and caching"""
        df_filtered = df.copy()
        
        for column, category in active_filters.items():
            st.write(f"### {category.replace('📊 ', '').replace('🔢 ', '').replace('📅 ', '').replace('📝 ', '')} Filter: {column}")
            
            if "Categorical" in category:
                df_filtered = DataProcessor._apply_categorical_filter(df_filtered, column, profile)
            elif "Numeric" in category:
                df_filtered = DataProcessor._apply_numeric_filter(df_filtered, column, profile)
            elif "DateTime" in category:
                df_filtered = DataProcessor._apply_datetime_filter(df_filtered, column, profile)
            elif "Text" in category:
                df_filtered = DataProcessor._apply_text_filter(df_filtered, column)
        
        return df_filtered
    
    @staticmethod
    def _apply_categorical_filter(df: pd.DataFrame, column: str, profile: Dict[str, Any]) -> pd.DataFrame:
        """Apply enhanced categorical filtering with visualization"""
        from utils.visualization import UAP_Visualizer
        
        col_info = next((col for col in profile['categorical_columns'] if col['name'] == column), None)
        if not col_info:
            return df
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Smart selection options
            selection_mode = st.radio(
                f"Selection mode for {column}",
                ["Select specific values", "Select top N", "Exclude values"],
                key=f"selection_mode_{column}"
            )
            
            if selection_mode == "Select specific values":
                available_values = df[column].value_counts().index.tolist()
                selected_values = st.multiselect(
                    f"Values for {column}",
                    options=available_values,
                    default=available_values[:min(5, len(available_values))],
                    key=f"multiselect_{column}"
                )
                df_filtered = df[df[column].isin(selected_values)]
                
            elif selection_mode == "Select top N":
                top_n = st.slider(f"Top N categories for {column}", 1, min(20, col_info['unique_count']), 5, key=f"topn_{column}")
                top_values = df[column].value_counts().head(top_n).index.tolist()
                df_filtered = df[df[column].isin(top_values)]
                
            else:  # Exclude values
                exclude_values = st.multiselect(
                    f"Exclude values from {column}",
                    options=df[column].value_counts().index.tolist(),
                    key=f"exclude_{column}"
                )
                df_filtered = df[~df[column].isin(exclude_values)]
        
        with col2:
            if len(df_filtered) > 0:
                with st.container():
                    st.pyplot(UAP_Visualizer.plot_treemap(df_filtered, column, top_n=15))
        
        return df_filtered
    
    @staticmethod
    def _apply_numeric_filter(df: pd.DataFrame, column: str, profile: Dict[str, Any]) -> pd.DataFrame:
        """Apply enhanced numeric filtering with statistics"""
        from utils.visualization import UAP_Visualizer
        
        col_info = next((col for col in profile['numeric_columns'] if col['name'] == column), None)
        if not col_info:
            return df

        # Binary boolean columns (values ⊆ {0, 1}) — Range/Percentile/StdDev are
        # meaningless here, so offer a 0/1 value picker instead of a slider.
        _nonnull = df[column].dropna()
        _uniq = set(_nonnull.unique())
        if _uniq and _uniq.issubset({0, 1}):
            bcol1, bcol2 = st.columns([1, 2])
            with bcol1:
                _opts = sorted(_uniq)
                picked = st.multiselect(
                    f"Values for {column}",
                    _opts,
                    default=_opts,
                    format_func=lambda v: f"{int(v)} — {'true' if int(v) == 1 else 'false'}",
                    key=f"binary_{column}",
                )
                df_filtered = df[df[column].isin(picked)]
            with bcol2:
                if len(df_filtered) > 0:
                    _vc = df_filtered[column].value_counts().sort_index()
                    _vc.index = _vc.index.map(
                        lambda v: f"{int(v)} ({'true' if int(v) == 1 else 'false'})"
                    )
                    st.bar_chart(_vc)
            return df_filtered

        col1, col2 = st.columns([1, 2])

        with col1:
            filter_mode = st.radio(
                f"Filter mode for {column}",
                ["Range", "Percentile", "Standard Deviation"],
                key=f"numeric_mode_{column}"
            )
            
            if filter_mode == "Range":
                min_val, max_val = st.slider(
                    f"Range for {column}",
                    min_value=col_info['min'],
                    max_value=col_info['max'],
                    value=(col_info['min'], col_info['max']),
                    key=f"range_{column}"
                )
                df_filtered = df[df[column].between(min_val, max_val)]
                
            elif filter_mode == "Percentile":
                lower_pct, upper_pct = st.slider(
                    f"Percentile range for {column}",
                    0, 100, (10, 90),
                    key=f"percentile_{column}"
                )
                lower_val = df[column].quantile(lower_pct / 100)
                upper_val = df[column].quantile(upper_pct / 100)
                df_filtered = df[df[column].between(lower_val, upper_val)]
                
            else:  # Standard Deviation
                std_range = st.slider(
                    f"Standard deviations from mean for {column}",
                    0.5, 3.0, 2.0, step=0.5,
                    key=f"std_{column}"
                )
                mean_val = col_info['mean']
                std_val = col_info['std']
                lower_bound = mean_val - (std_range * std_val)
                upper_bound = mean_val + (std_range * std_val)
                df_filtered = df[df[column].between(lower_bound, upper_bound)]
        
        with col2:
            if len(df_filtered) > 0:
                bins = min(50, max(10, len(df_filtered[column].unique())))
                st.pyplot(UAP_Visualizer.plot_hist(df_filtered, column, bins=bins))
        
        return df_filtered
    
    @staticmethod
    def _apply_datetime_filter(df: pd.DataFrame, column: str, profile: Dict[str, Any]) -> pd.DataFrame:
        """Apply enhanced datetime filtering with multiple modes and preview."""
        # Ensure datetime dtype
        try:
            df_local = df.copy()
            df_local[column] = pd.to_datetime(df_local[column], errors='coerce')
        except Exception:
            st.warning(f"Could not parse {column} as datetime")
            return df

        if not is_datetime64_any_dtype(df_local[column]):
            st.info(f"{column} is not datetime-like; skipping date filter")
            return df

        # Normalize timezone and precision
        try:
            df_local[column] = df_local[column].dt.tz_localize(None)
        except Exception:
            pass
        df_local[column] = df_local[column].dt.floor('ms')

        valid = df_local[column].dropna()
        if valid.empty:
            st.info(f"No valid datetime values in {column}")
            return df

        try:
            min_date = valid.min().date()
            max_date = valid.max().date()
        except (OverflowError, ValueError, OSError):
            st.info(f"{column} contains dates outside the supported range; skipping date filter")
            return df

        col1, col2 = st.columns([1, 2])
        with col1:
            mode = st.radio(
                f"Date filter for {column}",
                ["Date Range", "Relative Period", "Specific Years"],
                key=f"dt_mode_{column}"
            )

            if mode == "Date Range":
                start_date, end_date = st.date_input(
                    f"Range for {column}",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key=f"dt_range_{column}"
                )
                if isinstance(start_date, tuple):
                    start_date, end_date = start_date
                mask = df_local[column].dt.date.between(start_date, end_date)
                df_filtered = df_local[mask]

            elif mode == "Relative Period":
                choice = st.selectbox(
                    f"Relative period for {column}",
                    ["Last 7 days", "Last 30 days", "Last 90 days", "Last year"],
                    key=f"dt_rel_{column}"
                )
                today = pd.Timestamp.now().normalize()
                if choice == "Last 7 days":
                    cutoff = today - pd.Timedelta(days=7)
                elif choice == "Last 30 days":
                    cutoff = today - pd.Timedelta(days=30)
                elif choice == "Last 90 days":
                    cutoff = today - pd.Timedelta(days=90)
                else:
                    cutoff = today - pd.Timedelta(days=365)
                mask = df_local[column] >= cutoff
                df_filtered = df_local[mask]

            else:  # Specific Years
                years = sorted(valid.dt.year.unique())
                selected_years = st.multiselect(
                    f"Years for {column}", years,
                    default=years[-3:] if len(years) >= 3 else years,
                    key=f"dt_years_{column}"
                )
                mask = df_local[column].dt.year.isin(selected_years)
                df_filtered = df_local[mask]

        with col2:
            # Preview distribution bar using helper
            try:
                start = df_filtered[column].min() if not df_filtered.empty else valid.min()
                end = df_filtered[column].max() if not df_filtered.empty else valid.max()
                if pd.notna(start) and pd.notna(end):
                    DataProcessor._visualize_date_distribution(df_local, column, start, end)
            except Exception:
                pass

        return df_filtered

    @staticmethod
    def _apply_text_filter(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply flexible text filtering: contains/starts/ends/regex/length."""
        col1, col2 = st.columns([1, 2])
        with col1:
            mode = st.radio(
                f"Text filter for {column}",
                ["Contains", "Starts with", "Ends with", "Regex", "Length"],
                key=f"txt_mode_{column}"
            )
            if mode == "Length":
                lengths = df[column].astype(str).str.len()
                min_len = int(lengths.min() if len(lengths) else 0)
                max_len = int(lengths.max() if len(lengths) else 0)
                lo, hi = st.slider(
                    f"Length range for {column}",
                    min_len, max_len, (min_len, max_len), key=f"txt_len_{column}"
                )
                mask = lengths.between(lo, hi)
                return df[mask]
            else:
                query = st.text_input(f"Search in {column}", key=f"txt_q_{column}")
                if not query:
                    return df
                series = df[column].astype(str)
                try:
                    if mode == "Contains":
                        mask = series.str.contains(query, case=False, na=False)
                    elif mode == "Starts with":
                        mask = series.str.startswith(query, na=False)
                    elif mode == "Ends with":
                        mask = series.str.endswith(query, na=False)
                    else:  # Regex
                        mask = series.str.contains(query, regex=True, na=False)
                except Exception:
                    st.warning("Invalid pattern; no filter applied")
                    return df
                return df[mask]

    @staticmethod
    def _prepare_for_mapping(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for mapping libraries by converting datetime columns to strings"""
        df_map = df.copy()
        
        for col in df_map.columns:
            if is_datetime64_any_dtype(df_map[col]):
                try:
                    df_map[col] = df_map[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    # If conversion fails, leave column as-is
                    pass
        
        return df_map 