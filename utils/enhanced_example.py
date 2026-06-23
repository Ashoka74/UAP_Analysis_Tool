"""
Enhanced filtering and visualization example for UAP Data Analysis Tool
Demonstrates the improved dynamic filtering and visualization pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
from data_processing import DataProcessor
from visualization import UAP_Visualizer
from session_manager import SessionStateManager
import plotly.graph_objects as go

def main():
    """Main function demonstrating enhanced filtering and visualization"""
    
    st.title("🚀 Enhanced UAP Data Analysis Pipeline")
    st.markdown("### Dynamic Filtering & Interactive Visualization System")
    
    # Initialize session state
    SessionStateManager.initialize()
    
    # Load data with caching
    @st.cache_data
    def load_sample_data():
        """Load sample data for demonstration"""
        try:
            # Try to load the actual UAP dataset
            df = DataProcessor.load_data('final_ufoseti_dataset.h5', key='df')
            st.success(f"✅ Loaded UAP dataset with {len(df):,} records")
            return df
        except Exception as e:
            st.warning(f"Could not load UAP dataset: {e}")
            # Create sample data if real data not available
            np.random.seed(42)
            n_samples = 10000
            
            sample_data = {
                'date': pd.date_range('2020-01-01', periods=n_samples, freq='H'),
                'latitude': np.random.uniform(-90, 90, n_samples),
                'longitude': np.random.uniform(-180, 180, n_samples),
                'duration_minutes': np.random.exponential(10, n_samples),
                'shape': np.random.choice(['circle', 'triangle', 'disk', 'light', 'other'], n_samples),
                'color': np.random.choice(['white', 'red', 'orange', 'blue', 'green', 'unknown'], n_samples),
                'altitude': np.random.uniform(100, 50000, n_samples),
                'witnesses': np.random.poisson(2, n_samples) + 1,
                'credibility_score': np.random.beta(2, 5, n_samples),
                'description_length': np.random.lognormal(3, 1, n_samples).astype(int)
            }
            
            df = pd.DataFrame(sample_data)
            st.info(f"📊 Using sample dataset with {len(df):,} records for demonstration")
            return df
    
    # Load the data
    df = load_sample_data()
    
    # Sidebar for analysis options
    with st.sidebar:
        st.header("🔧 Analysis Options")
        
        analysis_mode = st.radio(
            "Select Analysis Mode",
            ["Enhanced Filtering", "Interactive Visualizations", "Dashboard View", "Performance Demo"]
        )
        
        enable_quick_filters = st.checkbox("Enable Quick Filters", value=False)
        enable_advanced_filters = st.checkbox("Enable Advanced Filters", value=True)
        max_viz_points = st.slider("Max Visualization Points", 1000, 50000, 10000, step=1000)
    
    # Main content based on selected mode
    if analysis_mode == "Enhanced Filtering":
        show_enhanced_filtering(df, enable_quick_filters, enable_advanced_filters)
        
    elif analysis_mode == "Interactive Visualizations":
        show_interactive_visualizations(df, max_viz_points)
        
    elif analysis_mode == "Dashboard View":
        show_dashboard_view(df, max_viz_points)
        
    elif analysis_mode == "Performance Demo":
        show_performance_demo(df)

def show_enhanced_filtering(df: pd.DataFrame, enable_quick_filters: bool, enable_advanced_filters: bool):
    """Demonstrate enhanced filtering capabilities"""
    
    st.header("🔍 Enhanced Dynamic Filtering")
    
    # Show data profile first
    with st.expander("📊 Data Profile Analysis", expanded=True):
        profile = DataProcessor.profile_data(df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Categorical Columns", len(profile['categorical_columns']))
        with col2:
            st.metric("Numeric Columns", len(profile['numeric_columns']))
        with col3:
            st.metric("DateTime Columns", len(profile['datetime_columns']))
        with col4:
            st.metric("Text Columns", len(profile['text_columns']))
        
        # Show column details
        if st.checkbox("Show detailed column analysis"):
            st.json(profile)
    
    # Apply enhanced filtering
    st.subheader("Apply Filters")
    filtered_df = DataProcessor.filter_dataframe_enhanced(
        df, 
        enable_quick_filters=enable_quick_filters,
        enable_advanced_filters=enable_advanced_filters
    )
    
    # Show filtered results
    if len(filtered_df) > 0:
        st.subheader("📋 Filtered Data Preview")
        st.dataframe(filtered_df.head(100), use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Filtered Data"):
                # In a real app, you'd save to a file
                SessionStateManager.set('last_filtered_data', filtered_df)
                st.success("Filtered data saved to session!")
        
        with col2:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="filtered_uap_data.csv",
                mime="text/csv"
            )

def show_interactive_visualizations(df: pd.DataFrame, max_points: int):
    """Demonstrate interactive visualization capabilities"""
    
    st.header("📊 Interactive Visualizations")
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = []
    
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Scatter Plot", "Histogram", "Treemap", "Correlation Matrix", "Time Series"]
    )
    
    if viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        with col2:
            y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col], key="scatter_y")
        with col3:
            color_col = st.selectbox("Color by", ["None"] + categorical_cols, key="scatter_color")
            color_col = None if color_col == "None" else color_col
        
        if st.button("Generate Scatter Plot"):
            fig = UAP_Visualizer.plot_interactive_scatter(
                df, x_col, y_col, color_col=color_col, max_points=max_points
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Histogram" and len(numeric_cols) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            hist_col = st.selectbox("Column to analyze", numeric_cols + categorical_cols, key="hist_col")
        with col2:
            bins = st.slider("Number of bins", 10, 100, 50, key="hist_bins")
        
        if st.button("Generate Histogram"):
            if hist_col in numeric_cols:
                fig = UAP_Visualizer.plot_interactive_histogram(df, hist_col, bins=bins)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # For categorical columns, use treemap instead
                fig = UAP_Visualizer.plot_interactive_treemap(df, hist_col, top_n=20)
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Treemap" and len(categorical_cols) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            tree_col = st.selectbox("Categorical column", categorical_cols, key="tree_col")
        with col2:
            top_n = st.slider("Top N categories", 5, 50, 20, key="tree_n")
        
        if st.button("Generate Treemap"):
            fig = UAP_Visualizer.plot_interactive_treemap(df, tree_col, top_n=top_n)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Matrix" and len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], key="corr_method")
        with col2:
            selected_cols = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:10], key="corr_cols")
        
        if selected_cols and st.button("Generate Correlation Matrix"):
            fig = UAP_Visualizer.plot_correlation_matrix(df[selected_cols], method=corr_method)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Time Series" and len(datetime_cols) > 0 and len(numeric_cols) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_col = st.selectbox("Date column", datetime_cols, key="ts_date")
        with col2:
            value_cols = st.multiselect("Value columns", numeric_cols, default=numeric_cols[:3], key="ts_values")
        with col3:
            resample_freq = st.selectbox("Resample frequency", ["None", "D", "W", "M"], key="ts_freq")
            resample_freq = None if resample_freq == "None" else resample_freq
        
        if value_cols and st.button("Generate Time Series"):
            fig = UAP_Visualizer.plot_time_series(df, date_col, value_cols, resample_freq=resample_freq)
            st.plotly_chart(fig, use_container_width=True)

def show_dashboard_view(df: pd.DataFrame, max_points: int):
    """Demonstrate dashboard capabilities"""
    
    st.header("📈 Interactive Dashboard")
    
    # Create multiple charts for dashboard
    charts = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(numeric_cols) >= 2 and len(categorical_cols) >= 1:
        with st.spinner("Generating dashboard charts..."):
            # Chart 1: Scatter plot
            if len(numeric_cols) >= 2:
                fig1 = UAP_Visualizer.plot_interactive_scatter(
                    df, numeric_cols[0], numeric_cols[1], 
                    color_col=categorical_cols[0] if categorical_cols else None,
                    max_points=max_points//4
                )
                charts.append(fig1)
            
            # Chart 2: Histogram
            if len(numeric_cols) >= 1:
                fig2 = UAP_Visualizer.plot_interactive_histogram(df, numeric_cols[0])
                charts.append(fig2)
            
            # Chart 3: Treemap
            if len(categorical_cols) >= 1:
                fig3 = UAP_Visualizer.plot_interactive_treemap(df, categorical_cols[0], top_n=15)
                charts.append(fig3)
            
            # Chart 4: Correlation matrix (if enough numeric columns)
            if len(numeric_cols) >= 3:
                fig4 = UAP_Visualizer.plot_correlation_matrix(df[numeric_cols[:5]])
                charts.append(fig4)
        
        # Display individual charts
        if len(charts) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(charts[0], use_container_width=True)
                if len(charts) >= 3:
                    st.plotly_chart(charts[2], use_container_width=True)
            
            with col2:
                st.plotly_chart(charts[1], use_container_width=True)
                if len(charts) >= 4:
                    st.plotly_chart(charts[3], use_container_width=True)
        
        # Combined dashboard view
        if st.button("Generate Combined Dashboard"):
            dashboard_fig = UAP_Visualizer.create_dashboard_layout(charts[:4], layout="2x2")
            st.plotly_chart(dashboard_fig, use_container_width=True)
    
    else:
        st.warning("Not enough numeric or categorical columns for dashboard generation")

def show_performance_demo(df: pd.DataFrame):
    """Demonstrate performance improvements"""
    
    st.header("⚡ Performance Demonstration")
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dataset Size", f"{len(df):,} rows")
    with col2:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col3:
        cache_info = st.session_state.get('cached_visualizations', {})
        st.metric("Cached Visualizations", len(cache_info))
    
    # Performance comparison
    st.subheader("🏃‍♂️ Speed Comparison")
    
    if st.button("Run Performance Test"):
        import time
        
        # Test data profiling speed
        start_time = time.time()
        profile = DataProcessor.profile_data(df)
        profile_time = time.time() - start_time
        
        # Test visualization generation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 1:
            start_time = time.time()
            fig = UAP_Visualizer.plot_interactive_histogram(df, numeric_cols[0])
            viz_time = time.time() - start_time
        else:
            viz_time = 0
        
        # Display results
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            st.metric("Data Profiling", f"{profile_time:.3f} seconds")
        with perf_col2:
            st.metric("Visualization Generation", f"{viz_time:.3f} seconds")
        
        # Show caching benefits
        st.info("🚀 Subsequent calls to the same functions will be much faster due to caching!")
        
        # Memory optimization demo
        if len(df) > 10000:
            st.subheader("📊 Smart Sampling Demo")
            
            sample_sizes = [1000, 5000, 10000, len(df)]
            sample_times = []
            
            for size in sample_sizes:
                if size <= len(df):
                    start_time = time.time()
                    sampled_df = UAP_Visualizer._smart_sampling(df, max_points=size)
                    sample_time = time.time() - start_time
                    sample_times.append(sample_time)
                else:
                    sample_times.append(None)
            
            # Create performance chart
            perf_data = {
                'Sample Size': [f"{size:,}" for size in sample_sizes if sample_times[sample_sizes.index(size)] is not None],
                'Processing Time': [t for t in sample_times if t is not None]
            }
            
            perf_df = pd.DataFrame(perf_data)
            st.line_chart(perf_df.set_index('Sample Size'))

if __name__ == "__main__":
    main()


