"""
Example usage of UAP Analytics utilities
Demonstrates best practices and common workflows
"""

import streamlit as st
import pandas as pd
from utils import (
    SessionStateManager, 
    DataProcessor, 
    UAP_Visualizer,
    MemoryManager,
    APIKeyValidator,
    create_uap_analysis_pipeline,
    log_performance
)

# Initialize session state at app start
SessionStateManager.initialize()

st.title("UAP Analytics Example")

# Example 1: Loading and Processing Data
st.header("1. Data Loading with Memory Optimization")

@st.cache_data
@log_performance
def load_uap_data(file_path):
    """Load UAP data with automatic optimization based on file size"""
    import os
    
    file_size_gb = os.path.getsize(file_path) / (1024**3)
    
    if file_size_gb > 1:
        st.info(f"Large file ({file_size_gb:.1f}GB) - using sampling")
        return MemoryManager.sample_large_dataset(file_path, sample_size=50000)
    else:
        data = DataProcessor.load_data(file_path)
        return MemoryManager.optimize_dataframe_memory(data)

# Example usage
if st.button("Load Sample Data"):
    try:
        data = load_uap_data("sample_data.h5")
        SessionStateManager.set('data', data)
        st.success(f"Loaded {len(data)} records")
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Example 2: Data Filtering with Visualizations
st.header("2. Interactive Data Filtering")

if SessionStateManager.exists('data'):
    data = SessionStateManager.get('data')
    
    # Apply filtering with automatic visualizations
    filtered_data = DataProcessor.filter_dataframe(data)
    
    # Save filtered data
    SessionStateManager.set('filtered_data', filtered_data)
    
    # Display results
    st.write(f"Filtered to {len(filtered_data)} records")
    st.dataframe(filtered_data.head())

# Example 3: API Key Validation
st.header("3. API Key Management")

api_key = st.text_input("Enter API Key", type="password")
api_provider = st.selectbox("Select Provider", ["openai", "cohere", "gemini"])

if api_key and st.button("Validate API Key"):
    # Check if already validated
    cache_key = f"{api_provider}_validated"
    
    if SessionStateManager.get(cache_key, False):
        st.info("API key already validated!")
    else:
        with st.spinner("Validating..."):
            validators = {
                'openai': APIKeyValidator.validate_openai_key,
                'cohere': APIKeyValidator.validate_cohere_key,
                'gemini': APIKeyValidator.validate_gemini_key
            }
            
            if validators[api_provider](api_key):
                SessionStateManager.set(cache_key, True)
                st.success("API key is valid!")
            else:
                st.error("Invalid API key")

# Example 4: Pipeline Processing
st.header("4. Data Pipeline Processing")

uploaded_file = st.file_uploader("Upload UAP Data", type=['csv', 'h5', 'xlsx'])

if uploaded_file and st.button("Process with Pipeline"):
    # Create and run pipeline
    pipeline = create_uap_analysis_pipeline()
    
    try:
        with st.spinner("Running pipeline..."):
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Run pipeline
            result = pipeline.run(temp_path)
            
            # Display results
            st.success("Pipeline completed!")
            st.dataframe(result.head())
            
            # Show execution summary
            summary = pipeline.get_execution_summary()
            st.write("Execution Summary:")
            st.dataframe(summary)
            
            # Clean up
            import os
            os.remove(temp_path)
            
    except Exception as e:
        st.error(f"Pipeline error: {e}")

# Example 5: Custom Visualizations
st.header("5. Cached Visualizations")

if SessionStateManager.exists('filtered_data'):
    data = SessionStateManager.get('filtered_data')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Categorical visualization
        cat_columns = data.select_dtypes(include=['object', 'category']).columns
        if len(cat_columns) > 0:
            selected_cat = st.selectbox("Select category column", cat_columns)
            if selected_cat:
                fig = UAP_Visualizer.plot_treemap(data, selected_cat, top_n=15)
                st.pyplot(fig)
    
    with col2:
        # Numeric visualization
        num_columns = data.select_dtypes(include=['number']).columns
        if len(num_columns) > 0:
            selected_num = st.selectbox("Select numeric column", num_columns)
            if selected_num:
                fig = UAP_Visualizer.plot_hist(data, selected_num)
                st.pyplot(fig)

# Example 6: Memory Management
st.header("6. Memory Monitoring")

col1, col2, col3 = st.columns(3)

memory_stats = MemoryManager.get_memory_usage()

col1.metric("Memory Used", f"{memory_stats['rss_mb']:.1f} MB")
col2.metric("Memory %", f"{memory_stats['percent']:.1f}%")
col3.metric("Available", f"{memory_stats['available_mb']:.1f} MB")

if st.button("Clear All Caches"):
    MemoryManager.clear_memory_cache()
    SessionStateManager.clear(['data', 'filtered_data'])
    st.success("Caches cleared!")
    st.experimental_rerun()

# Example 7: Session State Management
st.header("7. Session State Debugging")

with st.expander("Session State Inspector", expanded=False):
    # Display current session state
    st.json(SessionStateManager.get_state_summary())
    
    # Manual state management
    col1, col2 = st.columns(2)
    
    with col1:
        key = st.text_input("State Key")
        value = st.text_input("State Value")
        if st.button("Set State"):
            SessionStateManager.set(key, value)
            st.success(f"Set {key} = {value}")
    
    with col2:
        get_key = st.text_input("Get Key")
        if st.button("Get State"):
            value = SessionStateManager.get(get_key, "Not found")
            st.write(f"Value: {value}")

# Footer with performance tips
st.markdown("---")
st.markdown("""
### Performance Tips:
- 🚀 GPU acceleration is {'**enabled**' if torch.cuda.is_available() else '**disabled**'}
- 💾 Use chunked processing for files > 1GB
- 🔑 API keys are cached for 1 hour after validation
- 📊 Visualizations are cached automatically
- 🧹 Clear caches periodically to free memory
""")

if __name__ == "__main__":
    import torch
    
    # Display GPU status
    if torch.cuda.is_available():
        st.sidebar.success("🚀 GPU Acceleration Active")
        st.sidebar.write(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.info("�� Running on CPU") 