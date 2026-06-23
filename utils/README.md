# UAP Analytics Utilities Package

This package contains centralized utilities for the UAP Data Analysis Tool, designed to improve code organization, performance, and maintainability.

## Components

### 1. Visualization (`visualization.py`)
Centralized plotting functions with caching for improved performance.

```python
from utils import UAP_Visualizer

# Create a treemap
fig = UAP_Visualizer.plot_treemap(df, 'category_column', top_n=20)

# Create a histogram
fig = UAP_Visualizer.plot_hist(df, 'numeric_column', bins=30)

# Create a line plot
fig = UAP_Visualizer.plot_line(df, 'date_column', ['value1', 'value2'])
```

### 2. Data Processing (`data_processing.py`)
Centralized data filtering, loading, and transformation utilities.

```python
from utils import DataProcessor

# Filter dataframe with UI
filtered_df = DataProcessor.filter_dataframe(df)

# Load data with caching
data = DataProcessor.load_data('file.h5')

# Parse JSON responses in parallel
parsed = DataProcessor.parse_responses_parallel(responses_dict)

# Find lat/lon columns
lat_col, lon_col = DataProcessor.find_lat_lon_columns(df)
```

### 3. Session State Management (`session_manager.py`)
Centralized session state handling for Streamlit apps.

```python
from utils import SessionStateManager

# Initialize session state
SessionStateManager.initialize()

# Get/Set values
value = SessionStateManager.get('key', default_value)
SessionStateManager.set('key', value)

# Update multiple values
SessionStateManager.update({'key1': value1, 'key2': value2})

# Clear specific keys or all
SessionStateManager.clear(['key1', 'key2'])
```

### 4. API Key Validation (`api_validators.py`)
Cached API key validation to reduce redundant API calls.

```python
from utils import APIKeyValidator

# Validate individual keys
is_valid = APIKeyValidator.validate_openai_key(api_key)
is_valid = APIKeyValidator.validate_cohere_key(api_key)
is_valid = APIKeyValidator.validate_gemini_key(api_key)

# Validate multiple keys
results = APIKeyValidator.validate_all_keys({
    'openai': openai_key,
    'cohere': cohere_key,
    'gemini': gemini_key
})
```

### 5. Memory Management (`memory_manager.py`)
Utilities for handling large datasets efficiently.

```python
from utils import MemoryManager

# Get memory usage
stats = MemoryManager.get_memory_usage()

# Process large file in chunks
iterator = MemoryManager.get_data_iterator('large_file.h5', chunksize=10000)
result = MemoryManager.process_data_in_chunks(iterator, process_func)

# Optimize DataFrame memory
optimized_df = MemoryManager.optimize_dataframe_memory(df)

# Sample large dataset
sample = MemoryManager.sample_large_dataset('huge_file.h5', sample_size=10000)
```

### 6. Pipeline Architecture (`pipeline.py`)
ETL pipeline pattern for structured data processing.

```python
from utils import UAP_Pipeline, PipelineComponents, create_uap_analysis_pipeline

# Create custom pipeline
pipeline = UAP_Pipeline("My Pipeline")
pipeline.add_extractor("Load Data", PipelineComponents.extract_from_file, use_chunks=True)
pipeline.add_transformer("Parse JSON", PipelineComponents.parse_json_responses)
pipeline.add_validator("Validate Schema", PipelineComponents.validate_schema, required_columns=['date'])
pipeline.add_loader("Save Cache", PipelineComponents.save_to_cache, cache_key='results')

# Run pipeline
result = pipeline.run(initial_data='data.csv')

# Use pre-built pipeline
standard_pipeline = create_uap_analysis_pipeline()
result = standard_pipeline.run('uap_data.h5')
```

### 7. Logging Configuration (`logger_config.py`)
Centralized logging with decorators for performance tracking.

```python
from utils import setup_logging, log_performance, log_errors

# Setup custom logging
logger = setup_logging(log_level="DEBUG", log_file="app.log")

# Use decorators
@log_performance
@log_errors
def process_data(df):
    # Function will log execution time and any errors
    return df.groupby('category').mean()
```

## GPU Acceleration

The utils package automatically detects and uses GPU acceleration when available:

```python
# In app.py
import torch
GPU_AVAILABLE = torch.cuda.is_available()

if GPU_AVAILABLE:
    import cuml.accel
    cuml.accel.install()
```

## Best Practices

1. **Always initialize session state** at the beginning of your Streamlit app:
   ```python
   SessionStateManager.initialize()
   ```

2. **Use cached functions** for expensive operations:
   ```python
   @st.cache_data
   def expensive_computation(data):
       return DataProcessor.parse_responses_parallel(data)
   ```

3. **Handle large files** with memory manager:
   ```python
   # Check file size first
   if file_size_gb > 1:
       data = MemoryManager.sample_large_dataset(file_path)
   else:
       data = DataProcessor.load_data(file_path)
   ```

4. **Validate API keys** before use:
   ```python
   if not SessionStateManager.get('api_key_validated'):
       if APIKeyValidator.validate_openai_key(key):
           SessionStateManager.set('api_key_validated', True)
   ```

5. **Use pipelines** for complex data processing:
   ```python
   pipeline = create_uap_analysis_pipeline()
   processed_data = pipeline.run(raw_data_path)
   ```

## Performance Tips

- Enable GPU acceleration when available for faster processing
- Use chunked processing for files larger than 1GB
- Cache visualization results for repeated plots
- Validate API keys once and cache results
- Use parallel processing for JSON parsing
- Optimize DataFrame memory for large datasets

## Debugging

View session state summary:
```python
with st.expander("Debug Info"):
    st.json(SessionStateManager.get_state_summary())
```

Check memory usage:
```python
stats = MemoryManager.get_memory_usage()
st.metric("Memory", f"{stats['rss_mb']:.1f} MB")
``` 