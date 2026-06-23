"""
Data pipeline utilities for UAP Data Analysis Tool
Implements ETL pipeline pattern for data processing
"""

import pandas as pd
import numpy as np
from typing import List, Callable, Any, Dict, Optional, Union
import logging
from functools import wraps
import time
import streamlit as st

logger = logging.getLogger(__name__)

class PipelineStep:
    """Base class for pipeline steps"""
    
    def __init__(self, name: str, func: Callable, **kwargs):
        self.name = name
        self.func = func
        self.kwargs = kwargs
        self.execution_time = None
        self.error = None
        
    def __call__(self, data: Any) -> Any:
        """Execute the pipeline step"""
        start_time = time.time()
        try:
            result = self.func(data, **self.kwargs)
            self.execution_time = time.time() - start_time
            logger.info(f"Step '{self.name}' completed in {self.execution_time:.2f}s")
            return result
        except Exception as e:
            self.error = e
            logger.error(f"Error in step '{self.name}': {e}")
            raise
            
    def __repr__(self):
        return f"PipelineStep(name='{self.name}', func={self.func.__name__})"

class UAP_Pipeline:
    """ETL Pipeline for UAP data processing"""
    
    def __init__(self, name: str = "UAP Pipeline"):
        self.name = name
        self.extractors: List[PipelineStep] = []
        self.transformers: List[PipelineStep] = []
        self.loaders: List[PipelineStep] = []
        self.validators: List[PipelineStep] = []
        self.execution_history: List[Dict[str, Any]] = []
        
    def add_extractor(self, name: str, func: Callable, **kwargs) -> 'UAP_Pipeline':
        """Add an extraction step"""
        self.extractors.append(PipelineStep(name, func, **kwargs))
        return self
        
    def add_transformer(self, name: str, func: Callable, **kwargs) -> 'UAP_Pipeline':
        """Add a transformation step"""
        self.transformers.append(PipelineStep(name, func, **kwargs))
        return self
        
    def add_loader(self, name: str, func: Callable, **kwargs) -> 'UAP_Pipeline':
        """Add a loading step"""
        self.loaders.append(PipelineStep(name, func, **kwargs))
        return self
        
    def add_validator(self, name: str, func: Callable, **kwargs) -> 'UAP_Pipeline':
        """Add a validation step"""
        self.validators.append(PipelineStep(name, func, **kwargs))
        return self
        
    def run(self, initial_data: Optional[Any] = None, show_progress: bool = True) -> Any:
        """Execute the complete pipeline"""
        start_time = time.time()
        data = initial_data
        
        all_steps = [
            ("Extractors", self.extractors),
            ("Transformers", self.transformers),
            ("Validators", self.validators),
            ("Loaders", self.loaders)
        ]
        
        total_steps = sum(len(steps) for _, steps in all_steps)
        current_step = 0
        
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        execution_record = {
            'pipeline': self.name,
            'start_time': start_time,
            'steps': []
        }
        
        try:
            for stage_name, steps in all_steps:
                for step in steps:
                    current_step += 1
                    
                    if show_progress:
                        progress = current_step / total_steps
                        progress_bar.progress(progress)
                        status_text.text(f"{stage_name}: {step.name}")
                    
                    # Execute step
                    step_start = time.time()
                    data = step(data)
                    step_time = time.time() - step_start
                    
                    # Record execution
                    execution_record['steps'].append({
                        'stage': stage_name,
                        'name': step.name,
                        'execution_time': step_time,
                        'success': True
                    })
                    
        except Exception as e:
            execution_record['error'] = str(e)
            execution_record['failed_step'] = step.name
            raise
        finally:
            if show_progress:
                progress_bar.empty()
                status_text.empty()
                
            execution_record['total_time'] = time.time() - start_time
            self.execution_history.append(execution_record)
            
        logger.info(f"Pipeline '{self.name}' completed in {execution_record['total_time']:.2f}s")
        return data
        
    def get_execution_summary(self) -> pd.DataFrame:
        """Get summary of pipeline executions"""
        if not self.execution_history:
            return pd.DataFrame()
            
        summaries = []
        for execution in self.execution_history:
            summary = {
                'pipeline': execution['pipeline'],
                'total_time': execution['total_time'],
                'num_steps': len(execution.get('steps', [])),
                'success': 'error' not in execution
            }
            summaries.append(summary)
            
        return pd.DataFrame(summaries)
        
    def visualize_pipeline(self) -> None:
        """Visualize the pipeline structure"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        stages = [
            ("Extract", self.extractors, '#FF6B6B'),
            ("Transform", self.transformers, '#4ECDC4'),
            ("Validate", self.validators, '#45B7D1'),
            ("Load", self.loaders, '#96CEB4')
        ]
        
        y_pos = 0.8
        x_pos = 0.1
        box_height = 0.15
        box_width = 0.15
        
        for stage_name, steps, color in stages:
            # Stage label
            ax.text(x_pos, y_pos + 0.1, stage_name, fontsize=14, fontweight='bold')
            
            # Draw steps
            for i, step in enumerate(steps):
                rect = Rectangle((x_pos, y_pos - i * 0.2), box_width, box_height, 
                               facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                ax.text(x_pos + box_width/2, y_pos - i * 0.2 + box_height/2, 
                       step.name, ha='center', va='center', fontsize=10)
                
            x_pos += 0.25
            
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f"Pipeline: {self.name}", fontsize=16, fontweight='bold')
        
        return fig

# Pre-built pipeline components
class PipelineComponents:
    """Common pipeline components for UAP data processing"""
    
    @staticmethod
    def extract_from_file(file_path: str, **kwargs) -> pd.DataFrame:
        """Extract data from file"""
        from utils.memory_manager import MemoryManager
        
        if kwargs.get('use_chunks', False):
            # Use chunked loading for large files
            iterator = MemoryManager.get_data_iterator(file_path, kwargs.get('chunksize', 10000))
            return MemoryManager.process_data_in_chunks(iterator, lambda x: x)
        else:
            # Regular loading
            from utils.data_processing import DataProcessor
            return DataProcessor.load_data(file_path)
    
    @staticmethod
    def parse_json_responses(data: pd.DataFrame, response_column: str = 'response') -> pd.DataFrame:
        """Parse JSON responses in parallel"""
        from utils.data_processing import DataProcessor
        
        if response_column in data.columns:
            responses_dict = data[response_column].to_dict()
            parsed = DataProcessor.parse_responses_parallel(responses_dict)
            
            # Convert back to DataFrame
            parsed_df = pd.DataFrame.from_dict(parsed, orient='index')
            return pd.concat([data, parsed_df], axis=1)
        else:
            logger.warning(f"Column '{response_column}' not found in data")
            return data
    
    @staticmethod
    def optimize_memory(data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        from utils.memory_manager import MemoryManager
        return MemoryManager.optimize_dataframe_memory(data)
    
    @staticmethod
    def validate_schema(data: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """Validate DataFrame has required columns"""
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        return data
    
    @staticmethod
    def filter_outliers(data: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        """Filter outliers from specified columns"""
        data = data.copy()
        
        for col in columns:
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                if method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    data = data[(data[col] >= lower) & (data[col] <= upper)]
                elif method == 'zscore':
                    from scipy import stats
                    z_scores = np.abs(stats.zscore(data[col].dropna()))
                    data = data[z_scores < 3]
                    
        return data
    
    @staticmethod
    def add_derived_features(data: pd.DataFrame) -> pd.DataFrame:
        """Add commonly used derived features for UAP analysis"""
        data = data.copy()
        
        # Add time-based features if date column exists
        date_columns = [col for col in data.columns if 'date' in col.lower()]
        for date_col in date_columns:
            try:
                data[date_col] = pd.to_datetime(data[date_col])
                data[f'{date_col}_year'] = data[date_col].dt.year
                data[f'{date_col}_month'] = data[date_col].dt.month
                data[f'{date_col}_dayofweek'] = data[date_col].dt.dayofweek
                data[f'{date_col}_hour'] = data[date_col].dt.hour
            except:
                pass
                
        # Add location-based features if lat/lon exist
        lat_cols = [col for col in data.columns if 'lat' in col.lower()]
        lon_cols = [col for col in data.columns if 'lon' in col.lower() or 'lng' in col.lower()]
        
        if lat_cols and lon_cols:
            lat_col = lat_cols[0]
            lon_col = lon_cols[0]
            
            # Add hemisphere indicators
            data['northern_hemisphere'] = (data[lat_col] > 0).astype(int)
            data['eastern_hemisphere'] = (data[lon_col] > 0).astype(int)
            
        return data
    
    @staticmethod
    def save_to_cache(data: pd.DataFrame, cache_key: str) -> pd.DataFrame:
        """Save data to session state cache"""
        from utils.session_manager import SessionStateManager
        SessionStateManager.set(f'pipeline_cache_{cache_key}', data)
        logger.info(f"Data saved to cache with key: {cache_key}")
        return data

# Example usage function
def create_uap_analysis_pipeline() -> UAP_Pipeline:
    """Create a standard UAP analysis pipeline"""
    pipeline = UAP_Pipeline("Standard UAP Analysis")
    
    # Add extraction steps
    pipeline.add_extractor(
        "Load Data", 
        PipelineComponents.extract_from_file,
        use_chunks=True
    )
    
    # Add transformation steps
    pipeline.add_transformer(
        "Parse JSON",
        PipelineComponents.parse_json_responses
    )
    
    pipeline.add_transformer(
        "Optimize Memory",
        PipelineComponents.optimize_memory
    )
    
    pipeline.add_transformer(
        "Add Features",
        PipelineComponents.add_derived_features
    )
    
    # Add validation steps
    pipeline.add_validator(
        "Validate Schema",
        PipelineComponents.validate_schema,
        required_columns=['date', 'location']
    )
    
    pipeline.add_transformer(
        "Filter Outliers",
        PipelineComponents.filter_outliers,
        columns=['altitude', 'speed'],
        method='iqr'
    )
    
    # Add loader steps
    pipeline.add_loader(
        "Cache Results",
        PipelineComponents.save_to_cache,
        cache_key='processed_uap_data'
    )
    
    return pipeline 