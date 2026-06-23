"""
Memory management utilities for UAP Data Analysis Tool
Handles large file processing and memory optimization
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Iterator, Callable, Any, Optional, List
import logging
import gc
import psutil
import os

logger = logging.getLogger(__name__)

class MemoryManager:
    """Memory management for processing large UAP datasets"""
    
    @staticmethod
    def get_memory_usage() -> dict:
        """Get current memory usage statistics"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    @staticmethod
    @st.cache_resource
    def get_data_iterator(file_path: str, chunksize: int = 10000) -> Iterator[pd.DataFrame]:
        """
        Create an iterator for reading large HDF5 files in chunks
        
        Args:
            file_path: Path to HDF5 file
            chunksize: Number of rows per chunk
            
        Returns:
            Iterator yielding DataFrame chunks
        """
        try:
            # For HDF5 files
            if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
                return pd.read_hdf(file_path, iterator=True, chunksize=chunksize)
            # For CSV files
            elif file_path.endswith('.csv'):
                return pd.read_csv(file_path, chunksize=chunksize)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
        except Exception as e:
            logger.error(f"Error creating data iterator: {e}")
            raise
    
    @staticmethod
    def process_data_in_chunks(
        iterator: Iterator[pd.DataFrame], 
        process_func: Callable[[pd.DataFrame], Any],
        combine_func: Optional[Callable[[List[Any]], Any]] = None,
        progress_bar: bool = True
    ) -> Any:
        """
        Process data in chunks to manage memory usage
        
        Args:
            iterator: Data chunk iterator
            process_func: Function to apply to each chunk
            combine_func: Function to combine results (default: pd.concat)
            progress_bar: Show progress bar
            
        Returns:
            Combined results from all chunks
        """
        results = []
        chunk_count = 0
        
        if progress_bar:
            progress = st.progress(0)
            status_text = st.empty()
        
        try:
            for chunk in iterator:
                # Process chunk
                result = process_func(chunk)
                results.append(result)
                chunk_count += 1
                
                # Update progress
                if progress_bar:
                    memory_stats = MemoryManager.get_memory_usage()
                    status_text.text(
                        f"Processed {chunk_count} chunks | "
                        f"Memory: {memory_stats['rss_mb']:.1f}MB ({memory_stats['percent']:.1f}%)"
                    )
                
                # Garbage collection every 10 chunks
                if chunk_count % 10 == 0:
                    gc.collect()
                    
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_count}: {e}")
            raise
        finally:
            if progress_bar:
                progress.empty()
                status_text.empty()
        
        # Combine results
        if combine_func:
            return combine_func(results)
        elif results and isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        else:
            return results
    
    @staticmethod
    @st.cache_data(persist="disk", max_entries=10)
    def load_data_subset(
        file_path: str, 
        start_row: int = 0, 
        num_rows: int = 10000,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load a subset of data from file with disk caching
        
        Args:
            file_path: Path to data file
            start_row: Starting row index
            num_rows: Number of rows to load
            columns: Specific columns to load
            
        Returns:
            DataFrame subset
        """
        try:
            if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
                # For HDF5, we need to load and then slice
                with pd.HDFStore(file_path, mode='r') as store:
                    # Get the first key (assumes single dataset)
                    key = list(store.keys())[0]
                    
                    # Use where parameter for efficient loading
                    if columns:
                        df = store.select(
                            key, 
                            start=start_row, 
                            stop=start_row + num_rows,
                            columns=columns
                        )
                    else:
                        df = store.select(
                            key, 
                            start=start_row, 
                            stop=start_row + num_rows
                        )
                return df
                
            elif file_path.endswith('.csv'):
                # For CSV, use skiprows and nrows
                if columns:
                    return pd.read_csv(
                        file_path, 
                        skiprows=range(1, start_row + 1),  # Skip header + rows
                        nrows=num_rows,
                        usecols=columns
                    )
                else:
                    return pd.read_csv(
                        file_path, 
                        skiprows=range(1, start_row + 1),
                        nrows=num_rows
                    )
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
                
        except Exception as e:
            logger.error(f"Error loading data subset: {e}")
            raise
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame, deep: bool = True) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types
        
        Args:
            df: DataFrame to optimize
            deep: Whether to return a deep copy
            
        Returns:
            Memory-optimized DataFrame
        """
        if deep:
            df = df.copy()
            
        # Optimize numeric columns
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                # Integer optimization
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                        
                # Float optimization
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                        
        # Convert string columns with low cardinality to category
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
                
        return df
    
    @staticmethod
    def estimate_dataframe_memory(df: pd.DataFrame) -> dict:
        """Estimate memory usage of a DataFrame"""
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        return {
            'total_mb': total_memory / 1024 / 1024,
            'columns': {
                col: mem / 1024 / 1024 
                for col, mem in memory_usage.items() 
                if col != 'Index'
            }
        }
    
    @staticmethod
    def clear_memory_cache() -> None:
        """Clear memory and caches"""
        # Clear Streamlit cache
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Memory cache cleared")
        
    @staticmethod
    @st.cache_data
    def sample_large_dataset(
        file_path: str, 
        sample_size: int = 10000,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Get a random sample from a large dataset
        
        Args:
            file_path: Path to data file
            sample_size: Number of rows to sample
            random_state: Random seed for reproducibility
            
        Returns:
            Sampled DataFrame
        """
        try:
            # First, get total number of rows
            if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
                with pd.HDFStore(file_path, mode='r') as store:
                    key = list(store.keys())[0]
                    total_rows = store.get_storer(key).nrows
            elif file_path.endswith('.csv'):
                # Count rows without loading entire file
                total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            # Generate random indices
            if sample_size >= total_rows:
                return MemoryManager.load_data_subset(file_path, 0, total_rows)
            
            np.random.seed(random_state)
            sample_indices = np.sort(np.random.choice(total_rows, sample_size, replace=False))
            
            # Load sampled data
            sampled_dfs = []
            for idx in sample_indices:
                df_row = MemoryManager.load_data_subset(file_path, idx, 1)
                sampled_dfs.append(df_row)
                
            return pd.concat(sampled_dfs, ignore_index=True)
            
        except Exception as e:
            logger.error(f"Error sampling dataset: {e}")
            raise 