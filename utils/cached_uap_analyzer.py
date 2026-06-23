"""
Cached UAP Analyzer wrapper that uses embedding caching
Extends UAPAnalyzer with cached embedding support
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Any
import logging
from sentence_transformers import SentenceTransformer
import torch
from uap_analyzer import get_embed_model

from uap_analyzer import UAPAnalyzer
from .embedding_cache import compute_embeddings_with_cache, get_embedding_cache

logger = logging.getLogger(__name__)

class CachedUAPAnalyzer(UAPAnalyzer):
    """
    Extended UAPAnalyzer that uses cached embeddings
    """
    
    def __init__(self, data: pd.DataFrame, column: str, model_name: str = "microsoft/harrier-oss-v1-0.6b"):
        """
        Initialize the cached analyzer
        
        Args:
            data: DataFrame containing the data
            column: Column name to analyze
            model_name: Sentence transformer model name
        """
        super().__init__(data, column)
        self.model_name = model_name
        self._embedding_model = None
        self._embeddings_cached = False
        
    @property
    def embedding_model(self):
        """Lazy load the embedding model"""
        if self._embedding_model is None:
            if self.model_name == "microsoft/harrier-oss-v1-0.6b":
                self._embedding_model = get_embed_model()
            else:
                self._embedding_model = SentenceTransformer(self.model_name, model_kwargs={"dtype": "auto"})
        return self._embedding_model
    
    def preprocess_data(self, top_n: int = 32) -> None:
        """
        Preprocess data with cached embeddings
        
        Args:
            top_n: Number of top items to keep
        """
        # Call parent preprocessing (if it does other things)
        super().preprocess_data(top_n)
        
        # Now compute embeddings with caching
        column_data = self.data[self.column]
        
        # Log cache status
        cache_info = get_embedding_cache().get_cache_info()
        if self.column in cache_info['columns']:
            logger.info(f"Found cached embeddings for column '{self.column}'")
            st.info(f"✓ Using cached embeddings for '{self.column}'")
        else:
            st.info(f"⏳ Computing new embeddings for '{self.column}'...")
        
        # Compute embeddings with cache
        self.embeddings = compute_embeddings_with_cache(
            data=column_data,
            column_name=self.column,
            model_name=self.model_name,
            encoder_func=self.embedding_model.encode
        )
        
        self._embeddings_cached = True
        
        # Store embeddings in the expected format for UAPAnalyzer
        if hasattr(self, '__dict__'):
            self.__dict__['embeddings'] = self.embeddings
    
    def compute_embeddings(self, texts: list) -> np.ndarray:
        """
        Override the compute_embeddings method if it exists
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Embeddings array
        """
        # Convert to pandas Series for caching
        data_series = pd.Series(texts)
        
        return compute_embeddings_with_cache(
            data=data_series,
            column_name=self.column,
            model_name=self.model_name,
            encoder_func=self.embedding_model.encode
        )
    
    def get_cache_status(self) -> dict:
        """Get cache status for this analyzer"""
        cache_info = get_embedding_cache().get_cache_info()
        
        column_cache = cache_info['columns'].get(self.column, [])
        
        return {
            'column': self.column,
            'cached': len(column_cache) > 0,
            'cache_entries': column_cache,
            'embeddings_loaded': self._embeddings_cached
        }

# Convenience function to create cached analyzer with progress tracking
@st.cache_resource
def create_cached_analyzer(_data: pd.DataFrame, column: str, model_name: str = "microsoft/harrier-oss-v1-0.6b") -> CachedUAPAnalyzer:
    """
    Create a cached UAP analyzer instance
    
    Args:
        _data: DataFrame (underscore prefix for Streamlit caching)
        column: Column to analyze
        model_name: Model name for embeddings
        
    Returns:
        CachedUAPAnalyzer instance
    """
    return CachedUAPAnalyzer(_data, column, model_name)

# Function to clear embedding cache with UI feedback
def clear_embedding_cache_ui(column: Optional[str] = None) -> None:
    """
    Clear embedding cache with UI feedback
    
    Args:
        column: Specific column to clear, or None for all
    """
    cache_manager = get_embedding_cache()
    
    if column:
        cache_manager.clear_cache(column)
        st.success(f"✓ Cleared embedding cache for column '{column}'")
    else:
        cache_manager.clear_cache()
        st.success("✓ Cleared all embedding caches")
        
# Function to display cache info in UI
def display_cache_info() -> None:
    """Display embedding cache information in Streamlit UI"""
    cache_info = get_embedding_cache().get_cache_info()
    
    with st.expander("📊 Embedding Cache Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Cached", cache_info['total_cached'])
        
        with col2:
            st.metric("Memory Cached", cache_info['memory_cached'])
            
        with col3:
            st.metric("Disk Size", f"{cache_info['disk_size_mb']:.1f} MB")
        
        if cache_info['columns']:
            st.subheader("Cached Columns:")
            for col_name, entries in cache_info['columns'].items():
                st.write(f"**{col_name}**")
                for entry in entries:
                    st.write(f"  - Model: {entry['model']}, Shape: {entry['shape']}, Cached: {entry['cached_at']}")
        else:
            st.info("No embeddings cached yet")
            
        # Clear cache buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear All Cache", key="clear_all_cache"):
                clear_embedding_cache_ui()
                st.experimental_rerun()
                
        with col2:
            selected_col = st.selectbox(
                "Clear specific column",
                options=list(cache_info['columns'].keys()) if cache_info['columns'] else [],
                key="clear_specific_cache"
            )
            if selected_col and st.button(f"Clear {selected_col}", key=f"clear_{selected_col}"):
                clear_embedding_cache_ui(selected_col)
                st.experimental_rerun() 