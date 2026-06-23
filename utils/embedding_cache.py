"""
Embedding cache management for UAP Data Analysis Tool
Caches embeddings per column to avoid recomputation
"""

import pandas as pd
import numpy as np
import streamlit as st
import hashlib
import pickle
import os
import json
from typing import Optional, Dict, Any, Tuple
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class EmbeddingCacheManager:
    """Manages caching of embeddings for text columns"""
    
    # Cache directory
    CACHE_DIR = Path(".embedding_cache")
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the embedding cache manager"""
        self.cache_dir = cache_dir or self.CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache for current session
        self._memory_cache: Dict[str, np.ndarray] = {}
        
        # Metadata file to track cached embeddings
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    @staticmethod
    def generate_cache_key(data: pd.Series, column_name: str, model_name: str = "default") -> str:
        """
        Generate a unique cache key for column data
        
        Args:
            data: The column data
            column_name: Name of the column
            model_name: Embedding model name
            
        Returns:
            Unique cache key
        """
        # Create a hash of the data content
        data_str = ''.join(str(x) for x in data.dropna().unique())
        data_hash = hashlib.md5(data_str.encode()).hexdigest()[:16]
        
        # Include column name and model in the key
        cache_key = f"{column_name}_{model_name}_{data_hash}"
        
        return cache_key
    
    def get_embeddings(self, 
                      data: pd.Series, 
                      column_name: str,
                      model_name: str = "microsoft/harrier-oss-v1-0.6b",
                      compute_func: Optional[callable] = None) -> Optional[np.ndarray]:
        """
        Get embeddings from cache or compute if not exists
        
        Args:
            data: The column data
            column_name: Name of the column
            model_name: Embedding model name
            compute_func: Function to compute embeddings if not cached
            
        Returns:
            Embeddings array or None
        """
        cache_key = self.generate_cache_key(data, column_name, model_name)
        
        # Check in-memory cache first
        if cache_key in self._memory_cache:
            logger.info(f"Using in-memory cached embeddings for {column_name}")
            return self._memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                logger.info(f"Loading cached embeddings from disk for {column_name}")
                with open(cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                    
                # Store in memory cache
                self._memory_cache[cache_key] = embeddings
                
                return embeddings
            except Exception as e:
                logger.error(f"Error loading cached embeddings: {e}")
                # Continue to recompute if loading fails
        
        # Compute embeddings if not cached and compute function provided
        if compute_func is not None:
            logger.info(f"Computing new embeddings for {column_name}")
            start_time = time.time()
            
            embeddings = compute_func(data)
            
            compute_time = time.time() - start_time
            logger.info(f"Computed embeddings in {compute_time:.2f}s")
            
            # Cache the embeddings
            self.cache_embeddings(embeddings, data, column_name, model_name)
            
            return embeddings
        
        return None
    
    def cache_embeddings(self,
                        embeddings: np.ndarray,
                        data: pd.Series,
                        column_name: str,
                        model_name: str = "microsoft/harrier-oss-v1-0.6b") -> None:
        """
        Cache embeddings to disk and memory
        
        Args:
            embeddings: The embeddings array
            data: The original column data
            column_name: Name of the column
            model_name: Embedding model name
        """
        cache_key = self.generate_cache_key(data, column_name, model_name)
        
        # Store in memory cache
        self._memory_cache[cache_key] = embeddings
        
        # Store on disk
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            
            # Update metadata
            self.metadata[cache_key] = {
                'column_name': column_name,
                'model_name': model_name,
                'shape': embeddings.shape,
                'cached_at': time.time(),
                'data_size': len(data),
                'unique_values': int(data.nunique())
            }
            self._save_metadata()
            
            logger.info(f"Cached embeddings for {column_name} to disk")
            
        except Exception as e:
            logger.error(f"Error caching embeddings: {e}")
    
    def clear_cache(self, column_name: Optional[str] = None) -> None:
        """
        Clear embedding cache
        
        Args:
            column_name: If provided, only clear cache for this column
        """
        if column_name:
            # Clear specific column
            keys_to_remove = [k for k in self.metadata.keys() 
                            if k.startswith(f"{column_name}_")]
            
            for key in keys_to_remove:
                # Remove from memory
                self._memory_cache.pop(key, None)
                
                # Remove from disk
                cache_file = self.cache_dir / f"{key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                
                # Remove from metadata
                self.metadata.pop(key, None)
            
            self._save_metadata()
            logger.info(f"Cleared cache for column: {column_name}")
            
        else:
            # Clear all cache
            self._memory_cache.clear()
            
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            
            # Clear metadata
            self.metadata.clear()
            self._save_metadata()
            
            logger.info("Cleared all embedding cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached embeddings"""
        info = {
            'total_cached': len(self.metadata),
            'memory_cached': len(self._memory_cache),
            'disk_size_mb': sum(
                (self.cache_dir / f"{k}.pkl").stat().st_size / 1024 / 1024
                for k in self.metadata.keys()
                if (self.cache_dir / f"{k}.pkl").exists()
            ),
            'columns': {}
        }
        
        # Group by column name
        for key, meta in self.metadata.items():
            col_name = meta['column_name']
            if col_name not in info['columns']:
                info['columns'][col_name] = []
            
            info['columns'][col_name].append({
                'model': meta['model_name'],
                'shape': meta['shape'],
                'cached_at': time.strftime('%Y-%m-%d %H:%M:%S', 
                                         time.localtime(meta['cached_at']))
            })
        
        return info

# Global instance for easy access
_embedding_cache = None

def get_embedding_cache() -> EmbeddingCacheManager:
    """Get or create global embedding cache instance"""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCacheManager()
    return _embedding_cache

# Streamlit-specific caching decorator
@st.cache_data(persist="disk", show_spinner=False)
def get_cached_embeddings(data_hash: str, 
                         column_name: str,
                         model_name: str,
                         compute_func: callable,
                         data_values: list) -> np.ndarray:
    """
    Streamlit-cached wrapper for embedding computation
    
    Args:
        data_hash: Hash of the data
        column_name: Column name
        model_name: Model name
        compute_func: Function to compute embeddings
        data_values: The actual data values
        
    Returns:
        Embeddings array
    """
    # Convert back to Series
    data = pd.Series(data_values)
    
    # Use the cache manager
    cache_manager = get_embedding_cache()
    embeddings = cache_manager.get_embeddings(
        data, 
        column_name, 
        model_name,
        lambda x: compute_func(x.tolist())
    )
    
    return embeddings

# Helper function for the UAPAnalyzer integration
def compute_embeddings_with_cache(data: pd.Series,
                                column_name: str,
                                model_name: str = "microsoft/harrier-oss-v1-0.6b",
                                encoder_func: callable = None) -> np.ndarray:
    """
    Compute embeddings with caching support
    
    Args:
        data: Column data
        column_name: Name of the column
        model_name: Embedding model name
        encoder_func: Function that takes a list of texts and returns embeddings
        
    Returns:
        Embeddings array
    """
    if encoder_func is None:
        from uap_analyzer import get_embed_model
        from sentence_transformers import SentenceTransformer
        if model_name == "microsoft/harrier-oss-v1-0.6b":
            encoder_func = get_embed_model().encode
        else:
            model = SentenceTransformer(model_name, model_kwargs={"dtype": "auto"})
            encoder_func = model.encode
    
    # Generate hash for Streamlit caching
    data_str = ''.join(str(x) for x in data.dropna().unique())
    data_hash = hashlib.md5(data_str.encode()).hexdigest()
    
    # Use Streamlit caching
    return get_cached_embeddings(
        data_hash,
        column_name,
        model_name,
        encoder_func,
        data.tolist()
    ) 