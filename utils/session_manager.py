"""
Session state management for UAP Data Analysis Tool
Centralizes session state initialization and management
"""

import streamlit as st
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SessionStateManager:
    """Centralized session state management"""
    
    # Default values for all session state variables
    DEFAULTS = {
        # Data storage
        'analyzers': [],
        'col_names': [],
        'clusters': {},
        'new_data': None,
        'dataset': None,
        'parsed_responses': None,
        'parsed_responses_df': None,
        'filtered_data': None,
        
        # Processing flags
        'data_processed': False,
        'data_loaded': False,
        'map_generated': False,
        
        # Authentication and subscription
        'user_subscribed': False,
        'email': '',
        
        # API keys validation
        'api_key_valid': False,
        'previous_api_key': None,
        'api_keys_validated': False,
        
        # Analysis results
        'gemini_answer': None,
        'json_format': None,
        
        # UI state
        'stage': 0,
        'buttons': {},
        
        # Cache for expensive operations
        'cached_embeddings': {},
        'cached_clusters': {},
        'cached_visualizations': {}
    }
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize all session state variables with defaults"""
        for key, default_value in cls.DEFAULTS.items():
            if key not in st.session_state:
                # Handle special cases where default needs to be a new instance
                if key in ['new_data', 'dataset']:
                    import pandas as pd
                    st.session_state[key] = pd.DataFrame()
                elif isinstance(default_value, (dict, list)):
                    # Create new instance to avoid shared references
                    st.session_state[key] = type(default_value)(default_value)
                else:
                    st.session_state[key] = default_value
                    
        logger.info("Session state initialized")
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Safely get a session state value"""
        return st.session_state.get(key, default if default is not None else cls.DEFAULTS.get(key))
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Safely set a session state value"""
        st.session_state[key] = value
        logger.debug(f"Session state '{key}' set to {type(value).__name__}")
    
    @classmethod
    def update(cls, updates: Dict[str, Any]) -> None:
        """Update multiple session state values at once"""
        for key, value in updates.items():
            cls.set(key, value)
    
    @classmethod
    def clear(cls, keys: Optional[list] = None) -> None:
        """Clear specific session state keys or all if none specified"""
        if keys is None:
            # Clear all except authentication-related keys
            preserve_keys = {'email', 'user_subscribed'}
            keys_to_clear = [k for k in st.session_state.keys() if k not in preserve_keys]
        else:
            keys_to_clear = keys
            
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
                
        # Reinitialize cleared keys with defaults
        for key in keys_to_clear:
            if key in cls.DEFAULTS:
                if key in ['new_data', 'dataset']:
                    import pandas as pd
                    st.session_state[key] = pd.DataFrame()
                elif isinstance(cls.DEFAULTS[key], (dict, list)):
                    st.session_state[key] = type(cls.DEFAULTS[key])(cls.DEFAULTS[key])
                else:
                    st.session_state[key] = cls.DEFAULTS[key]
                    
        logger.info(f"Cleared session state keys: {keys_to_clear}")
    
    @classmethod
    def exists(cls, key: str) -> bool:
        """Check if a session state key exists"""
        return key in st.session_state
    
    @classmethod
    def increment(cls, key: str, amount: int = 1) -> None:
        """Increment a numeric session state value"""
        current = cls.get(key, 0)
        if isinstance(current, (int, float)):
            cls.set(key, current + amount)
        else:
            logger.warning(f"Cannot increment non-numeric session state key: {key}")
    
    @classmethod
    def append(cls, key: str, value: Any) -> None:
        """Append to a list session state value"""
        current = cls.get(key, [])
        if isinstance(current, list):
            current.append(value)
            cls.set(key, current)
        else:
            logger.warning(f"Cannot append to non-list session state key: {key}")
    
    @classmethod
    def update_dict(cls, key: str, updates: Dict[str, Any]) -> None:
        """Update a dictionary session state value"""
        current = cls.get(key, {})
        if isinstance(current, dict):
            current.update(updates)
            cls.set(key, current)
        else:
            logger.warning(f"Cannot update non-dict session state key: {key}")
    
    @classmethod
    def get_state_summary(cls) -> Dict[str, Any]:
        """Get a summary of current session state for debugging"""
        summary = {}
        for key in st.session_state:
            value = st.session_state[key]
            if hasattr(value, '__len__') and not isinstance(value, str):
                summary[key] = f"{type(value).__name__} (length: {len(value)})"
            else:
                summary[key] = f"{type(value).__name__}"
        return summary
    
    @classmethod
    def cache_visualization(cls, viz_type: str, data_hash: str, figure: Any) -> None:
        """Cache a visualization for reuse"""
        cache = cls.get('cached_visualizations', {})
        cache_key = f"{viz_type}_{data_hash}"
        cache[cache_key] = figure
        cls.set('cached_visualizations', cache)
        
    @classmethod
    def get_cached_visualization(cls, viz_type: str, data_hash: str) -> Optional[Any]:
        """Retrieve a cached visualization"""
        cache = cls.get('cached_visualizations', {})
        cache_key = f"{viz_type}_{data_hash}"
        return cache.get(cache_key) 