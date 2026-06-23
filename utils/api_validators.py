"""
API key validation utilities for UAP Data Analysis Tool
Implements cached validation to avoid repeated API calls
"""

import streamlit as st
from typing import Dict, Optional, Callable
import logging
import time
from functools import lru_cache
import os

logger = logging.getLogger(__name__)

class APIKeyValidator:
    """Centralized API key validation with caching"""
    
    # Cache validation results for 1 hour (3600 seconds)
    CACHE_TTL = 3600
    _validation_cache: Dict[str, Dict[str, any]] = {}
    
    @classmethod
    @st.cache_data(ttl=3600)
    def validate_openai_key(cls, api_key: str, model: str = 'gpt-4o-mini') -> bool:
        """Validate OpenAI API key with caching"""
        if not api_key:
            return False
            
        cache_key = f"openai_{api_key[:8]}..."  # Use first 8 chars for cache key
        
        # Check cache first
        if cache_key in cls._validation_cache:
            cached = cls._validation_cache[cache_key]
            if time.time() - cached['timestamp'] < cls.CACHE_TTL:
                logger.info(f"Using cached validation result for OpenAI key")
                return cached['valid']
        
        try:
            from openai import OpenAI
            
            os.environ['OPENAI_API_KEY'] = api_key
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": 'Say "test"'}],
                max_tokens=5
            )
            
            valid = len(response.choices[0].message.content) > 0
            
            # Cache the result
            cls._validation_cache[cache_key] = {
                'valid': valid,
                'timestamp': time.time()
            }
            
            logger.info(f"OpenAI API key validation: {'success' if valid else 'failed'}")
            return valid
            
        except Exception as e:
            logger.error(f'Error validating OpenAI API key: {e}')
            # Cache the negative result
            cls._validation_cache[cache_key] = {
                'valid': False,
                'timestamp': time.time()
            }
            return False
    
    @classmethod
    @st.cache_data(ttl=3600)
    def validate_cohere_key(cls, api_key: str) -> bool:
        """Validate Cohere API key with caching"""
        if not api_key:
            return False
            
        cache_key = f"cohere_{api_key[:8]}..."
        
        # Check cache first
        if cache_key in cls._validation_cache:
            cached = cls._validation_cache[cache_key]
            if time.time() - cached['timestamp'] < cls.CACHE_TTL:
                logger.info(f"Using cached validation result for Cohere key")
                return cached['valid']
        
        try:
            import cohere
            
            co = cohere.Client(api_key=api_key)
            # Test with a simple rerank call
            response = co.rerank(
                model="rerank-english-v3.0",
                query="test",
                documents=["test document"],
                top_n=1
            )
            
            valid = response is not None
            
            # Cache the result
            cls._validation_cache[cache_key] = {
                'valid': valid,
                'timestamp': time.time()
            }
            
            logger.info(f"Cohere API key validation: {'success' if valid else 'failed'}")
            return valid
            
        except Exception as e:
            logger.error(f'Error validating Cohere API key: {e}')
            # Cache the negative result
            cls._validation_cache[cache_key] = {
                'valid': False,
                'timestamp': time.time()
            }
            return False
    
    @classmethod
    @st.cache_data(ttl=3600)
    def validate_gemini_key(cls, api_key: str) -> bool:
        """Validate Google Gemini API key with caching"""
        if not api_key:
            return False
            
        cache_key = f"gemini_{api_key[:8]}..."
        
        # Check cache first
        if cache_key in cls._validation_cache:
            cached = cls._validation_cache[cache_key]
            if time.time() - cached['timestamp'] < cls.CACHE_TTL:
                logger.info(f"Using cached validation result for Gemini key")
                return cached['valid']
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('models/gemini-3.1-pro-preview')
            response = model.generate_content("Say test")
            
            valid = response.text is not None and len(response.text) > 0
            
            # Cache the result
            cls._validation_cache[cache_key] = {
                'valid': valid,
                'timestamp': time.time()
            }
            
            logger.info(f"Gemini API key validation: {'success' if valid else 'failed'}")
            return valid
            
        except Exception as e:
            logger.error(f'Error validating Gemini API key: {e}')
            # Cache the negative result
            cls._validation_cache[cache_key] = {
                'valid': False,
                'timestamp': time.time()
            }
            return False
    
    @classmethod
    def validate_all_keys(cls, api_keys: Dict[str, str]) -> Dict[str, bool]:
        """Validate multiple API keys and return results"""
        validators = {
            'openai': cls.validate_openai_key,
            'cohere': cls.validate_cohere_key,
            'gemini': cls.validate_gemini_key
        }
        
        results = {}
        for provider, key in api_keys.items():
            if provider in validators and key:
                results[provider] = validators[provider](key)
            else:
                results[provider] = False
                
        return results
    
    @classmethod
    def clear_cache(cls, provider: Optional[str] = None) -> None:
        """Clear validation cache for a specific provider or all"""
        if provider:
            # Clear specific provider's cache
            keys_to_remove = [k for k in cls._validation_cache.keys() 
                            if k.startswith(f"{provider}_")]
            for key in keys_to_remove:
                del cls._validation_cache[key]
            logger.info(f"Cleared validation cache for {provider}")
        else:
            # Clear all cache
            cls._validation_cache.clear()
            logger.info("Cleared all validation cache")
            
    @classmethod
    def get_validation_status(cls) -> Dict[str, int]:
        """Get current validation cache status"""
        status = {}
        current_time = time.time()
        
        for provider in ['openai', 'cohere', 'gemini']:
            provider_keys = [k for k in cls._validation_cache.keys() 
                           if k.startswith(f"{provider}_")]
            valid_count = sum(1 for k in provider_keys 
                            if cls._validation_cache[k]['valid'] and 
                            current_time - cls._validation_cache[k]['timestamp'] < cls.CACHE_TTL)
            status[provider] = valid_count
            
        return status 