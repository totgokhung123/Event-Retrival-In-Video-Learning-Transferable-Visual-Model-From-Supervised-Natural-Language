"""
Services package for video search backend
"""

import os
import clip
import torch
import numpy as np
from pathlib import Path

def initialize_services(base_dir, metadata_dir, embedding_dir, finetuned_model_path=None):
    """
    Initialize all services needed by the application
    
    Args:
        base_dir: Base directory for frames
        metadata_dir: Directory for metadata
        embedding_dir: Directory for embeddings
        finetuned_model_path: Path to the finetuned CLIP model (optional)
        
    Returns:
        Dictionary containing all service instances
    """
    # Import services from the current package
    from .path_service import PathService
    from .cache_service import CacheService
    from .data_service import DataService
    
    # Initialize basic services
    path_service = PathService(base_dir, metadata_dir, embedding_dir)
    cache_service = CacheService()
    data_service = DataService(path_service, cache_service)
    
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Try to initialize advanced services
    embedding_service = None
    search_service = None
    
    try:
        from .embedding_service import EmbeddingService
        from .search_service import SearchService
        
        # Initialize embedding service with finetuned model path
        embedding_service = EmbeddingService(cache_service, path_service, data_service, device)
        
        # Initialize search service
        search_service = SearchService(embedding_service, data_service, path_service, cache_service)
        print("Advanced services loaded successfully")
        services_loaded = True
    except ImportError as e:
        print(f"Could not import advanced services: {e}")
        services_loaded = False
    
    # Return all services in a dictionary
    return {
        'path_service': path_service,
        'cache_service': cache_service,
        'data_service': data_service,
        'embedding_service': embedding_service,
        'search_service': search_service,
        'device': device,
        'services_loaded': services_loaded
    } 