"""
Modular model architecture for MLIR-RL.

Provides base classes and factory functions for creating embeddings,
policy heads, and value heads.
"""
from .base import BaseEmbedding, BasePolicyHead, BaseValueHead
from .embeddings import LSTMEmbedding, DistilBertEmbedding
from utils.config import Config


# Registry of available embedding models
EMBEDDING_REGISTRY = {
    'lstm': LSTMEmbedding,
    'distilbert': DistilBertEmbedding,
}


def get_embedding_layer(model_type: str = None) -> BaseEmbedding:
    """
    Factory function to create embedding layers.
    
    Args:
        model_type: Type of embedding model. If None, reads from Config.
                   Options: 'lstm', 'distilbert'
    
    Returns:
        Embedding layer instance
        
    Raises:
        ValueError: If model_type is not supported
        
    Examples:
        >>> embedding = get_embedding_layer('lstm')
        >>> embedding = get_embedding_layer()  # Uses Config().model_type
    """
    if model_type is None:
        model_type = Config().model_type
    
    if model_type not in EMBEDDING_REGISTRY:
        available = ', '.join(EMBEDDING_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available models: {available}"
        )
    
    embedding_class = EMBEDDING_REGISTRY[model_type]
    return embedding_class()


def register_embedding(name: str, embedding_class: type):
    """
    Register a new embedding model type.
    
    This allows dynamically adding new embedding models without
    modifying this file.
    
    Args:
        name: Name to register the model under
        embedding_class: Class that inherits from BaseEmbedding
        
    Example:
        >>> class MyEmbedding(BaseEmbedding):
        ...     pass
        >>> register_embedding('my_model', MyEmbedding)
    """
    if not issubclass(embedding_class, BaseEmbedding):
        raise TypeError(f"{embedding_class} must inherit from BaseEmbedding")
    
    EMBEDDING_REGISTRY[name] = embedding_class


def list_available_models() -> list[str]:
    """
    Get list of available embedding model types.
    
    Returns:
        List of model type names
    """
    return list(EMBEDDING_REGISTRY.keys())


__all__ = [
    'BaseEmbedding',
    'BasePolicyHead', 
    'BaseValueHead',
    'LSTMEmbedding',
    'DistilBertEmbedding',
    'get_embedding_layer',
    'register_embedding',
    'list_available_models',
    'EMBEDDING_REGISTRY',
]
