"""
Base classes for modular model architecture.

Defines abstract interfaces for embeddings, policy heads, and value heads
to enable easy addition of new model types.
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Optional
from torch.distributions import Distribution


class BaseEmbedding(ABC, nn.Module):
    """
    Abstract base class for all embedding models.
    
    Embeddings process raw observations and produce feature representations
    that are then used by policy and value heads.
    """
    
    def __init__(self):
        super().__init__()
        self._output_size = None
    
    @property
    def output_size(self) -> int:
        """Return the dimensionality of the output embedding."""
        if self._output_size is None:
            raise NotImplementedError("output_size must be set in __init__")
        return self._output_size
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Process observation and return embedding.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            
        Returns:
            embedding: Tensor of shape (batch_size, output_size)
        """
        pass
    
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass wrapper."""
        return super().__call__(obs)


class BasePolicyHead(ABC, nn.Module):
    """
    Abstract base class for policy heads.
    
    Policy heads take embeddings and produce action distributions.
    """
    
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
    
    @abstractmethod
    def forward(self, embedding: torch.Tensor) -> list[Optional[Distribution]]:
        """
        Generate action distributions.
        
        Args:
            embedding: Feature embedding of shape (batch_size, input_size)
            
        Returns:
            distributions: List of action distributions (one per action type)
        """
        pass


class BaseValueHead(ABC, nn.Module):
    """
    Abstract base class for value heads.
    
    Value heads take embeddings and predict state values.
    """
    
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
    
    @abstractmethod
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict state value.
        
        Args:
            embedding: Feature embedding of shape (batch_size, input_size)
            
        Returns:
            value: Tensor of shape (batch_size,) with value predictions
        """
        pass
    
    @abstractmethod
    def loss(self, new_values: torch.Tensor, values: torch.Tensor, 
             returns: torch.Tensor) -> torch.Tensor:
        """
        Calculate value loss.
        
        Args:
            new_values: New value predictions
            values: Old value predictions
            returns: Target returns
            
        Returns:
            loss: Scalar loss tensor
        """
        pass
