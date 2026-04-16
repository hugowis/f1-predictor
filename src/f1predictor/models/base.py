"""
Base model class for F1 lap time prediction.

Provides common interface for all models (GRU, LSTM, Transformer, etc).
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for F1 lap time prediction models.
    
    All models should inherit from this and implement:
    - forward()
    - encode()
    - decode() (if applicable)
    
    This class provides:
    - Standard initialization
    - Device management
    - Configuration storage
    - Model info methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base model.
        
        Parameters
        ----------
        config : dict
            Model configuration containing:
            - input_size: number of input features
            - hidden_size: GRU/LSTM hidden dimension
            - num_layers: number of stacked layers
            - output_size: number of output features (usually 1 for lap time)
            - dropout: dropout rate
            - device: torch device
        """
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cpu')
        
    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        
        Returns
        -------
        torch.Tensor
            Model output
        """
        pass
    
    def to_device(self, *tensors):
        """Move tensors to model's device."""
        return [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in tensors]
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return self.config
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def model_info(self) -> str:
        """Return human-readable model information."""
        num_params = self.count_parameters()
        return f"{self.__class__.__name__}(params={num_params:,})"
