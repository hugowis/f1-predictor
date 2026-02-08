"""
Configuration module for F1 lap time prediction.
"""

from .base import Config, ModelConfig, TrainingConfig, DataConfig
from .base import get_phase1_config, get_phase2_config

__all__ = [
    'Config',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'get_phase1_config',
    'get_phase2_config',
]

__version__ = '1.0.0'
