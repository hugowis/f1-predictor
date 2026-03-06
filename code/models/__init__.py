"""
Models package for F1 lap time prediction.
"""

from .base import BaseModel
from .seq2seq import Seq2Seq
from .trainer import Trainer, create_scheduler
from .evaluator import Evaluator, compute_regression_metrics, report_evaluation

__all__ = [
    'BaseModel',
    'Seq2Seq',
    'Trainer',
    'Evaluator',
    'create_scheduler',
    'compute_regression_metrics',
    'report_evaluation',
]

__version__ = '1.0.0'
