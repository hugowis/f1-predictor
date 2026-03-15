"""
Models package for F1 lap time prediction.
"""

from .base import BaseModel
from .seq2seq import Seq2Seq
from .trainer import Trainer, create_scheduler
from .evaluator import Evaluator, compute_regression_metrics, report_evaluation
from .rollout_evaluator import (
    evaluate_autoregressive_rollout,
    report_rollout_evaluation,
    denormalize_rollout_metrics,
)

__all__ = [
    'BaseModel',
    'Seq2Seq',
    'Trainer',
    'Evaluator',
    'create_scheduler',
    'compute_regression_metrics',
    'report_evaluation',
    'evaluate_autoregressive_rollout',
    'report_rollout_evaluation',
    'denormalize_rollout_metrics',
]

__version__ = '1.0.0'
