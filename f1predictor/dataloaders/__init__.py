"""
F1 Lap Time Prediction - Data Loaders

This package provides PyTorch Dataset classes for loading F1 race data
with support for multi-year datasets, data augmentation, and flexible
normalization strategies.

Two main dataloaders are provided:

1. **StintDataloader**: Stint-based sequences for teacher-forcing training
   - Groups laps into stints
   - Automatically splits on safety cars, red flags, yellow flags, VSC
   - Provides (input_sequence, target_laptime) pairs
   - Supports augmentation: tyre degradation, fuel variation, weather jitter

2. **AutoregressiveLapDataloader**: Lap-by-lap sequences for autoregressive training
   - Returns (context_laps, next_laptime) pairs
   - Flexible context window size
   - Supports both teacher-forcing and autoregressive inference
   - Augmentation: temporal jitter, trend scaling, feature dropout

**Normalization**:
- StandardScaler (recommended): mean=0, std=1, best for neural networks
- MinMaxScaler: [0, 1] range, compact bounded
- RobustScaler: resistant to outliers, good for noisy sensors

**Multi-year support**:
All dataloaders accept single year (int) or multiple years (list of ints).
Scaler is fit globally on combined data from all years.

Examples
--------

Single-year stint dataloader:

    >>> from code.dataloaders import StintDataloader
    >>> ds = StintDataloader(year=2019, window_size=20)
    >>> print(f"Total stints: {len(ds)}")
    >>> sample = ds[0]
    >>> features, target_laptime, metadata = sample
    >>> print(features.shape)  # (20, num_features)

Multi-year autoregressive dataloader:

    >>> from code.dataloaders import AutoregressiveLapDataloader
    >>> from torch.utils.data import DataLoader
    >>> ds = AutoregressiveLapDataloader(
    ...     year=[2019, 2020, 2021],
    ...     context_window=5,
    ...     augment_prob=0.5
    ... )
    >>> print(f"Total lap pairs: {len(ds)}")
    >>> loader = DataLoader(ds, batch_size=32, shuffle=True)
    >>> for context_batch, target_batch, metadata in loader:
    ...     print(context_batch.shape)  # (32, 5, num_features)
    ...     print(target_batch.shape)   # (32,)
    ...     break

Accessing metadata:

    >>> ds = AutoregressiveLapDataloader(year=[2020, 2021])
    >>> _, _, metadata = ds[100]
    >>> print(f"Driver {metadata['driver']} at {metadata['race_name']}")

Using normalization:

    >>> from code.dataloaders import LapTimeNormalizer
    >>> norm = LapTimeNormalizer(scaler_type="robust")  # Use RobustScaler
    >>> ds = StintDataloader(year=2019)  # Uses normalizer internally
    >>> stats = ds.normalizer.get_statistics()
    >>> print(f"LapTime mean: {stats['mean'][0]:.0f} ms")
"""

from .stint_dataloader import StintDataloader
from .autoregressive_dataloader import AutoregressiveLapDataloader
from .normalization import LapTimeNormalizer
from . import utils

__all__ = [
    'StintDataloader',
    'AutoregressiveLapDataloader',
    'LapTimeNormalizer',
    'utils',
]

__version__ = '1.0.0'
