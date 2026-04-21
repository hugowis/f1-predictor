"""
Data normalization utilities for F1 lap data.

This module provides efficient normalization handling for continuous features
across multiple years of F1 data. It supports three main normalization strategies:

1. StandardScaler (RECOMMENDED)
   - Transforms: (x - mean) / std
   - Range: unbounded, centered at 0 with std=1
   - Best for: neural networks with centered activations (ReLU, Tanh)
   - Pros: No information loss, symmetric, good numerical stability
   - Cons: Sensitive to outliers
   - Use when: Training deep learning models

2. MinMaxScaler
   - Transforms: (x - min) / (max - min)
   - Range: [0, 1]
   - Best for: neural networks where bounded input helps regularization
   - Pros: Interpretable, compact range, preserves relationships
   - Cons: Very sensitive to outliers (extreme values shrink all others)
   - Use when: Data is clean and bounded, or you need interpretability
   
3. RobustScaler
   - Transforms: (x - median) / IQR
   - Range: typically [-0.7, 0.7] for normal distributions
   - Best for: data with outliers or noisy sensors
   - Pros: Robust to outliers, uses medians/quartiles
   - Cons: Less information from tails, unbounded range
   - Use when: Data contains sensor errors or measurement noise

For F1 lap data specifically:
- LapTime: Use StandardScaler (mostly continuous, few extreme outliers)
- AirTemp/TrackTemp: StandardScaler (well-behaved distributions)
- Humidity/Pressure: StandardScaler (bounded but not [0,1] naturally)
- Position: Consider RobustScaler (can have extreme values P1 vs P20)
- delta_to_car_ahead: RobustScaler (occasionally very large gaps)
"""

import pickle
import logging
from pathlib import Path
from typing import Union, List, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from .utils import get_numeric_columns, get_scaler_key, normalize_year_input


logger = logging.getLogger(__name__)


class LapTimeNormalizer:
    """
    Manage normalization of continuous F1 lap data.
    
    This class fits a StandardScaler (recommended) on data from one or more years
    and provides transform/inverse_transform methods. Scalers are cached on disk
    to avoid refitting on repeated data loads.
    
    Parameters
    ----------
    scaler_type : str, optional
        Type of scaler: "standard", "minmax", or "robust". Default is "standard".
    scaler_dir : Path, optional
        Directory to save/load fitted scalers. Default is ./f1predictor/dataloaders/scalers/
    columns : list of str, optional
        Specific columns to normalize. If None, uses default numeric columns.
    
    Attributes
    ----------
    scaler : sklearn scaler object
        The fitted scaler
    columns : list of str
        Columns that are normalized
    years : list of int
        Years the scaler was fit on
    """
    
    def __init__(
        self,
        scaler_type: str = "standard",
        scaler_dir: Path = None,
        columns: List[str] = None,
    ):
        self.scaler_type = scaler_type
        self.scaler_dir = scaler_dir or Path(__file__).parent / "scalers"
        self.scaler_dir.mkdir(parents=True, exist_ok=True)
        
        # Select scaler class
        if scaler_type == "standard":
            self.ScalerClass = StandardScaler
        elif scaler_type == "minmax":
            self.ScalerClass = MinMaxScaler
        elif scaler_type == "robust":
            self.ScalerClass = RobustScaler
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.scaler = None
        self.columns = columns or get_numeric_columns()
        self.years = None
    
    def fit(self, data: pd.DataFrame, years: Union[int, List[int]] = None):
        """
        Fit the scaler on data from specified year(s).
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to fit on (should be pre-filtered to numeric columns)
        years : int or list of int, optional
            Years the data came from (for caching). If provided, scaler will be saved.
        
        Returns
        -------
        self
            For method chaining
        
        Examples
        --------
        >>> from f1predictor.dataloaders.utils import load_all_races
        >>> df = load_all_races([2019, 2020])
        >>> norm = LapTimeNormalizer()
        >>> norm.fit(df[[col for col in df.columns if col in norm.columns]], years=[2019, 2020])
        >>> normalized = norm.transform(df)
        """
        if self.years is None and years is not None:
            self.years = normalize_year_input(years)
        
        # Select only columns that exist and should be normalized
        cols_to_fit = [c for c in self.columns if c in data.columns]
        
        if not cols_to_fit:
            raise ValueError(f"No columns to fit. Expected {self.columns}, got {data.columns.tolist()}")
        
        # Drop NaN values for fitting
        data_clean = data[cols_to_fit].dropna()
        
        logger.info(f"Fitting {self.scaler_type}Scaler on {len(data_clean)} samples")
        self.scaler = self.ScalerClass()
        self.scaler.fit(data_clean)
        
        # Save scaler if years are provided
        if self.years is not None:
            self.save(self.years)
        
        return self

    def _get_column_transform_params(self, column_index: int):
        """Return per-column affine transform parameters for the fitted scaler."""
        if isinstance(self.scaler, StandardScaler):
            offset = float(self.scaler.mean_[column_index])
            scale = float(self.scaler.scale_[column_index])
            return offset, scale, 'standard'
        if isinstance(self.scaler, MinMaxScaler):
            offset = float(self.scaler.min_[column_index])
            scale = float(self.scaler.scale_[column_index])
            return offset, scale, 'minmax'
        if isinstance(self.scaler, RobustScaler):
            offset = float(self.scaler.center_[column_index])
            scale = float(self.scaler.scale_[column_index])
            return offset, scale, 'robust'
        raise RuntimeError(f"Unsupported scaler type: {type(self.scaler).__name__}")

    def _transform_column_values(self, values: pd.Series, column_index: int) -> pd.Series:
        """Transform a single numeric column while preserving NaNs."""
        offset, scale, mode = self._get_column_transform_params(column_index)
        if np.isclose(scale, 0.0):
            scale = 1.0

        valid_mask = values.notna()
        if not valid_mask.any():
            return values

        if mode == 'minmax':
            values.loc[valid_mask] = values.loc[valid_mask] * scale + offset
        else:
            values.loc[valid_mask] = (values.loc[valid_mask] - offset) / scale
        return values

    def _inverse_transform_column_values(self, values: pd.Series, column_index: int) -> pd.Series:
        """Inverse-transform a single numeric column while preserving NaNs."""
        offset, scale, mode = self._get_column_transform_params(column_index)
        if np.isclose(scale, 0.0):
            scale = 1.0

        valid_mask = values.notna()
        if not valid_mask.any():
            return values

        if mode == 'minmax':
            values.loc[valid_mask] = (values.loc[valid_mask] - offset) / scale
        else:
            values.loc[valid_mask] = values.loc[valid_mask] * scale + offset
        return values
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization to data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to normalize (must have same columns as training data)
        
        Returns
        -------
        pd.DataFrame
            Normalized data with same shape and index
        
        Raises
        ------
        RuntimeError
            If scaler has not been fit yet
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")
        
        data_copy = data.copy()
        cols_to_transform = [c for c in self.columns if c in data_copy.columns]
        
        # Convert columns to float to avoid dtype warning when assigning normalized values
        data_copy[cols_to_transform] = data_copy[cols_to_transform].astype(np.float64)

        for column_index, column_name in enumerate(cols_to_transform):
            data_copy[column_name] = self._transform_column_values(
                data_copy[column_name].astype(np.float64).copy(),
                column_index,
            )
        
        return data_copy
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse normalization (denormalize) data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Normalized data
        
        Returns
        -------
        pd.DataFrame
            Original scale data
        
        Raises
        ------
        RuntimeError
            If scaler has not been fit yet
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")
        
        data_copy = data.copy()
        cols_to_inverse = [c for c in self.columns if c in data_copy.columns]
        
        # Convert columns to float to handle denormalization properly
        data_copy[cols_to_inverse] = data_copy[cols_to_inverse].astype(np.float64)

        for column_index, column_name in enumerate(cols_to_inverse):
            data_copy[column_name] = self._inverse_transform_column_values(
                data_copy[column_name].astype(np.float64).copy(),
                column_index,
            )
        
        return data_copy
    
    def save(self, years: Union[int, List[int]]):
        """
        Save fitted scaler to disk for later reuse.
        
        Parameters
        ----------
        years : int or list of int
            Years the scaler was fit on (used for filename)
        
        Examples
        --------
        >>> norm = LapTimeNormalizer()
        >>> norm.fit(train_data, years=[2019, 2020])
        >>> norm.save([2019, 2020])  # Saved as: scalers/2019_2020.pkl
        """
        years_list = normalize_year_input(years)
        key = get_scaler_key(years_list)
        
        filepath = self.scaler_dir / f"{key}_{self.scaler_type}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'columns': self.columns,
                'years': years_list,
                'scaler_type': self.scaler_type,
            }, f)
        
        logger.info(f"Scaler saved to {filepath}")
    
    def load(self, years: Union[int, List[int]]):
        """
        Load fitted scaler from disk.
        
        Parameters
        ----------
        years : int or list of int
            Years to load scaler for
        
        Returns
        -------
        self
            For method chaining
        
        Raises
        ------
        FileNotFoundError
            If no saved scaler found for the specified years
        
        Examples
        --------
        >>> norm = LapTimeNormalizer()
        >>> norm.load([2019, 2020])
        >>> normalized_data = norm.transform(data)
        """
        years_list = normalize_year_input(years)
        key = get_scaler_key(years_list)
        
        filepath = self.scaler_dir / f"{key}_{self.scaler_type}.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"No saved scaler found for years {years_list} at {filepath}. "
                f"Call fit() first to create a scaler."
            )
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        saved_years = data.get('years')
        if saved_years is not None and years_list != saved_years:
            raise ValueError(
                f"Scaler provenance mismatch: requested years {years_list} "
                f"but scaler at {filepath} was fit on {saved_years}. "
                f"Pass the correct years or re-fit the scaler on training data only."
            )

        # Reject scalers whose column set no longer matches the current
        # numeric-feature registry.  A mismatch means the scaler was fit
        # before a feature-schema change (e.g. Part C added 16 columns):
        # loading it would leave the new columns un-normalised and the
        # model would see raw millisecond values, producing NaN on the
        # first forward pass.  Force a refit instead of silently breaking.
        saved_columns = data.get('columns', [])
        expected_columns = get_numeric_columns()
        if list(saved_columns) != list(expected_columns):
            raise FileNotFoundError(
                f"Scaler at {filepath} was fit on columns "
                f"{saved_columns} but the current feature registry is "
                f"{expected_columns}. Delete the stale scaler and re-fit "
                f"on training data."
            )

        self.scaler = data['scaler']
        self.columns = data['columns']
        self.years = data['years']
        self.scaler_type = data['scaler_type']

        logger.info(f"Scaler loaded from {filepath}")
        return self
    
    def get_statistics(self) -> Dict[str, np.ndarray]:
        """
        Get normalization statistics (mean, std, etc.) for analysis.
        
        Returns
        -------
        dict
            Statistics depending on scaler type:
            - StandardScaler: {'mean': array, 'std': array}
            - MinMaxScaler: {'min': array, 'max': array}
            - RobustScaler: {'center': array, 'scale': array}
        
        Examples
        --------
        >>> norm = LapTimeNormalizer('standard')
        >>> norm.fit(data)
        >>> stats = norm.get_statistics()
        >>> print(f"LapTime mean: {stats['mean'][0]}, std: {stats['std'][0]}")
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not fit. Call fit() first.")
        
        if isinstance(self.scaler, StandardScaler):
            return {
                'mean': self.scaler.mean_,
                'std': self.scaler.scale_,
                'columns': self.columns,
            }
        elif isinstance(self.scaler, MinMaxScaler):
            return {
                'min': self.scaler.data_min_,
                'max': self.scaler.data_max_,
                'columns': self.columns,
            }
        elif isinstance(self.scaler, RobustScaler):
            return {
                'center': self.scaler.center_,
                'scale': self.scaler.scale_,
                'columns': self.columns,
            }
