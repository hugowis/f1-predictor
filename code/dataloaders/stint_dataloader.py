"""
Stint-based data loader for F1 lap prediction.

This dataloader groups laps into stints and provides sequences for training
recurrent models with teacher forcing. It automatically:
- Splits stints when encountering safety cars, red flags, yellow flags, or VSC
- Filters to only normal laps (no pit in/out laps)
- Normalizes continuous features using StandardScaler
- Supports multi-year loading
- Provides optional data augmentation

Use case: Training sequence-to-sequence models on stint-level sequences.
"""

import logging
from pathlib import Path
from typing import Union, List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import (
    normalize_year_input,
    load_all_races,
    get_numeric_columns,
    get_categorical_columns,
    get_boolean_columns,
    get_compound_columns,
)
from .normalization import LapTimeNormalizer


logger = logging.getLogger(__name__)


class StintDataloader(Dataset):
    """
    PyTorch Dataset for stint-based F1 lap sequences.
    
    Loads race data from one or more years, groups laps into stints,
    and provides sequences for teacher-forcing training.
    
    Parameters
    ----------
    year : int or list of int
        Season year(s) to load, e.g., 2019 or [2019, 2020, 2021]
    window_size : int, optional
        Maximum sequence length. Sequences longer than this are truncated,
        shorter ones are zero-padded. Default is 20.
    augment_prob : float, optional
        Probability of applying data augmentation to each sample. Default is 0.0 (no augmentation).
    normalize : bool, optional
        Whether to normalize continuous features. Default is True.
    data_path : Path, optional
        Path to data directory. Default is ./data
    device : str, optional
        Device to place tensors on ('cpu' or 'cuda'). Default is 'cpu'.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    
    Attributes
    ----------
    stints : list of dict
        List of stint sequences with metadata
    normalizer : LapTimeNormalizer
        Fitted scaler for continuously normalized features
    
    Data Augmentation
    ---------------
    Three augmentation strategies are implemented:
    
    1. Tyre Degradation Shift
       - Randomly shift TyreLife ±2 laps
       - Adjust LapTime based on simulated tyre performance curve
       - Simulates uncertainty in tyre age estimation
    
    2. Fuel Load Variation
       - Randomly scale fuel_proxy (0.8 to 1.2x)
       - Scale LapTime inversely (more fuel → slower)
       - Simulates different fuel strategies
    
    3. Weather Jitter
       - Add Gaussian noise ±5% to AirTemp, TrackTemp, Humidity, WindSpeed
       - Preserves weather patterns but adds sensor uncertainty
    
    Examples
    --------
    >>> # Load single year
    >>> ds = StintDataloader(year=2019, window_size=20)
    >>> print(f"Total stints: {len(ds)}")
    
    >>> # Load multiple years
    >>> ds_multi = StintDataloader(year=[2019, 2020, 2021], window_size=15)
    >>> sample = ds_multi[0]
    >>> print(sample[0].shape)  # Sequence features
    >>> print(sample[1])  # Target lap time
    >>> print(sample[2])  # Metadata
    
    >>> # With augmentation
    >>> ds_aug = StintDataloader(year=2021, augment_prob=0.5, seed=42)
    >>> from torch.utils.data import DataLoader
    >>> loader = DataLoader(ds_aug, batch_size=32, shuffle=True)
    """
    
    def __init__(
        self,
        year: Union[int, List[int]],
        window_size: int = 20,
        augment_prob: float = 0.0,
        normalize: bool = True,
        scaler_type: str = "standard",
        normalizer: Optional[LapTimeNormalizer] = None,
        data_path: Path = None,
        device: str = 'cpu',
        seed: int = None,
    ):
        self.years = normalize_year_input(year)
        self.window_size = window_size
        self.augment_prob = augment_prob
        self.device = device
        self.data_path = data_path or Path("./data")
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Define feature columns
        self.numeric_columns = get_numeric_columns()
        self.categorical_columns = get_categorical_columns()
        self.boolean_columns = get_boolean_columns()
        self.compound_columns = get_compound_columns()
        
        # Load and prepare data
        logger.info(f"Loading race data for years {self.years}...")
        self.data = load_all_races(self.years, session="Race", data_path=self.data_path)
        
        # Setup normalization
        self.normalizer = None
        if normalize:
            if normalizer is not None:
                self.normalizer = normalizer
                logger.info("Using provided normalizer")
            else:
                self.normalizer = LapTimeNormalizer(scaler_type=scaler_type)
                # Try to load fitted scaler, if not available fit on data
                try:
                    self.normalizer.load(self.years)
                    logger.info(f"Loaded pre-fitted scaler for years {self.years}")
                except FileNotFoundError:
                    logger.info(f"Fitting scaler on data from years {self.years}...")
                    numeric_data = self.data[self.numeric_columns].copy()
                    self.normalizer.fit(numeric_data, years=self.years)
        
        # Generate stint sequences
        logger.info("Generating stint sequences...")
        self.stints = self._generate_stints()
        logger.info(f"Created {len(self.stints)} stint sequences")
    
    def _generate_stints(self) -> List[Dict]:
        """
        Extract and prepare stint sequences from race data.
        
        Returns
        -------
        list of dict
            Each dict contains:
            - 'laps': sequence of lap data (one hot per lap)
            - 'driver': driver ID
            - 'year': year
            - 'circuit': circuit ID
            - 'stint_id': stint identifier
        """
        stints = []
        
        # Group by driver, year, circuit, and stint
        grouped = self.data.groupby(['Driver', 'Year', 'Circuit', 'Stint'])
        
        for (driver, year, circuit, stint_id), group in grouped:
            # Filter to only normal laps (no pit in/out laps)
            normal_laps = group[group['is_normal_lap'] == 1].copy()
            
            if len(normal_laps) < 2:  # Need at least 2 laps to form input->target
                continue
            
            # Sort by lap number to ensure temporal order
            normal_laps = normal_laps.sort_values('LapNumber').reset_index(drop=True)
            
            # Split stint if encountering abnormal track conditions
            stint_sequences = self._split_on_flags(normal_laps)
            
            for seq in stint_sequences:
                if len(seq) >= 2:  # Need at least input + target
                    stints.append({
                        'laps': seq,
                        'driver': int(driver),
                        'year': int(year),
                        'circuit': int(circuit),
                        'stint_id': int(stint_id),
                        'length': len(seq),
                    })
        
        return stints
    
    def _split_on_flags(self, laps: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Split a stint sequence at abnormal track conditions (flags, VSC, etc).
        
        Parameters
        ----------
        laps : pd.DataFrame
            Laps from a single stint
        
        Returns
        -------
        list of pd.DataFrame
            Split sequences, each a continuous normal-condition stint
        """
        # Identify rows with abnormal conditions
        abnormal = (
            (laps['safety_car'] == 1) |
            (laps['red_flag'] == 1) |
            (laps['yellow_flag'] == 1) |
            (laps['vsc'] == 1)
        )
        
        # Get indices where condition changes from normal to abnormal
        flag_indices = set(np.where(abnormal)[0].tolist())
        
        if not flag_indices:
            # No flags, return whole stint
            return [laps]
        
        # Split at flag boundaries
        sequences = []
        start = 0
        
        for flag_idx in sorted(flag_indices):
            if flag_idx > start:
                sequences.append(laps.iloc[start:flag_idx])
            start = flag_idx + 1
        
        # Add remaining laps after last flag
        if start < len(laps):
            sequences.append(laps.iloc[start:])
        
        # Filter out short sequences
        return [seq for seq in sequences if len(seq) >= 2]
    
    def _get_lap_features(self, lap: pd.Series) -> np.ndarray:
        """
        Extract and prepare features from a single lap.
        
        Parameters
        ----------
        lap : pd.Series
            Single lap row from dataframe
        
        Returns
        -------
        np.ndarray
            Feature vector [numeric_features + categorical_features + boolean_features + compound_features]
        """
        # Numeric features (will be normalized)
        numeric_feat = lap[self.numeric_columns].values.astype(np.float32)
        # Boolean features
        bool_feat = lap[self.boolean_columns].values.astype(np.float32)
        # Compound one-hot features
        compound_feat = lap[self.compound_columns].values.astype(np.float32)

        # Concatenate numeric, categorical, boolean and compound into single vector
        cat_feat = lap[self.categorical_columns].values.astype(np.int32)
        return np.concatenate([numeric_feat, cat_feat, bool_feat, compound_feat])
    
    def _apply_augmentation(self, laps: pd.DataFrame) -> pd.DataFrame:
        """
        Apply random data augmentation to stint sequence.
        
        Parameters
        ----------
        laps : pd.DataFrame
            Stint sequence to augment
        
        Returns
        -------
        pd.DataFrame
            Augmented stint sequence
        """
        laps = laps.copy()
        aug_type = np.random.choice(3)  # 3 augmentation strategies
        
        if aug_type == 0:
            # Tyre degradation shift
            shift = np.random.randint(-2, 3)  # ±2 laps
            laps['TyreLife'] = (laps['TyreLife'] + shift).clip(lower=1)
            # Adjust LapTime: fresh tyre ~2-3% faster, old tyre ~1-2% slower per lap
            tyre_age = laps['TyreLife'].values
            adjustment = 1.0 + (tyre_age - 1) * 0.01  # 1% per lap
            laps['LapTime'] = laps['LapTime'] * adjustment
        
        elif aug_type == 1:
            # Fuel load variation
            fuel_scale = np.random.uniform(0.8, 1.2)
            laps['fuel_proxy'] = laps['fuel_proxy'] * fuel_scale
            # More fuel → slightly slower
            laps['LapTime'] = laps['LapTime'] * (1 + (fuel_scale - 1) * 0.15)
        
        else:
            # Weather jitter
            weather_cols = ['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed']
            for col in weather_cols:
                if col in laps.columns and laps[col].notna().any():
                    noise = np.random.normal(0, 0.05)  # 5% std
                    laps[col] = laps[col] * (1 + noise)
        
        return laps
    
    def __len__(self) -> int:
        """Return total number of stint sequences."""
        return len(self.stints)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single stint sequence.
        
        Parameters
        ----------
        idx : int
            Index of stint sequence
        
        Returns
        -------
        torch.Tensor
            Input features of shape (window_size, num_features)
        torch.Tensor
            Target lap time (scalar, shape ())
        dict
            Metadata: {'driver', 'year', 'circuit', 'stint_id', 'length'}
        """
        stint = self.stints[idx]
        laps = stint['laps'].copy()
        
        # Apply augmentation if enabled
        if np.random.random() < self.augment_prob:
            laps = self._apply_augmentation(laps)
        
        # Normalize numeric features
        if self.normalizer is not None:
            laps = self.normalizer.transform(laps)
        
        # Extract features and target
        # Input: all laps except last
        # Target: last lap's time
        input_laps = laps.iloc[:-1]
        target_lap = laps.iloc[-1]
        
        # Pad or truncate to window size
        if len(input_laps) > self.window_size:
            input_laps = input_laps.iloc[-self.window_size:]
        
        # Build feature matrix
        features_list = []
        for _, lap in input_laps.iterrows():
            feat = self._get_lap_features(lap)
            features_list.append(feat)

        # Convert to tensor
        if features_list:
            features = np.array(features_list, dtype=np.float32)
        else:
            # Padding case: empty input sequence
            num_features = len(self._get_lap_features(input_laps.iloc[0])) if len(input_laps) > 0 else 100
            features = np.zeros((0, num_features), dtype=np.float32)

        # Pad sequence to window size
        if len(features) < self.window_size:
            padding = np.zeros((self.window_size - len(features), features.shape[1]), dtype=np.float32)
            features = np.vstack([padding, features])

        # Get target lap time
        target_laptime = float(target_lap['LapTime'])

        # Convert to tensors
        features_tensor = torch.from_numpy(features).to(self.device)
        target_tensor = torch.tensor(target_laptime, dtype=torch.float32, device=self.device)

        metadata = {
            'driver': stint['driver'],
            'year': stint['year'],
            'circuit': stint['circuit'],
            'stint_id': stint['stint_id'],
            'length': stint['length'],
        }

        return features_tensor, target_tensor, metadata
