"""
Autoregressive lap-by-lap data loader for F1 lap prediction.

This dataloader returns (context_laps, next_lap_time) pairs suitable for:
- Autoregressive inference (feed predictions back as context)
- Teacher-forcing training (feed ground truth as context)
- Evaluation with variable-length contexts

Use case: Training sequential models for next-lap-time prediction with flexible context windows.
"""

import logging
from pathlib import Path
from typing import Union, List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from .utils import (
    normalize_year_input,
    load_all_races,
    get_numeric_columns,
    get_categorical_columns,
    get_boolean_columns,
    get_compound_columns,
    extract_lap_features_vectorized,
)
from .normalization import LapTimeNormalizer


logger = logging.getLogger(__name__)


class AutoregressiveLapDataloader(Dataset):
    """
    PyTorch Dataset for autoregressive F1 lap-by-lap prediction.
    
    Loads race data from one or more years and generates (context_laps, next_lap_time)
    pairs for training sequence models. Supports both teacher-forcing and autoregressive
    inference paradigms.
    
    Parameters
    ----------
    year : int or list of int
        Season year(s) to load, e.g., 2019 or [2019, 2020, 2021]
    context_window : int, optional
        Number of previous laps to use as context. Default is 5.
        If a driver has fewer laps available, sequence is zero-padded.
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
    lap_pairs : list of dict
        List of (context_laps, next_lap_time) pairs with metadata
    normalizer : LapTimeNormalizer
        Fitted scaler for continuous features
    
    Data Augmentation
    ---------------
    Three augmentation strategies are implemented:
    
    1. Temporal Jitter
       - Add Gaussian noise ±2% to LapTime, delta_to_car_ahead
       - Simulates measurement uncertainty
    
    2. Trend Augmentation
       - Linearly scale all lap times in context and target by ±3%
       - Simulates driver form variation across session
    
    3. Feature Dropout
       - Randomly zero out weather columns with 10% probability
       - Simulates missing sensor data
    
    Examples
    --------
    >>> # Load and create dataloader
    >>> ds = AutoregressiveLapDataloader(year=2019, context_window=5)
    >>> print(f"Total lap pairs: {len(ds)}")
    
    >>> # Multiple years
    >>> ds_multi = AutoregressiveLapDataloader(year=[2019, 2020, 2021])
    >>> sample = ds_multi[100]
    >>> context, target, metadata = sample
    >>> print(context.shape)  # (context_window, num_features)
    >>> print(target)  # scalar target lap time
    
    >>> # With augmentation and batch loading
    >>> ds_aug = AutoregressiveLapDataloader(
    ...     year=[2020, 2021],
    ...     context_window=10,
    ...     augment_prob=0.5,
    ...     seed=42
    ... )
    >>> from torch.utils.data import DataLoader
    >>> loader = DataLoader(ds_aug, batch_size=64, shuffle=True, num_workers=2)
    >>> for context_batch, target_batch, metadata_batch in loader:
    ...     print(context_batch.shape)  # (batch_size, context_window, num_features)
    ...     break
    """
    
    def __init__(
        self,
        year: Union[int, List[int]],
        context_window: int = 5,
        multistep_horizon: int = 1,
        augment_prob: float = 0.0,
        normalize: bool = True,
        scaler_type: str = "standard",
        normalizer: Optional[LapTimeNormalizer] = None,
        require_normalizer: bool = False,
        data_path: Path = None,
        cache_path: Optional[Path] = None,
        device: str = 'cpu',
        seed: int = None,
    ):
        self.years = normalize_year_input(year)
        self.context_window = context_window
        self.multistep_horizon = max(1, multistep_horizon)
        self.augment_prob = augment_prob
        self.device = device
        self.data_path = data_path or Path("./data")
        # Cache file for precomputed contexts/targets
        self.cache_path = Path(cache_path) if cache_path is not None else None
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Define feature columns
        self.numeric_columns = get_numeric_columns()
        self.categorical_columns = get_categorical_columns()
        self.boolean_columns = get_boolean_columns()
        self.compound_columns = get_compound_columns()
        self.all_features = (
            self.numeric_columns +
            self.categorical_columns +
            self.boolean_columns +
            self.compound_columns
        )
        self._numeric_index = {c: i for i, c in enumerate(self.numeric_columns)}
        self._weather_numeric_indices = [
            self._numeric_index[c]
            for c in ['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed', 'WindDirection']
            if c in self._numeric_index
        ]
        
        # Load and prepare data
        logger.info(f"Loading race data for years {self.years}...")
        self.data = load_all_races(self.years, session="Race", data_path=self.data_path)
        
        # Filter out extreme outliers in lap times
        # Reasonable F1 lap times are 55s-200s (55K-200K ms)
        # Anything beyond this is likely red flags, data errors, or safety car crawling
        original_len = len(self.data)
        lap_time_max = 200000  # 200 seconds (3 min 20 sec) max reasonable lap
        self.data = self.data[self.data['LapTime'] <= lap_time_max].reset_index(drop=True)
        filtered_count = original_len - len(self.data)
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} laps with LapTime > {lap_time_max/1000:.0f}s ({filtered_count/original_len*100:.2f}%)")
        
        # Setup normalization
        self.normalizer = None
        if normalize:
            if normalizer is not None:
                self.normalizer = normalizer
                logger.info("Using provided normalizer")
            elif require_normalizer:
                raise ValueError(
                    f"require_normalizer=True but no normalizer was provided for years {self.years}. "
                    f"Fit a normalizer on training data and pass it explicitly to prevent data leakage."
                )
            else:
                self.normalizer = LapTimeNormalizer(scaler_type=scaler_type)
                # Try to load fitted scaler, if not available fit on data
                try:
                    self.normalizer.load(self.years)
                    logger.info(f"Loaded pre-fitted scaler for years {self.years}")
                except FileNotFoundError:
                    logger.warning(
                        f"No pre-fitted scaler found for years {self.years}. "
                        f"Fitting scaler on provided data. If these are val/test years this "
                        f"will cause DATA LEAKAGE — pass a train-fitted normalizer instead."
                    )
                    numeric_data = self.data[self.numeric_columns].copy()
                    self.normalizer.fit(numeric_data, years=self.years)
        
        # Generate lap pairs
        logger.info("Generating (context_lap, next_lap) pairs...")
        self.lap_pairs = self._generate_lap_pairs()
        logger.info(f"Created {len(self.lap_pairs)} lap pairs")

        # Optionally skip precomputation via environment variable to avoid
        # long startup on interactive runs. Set `SKIP_PRECOMPUTE=1` to skip.
        if os.environ.get('SKIP_PRECOMPUTE') == '1':
            logger.info("SKIP_PRECOMPUTE=1: skipping precompute of feature tensors")
            self._precomputed_contexts = [None] * len(self.lap_pairs)
            self._precomputed_targets = [None] * len(self.lap_pairs)
            self._precomputed_meta = [None] * len(self.lap_pairs)
        else:
            self._scaler_type = scaler_type
            if self.cache_path is None:
                cache_dir = Path(self.data_path) / "precomputed"
                cache_dir.mkdir(parents=True, exist_ok=True)
                years_key = "_".join(map(str, sorted(self.years)))
                cache_fname = f"ar_cache_{years_key}_cw{self.context_window}_h{self.multistep_horizon}_{scaler_type}.pt"
                self.cache_path = cache_dir / cache_fname

            if not self._load_cache():
                self._precompute_all()
                self._save_cache()
            else:
                logger.info("Using cached precomputed tensors; skipping recomputation.")
    
    @staticmethod
    def _safe_torch_load(p: Path):
        """Load a torch file with a progressive fallback chain for cross-version compat."""
        safe_globals = [
            np._core.multiarray.scalar,
            np._core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
        ]
        try:
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals(safe_globals)
        except Exception:
            pass

        for weights_only in (True, True, False):
            try:
                if weights_only and hasattr(torch.serialization, 'safe_globals'):
                    with torch.serialization.safe_globals(safe_globals):
                        return torch.load(p, weights_only=True)
                return torch.load(p, weights_only=weights_only)
            except Exception:
                pass

        import pickle
        with open(p, 'rb') as f:
            return pickle.load(f)

    def _load_cache(self) -> bool:
        """Try to load precomputed tensors from disk cache.

        Returns True if a valid, compatible cache was loaded; False otherwise.
        """
        if not self.cache_path.exists():
            return False

        try:
            logger.info(f"Loading cached precomputed dataset from {self.cache_path}...")
            cached = self._safe_torch_load(self.cache_path)

            normalizer_years = getattr(self.normalizer, 'years', None) if self.normalizer else None
            valid = (
                cached.get('years') == self.years
                and cached.get('context_window') == self.context_window
                and cached.get('multistep_horizon', 1) == self.multistep_horizon
                and cached.get('scaler_type') == self._scaler_type
                and cached.get('numeric_columns') == self.numeric_columns
            )
            cached_scaler_years = cached.get('scaler_fitted_years')
            if valid and cached_scaler_years is not None and normalizer_years is not None:
                if cached_scaler_years != normalizer_years:
                    logger.warning(
                        f"Cache scaler provenance mismatch: cache used scaler fit on "
                        f"{cached_scaler_years} but current normalizer fit on {normalizer_years}. "
                        f"Invalidating cache."
                    )
                    valid = False

            if not valid:
                logger.info("Cache file did not match current dataloader configuration; ignoring cache.")
                return False

            # Contexts: new caches store tensors directly; legacy caches may store lists/ndarrays
            loaded_contexts = []
            for x in cached.get('contexts', []):
                if x is None:
                    loaded_contexts.append(None)
                elif isinstance(x, torch.Tensor):
                    loaded_contexts.append(x)
                elif isinstance(x, np.ndarray):
                    loaded_contexts.append(torch.from_numpy(x.astype(np.float32)))
                elif isinstance(x, list):
                    loaded_contexts.append(torch.tensor(np.array(x), dtype=torch.float32))
                else:
                    loaded_contexts.append(None)
            self._precomputed_contexts = loaded_contexts

            # Targets: new caches store dicts of tensors directly
            loaded_targets = []
            for tp in cached.get('targets', []):
                if tp is None:
                    loaded_targets.append(None)
                    continue
                lt = tp['lap_time'] if isinstance(tp['lap_time'], torch.Tensor) else torch.tensor(tp['lap_time'], dtype=torch.float32)
                pit = tp.get('is_pitlap', 0)
                pit = pit if isinstance(pit, torch.Tensor) else torch.tensor(pit, dtype=torch.long)
                comp = tp.get('compound', 0)
                comp = comp if isinstance(comp, torch.Tensor) else torch.tensor(comp, dtype=torch.long)
                tmask = tp.get('target_mask')
                if tmask is None:
                    tmask = torch.ones_like(lt)
                elif not isinstance(tmask, torch.Tensor):
                    tmask = torch.tensor(tmask, dtype=torch.float32)
                loaded_targets.append({'lap_time': lt, 'is_pitlap': pit, 'compound': comp, 'target_mask': tmask})
            self._precomputed_targets = loaded_targets
            self._precomputed_meta = cached['meta']
            logger.info(f"Loaded {len(self._precomputed_contexts)} cached context tensors")
            return True

        except Exception:
            logger.exception("Failed to load cache; will recompute precomputed tensors.")
            return False

    def _precompute_all(self) -> None:
        """Compute and store feature tensors for all lap pairs."""
        H = self.multistep_horizon
        n = len(self.lap_pairs)
        n_feat = (len(self.numeric_columns) + len(self.categorical_columns)
                  + len(self.boolean_columns) + len(self.compound_columns))

        logger.info(f"Precomputing feature tensors for all {n} lap pairs...")
        self._precomputed_contexts = [None] * n
        self._precomputed_targets = [None] * n
        self._precomputed_meta = [None] * n

        for i in tqdm(range(n), desc="Precomputing contexts", unit="pairs"):
            pair = self.lap_pairs[i]
            context_indices = pair['context_indices']
            target_indices = pair.get('target_indices', [pair['target_index']])

            context_laps = self.data.loc[context_indices].copy()
            target_laps = self.data.loc[target_indices].copy()

            if self.normalizer is not None:
                context_laps = self.normalizer.transform(context_laps)
                target_laps = self.normalizer.transform(target_laps)

            if len(context_laps) > 0:
                context_array = extract_lap_features_vectorized(
                    context_laps,
                    self.numeric_columns, self.categorical_columns,
                    self.boolean_columns, self.compound_columns,
                )
            else:
                context_array = np.zeros((0, n_feat), dtype=np.float32)

            if len(context_array) < self.context_window:
                padding = np.zeros((self.context_window - len(context_array), n_feat), dtype=np.float32)
                context_array = np.vstack([padding, context_array])

            target_laptimes = torch.zeros(H, dtype=torch.float32)
            target_pitflags = torch.zeros(H, dtype=torch.long)
            target_compounds = torch.zeros(H, dtype=torch.long)
            target_mask = torch.zeros(H, dtype=torch.float32)

            for h in range(min(len(target_laps), H)):
                t_lap = target_laps.iloc[h]
                lap_time_val = float(t_lap['LapTime'])
                if not np.isfinite(lap_time_val):
                    target_laptimes[h] = 0.0
                    target_mask[h] = 0.0
                else:
                    target_laptimes[h] = lap_time_val
                    target_mask[h] = 1.0
                target_pitflags[h] = int(t_lap.get('is_pitlap', 0))
                comp_vals = t_lap[self.compound_columns].values.astype(np.int32)
                target_compounds[h] = int(np.argmax(comp_vals)) if comp_vals.sum() > 0 else len(self.compound_columns) - 1

            self._precomputed_contexts[i] = torch.from_numpy(context_array)
            self._precomputed_targets[i] = {
                'lap_time': target_laptimes,
                'is_pitlap': target_pitflags,
                'compound': target_compounds,
                'target_mask': target_mask,
            }
            self._precomputed_meta[i] = {
                'driver': pair['driver'],
                'year': pair['year'],
                'circuit': pair['circuit'],
                'race_name': pair['race_name'],
                'context_length': len(context_indices),
            }

        logger.info(f"Precomputed all {n} context tensors in RAM")

    def _save_cache(self) -> None:
        """Persist precomputed tensors to disk using native torch serialization."""
        try:
            logger.info(f"Saving precomputed cache to {self.cache_path} ...")
            torch.save({
                'years': self.years,
                'context_window': self.context_window,
                'multistep_horizon': self.multistep_horizon,
                'scaler_type': self._scaler_type,
                'numeric_columns': self.numeric_columns,
                'scaler_fitted_years': getattr(self.normalizer, 'years', None),
                'contexts': self._precomputed_contexts,
                'targets': self._precomputed_targets,
                'meta': self._precomputed_meta,
            }, self.cache_path)
            logger.info(f"Cache saved to {self.cache_path}")
        except Exception:
            logger.exception("Failed to save precomputed cache")

    def _generate_lap_pairs(self) -> List[Dict]:
        """
        Generate (context_laps, next_lap) pairs from race data.
        
        Returns
        -------
        list of dict
            Each dict contains:
            - 'context_indices': indices of context laps in self.data
            - 'target_index': index of target lap in self.data
            - 'driver': driver ID
            - 'year': year
            - 'circuit': circuit ID
            - 'race_name': race/circuit name
        """
        pairs = []
        
        # Group by driver, year, circuit to respect race structure
        grouped = self.data.groupby(['Driver', 'Year', 'Circuit'])
        
        for (driver, year, circuit), race_data in grouped:
            # Filter to only normal laps
            normal_laps = race_data[race_data['is_normal_lap'] == 1].copy()
            
            if len(normal_laps) < 2:  # Need at least 1 context + 1 target
                continue
            
            # Sort by lap number while preserving original global indices
            normal_laps = normal_laps.sort_values('LapNumber')
            
            # Keep original global indices for selecting rows from self.data
            original_indices = normal_laps.index.tolist()
            
            # Generate pairs
            for i in range(1, len(normal_laps)):
                context_start = max(0, i - self.context_window)
                context_indices = original_indices[context_start:i]
                target_index = original_indices[i]

                # Collect up to multistep_horizon consecutive target indices
                target_end = min(i + self.multistep_horizon, len(normal_laps))
                target_indices = original_indices[i:target_end]

                pairs.append({
                    'context_indices': context_indices,
                    'target_index': target_index,
                    'target_indices': target_indices,
                    'driver': int(driver),
                    'year': int(year),
                    'circuit': int(circuit),
                    'race_name': race_data.iloc[0].get('Circuit', 'Unknown'),
                })
        
        return pairs
    
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
            Feature vector
        """
        features = []
        
        # Numeric features
        numeric_feat = lap[self.numeric_columns].values.astype(np.float32)
        features.append(numeric_feat)
        
        # Categorical features
        cat_feat = lap[self.categorical_columns].values.astype(np.int32)
        # Convert categorical to float for consistency
        features.append(cat_feat.astype(np.float32))
        
        # Boolean features
        bool_feat = lap[self.boolean_columns].values.astype(np.float32)
        features.append(bool_feat)
        
        # Compound one-hot features
        compound_feat = lap[self.compound_columns].values.astype(np.float32)
        features.append(compound_feat)
        
        return np.concatenate(features)
    
    def _apply_augmentation(self, context_laps: pd.DataFrame, target_lap: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply random data augmentation to context and target laps.
        
        Parameters
        ----------
        context_laps : pd.DataFrame
            Context lap sequence
        target_lap : pd.Series
            Target lap
        
        Returns
        -------
        tuple of (pd.DataFrame, pd.Series)
            Augmented context and target
        """
        context_laps = context_laps.copy()
        target_lap = target_lap.copy()
        
        aug_type = np.random.choice(3)  # 3 augmentation strategies
        
        if aug_type == 0:
            # Temporal jitter: add noise to timing columns
            noise_scale = np.random.normal(0, 0.02)  # ±2% std
            for col in ['LapTime', 'delta_to_car_ahead']:
                if col in context_laps.columns:
                    mask = context_laps[col].notna()
                    context_laps.loc[mask, col] = context_laps.loc[mask, col] * (1 + noise_scale)
            
            if 'LapTime' in target_lap.index and pd.notna(target_lap['LapTime']):
                target_lap['LapTime'] = target_lap['LapTime'] * (1 + noise_scale)
        
        elif aug_type == 1:
            # Trend augmentation: scale lap times ±3%
            scale = np.random.uniform(0.97, 1.03)
            context_laps['LapTime'] = context_laps['LapTime'] * scale
            if 'LapTime' in target_lap.index:
                target_lap['LapTime'] = target_lap['LapTime'] * scale
        
        else:
            # Feature dropout: zero out weather columns
            weather_cols = ['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed', 'WindDirection']
            for col in weather_cols:
                if col in context_laps.columns and np.random.random() < 0.1:
                    context_laps[col] = 0.0
        
        return context_laps, target_lap
    
    def __len__(self) -> int:
        """Return total number of lap pairs."""
        return len(self.lap_pairs)

    def _apply_cached_tensor_augmentation(
        self,
        context_tensor: torch.Tensor,
        target_tensor: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply lightweight augmentation directly on precomputed normalized tensors.

        This keeps cache speed while restoring stochastic augmentation during training.
        """
        context_aug = context_tensor.clone()
        target_aug = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in target_tensor.items()
        }

        aug_type = int(np.random.choice(3))

        if aug_type == 0:
            noise_scale = float(np.random.normal(0.0, 0.02))
            lap_idx = self._numeric_index.get('LapTime', None)
            dca_idx = self._numeric_index.get('delta_to_car_ahead', None)
            for idx in [lap_idx, dca_idx]:
                if idx is not None:
                    context_aug[:, idx] = context_aug[:, idx] * (1.0 + noise_scale)
            if 'lap_time' in target_aug and isinstance(target_aug['lap_time'], torch.Tensor):
                target_aug['lap_time'] = target_aug['lap_time'] * (1.0 + noise_scale)

        elif aug_type == 1:
            scale = float(np.random.uniform(0.97, 1.03))
            lap_idx = self._numeric_index.get('LapTime', None)
            if lap_idx is not None:
                context_aug[:, lap_idx] = context_aug[:, lap_idx] * scale
            if 'lap_time' in target_aug and isinstance(target_aug['lap_time'], torch.Tensor):
                target_aug['lap_time'] = target_aug['lap_time'] * scale

        else:
            for idx in self._weather_numeric_indices:
                if np.random.random() < 0.1:
                    context_aug[:, idx] = 0.0

        return context_aug, target_aug
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single (context, target) lap pair.
        
        Parameters
        ----------
        idx : int
            Index of lap pair
        
        Returns
        -------
        torch.Tensor
            Context features of shape (actual_context_length, num_features).
            Note: actual_context_length may be <= context_window (not padded at tensor level,
            but caller should handle variable-length sequences or use DataLoader collate_fn)
        torch.Tensor
            Target lap time (scalar, shape ())
        dict
            Metadata: {'driver', 'year', 'circuit', 'race_name', 'context_length'}
        """
        # Fast path: if this index was precomputed, return cached tensors
        if idx < len(self._precomputed_contexts) and self._precomputed_contexts[idx] is not None:
            context_tensor = self._precomputed_contexts[idx].clone()
            target_tensor = {
                k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in self._precomputed_targets[idx].items()
            }

            if self.augment_prob > 0.0 and np.random.random() < self.augment_prob:
                context_tensor, target_tensor = self._apply_cached_tensor_augmentation(context_tensor, target_tensor)

            context_tensor = context_tensor.to(self.device)
            for k, v in list(target_tensor.items()):
                if isinstance(v, torch.Tensor):
                    target_tensor[k] = v.to(self.device)
            metadata = self._precomputed_meta[idx]
            return context_tensor, target_tensor, metadata

        # Fallback: compute on the fly for indices not precomputed
        pair = self.lap_pairs[idx]
        context_indices = pair['context_indices']
        target_indices = pair.get('target_indices', [pair['target_index']])
        H = self.multistep_horizon

        context_laps = self.data.loc[context_indices].copy()
        target_laps = self.data.loc[target_indices].copy()

        # Apply augmentation if enabled (only on first target for backward compat)
        if np.random.random() < self.augment_prob and len(target_laps) > 0:
            first_target = target_laps.iloc[0].copy()
            context_laps, first_target = self._apply_augmentation(context_laps, first_target)
            target_laps.iloc[0] = first_target

        # Normalize numeric features
        if self.normalizer is not None:
            context_laps = self.normalizer.transform(context_laps)
            target_laps = self.normalizer.transform(target_laps)

        # Extract features (vectorized)
        if len(context_laps) > 0:
            context_array = extract_lap_features_vectorized(
                context_laps,
                self.numeric_columns,
                self.categorical_columns,
                self.boolean_columns,
                self.compound_columns,
            )
        else:
            n_feat = len(self.numeric_columns) + len(self.categorical_columns) + len(self.boolean_columns) + len(self.compound_columns)
            context_array = np.zeros((0, n_feat), dtype=np.float32)

        # Pad context to window size
        if len(context_array) < self.context_window:
            padding = np.zeros((self.context_window - len(context_array), context_array.shape[1]), dtype=np.float32)
            context_array = np.vstack([padding, context_array])

        # Multi-step target values
        target_laptimes = torch.zeros(H, dtype=torch.float32)
        target_pitflags = torch.zeros(H, dtype=torch.long)
        target_compounds = torch.zeros(H, dtype=torch.long)
        target_mask = torch.zeros(H, dtype=torch.float32)

        for h in range(min(len(target_laps), H)):
            t_lap = target_laps.iloc[h]
            lap_time_val = float(t_lap['LapTime'])
            if not np.isfinite(lap_time_val):
                target_laptimes[h] = 0.0
                target_mask[h] = 0.0
            else:
                target_laptimes[h] = lap_time_val
                target_mask[h] = 1.0
            target_pitflags[h] = int(t_lap.get('is_pitlap', 0))
            comp_vals = t_lap[self.compound_columns].values.astype(np.int32)
            if comp_vals.sum() == 0:
                target_compounds[h] = len(self.compound_columns) - 1
            else:
                target_compounds[h] = int(np.argmax(comp_vals))

        context_tensor = torch.from_numpy(context_array).to(self.device)
        target_tensor = {
            'lap_time': target_laptimes.to(self.device),
            'is_pitlap': target_pitflags.to(self.device),
            'compound': target_compounds.to(self.device),
            'target_mask': target_mask.to(self.device),
        }

        metadata = {
            'driver': pair['driver'],
            'year': pair['year'],
            'circuit': pair['circuit'],
            'race_name': pair['race_name'],
            'context_length': len(context_features),
        }

        return context_tensor, target_tensor, metadata
