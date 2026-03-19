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
        augment_prob: float = 0.0,
        normalize: bool = True,
        scaler_type: str = "standard",
        normalizer: Optional[LapTimeNormalizer] = None,
        data_path: Path = None,
        cache_path: Optional[Path] = None,
        device: str = 'cpu',
        seed: int = None,
    ):
        self.years = normalize_year_input(year)
        self.context_window = context_window
        self.augment_prob = augment_prob
        self.scaler_type = scaler_type
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
            # Attempt to load precomputed cache if available to speed startup.
            # Cache must match: years, context_window, scaler_type and numeric column set.
            cache_loaded = False
            if self.cache_path is None:
                # default cache location inside data_path/precomputed/
                cache_dir = Path(self.data_path) / "precomputed"
                cache_dir.mkdir(parents=True, exist_ok=True)
                years_key = "_".join(map(str, sorted(self.years)))
                cache_fname = f"ar_cache_{years_key}_cw{self.context_window}_{scaler_type}.pt"
                self.cache_path = cache_dir / cache_fname

            if self.cache_path.exists():
                try:
                    logger.info(f"Loading cached precomputed dataset from {self.cache_path}...")
                    def _safe_torch_load(p):
                        safe_globals = [
                            np._core.multiarray.scalar,
                            np._core.multiarray._reconstruct,
                            np.ndarray,
                            np.dtype,
                        ]

                        # Pre-allowlist for PyTorch>=2.6 weights_only default behavior
                        try:
                            if hasattr(torch.serialization, 'add_safe_globals'):
                                torch.serialization.add_safe_globals(safe_globals)
                        except Exception:
                            pass

                        # 1) Preferred: safe weights-only load
                        try:
                            return torch.load(p, weights_only=True)
                        except Exception:
                            pass

                        # 2) Safe globals context fallback (older/alternate torch APIs)
                        try:
                            if hasattr(torch.serialization, 'safe_globals'):
                                with torch.serialization.safe_globals(safe_globals):
                                    return torch.load(p, weights_only=True)
                        except Exception:
                            pass

                        # 3) Trusted-local-file fallback for legacy caches
                        try:
                            logger.info("Falling back to torch.load(weights_only=False) for trusted local cache")
                            return torch.load(p, weights_only=False)
                        except Exception:
                            pass

                        # 4) Last resort: pickle
                        import pickle
                        with open(p, 'rb') as f:
                            return pickle.load(f)

                    cached = _safe_torch_load(self.cache_path)
                    # Validate cache compatibility
                    valid = (
                        cached.get('years') == self.years and
                        cached.get('context_window') == self.context_window and
                        cached.get('scaler_type') == scaler_type and
                        cached.get('numeric_columns') == self.numeric_columns
                    )
                    if valid:
                        # Reconstruct contexts: skip None entries safely and accept lists/ndarrays/tensors
                        loaded_contexts = []
                        for x in cached.get('contexts', []):
                            if x is None:
                                loaded_contexts.append(None)
                            elif isinstance(x, torch.Tensor):
                                loaded_contexts.append(x)
                            elif isinstance(x, np.ndarray):
                                loaded_contexts.append(torch.from_numpy(x))
                            elif isinstance(x, list):
                                try:
                                    loaded_contexts.append(torch.tensor(x, dtype=torch.float32))
                                except Exception:
                                    loaded_contexts.append(torch.tensor(np.array(x), dtype=torch.float32))
                            else:
                                # unknown type, try torch.tensor
                                try:
                                    loaded_contexts.append(torch.tensor(x))
                                except Exception:
                                    loaded_contexts.append(None)
                        self._precomputed_contexts = loaded_contexts

                        # Reconstruct target tensors from saved primitives (handle None)
                        loaded_targets = []
                        for tp in cached.get('targets', []):
                            if tp is None:
                                loaded_targets.append(None)
                                continue
                            lt = torch.tensor(tp['lap_time'], dtype=torch.float32)
                            is_pit = torch.tensor(int(tp.get('is_pitlap', 0)), dtype=torch.long)
                            comp = torch.tensor(int(tp.get('compound', 0)), dtype=torch.long)
                            loaded_targets.append({'lap_time': lt, 'is_pitlap': is_pit, 'compound': comp})
                        self._precomputed_targets = loaded_targets
                        self._precomputed_meta = cached['meta']
                        cache_loaded = True
                        logger.info(f"Loaded {len(self._precomputed_contexts)} cached context tensors")
                    else:
                        logger.info("Cache file did not match current dataloader configuration; ignoring cache.")
                except Exception:
                    logger.exception("Failed to load cache; will recompute precomputed tensors.")

            if not cache_loaded:
                # Precompute feature tensors for all lap pairs to avoid heavy
                # pandas->numpy work inside __getitem__ which blocks the GPU.
                # This produces slightly higher memory usage but greatly reduces
                # per-sample CPU overhead during training.
                logger.info(f"Precomputing feature tensors for all {len(self.lap_pairs)} lap pairs...")
                self._precomputed_contexts = [None] * len(self.lap_pairs)
                self._precomputed_targets = [None] * len(self.lap_pairs)
                self._precomputed_meta = [None] * len(self.lap_pairs)

            if not cache_loaded:
                for i in tqdm(range(len(self.lap_pairs)), desc="Precomputing contexts", unit="pairs"):
                    pair = self.lap_pairs[i]
                    context_indices = pair['context_indices']
                    target_index = pair['target_index']

                    context_laps = self.data.loc[context_indices].copy()
                    target_lap = self.data.loc[target_index].copy()

                    # NOTE: precompute without augmentation (augmentation per-sample
                    # would re-introduce CPU work). If augment_prob > 0, augmentation
                    # will be skipped for precomputed path to maximize throughput.
                    if self.normalizer is not None:
                        context_laps = self.normalizer.transform(context_laps)
                        target_lap_norm = self.normalizer.transform(target_lap.to_frame().T)
                        target_lap = target_lap_norm.iloc[0]

                    # Extract features for context
                    context_features = []
                    for _, lap in context_laps.iterrows():
                        feat = self._get_lap_features(lap)
                        context_features.append(feat)

                    if context_features:
                        context_array = np.array(context_features, dtype=np.float32)
                    else:
                        # Empty context
                        num_features = len(self._get_lap_features(context_laps.iloc[0])) if len(context_laps) > 0 else 100
                        context_array = np.zeros((0, num_features), dtype=np.float32)

                    # Pad context to window size
                    if len(context_array) < self.context_window:
                        padding = np.zeros((self.context_window - len(context_array), context_array.shape[1]), dtype=np.float32)
                        context_array = np.vstack([padding, context_array])

                    # Target values
                    target_laptime = float(target_lap['LapTime'])
                    pit_flag = int(target_lap.get('is_pitlap', 0))
                    comp_vals = target_lap[self.compound_columns].values.astype(np.int32)
                    if comp_vals.sum() == 0:
                        compound_idx = len(self.compound_columns) - 1
                    else:
                        compound_idx = int(np.argmax(comp_vals))

                    # Convert to tensors and store
                    context_tensor = torch.from_numpy(context_array)
                    target_tensor = {
                        'lap_time': torch.tensor(target_laptime, dtype=torch.float32),
                        'is_pitlap': torch.tensor(pit_flag, dtype=torch.long),
                        'compound': torch.tensor(compound_idx, dtype=torch.long),
                    }

                    metadata = {
                        'driver': pair['driver'],
                        'year': pair['year'],
                        'circuit': pair['circuit'],
                        'race_name': pair['race_name'],
                        'context_length': len(context_features),
                    }

                    self._precomputed_contexts[i] = context_tensor
                    self._precomputed_targets[i] = target_tensor
                    self._precomputed_meta[i] = metadata

                logger.info(f"Precomputed all {len(self.lap_pairs)} context tensors in RAM for maximum training speed")

                # Persist cache to disk for faster subsequent loads
                try:
                    logger.info(f"Saving precomputed cache to {self.cache_path} ...")
                    # Convert contexts to plain lists to avoid numpy reconstruction issues on load
                    contexts_np = []
                    for c in self._precomputed_contexts:
                        if c is None:
                            contexts_np.append(None)
                        elif isinstance(c, torch.Tensor):
                            contexts_np.append(c.cpu().numpy().tolist())
                        elif isinstance(c, np.ndarray):
                            contexts_np.append(c.tolist())
                        elif isinstance(c, list):
                            contexts_np.append(c)
                        else:
                            try:
                                contexts_np.append(np.array(c).tolist())
                            except Exception:
                                contexts_np.append(None)
                    # targets are small dicts with tensors; convert to primitives (handle None)
                    targets_primitives = []
                    for t in self._precomputed_targets:
                        if t is None:
                            targets_primitives.append(None)
                            continue
                        tp = {}
                        for k, v in t.items():
                            if isinstance(v, torch.Tensor):
                                tp[k] = v.cpu().item() if v.numel() == 1 else v.cpu().numpy()
                            else:
                                tp[k] = v
                        targets_primitives.append(tp)

                    meta_primitives = []
                    for m in self._precomputed_meta:
                        if m is None:
                            meta_primitives.append(None)
                            continue
                        meta_primitives.append({
                            'driver': int(m.get('driver', 0)),
                            'year': int(m.get('year', 0)),
                            'circuit': int(m.get('circuit', 0)),
                            'race_name': str(m.get('race_name', 'Unknown')),
                            'context_length': int(m.get('context_length', 0)),
                        })

                    torch.save({
                        'years': self.years,
                        'context_window': self.context_window,
                        'scaler_type': scaler_type,
                        'numeric_columns': self.numeric_columns,
                        'contexts': contexts_np,
                        'targets': targets_primitives,
                        'meta': meta_primitives,
                    }, self.cache_path)
                    logger.info(f"Cache saved to {self.cache_path}")
                except Exception:
                    logger.exception("Failed to save precomputed cache")
            else:
                logger.info("Using cached precomputed tensors; skipping recomputation.")
    
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
                
                pairs.append({
                    'context_indices': context_indices,
                    'target_index': target_index,
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
        target_index = pair['target_index']

        context_laps = self.data.loc[context_indices].copy()
        target_lap = self.data.loc[target_index].copy()

        # Apply augmentation if enabled
        if np.random.random() < self.augment_prob:
            context_laps, target_lap = self._apply_augmentation(context_laps, target_lap)

        # Normalize numeric features
        if self.normalizer is not None:
            context_laps = self.normalizer.transform(context_laps)
            target_lap_normalized = self.normalizer.transform(target_lap.to_frame().T)
            target_lap = target_lap_normalized.iloc[0]

        # Extract features
        context_features = []
        for _, lap in context_laps.iterrows():
            feat = self._get_lap_features(lap)
            context_features.append(feat)

        if context_features:
            context_array = np.array(context_features, dtype=np.float32)
        else:
            num_features = len(self._get_lap_features(context_laps.iloc[0])) if len(context_laps) > 0 else 100
            context_array = np.zeros((0, num_features), dtype=np.float32)

        # Pad context to window size
        if len(context_array) < self.context_window:
            padding = np.zeros((self.context_window - len(context_array), context_array.shape[1]), dtype=np.float32)
            context_array = np.vstack([padding, context_array])

        target_laptime = float(target_lap['LapTime'])
        pit_flag = int(target_lap.get('is_pitlap', 0))
        comp_vals = target_lap[self.compound_columns].values.astype(np.int32)
        if comp_vals.sum() == 0:
            compound_idx = len(self.compound_columns) - 1
        else:
            compound_idx = int(np.argmax(comp_vals))

        context_tensor = torch.from_numpy(context_array).to(self.device)
        target_tensor = {
            'lap_time': torch.tensor(target_laptime, dtype=torch.float32, device=self.device),
            'is_pitlap': torch.tensor(pit_flag, dtype=torch.long, device=self.device),
            'compound': torch.tensor(compound_idx, dtype=torch.long, device=self.device),
        }

        metadata = {
            'driver': pair['driver'],
            'year': pair['year'],
            'circuit': pair['circuit'],
            'race_name': pair['race_name'],
            'context_length': len(context_features),
        }

        return context_tensor, target_tensor, metadata


# Columns extracted from future target laps and fed to the decoder at each
# rollout step as additional "known future" context.  These are features that
# are determined by race strategy / race schedule and are therefore available
# at inference time even though we haven't predicted them yet.
ROLLOUT_DECODER_FEATURE_COLS: List[str] = [
    # Normalized numeric (strategy-determined)
    'TyreLife',
    'fuel_proxy',
    # Boolean flags (strategy-determined)
    'FreshTyre',
    'is_outlap',
    'is_inlap',
    'is_pitlap',
    # Tire compound one-hot (strategy-determined)
    'compound_soft',
    'compound_medium',
    'compound_hard',
    'compound_unknown',
]


class AutoregressiveRolloutDataset(Dataset):
    """
    Dataset that provides multi-step rollout sequences for training.

    Wraps an existing ``AutoregressiveLapDataloader`` and generates
    (context, future_targets) pairs where the context has ``context_window``
    laps and future_targets contain ``rollout_steps`` consecutive ground-truth
    lap targets for computing loss over an unrolled autoregressive sequence.

    The targets dict now also contains ``future_features``, a tensor of shape
    ``(rollout_steps, n_decoder_features)`` with the known-future features for
    each rollout step (TyreLife, fuel_proxy, compound, pit flags, etc.).  These
    are fed as additional decoder inputs so the model knows the race context
    without having to predict it.

    Parameters
    ----------
    base_dataset : AutoregressiveLapDataloader
        The base dataset with data, normalizer, feature extraction, etc.
    rollout_steps : int
        Number of future autoregressive steps per sample.
    """

    def __init__(self, base_dataset: AutoregressiveLapDataloader, rollout_steps: int = 5):
        self.base = base_dataset
        self.rollout_steps = rollout_steps
        self.context_window = base_dataset.context_window
        self.numeric_columns = base_dataset.numeric_columns
        self.compound_columns = base_dataset.compound_columns
        self.device = base_dataset.device
        self.normalizer = base_dataset.normalizer
        self.years = list(getattr(base_dataset, 'years', []))
        self.data_path = getattr(base_dataset, 'data_path', Path('./data'))
        self.scaler_type = getattr(base_dataset, 'scaler_type', 'standard')
        # Determine which future-feature columns are available in this dataset
        available_cols = set(base_dataset.data.columns) if base_dataset.data is not None else set()
        self.decoder_feature_cols = [c for c in ROLLOUT_DECODER_FEATURE_COLS if c in available_cols]
        self.n_decoder_features = len(self.decoder_feature_cols)
        self.sequences = self._generate_sequences()
        self.cache_path = self._build_cache_path()
        self._precomputed_contexts = [None] * len(self.sequences)
        self._precomputed_targets = [None] * len(self.sequences)
        self._precomputed_meta = [None] * len(self.sequences)
        self._load_or_precompute_cache()
        logger.info(
            f"AutoregressiveRolloutDataset: {len(self.sequences)} sequences "
            f"(context_window={self.context_window}, rollout_steps={rollout_steps}, "
            f"decoder_features={self.n_decoder_features})"
        )

    def _build_cache_path(self) -> Path:
        cache_dir = Path(self.data_path) / 'precomputed'
        cache_dir.mkdir(parents=True, exist_ok=True)
        years_key = "_".join(map(str, sorted(self.years)))
        cache_fname = (
            f"ar_rollout_cache_{years_key}_cw{self.context_window}"
            f"_rw{self.rollout_steps}_{self.scaler_type}.pt"
        )
        return cache_dir / cache_fname

    def _load_cache(self) -> bool:
        if not self.cache_path.exists():
            return False

        try:
            logger.info(f"Loading rollout cache from {self.cache_path}...")

            def _safe_torch_load(path: Path):
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

                try:
                    return torch.load(path, weights_only=True)
                except Exception:
                    pass

                try:
                    if hasattr(torch.serialization, 'safe_globals'):
                        with torch.serialization.safe_globals(safe_globals):
                            return torch.load(path, weights_only=True)
                except Exception:
                    pass

                try:
                    logger.info("Falling back to torch.load(weights_only=False) for trusted local rollout cache")
                    return torch.load(path, weights_only=False)
                except Exception:
                    pass

                import pickle
                with open(path, 'rb') as file_handle:
                    return pickle.load(file_handle)

            cached = _safe_torch_load(self.cache_path)
            valid = (
                cached.get('years') == self.years and
                cached.get('context_window') == self.context_window and
                cached.get('rollout_steps') == self.rollout_steps and
                cached.get('scaler_type') == self.scaler_type and
                cached.get('numeric_columns') == self.numeric_columns and
                cached.get('decoder_feature_cols') == self.decoder_feature_cols
            )
            if not valid:
                logger.info("Rollout cache file did not match current configuration; ignoring cache.")
                return False

            loaded_contexts = []
            for context in cached.get('contexts', []):
                if context is None:
                    loaded_contexts.append(None)
                elif isinstance(context, torch.Tensor):
                    loaded_contexts.append(context)
                elif isinstance(context, np.ndarray):
                    loaded_contexts.append(torch.from_numpy(context))
                else:
                    loaded_contexts.append(torch.tensor(context, dtype=torch.float32))

            loaded_targets = []
            for target in cached.get('targets', []):
                if target is None:
                    loaded_targets.append(None)
                    continue
                t_dict = {
                    'lap_time': torch.tensor(target['lap_time'], dtype=torch.float32),
                    'is_pitlap': torch.tensor(target['is_pitlap'], dtype=torch.long),
                    'compound': torch.tensor(target['compound'], dtype=torch.long),
                }
                if 'future_features' in target and target['future_features'] is not None:
                    t_dict['future_features'] = torch.tensor(target['future_features'], dtype=torch.float32)
                loaded_targets.append(t_dict)

            self._precomputed_contexts = loaded_contexts
            self._precomputed_targets = loaded_targets
            self._precomputed_meta = cached.get('meta', self._precomputed_meta)
            logger.info(f"Loaded {len(self._precomputed_contexts)} cached rollout contexts")
            return True
        except Exception:
            logger.exception("Failed to load rollout cache; will recompute rollout tensors.")
            return False

    def _save_cache(self) -> None:
        try:
            logger.info(f"Saving rollout cache to {self.cache_path} ...")
            contexts_np = []
            for context in self._precomputed_contexts:
                if context is None:
                    contexts_np.append(None)
                elif isinstance(context, torch.Tensor):
                    contexts_np.append(context.cpu().numpy().tolist())
                elif isinstance(context, np.ndarray):
                    contexts_np.append(context.tolist())
                else:
                    contexts_np.append(np.array(context).tolist())

            targets_primitives = []
            for target in self._precomputed_targets:
                if target is None:
                    targets_primitives.append(None)
                    continue
                def _to_list(v):
                    if isinstance(v, torch.Tensor):
                        return v.cpu().numpy().tolist()
                    return v
                t_prim = {
                    'lap_time': _to_list(target['lap_time']),
                    'is_pitlap': _to_list(target['is_pitlap']),
                    'compound': _to_list(target['compound']),
                }
                if 'future_features' in target and target['future_features'] is not None:
                    t_prim['future_features'] = _to_list(target['future_features'])
                targets_primitives.append(t_prim)

            meta_primitives = []
            for meta in self._precomputed_meta:
                if meta is None:
                    meta_primitives.append(None)
                    continue
                meta_primitives.append({
                    'driver': int(meta.get('driver', 0)),
                    'year': int(meta.get('year', 0)),
                    'circuit': int(meta.get('circuit', 0)),
                    'context_length': int(meta.get('context_length', 0)),
                })

            torch.save({
                'years': self.years,
                'context_window': self.context_window,
                'rollout_steps': self.rollout_steps,
                'scaler_type': self.scaler_type,
                'numeric_columns': self.numeric_columns,
                'decoder_feature_cols': self.decoder_feature_cols,
                'contexts': contexts_np,
                'targets': targets_primitives,
                'meta': meta_primitives,
            }, self.cache_path)
            logger.info(f"Rollout cache saved to {self.cache_path}")
        except Exception:
            logger.exception("Failed to save rollout cache")

    def _load_or_precompute_cache(self) -> None:
        if os.environ.get('SKIP_PRECOMPUTE') == '1':
            logger.info("SKIP_PRECOMPUTE=1: skipping rollout precompute cache")
            return

        if self._load_cache():
            return

        logger.info(f"Precomputing rollout tensors for all {len(self.sequences)} sequences...")
        for idx in tqdm(range(len(self.sequences)), desc="Precomputing rollouts", unit="seq"):
            seq = self.sequences[idx]
            context_laps = self.base.data.loc[seq['context_indices']].copy()
            target_laps = self.base.data.loc[seq['target_indices']].copy()

            if self.normalizer is not None:
                context_laps = self.normalizer.transform(context_laps)
                target_laps = self.normalizer.transform(target_laps)

            context_features = []
            for _, lap in context_laps.iterrows():
                context_features.append(self.base._get_lap_features(lap))
            context_array = np.array(context_features, dtype=np.float32)

            if len(context_array) < self.context_window:
                padding = np.zeros(
                    (self.context_window - len(context_array), context_array.shape[1]),
                    dtype=np.float32,
                )
                context_array = np.vstack([padding, context_array])

            target_laptimes = target_laps['LapTime'].values.astype(np.float32)
            if 'is_pitlap' in target_laps.columns:
                target_pitlaps = target_laps['is_pitlap'].values.astype(np.int64)
            else:
                target_pitlaps = np.zeros(len(target_laps), dtype=np.int64)

            compound_indices = []
            for _, lap in target_laps.iterrows():
                comp_vals = lap[self.compound_columns].values.astype(np.int32)
                if comp_vals.sum() == 0:
                    compound_indices.append(len(self.compound_columns) - 1)
                else:
                    compound_indices.append(int(np.argmax(comp_vals)))
            compound_indices = np.array(compound_indices, dtype=np.int64)

            np.nan_to_num(context_array, copy=False, nan=0.0)
            np.nan_to_num(target_laptimes, copy=False, nan=0.0)

            # Extract known-future decoder features from target laps
            future_feat_tensor = None
            if self.decoder_feature_cols:
                avail = [c for c in self.decoder_feature_cols if c in target_laps.columns]
                if avail:
                    future_feat_arr = target_laps[avail].values.astype(np.float32)
                    np.nan_to_num(future_feat_arr, copy=False, nan=0.0)
                    future_feat_tensor = torch.from_numpy(future_feat_arr)

            self._precomputed_contexts[idx] = torch.from_numpy(context_array)
            t_dict = {
                'lap_time': torch.from_numpy(target_laptimes),
                'is_pitlap': torch.from_numpy(target_pitlaps),
                'compound': torch.from_numpy(compound_indices),
            }
            if future_feat_tensor is not None:
                t_dict['future_features'] = future_feat_tensor
            self._precomputed_targets[idx] = t_dict
            self._precomputed_meta[idx] = {
                'driver': seq['driver'],
                'year': seq['year'],
                'circuit': seq['circuit'],
                'context_length': len(context_features),
            }

        logger.info(f"Precomputed all {len(self.sequences)} rollout tensors in RAM")
        self._save_cache()

    def _generate_sequences(self) -> List[Dict]:
        """Generate sliding-window rollout sequences from driver-race groups."""
        sequences = []
        data = self.base.data
        normal_laps = data[data['is_normal_lap'] == 1].copy()
        grouped = normal_laps.groupby(['Driver', 'Year', 'Circuit'])

        total_needed = self.context_window + self.rollout_steps
        for (driver, year, circuit), group_data in grouped:
            group_data = group_data.sort_values('LapNumber')
            if len(group_data) < total_needed:
                continue

            indices = group_data.index.tolist()
            for start in range(len(indices) - total_needed + 1):
                ctx_indices = indices[start:start + self.context_window]
                tgt_indices = indices[start + self.context_window:start + total_needed]
                sequences.append({
                    'context_indices': ctx_indices,
                    'target_indices': tgt_indices,
                    'driver': int(driver),
                    'year': int(year),
                    'circuit': int(circuit),
                })
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        if idx < len(self._precomputed_contexts) and self._precomputed_contexts[idx] is not None:
            context_tensor = self._precomputed_contexts[idx].clone().to(self.device)
            target_dict = {
                key: (value.clone().to(self.device) if isinstance(value, torch.Tensor) else value)
                for key, value in self._precomputed_targets[idx].items()
            }
            metadata = self._precomputed_meta[idx]
            return context_tensor, target_dict, metadata

        seq = self.sequences[idx]

        context_laps = self.base.data.loc[seq['context_indices']].copy()
        target_laps = self.base.data.loc[seq['target_indices']].copy()

        # Normalize
        if self.normalizer is not None:
            context_laps = self.normalizer.transform(context_laps)
            target_laps = self.normalizer.transform(target_laps)

        # Extract context features
        context_features = []
        for _, lap in context_laps.iterrows():
            feat = self.base._get_lap_features(lap)
            context_features.append(feat)
        context_array = np.array(context_features, dtype=np.float32)

        # Pad context if needed
        if len(context_array) < self.context_window:
            padding = np.zeros(
                (self.context_window - len(context_array), context_array.shape[1]),
                dtype=np.float32,
            )
            context_array = np.vstack([padding, context_array])

        # Extract target data for all rollout steps
        target_laptimes = target_laps['LapTime'].values.astype(np.float32)
        if 'is_pitlap' in target_laps.columns:
            target_pitlaps = target_laps['is_pitlap'].values.astype(np.int64)
        else:
            target_pitlaps = np.zeros(len(target_laps), dtype=np.int64)

        compound_indices = []
        for _, lap in target_laps.iterrows():
            comp_vals = lap[self.compound_columns].values.astype(np.int32)
            if comp_vals.sum() == 0:
                compound_indices.append(len(self.compound_columns) - 1)
            else:
                compound_indices.append(int(np.argmax(comp_vals)))
        compound_indices = np.array(compound_indices, dtype=np.int64)

        # Replace NaN
        np.nan_to_num(context_array, copy=False, nan=0.0)
        np.nan_to_num(target_laptimes, copy=False, nan=0.0)

        context_tensor = torch.from_numpy(context_array).to(self.device)
        target_dict = {
            'lap_time': torch.from_numpy(target_laptimes).to(self.device),
            'is_pitlap': torch.from_numpy(target_pitlaps).to(self.device),
            'compound': torch.from_numpy(compound_indices).to(self.device),
        }

        # Extract known-future decoder features
        if self.decoder_feature_cols:
            avail = [c for c in self.decoder_feature_cols if c in target_laps.columns]
            if avail:
                future_feat_arr = target_laps[avail].values.astype(np.float32)
                np.nan_to_num(future_feat_arr, copy=False, nan=0.0)
                target_dict['future_features'] = torch.from_numpy(future_feat_arr).to(self.device)

        metadata = {
            'driver': seq['driver'],
            'year': seq['year'],
            'circuit': seq['circuit'],
            'context_length': len(context_features),
        }

        return context_tensor, target_dict, metadata
