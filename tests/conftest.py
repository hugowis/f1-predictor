"""
Shared fixtures for F1 predictor test suite.

Provides minimal configs, synthetic data, and small models
for fast, deterministic testing.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pytest

# Ensure f1predictor library is on the path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent / "src" / "f1predictor"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.base import Config, ModelConfig, TrainingConfig, DataConfig
from dataloaders.utils import (
    get_numeric_columns,
    get_categorical_columns,
    get_boolean_columns,
    get_compound_columns,
)
from dataloaders.normalization import LapTimeNormalizer


# ── Column definitions (computed once) ──────────────────────────────────
NUMERIC_COLS = get_numeric_columns()
CATEGORICAL_COLS = get_categorical_columns()
BOOLEAN_COLS = get_boolean_columns()
COMPOUND_COLS = get_compound_columns()
ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS + BOOLEAN_COLS + COMPOUND_COLS

# Expected total raw feature count (before embeddings)
N_NUM = len(NUMERIC_COLS)       # 31 post-Part-C (was 15)
N_CAT = len(CATEGORICAL_COLS)   # 4
N_BOOL = len(BOOLEAN_COLS)      # 12
N_COMP = len(COMPOUND_COLS)     # 4
TOTAL_RAW = N_NUM + N_CAT + N_BOOL + N_COMP  # 51


@pytest.fixture
def minimal_model_config():
    """Small ModelConfig for fast unit tests."""
    return ModelConfig(
        name="seq2seq",
        input_size=TOTAL_RAW,
        output_size=1,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        encoder='gru',
        embedding_dims={'Driver': 4, 'Team': 4, 'Circuit': 4, 'Year': 4},
        vocab_sizes={'Driver': 10, 'Team': 5, 'Circuit': 8, 'Year': 4},
    )


@pytest.fixture
def minimal_config(minimal_model_config):
    """Full Config with small dimensions for fast tests."""
    return Config(
        model=minimal_model_config,
        training=TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            num_epochs=2,
            weight_decay=0.0,
            gradient_clip=1.0,
            accumulation_steps=1,
            use_mixed_precision=False,
            warm_up_epochs=0,
            teacher_forcing_start=1.0,
            teacher_forcing_end=1.0,
            early_stopping_patience=5,
            train_years=[2020, 2021],
            val_years=[2022],
            test_years=[2023],
        ),
        data=DataConfig(
            window_size=5,
            context_window=3,
            augment_prob=0.0,
            normalize=False,
            num_workers=0,
        ),
        device='cpu',
        seed=42,
    )


@pytest.fixture
def small_seq2seq(minimal_model_config):
    """Small Seq2Seq model on CPU for fast tests."""
    from models.seq2seq import Seq2Seq

    torch.manual_seed(42)
    model = Seq2Seq(
        input_size=minimal_model_config.input_size,
        output_size=minimal_model_config.output_size,
        hidden_size=minimal_model_config.hidden_size,
        num_layers=minimal_model_config.num_layers,
        dropout=minimal_model_config.dropout,
        decoder_type=minimal_model_config.encoder,
        embedding_dims=minimal_model_config.embedding_dims,
        vocab_sizes=minimal_model_config.vocab_sizes,
        device='cpu',
        decoder_extra_features_size=0,
    )
    model.eval()
    return model


def _make_synthetic_rows(n_rows: int = 30, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic DataFrame that mirrors real clean race data."""
    rng = np.random.RandomState(seed)

    data = {}
    # Numeric columns
    data['LapTime'] = rng.uniform(80000, 100000, n_rows)  # ms
    data['TyreLife'] = rng.randint(1, 30, n_rows).astype(float)
    data['Position'] = rng.randint(1, 21, n_rows).astype(float)
    data['TrackLength'] = np.full(n_rows, 5412.0)
    data['RaceLaps'] = np.full(n_rows, 57.0)
    data['AirTemp'] = rng.uniform(15, 35, n_rows)
    data['TrackTemp'] = rng.uniform(20, 50, n_rows)
    data['Humidity'] = rng.uniform(30, 90, n_rows)
    data['Pressure'] = rng.uniform(990, 1030, n_rows)
    data['WindSpeed'] = rng.uniform(0, 10, n_rows)
    data['WindDirection'] = rng.uniform(0, 360, n_rows)
    data['delta_to_car_ahead'] = rng.uniform(0, 5000, n_rows)
    data['fuel_proxy'] = rng.uniform(0.2, 1.0, n_rows)
    data['stint_lap'] = rng.randint(1, 20, n_rows).astype(float)
    data['cumulative_stint_time'] = rng.uniform(80000, 1500000, n_rows)

    # Part-C engineered features (fill with reasonable ranges so tests run)
    # C1: rolling lap-time trend
    data['laptime_ema_3'] = rng.uniform(80000, 100000, n_rows)
    data['laptime_ema_5'] = rng.uniform(80000, 100000, n_rows)
    data['laptime_delta_1'] = rng.uniform(-500, 500, n_rows)
    data['laptime_delta_3'] = rng.uniform(-1000, 1000, n_rows)
    # C2: tyre-deg slope
    data['stint_deg_slope_3'] = rng.uniform(-50, 200, n_rows)
    # C3: weather deltas vs race start
    data['d_airtemp_vs_start'] = rng.uniform(-5, 5, n_rows)
    data['d_tracktemp_vs_start'] = rng.uniform(-10, 10, n_rows)
    data['d_humidity_vs_start'] = rng.uniform(-20, 20, n_rows)
    # C4: traffic rate
    data['d_delta_to_car_ahead'] = rng.uniform(-1000, 1000, n_rows)
    # C5: sector times + deltas (ms)
    data['Sector1Time_ms'] = rng.uniform(20000, 35000, n_rows)
    data['Sector2Time_ms'] = rng.uniform(25000, 40000, n_rows)
    data['Sector3Time_ms'] = rng.uniform(25000, 40000, n_rows)
    data['sector1_delta'] = rng.uniform(-500, 500, n_rows)
    data['sector2_delta'] = rng.uniform(-500, 500, n_rows)
    data['sector3_delta'] = rng.uniform(-500, 500, n_rows)
    # C6: stint progress %
    data['stint_progress_pct'] = rng.uniform(0, 1, n_rows)

    # Categorical columns (small integer ids)
    data['Driver'] = rng.randint(0, 8, n_rows)
    data['Team'] = rng.randint(0, 4, n_rows)
    data['Year'] = rng.randint(0, 3, n_rows)
    data['Circuit'] = rng.randint(0, 6, n_rows)

    # Boolean columns
    for col in BOOLEAN_COLS:
        data[col] = rng.choice([0.0, 1.0], n_rows)

    # Compound one-hot (exactly one active per row)
    compounds = np.zeros((n_rows, len(COMPOUND_COLS)), dtype=float)
    chosen = rng.randint(0, len(COMPOUND_COLS), n_rows)
    compounds[np.arange(n_rows), chosen] = 1.0
    for i, col in enumerate(COMPOUND_COLS):
        data[col] = compounds[:, i]

    # Extra metadata columns (not features, but present in real data)
    data['LapNumber'] = np.arange(1, n_rows + 1, dtype=float)

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_df():
    """30-row DataFrame mimicking clean race data."""
    return _make_synthetic_rows(30, seed=42)


@pytest.fixture
def synthetic_batch_dict():
    """Dict-structured batch as produced by AutoregressiveLapDataloader."""
    torch.manual_seed(42)
    batch_size, seq_len = 4, 5

    numeric = torch.randn(batch_size, seq_len, N_NUM + N_BOOL + N_COMP)
    categorical = torch.randint(0, 3, (batch_size, seq_len, N_CAT))  # max 3 to stay within all vocab sizes

    encoder_input = {
        'numeric': numeric,
        'categorical': categorical,
        'cat_names': CATEGORICAL_COLS,
    }
    decoder_input = torch.randn(batch_size, 1, 1)
    targets = {'lap_time': torch.randn(batch_size)}
    metadata = [{'driver': 'VER', 'race_name': 'Monza'}] * batch_size

    return encoder_input, decoder_input, targets, metadata


@pytest.fixture
def synthetic_batch_tensor():
    """Raw-tensor batch (old dataloader format) with all columns concatenated."""
    torch.manual_seed(42)
    batch_size, seq_len = 4, 5

    encoder_input = torch.randn(batch_size, seq_len, TOTAL_RAW)
    # Ensure categorical columns are valid int indices (max 3 to stay within vocab sizes)
    cat_start = N_NUM
    cat_end = cat_start + N_CAT
    encoder_input[:, :, cat_start:cat_end] = torch.randint(0, 3, (batch_size, seq_len, N_CAT)).float()

    decoder_input = torch.randn(batch_size, 1, 1)
    targets = {'lap_time': torch.randn(batch_size)}
    metadata = [{'driver': 'HAM', 'race_name': 'Spa'}] * batch_size

    return encoder_input, decoder_input, targets, metadata


@pytest.fixture
def trained_normalizer(synthetic_df):
    """LapTimeNormalizer fit on synthetic data (no disk I/O)."""
    norm = LapTimeNormalizer(scaler_type='standard')
    numeric_data = synthetic_df[NUMERIC_COLS].copy()
    norm.scaler = norm.ScalerClass()
    norm.scaler.fit(numeric_data)
    norm.years = [2020, 2021]
    return norm
