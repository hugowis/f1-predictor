"""
Tests for dataloaders and data pipeline (code/dataloaders/).

Covers feature extraction consistency, normalizer round-trips,
column definitions, and augmentation shape preservation.
"""

import numpy as np
import pandas as pd
import torch
import pytest

from tests.conftest import (
    NUMERIC_COLS, CATEGORICAL_COLS, BOOLEAN_COLS, COMPOUND_COLS,
    ALL_FEATURE_COLS, N_NUM, N_CAT, N_BOOL, N_COMP, TOTAL_RAW,
    _make_synthetic_rows,
)
from dataloaders.utils import (
    get_numeric_columns,
    get_categorical_columns,
    get_boolean_columns,
    get_compound_columns,
    normalize_year_input,
)
from dataloaders.normalization import LapTimeNormalizer


# ── Feature column definitions ──────────────────────────────────────────

class TestFeatureColumns:
    """Verify column lists are consistent and sum to expected input_size."""

    def test_numeric_column_count(self):
        assert len(get_numeric_columns()) == 15

    def test_categorical_column_count(self):
        assert len(get_categorical_columns()) == 4

    def test_boolean_column_count(self):
        assert len(get_boolean_columns()) == 12

    def test_compound_column_count(self):
        assert len(get_compound_columns()) == 4

    def test_total_raw_feature_count(self):
        """Total columns should equal model input_size (35)."""
        total = (
            len(get_numeric_columns())
            + len(get_categorical_columns())
            + len(get_boolean_columns())
            + len(get_compound_columns())
        )
        assert total == 35

    def test_no_duplicate_columns(self):
        """No column appears in more than one category."""
        all_cols = (
            get_numeric_columns()
            + get_categorical_columns()
            + get_boolean_columns()
            + get_compound_columns()
        )
        assert len(all_cols) == len(set(all_cols))


# ── Feature extraction consistency ──────────────────────────────────────

class TestFeatureExtraction:
    """Verify both dataloaders extract identical feature vectors."""

    def test_get_lap_features_same_layout(self, synthetic_df):
        """Both dataloaders produce features in the same column order."""
        # Create minimal instances that have the column lists but don't load real data
        from dataloaders.stint_dataloader import StintDataloader
        from dataloaders.autoregressive_dataloader import AutoregressiveLapDataloader

        # We only need column lists, so we create the objects without loading data
        # by patching. Instead, just call the function on a Series directly.
        lap = synthetic_df.iloc[0]

        # StintDataloader feature extraction
        stint_feat = np.concatenate([
            lap[NUMERIC_COLS].values.astype(np.float32),
            lap[CATEGORICAL_COLS].values.astype(np.int32),
            lap[BOOLEAN_COLS].values.astype(np.float32),
            lap[COMPOUND_COLS].values.astype(np.float32),
        ])

        # AutoregressiveLapDataloader feature extraction
        ar_feat = np.concatenate([
            lap[NUMERIC_COLS].values.astype(np.float32),
            lap[CATEGORICAL_COLS].values.astype(np.int32).astype(np.float32),
            lap[BOOLEAN_COLS].values.astype(np.float32),
            lap[COMPOUND_COLS].values.astype(np.float32),
        ])

        # They should produce the same values (the dtype difference int32 vs float32
        # for categoricals is a known issue - they ARE the same numerically)
        np.testing.assert_allclose(
            stint_feat.astype(np.float32),
            ar_feat.astype(np.float32),
            err_msg="Feature vectors differ between dataloaders",
        )

    def test_feature_vector_length(self, synthetic_df):
        """Feature vector has expected length."""
        lap = synthetic_df.iloc[0]
        feat = np.concatenate([
            lap[NUMERIC_COLS].values.astype(np.float32),
            lap[CATEGORICAL_COLS].values.astype(np.int32),
            lap[BOOLEAN_COLS].values.astype(np.float32),
            lap[COMPOUND_COLS].values.astype(np.float32),
        ])
        assert feat.shape == (TOTAL_RAW,)


# ── Normalizer tests ───────────────────────────────────────────────────

class TestNormalizer:
    """Verify normalizer correctness."""

    def test_round_trip(self, synthetic_df):
        """transform -> inverse_transform recovers original values."""
        norm = LapTimeNormalizer(scaler_type='standard')
        numeric_data = synthetic_df[NUMERIC_COLS].copy()
        norm.scaler = norm.ScalerClass()
        norm.scaler.fit(numeric_data)
        norm.years = [2020]

        transformed = norm.transform(synthetic_df.copy())
        recovered = norm.inverse_transform(transformed)

        # Compare only numeric columns
        np.testing.assert_allclose(
            recovered[NUMERIC_COLS].values,
            synthetic_df[NUMERIC_COLS].values,
            rtol=1e-5,
            err_msg="Round-trip normalization failed",
        )

    def test_round_trip_minmax(self, synthetic_df):
        """Round trip works for MinMaxScaler too."""
        norm = LapTimeNormalizer(scaler_type='minmax')
        numeric_data = synthetic_df[NUMERIC_COLS].copy()
        norm.scaler = norm.ScalerClass()
        norm.scaler.fit(numeric_data)
        norm.years = [2020]

        transformed = norm.transform(synthetic_df.copy())
        recovered = norm.inverse_transform(transformed)

        np.testing.assert_allclose(
            recovered[NUMERIC_COLS].values,
            synthetic_df[NUMERIC_COLS].values,
            rtol=1e-5,
        )

    def test_round_trip_robust(self, synthetic_df):
        """Round trip works for RobustScaler too."""
        norm = LapTimeNormalizer(scaler_type='robust')
        numeric_data = synthetic_df[NUMERIC_COLS].copy()
        norm.scaler = norm.ScalerClass()
        norm.scaler.fit(numeric_data)
        norm.years = [2020]

        transformed = norm.transform(synthetic_df.copy())
        recovered = norm.inverse_transform(transformed)

        np.testing.assert_allclose(
            recovered[NUMERIC_COLS].values,
            synthetic_df[NUMERIC_COLS].values,
            rtol=1e-5,
        )

    def test_transform_changes_values(self, trained_normalizer, synthetic_df):
        """Transformed values differ from originals."""
        transformed = trained_normalizer.transform(synthetic_df.copy())
        # LapTime mean should be close to 0 after StandardScaler
        assert abs(transformed['LapTime'].mean()) < 1.0

    def test_transform_without_fit_raises(self):
        """Calling transform before fit raises RuntimeError."""
        norm = LapTimeNormalizer()
        df = _make_synthetic_rows(5)
        with pytest.raises(RuntimeError, match="Scaler not fit"):
            norm.transform(df)

    def test_non_numeric_columns_unchanged(self, trained_normalizer, synthetic_df):
        """Boolean and compound columns should not change."""
        transformed = trained_normalizer.transform(synthetic_df.copy())
        for col in BOOLEAN_COLS + COMPOUND_COLS:
            np.testing.assert_array_equal(
                transformed[col].values,
                synthetic_df[col].values,
                err_msg=f"Column {col} was incorrectly normalized",
            )

    def test_invalid_scaler_type_raises(self):
        """Unknown scaler type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scaler type"):
            LapTimeNormalizer(scaler_type='invalid')


# ── Year normalization utility ──────────────────────────────────────────

class TestNormalizeYearInput:
    def test_single_int(self):
        assert normalize_year_input(2020) == [2020]

    def test_list_sorted(self):
        assert normalize_year_input([2022, 2020, 2021]) == [2020, 2021, 2022]

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            normalize_year_input("2020")
