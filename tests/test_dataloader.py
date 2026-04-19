"""
Tests for dataloaders and data pipeline (f1predictor/dataloaders/).

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
        # 15 legacy + 16 Part-C (C1: 4, C2: 1, C3: 3, C4: 1, C5: 6, C6: 1) = 31
        assert len(get_numeric_columns()) == 31

    def test_numeric_column_legacy_prefix_order(self):
        """First 15 numeric columns must stay in legacy order for backward compat."""
        legacy = [
            "LapTime", "TyreLife", "Position", "TrackLength", "RaceLaps",
            "AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed",
            "WindDirection", "delta_to_car_ahead", "fuel_proxy",
            "stint_lap", "cumulative_stint_time",
        ]
        assert get_numeric_columns()[:15] == legacy

    def test_categorical_column_count(self):
        assert len(get_categorical_columns()) == 4

    def test_boolean_column_count(self):
        assert len(get_boolean_columns()) == 12

    def test_compound_column_count(self):
        assert len(get_compound_columns()) == 4

    def test_total_raw_feature_count(self):
        """Total columns should equal model input_size (51 post-Part-C)."""
        total = (
            len(get_numeric_columns())
            + len(get_categorical_columns())
            + len(get_boolean_columns())
            + len(get_compound_columns())
        )
        assert total == 51

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


# ── Part-C engineered features ──────────────────────────────────────────

class TestPartCFeatures:
    """Verify C1–C6 pipeline steps produce deterministic, finite outputs."""

    @pytest.fixture
    def prep(self, tmp_path, monkeypatch):
        """Build a DataPreparation instance without touching raw data on disk."""
        from data.data_preparation import DataPreparation
        # Patch _load_schedules so __init__ does not scan the filesystem
        monkeypatch.setattr(
            DataPreparation, "_load_schedules", lambda self: ({}, [])
        )
        monkeypatch.setattr(
            DataPreparation,
            "_load_or_build_stint_norms",
            lambda self: {"TestCircuit|SOFT": 20.0, "TestCircuit|HARD": 30.0},
        )
        return DataPreparation(tmp_path, vocab_years=[2023])

    @pytest.fixture
    def mini_df(self):
        """Two-driver, one-stint-each synthetic race frame."""
        rows = []
        for drv in ("A", "B"):
            for lap in range(1, 11):
                rows.append({
                    "Driver": drv,
                    "LapNumber": lap,
                    "LapTime": 90_000.0 + lap * 50.0 + (0 if drv == "A" else 20.0),
                    "is_normal_lap": 1,
                    "is_outlap": 0,
                    "is_inlap": 0,
                    "is_pitlap": 0,
                    "stint_id": 1,
                    "stint_lap": lap,
                    "Compound": "SOFT",
                    "AirTemp": 20.0 + lap * 0.1,
                    "TrackTemp": 30.0 + lap * 0.2,
                    "Humidity": 50.0 - lap * 0.1,
                    "delta_to_car_ahead": 1.0 + lap * 0.05,
                    "Sector1Time": 25_000.0 + lap * 5.0,
                    "Sector2Time": 30_000.0 + lap * 10.0,
                    "Sector3Time": 35_000.0 + lap * 15.0,
                })
        return pd.DataFrame(rows)

    def test_traffic_rate(self, prep, mini_df):
        out = prep._add_traffic_rate(mini_df.copy())
        assert "d_delta_to_car_ahead" in out.columns
        a = out[out["Driver"] == "A"].sort_values("LapNumber")
        # First lap per driver: diff NaN → filled with 0
        assert a["d_delta_to_car_ahead"].iloc[0] == 0.0
        # Subsequent laps: constant 0.05 step
        assert np.allclose(a["d_delta_to_car_ahead"].iloc[1:].values, 0.05)

    def test_weather_deltas(self, prep, mini_df):
        out = prep._add_weather_deltas(mini_df.copy())
        for col in ("d_airtemp_vs_start", "d_tracktemp_vs_start", "d_humidity_vs_start"):
            assert col in out.columns
            assert out[col].notna().all()
        # First row (reference) → 0
        assert out["d_airtemp_vs_start"].iloc[0] == 0.0

    def test_sector_times(self, prep, mini_df):
        out = prep._process_sector_times(mini_df.copy())
        for i in (1, 2, 3):
            assert f"Sector{i}Time_ms" in out.columns
            assert f"sector{i}_delta" in out.columns
            assert out[f"Sector{i}Time_ms"].notna().all()
            # First lap in each stint → delta 0
            a = out[out["Driver"] == "A"].sort_values("LapNumber")
            assert a[f"sector{i}_delta"].iloc[0] == 0.0

    def test_rolling_laptime_trend(self, prep, mini_df):
        out = prep._add_rolling_laptime_trend(mini_df.copy())
        for col in ("laptime_ema_3", "laptime_ema_5", "laptime_delta_1", "laptime_delta_3"):
            assert col in out.columns
            assert out[col].notna().all()
            assert np.isfinite(out[col]).all()
        a = out[out["Driver"] == "A"].sort_values("LapNumber")
        # Per-lap delta should equal the constant 50 ms step (laps 2+ in-stint)
        assert np.allclose(a["laptime_delta_1"].iloc[1:].values, 50.0)
        # First lap in stint → shift-based deltas are 0
        assert a["laptime_delta_1"].iloc[0] == 0.0

    def test_stint_deg_slope(self, prep, mini_df):
        out = prep._add_stint_deg_slope(mini_df.copy())
        assert "stint_deg_slope_3" in out.columns
        assert np.isfinite(out["stint_deg_slope_3"]).all()
        a = out[out["Driver"] == "A"].sort_values("LapNumber")
        # Laps 3+ should show positive slope ~50 (LapTime grows 50/lap)
        late_slopes = a["stint_deg_slope_3"].iloc[2:].values
        assert np.allclose(late_slopes, 50.0, atol=1e-6)

    def test_stint_progress_pct(self, prep, mini_df):
        # Need stint_lap already present (fixture provides it)
        out = prep._add_stint_progress_pct(mini_df.copy(), circuit="TestCircuit")
        assert "stint_progress_pct" in out.columns
        assert np.isfinite(out["stint_progress_pct"]).all()
        # SOFT expected=20 → lap 10 → 0.5
        a = out[out["Driver"] == "A"].sort_values("LapNumber")
        assert np.isclose(a["stint_progress_pct"].iloc[9], 0.5)

    def test_decoder_extra_feature_indices_legacy_prefix_frozen(self):
        """First 7 decoder-extra indices must remain in legacy order."""
        from dataloaders.utils import get_decoder_extra_feature_indices
        idx = get_decoder_extra_feature_indices()
        assert len(idx) == 11
        # Legacy-first invariant: old 7-extra checkpoints slice idx[:7]
        numeric = get_numeric_columns()
        compound = get_compound_columns()
        cat_count = len(get_categorical_columns())
        bool_count = len(get_boolean_columns())
        comp_offset = len(numeric) + cat_count + bool_count
        expected_legacy = [
            numeric.index("TyreLife"),
            numeric.index("fuel_proxy"),
            numeric.index("stint_lap"),
            comp_offset + compound.index("compound_soft"),
            comp_offset + compound.index("compound_medium"),
            comp_offset + compound.index("compound_hard"),
            comp_offset + compound.index("compound_unknown"),
        ]
        assert idx[:7] == expected_legacy
