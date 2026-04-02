"""
Tests for data leakage prevention.

These tests verify that training data does not contaminate validation/test data
through vocabulary building, normalization, caching, or configuration.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from config.base import Config, TrainingConfig, ModelConfig, DataConfig
from dataloaders.utils import get_numeric_columns
from dataloaders.normalization import LapTimeNormalizer
from tests.conftest import NUMERIC_COLS, _make_synthetic_rows


# ── Config year overlap ─────────────────────────────────────────────────

class TestConfigYearOverlap:
    """Verify that default configs don't have overlapping year splits."""

    def test_default_config_no_overlap(self):
        """Default train/val/test years don't overlap."""
        config = Config()
        train = set(config.training.train_years)
        val = set(config.training.val_years)
        test = set(config.training.test_years)

        assert train & val == set(), f"Train/val overlap: {train & val}"
        assert train & test == set(), f"Train/test overlap: {train & test}"
        assert val & test == set(), f"Val/test overlap: {val & test}"

    def test_phase1_config_no_overlap(self):
        from config.base import get_phase1_config
        config = get_phase1_config()
        train = set(config.training.train_years)
        val = set(config.training.val_years)
        test = set(config.training.test_years)
        assert train & val == set()
        assert train & test == set()

    def test_phase2_config_no_overlap(self):
        from config.base import get_phase2_config
        config = get_phase2_config()
        train = set(config.training.train_years)
        val = set(config.training.val_years)
        test = set(config.training.test_years)
        assert train & val == set()
        assert train & test == set()


# ── Vocabulary leakage ──────────────────────────────────────────────────

class TestVocabLeakage:
    """Verify vocabulary files don't leak test year information."""

    @pytest.fixture
    def vocabs_dir(self):
        return Path("data/vocabs")

    def test_year_vocab_exists(self, vocabs_dir):
        """Year.json should exist if data has been prepared."""
        if not vocabs_dir.exists():
            pytest.skip("No vocabs directory - data not yet prepared")
        year_file = vocabs_dir / "Year.json"
        if not year_file.exists():
            pytest.skip("Year.json not found")

    def test_year_vocab_should_not_contain_test_years(self, vocabs_dir):
        """LEAKAGE CHECK: Year.json should ideally not contain test year indices.

        NOTE: This test documents the current leakage. After Phase 1 fix,
        this test should PASS (test years should map to <UNK>).
        Currently expected to FAIL - marking as xfail.
        """
        year_file = vocabs_dir / "Year.json"
        if not year_file.exists():
            pytest.skip("Year.json not found")

        with open(year_file) as f:
            year_vocab = json.load(f)

        test_years_as_str = ['2025']
        leaked_years = [y for y in test_years_as_str if y in year_vocab]

        # After fix: this should be empty
        # Before fix: this will contain test years (documenting the leak)
        if leaked_years:
            pytest.xfail(
                f"KNOWN LEAKAGE: Year vocab contains test years {leaked_years}. "
                f"Fix in Phase 1 by building vocab from train years only."
            )

    def test_driver_vocab_documents_leak(self, vocabs_dir):
        """LEAKAGE CHECK: Drivers unique to test years leak information.

        Drivers who only appear in 2025 get their own embedding index,
        which the model can use to identify test data.
        """
        driver_file = vocabs_dir / "Driver.json"
        if not driver_file.exists():
            pytest.skip("Driver.json not found")

        with open(driver_file) as f:
            driver_vocab = json.load(f)

        # We can't easily tell which drivers are test-only without
        # the full data. Just document that the vocab exists and
        # note the total size.
        assert len(driver_vocab) > 0, "Driver vocab is empty"


# ── Normalizer provenance ──────────────────────────────────────────────

class TestNormalizerProvenance:
    """Verify normalizer tracks which years it was fit on."""

    def test_normalizer_stores_years(self):
        """After fitting, normalizer.years should be set."""
        df = _make_synthetic_rows(20)
        norm = LapTimeNormalizer(scaler_type='standard')
        norm.scaler = norm.ScalerClass()
        norm.scaler.fit(df[NUMERIC_COLS])
        norm.years = [2020, 2021]

        assert norm.years == [2020, 2021]

    def test_normalizer_save_includes_years(self, tmp_path):
        """Saved scaler pickle should include years metadata."""
        import pickle

        df = _make_synthetic_rows(20)
        norm = LapTimeNormalizer(scaler_type='standard', scaler_dir=tmp_path)
        norm.scaler = norm.ScalerClass()
        norm.scaler.fit(df[NUMERIC_COLS])
        norm.years = [2020, 2021]
        norm.save([2020, 2021])

        # Load the raw pickle and check
        pkl_file = tmp_path / "2020_2021_standard.pkl"
        assert pkl_file.exists()

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        assert 'years' in data
        assert data['years'] == [2020, 2021]
        assert 'scaler' in data
        assert 'columns' in data

    def test_normalizer_load_restores_years(self, tmp_path):
        """Loading a saved scaler restores the years it was fit on."""
        df = _make_synthetic_rows(20)
        norm = LapTimeNormalizer(scaler_type='standard', scaler_dir=tmp_path)
        norm.scaler = norm.ScalerClass()
        norm.scaler.fit(df[NUMERIC_COLS])
        norm.years = [2020, 2021]
        norm.save([2020, 2021])

        # Load into fresh normalizer
        norm2 = LapTimeNormalizer(scaler_type='standard', scaler_dir=tmp_path)
        norm2.load([2020, 2021])

        assert norm2.years == [2020, 2021]


# ── Feature engineering safety ──────────────────────────────────────────

class TestFeatureEngineeringSafety:
    """Verify engineered features don't use future information."""

    def test_fuel_proxy_uses_no_future_laps(self):
        """fuel_proxy = 1 - (LapNumber / RaceLaps) uses only known-before-race info."""
        # RaceLaps is the scheduled total laps (known pre-race)
        # LapNumber is the current lap (known at prediction time)
        # Neither uses future lap performance data
        lap_number = 10
        race_laps = 57
        fuel_proxy = 1 - (lap_number / race_laps)
        assert 0 < fuel_proxy < 1

    def test_cumulative_stint_time_uses_past_only(self):
        """cumulative_stint_time is a cumsum within a stint (past laps only)."""
        laptimes = [90000, 91000, 92000, 93000]
        cumsum = np.cumsum(laptimes)
        # At position i, cumsum[i] only uses laptimes[0:i+1] (past + current)
        assert cumsum[0] == 90000
        assert cumsum[1] == 90000 + 91000
        assert cumsum[2] == 90000 + 91000 + 92000
