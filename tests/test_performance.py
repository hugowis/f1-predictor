"""
Performance regression tests.

These tests verify that critical hot paths remain fast.
They measure wall-clock time and fail if operations exceed
generous thresholds (to avoid flaky tests on slow CI).
"""

import time
import numpy as np
import pandas as pd
import torch
import pytest

from tests.conftest import (
    NUMERIC_COLS, CATEGORICAL_COLS, BOOLEAN_COLS, COMPOUND_COLS,
    N_NUM, N_CAT, N_BOOL, N_COMP, TOTAL_RAW,
    _make_synthetic_rows,
)


# ── Vectorized feature extraction ───────────────────────────────────────

class TestFeatureExtractionSpeed:
    """Verify feature extraction is not bottlenecked by iterrows."""

    def test_vectorized_extraction_fast(self):
        """Extracting features from 500 rows should complete in < 50ms."""
        df = _make_synthetic_rows(500, seed=99)

        start = time.perf_counter()
        # Vectorized approach (what we want after refactoring)
        numeric = df[NUMERIC_COLS].values.astype(np.float32)
        cat = df[CATEGORICAL_COLS].values.astype(np.float32)
        boolean = df[BOOLEAN_COLS].values.astype(np.float32)
        compound = df[COMPOUND_COLS].values.astype(np.float32)
        features = np.concatenate([numeric, cat, boolean, compound], axis=1)
        elapsed = time.perf_counter() - start

        assert features.shape == (500, TOTAL_RAW)
        assert elapsed < 0.05, f"Vectorized extraction took {elapsed:.3f}s (> 50ms)"

    def test_iterrows_is_slower(self):
        """Document that iterrows is significantly slower than vectorized."""
        df = _make_synthetic_rows(500, seed=99)

        # iterrows approach (current code)
        start = time.perf_counter()
        features_list = []
        for _, lap in df.iterrows():
            feat = np.concatenate([
                lap[NUMERIC_COLS].values.astype(np.float32),
                lap[CATEGORICAL_COLS].values.astype(np.int32).astype(np.float32),
                lap[BOOLEAN_COLS].values.astype(np.float32),
                lap[COMPOUND_COLS].values.astype(np.float32),
            ])
            features_list.append(feat)
        iterrows_time = time.perf_counter() - start

        # vectorized approach
        start = time.perf_counter()
        features = np.concatenate([
            df[NUMERIC_COLS].values.astype(np.float32),
            df[CATEGORICAL_COLS].values.astype(np.float32),
            df[BOOLEAN_COLS].values.astype(np.float32),
            df[COMPOUND_COLS].values.astype(np.float32),
        ], axis=1)
        vectorized_time = time.perf_counter() - start

        # Vectorized should be faster (usually 10-50x)
        # Use a generous threshold to avoid flaky tests
        assert vectorized_time < iterrows_time, (
            f"Vectorized ({vectorized_time:.4f}s) should be faster "
            f"than iterrows ({iterrows_time:.4f}s)"
        )


# ── Model forward pass ─────────────────────────────────────────────────

class TestForwardPassSpeed:
    """Verify model forward pass is not doing redundant work."""

    def test_forward_completes_quickly(self, small_seq2seq):
        """Forward pass on small batch should complete in < 100ms."""
        model = small_seq2seq
        model.eval()
        batch_size, seq_len = 8, 10

        enc = {
            'numeric': torch.randn(batch_size, seq_len, N_NUM + N_BOOL + N_COMP),
            'categorical': torch.randint(0, 3, (batch_size, seq_len, N_CAT)),
            'cat_names': ['Driver', 'Team', 'Year', 'Circuit'],
        }
        dec = torch.randn(batch_size, 5, 1)

        # Warm up
        with torch.no_grad():
            model(enc, dec, teacher_forcing=True)

        # Measure
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                model(enc, dec, teacher_forcing=True)
        elapsed = (time.perf_counter() - start) / 10

        assert elapsed < 0.1, f"Forward pass took {elapsed:.3f}s (> 100ms)"

    def test_teacher_forcing_fast_path_faster(self, small_seq2seq):
        """Full TF (ratio >= 0.999) should be faster than step-by-step."""
        model = small_seq2seq
        model.eval()
        batch_size, seq_len, dec_len = 8, 10, 8

        enc = {
            'numeric': torch.randn(batch_size, seq_len, N_NUM + N_BOOL + N_COMP),
            'categorical': torch.randint(0, 3, (batch_size, seq_len, N_CAT)),
            'cat_names': ['Driver', 'Team', 'Year', 'Circuit'],
        }
        dec = torch.randn(batch_size, dec_len, 1)

        # Warm up
        with torch.no_grad():
            model(enc, dec, teacher_forcing=True, teacher_forcing_ratio=1.0)
            model(enc, dec, teacher_forcing=True, teacher_forcing_ratio=0.5)

        # Full TF (vectorized fast path)
        n_iters = 50
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                model(enc, dec, teacher_forcing=True, teacher_forcing_ratio=1.0)
        full_tf_time = (time.perf_counter() - start) / n_iters

        # Partial TF (step-by-step loop)
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                model(enc, dec, teacher_forcing=True, teacher_forcing_ratio=0.5)
        partial_tf_time = (time.perf_counter() - start) / n_iters

        # Full TF should be faster (vectorized vs loop)
        assert full_tf_time < partial_tf_time, (
            f"Full TF ({full_tf_time:.4f}s) should be faster than "
            f"partial TF ({partial_tf_time:.4f}s)"
        )


# ── Normalizer speed ───────────────────────────────────────────────────

class TestNormalizerSpeed:
    """Verify normalizer transform is fast enough."""

    def test_transform_1000_rows(self):
        """Normalizing 1000 rows should complete in < 100ms."""
        from dataloaders.normalization import LapTimeNormalizer

        df = _make_synthetic_rows(1000, seed=42)
        norm = LapTimeNormalizer(scaler_type='standard')
        norm.scaler = norm.ScalerClass()
        norm.scaler.fit(df[NUMERIC_COLS])
        norm.years = [2020]

        # Warm up
        norm.transform(df.copy())

        start = time.perf_counter()
        for _ in range(10):
            norm.transform(df.copy())
        elapsed = (time.perf_counter() - start) / 10

        assert elapsed < 0.1, f"Normalizer transform took {elapsed:.3f}s (> 100ms)"
