"""
Tests for the Trainer class (f1predictor/models/trainer.py).

Covers batch unpacking, single-batch training, validation metrics,
and gradient accumulation.
"""

import numpy as np
import torch
import torch.nn as nn
import pytest

from tests.conftest import N_NUM, N_CAT, N_BOOL, N_COMP, CATEGORICAL_COLS
from models.trainer import Trainer


def _make_small_trainer(model, device='cpu'):
    """Create a Trainer with minimal settings for testing."""
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    # Temporarily disable torch.compile for test stability
    _orig_compile = getattr(torch, 'compile', None)
    torch.compile = lambda m, **kw: m
    try:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            gradient_clip=1.0,
            accumulation_steps=1,
            use_mixed_precision=False,
            dynamic_aux_balance=False,
            pit_loss_weight=1e-3,
            compound_loss_weight=0.01,
        )
    finally:
        if _orig_compile is not None:
            torch.compile = _orig_compile
    return trainer


def _make_batch(batch_size=4, seq_len=3, format='4tuple'):
    """Create a synthetic batch in the expected format."""
    torch.manual_seed(42)

    encoder_input = {
        'numeric': torch.randn(batch_size, seq_len, N_NUM + N_BOOL + N_COMP),
        'categorical': torch.randint(0, 3, (batch_size, seq_len, N_CAT)),
        'cat_names': CATEGORICAL_COLS,
    }
    decoder_input = torch.randn(batch_size, 1, 1)
    targets = {
        'lap_time': torch.randn(batch_size),
        'pit': torch.zeros(batch_size),
        'compound': torch.randint(0, 4, (batch_size,)),
    }
    metadata = [{'driver': 'VER'}] * batch_size

    if format == '4tuple':
        return encoder_input, decoder_input, targets, metadata
    elif format == '3tuple':
        return encoder_input, targets, metadata


# ── Batch unpacking ─────────────────────────────────────────────────────

class TestUnpackBatch:
    """Verify _unpack_batch handles different batch formats correctly."""

    def test_4tuple_passthrough(self, small_seq2seq):
        trainer = _make_small_trainer(small_seq2seq)
        batch = _make_batch(format='4tuple')
        enc, dec, tgt, meta = trainer._unpack_batch(batch)

        assert isinstance(enc, dict)
        assert isinstance(dec, torch.Tensor)

    def test_3tuple_creates_decoder_input(self, small_seq2seq):
        trainer = _make_small_trainer(small_seq2seq)
        batch = _make_batch(format='3tuple')
        enc, dec, tgt, meta = trainer._unpack_batch(batch)

        # Should create decoder_input from lap_time target
        assert isinstance(dec, torch.Tensor)
        assert dec.dim() == 3  # (batch, seq_len, 1)

    def test_3tuple_scalar_target(self, small_seq2seq):
        """1D scalar targets get expanded to (batch, 1, 1)."""
        trainer = _make_small_trainer(small_seq2seq)
        targets_scalar = {'lap_time': torch.randn(4)}
        batch = (
            {'numeric': torch.randn(4, 3, N_NUM + N_BOOL + N_COMP),
             'categorical': torch.randint(0, 3, (4, 3, N_CAT)),
             'cat_names': CATEGORICAL_COLS},
            targets_scalar,
            [{}] * 4,
        )
        _, dec, _, _ = trainer._unpack_batch(batch)
        assert dec.shape == (4, 1, 1)

    def test_3tuple_multistep_target(self, small_seq2seq):
        """2D multi-step targets (batch, H) get expanded to (batch, H, 1)."""
        trainer = _make_small_trainer(small_seq2seq)
        H = 3
        targets_multi = {'lap_time': torch.randn(4, H)}
        batch = (
            {'numeric': torch.randn(4, 5, N_NUM + N_BOOL + N_COMP),
             'categorical': torch.randint(0, 3, (4, 5, N_CAT)),
             'cat_names': CATEGORICAL_COLS},
            targets_multi,
            [{}] * 4,
        )
        _, dec, _, _ = trainer._unpack_batch(batch)
        assert dec.shape == (4, H, 1)

    def test_invalid_batch_raises(self, small_seq2seq):
        trainer = _make_small_trainer(small_seq2seq)
        with pytest.raises(ValueError, match="Unexpected batch format"):
            trainer._unpack_batch((torch.randn(4, 3),))


# ── Single-batch training ──────────────────────────────────────────────

class TestTrainStep:
    """Verify a single training step produces valid results."""

    def test_single_batch_loss_finite(self, small_seq2seq):
        """One batch through train_epoch produces finite loss."""
        model = small_seq2seq
        model.train()
        trainer = _make_small_trainer(model)
        trainer._show_progress = False

        batch = _make_batch(batch_size=4, seq_len=3)
        # Create a minimal DataLoader-like iterable
        dataset = type('FakeDataset', (), {'__len__': lambda s: 1})()
        loader = type('FakeLoader', (), {
            '__iter__': lambda s: iter([batch]),
            '__len__': lambda s: 1,
            'dataset': dataset,
        })()

        result = trainer.train_epoch(loader, epoch=0)

        assert isinstance(result, dict)
        assert 'loss' in result
        assert np.isfinite(result['loss'])
        assert result['loss'] >= 0


# ── Validation ──────────────────────────────────────────────────────────

class TestValidation:
    """Verify validation returns expected metric keys."""

    def test_validate_returns_metrics(self, small_seq2seq):
        """validate() returns loss and metric dict."""
        model = small_seq2seq
        model.eval()
        trainer = _make_small_trainer(model)
        trainer._show_progress = False

        batch = _make_batch(batch_size=4, seq_len=3)
        dataset = type('FakeDataset', (), {'__len__': lambda s: 1})()
        loader = type('FakeLoader', (), {
            '__iter__': lambda s: iter([batch]),
            '__len__': lambda s: 1,
            'dataset': dataset,
        })()

        val_loss, metrics = trainer.validate(loader)

        assert np.isfinite(val_loss)
        assert isinstance(metrics, dict)
        assert 'mae' in metrics


# ── Scheduled sampling ──────────────────────────────────────────────────

class TestScheduledSampling:
    """Verify scheduled sampling noise injection."""

    def test_zero_prob_no_change(self):
        """prob=0 returns input unchanged."""
        x = torch.randn(2, 5, 10)
        out = Trainer._apply_scheduled_sampling(x, prob=0.0, noise_std=0.1)
        assert torch.allclose(x, out)

    def test_nonzero_prob_modifies_laptime(self):
        """prob=1 with noise always modifies the LapTime column."""
        torch.manual_seed(0)
        x = torch.zeros(4, 10, 5)
        out = Trainer._apply_scheduled_sampling(x, prob=1.0, noise_std=1.0, lap_time_idx=0)
        # Column 0 (LapTime) should have been modified
        assert not torch.allclose(x[:, :, 0], out[:, :, 0])
        # Other columns should be unchanged
        assert torch.allclose(x[:, :, 1:], out[:, :, 1:])
