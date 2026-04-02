"""
Tests for the Seq2Seq model (code/models/seq2seq.py).

Covers forward pass shapes, embedding application, teacher forcing,
and decode() consistency.
"""

import torch
import pytest

from tests.conftest import N_NUM, N_CAT, N_BOOL, N_COMP, TOTAL_RAW, CATEGORICAL_COLS


# ── Shape tests ─────────────────────────────────────────────────────────

class TestForwardShapes:
    """Verify output tensor shapes for all input formats."""

    def test_forward_dict_input_shape(self, small_seq2seq):
        """Dict-structured input returns correct output dict and shapes."""
        model = small_seq2seq
        batch_size, seq_len = 4, 5

        encoder_input = {
            'numeric': torch.randn(batch_size, seq_len, N_NUM + N_BOOL + N_COMP),
            'categorical': torch.randint(0, 3, (batch_size, seq_len, N_CAT), dtype=torch.long),
            'cat_names': CATEGORICAL_COLS,
        }
        decoder_input = torch.randn(batch_size, 3, 1)

        out = model(encoder_input, decoder_input, teacher_forcing=True)

        assert isinstance(out, dict)
        assert set(out.keys()) == {'lap', 'pit_logits', 'compound_logits'}
        assert out['lap'].shape == (batch_size, 3, 1)
        assert out['pit_logits'].shape == (batch_size, 3)
        assert out['compound_logits'].shape == (batch_size, 3, model.compound_classes)

    def test_forward_tensor_input_shape(self, small_seq2seq):
        """Raw tensor input (old format) with correct total_dim is handled."""
        model = small_seq2seq
        batch_size, seq_len = 4, 5

        encoder_input = torch.randn(batch_size, seq_len, TOTAL_RAW)
        # Make categorical columns valid int indices
        cat_start, cat_end = N_NUM, N_NUM + N_CAT
        encoder_input[:, :, cat_start:cat_end] = torch.randint(0, 3, (batch_size, seq_len, N_CAT), dtype=torch.long).float()

        decoder_input = torch.randn(batch_size, 2, 1)

        out = model(encoder_input, decoder_input, teacher_forcing=True)

        assert out['lap'].shape == (batch_size, 2, 1)
        assert out['pit_logits'].shape == (batch_size, 2)

    def test_forward_no_teacher_forcing(self, small_seq2seq):
        """Autoregressive decode produces correct output shape."""
        model = small_seq2seq
        batch_size, decoder_len = 4, 6

        encoder_input = {
            'numeric': torch.randn(batch_size, 3, N_NUM + N_BOOL + N_COMP),
            'categorical': torch.randint(0, 3, (batch_size, 3, N_CAT)),
            'cat_names': CATEGORICAL_COLS,
        }
        decoder_input = torch.randn(batch_size, decoder_len, 1)

        out = model(encoder_input, decoder_input, teacher_forcing=False, teacher_forcing_ratio=0.0)

        assert out['lap'].shape == (batch_size, decoder_len, 1)
        assert out['pit_logits'].shape == (batch_size, decoder_len)
        assert out['compound_logits'].shape == (batch_size, decoder_len, model.compound_classes)

    def test_multistep_output_shape(self, small_seq2seq):
        """Decoder with seq_len > 1 works for multi-step predictions."""
        model = small_seq2seq
        batch_size, decoder_len = 4, 5

        encoder_input = {
            'numeric': torch.randn(batch_size, 3, N_NUM + N_BOOL + N_COMP),
            'categorical': torch.randint(0, 3, (batch_size, 3, N_CAT)),
            'cat_names': CATEGORICAL_COLS,
        }
        decoder_input = torch.randn(batch_size, decoder_len, 1)

        out = model(encoder_input, decoder_input, teacher_forcing=True)

        assert out['lap'].shape == (batch_size, decoder_len, 1)


# ── Decode consistency ──────────────────────────────────────────────────

class TestDecodeConsistency:
    """Verify decode() produces same structure as forward(teacher_forcing=False)."""

    def test_decode_matches_forward_shape(self, small_seq2seq):
        """decode() output has same shape as forward(tf=False)."""
        model = small_seq2seq
        batch_size, enc_len, dec_len = 4, 5, 3

        encoder_input = {
            'numeric': torch.randn(batch_size, enc_len, N_NUM + N_BOOL + N_COMP),
            'categorical': torch.randint(0, 3, (batch_size, enc_len, N_CAT)),
            'cat_names': CATEGORICAL_COLS,
        }
        decoder_seed = torch.randn(batch_size, dec_len, 1)

        # Forward with no teacher forcing
        fwd_out = model(encoder_input, decoder_seed, teacher_forcing=False, teacher_forcing_ratio=0.0)

        # Use encode + decode directly
        # First, apply embeddings manually to get a plain tensor for encode()
        with torch.no_grad():
            # Run forward once to get the encoder hidden state shape
            # Instead, use encode on the processed input
            enc_in = model._apply_embeddings(encoder_input) if hasattr(model, '_apply_embeddings') else None

        # If _apply_embeddings doesn't exist yet (pre-refactor), skip this test
        if enc_in is None:
            # Fallback: just verify decode() with a dummy hidden state
            dummy_hidden = torch.zeros(model.num_layers, batch_size, model.hidden_size)
            dec_out = model.decode(decoder_seed, dummy_hidden, max_length=dec_len)
        else:
            enc_output, enc_hidden = model.encode(enc_in)
            base_hidden = model.fc_encoder_to_hidden(enc_output[:, -1, :]).unsqueeze(0)
            expanded = base_hidden.expand(model.num_layers, batch_size, model.hidden_size).contiguous()
            dec_out = model.decode(decoder_seed, expanded, max_length=dec_len)

        assert dec_out['lap'].shape == fwd_out['lap'].shape
        assert dec_out['pit_logits'].shape == fwd_out['pit_logits'].shape
        assert dec_out['compound_logits'].shape == fwd_out['compound_logits'].shape

    def test_decode_returns_dict(self, small_seq2seq):
        """decode() returns a dict with the same keys as forward()."""
        model = small_seq2seq
        batch_size = 2

        hidden = torch.zeros(model.num_layers, batch_size, model.hidden_size)
        dec_input = torch.randn(batch_size, 1, 1)

        out = model.decode(dec_input, hidden, max_length=3)

        assert isinstance(out, dict)
        assert set(out.keys()) == {'lap', 'pit_logits', 'compound_logits'}


# ── Embedding tests ─────────────────────────────────────────────────────

class TestEmbeddings:
    """Verify embedding application changes the effective input dimension."""

    def test_embeddings_affect_output(self, small_seq2seq):
        """Output changes when categorical ids differ (embeddings are used)."""
        model = small_seq2seq
        model.eval()
        batch_size, seq_len = 2, 3

        numeric = torch.randn(batch_size, seq_len, N_NUM + N_BOOL + N_COMP)
        dec = torch.randn(batch_size, 1, 1)

        # Two different categorical inputs
        cat_a = torch.zeros(batch_size, seq_len, N_CAT, dtype=torch.long)
        cat_b = torch.ones(batch_size, seq_len, N_CAT, dtype=torch.long)

        inp_a = {'numeric': numeric.clone(), 'categorical': cat_a, 'cat_names': CATEGORICAL_COLS}
        inp_b = {'numeric': numeric.clone(), 'categorical': cat_b, 'cat_names': CATEGORICAL_COLS}

        with torch.no_grad():
            out_a = model(inp_a, dec.clone(), teacher_forcing=True)
            out_b = model(inp_b, dec.clone(), teacher_forcing=True)

        # Outputs should differ because embeddings differ
        assert not torch.allclose(out_a['lap'], out_b['lap'], atol=1e-6)


# ── Determinism ─────────────────────────────────────────────────────────

class TestDeterminism:
    """Verify reproducibility with fixed seeds."""

    def test_forward_deterministic_with_seed(self, small_seq2seq):
        """Same input + eval mode = same output on repeated calls."""
        model = small_seq2seq
        model.eval()
        batch_size, seq_len = 2, 4

        enc = {
            'numeric': torch.randn(batch_size, seq_len, N_NUM + N_BOOL + N_COMP),
            'categorical': torch.randint(0, 3, (batch_size, seq_len, N_CAT), dtype=torch.long),
            'cat_names': CATEGORICAL_COLS,
        }
        dec = torch.randn(batch_size, 2, 1)

        with torch.no_grad():
            out1 = model(enc, dec, teacher_forcing=True)
            out2 = model(enc, dec, teacher_forcing=True)

        assert torch.allclose(out1['lap'], out2['lap'], atol=1e-7)
        assert torch.allclose(out1['pit_logits'], out2['pit_logits'], atol=1e-7)

    def test_output_is_finite(self, small_seq2seq):
        """Forward pass produces finite values (no NaN/Inf)."""
        model = small_seq2seq
        model.eval()
        batch_size, seq_len = 2, 4

        enc = {
            'numeric': torch.randn(batch_size, seq_len, N_NUM + N_BOOL + N_COMP),
            'categorical': torch.randint(0, 3, (batch_size, seq_len, N_CAT), dtype=torch.long),
            'cat_names': CATEGORICAL_COLS,
        }
        dec = torch.randn(batch_size, 2, 1)

        with torch.no_grad():
            out = model(enc, dec, teacher_forcing=True)

        assert torch.isfinite(out['lap']).all()
        assert torch.isfinite(out['pit_logits']).all()
        assert torch.isfinite(out['compound_logits']).all()
