"""
On-demand model loading and autoregressive rollout for the web dashboard.

All heavy work (model load, data load, rollout) is wrapped in Streamlit
cache decorators so it only runs once per unique (checkpoint, test_years)
combination and is reused on subsequent page interactions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Make sure the f1predictor library and scripts are importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent.parent  # f1-predictor/
_SRC = _REPO_ROOT / "src"
_CODE_DIR = _SRC / "f1predictor"
_SCRIPTS_DIR = _SRC / "scripts"
for _p in (str(_CODE_DIR), str(_SCRIPTS_DIR), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Model + normalizer loading (cached for the lifetime of the server process)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model checkpoint…")
def load_model_and_normalizer(checkpoint_path: str, config_path: str, device: str):
    """
    Load a Seq2Seq model and its associated normalizer.

    Cached by (checkpoint_path, config_path, device) — re-runs only when
    a different checkpoint or config is requested.

    Returns
    -------
    tuple of (model, normalizer, config)
        *model*      — loaded Seq2Seq in eval mode on *device*
        *normalizer* — fitted LapTimeNormalizer (may be None)
        *config*     — Config object (may be None)
    """
    from evaluate import load_model_from_checkpoint, _load_eval_runtime_config
    from config.base import Config

    ckpt = Path(checkpoint_path)
    cfg_path = Path(config_path) if config_path else None

    model, _ = load_model_from_checkpoint(ckpt, device=device)

    runtime = _load_eval_runtime_config(cfg_path)
    normalizer = runtime.get("normalizer")

    config = None
    if cfg_path and cfg_path.exists():
        config = Config.load(cfg_path)

    return model, normalizer, config


# ---------------------------------------------------------------------------
# Full rollout inference (cached per checkpoint + test years)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running autoregressive rollout (this may take ~30 s)…")
def run_rollout_sequences(
    checkpoint_path: str,
    config_path: str,
    test_years: tuple[int, ...],
    device: str,
    data_root: str,
) -> list[dict]:
    """
    Run the full autoregressive rollout for *test_years* and return
    per-sequence lap predictions (denormalized to ms).

    Cached by (checkpoint_path, test_years, device) — selecting a different
    seed or model re-runs; same seed reuses the cache instantly.

    Returns
    -------
    list of dict, one per driver-circuit-year sequence::

        {
            "driver_name":        str,
            "circuit_name":       str,
            "year":               str,
            "context_actual_ms":  list[float],   # context-window ground-truth laps
            "predicted_ms":       list[float],   # autoregressive predictions
            "actual_ms":          list[float],   # ground-truth for predicted steps
            "lap_numbers":        list[int],     # 1-based lap indices (context + predicted)
        }
    """
    from dataloaders.autoregressive_dataloader import AutoregressiveLapDataloader
    from models.rollout_evaluator import (
        evaluate_autoregressive_rollout,
        denormalize_rollout_metrics,
    )
    from dashboard.utils.data_loader import load_vocabs

    model, normalizer, config = load_model_and_normalizer(
        checkpoint_path, config_path, device
    )

    context_window = config.data.context_window if config else 10
    normalize = config.data.normalize if config else True
    scaler_type = config.data.scaler_type if config else "standard"

    # Load test dataset (skip precompute — we only need the raw lap data)
    old_skip = os.environ.get("SKIP_PRECOMPUTE")
    os.environ["SKIP_PRECOMPUTE"] = "1"
    try:
        test_ds = AutoregressiveLapDataloader(
            year=list(test_years),
            context_window=context_window,
            normalize=normalize,
            scaler_type=scaler_type,
            normalizer=normalizer,
            augment_prob=0.0,
        )
    finally:
        if old_skip is None:
            os.environ.pop("SKIP_PRECOMPUTE", None)
        else:
            os.environ["SKIP_PRECOMPUTE"] = old_skip

    # Run rollout — request per-sequence predictions
    _, sequence_rollouts = evaluate_autoregressive_rollout(
        model=model,
        test_dataset=test_ds,
        device=device,
        return_sequences=True,
    )

    # Determine denormalization parameters (offset + scale for full inverse transform)
    offset, scale = _get_laptime_denorm_params(normalizer)

    # Load vocabs for human-readable names
    vocabs = load_vocabs(Path(data_root))
    driver_map = vocabs.get("Driver", {})
    circuit_map = vocabs.get("Circuit", {})
    year_map = vocabs.get("Year", {})

    results = []
    for seq in sequence_rollouts:
        ctx_norm = seq.get("context_actual_norm", [])
        pred_norm = seq.get("predicted_laps_norm", [])
        act_norm = seq.get("actual_laps_norm", [])

        ctx_ms = [v * scale + offset for v in ctx_norm]
        pred_ms = [v * scale + offset for v in pred_norm]
        act_ms = [v * scale + offset for v in act_norm]

        n_ctx = len(ctx_ms)
        n_pred = len(pred_ms)
        lap_numbers = list(range(1, n_ctx + n_pred + 1))

        results.append(
            {
                "driver_name": driver_map.get(seq["driver"], str(seq["driver"])),
                "circuit_name": circuit_map.get(seq["circuit"], str(seq["circuit"])),
                "year": year_map.get(seq["year"], str(seq["year"])),
                "context_actual_ms": ctx_ms,
                "predicted_ms": pred_ms,
                "actual_ms": act_ms,
                "lap_numbers": lap_numbers,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_laptime_denorm_params(normalizer) -> tuple[float, float]:
    """Return (offset, scale) to fully invert LapTime normalization.

    For all scaler types the inverse transform is: x = z * scale + offset
      - StandardScaler:  offset=mean,   scale=std
      - MinMaxScaler:    offset=min,    scale=(max - min)
      - RobustScaler:    offset=center, scale=scale
    """
    if normalizer is None:
        return 0.0, 1.0
    stats = normalizer.get_statistics()
    if stats is None:
        return 0.0, 1.0
    col_index = stats["columns"].index("LapTime") if "columns" in stats else 0
    if "std" in stats:
        # StandardScaler: z = (x - mean) / std  →  x = z * std + mean
        return float(stats["mean"][col_index]), float(stats["std"][col_index])
    if "max" in stats and "min" in stats:
        # MinMaxScaler: z = (x - min) / (max - min)  →  x = z * (max-min) + min
        return float(stats["min"][col_index]), float(stats["max"][col_index] - stats["min"][col_index])
    if "scale" in stats:
        # RobustScaler: z = (x - center) / scale  →  x = z * scale + center
        return float(stats["center"][col_index]), float(stats["scale"][col_index])
    return 0.0, 1.0


def get_test_years_from_config(config_path: str) -> list[int]:
    """Read test_years from a config.json; fall back to [2025]."""
    if not config_path:
        return [2025]
    try:
        from config.base import Config
        cfg = Config.load(Path(config_path))
        return list(cfg.training.test_years)
    except Exception:
        return [2025]


def resolve_checkpoint_and_config(seed_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """
    Given a seed directory, return (checkpoint_path, config_path).

    Tries ``checkpoints/best_model.pt`` and ``config.json`` in the same dir.
    """
    ckpt = seed_dir / "checkpoints" / "best_model.pt"
    if not ckpt.exists():
        # Single-seed run might store checkpoint in a different location
        alts = list(seed_dir.rglob("best_model.pt"))
        ckpt = alts[0] if alts else None

    cfg = seed_dir / "config.json"
    if not cfg.exists():
        cfg = None

    return ckpt, cfg
