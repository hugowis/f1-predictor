"""
Autoregressive rollout evaluation for measuring error accumulation.

Runs the model autoregressively over full driver-race sequences to measure
how prediction errors compound over multiple laps. Complements the existing
one-step evaluation with multi-step metrics.

Metrics computed:
1. Horizon-bucketed MAE/RMSE — error at each prediction step
2. Cumulative drift — signed error accumulation over time
3. Stint total time error — error on total predicted stint duration
4. Stability ratio — RMSE at last horizon / RMSE at horizon 1
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from dataloaders.utils import get_decoder_extra_feature_indices as _get_decoder_extra_feat_idx
    _DECODER_EXTRA_FEAT_IDX = _get_decoder_extra_feat_idx()
except Exception:
    _DECODER_EXTRA_FEAT_IDX = []

logger = logging.getLogger(__name__)


def evaluate_autoregressive_rollout(
    model: nn.Module,
    test_dataset,
    device: str = 'cpu',
    max_horizon: int = 50,
    return_sequences: bool = False,
) -> Dict:
    """
    Evaluate model via full autoregressive rollout over driver-race sequences.

    For each driver in each race:
    1. Use first ``context_window`` ground-truth laps as encoder input
    2. Predict the next lap (single decoder step, no teacher forcing)
    3. Shift the context window: drop oldest lap, append new row whose
       LapTime is the model's prediction (other features from ground truth)
    4. Repeat until end of sequence or ``max_horizon`` steps

    Parameters
    ----------
    model : nn.Module
        Trained model (will be set to eval mode).
    test_dataset : AutoregressiveLapDataloader
        Test dataset containing ``data``, ``normalizer``, ``_get_lap_features``,
        ``context_window``, and ``numeric_columns``.
    device : str
        Torch device.
    max_horizon : int
        Maximum number of autoregressive steps per sequence.
    return_sequences : bool
        If True, return a tuple ``(metrics, sequence_rollouts)`` where
        ``sequence_rollouts`` is the list of per-sequence dicts augmented with
        ``predicted_laps_norm`` and ``actual_laps_norm`` arrays (normalized
        space) and ``context_actual_norm`` (the seed context laps).
        If False (default), return only the metrics dict.

    Returns
    -------
    dict or tuple
        Metrics dict, or ``(metrics, sequence_rollouts)`` when
        ``return_sequences=True``.
    """
    model.eval()
    context_window = test_dataset.context_window
    numeric_columns = test_dataset.numeric_columns
    lap_time_feat_idx = numeric_columns.index('LapTime')  # index inside feature vector
    cum_stint_time_feat_idx = (numeric_columns.index('cumulative_stint_time')
                               if 'cumulative_stint_time' in numeric_columns else None)
    stint_lap_feat_idx = (numeric_columns.index('stint_lap')
                          if 'stint_lap' in numeric_columns else None)

    # Work with normal laps only (matching training/pair-generation filter)
    data = test_dataset.data.copy()
    data = data[data['is_normal_lap'] == 1].copy()

    grouped = data.groupby(['Driver', 'Year', 'Circuit'])
    total_groups = len(grouped)

    # Collect per-sequence rollout data
    sequence_rollouts = []

    with torch.no_grad():
        for (driver, year, circuit), group_data in grouped:
            group_data = group_data.sort_values('LapNumber')

            if len(group_data) < context_window + 1:
                continue

            # Normalize the group using the dataset's fitted normalizer
            if test_dataset.normalizer is not None:
                group_norm = test_dataset.normalizer.transform(group_data)
            else:
                group_norm = group_data

            # Extract per-lap feature vectors and lap times
            all_features = []
            all_laptimes = []
            for _, lap in group_norm.iterrows():
                feat = test_dataset._get_lap_features(lap)
                all_features.append(feat)
                all_laptimes.append(float(lap['LapTime']))

            all_features = np.array(all_features, dtype=np.float32)
            all_laptimes = np.array(all_laptimes, dtype=np.float32)

            # Replace NaN in features (e.g. delta_to_car_ahead for lead car,
            # missing weather data) with 0.0 to prevent NaN propagation.
            np.nan_to_num(all_features, copy=False, nan=0.0)
            np.nan_to_num(all_laptimes, copy=False, nan=0.0)

            num_laps = len(all_features)
            num_predictions = min(num_laps - context_window, max_horizon)
            if num_predictions <= 0:
                continue

            # Initial context from ground truth
            context = all_features[:context_window].copy()

            # Accumulator for cumulative_stint_time feedback
            raw_cum_stint_time = 0.0
            normalizer = test_dataset.normalizer

            errors = []        # |pred - target| per step
            signed_errors = [] # pred - target per step
            predicted_laps_norm = []  # predicted lap times (normalized)
            actual_laps_norm = []     # ground-truth lap times (normalized)

            for h in range(num_predictions):
                target_idx = context_window + h
                target_laptime = all_laptimes[target_idx]

                # Build encoder input tensor
                context_tensor = torch.from_numpy(context).unsqueeze(0).to(device)

                # Decoder input: previous lap time + strategy-known extra features
                # Extra features (tyre life, fuel proxy, stint lap, compound)
                # come from the TARGET lap since they are known from race strategy.
                last_laptime = float(context[-1, lap_time_feat_idx])
                if _DECODER_EXTRA_FEAT_IDX:
                    target_extra = all_features[target_idx][_DECODER_EXTRA_FEAT_IDX].astype(np.float32)
                    decoder_data = np.concatenate([[last_laptime], target_extra])[np.newaxis, np.newaxis, :]
                else:
                    decoder_data = np.array([[[last_laptime]]], dtype=np.float32)
                decoder_input = torch.tensor(decoder_data, dtype=torch.float32, device=device)

                # Single-step forward pass (no teacher forcing)
                outputs = model(context_tensor, decoder_input, teacher_forcing=False)
                if isinstance(outputs, dict):
                    pred_laptime = outputs['lap'].squeeze().item()
                else:
                    pred_laptime = outputs.squeeze().item()

                # If model produces NaN/Inf, stop this sequence's rollout
                if not np.isfinite(pred_laptime):
                    break

                errors.append(abs(pred_laptime - target_laptime))
                signed_errors.append(pred_laptime - target_laptime)
                predicted_laps_norm.append(pred_laptime)
                actual_laps_norm.append(target_laptime)

                # Shift context window: use ground-truth features for the new
                # row but override LapTime with the model's prediction.  This
                # isolates the metric to lap-time error accumulation without
                # leaking future non-LapTime features being a confound, because
                # the model only feeds back its own LapTime prediction.
                new_row = all_features[target_idx].copy()
                new_row[lap_time_feat_idx] = pred_laptime

                # Feed back predicted cumulative_stint_time so the model
                # sees its own accumulated stint duration during rollout.
                if cum_stint_time_feat_idx is not None and normalizer is not None:
                    raw_pred = float(normalizer._inverse_transform_column_values(
                        pd.Series([pred_laptime]), lap_time_feat_idx).iloc[0])
                    raw_stint_lap = float(normalizer._inverse_transform_column_values(
                        pd.Series([new_row[stint_lap_feat_idx]]), stint_lap_feat_idx).iloc[0])
                    if raw_stint_lap <= 1.0:
                        raw_cum_stint_time = raw_pred
                    else:
                        raw_cum_stint_time += raw_pred
                    norm_cum = float(normalizer._transform_column_values(
                        pd.Series([raw_cum_stint_time]), cum_stint_time_feat_idx).iloc[0])
                    new_row[cum_stint_time_feat_idx] = norm_cum

                context = np.vstack([context[1:], new_row[np.newaxis, :]])

            # Only include sequences that produced at least one valid step
            if errors:
                entry = {
                    'driver': int(driver),
                    'year': int(year),
                    'circuit': int(circuit),
                    'errors': errors,
                    'signed_errors': signed_errors,
                }
                if return_sequences:
                    entry['predicted_laps_norm'] = predicted_laps_norm
                    entry['actual_laps_norm'] = actual_laps_norm
                    # Context window lap times (ground-truth seed, normalized)
                    entry['context_actual_norm'] = all_laptimes[:context_window].tolist()
                sequence_rollouts.append(entry)

    total_groups = len(grouped)
    logger.info(
        f"Rollout evaluation: {len(sequence_rollouts)}/{total_groups} sequences "
        f"produced valid predictions"
    )

    metrics = _compute_rollout_metrics(sequence_rollouts)
    if return_sequences:
        return metrics, sequence_rollouts
    return metrics


def _compute_rollout_metrics(sequence_rollouts) -> Dict:
    """
    Compute rollout metrics from per-sequence rollout data.

    Metrics
    -------
    1. ``horizon_mae`` / ``horizon_rmse``: dict mapping horizon step (1-based)
       to mean absolute / root-mean-square error across all sequences that
       reached that horizon.
    2. ``horizon_mean_drift`` / ``horizon_median_abs_drift``: cumulative signed
       error at each horizon step averaged across sequences.
    3. ``stint_total_time_mae`` / ``stint_total_time_rmse``: error on total
       predicted stint duration (sum of predicted laps vs. sum of actual laps).
    4. ``stability_ratio``: RMSE at last horizon / RMSE at horizon 1.
    """
    if not sequence_rollouts:
        return {
            'horizon_mae': {},
            'horizon_rmse': {},
            'horizon_mean_drift': {},
            'horizon_median_abs_drift': {},
            'stint_total_time_mae': 0.0,
            'stint_total_time_rmse': 0.0,
            'stability_ratio': 1.0,
            'num_sequences': 0,
        }

    # --- 1. Horizon-bucketed MAE / RMSE ---
    horizon_errors = defaultdict(list)
    for seq in sequence_rollouts:
        for h, err in enumerate(seq['errors'], start=1):
            horizon_errors[h].append(err)

    horizon_mae = {}
    horizon_rmse = {}
    for h in sorted(horizon_errors.keys()):
        errs = np.array(horizon_errors[h])
        horizon_mae[h] = float(np.nanmean(errs))
        horizon_rmse[h] = float(np.sqrt(np.nanmean(errs ** 2)))

    # --- 2. Cumulative drift (per-sequence, then averaged) ---
    max_h = max(len(s['signed_errors']) for s in sequence_rollouts)
    # For each horizon, collect the cumulative signed error across sequences
    horizon_drift_lists = defaultdict(list)
    for seq in sequence_rollouts:
        cumulative = 0.0
        for h, se in enumerate(seq['signed_errors'], start=1):
            cumulative += se
            horizon_drift_lists[h].append(cumulative)

    horizon_mean_drift = {}
    horizon_median_abs_drift = {}
    for h in sorted(horizon_drift_lists.keys()):
        drifts = np.array(horizon_drift_lists[h])
        horizon_mean_drift[h] = float(np.nanmean(drifts))
        horizon_median_abs_drift[h] = float(np.nanmedian(np.abs(drifts)))

    # --- 3. Stint total time error ---
    # total time error = |sum(signed_errors)| per sequence, since
    # sum(pred) - sum(target) = sum(signed_errors).
    stint_total_errors = []
    for seq in sequence_rollouts:
        total_signed = np.nansum(seq['signed_errors'])
        if np.isfinite(total_signed):
            stint_total_errors.append(abs(total_signed))

    if stint_total_errors:
        stint_total_errors = np.array(stint_total_errors)
        stint_total_time_mae = float(np.mean(stint_total_errors))
        stint_total_time_rmse = float(np.sqrt(np.mean(stint_total_errors ** 2)))
    else:
        stint_total_time_mae = float('nan')
        stint_total_time_rmse = float('nan')

    # --- 4. Stability ratio ---
    horizons = sorted(horizon_rmse.keys())
    if len(horizons) >= 2:
        first_rmse = horizon_rmse[horizons[0]]
        last_rmse = horizon_rmse[horizons[-1]]
        if first_rmse > 1e-12:
            stability_ratio = float(last_rmse / first_rmse)
        else:
            stability_ratio = float('inf')
    else:
        stability_ratio = 1.0

    return {
        'horizon_mae': {str(k): v for k, v in horizon_mae.items()},
        'horizon_rmse': {str(k): v for k, v in horizon_rmse.items()},
        'horizon_mean_drift': {str(k): v for k, v in horizon_mean_drift.items()},
        'horizon_median_abs_drift': {str(k): v for k, v in horizon_median_abs_drift.items()},
        'stint_total_time_mae': stint_total_time_mae,
        'stint_total_time_rmse': stint_total_time_rmse,
        'stability_ratio': stability_ratio,
        'num_sequences': len(sequence_rollouts),
        'max_horizon_reached': int(max_h),
    }


def report_rollout_evaluation(
    metrics: Dict,
    denorm_metrics: Optional[Dict] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Generate a human-readable report of rollout metrics.

    Parameters
    ----------
    metrics : dict
        Rollout metrics in normalized space.
    denorm_metrics : dict, optional
        Rollout metrics in denormalized (ms) space.
    save_path : str, optional
        Path to save the report text file.

    Returns
    -------
    str
        Formatted report string.
    """
    lines = []
    lines.append("=" * 65)
    lines.append("AUTOREGRESSIVE ROLLOUT EVALUATION REPORT")
    lines.append("=" * 65)
    lines.append(f"\nSequences evaluated: {metrics.get('num_sequences', 0)}")
    lines.append(f"Max horizon reached: {metrics.get('max_horizon_reached', 0)}")
    lines.append(f"Stability ratio (RMSE_last / RMSE_1): {metrics.get('stability_ratio', 0):.3f}")

    # Use denormalized metrics for the per-horizon table if available
    display = denorm_metrics if denorm_metrics is not None else metrics
    unit = "ms" if denorm_metrics is not None else "norm"

    # Horizon table
    h_mae = display.get('horizon_mae', {})
    h_rmse = display.get('horizon_rmse', {})
    h_drift = display.get('horizon_mean_drift', {})
    h_abs_drift = display.get('horizon_median_abs_drift', {})

    lines.append(f"\n{'Horizon':>8} {'MAE':>10} {'RMSE':>10} {'MeanDrift':>12} {'Med|Drift|':>12}  ({unit})")
    lines.append("-" * 65)
    for h_key in sorted(h_mae.keys(), key=lambda x: int(x)):
        mae_val = h_mae.get(h_key, 0)
        rmse_val = h_rmse.get(h_key, 0)
        drift_val = h_drift.get(h_key, 0)
        abs_drift_val = h_abs_drift.get(h_key, 0)
        lines.append(
            f"{h_key:>8} {mae_val:>10.2f} {rmse_val:>10.2f} {drift_val:>12.2f} {abs_drift_val:>12.2f}"
        )

    # Stint total time
    lines.append(f"\nStint Total Time Error ({unit}):")
    lines.append(f"  MAE:  {display.get('stint_total_time_mae', 0):.2f}")
    lines.append(f"  RMSE: {display.get('stint_total_time_rmse', 0):.2f}")

    report_str = "\n".join(lines)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_str)
        logger.info(f"Rollout report saved to {save_path}")

    return report_str


def denormalize_rollout_metrics(metrics: Dict, normalizer) -> Optional[Dict]:
    """
    Convert horizon-level normalized metrics to milliseconds.

    Multiplies MAE, RMSE, drift, and stint-total-time metrics by the
    normalizer's LapTime standard deviation (for standard scaling).

    Parameters
    ----------
    metrics : dict
        Rollout metrics in normalized space.
    normalizer : LapTimeNormalizer
        Fitted normalizer with ``get_statistics()``.

    Returns
    -------
    dict or None
        Metrics in ms, or None if normalizer stats are unavailable.
    """
    if normalizer is None:
        return None
    stats = normalizer.get_statistics()
    if stats is None:
        return None

    col_index = stats['columns'].index('LapTime') if 'columns' in stats else 0

    # Determine scale factor based on normalizer type
    if 'std' in stats:
        scale = stats['std'][col_index]
    elif 'max' in stats and 'min' in stats:
        scale = stats['max'][col_index] - stats['min'][col_index]
    elif 'scale' in stats:
        scale = stats['scale'][col_index]
    else:
        return None

    def _scale_dict(d, s):
        return {k: v * s for k, v in d.items()}

    return {
        'horizon_mae': _scale_dict(metrics.get('horizon_mae', {}), scale),
        'horizon_rmse': _scale_dict(metrics.get('horizon_rmse', {}), scale),
        'horizon_mean_drift': _scale_dict(metrics.get('horizon_mean_drift', {}), scale),
        'horizon_median_abs_drift': _scale_dict(metrics.get('horizon_median_abs_drift', {}), scale),
        'stint_total_time_mae': metrics.get('stint_total_time_mae', 0) * scale,
        'stint_total_time_rmse': metrics.get('stint_total_time_rmse', 0) * scale,
        'stability_ratio': metrics.get('stability_ratio', 1.0),
        'num_sequences': metrics.get('num_sequences', 0),
        'max_horizon_reached': metrics.get('max_horizon_reached', 0),
    }
