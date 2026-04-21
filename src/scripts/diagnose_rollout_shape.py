"""
Quantify how "linear" a model's autoregressive rollout is.

For each (driver, race) sequence, we run the model autoregressively for up to
``--max-horizon`` laps and compute:

- **linearity_r2**: R^2 of ``predicted_laptime ~ a + b * step``.  A value
  close to 1.0 means the rollout is essentially a straight line; lower
  values mean the model is bending around tyre-degradation cliffs, weather
  shifts, etc.
- **mae** in ms (inverse-transformed).
- Optional pred-vs-actual line plots for the longest rollouts per requested
  circuit.

The ``--freeze-decoder-extras`` flag reruns the rollout while pinning the
decoder's strategy-known features (TyreLife, fuel_proxy, stint_lap, compound)
to the last context lap.  If MAE barely moves with the freeze, the decoder is
ignoring those signals, which is the proximate cause of the linear look.

Usage
-----
    python src/scripts/diagnose_rollout_shape.py \\
        --results-dir results/repro_phase2_seed42 \\
        --races "Silverstone,Monaco,Singapore,Spa-Francorchamps,Suzuka"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "f1predictor"))

from dataloaders.autoregressive_dataloader import AutoregressiveLapDataloader  # noqa: E402
from dataloaders import LapTimeNormalizer  # noqa: E402
from dataloaders.utils import load_all_races  # noqa: E402
from models.rollout_evaluator import evaluate_autoregressive_rollout  # noqa: E402
from config.base import Config  # noqa: E402

from evaluate import load_model_from_checkpoint  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def _linearity_r2(values: np.ndarray) -> float:
    """R^2 of a simple linear regression ``values ~ a + b * step``.

    Returns 0.0 if the series is too short or flat.
    """
    y = np.asarray(values, dtype=np.float64)
    if y.size < 3:
        return float('nan')
    x = np.arange(y.size, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = intercept + slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def _load_circuit_vocab(data_dir: Path) -> dict[str, int]:
    path = data_dir / "vocabs" / "Circuit.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _denormalize_lap(values: np.ndarray, normalizer: LapTimeNormalizer) -> np.ndarray:
    """Inverse-transform a 1-D array of normalized lap times back to ms."""
    if normalizer is None:
        return values
    lap_idx = normalizer.columns.index('LapTime') if hasattr(normalizer, 'columns') else 0
    try:
        stats = normalizer.get_statistics()
        lap_idx = stats['columns'].index('LapTime')
    except Exception:
        pass
    series = pd.Series(values.astype(np.float64))
    return normalizer._inverse_transform_column_values(series, lap_idx).to_numpy()


def _build_test_dataset(config: Config, device: str) -> AutoregressiveLapDataloader:
    normalizer = None
    if config.data.normalize:
        normalizer = LapTimeNormalizer(scaler_type=config.data.scaler_type)
        try:
            normalizer.load(config.training.train_years)
            logger.info("Loaded training normalizer from disk")
        except FileNotFoundError:
            logger.info("No saved normalizer; fitting on train years %s", config.training.train_years)
            train_data = load_all_races(config.training.train_years, session="Race")
            normalizer.fit(train_data, years=config.training.train_years)

    ds = AutoregressiveLapDataloader(
        year=config.training.test_years,
        context_window=getattr(config.data, 'context_window', config.data.window_size),
        multistep_horizon=1,
        augment_prob=0.0,
        normalize=config.data.normalize,
        scaler_type=config.data.scaler_type,
        normalizer=normalizer,
        require_normalizer=config.data.normalize,
    )
    return ds


def _rollout_summary(sequence_rollouts: list[dict], normalizer) -> dict:
    """Aggregate per-sequence rollouts into summary stats."""
    r2s = []
    maes_ms = []
    per_seq = []
    for seq in sequence_rollouts:
        preds = np.array(seq['predicted_laps_norm'], dtype=np.float64)
        acts = np.array(seq['actual_laps_norm'], dtype=np.float64)
        if preds.size < 3:
            continue
        r2 = _linearity_r2(preds)
        if normalizer is not None:
            preds_ms = _denormalize_lap(preds, normalizer)
            acts_ms = _denormalize_lap(acts, normalizer)
            mae_ms = float(np.mean(np.abs(preds_ms - acts_ms)))
        else:
            mae_ms = float(np.mean(np.abs(preds - acts)))
        r2s.append(r2)
        maes_ms.append(mae_ms)
        per_seq.append({
            'driver': seq['driver'],
            'year': seq['year'],
            'circuit': seq['circuit'],
            'num_steps': int(preds.size),
            'linearity_r2': float(r2),
            'mae_ms': mae_ms,
        })

    r2s = np.array(r2s, dtype=np.float64)
    maes_ms = np.array(maes_ms, dtype=np.float64)
    return {
        'num_sequences': int(r2s.size),
        'linearity_r2_median': float(np.nanmedian(r2s)) if r2s.size else float('nan'),
        'linearity_r2_mean': float(np.nanmean(r2s)) if r2s.size else float('nan'),
        'linearity_r2_p25': float(np.nanpercentile(r2s, 25)) if r2s.size else float('nan'),
        'linearity_r2_p75': float(np.nanpercentile(r2s, 75)) if r2s.size else float('nan'),
        'frac_r2_above_095': float(np.mean(r2s > 0.95)) if r2s.size else float('nan'),
        'mae_ms_mean': float(np.nanmean(maes_ms)) if maes_ms.size else float('nan'),
        'mae_ms_median': float(np.nanmedian(maes_ms)) if maes_ms.size else float('nan'),
        'per_sequence': per_seq,
    }


def _select_sequences_for_plots(
    sequence_rollouts: list[dict],
    circuit_vocab: dict[str, int],
    requested_races: list[str],
    max_per_race: int = 2,
) -> list[dict]:
    """Pick the longest rollout(s) per requested circuit."""
    if not requested_races:
        sorted_seqs = sorted(sequence_rollouts, key=lambda s: -len(s['predicted_laps_norm']))
        return sorted_seqs[:5]

    chosen = []
    for race in requested_races:
        race = race.strip()
        cid = circuit_vocab.get(race)
        if cid is None:
            logger.warning("Circuit %r not in vocab; skipping", race)
            continue
        matching = [s for s in sequence_rollouts if s['circuit'] == cid]
        matching.sort(key=lambda s: -len(s['predicted_laps_norm']))
        for seq in matching[:max_per_race]:
            seq['_race_name'] = race
            chosen.append(seq)
    return chosen


def _plot_sequence(seq: dict, normalizer, race_name: str, output_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    preds = np.array(seq['predicted_laps_norm'], dtype=np.float64)
    acts = np.array(seq['actual_laps_norm'], dtype=np.float64)
    if normalizer is not None:
        preds = _denormalize_lap(preds, normalizer)
        acts = _denormalize_lap(acts, normalizer)

    x = np.arange(preds.size)
    slope, intercept = np.polyfit(x, preds, 1)
    lin_fit = intercept + slope * x
    r2 = _linearity_r2(seq['predicted_laps_norm'])

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax = axes[0]
    ax.plot(x, acts / 1000.0, label='actual', color='black', linewidth=1.6)
    ax.plot(x, preds / 1000.0, label='predicted', color='tab:blue', linewidth=1.6)
    ax.plot(x, lin_fit / 1000.0, label=f'linear fit (R²={r2:.3f})', color='tab:red', linestyle='--', linewidth=1.2)
    ax.set_ylabel('Lap time (s)')
    ax.set_title(f"{race_name} — driver={seq['driver']} year={seq['year']} (steps={preds.size})")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    residual_ms = preds - acts
    ax.plot(x, residual_ms, color='tab:purple', linewidth=1.2)
    ax.axhline(0.0, color='black', linewidth=0.6)
    ax.set_ylabel('Residual (ms)')
    ax.set_xlabel('Rollout step')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def diagnose(
    results_dir: Path,
    checkpoint_path: Path,
    config_path: Path,
    test_years: list[int] | None,
    races: list[str],
    max_horizon: int,
    device: str,
    freeze_decoder_extras: bool,
    output_dir: Path,
) -> dict:
    config = Config.load(config_path)
    if test_years:
        config.training.test_years = test_years
    config.device = device

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading checkpoint from %s", checkpoint_path)
    model, _ = load_model_from_checkpoint(checkpoint_path, device=device)

    logger.info("Building test dataset (years=%s, context_window=%s)",
                config.training.test_years, getattr(config.data, 'context_window', config.data.window_size))
    test_ds = _build_test_dataset(config, device=device)

    logger.info("Running rollout (freeze_decoder_extras=%s, max_horizon=%d)",
                freeze_decoder_extras, max_horizon)
    _, sequence_rollouts = evaluate_autoregressive_rollout(
        model=model,
        test_dataset=test_ds,
        device=device,
        max_horizon=max_horizon,
        return_sequences=True,
        freeze_decoder_extras=freeze_decoder_extras,
    )

    summary = _rollout_summary(sequence_rollouts, test_ds.normalizer)
    summary['freeze_decoder_extras'] = bool(freeze_decoder_extras)
    summary['max_horizon'] = int(max_horizon)
    summary['checkpoint'] = str(checkpoint_path)
    summary['test_years'] = list(config.training.test_years)

    summary_path = output_dir / "linearity_r2.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s", summary_path)

    logger.info(
        "Linearity R² — median=%.3f  mean=%.3f  p25=%.3f  p75=%.3f  frac>0.95=%.2f",
        summary['linearity_r2_median'], summary['linearity_r2_mean'],
        summary['linearity_r2_p25'], summary['linearity_r2_p75'],
        summary['frac_r2_above_095'],
    )
    logger.info("Rollout MAE (ms) — mean=%.2f  median=%.2f  sequences=%d",
                summary['mae_ms_mean'], summary['mae_ms_median'], summary['num_sequences'])

    circuit_vocab = _load_circuit_vocab(Path("data"))
    selected = _select_sequences_for_plots(sequence_rollouts, circuit_vocab, races)
    for seq in selected:
        race_name = seq.get('_race_name', f"circuit_{seq['circuit']}")
        safe = race_name.replace(' ', '_').replace('/', '_')
        fname = f"{safe}_driver{seq['driver']}_year{seq['year']}.png"
        _plot_sequence(seq, test_ds.normalizer, race_name, output_dir / fname)
    logger.info("Plotted %d sequences to %s", len(selected), output_dir)

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True,
                        help='Run directory containing config.json and checkpoints/best_model.pt')
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='Path to checkpoint (defaults to <results-dir>/checkpoints/best_model.pt)')
    parser.add_argument('--config', type=Path, default=None,
                        help='Path to config.json (defaults to <results-dir>/config.json)')
    parser.add_argument('--test-years', type=int, nargs='+', default=None,
                        help='Override test years from config')
    parser.add_argument('--races', default="Silverstone,Monaco,Singapore,Spa-Francorchamps,Suzuka",
                        help='Comma-separated circuit names to plot')
    parser.add_argument('--max-horizon', type=int, default=50)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--freeze-decoder-extras', action='store_true',
                        help='Pin decoder strategy-known features to the last context lap (ablation)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Where to write plots/summary (defaults to <results-dir>/diagnostics/rollout_shape[_frozen])')

    args = parser.parse_args()

    ckpt = args.checkpoint or (args.results_dir / "checkpoints" / "best_model.pt")
    cfg = args.config or (args.results_dir / "config.json")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not cfg.exists():
        raise FileNotFoundError(f"Config not found: {cfg}")

    if args.output_dir is None:
        suffix = "_frozen" if args.freeze_decoder_extras else ""
        args.output_dir = args.results_dir / "diagnostics" / f"rollout_shape{suffix}"

    races = [r.strip() for r in args.races.split(',') if r.strip()]

    diagnose(
        results_dir=args.results_dir,
        checkpoint_path=ckpt,
        config_path=cfg,
        test_years=args.test_years,
        races=races,
        max_horizon=args.max_horizon,
        device=args.device,
        freeze_decoder_extras=args.freeze_decoder_extras,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
