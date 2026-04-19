"""
Evaluation script for F1 lap time prediction models.

Usage:
    python scripts/evaluate.py --checkpoint results/phase1/checkpoints/best_model.pt --test-years 2022
    python scripts/evaluate.py --checkpoint model.pt --config config.json
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

# Add f1predictor library to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "f1predictor"))

from dataloaders import StintDataloader, LapTimeNormalizer
from dataloaders.autoregressive_dataloader import AutoregressiveLapDataloader
from dataloaders.utils import load_all_races
from models import Seq2Seq, Evaluator, compute_regression_metrics, report_evaluation
from models.rollout_evaluator import (
    evaluate_autoregressive_rollout,
    report_rollout_evaluation,
    denormalize_rollout_metrics,
)
from config.base import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def _collate_with_none_filter(batch):
    """Filter out None samples before default collation."""
    return default_collate([sample for sample in batch if sample is not None])


def _load_eval_runtime_config(config_path: Path):
    """Load evaluation runtime settings from config and optional normalizer."""
    runtime = {
        'batch_size': 32,
        'window_size': 20,
        'context_window': 20,
        'multistep_horizon': 1,
        'normalize': True,
        'scaler_type': 'standard',
        'normalizer': None,
    }

    if not (config_path and config_path.exists()):
        return runtime

    config = Config.load(config_path)
    runtime['batch_size'] = config.training.batch_size
    runtime['window_size'] = config.data.window_size
    runtime['context_window'] = getattr(config.data, 'context_window', config.data.window_size)
    runtime['multistep_horizon'] = getattr(config.training, 'multistep_horizon', 1)
    runtime['normalize'] = config.data.normalize
    runtime['scaler_type'] = config.data.scaler_type

    if runtime['normalize']:
        normalizer = LapTimeNormalizer(scaler_type=runtime['scaler_type'])
        try:
            normalizer.load(config.training.train_years)
            logger.info("Loaded training normalizer for evaluation")
        except FileNotFoundError:
            logger.info("Training normalizer not found; fitting on train data")
            train_data = load_all_races(config.training.train_years, session="Race")
            normalizer.fit(train_data, years=config.training.train_years)
        runtime['normalizer'] = normalizer

    return runtime


def _log_core_metrics(metrics: dict):
    """Log core regression metrics."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST METRICS")
    logger.info("=" * 60)

    for key in ['loss', 'mae', 'rmse', 'mape', 'median_ae']:
        value = metrics.get(key, 0)
        if key == 'mape':
            logger.info(f"  {key.upper()}: {value:.2f}%")
        else:
            logger.info(f"  {key.upper()}: {value:.2f}")


def _attach_auxiliary_metrics(metrics: dict, predictions: dict):
    """Compute and attach pit/compound metrics if auxiliary outputs exist."""
    pit_preds = predictions.get('pit_predictions')
    pit_targs = predictions.get('pit_targets')
    if pit_preds is not None and pit_targs is not None:
        try:
            from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

            pit_probs = pit_preds.reshape(-1)
            pit_labels = pit_targs.reshape(-1).astype(int)
            try:
                pit_auroc = float(roc_auc_score(pit_labels, pit_probs))
            except Exception:
                pit_auroc = float('nan')
            pit_pred_binary = (pit_probs >= 0.5).astype(int)
            pit_acc = float(accuracy_score(pit_labels, pit_pred_binary))
            pit_cm = confusion_matrix(pit_labels, pit_pred_binary).tolist()
            logger.info(f"  PIT_AUROC: {pit_auroc:.3f}")
            logger.info(f"  PIT_ACC: {pit_acc:.3f}")
            logger.info(f"  PIT_CONFUSION: {pit_cm}")
            metrics['pit_auroc'] = pit_auroc
            metrics['pit_accuracy'] = pit_acc
            metrics['pit_confusion'] = pit_cm
        except Exception:
            try:
                pit_acc = float(((pit_preds >= 0.5).astype(int).reshape(-1) == pit_targs.reshape(-1)).mean())
            except Exception:
                pit_acc = float('nan')
            logger.info(f"  PIT_ACC: {pit_acc:.3f}")
            metrics['pit_accuracy'] = pit_acc

    comp_preds = predictions.get('compound_predictions')
    comp_targs = predictions.get('compound_targets')
    if comp_preds is not None and comp_targs is not None:
        try:
            from sklearn.metrics import confusion_matrix, accuracy_score

            comp_pred_flat = comp_preds.reshape(-1)
            comp_targ_flat = comp_targs.reshape(-1)
            comp_acc = float(accuracy_score(comp_targ_flat, comp_pred_flat))
            comp_cm = confusion_matrix(comp_targ_flat, comp_pred_flat).tolist()
            logger.info(f"  COMPOUND_ACC: {comp_acc:.3f}")
            logger.info(f"  COMPOUND_CONFUSION: {comp_cm}")
            metrics['compound_accuracy'] = comp_acc
            metrics['compound_confusion'] = comp_cm
        except Exception:
            try:
                comp_acc = float((comp_preds.reshape(-1) == comp_targs.reshape(-1)).mean())
            except Exception:
                comp_acc = float('nan')
            logger.info(f"  COMPOUND_ACC: {comp_acc:.3f}")
            metrics['compound_accuracy'] = comp_acc


def _denormalize_predictions(predictions: dict, normalizer: LapTimeNormalizer):
    """Convert normalized outputs back to milliseconds when normalizer stats are known."""
    if normalizer is None:
        logger.warning("No normalizer available; skipping denormalized metrics")
        return predictions['predictions'], predictions['targets']

    stats = normalizer.get_statistics()
    if stats is None:
        logger.warning("No normalizer statistics available; skipping denormalized metrics")
        return predictions['predictions'], predictions['targets']

    col_index = stats['columns'].index('LapTime') if 'columns' in stats else 0
    if 'std' in stats and 'mean' in stats:
        pred_denorm = predictions['predictions'] * stats['std'][col_index] + stats['mean'][col_index]
        targ_denorm = predictions['targets'] * stats['std'][col_index] + stats['mean'][col_index]
        return pred_denorm, targ_denorm
    if 'min' in stats and 'max' in stats:
        span = stats['max'][col_index] - stats['min'][col_index]
        pred_denorm = predictions['predictions'] * span + stats['min'][col_index]
        targ_denorm = predictions['targets'] * span + stats['min'][col_index]
        return pred_denorm, targ_denorm
    if 'center' in stats and 'scale' in stats:
        pred_denorm = predictions['predictions'] * stats['scale'][col_index] + stats['center'][col_index]
        targ_denorm = predictions['targets'] * stats['scale'][col_index] + stats['center'][col_index]
        return pred_denorm, targ_denorm

    logger.warning("Unknown normalizer stats; skipping denormalized metrics")
    return predictions['predictions'], predictions['targets']


def _build_ms_metrics(pred_denorm: np.ndarray, target_denorm: np.ndarray) -> tuple:
    """Build denormalized metrics and error bucket breakdown."""
    metrics_denorm_full = compute_regression_metrics(pred_denorm, target_denorm)
    metrics_denorm = {
        'mae_ms': metrics_denorm_full['mae'],
        'rmse_ms': metrics_denorm_full['rmse'],
        'median_ae_ms': metrics_denorm_full['median_ae'],
        'mape_percent': metrics_denorm_full['mape'],
        'q25_ae_ms': metrics_denorm_full['q25_ae'],
        'q50_ae_ms': metrics_denorm_full['q50_ae'],
        'q75_ae_ms': metrics_denorm_full['q75_ae'],
        'q95_ae_ms': metrics_denorm_full['q95_ae'],
        'q99_ae_ms': metrics_denorm_full['q99_ae'],
        'mean_bias_ms': metrics_denorm_full['mean_bias'],
        'median_bias_ms': metrics_denorm_full['median_bias'],
        'mse_ms2': metrics_denorm_full['loss'],
    }

    errors = np.abs(pred_denorm - target_denorm)
    error_breakdown = {
        'error_0_10ms': float((errors < 10).sum() / len(errors) * 100),
        'error_10_50ms': float(((errors >= 10) & (errors < 50)).sum() / len(errors) * 100),
        'error_50_100ms': float(((errors >= 50) & (errors < 100)).sum() / len(errors) * 100),
        'error_100_200ms': float(((errors >= 100) & (errors < 200)).sum() / len(errors) * 100),
        'error_200plus_ms': float((errors >= 200).sum() / len(errors) * 100),
    }
    return metrics_denorm_full, metrics_denorm, error_breakdown


def _save_evaluation_artifacts(
    output_dir: Path,
    checkpoint_path: Path,
    test_years: list,
    metrics: dict,
    metrics_denorm: dict,
    error_breakdown: dict,
    pred_denorm: np.ndarray,
    target_denorm: np.ndarray,
    metadata: list,
):
    """Persist evaluation outputs and return results path."""
    preds_file = output_dir / 'predictions.npz'
    meta_file = output_dir / 'predictions_metadata.json'
    try:
        np.savez_compressed(preds_file, predictions=pred_denorm, targets=target_denorm)
        with open(meta_file, 'w', encoding='utf-8') as mf:
            json.dump(metadata, mf, default=str, indent=2)
        logger.info(f"Saved per-sample predictions to {preds_file} and metadata to {meta_file}")
    except Exception as exc:
        logger.warning(f"Failed to save predictions/metadata: {exc}")

    results = {
        'checkpoint': str(checkpoint_path),
        'test_years': test_years,
        'metrics_normalized': metrics,
        'metrics_denormalized_ms': metrics_denorm,
        'error_breakdown': error_breakdown,
        'predictions_file': str(preds_file),
        'predictions_metadata_file': str(meta_file),
        'metrics': metrics,
    }
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as out:
        json.dump(results, out, indent=2)
    logger.info(f"\nEvaluation results saved to {results_path}")
    return results_path


def _run_rollout_evaluation(
    model,
    test_years: list,
    config_path: Path,
    normalizer,
    device: str,
    output_dir: Path,
):
    """
    Run autoregressive rollout evaluation and save results.

    Creates an AutoregressiveLapDataloader for test years, rolls out the model
    over full driver-race sequences, and computes error-accumulation metrics.
    """
    from dataloaders.autoregressive_dataloader import AutoregressiveLapDataloader

    config = Config.load(config_path) if config_path and config_path.exists() else None
    context_window = config.data.context_window if config else 10
    normalize = config.data.normalize if config else True
    scaler_type = config.data.scaler_type if config else 'standard'

    logger.info("\nRunning autoregressive rollout evaluation...")
    import os
    old_skip = os.environ.get('SKIP_PRECOMPUTE')
    os.environ['SKIP_PRECOMPUTE'] = '1'  # skip precompute; we only need raw data
    try:
        test_ds = AutoregressiveLapDataloader(
            year=test_years,
            context_window=context_window,
            normalize=normalize,
            scaler_type=scaler_type,
            normalizer=normalizer,
            augment_prob=0.0,
        )
    finally:
        if old_skip is None:
            os.environ.pop('SKIP_PRECOMPUTE', None)
        else:
            os.environ['SKIP_PRECOMPUTE'] = old_skip

    rollout_metrics = evaluate_autoregressive_rollout(
        model=model,
        test_dataset=test_ds,
        device=device,
    )

    # Denormalize to ms
    rollout_denorm = denormalize_rollout_metrics(rollout_metrics, normalizer)

    # Save results
    rollout_output = {
        'rollout_metrics_normalized': rollout_metrics,
    }
    if rollout_denorm is not None:
        rollout_output['rollout_metrics_ms'] = rollout_denorm

    rollout_path = output_dir / "rollout_evaluation.json"
    with open(rollout_path, 'w') as f:
        json.dump(rollout_output, f, indent=2)
    logger.info(f"Rollout metrics saved to {rollout_path}")

    # Print report
    report = report_rollout_evaluation(
        metrics=rollout_metrics,
        denorm_metrics=rollout_denorm,
        save_path=str(output_dir / "rollout_report.txt"),
    )
    print("\n" + report)


def load_model_from_checkpoint(checkpoint_path: Path, device: str = 'cpu'):
    """
    Load model from checkpoint.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file
    device : str
        Device to load on
    
    Returns
    -------
    tuple of (model, checkpoint)
        (loaded model, checkpoint dict)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Allow numpy scalar type in older checkpoints and load full pickle (trusted local file)
    try:
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    except Exception:
        pass
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Reconstruct model config from checkpoint
    model_config = checkpoint.get('model_config', {})
    
    if not model_config:
        raise ValueError("No model config found in checkpoint. Use --config flag or retrain.")
    
    # Detect architecture generation from state dict keys so old checkpoints
    # (single Linear decoder head) and new ones (MLP Sequential) both load.
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}

    # Infer decoder head shape from Sequential indices so runs with non-default
    # head widths / layer counts reload correctly.  Sequential layout:
    #   dropout==0: [Linear(0), GELU(1), Linear(2), GELU(3), ..., Linear(N)]
    #   dropout>0:  [Linear(0), GELU(1), Dropout(2), Linear(3), ..., Linear(N)]
    linear_keys = [
        k for k in state_dict
        if k.startswith('fc_decoder_output.') and k.endswith('.weight')
        and k.split('.')[1].isdigit()
    ]
    if linear_keys:
        use_mlp_decoder_head = True
        linear_indices = sorted(int(k.split('.')[1]) for k in linear_keys)
        decoder_head_layers = len(linear_indices)
        decoder_head_hidden_inferred = int(state_dict[f'fc_decoder_output.{linear_indices[0]}.weight'].shape[0])
        step = (linear_indices[1] - linear_indices[0]) if len(linear_indices) >= 2 else 2
        decoder_head_dropout_inferred = 0.1 if step >= 3 else 0.0
    else:
        use_mlp_decoder_head = False
        decoder_head_layers = 1
        decoder_head_hidden_inferred = model_config.get('decoder_head_hidden', 64)
        decoder_head_dropout_inferred = 0.0
    logger.info(
        f"Checkpoint decoder head: {'MLP' if use_mlp_decoder_head else 'Linear (legacy)'} "
        f"(layers={decoder_head_layers}, hidden={decoder_head_hidden_inferred}, "
        f"dropout_present={decoder_head_dropout_inferred > 0})"
    )

    # Infer encoder_pool from fc_encoder_to_hidden input dim (B3).
    # Input dim = hidden -> legacy; 2*hidden -> encoder_pool=True.
    hidden_size = model_config.get('hidden_size', 128)
    encoder_pool_inferred = False
    fc_key = 'fc_encoder_to_hidden.weight'
    if fc_key in state_dict:
        fc_in = int(state_dict[fc_key].shape[1])
        encoder_pool_inferred = fc_in == 2 * hidden_size
    encoder_pool_k_inferred = model_config.get('encoder_pool_k', 5)
    logger.info(f"Checkpoint encoder_pool: {encoder_pool_inferred} (k={encoder_pool_k_inferred})")

    # Infer skip-connection type from state_dict keys (B2).
    if any(k.startswith('decoder_film.') for k in state_dict):
        decoder_skip_type_inferred = 'film'
    elif any(k.startswith('decoder_skip.') for k in state_dict):
        decoder_skip_type_inferred = 'additive'
    else:
        decoder_skip_type_inferred = 'none'
    logger.info(f"Checkpoint decoder skip: {decoder_skip_type_inferred}")

    # Infer decoder_extra_features_size from the checkpoint's decoder layer
    # shape.  `decoder.weight_ih_l0` is (3*hidden, decoder_input_size) for a
    # GRU / (4*hidden, decoder_input_size) for an LSTM, where
    # decoder_input_size = output_size + decoder_extra_features_size.  This
    # overrides the saved model_config entry because older configs omitted
    # the field entirely and would otherwise default to 0.
    output_size = model_config.get('output_size', 1)
    decoder_extra_saved = model_config.get('decoder_extra_features_size')
    decoder_extra_inferred = decoder_extra_saved
    ih_key = 'decoder.weight_ih_l0'
    if ih_key in state_dict:
        inferred_input = int(state_dict[ih_key].shape[1])
        decoder_extra_inferred = max(0, inferred_input - output_size)
        if decoder_extra_saved is not None and decoder_extra_saved != decoder_extra_inferred:
            logger.warning(
                "decoder_extra_features_size in config (%s) disagrees with checkpoint (%s); "
                "using checkpoint-inferred value",
                decoder_extra_saved, decoder_extra_inferred,
            )
    decoder_extra_features_size = decoder_extra_inferred if decoder_extra_inferred is not None else 0

    # Create model with saved config
    model = Seq2Seq(
        input_size=model_config.get('input_size', 33),
        output_size=output_size,
        hidden_size=model_config.get('hidden_size', 128),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.2),
        decoder_type=model_config.get('decoder_type', 'gru'),
        encoder_type=model_config.get('encoder', model_config.get('encoder_type', 'gru')),
        embedding_dims=model_config.get('embedding_dims', {}),
        vocab_sizes=model_config.get('vocab_sizes', {}),
        device=device,
        decoder_extra_features_size=decoder_extra_features_size,
        use_mlp_decoder_head=use_mlp_decoder_head,
        decoder_head_hidden=decoder_head_hidden_inferred,
        decoder_head_layers=decoder_head_layers,
        decoder_head_dropout=decoder_head_dropout_inferred,
        decoder_skip_type=decoder_skip_type_inferred,
        encoder_pool=encoder_pool_inferred,
        encoder_pool_k=encoder_pool_k_inferred,
    )
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    logger.info(f"Model loaded: {model.model_info()}")
    logger.info(f"Checkpoint epoch: {checkpoint['epoch']}, Val loss: {checkpoint['val_loss']:.6f}")
    
    return model, checkpoint


def create_test_dataloader(
    years: list,
    batch_size: int = 32,
    window_size: int = 20,
    context_window: int = None,
    multistep_horizon: int = 1,
    normalize: bool = True,
    scaler_type: str = "standard",
    normalizer: LapTimeNormalizer = None,
):
    """Create test dataloader using AutoregressiveLapDataloader."""
    context_window = context_window or window_size
    logger.info(f"Creating test dataloader for years {years}")

    test_ds = AutoregressiveLapDataloader(
        year=years,
        context_window=context_window,
        multistep_horizon=multistep_horizon,
        augment_prob=0.0,
        normalize=normalize,
        scaler_type=scaler_type,
        normalizer=normalizer,
        require_normalizer=normalize,
    )

    logger.info(f"Test set: {len(test_ds)} lap pairs")

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_with_none_filter,
    )

    return test_loader


def evaluate(
    checkpoint_path: Path,
    test_years: list,
    config_path: Path = None,
    device: str = 'cpu',
    output_dir: Path = None,
):
    """
    Run evaluation on test set.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to model checkpoint
    test_years : list
        Years to test on
    config_path : Path, optional
        Path to config file (for loading exact hyperparameters)
    device : str
        Device to use
    output_dir : Path, optional
        Directory to save results
    """
    if output_dir is None:
        output_dir = checkpoint_path.parent.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("F1 LAP TIME PREDICTION - EVALUATION")
    logger.info("=" * 60)
    
    # Load model
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    
    runtime = _load_eval_runtime_config(config_path)
    
    # Create test dataloader — always H=1 so next-lap MAE is comparable across
    # runs regardless of training multistep_horizon. Rollout evaluation below
    # handles multi-step horizon separately via evaluate_autoregressive_rollout.
    test_loader = create_test_dataloader(
        years=test_years,
        batch_size=runtime['batch_size'],
        window_size=runtime['window_size'],
        context_window=runtime['context_window'],
        multistep_horizon=1,
        normalize=runtime['normalize'],
        scaler_type=runtime['scaler_type'],
        normalizer=runtime['normalizer'],
    )
    
    # Create evaluator
    criterion = nn.MSELoss()
    evaluator = Evaluator(model, criterion, device=device)
    
    # Run evaluation
    logger.info("\nRunning evaluation...")
    metrics, predictions = evaluator.evaluate(test_loader, return_predictions=True)
    
    _log_core_metrics(metrics)
    _attach_auxiliary_metrics(metrics, predictions)
    
    # Denormalize predictions for interpretability
    logger.info("\nDenormalizing predictions...")

    normalizer = runtime['normalizer']
    if normalizer is None:
        normalizer = test_loader.dataset.normalizer
    pred_denorm, target_denorm = _denormalize_predictions(predictions, normalizer)
    
    metrics_denorm_full, metrics_denorm, error_breakdown = _build_ms_metrics(pred_denorm, target_denorm)
    
    logger.info("\nDenormalized Metrics (ms):")
    for key, value in metrics_denorm.items():
        logger.info(f"  {key}: {value:.2f}")
    
    logger.info("\nError Breakdown:")
    for category, percentage in error_breakdown.items():
        logger.info(f"  {category}: {percentage:.2f}%")

    _save_evaluation_artifacts(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        test_years=test_years,
        metrics=metrics,
        metrics_denorm=metrics_denorm,
        error_breakdown=error_breakdown,
        pred_denorm=pred_denorm,
        target_denorm=target_denorm,
        metadata=predictions.get('metadata', []),
    )
    
    # Generate report on denormalized metrics for interpretability
    report_path = output_dir / "evaluation_report.txt"
    report = report_evaluation(metrics_denorm_full, save_path=report_path, unit_label="ms")
    print("\n" + report)
    
    # --- Autoregressive rollout evaluation ---
    try:
        _run_rollout_evaluation(
            model=model,
            test_years=test_years,
            config_path=config_path,
            normalizer=normalizer,
            device=device,
            output_dir=output_dir,
        )
    except Exception as exc:
        logger.warning(f"Rollout evaluation failed (non-fatal): {exc}")

    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate F1 lap time prediction model")
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to model checkpoint')
    parser.add_argument('--test-years', type=int, nargs='+', default=[2025], help='Test years')
    parser.add_argument('--config', type=Path, help='Path to config file')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--output', type=Path, help='Output directory')
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate(
        checkpoint_path=args.checkpoint,
        test_years=args.test_years,
        config_path=args.config,
        device=args.device,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
