"""
Evaluation script for F1 lap time prediction models.

Usage:
    python code/evaluate.py --checkpoint results/phase1/checkpoints/best_model.pt --test-years 2022
    python code/evaluate.py --checkpoint model.pt --config config.json
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

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataloaders import StintDataloader, LapTimeNormalizer
from dataloaders.utils import load_all_races
from models import Seq2SeqGRU, Evaluator, report_evaluation
from config.base import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


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
    
    # Create model with saved config
    model = Seq2SeqGRU(
        input_size=model_config.get('input_size', 33),
        output_size=model_config.get('output_size', 1),
        hidden_size=model_config.get('hidden_size', 128),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.2),
        embedding_dims=model_config.get('embedding_dims', {}),
        vocab_sizes=model_config.get('vocab_sizes', {}),
        device=device,
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    logger.info(f"Model loaded: {model.model_info()}")
    logger.info(f"Checkpoint epoch: {checkpoint['epoch']}, Val loss: {checkpoint['val_loss']:.6f}")
    
    return model, checkpoint


def create_test_dataloader(
    years: list,
    batch_size: int = 32,
    window_size: int = 20,
    normalize: bool = True,
    scaler_type: str = "standard",
    normalizer: LapTimeNormalizer = None,
):
    """
    Create test dataloader.
    
    Parameters
    ----------
    years : list
        Years to load
    batch_size : int
        Batch size
    window_size : int
        Window size for stint sequences
    
    Returns
    -------
    DataLoader
        Test dataloader
    """
    logger.info(f"Creating test dataloader for years {years}")
    
    test_ds = StintDataloader(
        year=years,
        window_size=window_size,
        augment_prob=0.0,
        normalize=normalize,
        scaler_type=scaler_type,
        normalizer=normalizer,
    )
    
    logger.info(f"Test set: {len(test_ds)} stints")
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
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
    
    # Load config if available
    config = None
    normalizer = None
    normalize = True
    scaler_type = "standard"
    if config_path and config_path.exists():
        config = Config.load(config_path)
        batch_size = config.training.batch_size
        window_size = config.data.window_size
        normalize = config.data.normalize
        scaler_type = config.data.scaler_type
        if normalize:
            normalizer = LapTimeNormalizer(scaler_type=scaler_type)
            try:
                normalizer.load(config.training.train_years)
                logger.info("Loaded training normalizer for evaluation")
            except FileNotFoundError:
                logger.info("Training normalizer not found; fitting on train data")
                train_data = load_all_races(config.training.train_years, session="Race")
                normalizer.fit(train_data, years=config.training.train_years)
    else:
        batch_size = 32
        window_size = 20
    
    # Create test dataloader
    test_loader = create_test_dataloader(
        years=test_years,
        batch_size=batch_size,
        window_size=window_size,
        normalize=normalize,
        scaler_type=scaler_type,
        normalizer=normalizer,
    )
    
    # Create evaluator
    criterion = nn.MSELoss()
    evaluator = Evaluator(model, criterion, device=device)
    
    # Run evaluation
    logger.info("\nRunning evaluation...")
    metrics, predictions = evaluator.evaluate(test_loader, return_predictions=True)
    
    # Log metrics
    logger.info("\n" + "=" * 60)
    logger.info("TEST METRICS")
    logger.info("=" * 60)
    
    for key in ['loss', 'mae', 'rmse', 'mape', 'median_ae']:
        value = metrics.get(key, 0)
        if key == 'mape':
            logger.info(f"  {key.upper()}: {value:.2f}%")
        else:
            logger.info(f"  {key.upper()}: {value:.2f}")
    
    # Denormalize predictions for interpretability
    logger.info("\nDenormalizing predictions...")

    if normalizer is None:
        normalizer = test_loader.dataset.normalizer

    stats = normalizer.get_statistics() if normalizer is not None else None
    if stats is None:
        logger.warning("No normalizer available; skipping denormalized metrics")
        pred_denorm = predictions['predictions']
        target_denorm = predictions['targets']
    else:
        col_index = stats['columns'].index('LapTime') if 'columns' in stats else 0
        if 'std' in stats and 'mean' in stats:
            pred_denorm = predictions['predictions'] * stats['std'][col_index]
            pred_denorm += stats['mean'][col_index]
            target_denorm = predictions['targets'] * stats['std'][col_index]
            target_denorm += stats['mean'][col_index]
        elif 'min' in stats and 'max' in stats:
            span = stats['max'][col_index] - stats['min'][col_index]
            pred_denorm = predictions['predictions'] * span + stats['min'][col_index]
            target_denorm = predictions['targets'] * span + stats['min'][col_index]
        elif 'center' in stats and 'scale' in stats:
            pred_denorm = predictions['predictions'] * stats['scale'][col_index]
            pred_denorm += stats['center'][col_index]
            target_denorm = predictions['targets'] * stats['scale'][col_index]
            target_denorm += stats['center'][col_index]
        else:
            logger.warning("Unknown normalizer stats; skipping denormalized metrics")
            pred_denorm = predictions['predictions']
            target_denorm = predictions['targets']
    
    # Compute denormalized metrics
    metrics_denorm = {
        'mae_ms': float(np.mean(np.abs(pred_denorm - target_denorm))),
        'rmse_ms': float(np.sqrt(np.mean((pred_denorm - target_denorm) ** 2))),
        'median_ae_ms': float(np.median(np.abs(pred_denorm - target_denorm))),
    }
    
    logger.info("\nDenormalized Metrics (ms):")
    for key, value in metrics_denorm.items():
        logger.info(f"  {key}: {value:.2f}")
    
    # Error breakdown
    errors = np.abs(pred_denorm - target_denorm)
    error_breakdown = {
        'error_0_10ms': float((errors < 10).sum() / len(errors) * 100),
        'error_10_50ms': float(((errors >= 10) & (errors < 50)).sum() / len(errors) * 100),
        'error_50_100ms': float(((errors >= 50) & (errors < 100)).sum() / len(errors) * 100),
        'error_100_200ms': float(((errors >= 100) & (errors < 200)).sum() / len(errors) * 100),
        'error_200plus_ms': float((errors >= 200).sum() / len(errors) * 100),
    }
    
    logger.info("\nError Breakdown:")
    for category, percentage in error_breakdown.items():
        logger.info(f"  {category}: {percentage:.2f}%")
    
    # Save evaluation results
    results = {
        'checkpoint': str(checkpoint_path),
        'test_years': test_years,
        'metrics': metrics,
        'metrics_denormalized_ms': metrics_denorm,
        'error_breakdown': error_breakdown,
    }
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nEvaluation results saved to {results_path}")
    
    # Generate report
    report_path = output_dir / "evaluation_report.txt"
    report = report_evaluation(metrics, save_path=report_path)
    print("\n" + report)
    
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
