"""
Training script for F1 lap time prediction models.

Usage:
    python code/train.py --config config/phase1.json --device cuda
    python code/train.py --phase 1  # Use preset Phase 1 config
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from analyze_results import analyze_results as run_analysis

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataloaders import StintDataloader, LapTimeNormalizer
from dataloaders.autoregressive_dataloader import AutoregressiveLapDataloader
from dataloaders.utils import get_compound_columns
from models import Seq2Seq, Trainer, create_scheduler
from config.base import Config, get_phase1_config, get_phase2_config
from evaluate import evaluate as run_evaluation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def create_dataloaders(config: Config, batch_size: int = 32):
    """
    Create training, validation, and test dataloaders.
    
    Parameters
    ----------
    config : Config
        Configuration object
    batch_size : int
        Batch size
    
    Returns
    -------
    tuple of DataLoaders
        (train_loader, val_loader, test_loader)
    """
    logger.info("Creating dataloaders...")
    
    # Create datasets for each split (stint-based)
    train_ds = StintDataloader(
        year=config.training.train_years,
        window_size=config.data.window_size,
        augment_prob=config.data.augment_prob,
        normalize=config.data.normalize,
        scaler_type=config.data.scaler_type,
    )

    shared_normalizer = train_ds.normalizer if config.data.normalize else None

    val_ds = StintDataloader(
        year=config.training.val_years,
        window_size=config.data.window_size,
        augment_prob=0.0,  # No augmentation on validation
        normalize=config.data.normalize,
        scaler_type=config.data.scaler_type,
        normalizer=shared_normalizer,
    )

    test_ds = StintDataloader(
        year=config.training.test_years,
        window_size=config.data.window_size,
        augment_prob=0.0,  # No augmentation on test
        normalize=config.data.normalize,
        scaler_type=config.data.scaler_type,
        normalizer=shared_normalizer,
    )
    
    logger.info(f"Train set: {len(train_ds)} stints")
    logger.info(f"Val set: {len(val_ds)} stints")
    logger.info(f"Test set: {len(test_ds)} stints")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=config.data.shuffle_train,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=config.data.shuffle_val,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=config.data.shuffle_test,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def create_autoregressive_dataloaders(config: Config, batch_size: int = 32):
    """
    Create dataloaders for autoregressive lap-by-lap training.
    """
    logger.info("Creating autoregressive dataloaders...")

    train_ds = AutoregressiveLapDataloader(
        year=config.training.train_years,
        context_window=config.data.context_window,
        augment_prob=config.data.augment_prob,
        normalize=config.data.normalize,
        scaler_type=config.data.scaler_type,
    )

    shared_normalizer = train_ds.normalizer if config.data.normalize else None
    
    # DEBUG: Log normalizer statistics
    if shared_normalizer is not None:
        stats = shared_normalizer.get_statistics()
        lap_idx = stats['columns'].index('LapTime')
        logger.info(f"Train normalizer stats: mean={stats['mean'][lap_idx]:.2f}, std={stats['std'][lap_idx]:.2f}")

    val_ds = AutoregressiveLapDataloader(
        year=config.training.val_years,
        context_window=config.data.context_window,
        augment_prob=0.0,
        normalize=config.data.normalize,
        scaler_type=config.data.scaler_type,
        normalizer=shared_normalizer,
    )
    
    # DEBUG: Verify val dataset is using shared normalizer
    if val_ds.normalizer is not None:
        val_stats = val_ds.normalizer.get_statistics()
        val_lap_idx = val_stats['columns'].index('LapTime')
        logger.info(f"Val normalizer stats: mean={val_stats['mean'][val_lap_idx]:.2f}, std={val_stats['std'][val_lap_idx]:.2f}")
        if shared_normalizer is not None:
            is_same = (val_ds.normalizer is shared_normalizer)
            logger.info(f"Val normalizer is same object as train normalizer: {is_same}")

    test_ds = AutoregressiveLapDataloader(
        year=config.training.test_years,
        context_window=config.data.context_window,
        augment_prob=0.0,
        normalize=config.data.normalize,
        scaler_type=config.data.scaler_type,
        normalizer=shared_normalizer,
    )

    logger.info(f"Train set: {len(train_ds)} lap pairs")
    logger.info(f"Val set: {len(val_ds)} lap pairs")
    logger.info(f"Test set: {len(test_ds)} lap pairs")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=config.data.shuffle_train, num_workers=config.data.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=config.data.shuffle_val, num_workers=config.data.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=config.data.shuffle_test, num_workers=config.data.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def create_model(config: Config, device: str = 'cpu'):
    """
    Create model instance.
    
    Parameters
    ----------
    config : Config
        Configuration object
    device : str
        Device to use
    
    Returns
    -------
    nn.Module
        Model instance
    """
    logger.info("Creating model...")
    
    model_config = {
        'input_size': config.model.input_size,
        'output_size': config.model.output_size,
        'hidden_size': config.model.hidden_size,
        'num_layers': config.model.num_layers,
        'dropout': config.model.dropout,
        'embedding_dims': config.model.embedding_dims,
        'vocab_sizes': config.model.vocab_sizes,
        'decoder_type': getattr(config.model, 'decoder_type', 'gru'),
        'encoder_type': getattr(config.model, 'encoder', 'gru'),
        'device': device,
    }
    
    # If autoregressive dataloader is used, pass compound_classes to model
    try:
        compound_classes = len(get_compound_columns())
    except Exception:
        compound_classes = getattr(config.model, 'compound_classes', 4)
    model_config['compound_classes'] = compound_classes

    if config.model.name in ("seq2seq", "seq2seq_gru"):
        model = Seq2Seq(**model_config)
    else:
        raise ValueError(f"Unknown model: {config.model.name}")
    
    logger.info(model.model_info())
    return model


def teacher_forcing_schedule(epoch: int, config: Config) -> float:
    """
    Compute teacher forcing ratio for an epoch.
    
    Supports linear and exponential decay.
    
    Parameters
    ----------
    epoch : int
        Current epoch
    config : Config
        Configuration with teacher forcing settings
    
    Returns
    -------
    float
        Teacher forcing ratio for this epoch (0.0 to 1.0)
    """
    start = config.training.teacher_forcing_start
    end = config.training.teacher_forcing_end
    total_epochs = config.training.num_epochs
    
    if config.training.teacher_forcing_decay == "linear":
        # Linear decay from start to end
        ratio = start - (start - end) * (epoch / total_epochs)
    elif config.training.teacher_forcing_decay == "exponential":
        # Exponential decay
        decay = (end / start) ** (1.0 / total_epochs)
        ratio = start * (decay ** epoch)
    else:
        ratio = start  # Constant
    
    return ratio


def train(config: Config, output_dir: Path = None):
    """
    Run training pipeline.
    
    Parameters
    ----------
    config : Config
        Configuration object
    output_dir : Path, optional
        Directory to save results
    
    Returns
    -------
    dict
        Training history
    """
    if output_dir is None:
        output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for full reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info("=" * 60)
    logger.info("F1 LAP TIME PREDICTION - TRAINING")
    logger.info("=" * 60)
    logger.info(config)
    
    # Create dataloaders
    if getattr(config, 'use_autoregressive', False):
        train_loader, val_loader, test_loader = create_autoregressive_dataloaders(
            config,
            batch_size=config.training.batch_size
        )
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            config,
            batch_size=config.training.batch_size
        )

    # Sanity-check that dataset.normalizer.transform applied to raw rows matches
    # the normalized lap_time returned by the dataset's __getitem__ implementation.
    try:
        dataset = train_loader.dataset
        if hasattr(dataset, 'normalizer') and dataset.normalizer is not None:
            import numpy as _np
            import pandas as _pd

            numeric_cols = getattr(dataset, 'numeric_columns', ['LapTime'])
            max_checks = min(50, len(getattr(dataset, 'lap_pairs', [])))
            mismatches = []

            # Temporarily disable augmentation to get deterministic __getitem__ behavior
            orig_aug = getattr(dataset, 'augment_prob', 0.0)
            try:
                dataset.augment_prob = 0.0
                for i in range(max_checks):
                    pair = dataset.lap_pairs[i]
                    tidx = pair['target_index']
                    raw_row = dataset.data.loc[[tidx]]
                    # Ensure numeric columns exist in raw_row
                    raw_numeric = raw_row[numeric_cols]
                    try:
                        transformed = dataset.normalizer.transform(raw_numeric)
                        transformed_val = float(transformed['LapTime'].iloc[0])
                    except Exception:
                        transformed_val = float('nan')

                    try:
                        _, tgt, _ = dataset.__getitem__(i)
                        returned = float(tgt['lap_time'].cpu().numpy()) if hasattr(tgt['lap_time'], 'cpu') else float(tgt['lap_time'])
                    except Exception:
                        returned = float('nan')

                    if not (_np.isfinite(transformed_val) and _np.isfinite(returned) and abs(transformed_val - returned) < 1e-3):
                        mismatches.append((tidx, float(raw_row['LapTime'].iloc[0]), transformed_val, returned))

            finally:
                # Restore augmentation probability
                dataset.augment_prob = orig_aug

            if mismatches:
                logger.error(f"Normalizer consistency check failed: {len(mismatches)} mismatches (showing up to 20)")
                for row in mismatches[:20]:
                    logger.error(f"{row[0]}: raw={row[1]} -> transformed={row[2]} -> returned={row[3]}")
                raise RuntimeError("Dataset normalizer.transform does not match dataset.__getitem__ returned lap_time values. See logs for samples.")
            else:
                logger.info("Normalizer consistency check passed: transformed values match dataset returns on sample rows")
    except StopIteration:
        logger.warning("Could not sample a batch from train_loader for normalizer check")
    except Exception:
        logger.exception("Normalizer sanity check failed during consistency verification")
        raise
    
    # Create model
    model = create_model(config, device=config.device)
    
    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, {
        'scheduler_type': config.training.scheduler_type,
        'warm_up_epochs': config.training.warm_up_epochs,
        'total_epochs': config.training.num_epochs,
    })
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=config.device,
        scheduler=scheduler,
        checkpoint_dir=output_dir / "checkpoints",
        gradient_clip=config.training.gradient_clip,
        accumulation_steps=config.training.accumulation_steps,
        pit_loss_weight=getattr(config.training, 'pit_loss_weight', 0.5),
        compound_loss_weight=getattr(config.training, 'compound_loss_weight', 0.5),
        use_mixed_precision=getattr(config.training, 'use_mixed_precision', True),
        early_stopping_use_ema=getattr(config.training, 'early_stopping_use_ema', False),
        early_stopping_ema_alpha=getattr(config.training, 'early_stopping_ema_alpha', 0.3),
    )
    
    # Training loop
    logger.info("\nStarting training...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_min_epochs=getattr(config.training, 'early_stopping_min_epochs', 0),
        teacher_forcing_schedule=lambda epoch: teacher_forcing_schedule(epoch, config),
    )
    
    # Save config and history
    config.save(output_dir / "config.json")
    
    with open(output_dir / "history.json", 'w') as f:
        # Convert numpy to native Python types for JSON serialization
        history_json = {
            k: [float(v) for v in vs] if isinstance(vs, (list, np.ndarray)) else vs
            for k, vs in history.items()
        }
        json.dump(history_json, f, indent=2)
    
    logger.info(f"\nTraining completed! Results saved to {output_dir}")
    
    return history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train F1 lap time prediction model")
    parser.add_argument('--phase', type=int, choices=[1, 2], help='Use preset Phase N config')
    parser.add_argument('--config', type=Path, help='Path to config JSON file')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--output', type=Path, help='Output directory for results')
    parser.add_argument('--autoregressive', action='store_true', help='Use autoregressive lap-by-lap dataloader')
    parser.add_argument('--decoder-type', choices=['gru', 'lstm'], help='Decoder type to use (overrides config)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--no-mixed-precision', action='store_true', help='Disable mixed precision training (FP16)')
    parser.add_argument('--seed', type=int, help='Random seed (overrides config)')
    parser.add_argument('--pit-loss-weight', type=float, help='Pit stop auxiliary loss weight (overrides config)')
    parser.add_argument('--compound-loss-weight', type=float, help='Compound auxiliary loss weight (overrides config)')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = Config.load(args.config)
        logger.info(f"Loaded config from {args.config}")
    elif args.phase == 1:
        config = get_phase1_config()
        logger.info("Using Phase 1 preset config")
    elif args.phase == 2:
        config = get_phase2_config()
        logger.info("Using Phase 2 preset config")
    elif args.autoregressive:
        config = get_phase2_config()
        logger.info("Using default Phase 2 config for autoregressive training")
    else:
        config = get_phase1_config()
        logger.info("Using default Phase 1 config")
    
    # Override config with command line arguments
    if args.device:
        config.device = args.device
    if getattr(args, 'decoder_type', None):
        # Attach decoder_type to model config dynamically
        try:
            config.model.decoder_type = args.decoder_type
        except Exception:
            setattr(config.model, 'decoder_type', args.decoder_type)

    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.no_mixed_precision:
        config.training.use_mixed_precision = False
    if args.seed is not None:
        config.seed = args.seed
    if args.pit_loss_weight is not None:
        config.training.pit_loss_weight = args.pit_loss_weight
    if args.compound_loss_weight is not None:
        config.training.compound_loss_weight = args.compound_loss_weight
    
    output_dir = args.output or Path(config.output_dir)
    
    # Run training
    # Choose dataloader mode (stint vs autoregressive) inside `train()` by flag
    if args.autoregressive:
        # attach a marker to config so train() can choose
        setattr(config, 'use_autoregressive', True)
    history = train(config, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {min(history['val_loss']):.6f}")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.6f}")
    logger.info(f"Total epochs trained: {len(history['train_loss'])}")

    # Post-training: run evaluation using the best checkpoint then analysis
    try:
        checkpoint_path = (output_dir / "checkpoints" / "best_model.pt")
        if checkpoint_path.exists():
            logger.info(f"Running evaluation on checkpoint: {checkpoint_path}")
            run_evaluation(
                checkpoint_path=checkpoint_path,
                test_years=config.training.test_years,
                config_path=output_dir / "config.json",
                device=config.device,
                output_dir=output_dir / "evaluation",
            )
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}. Skipping evaluation.")
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")

    # Run post-training analysis (call function directly)
    try:
        logger.info("Running post-training analysis...")
        # call with explicit results_dir to avoid relying on run name
        run_analysis(run=output_dir.name, results_dir=output_dir)
    except Exception as e:
        logger.exception(f"Post-training analysis failed: {e}")


if __name__ == "__main__":

    main()

