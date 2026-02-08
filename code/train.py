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

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataloaders import StintDataloader, LapTimeNormalizer
from models import Seq2SeqGRU, Trainer, create_scheduler
from config.base import Config, get_phase1_config, get_phase2_config

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
    
    # Create datasets for each split
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
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=config.data.shuffle_val,
        num_workers=config.data.num_workers,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=config.data.shuffle_test,
        num_workers=config.data.num_workers,
    )
    
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
        'device': device,
    }
    
    if config.model.name == "seq2seq_gru":
        model = Seq2SeqGRU(**model_config)
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
    
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    logger.info("=" * 60)
    logger.info("F1 LAP TIME PREDICTION - TRAINING")
    logger.info("=" * 60)
    logger.info(config)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        batch_size=config.training.batch_size
    )
    
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
    )
    
    # Training loop
    logger.info("\nStarting training...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
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
    parser.add_argument('--postprocess', action='store_true', help='Run evaluation and analysis after training')
    parser.add_argument('--output', type=Path, help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    
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
    else:
        config = get_phase1_config()
        logger.info("Using default Phase 1 config")
    
    # Override config with command line arguments
    if args.device:
        config.device = args.device
    # Postprocessing flag: if provided, request evaluation+analysis after training
    if getattr(args, 'postprocess', False):
        try:
            config.training.run_postprocessing = True
        except Exception:
            # Ensure training namespace exists
            if not hasattr(config, 'training'):
                config.training = type('T', (), {})()
            config.training.run_postprocessing = True
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    output_dir = args.output or Path(config.output_dir)
    
    # Run training
    history = train(config, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {min(history['val_loss']):.6f}")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.6f}")
    logger.info(f"Total epochs trained: {len(history['train_loss'])}")


if __name__ == "__main__":
    # Optionally run postprocessing (evaluation + analysis) if requested
    args = None
    try:
        # Reparse to check for postprocess flag
        import sys as _sys
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--postprocess', action='store_true')
        args, _ = parser.parse_known_args(_sys.argv[1:])
    except Exception:
        args = None

    main()

    # If postprocess requested, call the wrapper in code/postprocess.py
    try:
        if args and getattr(args, 'postprocess', False):
            from postprocess import run_postprocessing
            from pathlib import Path as _Path
            # Best checkpoint location
            cp = _Path(config.output_dir) / 'checkpoints' / 'best_model.pt'
            # If main didn't expose `config`, try loading saved config
            try:
                run_postprocessing(cp, config_path=_Path(config.output_dir) / 'config.json', device=config.device, output_dir=_Path(config.output_dir))
            except Exception:
                # Fallback: use output_dir and device derived from arguments
                run_postprocessing(cp, config_path=_Path(config.output_dir) / 'config.json', device=config.device, output_dir=_Path(config.output_dir))
    except Exception:
        pass
