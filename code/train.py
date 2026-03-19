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
from torch.utils.data import DataLoader
from analyze_results import analyze_results as run_analysis

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataloaders import StintDataloader
from dataloaders.autoregressive_dataloader import AutoregressiveLapDataloader, AutoregressiveRolloutDataset
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


def _set_global_seed(seed: int):
    """Set random seeds for deterministic training behavior."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_dataset_normalizer_consistency_check(train_loader: DataLoader):
    """Validate that dataset normalizer.transform matches __getitem__ lap_time output."""
    try:
        dataset = train_loader.dataset
        if not (hasattr(dataset, 'normalizer') and dataset.normalizer is not None):
            return

        import numpy as _np

        numeric_cols = getattr(dataset, 'numeric_columns', ['LapTime'])
        max_checks = min(50, len(getattr(dataset, 'lap_pairs', [])))
        mismatches = []

        # Temporarily disable augmentation to keep __getitem__ deterministic.
        orig_aug = getattr(dataset, 'augment_prob', 0.0)
        try:
            dataset.augment_prob = 0.0
            for i in range(max_checks):
                pair = dataset.lap_pairs[i]
                tidx = pair['target_index']
                raw_row = dataset.data.loc[[tidx]]
                raw_numeric = raw_row[numeric_cols]

                try:
                    transformed = dataset.normalizer.transform(raw_numeric)
                    transformed_val = float(transformed['LapTime'].iloc[0])
                except Exception:
                    transformed_val = float('nan')

                try:
                    _, tgt, _ = dataset.__getitem__(i)
                    lap_val = tgt['lap_time']
                    returned = float(lap_val.cpu().numpy()) if hasattr(lap_val, 'cpu') else float(lap_val)
                except Exception:
                    returned = float('nan')

                if not (_np.isfinite(transformed_val) and _np.isfinite(returned) and abs(transformed_val - returned) < 1e-3):
                    mismatches.append((tidx, float(raw_row['LapTime'].iloc[0]), transformed_val, returned))
        finally:
            dataset.augment_prob = orig_aug

        if mismatches:
            logger.error(f"Normalizer consistency check failed: {len(mismatches)} mismatches (showing up to 20)")
            for row in mismatches[:20]:
                logger.error(f"{row[0]}: raw={row[1]} -> transformed={row[2]} -> returned={row[3]}")
            raise RuntimeError(
                "Dataset normalizer.transform does not match dataset.__getitem__ returned lap_time values. "
                "See logs for samples."
            )
        logger.info("Normalizer consistency check passed: transformed values match dataset returns on sample rows")
    except StopIteration:
        logger.warning("Could not sample a batch from train_loader for normalizer check")
    except Exception:
        logger.exception("Normalizer sanity check failed during consistency verification")
        raise


def _apply_cli_overrides(config: Config, args: argparse.Namespace):
    """Apply CLI-provided configuration overrides in one place."""
    if getattr(args, 'autoregressive', False):
        config.training.rollout_training = True

    if args.device:
        config.device = args.device

    if getattr(args, 'decoder_type', None):
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
    if args.augment_prob is not None:
        if args.augment_prob < 0.0 or args.augment_prob > 1.0:
            raise ValueError(f"--augment-prob must be between 0.0 and 1.0, got {args.augment_prob}")
        config.data.augment_prob = args.augment_prob
    if args.no_mixed_precision:
        config.training.use_mixed_precision = False
    if args.seed is not None:
        config.seed = args.seed
    if args.pit_loss_weight is not None:
        config.training.pit_loss_weight = args.pit_loss_weight
    if args.compound_loss_weight is not None:
        config.training.compound_loss_weight = args.compound_loss_weight
    if getattr(args, 'disable_aux_scaling', False):
        config.training.dynamic_aux_balance = False
    # Teacher forcing CLI overrides
    if getattr(args, 'teacher_forcing_decay', None):
        config.training.teacher_forcing_decay = args.teacher_forcing_decay
    if getattr(args, 'teacher_forcing_start', None) is not None:
        if not (0.0 <= args.teacher_forcing_start <= 1.0):
            raise ValueError("--teacher-forcing-start must be between 0.0 and 1.0")
        config.training.teacher_forcing_start = args.teacher_forcing_start
    if getattr(args, 'teacher_forcing_end', None) is not None:
        if not (0.0 <= args.teacher_forcing_end <= 1.0):
            raise ValueError("--teacher-forcing-end must be between 0.0 and 1.0")
        config.training.teacher_forcing_end = args.teacher_forcing_end
    if getattr(args, 'teacher_forcing_hold_epochs', None) is not None:
        if args.teacher_forcing_hold_epochs < 0:
            raise ValueError("--teacher-forcing-hold-epochs must be >= 0")
        config.training.teacher_forcing_hold_epochs = args.teacher_forcing_hold_epochs
    if getattr(args, 'rollout_steps', None) is not None:
        config.training.rollout_steps = args.rollout_steps
    if getattr(args, 'rollout_weight', None) is not None:
        config.training.rollout_weight = args.rollout_weight
    if getattr(args, 'rollout_start_epoch', None) is not None:
        config.training.rollout_start_epoch = args.rollout_start_epoch
    # Rollout scheduled sampling / curriculum overrides
    if getattr(args, 'rollout_tf_start', None) is not None:
        if not (0.0 <= args.rollout_tf_start <= 1.0):
            raise ValueError("--rollout-tf-start must be between 0.0 and 1.0")
        config.training.rollout_teacher_forcing_start = args.rollout_tf_start
    if getattr(args, 'rollout_tf_end', None) is not None:
        if not (0.0 <= args.rollout_tf_end <= 1.0):
            raise ValueError("--rollout-tf-end must be between 0.0 and 1.0")
        config.training.rollout_teacher_forcing_end = args.rollout_tf_end
    if getattr(args, 'rollout_warmup_epochs', None) is not None:
        if args.rollout_warmup_epochs < 0:
            raise ValueError("--rollout-warmup-epochs must be >= 0")
        config.training.rollout_warmup_epochs = args.rollout_warmup_epochs
    if getattr(args, 'no_rollout_curriculum', False):
        config.training.rollout_curriculum = False


def _run_post_training_steps(config: Config, output_dir: Path):
    """Run evaluation and analysis after training finishes."""
    try:
        checkpoint_path = output_dir / "checkpoints" / "best_model.pt"
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
    except Exception as exc:
        logger.exception(f"Evaluation failed: {exc}")

    try:
        logger.info("Running post-training analysis...")
        run_analysis(run=output_dir.name, results_dir=output_dir)
    except Exception as exc:
        logger.exception(f"Post-training analysis failed: {exc}")


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
        'decoder_future_features': getattr(config.model, 'decoder_future_features', 0),
        'use_attention': getattr(config.model, 'use_attention', False),
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
    
    Supports constant, linear, exponential, and hold-then-decay schedules.
    
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

    decay_type = getattr(config.training, 'teacher_forcing_decay', 'linear')
    hold_epochs = getattr(config.training, 'teacher_forcing_hold_epochs', 0)

    # If hold_epochs covers the whole schedule, keep start constant
    if hold_epochs >= total_epochs:
        return start

    # If we're still in the hold period, return start
    if epoch < hold_epochs:
        return start

    # Compute effective epoch/count after hold
    eff_epoch = epoch - hold_epochs
    eff_total = max(1, total_epochs - hold_epochs)

    if decay_type in ("linear", "hold_then_decay"):
        # Linear decay from start to end over effective total
        ratio = start - (start - end) * (eff_epoch / eff_total)
    elif decay_type == "exponential":
        # Exponential decay over effective total.
        # Guard against end=0 which makes the base 0 and collapses the ratio
        # to 0 on the very first step.  Use a small epsilon as a floor for the
        # base calculation so the curve decays smoothly; the result is then
        # clamped down to the true `end` value.
        if start <= 0.0:
            ratio = end
        else:
            end_eff = max(end, 1e-6)
            decay = (end_eff / start) ** (1.0 / eff_total)
            ratio = max(start * (decay ** eff_epoch), end)
    else:
        # constant or unknown: return start
        ratio = start
    
    return ratio


def rollout_teacher_forcing_schedule(epoch: int, config: Config) -> float:
    """
    Compute the teacher-forcing ratio applied *inside* the rollout decoder
    (scheduled sampling) for a given epoch.

    At epoch ``rollout_start_epoch`` the ratio equals
    ``rollout_teacher_forcing_start`` (default 1.0 = fully teacher-forced).
    It then decays via a cosine schedule to ``rollout_teacher_forcing_end``
    (default 0.0 = fully autoregressive) over ``rollout_warmup_epochs`` epochs.
    The cosine schedule stays near ``tf_start`` longer and drops steeply only
    toward the end, giving the model more time to adapt at low TF ratios.  After the
    warm-up period the end value is held constant.

    If rollout has not started yet (epoch < rollout_start_epoch), 0.0 is
    returned (not used because rollout is inactive).

    Parameters
    ----------
    epoch : int
        Current epoch.
    config : Config
        Training configuration.

    Returns
    -------
    float
        Teacher-forcing ratio for rollout at this epoch (0.0–1.0).
    """
    start_epoch = getattr(config.training, 'rollout_start_epoch', 0)
    tf_start = float(getattr(config.training, 'rollout_teacher_forcing_start', 1.0))
    tf_end = float(getattr(config.training, 'rollout_teacher_forcing_end', 0.0))
    warmup = int(getattr(config.training, 'rollout_warmup_epochs', 20))

    if epoch < start_epoch:
        return 0.0

    eff_epoch = epoch - start_epoch
    if warmup <= 0 or eff_epoch >= warmup:
        return tf_end

    # Cosine decay from tf_start to tf_end over warmup epochs.
    # Stays near tf_start longer and drops steeply only at the end,
    # giving the model more time to adapt at low teacher-forcing ratios.
    cosine_factor = 0.5 * (1.0 + np.cos(np.pi * eff_epoch / warmup))
    ratio = tf_end + (tf_start - tf_end) * cosine_factor
    return float(np.clip(ratio, min(tf_start, tf_end), max(tf_start, tf_end)))


def rollout_curriculum_steps(epoch: int, config: Config) -> int:
    """
    Compute the active rollout horizon (number of future steps to unroll) for a
    given epoch, implementing curriculum learning.

    The horizon starts at 1 step at ``rollout_start_epoch`` and grows linearly
    to ``rollout_steps`` over ``rollout_warmup_epochs`` epochs.  If
    ``rollout_curriculum`` is ``False`` the full ``rollout_steps`` is always
    returned.

    Parameters
    ----------
    epoch : int
        Current epoch.
    config : Config
        Training configuration.

    Returns
    -------
    int
        Number of rollout steps to use this epoch.
    """
    use_curriculum = bool(getattr(config.training, 'rollout_curriculum', True))
    max_steps = int(getattr(config.training, 'rollout_steps', 5))

    if not use_curriculum:
        return max_steps

    start_epoch = getattr(config.training, 'rollout_start_epoch', 0)
    warmup = int(getattr(config.training, 'rollout_warmup_epochs', 20))

    if epoch < start_epoch:
        return 1

    eff_epoch = epoch - start_epoch
    if warmup <= 0 or eff_epoch >= warmup:
        return max_steps

    # Linearly grow from 1 to max_steps over warmup epochs
    steps = 1 + (max_steps - 1) * (eff_epoch / warmup)
    return max(1, min(max_steps, int(np.ceil(steps))))


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

    if getattr(config, 'use_autoregressive', False):
        config.training.rollout_training = True
    
    # Set random seed for full reproducibility
    _set_global_seed(config.seed)
    
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

    _run_dataset_normalizer_consistency_check(train_loader)
    
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
        pit_loss_weight=getattr(config.training, 'pit_loss_weight', 1e-3),
        compound_loss_weight=getattr(config.training, 'compound_loss_weight', 0.01),
        use_mixed_precision=getattr(config.training, 'use_mixed_precision', True),
        early_stopping_use_ema=getattr(config.training, 'early_stopping_use_ema', False),
        early_stopping_ema_alpha=getattr(config.training, 'early_stopping_ema_alpha', 0.3),
        dynamic_aux_balance=getattr(config.training, 'dynamic_aux_balance', True),
        dynamic_aux_ema_alpha=getattr(config.training, 'dynamic_aux_ema_alpha', 0.05),
        dynamic_aux_min_scale=getattr(config.training, 'dynamic_aux_min_scale', 0.001),
        dynamic_aux_max_scale=getattr(config.training, 'dynamic_aux_max_scale', 20.0),
    )
    
    # Training loop
    logger.info("\nStarting training...")

    # Rollout training is always enabled in autoregressive mode.
    rollout_loader = None
    rollout_val_loader = None
    if getattr(config, 'use_autoregressive', False):
        rollout_steps = getattr(config.training, 'rollout_steps', 5)
        logger.info(f"Creating rollout dataset with {rollout_steps} steps...")
        rollout_train_ds = AutoregressiveRolloutDataset(
            base_dataset=train_loader.dataset,
            rollout_steps=rollout_steps,
        )
        rollout_loader = DataLoader(
            rollout_train_ds,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
        )

        # Rollout validation dataset (uses val split, no shuffle)
        if val_loader is not None:
            rollout_val_ds = AutoregressiveRolloutDataset(
                base_dataset=val_loader.dataset,
                rollout_steps=rollout_steps,
            )
            if len(rollout_val_ds) > 0:
                rollout_val_loader = DataLoader(
                    rollout_val_ds,
                    batch_size=config.training.batch_size,
                    shuffle=False,
                    num_workers=config.data.num_workers,
                    pin_memory=True,
                )
                logger.info(f"Rollout val dataset: {len(rollout_val_ds)} sequences")
            else:
                logger.warning("Rollout val dataset is empty — rollout validation skipped")

    _rollout_warmup = int(getattr(config.training, 'rollout_warmup_epochs', 40))
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_min_epochs=getattr(config.training, 'early_stopping_min_epochs', 0),
        teacher_forcing_schedule=lambda epoch: teacher_forcing_schedule(epoch, config),
        rollout_loader=rollout_loader,
        rollout_val_loader=rollout_val_loader,
        rollout_weight=getattr(config.training, 'rollout_weight', 1.0),
        rollout_start_epoch=getattr(config.training, 'rollout_start_epoch', 0),
        rollout_teacher_forcing_schedule=lambda epoch: rollout_teacher_forcing_schedule(epoch, config),
        rollout_curriculum_steps_schedule=lambda epoch: rollout_curriculum_steps(epoch, config),
        early_stopping_metric=getattr(config.training, 'early_stopping_metric', 'val_loss'),
        disable_single_step_after_warmup=getattr(config.training, 'disable_single_step_after_warmup', False),
        rollout_warmup_epochs=_rollout_warmup,
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
    parser.add_argument('--autoregressive', action='store_true', help='Use autoregressive lap-by-lap dataloader and enable rollout training')
    parser.add_argument('--decoder-type', choices=['gru', 'lstm'], help='Decoder type to use (overrides config)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--augment-prob', type=float, help='Training data augmentation probability (overrides config.data.augment_prob)')
    parser.add_argument('--no-mixed-precision', action='store_true', help='Disable mixed precision training (FP16)')
    parser.add_argument('--seed', type=int, help='Random seed (overrides config)')
    parser.add_argument('--pit-loss-weight', type=float, help='Pit stop auxiliary loss weight (overrides config)')
    parser.add_argument('--compound-loss-weight', type=float, help='Compound auxiliary loss weight (overrides config)')
    parser.add_argument('--disable-aux-scaling', action='store_true', help='Disable dynamic auxiliary loss scaling (pit/compound)')
    # Teacher forcing CLI options
    parser.add_argument('--teacher-forcing-decay', choices=['linear', 'exponential', 'hold_then_decay', 'constant'], help='Teacher forcing decay type (overrides config.training.teacher_forcing_decay)')
    parser.add_argument('--teacher-forcing-start', type=float, help='Start teacher forcing ratio (0.0-1.0)')
    parser.add_argument('--teacher-forcing-end', type=float, help='End teacher forcing ratio (0.0-1.0)')
    parser.add_argument('--teacher-forcing-hold-epochs', type=int, help='Number of epochs to hold start ratio before decaying (used with hold_then_decay)')
    parser.add_argument('--rollout-steps', type=int, help='Number of autoregressive rollout steps per sample (default: 5)')
    parser.add_argument('--rollout-weight', type=float, help='Weight multiplier for rollout loss (default: 1.0)')
    parser.add_argument('--rollout-start-epoch', type=int, help='Epoch at which to start rollout training (default: 0)')
    # Rollout scheduled sampling / curriculum options
    parser.add_argument(
        '--rollout-tf-start', type=float,
        help='Initial teacher-forcing ratio inside rollout decoder (scheduled sampling start). '
             'Default: 1.0 (fully teacher-forced at rollout start, smoothly bridges to autoregressive).',
    )
    parser.add_argument(
        '--rollout-tf-end', type=float,
        help='Final teacher-forcing ratio inside rollout decoder. Default: 0.0 (fully autoregressive).',
    )
    parser.add_argument(
        '--rollout-warmup-epochs', type=int,
        help='Number of epochs (measured from rollout_start_epoch) over which the rollout '
             'teacher-forcing ratio decays and the curriculum horizon grows. Default: 20.',
    )
    parser.add_argument(
        '--no-rollout-curriculum', action='store_true',
        help='Disable curriculum rollout (always use full rollout_steps from the first rollout epoch).',
    )
    
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
    
    _apply_cli_overrides(config, args)
    
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

    _run_post_training_steps(config, output_dir)


if __name__ == "__main__":

    main()

