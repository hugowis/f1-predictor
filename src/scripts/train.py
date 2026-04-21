"""
Training script for F1 lap time prediction models.

Usage:
    python scripts/train.py --config config/phase1.json --device cuda
    python scripts/train.py --phase 1  # Use preset Phase 1 config
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

# Switch multiprocessing to file_system sharing so DataLoader workers don't
# exhaust the forkserver's unix-socket fd budget when multiple training
# processes run in parallel against the big precomputed AR cache.
torch.multiprocessing.set_sharing_strategy('file_system')
from analyze_results import analyze_results as run_analysis

# Add f1predictor library to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "f1predictor"))

from dataloaders import StintDataloader
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
                    # Multi-step targets return (H,) tensor; check first element only
                    if hasattr(lap_val, 'dim') and lap_val.dim() >= 1:
                        lap_val = lap_val[0]
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
    if args.device:
        config.device = args.device

    if getattr(args, 'decoder_type', None):
        try:
            config.model.decoder_type = args.decoder_type
        except Exception:
            setattr(config.model, 'decoder_type', args.decoder_type)

    # Decoder head overrides (B1)
    if getattr(args, 'no_mlp_decoder_head', False):
        config.model.use_mlp_decoder_head = False
    if getattr(args, 'decoder_head_hidden', None) is not None:
        if args.decoder_head_hidden < 1:
            raise ValueError("--decoder-head-hidden must be >= 1")
        config.model.decoder_head_hidden = args.decoder_head_hidden
    if getattr(args, 'decoder_head_layers', None) is not None:
        if args.decoder_head_layers < 1:
            raise ValueError("--decoder-head-layers must be >= 1")
        config.model.decoder_head_layers = args.decoder_head_layers
    if getattr(args, 'decoder_head_dropout', None) is not None:
        if not (0.0 <= args.decoder_head_dropout < 1.0):
            raise ValueError("--decoder-head-dropout must be in [0.0, 1.0)")
        config.model.decoder_head_dropout = args.decoder_head_dropout
    if getattr(args, 'decoder_skip_type', None):
        config.model.decoder_skip_type = args.decoder_skip_type
    if getattr(args, 'encoder_pool', False):
        config.model.encoder_pool = True
    if getattr(args, 'encoder_pool_k', None) is not None:
        if args.encoder_pool_k < 1:
            raise ValueError("--encoder-pool-k must be >= 1")
        config.model.encoder_pool_k = args.encoder_pool_k

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

    # Multi-step autoregressive training overrides
    if getattr(args, 'multistep_horizon', None) is not None:
        if args.multistep_horizon < 1:
            raise ValueError("--multistep-horizon must be >= 1")
        config.training.multistep_horizon = args.multistep_horizon
    if getattr(args, 'multistep_curriculum', None):
        config.training.multistep_curriculum = args.multistep_curriculum
    if getattr(args, 'multistep_curriculum_start_epoch', None) is not None:
        config.training.multistep_curriculum_start_epoch = args.multistep_curriculum_start_epoch
    if getattr(args, 'multistep_curriculum_end_epoch', None) is not None:
        config.training.multistep_curriculum_end_epoch = args.multistep_curriculum_end_epoch

    # Scheduled sampling overrides
    if getattr(args, 'scheduled_sampling', False):
        config.training.scheduled_sampling_enabled = True
    if getattr(args, 'ss_max_prob', None) is not None:
        config.training.scheduled_sampling_max_prob = args.ss_max_prob
    if getattr(args, 'ss_noise_std', None) is not None:
        config.training.scheduled_sampling_noise_std = args.ss_noise_std
    if getattr(args, 'ss_start_epoch', None) is not None:
        config.training.scheduled_sampling_start_epoch = args.ss_start_epoch
    if getattr(args, 'ss_end_epoch', None) is not None:
        config.training.scheduled_sampling_end_epoch = args.ss_end_epoch

    # Data loading overrides
    if getattr(args, 'context_window', None) is not None:
        if args.context_window < 1:
            raise ValueError("--context-window must be >= 1")
        config.data.context_window = args.context_window
    if getattr(args, 'num_workers', None) is not None:
        config.data.num_workers = args.num_workers

    # Validation frequency override
    if getattr(args, 'validation_freq', None) is not None:
        if args.validation_freq < 1:
            raise ValueError("--validation-freq must be >= 1")
        config.training.validation_freq = args.validation_freq

    # LR scaling override
    if getattr(args, 'no_lr_scaling', False):
        config.training.lr_scale_with_batch = False


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
        require_normalizer=config.data.normalize,
    )

    test_ds = StintDataloader(
        year=config.training.test_years,
        window_size=config.data.window_size,
        augment_prob=0.0,  # No augmentation on test
        normalize=config.data.normalize,
        scaler_type=config.data.scaler_type,
        normalizer=shared_normalizer,
        require_normalizer=config.data.normalize,
    )
    
    logger.info(f"Train set: {len(train_ds)} stints")
    logger.info(f"Val set: {len(val_ds)} stints")
    logger.info(f"Test set: {len(test_ds)} stints")
    
    # Create dataloaders
    nw = config.data.num_workers
    loader_kwargs: dict = dict(pin_memory=True, num_workers=nw)
    if nw > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=config.data.shuffle_train,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=config.data.shuffle_val,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=config.data.shuffle_test,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


def create_autoregressive_dataloaders(config: Config, batch_size: int = 32):
    """
    Create dataloaders for autoregressive lap-by-lap training.
    """
    logger.info("Creating autoregressive dataloaders...")

    ms_horizon = getattr(config.training, 'multistep_horizon', 1)
    train_ds = AutoregressiveLapDataloader(
        year=config.training.train_years,
        context_window=config.data.context_window,
        multistep_horizon=ms_horizon,
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
        multistep_horizon=ms_horizon,
        augment_prob=0.0,
        normalize=config.data.normalize,
        scaler_type=config.data.scaler_type,
        normalizer=shared_normalizer,
        require_normalizer=config.data.normalize,
    )

    test_ds = AutoregressiveLapDataloader(
        year=config.training.test_years,
        context_window=config.data.context_window,
        multistep_horizon=ms_horizon,
        augment_prob=0.0,
        normalize=config.data.normalize,
        scaler_type=config.data.scaler_type,
        normalizer=shared_normalizer,
        require_normalizer=config.data.normalize,
    )

    logger.info(f"Train set: {len(train_ds)} lap pairs")
    logger.info(f"Val set: {len(val_ds)} lap pairs")
    logger.info(f"Test set: {len(test_ds)} lap pairs")

    nw = config.data.num_workers
    loader_kwargs: dict = dict(pin_memory=True, num_workers=nw)
    if nw > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=config.data.shuffle_train,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=config.data.shuffle_val,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=config.data.shuffle_test,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


def create_model(config: Config, device: str = 'cpu', use_autoregressive: bool = False):
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
    
    # Ensure decoder_extra_features_size matches the training mode.  Phase-1
    # (stint) uses 0; phase-2 / autoregressive uses 11 (post-Part C).  If the
    # config was created from a phase-1 preset but the user passed
    # --autoregressive, bump the value so the model is built correctly and so
    # the resolved config persisted to config.json reflects the actual model
    # architecture. Legacy 7-extra checkpoints still roll out via
    # rollout_evaluator's slice-to-model-size guard.
    if use_autoregressive and config.model.decoder_extra_features_size == 0:
        config.model.decoder_extra_features_size = 11
    elif not use_autoregressive:
        config.model.decoder_extra_features_size = 0

    # Auto-derive encoder input_size from the feature registry so schema
    # changes (e.g. Part C) don't silently break training.  Warn loudly if
    # the config value disagrees with the registry — that's the classic
    # "Expected 103, got 119" mismatch at the encoder GRU.
    from dataloaders.utils import (
        get_numeric_columns, get_categorical_columns,
        get_boolean_columns, get_compound_columns,
    )
    derived_input_size = (
        len(get_numeric_columns()) + len(get_categorical_columns())
        + len(get_boolean_columns()) + len(get_compound_columns())
    )
    if config.model.input_size != derived_input_size:
        logger.warning(
            f"ModelConfig.input_size={config.model.input_size} disagrees with "
            f"feature-registry count {derived_input_size}; overriding to "
            f"{derived_input_size} to match the AR cache."
        )
        config.model.input_size = derived_input_size

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
        'decoder_extra_features_size': config.model.decoder_extra_features_size,
        'use_mlp_decoder_head': getattr(config.model, 'use_mlp_decoder_head', True),
        'decoder_head_hidden': getattr(config.model, 'decoder_head_hidden', 64),
        'decoder_head_layers': getattr(config.model, 'decoder_head_layers', 2),
        'decoder_head_dropout': getattr(config.model, 'decoder_head_dropout', 0.0),
        'decoder_skip_type': getattr(config.model, 'decoder_skip_type', 'none'),
        'encoder_pool': getattr(config.model, 'encoder_pool', False),
        'encoder_pool_k': getattr(config.model, 'encoder_pool_k', 5),
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
        # Linear decay from start to end over effective total.
        # For hold_then_decay the hold period is already consumed above;
        # what remains is a standard linear decay from start to end.
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


def multistep_horizon_schedule(epoch: int, config: Config) -> int:
    """
    Compute the multi-step prediction horizon H for an epoch.

    Supports:
    - 'none': always use max horizon from config
    - 'linear': H grows linearly from 1 to max over the curriculum window

    Parameters
    ----------
    epoch : int
        Current epoch
    config : Config
        Configuration with multistep settings

    Returns
    -------
    int
        Prediction horizon H for this epoch (>= 1)
    """
    max_h = getattr(config.training, 'multistep_horizon', 1)
    if max_h <= 1:
        return 1

    curriculum = getattr(config.training, 'multistep_curriculum', 'none')
    if curriculum == 'none':
        return max_h

    start_epoch = getattr(config.training, 'multistep_curriculum_start_epoch', 0)
    end_epoch = getattr(config.training, 'multistep_curriculum_end_epoch', -1)
    if end_epoch < 0:
        end_epoch = config.training.num_epochs

    if epoch < start_epoch:
        return 1

    if curriculum == 'linear':
        progress = min(1.0, (epoch - start_epoch) / max(1, end_epoch - start_epoch))
        return max(1, int(1 + (max_h - 1) * progress))

    # Unknown curriculum type: fall back to max
    return max_h


def scheduled_sampling_schedule(epoch: int, config: Config) -> float:
    """
    Compute the scheduled sampling probability for an epoch.

    Returns the probability of corrupting each context lap's LapTime with
    Gaussian noise during training. Linearly ramps from 0 to max_prob
    between start_epoch and end_epoch.

    Parameters
    ----------
    epoch : int
        Current epoch
    config : Config
        Configuration with scheduled_sampling settings

    Returns
    -------
    float
        Corruption probability for this epoch (0.0 to max_prob)
    """
    if not getattr(config.training, 'scheduled_sampling_enabled', False):
        return 0.0

    max_prob = getattr(config.training, 'scheduled_sampling_max_prob', 0.5)
    start_epoch = getattr(config.training, 'scheduled_sampling_start_epoch', 10)
    end_epoch = getattr(config.training, 'scheduled_sampling_end_epoch', -1)
    if end_epoch < 0:
        end_epoch = config.training.num_epochs

    if epoch < start_epoch:
        return 0.0

    if epoch >= end_epoch:
        return max_prob

    progress = (epoch - start_epoch) / max(1, end_epoch - start_epoch)
    return max_prob * progress


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
    model = create_model(config, device=config.device, use_autoregressive=getattr(config, 'use_autoregressive', False))
    
    # Apply sqrt LR scaling when batch size differs from reference
    # (sqrt scaling is more stable for RNNs than linear scaling)
    if config.training.lr_scale_with_batch:
        ref_bs = config.training.lr_scale_reference_batch
        actual_bs = config.training.batch_size
        if actual_bs != ref_bs:
            scale = (actual_bs / ref_bs) ** 0.5
            old_lr = config.training.learning_rate
            config.training.learning_rate = old_lr * scale
            # Scale warmup UP for larger batches (larger LR needs more warmup)
            base_warmup = config.training.warm_up_epochs
            config.training.warm_up_epochs = max(base_warmup, int(base_warmup * scale))
            logger.info(
                f"Sqrt LR scaling: {old_lr:.2e} * {scale:.2f} = {config.training.learning_rate:.2e} "
                f"(batch_size={actual_bs}, ref={ref_bs}), "
                f"warmup={config.training.warm_up_epochs} epochs"
            )

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
    # Log scheduled sampling settings
    ss_enabled = getattr(config.training, 'scheduled_sampling_enabled', False)
    if ss_enabled:
        logger.info(f"Scheduled sampling ENABLED: max_prob={config.training.scheduled_sampling_max_prob}, "
                     f"noise_std={config.training.scheduled_sampling_noise_std}, "
                     f"start_epoch={config.training.scheduled_sampling_start_epoch}")

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_min_epochs=getattr(config.training, 'early_stopping_min_epochs', 0),
        validation_freq=getattr(config.training, 'validation_freq', 1),
        teacher_forcing_schedule=lambda epoch: teacher_forcing_schedule(epoch, config),
        multistep_horizon_schedule=lambda epoch: multistep_horizon_schedule(epoch, config),
        scheduled_sampling_schedule=lambda epoch: scheduled_sampling_schedule(epoch, config),
        scheduled_sampling_noise_std=getattr(config.training, 'scheduled_sampling_noise_std', 0.02),
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
    # Decoder output head (B1)
    parser.add_argument('--no-mlp-decoder-head', action='store_true', help='Use single Linear decoder head instead of MLP (legacy)')
    parser.add_argument('--decoder-head-hidden', type=int, help='MLP decoder head hidden width (default: 64)')
    parser.add_argument('--decoder-head-layers', type=int, help='MLP decoder head number of Linear layers (1 = single Linear, default: 2)')
    parser.add_argument('--decoder-head-dropout', type=float, help='Dropout between MLP decoder head layers (default: 0.0)')
    parser.add_argument('--decoder-skip-type', choices=['none', 'additive', 'film'], help='Skip connection from decoder input to head (B2). none|additive|film (default: none)')
    parser.add_argument('--encoder-pool', action='store_true', help='B3: init decoder hidden from concat(last_output, mean_pool_last_k) instead of only last_output')
    parser.add_argument('--encoder-pool-k', type=int, help='B3: pool window size k (default: 5); clamped to encoder seq len at runtime')
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
    # Multi-step autoregressive training
    parser.add_argument('--multistep-horizon', type=int, help='Max prediction horizon H for multi-step training (1=single-step default)')
    parser.add_argument('--multistep-curriculum', choices=['none', 'linear'], help='Curriculum for increasing H over training epochs')
    parser.add_argument('--multistep-curriculum-start-epoch', type=int, help='Epoch to start increasing H (default: 0)')
    parser.add_argument('--multistep-curriculum-end-epoch', type=int, help='Epoch to reach max H (default: num_epochs)')
    # Scheduled sampling (exposure bias correction)
    parser.add_argument('--scheduled-sampling', action='store_true', help='Enable scheduled sampling (encoder context noise injection)')
    parser.add_argument('--ss-max-prob', type=float, help='Max probability of corrupting each context lap LapTime (default: 0.5)')
    parser.add_argument('--ss-noise-std', type=float, help='Noise std in normalized space (default: 0.02, ~300ms)')
    parser.add_argument('--ss-start-epoch', type=int, help='Epoch to start scheduled sampling (default: 10)')
    parser.add_argument('--ss-end-epoch', type=int, help='Epoch to reach max noise (default: num_epochs)')
    parser.add_argument('--context-window', type=int, help='Number of past laps used as encoder context (overrides config.data.context_window)')
    parser.add_argument('--num-workers', type=int, help='Number of data loading workers (overrides config)')
    parser.add_argument('--validation-freq', type=int, help='Validate every N epochs (default: 1)')
    parser.add_argument('--no-lr-scaling', action='store_true', help='Disable automatic linear LR scaling with batch size')

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

