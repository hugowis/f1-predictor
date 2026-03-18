"""
Training utilities for F1 lap time prediction models.

Provides a flexible Trainer class that can work with different model architectures.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Callable, List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import json
from datetime import datetime
from tqdm.auto import tqdm

from .evaluator import compute_regression_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Generic trainer for sequence prediction models.
    
    Handles:
    - Training loop with gradient accumulation
    - Validation during training
    - Learning rate scheduling
    - Model checkpointing
    - Metrics logging
    - Early stopping
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model to train
    optimizer : Optimizer
        Optimizer instance
    criterion : nn.Module
        Loss function
    device : str
        Device to train on ('cpu' or 'cuda')
    scheduler : callable, optional
        Learning rate scheduler
    checkpoint_dir : Path, optional
        Directory to save checkpoints
    
    Examples
    --------
    >>> model = Seq2Seq(...)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> criterion = nn.MSELoss()
    >>> 
    >>> trainer = Trainer(model, optimizer, criterion, device='cuda')
    >>> 
    >>> train_loss = trainer.train_epoch(train_loader)
    >>> val_loss, val_metrics = trainer.validate(val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        scheduler: Optional[Any] = None,
        checkpoint_dir: Optional[Path] = None,
        gradient_clip: Optional[float] = 1.0,
        accumulation_steps: int = 1,
        pit_loss_weight: float = 1e-3,
        compound_loss_weight: float = 0.01,
        use_mixed_precision: bool = True,
        early_stopping_use_ema: bool = False,
        early_stopping_ema_alpha: float = 0.3,
        dynamic_aux_balance: bool = True,
        dynamic_aux_ema_alpha: float = 0.05,
        dynamic_aux_min_scale: float = 0.001,
        dynamic_aux_max_scale: float = 20.0,
        compound_label_smoothing: float = 0.02,
        compound_class_weights: Optional[List[float]] = None,
        lap_loss_kind: str = 'mse',
        lap_huber_delta: float = 0.1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        self.pit_loss_weight = pit_loss_weight
        self.compound_loss_weight = compound_loss_weight
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        self.early_stopping_use_ema = early_stopping_use_ema
        self.early_stopping_ema_alpha = float(early_stopping_ema_alpha)
        self.dynamic_aux_balance = bool(dynamic_aux_balance)
        self.dynamic_aux_ema_alpha = float(dynamic_aux_ema_alpha)
        self.dynamic_aux_min_scale = float(dynamic_aux_min_scale)
        self.dynamic_aux_max_scale = float(dynamic_aux_max_scale)
        self.compound_label_smoothing = float(compound_label_smoothing)
        # Lap loss configuration: 'mse' or 'huber'
        self.lap_loss_kind = str(lap_loss_kind).lower()
        self.lap_huber_delta = float(lap_huber_delta)

        self._lap_loss_ema = None
        self._pit_loss_ema = None
        self._compound_loss_ema = None
        self._compound_class_weight_tensor = None
        if compound_class_weights is not None:
            try:
                self._compound_class_weight_tensor = torch.tensor(
                    compound_class_weights,
                    dtype=torch.float32,
                    device=self.device,
                )
            except Exception:
                logger.exception("Failed to set provided compound_class_weights; falling back to automatic inference")
                self._compound_class_weight_tensor = None
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Ensure model is on the correct device
        try:
            self.model.to(self.device)
        except Exception:
            pass
        
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.best_monitored_val = float('inf')
        self.val_ema = None
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'train_lap_loss': [],
            'train_pit_loss': [],
            'train_compound_loss': [],
            'val_loss': [],
            'val_loss_ema': [],
            'val_loss_monitored': [],
            'learning_rate': [],
        }
        self._show_progress = True
        
        if self.use_mixed_precision:
            logger.info("Mixed precision training enabled (FP16)")

    def _update_loss_ema(self, current_ema: Optional[float], value: float) -> float:
        if current_ema is None:
            return float(value)
        alpha = self.dynamic_aux_ema_alpha
        return alpha * float(value) + (1.0 - alpha) * float(current_ema)

    def _maybe_initialize_compound_class_weights(self, dataset: Optional[Any]) -> None:
        if self._compound_class_weight_tensor is not None or dataset is None:
            return

        try:
            num_classes = int(getattr(self.model, 'compound_classes', 4))
            counts = np.zeros(num_classes, dtype=np.float64)
            precomputed_targets = getattr(dataset, '_precomputed_targets', None)
            if not precomputed_targets:
                return

            for target in precomputed_targets:
                if target is None or 'compound' not in target:
                    continue
                comp_val = target['compound']
                if isinstance(comp_val, torch.Tensor):
                    comp_idx = int(comp_val.item())
                else:
                    comp_idx = int(comp_val)
                if 0 <= comp_idx < num_classes:
                    counts[comp_idx] += 1.0

            if counts.sum() <= 0:
                return

            counts = np.clip(counts, 1.0, None)
            inv_freq = counts.sum() / (len(counts) * counts)
            weights = inv_freq / max(inv_freq.mean(), 1e-8)
            self._compound_class_weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
            logger.info(f"Initialized compound class weights: {weights.tolist()}")
        except Exception:
            logger.exception("Failed to infer compound class weights from dataset")
            self._compound_class_weight_tensor = None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int = 0,
        teacher_forcing_ratio: float = 1.0,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        epoch : int, optional
            Epoch number (for logging)
        teacher_forcing_ratio : float, optional
            Probability of using teacher forcing. Default is 1.0.
        
        Returns
        -------
        dict
            Average total and component losses for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_lap_loss = 0.0
        total_pit_loss = 0.0
        total_comp_loss = 0.0
        total_pit_scale = 0.0
        total_comp_scale = 0.0
        num_multitask_batches = 0
        num_batches = 0

        self._maybe_initialize_compound_class_weights(getattr(train_loader, 'dataset', None))

        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1} [train]",
            leave=False,
            dynamic_ncols=True,
            disable=not self._show_progress,
        )

        for batch_idx, batch in progress:
            # Extract batch data (pass dataset so decoder targets can be normalized/asserted)
            encoder_input, decoder_input, targets, metadata = self._unpack_batch(batch, dataset=train_loader.dataset)
            # Encoder input may be a dict (structured numeric+categorical); let the
            # model handle device placement and embedding when it's a dict.
            if not isinstance(encoder_input, dict):
                encoder_input = encoder_input.to(self.device)
            decoder_input = decoder_input.to(self.device)
            # Move targets tensors to device if targets is a dict
            if isinstance(targets, dict):
                for k, v in list(targets.items()):
                    if isinstance(v, torch.Tensor):
                        targets[k] = v.to(self.device)
            else:
                targets = targets.to(self.device)
            
            # Forward pass with optional mixed precision
            with autocast(device_type='cuda', enabled=self.use_mixed_precision):
                outputs = self.model(
                    encoder_input,
                    decoder_input,
                    teacher_forcing=True,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                )

            # Determine if targets is a dict (multi-task) or single tensor
            if isinstance(targets, dict):
                # Lap time target
                lap_t = targets['lap_time']
                if lap_t.dim() == 1:
                    lap_t_rs = lap_t.unsqueeze(-1).unsqueeze(-1)
                else:
                    lap_t_rs = lap_t

                # Pit and compound targets
                pit_t = targets.get('is_pitlap', None)
                comp_t = targets.get('compound', None)

                with torch.no_grad():
                    mask = torch.isfinite(lap_t_rs)
                if mask.sum() == 0:
                    continue

                lap_pred = outputs['lap'] if isinstance(outputs, dict) else outputs
                if not torch.isfinite(lap_pred).all():
                    continue

                denom = mask.float().sum()
                # Compute lap loss: Huber (preferred) or MSE
                if self.lap_loss_kind == 'huber':
                    diff = lap_pred - lap_t_rs
                    ad = diff.abs()
                    delta = self.lap_huber_delta
                    huber = torch.where(ad <= delta, 0.5 * diff * diff, delta * (ad - 0.5 * delta))
                    masked = huber * mask.float()
                    lap_loss = masked.sum() / denom
                else:
                    se = (lap_pred - lap_t_rs) ** 2
                    masked_se = se * mask.float()
                    lap_loss = masked_se.sum() / denom

                # Pit loss (BCEWithLogits)
                pit_loss = torch.tensor(0.0, device=self.device)
                if pit_t is not None and 'pit_logits' in outputs:
                    pit_pred = outputs['pit_logits']  # (batch, seq_len)
                    # Align shapes: make pit_t same shape as pit_pred
                    if pit_t.dim() == 1:
                        pit_t_rs = pit_t.unsqueeze(-1)
                    else:
                        pit_t_rs = pit_t
                    bce = nn.BCEWithLogitsLoss(reduction='none')
                    pit_raw_loss = bce(pit_pred, pit_t_rs.float())
                    pit_loss = pit_raw_loss.mean()

                # Compound loss (CrossEntropy)
                comp_loss = torch.tensor(0.0, device=self.device)
                if comp_t is not None and 'compound_logits' in outputs:
                    comp_pred = outputs['compound_logits']  # (batch, seq_len, C)
                    # Flatten for cross-entropy: (batch*seq_len, C)
                    bsz, seq_len, C = comp_pred.shape
                    comp_pred_flat = comp_pred.reshape(bsz * seq_len, C)
                    # comp_t may be (batch,) or (batch, seq_len)
                    if comp_t.dim() == 1:
                        comp_t_rs = comp_t.unsqueeze(-1).expand(-1, seq_len).reshape(-1)
                    else:
                        comp_t_rs = comp_t.reshape(-1)
                    comp_weight = self._compound_class_weight_tensor
                    if comp_weight is not None:
                        comp_weight = comp_weight.to(device=comp_pred_flat.device, dtype=torch.float32)
                    try:
                        ce = nn.CrossEntropyLoss(
                            reduction='none',
                            weight=comp_weight,
                            label_smoothing=self.compound_label_smoothing,
                        )
                    except TypeError:
                        ce = nn.CrossEntropyLoss(
                            reduction='none',
                            weight=comp_weight,
                        )
                    with autocast(device_type='cuda', enabled=False):
                        comp_raw = ce(comp_pred_flat.float(), comp_t_rs.long())
                    comp_loss = comp_raw.mean()

                pit_scale = 1.0
                comp_scale = 1.0
                if self.dynamic_aux_balance:
                    self._lap_loss_ema = self._update_loss_ema(self._lap_loss_ema, float(lap_loss.detach().item()))
                    self._pit_loss_ema = self._update_loss_ema(self._pit_loss_ema, float(pit_loss.detach().item()))
                    self._compound_loss_ema = self._update_loss_ema(self._compound_loss_ema, float(comp_loss.detach().item()))

                    if self._pit_loss_ema is not None and self._pit_loss_ema > 0:
                        pit_scale = float(np.clip(
                            self._lap_loss_ema / (self._pit_loss_ema + 1e-8),
                            self.dynamic_aux_min_scale,
                            self.dynamic_aux_max_scale,
                        ))
                    if self._compound_loss_ema is not None and self._compound_loss_ema > 0:
                        comp_scale = float(np.clip(
                            self._lap_loss_ema / (self._compound_loss_ema + 1e-8),
                            self.dynamic_aux_min_scale,
                            self.dynamic_aux_max_scale,
                        ))

                # Combine losses with configurable weighting
                loss_batch = (
                    lap_loss +
                    float(self.pit_loss_weight) * pit_scale * pit_loss +
                    float(self.compound_loss_weight) * comp_scale * comp_loss
                )

                lap_loss_value = float(lap_loss.detach().item())
                pit_loss_value = float(pit_loss.detach().item())
                comp_loss_value = float(comp_loss.detach().item())
                total_pit_scale += pit_scale
                total_comp_scale += comp_scale
                num_multitask_batches += 1

            else:
                # Old single-target behavior
                if targets.dim() == 1:
                    targets_reshaped = targets.unsqueeze(-1).unsqueeze(-1)
                else:
                    targets_reshaped = targets

                with torch.no_grad():
                    mask = torch.isfinite(targets_reshaped)
                if mask.sum() == 0:
                    continue

                if isinstance(outputs, dict):
                    lap_pred = outputs['lap']
                else:
                    lap_pred = outputs

                if not torch.isfinite(lap_pred).all():
                    continue

                denom = mask.float().sum()
                if self.lap_loss_kind == 'huber':
                    diff = lap_pred - targets_reshaped
                    ad = diff.abs()
                    delta = self.lap_huber_delta
                    huber = torch.where(ad <= delta, 0.5 * diff * diff, delta * (ad - 0.5 * delta))
                    masked = huber * mask.float()
                    loss_batch = masked.sum() / denom
                else:
                    se = (lap_pred - targets_reshaped) ** 2
                    masked_se = se * mask.float()
                    loss_batch = masked_se.sum() / denom
                lap_loss_value = float(loss_batch.detach().item())
                pit_loss_value = 0.0
                comp_loss_value = 0.0

            # Skip pathological batches that would dominate epoch loss and destabilize training
            if not torch.isfinite(loss_batch):
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss_value = float(loss_batch.detach().item())
            if loss_value > 1e3:
                logger.debug(
                    f"Skipping pathological batch at epoch {epoch}, batch {batch_idx}: "
                    f"loss={loss_value:.6f}"
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # Scale loss for gradient accumulation (common to both branches)
            loss = loss_batch / self.accumulation_steps
            
            # Backward pass with gradient accumulation and mixed precision
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step with gradient accumulation and mixed precision
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_mixed_precision:
                    # Unscale gradients for clipping
                    if self.gradient_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip
                        )
                    
                    # Step optimizer with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard training without mixed precision
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip
                        )
                    
                    # Check for non-finite gradients
                    skip_step = False
                    for name, p in self.model.named_parameters():
                        if p.grad is None:
                            continue
                        if not torch.isfinite(p.grad).all():
                            logger.warning(f"Non-finite gradient detected for {name} at batch {batch_idx}, skipping optimizer step")
                            skip_step = True
                            break
                    
                    if not skip_step:
                        self.optimizer.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
            
            total_loss += loss_value
            total_lap_loss += lap_loss_value
            total_pit_loss += pit_loss_value
            total_comp_loss += comp_loss_value
            num_batches += 1

            if num_batches > 0:
                progress.set_postfix({
                    'loss': f"{(total_loss / num_batches):.4f}",
                    'lap': f"{(total_lap_loss / num_batches):.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })

        progress.close()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_lap_loss = total_lap_loss / num_batches if num_batches > 0 else 0.0
        avg_pit_loss = total_pit_loss / num_batches if num_batches > 0 else 0.0
        avg_comp_loss = total_comp_loss / num_batches if num_batches > 0 else 0.0
        avg_pit_scale = total_pit_scale / num_multitask_batches if num_multitask_batches > 0 else 1.0
        avg_comp_scale = total_comp_scale / num_multitask_batches if num_multitask_batches > 0 else 1.0
        logger.info(
            f"Epoch {epoch} - Train Loss: {avg_loss:.6f} "
            f"(lap={avg_lap_loss:.6f}, pit={avg_pit_loss:.6f}, compound={avg_comp_loss:.6f}, "
            f"pit_scale={avg_pit_scale:.3f}, comp_scale={avg_comp_scale:.3f})"
        )

        return {
            'loss': avg_loss,
            'lap_loss': avg_lap_loss,
            'pit_loss': avg_pit_loss,
            'compound_loss': avg_comp_loss,
        }

    def train_epoch_rollout(
        self,
        rollout_loader: DataLoader,
        epoch: int = 0,
        rollout_weight: float = 1.0,
        teacher_forcing_ratio: float = 0.0,
        current_rollout_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch using multi-step autoregressive rollout.

        Supports *scheduled sampling*: when ``teacher_forcing_ratio > 0`` the
        decoder uses ground-truth lap times as input with the given probability
        at each step, decaying towards fully autoregressive over training.  The
        default (``teacher_forcing_ratio=0``) preserves the original fully
        autoregressive behaviour.

        Supports *curriculum rollout*: ``current_rollout_steps`` limits how
        many future steps are unrolled this epoch.  When ``None`` the full
        horizon stored in the dataset is used.  The caller should gradually
        increase this from 1 to ``rollout_steps`` over the warm-up period.

        Parameters
        ----------
        rollout_loader : DataLoader
            DataLoader wrapping an ``AutoregressiveRolloutDataset``.
        epoch : int
            Epoch number (for logging).
        rollout_weight : float
            Scalar multiplier applied to the total rollout loss before
            backpropagation.
        teacher_forcing_ratio : float
            Probability (0–1) of using ground-truth decoder input at each
            rollout step (scheduled sampling).  0.0 = fully autoregressive.
        current_rollout_steps : int or None
            Curriculum horizon: truncate each rollout sequence to at most this
            many future steps.  ``None`` means use all steps in the batch.

        Returns
        -------
        dict
            Average total and component losses for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_lap_loss = 0.0
        total_pit_loss = 0.0
        total_comp_loss = 0.0
        num_batches = 0

        # Clamp scheduled-sampling ratio to valid range
        tf_ratio = float(np.clip(teacher_forcing_ratio, 0.0, 1.0))
        use_tf = tf_ratio > 0.0

        progress = tqdm(
            enumerate(rollout_loader),
            total=len(rollout_loader),
            desc=f"Epoch {epoch + 1} [rollout tf={tf_ratio:.2f}]",
            leave=False,
            dynamic_ncols=True,
            disable=not self._show_progress,
        )

        for batch_idx, batch in progress:
            encoder_input, targets, metadata = batch[0], batch[1], batch[2]

            # --- encoder input to device ---
            if isinstance(encoder_input, dict):
                for k, v in encoder_input.items():
                    if isinstance(v, torch.Tensor):
                        encoder_input[k] = v.to(self.device)
            else:
                encoder_input = encoder_input.to(self.device)

            # --- targets to device ---
            if isinstance(targets, dict):
                for k, v in list(targets.items()):
                    if isinstance(v, torch.Tensor):
                        targets[k] = v.to(self.device)
            else:
                targets = targets.to(self.device)

            lap_t = targets['lap_time'] if isinstance(targets, dict) else targets
            # lap_t shape: (batch, max_rollout_steps)
            max_steps = lap_t.size(1)

            # --- Curriculum: clamp to current_rollout_steps ---
            active_steps = max_steps
            if current_rollout_steps is not None:
                active_steps = max(1, min(int(current_rollout_steps), max_steps))
            lap_t = lap_t[:, :active_steps]

            # Construct decoder_input: (batch, active_steps, 1)
            # Position 0 = last context LapTime (LapTime is feature index 0)
            if isinstance(encoder_input, dict):
                last_lt = encoder_input['numeric'][:, -1, 0]
            else:
                last_lt = encoder_input[:, -1, 0]

            decoder_input = torch.zeros(
                lap_t.size(0), active_steps, 1,
                dtype=torch.float32, device=self.device,
            )
            decoder_input[:, 0, 0] = last_lt
            # Fill remaining positions with ground truth — used by the model
            # when teacher_forcing=True with the scheduled sampling ratio.
            if active_steps > 1:
                decoder_input[:, 1:, 0] = lap_t[:, :-1]

            # Forward pass — scheduled sampling: use teacher forcing at the
            # given ratio so the decoder gradually transitions from
            # teacher-forced to fully autoregressive.
            with autocast(device_type='cuda', enabled=self.use_mixed_precision):
                outputs = self.model(
                    encoder_input,
                    decoder_input,
                    teacher_forcing=use_tf,
                    teacher_forcing_ratio=tf_ratio,
                )

            # --- Loss computation (same logic as train_epoch) ---
            lap_pred = outputs['lap'] if isinstance(outputs, dict) else outputs  # (B, S, 1)
            lap_t_rs = lap_t.unsqueeze(-1)  # (B, S, 1)

            with torch.no_grad():
                mask = torch.isfinite(lap_t_rs)
            if mask.sum() == 0:
                continue
            if not torch.isfinite(lap_pred).all():
                continue

            denom = mask.float().sum()
            if self.lap_loss_kind == 'huber':
                diff = lap_pred - lap_t_rs
                ad = diff.abs()
                delta = self.lap_huber_delta
                huber = torch.where(ad <= delta, 0.5 * diff * diff, delta * (ad - 0.5 * delta))
                lap_loss = (huber * mask.float()).sum() / denom
            else:
                se = (lap_pred - lap_t_rs) ** 2
                lap_loss = (se * mask.float()).sum() / denom

            pit_loss = torch.tensor(0.0, device=self.device)
            pit_t = targets.get('is_pitlap', None) if isinstance(targets, dict) else None
            if pit_t is not None and 'pit_logits' in outputs:
                pit_pred = outputs['pit_logits']  # (B, S)
                # Align pit target to active steps
                pit_t_active = pit_t[:, :active_steps] if pit_t.dim() > 1 else pit_t
                bce = nn.BCEWithLogitsLoss(reduction='mean')
                pit_loss = bce(pit_pred, pit_t_active.float())

            comp_loss = torch.tensor(0.0, device=self.device)
            comp_t = targets.get('compound', None) if isinstance(targets, dict) else None
            if comp_t is not None and 'compound_logits' in outputs:
                comp_pred = outputs['compound_logits']  # (B, S, C)
                bsz, seq_len, C = comp_pred.shape
                comp_pred_flat = comp_pred.reshape(bsz * seq_len, C)
                # Align compound target to active steps
                comp_t_active = comp_t[:, :active_steps] if comp_t.dim() > 1 else comp_t
                comp_t_flat = comp_t_active.reshape(-1)
                comp_weight = self._compound_class_weight_tensor
                if comp_weight is not None:
                    comp_weight = comp_weight.to(device=comp_pred_flat.device, dtype=torch.float32)
                try:
                    ce = nn.CrossEntropyLoss(weight=comp_weight, label_smoothing=self.compound_label_smoothing)
                except TypeError:
                    ce = nn.CrossEntropyLoss(weight=comp_weight)
                with autocast(device_type='cuda', enabled=False):
                    comp_loss = ce(comp_pred_flat.float(), comp_t_flat.long())

            loss_batch = (
                lap_loss
                + float(self.pit_loss_weight) * pit_loss
                + float(self.compound_loss_weight) * comp_loss
            ) * rollout_weight

            if not torch.isfinite(loss_batch):
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss_value = float(loss_batch.detach().item())
            if loss_value > 1e3:
                logger.debug(f"Skipping pathological rollout batch at epoch {epoch}, batch {batch_idx}: loss={loss_value:.6f}")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss = loss_batch / self.accumulation_steps

            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_mixed_precision:
                    if self.gradient_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss_value
            total_lap_loss += float(lap_loss.detach().item())
            total_pit_loss += float(pit_loss.detach().item())
            total_comp_loss += float(comp_loss.detach().item())
            num_batches += 1

            if num_batches > 0:
                progress.set_postfix({
                    'r_loss': f"{(total_loss / num_batches):.4f}",
                    'r_lap': f"{(total_lap_loss / num_batches):.4f}",
                })

        progress.close()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_lap = total_lap_loss / num_batches if num_batches > 0 else 0.0
        avg_pit = total_pit_loss / num_batches if num_batches > 0 else 0.0
        avg_comp = total_comp_loss / num_batches if num_batches > 0 else 0.0
        logger.info(
            f"Epoch {epoch} - Rollout Loss: {avg_loss:.6f} "
            f"(lap={avg_lap:.6f}, pit={avg_pit:.6f}, compound={avg_comp:.6f}, "
            f"tf={tf_ratio:.2f}, steps={active_steps})"
        )
        return {
            'loss': avg_loss,
            'lap_loss': avg_lap,
            'pit_loss': avg_pit,
            'compound_loss': avg_comp,
            'active_steps': active_steps,
        }
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate model on validation set.
        
        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader
        
        Returns
        -------
        tuple of (float, dict)
            (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []

        progress = tqdm(
            val_loader,
            total=len(val_loader),
            desc="Validation",
            leave=False,
            dynamic_ncols=True,
            disable=not self._show_progress,
        )
        
        with torch.no_grad():
            for batch in progress:
                encoder_input, decoder_input, targets, metadata = self._unpack_batch(batch, dataset=val_loader.dataset)
                # Encoder input may be structured dict; don't call .to on dicts
                if not isinstance(encoder_input, dict):
                    encoder_input = encoder_input.to(self.device)
                decoder_input = decoder_input.to(self.device)
                # Move targets tensors to device if targets is a dict
                if isinstance(targets, dict):
                    for k, v in list(targets.items()):
                        if isinstance(v, torch.Tensor):
                            targets[k] = v.to(self.device)
                else:
                    targets = targets.to(self.device)

                # Forward pass with optional mixed precision (no teacher forcing for validation)
                with autocast(device_type='cuda', enabled=self.use_mixed_precision):
                    outputs = self.model(
                        encoder_input,
                        decoder_input,
                        teacher_forcing=False,
                    )

                # If targets is dict (multi-task), focus validation on lap_time
                if isinstance(targets, dict):
                    lap_t = targets['lap_time']
                    if lap_t.dim() == 1:
                        targets_reshaped = lap_t.unsqueeze(-1).unsqueeze(-1)
                    else:
                        targets_reshaped = lap_t
                else:
                    # Reshape targets to match output shape: (batch,) -> (batch, 1, 1)
                    if targets.dim() == 1:
                        targets_reshaped = targets.unsqueeze(-1).unsqueeze(-1)
                    else:
                        targets_reshaped = targets

                # Compute finite mask on the same tensor shape used in loss
                mask_t = torch.isfinite(targets_reshaped)
                if mask_t.sum() == 0:
                    continue

                # Use lap predictions if multi-head
                if isinstance(outputs, dict):
                    lap_pred = outputs['lap']
                else:
                    lap_pred = outputs

                if not torch.isfinite(lap_pred).all():
                    continue

                denom_val = mask_t.float().sum()
                if self.lap_loss_kind == 'huber':
                    diff = lap_pred - targets_reshaped
                    ad = diff.abs()
                    delta = self.lap_huber_delta
                    huber = torch.where(ad <= delta, 0.5 * diff * diff, delta * (ad - 0.5 * delta))
                    masked_huber = huber * mask_t.float()
                    loss_val = masked_huber.sum().item() / float(denom_val.item())
                else:
                    se = (lap_pred - targets_reshaped) ** 2
                    masked_se = se * mask_t.float()
                    loss_val = masked_se.sum().item() / float(denom_val.item())
                total_loss += loss_val
                num_batches += 1

                # Store predictions and valid targets for metrics
                # Use lap predictions and targets_reshaped for metrics
                if isinstance(outputs, dict):
                    lap_pred = outputs['lap']
                else:
                    lap_pred = outputs

                preds_np = lap_pred.cpu().numpy().reshape(-1)
                targs_np = targets_reshaped.cpu().numpy().reshape(-1)
                valid_idx = np.isfinite(targs_np) & np.isfinite(preds_np)
                all_predictions.append(preds_np[valid_idx])
                all_targets.append(targs_np[valid_idx])

                if num_batches > 0:
                    progress.set_postfix({'val_loss': f"{(total_loss / num_batches):.4f}"})

        progress.close()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Compute additional metrics
        if all_predictions and all_targets:
            predictions = np.concatenate(all_predictions, axis=0)
            targets_arr = np.concatenate(all_targets, axis=0)
            metrics = self._compute_metrics(predictions, targets_arr)
        else:
            metrics = {'mae': float('nan'), 'rmse': float('nan'), 'mape': float('nan')}
        metrics['loss'] = avg_loss
        
        logger.info(
            "Validation summary: loss=%.6f, mae=%.4f, rmse=%.4f, mape=%.2f%%",
            avg_loss,
            metrics.get('mae', float('nan')),
            metrics.get('rmse', float('nan')),
            metrics.get('mape', float('nan')),
        )
        
        return avg_loss, metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        early_stopping_min_epochs: int = 0,
        teacher_forcing_schedule: Optional[Callable[[int], float]] = None,
        rollout_loader: Optional[DataLoader] = None,
        rollout_weight: float = 1.0,
        rollout_start_epoch: int = 0,
        rollout_teacher_forcing_schedule: Optional[Callable[[int], float]] = None,
        rollout_curriculum_steps_schedule: Optional[Callable[[int], int]] = None,
    ) -> Dict[str, Any]:
        """
        Train model for multiple epochs.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        num_epochs : int
            Number of epochs to train
        early_stopping_patience : int
            Number of epochs without improvement before stopping
        early_stopping_min_epochs : int
            Grace period: patience counter won't start before this epoch.
            Prevents a lucky low val loss in the first few epochs from
            triggering early stopping before training has stabilised.
        teacher_forcing_schedule : callable, optional
            Function that takes epoch number and returns teacher_forcing_ratio
            for the *single-step* training pass.  If None, uses constant 1.0.
        rollout_loader : DataLoader, optional
            DataLoader wrapping an ``AutoregressiveRolloutDataset`` for
            multi-step rollout training.  When provided, an additional
            rollout training pass is run each epoch (after the single-step
            pass) starting from ``rollout_start_epoch``.
        rollout_weight : float
            Multiplier applied to rollout loss before combining with
            single-step loss for logging purposes.  The rollout pass runs
            its own backward so this weight scales the loss magnitude.
        rollout_start_epoch : int
            First epoch at which rollout training is activated.  This lets
            the model warm up with single-step training first.
        rollout_teacher_forcing_schedule : callable, optional
            Function ``epoch -> float`` returning the teacher-forcing ratio to
            use *inside* the rollout decoder (scheduled sampling).  A value
            of 1.0 means fully teacher-forced; 0.0 means fully autoregressive.
            When ``None`` defaults to 0.0 (backward-compatible).
        rollout_curriculum_steps_schedule : callable, optional
            Function ``epoch -> int`` returning the active rollout horizon for
            this epoch (curriculum learning).  Allows starting with a short
            horizon (e.g. 1 step) and gradually increasing to ``rollout_steps``.
            When ``None`` the full horizon stored in the dataset is always used.
        
        Returns
        -------
        dict
            Training history and best metrics
        """
        if teacher_forcing_schedule is None:
            teacher_forcing_schedule = lambda epoch: 1.0
        if rollout_teacher_forcing_schedule is None:
            rollout_teacher_forcing_schedule = lambda epoch: 0.0
        if rollout_curriculum_steps_schedule is None:
            rollout_curriculum_steps_schedule = lambda epoch: None

        # Extend history keys for rollout tracking
        if 'rollout_loss' not in self.history:
            self.history['rollout_loss'] = []
        if 'rollout_teacher_forcing' not in self.history:
            self.history['rollout_teacher_forcing'] = []
        if 'rollout_curriculum_steps' not in self.history:
            self.history['rollout_curriculum_steps'] = []
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Model: {self.model.model_info()}")
        logger.info(f"Device: {self.device}")
        if rollout_loader is not None:
            logger.info(
                f"Rollout training enabled: weight={rollout_weight}, "
                f"start_epoch={rollout_start_epoch}, "
                f"scheduled_sampling=True, curriculum=True"
            )
        if self.early_stopping_use_ema:
            logger.info(f"Early stopping monitor: EMA(val_loss), alpha={self.early_stopping_ema_alpha:.2f}")
        else:
            logger.info("Early stopping monitor: val_loss")
        if early_stopping_min_epochs > 0:
            logger.info(f"Early stopping grace period: {early_stopping_min_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Get teacher forcing ratio for this epoch
            tf_ratio = teacher_forcing_schedule(epoch)
            
            # Train (single-step)
            train_stats = self.train_epoch(train_loader, epoch, tf_ratio)
            self.history['train_loss'].append(float(train_stats.get('loss', 0.0)))
            self.history['train_lap_loss'].append(float(train_stats.get('lap_loss', train_stats.get('loss', 0.0))))
            self.history['train_pit_loss'].append(float(train_stats.get('pit_loss', 0.0)))
            self.history['train_compound_loss'].append(float(train_stats.get('compound_loss', 0.0)))

            # Rollout training pass (multi-step autoregressive with scheduled sampling)
            rollout_loss_value = 0.0
            rollout_tf_value = 0.0
            rollout_steps_value = 0
            if rollout_loader is not None and epoch >= rollout_start_epoch:
                rollout_tf = rollout_teacher_forcing_schedule(epoch)
                cur_steps = rollout_curriculum_steps_schedule(epoch)
                rollout_stats = self.train_epoch_rollout(
                    rollout_loader,
                    epoch,
                    rollout_weight,
                    teacher_forcing_ratio=rollout_tf,
                    current_rollout_steps=cur_steps,
                )
                rollout_loss_value = rollout_stats.get('loss', 0.0)
                rollout_tf_value = rollout_tf
                rollout_steps_value = rollout_stats.get('active_steps', cur_steps if cur_steps is not None else 0)
            self.history['rollout_loss'].append(rollout_loss_value)
            self.history['rollout_teacher_forcing'].append(rollout_tf_value)
            self.history['rollout_curriculum_steps'].append(rollout_steps_value)
            
            # Validate
            if val_loader is not None:
                val_loss, val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)

                # Monitor either raw val loss or EMA-smoothed val loss
                if self.early_stopping_use_ema:
                    if self.val_ema is None:
                        self.val_ema = float(val_loss)
                    else:
                        alpha = self.early_stopping_ema_alpha
                        self.val_ema = alpha * float(val_loss) + (1.0 - alpha) * self.val_ema
                    monitored_val = self.val_ema
                else:
                    monitored_val = float(val_loss)

                self.history['val_loss_ema'].append(float(self.val_ema) if self.val_ema is not None else float(val_loss))
                self.history['val_loss_monitored'].append(float(monitored_val))
                logger.info(f"Monitored Validation Loss: {monitored_val:.6f}")
                
                # Check for improvement
                if monitored_val < self.best_monitored_val:
                    self.best_val_loss = val_loss
                    self.best_monitored_val = monitored_val
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(epoch, val_loss, val_metrics)
                else:
                    self.epochs_without_improvement += 1
                
                # Early stopping (only after grace period)
                if epoch >= early_stopping_min_epochs and self.epochs_without_improvement >= early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(no improvement for {early_stopping_patience} epochs)"
                    )
                    break

                elapsed = time.perf_counter() - epoch_start
                logger.info(
                    "Epoch %d/%d | tf=%.3f | r_tf=%.3f | r_steps=%s | lr=%.2e | train=%.6f | val=%.6f | monitored=%.6f | best=%.6f | time=%.1fs",
                    epoch + 1,
                    num_epochs,
                    tf_ratio,
                    rollout_tf_value,
                    str(rollout_steps_value) if rollout_steps_value else "-",
                    current_lr,
                    train_stats.get('loss', float('nan')),
                    val_loss,
                    monitored_val,
                    self.best_monitored_val,
                    elapsed,
                )
            else:
                elapsed = time.perf_counter() - epoch_start
                logger.info(
                    "Epoch %d/%d | tf=%.3f | r_tf=%.3f | r_steps=%s | lr=%.2e | train=%.6f | time=%.1fs",
                    epoch + 1,
                    num_epochs,
                    tf_ratio,
                    rollout_tf_value,
                    str(rollout_steps_value) if rollout_steps_value else "-",
                    current_lr,
                    train_stats.get('loss', float('nan')),
                    elapsed,
                )
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
        
        logger.info("Training completed")
        return self.history
    
    def _unpack_batch(self, batch: Tuple, dataset: Optional[Any] = None) -> Tuple:
        """
        Unpack batch data from dataloader.
        
        Handles both StintDataloader and custom batch formats.
        For StintDataloader (3-element tuple), expands 1D targets to (batch, 1, 1)
        for seq2seq input/output.
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            # Already unpacked format
            return batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 3:
            # StintDataloader format: (features, target, metadata)
            encoder_input, targets, metadata = batch

            # If encoder_input is structured dict (numeric + categorical), return it
            # directly and let the model handle embeddings and concatenation.
            if isinstance(encoder_input, dict) and 'numeric' in encoder_input and 'categorical' in encoder_input:
                encoder_final = encoder_input
            else:
                encoder_final = encoder_input

            # Extract lap_time from targets dict (standardized format)
            # Targets must be a dict with 'lap_time' key for multi-task consistency
            if not isinstance(targets, dict):
                # Convert tensor targets to dict format for consistency
                targets = {'lap_time': targets}
            
            lap_t = targets.get('lap_time')
            if lap_t is None:
                raise ValueError("targets dict must contain 'lap_time' key")
            
            # Data is already normalized by the dataloader - no runtime normalization
            # This prevents data leakage from applying transforms at training time
            lap_tensor_norm = lap_t.clone().type(torch.float32)

            # Build decoder_input shape consistent with previous behavior
            if lap_tensor_norm.dim() == 1:
                decoder_input = lap_tensor_norm.unsqueeze(-1).unsqueeze(-1)
            elif lap_tensor_norm.dim() == 2:
                decoder_input = lap_tensor_norm.unsqueeze(-1)
            else:
                decoder_input = lap_tensor_norm.clone()

            return encoder_final, decoder_input, targets, metadata
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")
    
    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions
        targets : np.ndarray
            Ground truth targets
        
        Returns
        -------
        dict
            Metrics (MAE, RMSE, MAPE, etc.)
        """
        metrics = compute_regression_metrics(predictions, targets)
        return {
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape': metrics['mape'],
        }
    
    def _save_checkpoint(self, epoch: int, val_loss: float, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"best_model.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'model_config': self.model.get_config() if hasattr(self.model, 'get_config') else {},
            'timestamp': datetime.now().isoformat(),
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        # Also save metrics to JSON (convert numpy types to Python native types)
        metrics_path = self.checkpoint_dir / f"metrics_epoch_{epoch}.json"
        metrics_serializable = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load model from checkpoint.
        
        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file
        
        Returns
        -------
        dict
            Checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint


def create_scheduler(optimizer: Optimizer, config: Dict[str, Any]) -> Optional[Any]:
    """
    Create learning rate scheduler based on config.
    
    Parameters
    ----------
    optimizer : Optimizer
        PyTorch optimizer
    config : dict
        Scheduler config with keys:
        - scheduler_type: 'cosine', 'linear', 'lambda', etc.
        - warm_up_epochs: number of warmup epochs
        - total_epochs: total training epochs
    
    Returns
    -------
    scheduler or None
        Learning rate scheduler
    """
    scheduler_type = config.get('scheduler_type', 'cosine')
    
    if scheduler_type == 'cosine':
        total_epochs = config.get('total_epochs', 100)
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    
    elif scheduler_type == 'linear':
        total_epochs = config.get('total_epochs', 100)
        warm_up = config.get('warm_up_epochs', 5)
        
        def lr_lambda(epoch):
            if epoch < warm_up:
                return float(epoch) / float(max(1, warm_up))
            else:
                return max(0.0, float(total_epochs - epoch) / float(max(1, total_epochs - warm_up)))
        
        return LambdaLR(optimizer, lr_lambda)
    
    else:
        return None

