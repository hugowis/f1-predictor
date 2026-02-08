"""
Training utilities for F1 lap time prediction models.

Provides a flexible Trainer class that can work with different model architectures.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Callable, List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader
import json
from datetime import datetime

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
    >>> model = Seq2SeqGRU(...)
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
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        # Ensure model is on the correct device
        try:
            self.model.to(self.device)
        except Exception:
            pass
        
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int = 0,
        teacher_forcing_ratio: float = 1.0,
    ) -> float:
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
        float
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract batch data
            encoder_input, decoder_input, targets, metadata = self._unpack_batch(batch)
            # Encoder input may be a dict (structured numeric+categorical); let the
            # model handle device placement and embedding when it's a dict.
            if not isinstance(encoder_input, dict):
                encoder_input = encoder_input.to(self.device)
            decoder_input = decoder_input.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(
                encoder_input,
                decoder_input,
                teacher_forcing=True,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            
            # Reshape targets to match output shape: (batch,) -> (batch, 1, 1)
            if targets.dim() == 1:
                targets_reshaped = targets.unsqueeze(-1).unsqueeze(-1)
            else:
                targets_reshaped = targets

            # Compute masked loss: ignore NaN or non-finite targets
            with torch.no_grad():
                mask = torch.isfinite(targets_reshaped)
            if mask.sum() == 0:
                # Nothing to learn from this batch
                continue

            # If model outputs are non-finite, skip this batch to avoid NaNs
            if not torch.isfinite(outputs).all():
                # Skip silently to avoid cluttering logs
                continue

            se = (outputs - targets_reshaped) ** 2
            masked_se = se * mask.float()
            denom = mask.float().sum()
            loss_batch = masked_se.sum() / denom

            # Backward pass with gradient accumulation
            loss = loss_batch / self.accumulation_steps
            loss.backward()
            
            # Before stepping optimizer, check gradients for NaN/Inf
            skip_step = False
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                for name, p in self.model.named_parameters():
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all():
                        logger.warning(f"Non-finite gradient detected for {name} at batch {batch_idx}, skipping optimizer step and zeroing gradients")
                        skip_step = True
                        break
                if not skip_step:
                    self.optimizer.step()
                # Always zero grads to keep state consistent
                self.optimizer.zero_grad()
            
            total_loss += loss_batch.item()
            num_batches += 1
            
            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss_batch.item():.6f}"
                )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.6f}")
        
        return avg_loss
    
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
        
        with torch.no_grad():
            for batch in val_loader:
                encoder_input, decoder_input, targets, metadata = self._unpack_batch(batch)
                # Encoder input may be structured dict; don't call .to on dicts
                if not isinstance(encoder_input, dict):
                    encoder_input = encoder_input.to(self.device)
                decoder_input = decoder_input.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass (no teacher forcing for validation)
                outputs = self.model(
                    encoder_input,
                    decoder_input,
                    teacher_forcing=False,
                )
                
                # Reshape targets to match output shape: (batch,) -> (batch, 1, 1)
                if targets.dim() == 1:
                    targets_reshaped = targets.unsqueeze(-1).unsqueeze(-1)
                else:
                    targets_reshaped = targets

                # Mask invalid targets (NaN / inf)
                mask = np.isfinite(targets.cpu().numpy())
                # If no valid targets in this batch, skip
                if mask.sum() == 0:
                    continue

                # Compute masked loss using torch (ensure types on device)
                mask_t = torch.from_numpy(mask).to(self.device)
                se = (outputs - targets_reshaped) ** 2
                masked_se = se * mask_t.float()
                loss_val = masked_se.sum().item() / float(mask_t.sum().item())
                total_loss += loss_val
                num_batches += 1

                # Store predictions and valid targets for metrics
                preds_np = outputs.cpu().numpy().reshape(-1)
                targs_np = targets.cpu().numpy().reshape(-1)
                valid_idx = np.isfinite(targs_np)
                all_predictions.append(preds_np[valid_idx])
                all_targets.append(targs_np[valid_idx])
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Compute additional metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        metrics = self._compute_metrics(predictions, targets_arr)
        metrics['loss'] = avg_loss
        
        logger.info(f"Validation Loss: {avg_loss:.6f}")
        logger.info(f"Validation MAE: {metrics.get('mae', 0):.2f} ms")
        
        return avg_loss, metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        teacher_forcing_schedule: Optional[Callable[[int], float]] = None,
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
        teacher_forcing_schedule : callable, optional
            Function that takes epoch number and returns teacher_forcing_ratio
            If None, uses constant 1.0
        
        Returns
        -------
        dict
            Training history and best metrics
        """
        if teacher_forcing_schedule is None:
            teacher_forcing_schedule = lambda epoch: 1.0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Model: {self.model.model_info()}")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Get teacher forcing ratio for this epoch
            tf_ratio = teacher_forcing_schedule(epoch)
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch, tf_ratio)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            if val_loader is not None:
                val_loss, val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                
                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(epoch, val_loss, val_metrics)
                else:
                    self.epochs_without_improvement += 1
                
                # Early stopping
                if self.epochs_without_improvement >= early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {early_stopping_patience} epochs)"
                    )
                    break
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
        
        logger.info("Training completed")
        return self.history
    
    def _unpack_batch(self, batch: Tuple) -> Tuple:
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

            # Create decoder input from targets
            # Expand 1D targets (batch,) to (batch, 1, 1) for seq2seq
            if targets.dim() == 1:
                decoder_input = targets.unsqueeze(-1).unsqueeze(-1)
            elif targets.dim() == 2:
                decoder_input = targets.unsqueeze(-1)
            else:
                decoder_input = targets.clone()

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
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # MAPE (avoid division by zero)
        mask = targets != 0
        if mask.any():
            mape = np.mean(np.abs((predictions[mask] - targets[mask]) / targets[mask])) * 100
        else:
            mape = 0.0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
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

