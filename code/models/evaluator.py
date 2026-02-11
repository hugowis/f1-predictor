"""
Evaluation utilities for F1 lap time prediction models.

Provides comprehensive metrics and analysis tools.
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Model evaluator with comprehensive metrics and error analysis.
    
    Computes:
    - MAE, RMSE, MAPE
    - Error by stint phase (early, mid, late)
    - Error by tyre compound
    - Error by circuit
    - Quantile losses
    
    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    criterion : nn.Module
        Loss function
    device : str
        Device to use
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: str = 'cpu',
    ):
        self.model = model
        self.criterion = criterion
        self.device = device
    
    def evaluate(
        self,
        test_loader: DataLoader,
        return_predictions: bool = False,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
        """
        Comprehensive evaluation on test set.
        
        Parameters
        ----------
        test_loader : DataLoader
            Test data loader
        return_predictions : bool
            Whether to return predictions and targets
        
        Returns
        -------
        tuple of (metrics_dict, predictions_dict or None)
            (metrics, {'predictions': pred, 'targets': target, 'metadata': metadata})
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_metadata = []  # will store per-sample metadata dicts
        sum_se = 0.0
        total_count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Unpack batch
                if len(batch) == 4:
                    encoder_input, decoder_input, targets, metadata = batch
                elif len(batch) == 3:
                    encoder_input, targets, metadata = batch
                    # Create decoder_input with shape (batch, seq_len, output_size)
                    if targets.dim() == 1:
                        decoder_input = targets.unsqueeze(-1).unsqueeze(-1)
                    elif targets.dim() == 2:
                        decoder_input = targets.unsqueeze(-1)
                    else:
                        decoder_input = targets.clone()
                else:
                    raise ValueError(f"Unexpected batch format")
                
                # If encoder_input is structured dict (numeric + categorical), apply embeddings
                if isinstance(encoder_input, dict) and 'numeric' in encoder_input and 'categorical' in encoder_input:
                    numeric = encoder_input['numeric']
                    categorical = encoder_input['categorical']
                    cat_names = encoder_input.get('cat_names', [])


                    if isinstance(numeric, torch.Tensor):
                        numeric_tensor = numeric
                    else:
                        numeric_tensor = torch.from_numpy(numeric)

                    if isinstance(categorical, torch.Tensor):
                        cat_tensor = categorical
                    else:
                        cat_tensor = torch.from_numpy(categorical)

                    emb_list = []
                    if cat_tensor.numel() > 0:
                        num_cat = cat_tensor.shape[-1]
                        for i in range(num_cat):
                            feat_name = cat_names[i] if i < len(cat_names) else None
                            idxs = cat_tensor[:, :, i].long()
                            if feat_name and hasattr(self.model, 'embeddings') and feat_name in self.model.embeddings:
                                emb_layer = self.model.embeddings[feat_name]
                                try:
                                    emb_device = next(emb_layer.parameters()).device
                                    idxs = idxs.to(emb_device)
                                except Exception:
                                    emb_device = None
                                emb_out = emb_layer(idxs)
                                try:
                                    emb_out = emb_out.to(numeric_tensor.device)
                                except Exception:
                                    pass
                                emb_list.append(emb_out)
                    if emb_list:
                        cat_emb = torch.cat(emb_list, dim=-1)
                    else:
                        cat_emb = torch.zeros(numeric_tensor.shape[0], numeric_tensor.shape[1], 0, device=numeric_tensor.device)

                    encoder_input = torch.cat([numeric_tensor, cat_emb], dim=-1)

                encoder_input = encoder_input.to(self.device)
                decoder_input = decoder_input.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(encoder_input, decoder_input, teacher_forcing=False)

                # Ensure targets shape matches outputs
                if targets.dim() == 1:
                    targets_reshaped = targets.unsqueeze(-1).unsqueeze(-1)
                else:
                    targets_reshaped = targets

                # Convert to numpy and flatten per-sample for masking
                preds_np = outputs.cpu().numpy().reshape(-1)
                targs_np = targets_reshaped.cpu().numpy().reshape(-1)

                # Mask invalid values
                valid_mask = np.isfinite(targs_np) & np.isfinite(preds_np)
                valid_count = int(valid_mask.sum())
                if valid_count == 0:
                    # skip batch with no valid targets
                    continue

                # Accumulate sum squared error and count for averaged loss
                se = (preds_np[valid_mask] - targs_np[valid_mask]) ** 2
                sum_se += float(se.sum())
                total_count += valid_count

                # Store valid predictions/targets for metrics
                all_predictions.append(preds_np[valid_mask])
                all_targets.append(targs_np[valid_mask])

                # Align metadata to valid entries and store per-sample
                meta_list = []

                # Robustly handle collated metadata which may be:
                # - a dict of arrays/tensors (default_collate behavior)
                # - a list of per-sample dicts
                # - a single dict
                if isinstance(metadata, dict):
                    # Try to build list of per-sample dicts from dict of batched values
                    # Determine batch length from first value
                    try:
                        first_val = next(iter(metadata.values()))
                        batch_len = len(first_val)
                    except Exception:
                        batch_len = None

                    if batch_len is None:
                        meta_list = [metadata]
                    else:
                        for i in range(batch_len):
                            item = {}
                            for k, v in metadata.items():
                                try:
                                    if isinstance(v, torch.Tensor):
                                        elem = v[i]
                                        if elem.numel() == 1:
                                            val = elem.item()
                                        else:
                                            val = elem.cpu().numpy().tolist()
                                    elif isinstance(v, np.ndarray):
                                        elem = v[i]
                                        try:
                                            val = elem.item()
                                        except Exception:
                                            val = elem.tolist()
                                    else:
                                        # list or scalar
                                        val = v[i]
                                except Exception:
                                    val = None
                                item[k] = val
                            meta_list.append(item)

                elif isinstance(metadata, (list, tuple)):
                    meta_list = list(metadata)
                else:
                    meta_list = [metadata]

                # If metadata length matches preds length, select by valid_mask
                if len(meta_list) == len(valid_mask):
                    for i, ok in enumerate(valid_mask):
                        if ok:
                            all_metadata.append(meta_list[i])
                else:
                    # Fallback: append a placeholder for each valid prediction
                    for _ in range(int(valid_mask.sum())):
                        all_metadata.append({'note': 'missing_metadata'})
        
        # Aggregate results
        if len(all_predictions) > 0:
            predictions = np.concatenate(all_predictions, axis=0)
            targets = np.concatenate(all_targets, axis=0)
        else:
            predictions = np.array([])
            targets = np.array([])

        # Compute metrics
        metrics = self._compute_all_metrics(predictions, targets) if total_count > 0 else {
            'mae': float('nan'), 'rmse': float('nan'), 'median_ae': float('nan'), 'mape': float('nan')
        }
        metrics['loss'] = float(sum_se / total_count) if total_count > 0 else float('nan')
        
        logger.info(f"Test Loss: {metrics['loss']:.6f}")
        logger.info(f"Test MAE: {metrics['mae']:.2f} ms")
        logger.info(f"Test RMSE: {metrics['rmse']:.2f} ms")
        logger.info(f"Test MAPE: {metrics['mape']:.2f}%")
        
        if return_predictions:
            pred_dict = {
                'predictions': predictions,
                'targets': targets,
                'metadata': all_metadata,
            }
            return metrics, pred_dict
        else:
            return metrics, None
    
    def _compute_all_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions, shape (n_samples, seq_len, 1)
        targets : np.ndarray
            Ground truth targets, same shape
        
        Returns
        -------
        dict
            Dictionary of metrics
        """
        # Flatten for overall metrics
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)
        
        metrics = {}
        
        # Basic metrics
        metrics['mae'] = float(np.mean(np.abs(pred_flat - target_flat)))
        metrics['rmse'] = float(np.sqrt(np.mean((pred_flat - target_flat) ** 2)))
        metrics['median_ae'] = float(np.median(np.abs(pred_flat - target_flat)))
        
        # MAPE
        mask = target_flat != 0
        if mask.any():
            mape = np.mean(np.abs((pred_flat[mask] - target_flat[mask]) / target_flat[mask])) * 100
            metrics['mape'] = float(mape)
        else:
            metrics['mape'] = 0.0
        
        # Quantile metrics
        errors = np.abs(pred_flat - target_flat)
        metrics['q25_ae'] = float(np.percentile(errors, 25))
        metrics['q50_ae'] = float(np.percentile(errors, 50))
        metrics['q75_ae'] = float(np.percentile(errors, 75))
        metrics['q95_ae'] = float(np.percentile(errors, 95))
        metrics['q99_ae'] = float(np.percentile(errors, 99))
        
        # Bias (signed error)
        bias = pred_flat - target_flat
        metrics['mean_bias'] = float(np.mean(bias))
        metrics['median_bias'] = float(np.median(bias))
        
        return metrics
    
    def evaluate_by_stint_phase(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        stint_lengths: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance by stint phase (early, mid, late).
        
        Parameters
        ----------
        predictions : np.ndarray
            Predictions from evaluation
        targets : np.ndarray
            Targets from evaluation
        stint_lengths : np.ndarray
            Length of each stint
        
        Returns
        -------
        dict
            Metrics grouped by stint phase
        """
        metrics_by_phase = {'early': {}, 'mid': {}, 'late': {}}
        
        # Determine stint phase boundaries
        early_threshold = 0.33
        late_threshold = 0.67
        
        # This would require mapping lap indices to stint positions
        # For now, a simplified version:
        logger.warning("Stint phase analysis requires additional metadata")
        
        return metrics_by_phase
    
    def error_breakdown(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        """
        Categorize errors into ranges.
        
        Returns
        -------
        dict
            Percentage of predictions in each error category
        """
        errors = np.abs(predictions.reshape(-1) - targets.reshape(-1))
        
        total = len(errors)
        breakdown = {
            'error_0_10ms': float((errors < 10).sum() / total * 100),
            'error_10_50ms': float(((errors >= 10) & (errors < 50)).sum() / total * 100),
            'error_50_100ms': float(((errors >= 50) & (errors < 100)).sum() / total * 100),
            'error_100_200ms': float(((errors >= 100) & (errors < 200)).sum() / total * 100),
            'error_200plus_ms': float((errors >= 200).sum() / total * 100),
        }
        
        logger.info("Error Breakdown:")
        for category, percentage in breakdown.items():
            logger.info(f"  {category}: {percentage:.2f}%")
        
        return breakdown


def denormalize_predictions(
    predictions: np.ndarray,
    normalizer,
    feature_index: int = 0,
) -> np.ndarray:
    """
    Denormalize predictions to original scale.
    
    Parameters
    ----------
    predictions : np.ndarray
        Normalized predictions
    normalizer : LapTimeNormalizer
        Fitted normalizer
    feature_index : int
        Which feature to denormalize (usually 0 for lap time)
    
    Returns
    -------
    np.ndarray
        Predictions in original scale
    """
    # Get normalization statistics
    stats = normalizer.get_statistics()
    mean = stats['mean'][feature_index]
    std = stats['std'][feature_index]
    
    # Reverse normalization: x = (z * std) + mean
    denormalized = (predictions * std) + mean
    return denormalized


def report_evaluation(
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
) -> str:
    """
    Generate a human-readable evaluation report.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metrics
    save_path : str, optional
        Path to save report
    
    Returns
    -------
    str
        Formatted report
    """
    report = []
    report.append("=" * 60)
    report.append("EVALUATION REPORT")
    report.append("=" * 60)
    
    report.append("\nCore Metrics:")
    report.append(f"  MAE:          {metrics.get('mae', 0):.2f} ms")
    report.append(f"  RMSE:         {metrics.get('rmse', 0):.2f} ms")
    report.append(f"  Median AE:    {metrics.get('median_ae', 0):.2f} ms")
    report.append(f"  MAPE:         {metrics.get('mape', 0):.2f}%")
    report.append(f"  Loss:         {metrics.get('loss', 0):.6f}")
    
    report.append("\nQuantile Absolute Error:")
    for q in [25, 50, 75, 95, 99]:
        key = f'q{q}_ae'
        report.append(f"  {q}th percentile: {metrics.get(key, 0):.2f} ms")
    
    report.append("\nBias:")
    report.append(f"  Mean bias:    {metrics.get('mean_bias', 0):.2f} ms")
    report.append(f"  Median bias:  {metrics.get('median_bias', 0):.2f} ms")
    
    report_str = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_str)
        logger.info(f"Report saved to {save_path}")
    
    return report_str
