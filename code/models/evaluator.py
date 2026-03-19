"""Evaluation utilities for F1 lap time prediction models."""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
import logging

logger = logging.getLogger(__name__)


def compute_regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics from prediction and target arrays."""
    pred_flat = np.asarray(predictions).reshape(-1)
    target_flat = np.asarray(targets).reshape(-1)

    metrics = {
        'mae': float(np.mean(np.abs(pred_flat - target_flat))),
        'rmse': float(np.sqrt(np.mean((pred_flat - target_flat) ** 2))),
        'median_ae': float(np.median(np.abs(pred_flat - target_flat))),
        'loss': float(np.mean((pred_flat - target_flat) ** 2)),
    }

    mape_mask = target_flat != 0
    if mape_mask.any():
        metrics['mape'] = float(
            np.mean(np.abs((pred_flat[mape_mask] - target_flat[mape_mask]) / target_flat[mape_mask])) * 100
        )
    else:
        metrics['mape'] = 0.0

    errors = np.abs(pred_flat - target_flat)
    metrics['q25_ae'] = float(np.percentile(errors, 25))
    metrics['q50_ae'] = float(np.percentile(errors, 50))
    metrics['q75_ae'] = float(np.percentile(errors, 75))
    metrics['q95_ae'] = float(np.percentile(errors, 95))
    metrics['q99_ae'] = float(np.percentile(errors, 99))

    bias = pred_flat - target_flat
    metrics['mean_bias'] = float(np.mean(bias))
    metrics['median_bias'] = float(np.median(bias))
    return metrics


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
                # Move targets tensors to device if targets is a dict
                if isinstance(targets, dict):
                    for k, v in list(targets.items()):
                        if isinstance(v, torch.Tensor):
                            targets[k] = v.to(self.device)
                else:
                    targets = targets.to(self.device)

                # If model uses decoder future features, pad decoder_input with zeros
                # when future features are not provided by the dataloader.
                n_fut = getattr(self.model, 'decoder_future_features', 0)
                if n_fut > 0 and decoder_input.size(-1) < 1 + n_fut:
                    pad = torch.zeros(
                        *decoder_input.shape[:-1], n_fut,
                        dtype=decoder_input.dtype, device=decoder_input.device,
                    )
                    decoder_input = torch.cat([decoder_input, pad], dim=-1)

                # Forward pass with autocast for faster inference
                with autocast(device_type='cuda', enabled=self.device == 'cuda'):
                    outputs = self.model(encoder_input, decoder_input, teacher_forcing=False)

                # Handle multi-head outputs (dict) or single tensor
                if isinstance(outputs, dict):
                    lap_pred = outputs.get('lap')
                    pit_logits = outputs.get('pit_logits', None)
                    comp_logits = outputs.get('compound_logits', None)
                else:
                    lap_pred = outputs
                    pit_logits = None
                    comp_logits = None

                # Prepare target lap times
                if isinstance(targets, dict):
                    lap_t = targets['lap_time']
                    pit_t = targets.get('is_pitlap', None)
                    comp_t = targets.get('compound', None)
                else:
                    lap_t = targets
                    pit_t = None
                    comp_t = None

                if lap_t.dim() == 1:
                    targets_reshaped = lap_t.unsqueeze(-1).unsqueeze(-1)
                else:
                    targets_reshaped = lap_t

                # Convert lap predictions/targets to numpy and flatten per-sample for masking
                preds_np = lap_pred.cpu().numpy().reshape(-1)
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

                # If pit predictions exist, store them and targets
                if pit_logits is not None and pit_t is not None:
                    pit_probs = torch.sigmoid(pit_logits).cpu().numpy().reshape(-1)
                    pit_targets = pit_t.cpu().numpy().reshape(-1)
                    # Apply same valid_mask (based on lap targets) to align samples
                    pit_valid = np.isfinite(pit_targets) & np.isfinite(pit_probs)
                    # store under metadata keys for downstream aggregation
                    # Attach to all_metadata entries in same order
                    # We append aggregated arrays at the end of metadata list for simplicity
                    # Collect pit per-batch
                    try:
                        existing = batch_pit_preds
                    except NameError:
                        batch_pit_preds = []
                        batch_pit_targs = []
                    batch_pit_preds.append(pit_probs[pit_valid])
                    batch_pit_targs.append(pit_targets[pit_valid])

                if comp_logits is not None and comp_t is not None:
                    comp_preds = np.argmax(comp_logits.cpu().numpy(), axis=-1).reshape(-1)
                    comp_targets = comp_t.cpu().numpy().reshape(-1)
                    comp_valid = np.isfinite(comp_targets) & np.isfinite(comp_preds)
                    try:
                        existing = batch_comp_preds
                    except NameError:
                        batch_comp_preds = []
                        batch_comp_targs = []
                    batch_comp_preds.append(comp_preds[comp_valid])
                    batch_comp_targs.append(comp_targets[comp_valid])

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

        # Aggregate pit/compound if collected
        pit_pred_arr = None
        pit_targ_arr = None
        comp_pred_arr = None
        comp_targ_arr = None
        try:
            if 'batch_pit_preds' in locals():
                pit_pred_arr = np.concatenate(batch_pit_preds, axis=0)
                pit_targ_arr = np.concatenate(batch_pit_targs, axis=0)
        except Exception:
            pit_pred_arr = None
            pit_targ_arr = None

        try:
            if 'batch_comp_preds' in locals():
                comp_pred_arr = np.concatenate(batch_comp_preds, axis=0)
                comp_targ_arr = np.concatenate(batch_comp_targs, axis=0)
        except Exception:
            comp_pred_arr = None
            comp_targ_arr = None

        # Attach to predictions for external evaluation convenience
        # (keeps backward compatibility if keys absent)

        # Compute metrics
        metrics = compute_regression_metrics(predictions, targets) if total_count > 0 else {
            'mae': float('nan'), 'rmse': float('nan'), 'median_ae': float('nan'), 'mape': float('nan')
        }
        metrics['loss'] = float(sum_se / total_count) if total_count > 0 else float('nan')
        
        logger.info(f"Test Loss: {metrics['loss']:.6f}")
        logger.info(f"Test MAE (normalized): {metrics['mae']:.4f}")
        logger.info(f"Test RMSE (normalized): {metrics['rmse']:.4f}")
        logger.info(f"Test MAPE: {metrics['mape']:.2f}%")
        
        if return_predictions:
            pred_dict = {
                'predictions': predictions,
                'targets': targets,
                'metadata': all_metadata,
                'pit_predictions': pit_pred_arr,
                'pit_targets': pit_targ_arr,
                'compound_predictions': comp_pred_arr,
                'compound_targets': comp_targ_arr,
            }
            return metrics, pred_dict
        else:
            return metrics, None
    
def report_evaluation(
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
    unit_label: str = "normalized",
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
    report.append(f"  MAE:          {metrics.get('mae', 0):.2f} {unit_label}")
    report.append(f"  RMSE:         {metrics.get('rmse', 0):.2f} {unit_label}")
    report.append(f"  Median AE:    {metrics.get('median_ae', 0):.2f} {unit_label}")
    report.append(f"  MAPE:         {metrics.get('mape', 0):.2f}%")
    report.append(f"  Loss:         {metrics.get('loss', 0):.6f}")
    
    report.append("\nQuantile Absolute Error:")
    for q in [25, 50, 75, 95, 99]:
        key = f'q{q}_ae'
        report.append(f"  {q}th percentile: {metrics.get(key, 0):.2f} {unit_label}")
    
    report.append("\nBias:")
    report.append(f"  Mean bias:    {metrics.get('mean_bias', 0):.2f} {unit_label}")
    report.append(f"  Median bias:  {metrics.get('median_bias', 0):.2f} {unit_label}")
    
    report_str = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_str)
        logger.info(f"Report saved to {save_path}")
    
    return report_str
