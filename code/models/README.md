# F1 Lap Time Prediction - Models & Training

This directory contains the complete training and evaluation pipeline for F1 lap time prediction models.

## Overview

The project implements a sequence-to-sequence (seq2seq) model with GRU encoder-decoder architecture for predicting next lap times in Formula 1 races.

### Architecture

**Seq2Seq GRU Model:**
- **Encoder**: GRU that processes a sequence of past laps
- **Decoder**: GRU that generates predictions for future laps
- **Teacher Forcing**: During training, decoder receives ground-truth outputs
- **Embeddings**: Categorical features (Driver, Team, Circuit, Year) are embedded

### Training Phases

Following the project roadmap, models are trained in phases:

**Phase 1: Pure Teacher Forcing** (Current)
- Stint-based sequences (1-5 laps)
- Seq2seq with full teacher forcing
- Small models to validate architecture
- Training on 2019-2020, validation on 2021, test on 2022

**Phase 2: Full Race Sequences** (Future)
- Full-race sequences with auxiliary heads
- Teacher forcing ratio decay during training
- Larger models with more capacity
- Multi-year training

## Quick Start

### 1. Training a Model

```bash
# Train Phase 1 with default config
python train.py --phase 1

# Train with custom parameters
python train.py --phase 1 --epochs 100 --batch-size 32 --learning-rate 1e-3

# Train with custom config file
python train.py --config custom_config.json

# Use GPU
python train.py --phase 1 --device cuda
```

### 2. Evaluating a Model

```bash
# Evaluate best trained model
python evaluate.py --checkpoint results/phase1/checkpoints/best_model.pt --test-years 2022

# With multiple test years
python evaluate.py --checkpoint best_model.pt --test-years 2022 2023

# Use original config for consistency
python evaluate.py --checkpoint best_model.pt --config results/phase1/config.json
```

### 3. Programmatic Usage

```python
from pathlib import Path
import torch
from code.config import get_phase1_config
from code.models import Seq2SeqGRU, Trainer
from code.dataloaders import StintDataloader
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn

# Load config
config = get_phase1_config()

# Create datasets
train_ds = StintDataloader(year=[2019, 2020], window_size=20)
val_ds = StintDataloader(year=[2021], window_size=20)

# Create dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Create model
model = Seq2SeqGRU(
    input_size=33,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
)

# Create optimizer and trainer
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
trainer = Trainer(model, optimizer, criterion, device='cuda')

# Train
history = trainer.fit(
    train_loader,
    val_loader,
    num_epochs=100,
    early_stopping_patience=15
)

# Evaluate
from code.models import Evaluator
test_loader = DataLoader(val_ds, batch_size=32)
evaluator = Evaluator(model, criterion, device='cuda')
metrics, predictions = evaluator.evaluate(test_loader, return_predictions=True)

print(f"Test MAE: {metrics['mae']:.2f} ms")
print(f"Test RMSE: {metrics['rmse']:.2f} ms")
```

## Configuration

Configuration is managed through dataclass-based config objects in `code/config/base.py`.

### Key Config Parameters

**ModelConfig:**
- `input_size`: Number of input features (33 from dataloaders)
- `hidden_size`: GRU hidden dimension (128)
- `num_layers`: Number of stacked GRU layers (2)
- `dropout`: Dropout rate (0.2)
- `embedding_dims`: Dimensions for categorical embeddings
- `vocab_sizes`: Vocabulary sizes for categorical features

**TrainingConfig:**
- `batch_size`: Training batch size (32)
- `learning_rate`: Learning rate (1e-3)
- `num_epochs`: Training epochs (100)
- `teacher_forcing_start/end`: Teacher forcing ratio schedule
- `early_stopping_patience`: Patience for early stopping (15)

**DataConfig:**
- `window_size`: Max sequence length for stints (20)
- `augment_prob`: Probability of data augmentation (0.3)
- `normalize`: Whether to normalize features (True)

### Saving & Loading Configs

```python
from code.config import Config

# Save
config = get_phase1_config()
config.save(Path("my_config.json"))

# Load
loaded_config = Config.load(Path("my_config.json"))
```

## Data Augmentation

The dataloaders provide automatic augmentation:

**Stint Dataloader:**
1. **Tyre Degradation**: Shift TyreLife ±2 laps
2. **Fuel Variation**: Scale fuel_proxy 0.8-1.2x
3. **Weather Jitter**: Add ±5% noise to weather features

**AutoregressiveLapDataloader:**
1. **Temporal Jitter**: ±2% noise on timing features
2. **Trend Augmentation**: ±3% lap time scaling
3. **Feature Dropout**: 10% weather column dropout

## Output Structure

Training creates the following output:

```
results/phase1/
├── config.json                          # Full configuration
├── history.json                         # Training history (loss curves)
├── checkpoints/
│   ├── best_model.pt                    # Best checkpoint
│   └── metrics_epoch_*.json             # Per-epoch metrics
└── evaluation/
    ├── evaluation_results.json          # Test metrics
    └── evaluation_report.txt            # Human-readable report
```

## Model Checkpoints

Checkpoints save:
```python
{
    'epoch': int,
    'model_state_dict': state_dict,
    'optimizer_state_dict': state_dict,
    'val_loss': float,
    'metrics': dict,
    'model_config': dict,
    'timestamp': str,
}
```

Loading a checkpoint:
```python
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Extensibility

The code is designed for easy extension to future models:

### Adding a New Model Type

1. Create model class inheriting from `BaseModel`:

```python
class MyNewModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Your architecture...
    
    def forward(self, x):
        # Your forward pass...
        return output
```

2. Add to model factory in `train.py`:

```python
if config.model.name == "my_new_model":
    model = MyNewModel(model_config)
```

3. Use same Trainer, Evaluator, Config system

### Adding New Data Augmentations

Edit the augmentation methods in dataloaders:

```python
# In stint_dataloader.py
def _apply_augmentation(self, laps):
    # Modify _apply_augmentation() to add custom strategies
    ...
```

### Adding Custom Metrics

Extend `Evaluator.compute_metrics()`:

```python
# In evaluator.py
def _compute_all_metrics(self, predictions, targets):
    base_metrics = {...}  # Existing metrics
    # Add your custom metrics
    base_metrics['my_metric'] = compute_my_metric(predictions, targets)
    return base_metrics
```

## Key Components

### `code/models/base.py`
- `BaseModel`: Abstract base class for all models
- Provides common interface and utility methods

### `code/models/seq2seq.py`
- `Seq2SeqGRU`: Main model implementation
- Encoder-decoder with teacher forcing support

### `code/models/trainer.py`
- `Trainer`: Generic training loop
- Handles gradient accumulation, learning rate scheduling, checkpointing
- Early stopping and validation

### `code/models/evaluator.py`
- `Evaluator`: Comprehensive evaluation metrics
- Error breakdown and analysis tools

### `code/config/base.py`
- Configuration dataclasses
- Preset configurations for each phase
- Config save/load functionality

### `code/dataloaders/`
- `StintDataloader`: Stint-based sequences
- `AutoregressiveLapDataloader`: Lap-by-lap sequences
- `LapTimeNormalizer`: Data normalization

## Logging

All modules use Python's logging. To see detailed logs:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Known Limitations & Future Work

- [ ] Support for auxiliary prediction heads (pit probability, tyre wear)
- [ ] Multi-task learning with circuit/driver embeddings
- [ ] Uncertainty estimation (aleatoric + epistemic)
- [ ] Attention mechanisms for better feature importance
- [ ] Transformer-based architectures for Phase 3
- [ ] Distributed training support
- [ ] TorchScript export for inference optimization

## References

The architecture is based on attention-free seq2seq models. See:
- Sutskever et al., "Sequence to Sequence Learning with Neural Networks" (2014)
- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014)

## License

See LICENSE file in project root.
