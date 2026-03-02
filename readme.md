# F1 Lap Time Predictor

A deep learning solution to predict **next lap times in Formula 1 racing** using historical lap data and race telemetry.

## Overview

This project builds a **sequence-to-sequence (seq2seq) model** to solve the lap time prediction problem:

> **Given**: Lap times and telemetry data from laps 1 to t  
> **Predict**: Lap times for lap t+1, t+2, ... within a stint  

**Key constraints:**
- Only information available *up to lap t* can be used (no data leakage)
- Focus on race laps (not qualifying sessions)
- Must handle varying stint lengths, pit stops, and track conditions
- Real-world applicability to race strategy and driver performance analysis

---

## Quick Start - Training Pipeline

All training and evaluation scripts are located in the `code/` folder for better organization.

### Prepare the data

```bash
# Download data
python code/data/data_downloader.py 
# Prepare it
python code/data/data_preparation.py
```

### Precomputed / Cached Dataloader (new)

To speed up development and testing you can persist the autoregressive dataloader's
precomputed context tensors to disk so subsequent runs load instantly.

- Default behavior: the `AutoregressiveLapDataloader` will attempt to load a cache
  file under `data/precomputed/` named like `ar_cache_<years>_cw<context>_<scaler>.pt`.
- You can override or explicitly set the cache file with the `cache_path` parameter.
- The dataloader validates cache compatibility (years, `context_window`, `scaler_type`, numeric columns).
- To skip precomputation entirely (interactive debugging), set the environment variable `SKIP_PRECOMPUTE=1`.

Examples

Python API (use explicit cache path):

```python
from pathlib import Path
from code.dataloaders.autoregressive_dataloader import AutoregressiveLapDataloader

ds = AutoregressiveLapDataloader(
    year=[2019, 2020],
    context_window=5,
    cache_path=Path('data/precomputed/my_ar_cache.pt'),
    scaler_type='standard',
    device='cpu'
)
# On first instantiation this will precompute tensors and save the cache.
# On subsequent runs it will load the cache (if compatible) which is much faster.
```

Quick precompute from Python (bulk):

```python
from pathlib import Path
from code.dataloaders.autoregressive_dataloader import AutoregressiveLapDataloader

def build_cache(years, cw):
    p = Path(f'data/precomputed/ar_cache_{"_".join(map(str, years))}_cw{cw}_standard.pt')
    ds = AutoregressiveLapDataloader(year=years, context_window=cw, cache_path=p)

# Example: precompute caches for 2018-2020 with cw=5
build_cache([2018,2019,2020], 5)
```

Clearing cache

```bash
rm -f data/precomputed/ar_cache_*
```

Notes

- The cache stores context arrays (numpy), target primitives and metadata. If you change
  the dataloader feature set (numeric columns) or scaler type, the cache will be ignored
  and recomputed.
- Keep an eye on disk usage for very large year ranges; caches can become large when
  precomputing many races.


### Training a Model

```bash
# Train Phase 1 model (default config)
python code/train.py --phase 1 --device cuda

# Train with custom batch size and epochs
python code/train.py --phase 1 --batch-size 16 --epochs 100 --device cuda

# Train with custom config file
python code/train.py --config path/to/config.json --device cuda
```

### Evaluating a Model

```bash
# Evaluate checkpoint on 2025 test data
python code/evaluate.py --checkpoint results/phase1/checkpoints/best_model.pt --device cuda

# Evaluate on specific years
python code/evaluate.py --checkpoint results/phase1/checkpoints/best_model.pt --test-years 2022 2023 2024
```

### Visualizing Results

```bash

# Generate error analysis report, loss curves and error distribution plots
python code/analyze_results.py
```

All results are saved to `results/phase1/` including:
- `loss_curves.png` - Training/validation loss progression
- `error_breakdown.png` - Error distribution visualization
- `test_metrics_summary.png` - Test performance metrics
- `analysis_report.txt` - Comprehensive analysis report
- `evaluation_results.json` - Detailed evaluation metrics

---

## Data

All data is obtained using the **FastF1** Python framework and downloaded via:
```code/data/data_downloader.py```


### Raw Lap Data (FastF1)

Each row corresponds to a single lap for a driver in a race and includes:

- Lap timing information
- Sector times
- Speed traps
- Tyre information
- Driver and team metadata
- Track status flags (yellow flags, safety car, etc.)

Example columns:

```Time, Driver, LapTime, LapNumber, Stint, Sector1Time, Sector2Time, Sector3Time,SpeedI1, SpeedI2, SpeedFL, SpeedST, Compound, TyreLife, FreshTyre, Team, TrackStatus, Position, PitInTime, PitOutTime, ...```


### Additional Data

The following data sources are joined at lap level:

- **Weather data**
  - Track temperature
  - Air temperature
  - Humidity
  - Rain
  - Wind speed (if available)
- **Track / schedule metadata**
  - Circuit name
  - Track length
  - Season / year
  - Race total laps

---

## Task Definition

**Primary task**:  
Predict the **next lap times** for a given driver.

- Input: laps `[t-N, ..., t]`
- Output: `LapTime(t+1), LapTime(t+2), ...`
- Only information available **up to lap t** is allowed (no data leakage)

---

## Project Status & Roadmap

### ✅ Phase 1: Complete

**Completed tasks:**
- ✅ Data loading and preprocessing (stint-based sequences)
- ✅ Seq2Seq GRU model architecture
- ✅ Training infrastructure (trainer, scheduler, early stopping)
- ✅ Evaluation metrics (MAE, RMSE, MAPE, quantiles)
- ✅ GPU support (CUDA 13.0, automatic device detection)
- ✅ Masking for missing/non-finite data
- ✅ Full training pipeline (100 epochs, converged at epoch 60)
- ✅ Model checkpointing and restoration
- ✅ Visualization suite (loss curves, error breakdown, metrics plots)
- ✅ Comprehensive evaluation reports

---
### Performance Results
| Metric | Value |
|--------|-------|
| **MAE** | 51.10 ms |
| **RMSE** | 92.38 ms |
| **Median AE** | 31.92 ms |
| **MAPE** | 4.22 % |

### Error Distribution
- **0-10 ms** (Very Accurate): 16.74%
- **10-50 ms** (Accurate): 52.17% ⭐ *Most common*
- **50-100 ms** (Good): 20.18%
- **100-200 ms** (Fair): 7.38%
- **200+ ms** (Poor): 3.53%

**Key insight**: 68.91% of predictions have <50ms error (16.74% + 52.17%), demonstrating improved accuracy for the bs32 training run.

---


**LSTM vs GRU Comparison (concise)**: The LSTM Phase 1 run achieves similar MAE to the GRU run but shows notably lower RMSE and MAPE, while the median absolute error is slightly higher — see the GRU summary above and the LSTM metrics here for direct comparison.

### ✅ Phase 2: Complete

**Completed tasks:**
- ✅ Extend sequence length to full races
- ✅ Implement autoregressive predictions (free-running mode)
- ✅ Auxiliary pit head and compound head
- ✅ Scheduled sampling (gradual removal of teacher forcing)
- ✅ Error analysis by driver and circuit

**Current best test metrics with  compound-loss-weight = 0.01 and pit-loss-weight = 0.0:**
- **MAE:** 23.05 ms
- **RMSE:** 40.97 ms
- **Median AE:** 15.68 ms
- **MAPE:** 0.0256 %
- **Mean bias:** 3.20 ms

**Error distribution (percent):**
- 0-10 ms: 33.80%
- 10-50 ms: 56.85%
- 50-100 ms: 6.89%
- 100-200 ms: 2.05%
- 200+ ms: 0.41%

### Analysis & comparison with Phase 1

- **Summary:** The current best Phase 2 run substantially improves core metrics versus the Phase 1 baseline. MAE falls from 51.10 ms → 23.05 ms (~55% reduction), RMSE from 92.38 ms → 40.97 ms (~56% reduction), and median AE from 31.92 ms → 15.68 ms (~51% reduction).
- **Error distribution shift:** Predictions with error <50 ms increase from ~68.9% (Phase 1) to ~90.7% (Phase 2), indicating a marked shift toward tighter, more reliable predictions. 
- **Likely contributors:** autoregressive/free‑running training (reduces exposure bias), auxiliary heads for pit/compound (better modeling of strategy changes), scheduled sampling and full‑race context (longer sequences improve robustness).

---

### 🔮 Phase 3: Future Enhancements

**Long-term items:**
- [ ] Transformer-based architecture
- [ ] Uncertainty estimation (mean + variance predictions)
- [ ] Pretraining on FP/qualifying data for better generalization (train pit head)

### Additional experiments
- [ ] Web dashboard
- [ ] Real-time prediction pipeline for races
- [ ] Scenario-based predictions (e.g., "What if I pit on lap 10?")


---

## Detailed TODO List

### Data Pipeline
- ✅ Lap data loading (FastF1 framework)
- ✅ Weather data integration
- ✅ Track metadata joining
- ✅ Stint-based sequence creation
- ✅ Data normalization (per-year StandardScaler)
- ✅ Masking for non-finite values
- ✅ Add 2018 season (different compounds and missing data)

### Modeling - Phase 1
- ✅ GRU encoder-decoder architecture
- ✅ Teacher forcing (full schedule)
- ✅ Stint-based sequences (1-20 laps)
- ✅ Multi-layer RNN (2 layers)
- ✅ Dropout and gradient clipping
- ✅ LSTM variant comparison
- ✅ Driver embeddings
- ✅ Team/car embeddings

### Modeling - Phase 2
- [✅] Full-race sequences
- [✅] Autoregressive
- [✅] Pit stop modeling (head for pit lap prediction)
- [✅] Compound modeling (head for tire compound prediction)

### Modeling - Phase 3
- [ ] Transformer-based architecture
- [ ] Uncertainty estimation (mean + variance predictions)

### Training & Optimization
- ✅ PyTorch training loop
- ✅ GPU acceleration
- ✅ Learning rate scheduling (cosine annealing)
- ✅ Early stopping with patience
- ✅ Gradient accumulation support
- ✅ Checkpointing and restoration

### Evaluation & Analysis
- ✅ Test set evaluation
- ✅ Denormalization of predictions
- ✅ MAE/RMSE/MAPE/Quantiles computation
- ✅ Error breakdown by ranges
- ✅ Loss curve visualization
- ✅ Error distribution plots
- ✅ Driver-level error analysis
- ✅ Circuit-level error analysis
- ✅ Team-level error analysis
- ✅ Compound-specific analysis

### Infrastructure & Deployment
- ✅ Modular code structure
- ✅ Configuration system (dataclasses)
- ✅ Training script with CLI
- ✅ Evaluation script with checkpointing
- ✅ Visualization suite
- ✅ Git ignore for generated files
- [ ] Web dashboard
---

## Experiments
- ✅ Add 2018 season data
- [ ] Pretraining on FP sessions, qualifying data, sprint races, etc.
- [ ] Scenario-based predictions (e.g., "What if I pit on lap 10?")

## Further work
- [ ] Web dashboard
- [ ] Real-time prediction pipeline for races
- [ ] Scenario-based predictions (e.g., "What if I pit on lap 10?")