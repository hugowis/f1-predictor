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

## Current Solution: Phase 1 (✅ Complete)

### Architecture
- **Model**: Seq2Seq with GRU encoder-decoder
- **Parameters**: 363,921 trainable parameters
- **Hidden size**: 128 units, 2 stacked layers, 0.2 dropout
- **Teacher forcing**: 100% during training (Phase 1 approach)
- **Sequence length**: 20 laps max per stint

### Training Details
- **Data**: 5,311 stints (2019-2023 seasons)
  - Train: 2019-2023 (5,311 sequences)
  - Validation: 2024 (1,235 sequences)
  - Test: 2025 (1,225 sequences)
- **Hardware**: NVIDIA RTX 5080 (13.0 CUDA)
- **Optimizer**: SGD with learning rate 1e-3, cosine scheduler (5 warm-up epochs)
- **Loss**: MSE with gradient clipping (1.0)
- **Early stopping**: Patience=15 epochs → stopped at epoch 60

 - **Batch size**: 32 (this evaluation run)
 - **Early stopping**: Patience=15 epochs → stopped at epoch 57

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


## Quick Start - Training Pipeline

All training and evaluation scripts are located in the `code/` folder for better organization.

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

**Deliverables:**
- Trained model checkpoint: `results/phase1/checkpoints/best_model.pt`
- Training history: `results/phase1/history.json`
- Test metrics: `results/phase1/evaluation/evaluation_results.json`
- Visualizations: `.png` files in `results/phase1/`
- Configuration: `results/phase1/config.json`

---

### 🚀 Phase 2: In Planning

**Objectives:**
- [ ] Extend sequence length to full races, masking for not normal laps
- [ ] Implement autoregressive predictions (free-running mode)
- [ ] Auxiliary pit head and compound head
- [ ] Scheduled sampling (gradual removal of teacher forcing)
- ✅ Error analysis by driver and circuit

**Expected improvements:**
- Predictions across full race (not just stints)
- Handling of strategy changes and pit stops
- Driver-specific lap time signatures

---

### 🔮 Phase 3: Future Enhancements

**Long-term items:**
- [ ] Transformer-based architecture
- [ ] Uncertainty estimation (mean + variance predictions)
- [ ] Web dashboard
- [ ] Real-time prediction pipeline for races
- [ ] Pretraining on FP/qualifying data for better generalization
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
- [ ] Full-race sequences
- [ ] Autoregressive generation
- [ ] Pit stop modeling (head for pit lap prediction)
- [ ] Compound modeling (head for tire compound prediction)
- [ ] Transformer decoder

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
- [ ] Prediction dashboard
---

## Experiments
- ✅ Add 2018 season data
- [ ] Test with dropped features (sectors, speeds, position)
- [ ] Pretraining on FP sessions, qualifycation data, sprin, etc.



