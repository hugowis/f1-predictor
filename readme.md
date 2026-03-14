# F1 Lap Time Predictor

Deep learning project to predict next-lap F1 race pace from historical lap, weather, and race-context features.

## Highlights

- Seq2Seq recurrent models (GRU/LSTM) for lap-time forecasting.
- Two training modes:
  - Phase 1: stint-based supervised forecasting.
  - Phase 2: autoregressive full-race forecasting with auxiliary pit/compound heads.
- End-to-end pipeline:
  - data download, cleaning, training, evaluation, and analysis plots/reports.
- Built-in safeguards:
  - no-leakage feature design, normalization consistency checks, early stopping, checkpointing.

## Problem Definition

Given race data up to lap `t` for a driver, predict future lap time(s), while using only information available at or before `t`.

- Input: lap history window and race context.
- Output: next lap time (and, in autoregressive mode, iterative future lap times).
- Constraints:
  - no future leakage,
  - varying stint lengths,
  - pit events and compound changes,
  - weather and circuit variability.

## Latest Results

Results are read from tracked experiment outputs in `results/*/evaluation/evaluation_results.json`.

### Best Run (current best in repository)

 - Run: `results/step2_compound_0.01/`
 - Config family: Phase 2 autoregressive, `compound_loss_weight=0.01`, `pit_loss_weight=0.001`

| Metric | Value |
|---|---:|
| MAE (ms) | 23.05 |
| RMSE (ms) | 40.97 |
| Median AE (ms) | 15.68 |
| MAPE (%) | 0.0256 |
| Mean Bias (ms) | 3.20 |
| Errors < 50 ms (%) | 90.65 |

### Latest Run

- Run: `results/step6b_150ep_aug0.20/`
- Config family: Phase 2 autoregressive, `compound_loss_weight=0.01`, `pit_loss_weight=0.001`

| Metric | Value |
|---|---:|
| MAE (ms) | 29.43 |
| RMSE (ms) | 59.20 |
| Median AE (ms) | 16.61 |
| MAPE (%) | 0.0315 |
| Mean Bias (ms) | 12.91 |
| Errors < 50 ms (%) | 88.43 |



## Repository Structure

```text
code/
  train.py
  evaluate.py
  analyze_results.py
  config/
  data/
  dataloaders/
  models/
data/
  raw_data/
  clean_data/
  precomputed/
  vocabs/
results/
  <experiment_name>/
```

## Installation

### Requirements

- Python 3.10+
- Optional GPU for faster training (CUDA-compatible PyTorch)

### 1. Clone

```bash
git clone <your-repo-url>
cd f1-predictor
```

### 2. Create Environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

### 1. Download and prepare data

```bash
python code/data/data_downloader.py
python code/data/data_preparation.py
```

### 2. Train

Phase 1 (stint mode):

```bash
python code/train.py --phase 1 --device cuda
```

Phase 2 (autoregressive mode):

```bash
python code/train.py --phase 2 --autoregressive --device cuda
```

Custom run example:

```bash
python code/train.py \
  --phase 2 \
  --autoregressive \
  --epochs 150 \
  --batch-size 64 \
  --compound-loss-weight 0.01 \
  --pit-loss-weight 0.0 \
  --augment-prob 0.20 \
  --output results/my_run
```

### 3. Evaluate a checkpoint

```bash
python code/evaluate.py \
  --checkpoint results/my_run/checkpoints/best_model.pt \
  --config results/my_run/config.json \
  --test-years 2025 \
  --device cuda
```

### 4. Generate analysis plots/reports

```bash
python code/analyze_results.py --run my_run
```

## Reproducibility

This project is designed for practical reproducibility, but exact bitwise reproduction across different machines/driver stacks is not guaranteed.

### What is already implemented

- Fixed random seed support (`--seed` in `code/train.py`).
- Deterministic CuDNN settings enabled in training script.
- Config snapshot is saved per run (`results/<run>/config.json`).
- Training history, checkpoints, evaluation outputs, and reports are persisted under each run directory.

### Recommended reproducibility workflow

1. Pin environment:
   - use a fresh virtual environment,
   - install with `pip install -r requirements.txt`.
2. Use explicit run names and seed:
   - `--seed 42 --output results/repro_seed42`.
3. Reuse cached/precomputed data only when feature schema is unchanged.
4. Report both:
   - exact command used,
   - `results/<run>/config.json` and `history.json`.
5. Keep train/val/test year splits identical across comparisons.

### Example reproducible command

```bash
python code/train.py \
  --phase 2 \
  --autoregressive \
  --seed 42 \
  --epochs 150 \
  --batch-size 64 \
  --compound-loss-weight 0.01 \
  --pit-loss-weight 0.001 \
  --output results/repro_phase2_seed42
```

### Multi-seed launcher

Use the launcher when you want to repeat the same training configuration across several seeds and optionally run them in parallel.

```bash
python code/launch_seed_experiments.py \
  --config results/step2_compound_0.01/config.json \
  --seeds 42 123 789 \
  --batch-size 128 \
  --mode parallel \
  --max-parallel 3 \
  --output-root results/step2_multiseed
```

Any extra CLI flags that are not consumed by the launcher are forwarded to `code/train.py`, so you can still sweep things like `--epochs`, `--augment-prob`, or `--compound-loss-weight`. The launcher creates one subdirectory per seed, keeps a `launch_manifest.json`, and writes `leaderboard.csv` plus `leaderboard.json` at the output root after all runs finish.

### Grid-search launcher

Use the grid-search wrapper when you want to sweep several launcher or training arguments at once while keeping the existing multi-seed workflow.

```bash
python code/grid_search_experiments.py \
  --search-root results/tf_schedule_grid \
  --grid teacher-forcing-decay=linear,exponential \
  --grid teacher-forcing-hold-epochs=0,10,20 \
  --grid teacher-forcing-end=0.5,0.3,0.0 \
  --grid epochs=100,150,200 \
  --phase 2 \
  --autoregressive \
  --seeds 42 123 789 \
  --device cuda
```

How it works:

- Each `--grid` defines one Cartesian-product dimension.
- Any non-wrapper arguments are forwarded to `code/launch_seed_experiments.py`.
- Each hyperparameter combination gets its own output directory under `--search-root`.
- The wrapper writes `grid_manifest.json`, `grid_search_results.csv`, and `grid_search_results.json` at the search root.

Grid syntax notes:

- Write flags with or without leading dashes: `epochs=100,150` and `--epochs=100,150` both work.
- Use `true` or `false` for boolean flags.
- Use `none` to omit an optional flag for one branch.
- Use `|` inside a single value when an option needs multiple CLI tokens, for example `seeds=42|123|789,101|202|303`.
- You can also provide a JSON `--grid-file` mapping flag names to value lists.

## Data Notes

- Primary source: FastF1 race sessions.
- Typical features include lap timing, sector timing, speed traps, tire metadata, race status, weather, circuit metadata, and encoded entities (driver/team/circuit/year).
- Local data directories:
  - `data/raw_data/` (downloaded source data)
  - `data/clean_data/` (prepared datasets)
  - `data/precomputed/` (cached autoregressive tensors)

## Caching and Performance

`AutoregressiveLapDataloader` supports precomputed cache files under `data/precomputed/`.

- Default cache naming pattern:
  - `ar_cache_<years>_cw<context>_<scaler>.pt`
- Cache is validated for compatibility (years, context window, scaler type, numeric schema).
- To bypass precompute cache for debugging:
  - set `SKIP_PRECOMPUTE=1`.

## Outputs

Each run folder in `results/<run_name>/` can include:

- `config.json` (resolved run config)
- `history.json` (training curves)
- `checkpoints/best_model.pt`
- `evaluation/evaluation_results.json`
- `evaluation/predictions.npz`
- `evaluation/evaluation_report.txt`
- analysis figures (`loss_curves.png`, `error_breakdown.png`, etc.)

## Development Status

### Completed

- Phase 1 stint-based modeling and evaluation.
- Phase 2 autoregressive setup with auxiliary heads.
- Per-run reporting and grouped error analysis.

### Next

Cf. `ROADMAP_TODO.md` for detailed next steps.


## License

This project is released under the terms of the `LICENSE` file in the repository root.