# F1 Predictor — Experiment Roadmap

This file tracks the next experimentation steps ordered by priority. All experiments are run using `code/launch_seed_experiments.py` or `code/grid_search_experiment.py`, with 3 seeds per configuration for statistical significance. Results are stored in `results/` directories.

**Current best model**: Seq2Seq GRU (256 hidden, 3 layers, 35 features)
- Next-lap MAE: **25.08ms** (phase2_ar, seed 42, 150 epochs)
- Rollout stint MAE: **~75,100ms** (scheduled sampling, seed 111)
- Stability ratio: ~1.94

---

## High Priority

### E1. Rollout Grid Search — ✅ COMPLETED (negative result)

**Status**: Completed 2026-04-07. Multi-step training does NOT improve rollout under any configuration tested.

**Experiment**: 24-combo grid (H ∈ {1,3,5,10} × curriculum ∈ {none,linear} × start_epoch ∈ {0,10,20}), 3 seeds each.
- Results: `results/E1_rollout_grid/`

**Summary table** (sorted by stint MAE, mean over 3 seeds):

| Config | Next-lap MAE | Stint MAE | Stability |
|---|---|---|---|
| **H=1 (any)** | **37.5ms** | **82,344ms** | **1.63** |
| H=10 linear s=10 | 103ms | 100,502ms | 1.76 |
| H=10 linear s=20 | 93ms | 103,562ms | 1.78 |
| H=5 linear s=10-20 | 101ms | 116,460ms | 1.85 |
| H=10 none (any) | 96ms | 116,518ms | 1.85 |
| H=3 linear s=20 | 97ms | 116,920ms | 1.88 |
| H=5 none (any) | 162ms | 158,995ms | 2.24 |
| H=10 linear s=0 | 165ms | 171,788ms | 2.20 |

**Key findings**:
1. **H=1 wins on every single metric** — best stint MAE, best next-lap MAE, best stability ratio
2. **If you must use multi-step**: linear curriculum + late start (s=10 or s=20) is always better than no curriculum; `start_epoch=0` is always the worst
3. **H=10 linear s=10 is the best multi-step config** (100,502ms stint MAE) — but still 22% worse than H=1
4. **`curriculum=none` with H>1 is always bad** — dilutes the next-step gradient from the start with no benefit
5. Multi-step makes stability ratio *worse* (1.85-2.24) vs H=1 (1.63)

**Conclusion**: Multi-step training is definitively not the path to better rollout. The drift is structural — the model lacks information about its position in a race/stint. The way forward is richer contextual features or architectural changes, not loss function modifications.

---

### E2. Scheduled Sampling Hyperparameter Sweep

P0.2 validated the approach with a single config (p=0.5, std=0.02, start=10). A proper sweep may find a significantly better operating point.

Hypothesis:
- Higher noise or earlier start could further reduce exposure bias.
- Lower noise might preserve next-lap accuracy while still helping rollout.

Tasks:
- [ ] Grid search `ss-max-prob` in {0.3, 0.5, 0.7, 1.0}
- [ ] Grid search `ss-noise-std` in {0.01, 0.02, 0.05}
- [ ] Grid search `ss-start-epoch` in {5, 10, 20}
- [ ] Run via `grid_search_experiment.py`, 3 seeds per combo
- [ ] Analyze interaction between noise intensity and rollout stability

Success criteria:
- Find config with stint MAE < 73,000ms
- All 3 seeds converge (no catastrophic failures)

---

### E3. Teacher Forcing Schedule Sweep

Teacher forcing decay controls how quickly the model transitions from seeing ground-truth to its own predictions during decoder training.

Hypothesis:
- Current exponential decay to 0.3 may not be optimal. A hold-then-decay schedule could let the model learn basics before exposure to autoregressive noise.

Tasks:
- [ ] Grid search `teacher-forcing-decay` in {linear, exponential, hold_then_decay, constant}
- [ ] Grid search `teacher-forcing-end` in {0.0, 0.1, 0.3, 0.5}
- [ ] Grid search `teacher-forcing-hold-epochs` in {0, 10, 20, 30} (for hold_then_decay)
- [ ] Test interaction with scheduled sampling (on vs off)
- [ ] Test with different epoch counts {100, 150, 200}

Success criteria:
- Lower rollout error accumulation on long stints
- Identify best TF schedule to combine with scheduled sampling

---

## Medium Priority

### E4. Context Window & Sequence Strategy

The model currently uses a context window of 10 past laps. This may be too short to capture race dynamics or too long, adding noise.

Hypothesis:
- Larger context windows capture more race state (tire degradation trends, weather changes).
- Smaller windows reduce noise and may improve early-race predictions.

Tasks:
- [ ] Evaluate `context_window` in {5, 10, 15, 20}
- [ ] Compare full-race mode vs bounded horizon mode
- [ ] Analyze performance by lap index to detect where performance degrades
- [ ] Check if larger context helps rollout stability specifically

Success criteria:
- Improved late-race MAE/RMSE without worsening early-race accuracy

---

### E5. Feature Ablation Study

The model uses 35 features across 4 types (numeric, categorical embeddings, boolean, one-hot). Not all may contribute positively.

Hypothesis:
- Some feature groups add noise rather than signal. Removing them could improve generalization.

Tasks:
- [ ] Ablation groups to test (remove one at a time):
  - Weather features (AirTemp, TrackTemp, Humidity, Pressure, WindSpeed, WindDirection)
  - Compound features (compound_soft/medium/hard/unknown)
  - Position & track status (Position, track_clear, yellow_flag, safety_car, red_flag, vsc, vsc_ending)
  - Driver/Team embeddings
  - Tire degradation features (stint_lap, cumulative_stint_time, TyreLife, fuel_proxy)
- [ ] Measure delta MAE/RMSE for each group removed
- [ ] Document feature importance ranking
- [ ] Identify and remove noise features from default config

Success criteria:
- Clear signal of highest-value features
- Identify any features that hurt performance when included

---

### E6. Augmentation Policy Refinement

Data augmentation adds noise to training inputs to improve robustness. Current default is 0.2 probability with uniform noise.

Tasks:
- [ ] Sweep `augment_prob` in {0.0, 0.05, 0.1, 0.2, 0.3}, 3 seeds each
- [ ] Track if augmentation helps robustness (seed variance) or only hurts clean-data MAE
- [ ] Design feature-specific augmentation (e.g., higher noise on weather, lower on lap time)
- [ ] Test interaction with scheduled sampling (both add noise — may be redundant)

Success criteria:
- Determine optimal augmentation level
- Decide if feature-specific augmentation is worth the complexity

---

## Low Priority

### E7. Pretraining Data Expansion

Currently training only on Race sessions. Practice, qualifying, and sprint data is available but unused.

Hypothesis:
- Pretraining on non-race sessions could improve feature representations and auxiliary head performance, even if the data distribution differs.

Tasks:
- [ ] Pretrain on FP1/FP2/FP3 sessions, fine-tune on Race
- [ ] Pretrain on Qualifying sessions, fine-tune on Race
- [ ] Pretrain on Sprint sessions, fine-tune on Race
- [ ] Test if pretraining helps auxiliary heads (pit prediction, compound classification)
- [ ] Evaluate whether pretraining improves or hurts race-session generalization

Success criteria:
- Consistent gain on held-out race sessions without harming calibration or rollout stability

---

### E8. Transformer Architecture

Replace or augment the GRU encoder-decoder with a transformer.

Hypothesis:
- Attention mechanisms may better capture long-range dependencies in race sequences, especially for late-stint predictions where the GRU drifts.

Tasks:
- [ ] Build minimal transformer encoder baseline for lap time regression
- [ ] Keep same input pipeline, features, and metrics for fair comparison
- [ ] Start with small model (2-4 layers, 128-256 dim) to validate training stability
- [ ] Experiment with context window size (transformers may handle longer sequences better)
- [ ] Scale up if stable and promising
- [ ] Compare attention patterns to understand what the model learns

Risks:
- Overfitting on small dataset (only ~120k lap pairs)
- Higher compute cost per epoch

Success criteria:
- Must beat GRU baseline on RMSE or q95/q99, not just MAE
- Must maintain rollout stability (stability ratio < 2.0)

---

### E9. Uncertainty Prediction

Add probabilistic output to quantify prediction confidence.

Hypothesis:
- A model that knows when it's uncertain is more useful for strategy decisions than a point estimate.

Tasks:
- [ ] Add probabilistic head: predict mean + variance (Gaussian NLL loss)
- [ ] Evaluate calibration: actual coverage at 90%, 95% confidence intervals
- [ ] Use uncertainty for "confidence-aware" strategy outputs
- [ ] Analyze if uncertainty correlates with actual error (are uncertain predictions actually worse?)
- [ ] Test if uncertainty grows appropriately during rollout (later horizons should be more uncertain)

Success criteria:
- Well-calibrated uncertainty (90% CI covers ~90% of actual values)
- Uncertainty grows with rollout horizon

---

## Analysis & Product

### E10. Hard-case Diagnostics

Understand where and why the model fails badly to guide future improvements.

Tasks:
- [ ] Build dedicated report for top 5% largest errors
- [ ] Slice errors by:
  - Circuit (some tracks may be systematically harder)
  - Driver (skill variance)
  - Compound (soft vs hard degradation curves)
  - Pit-adjacent laps (in-laps, out-laps)
  - Weather regime (dry vs wet vs changing)
  - Stint phase (early vs late stint)
- [ ] Track signed bias by slice (does the model consistently over/under-predict for specific conditions?)
- [ ] Identify if hard cases are fixable with features/data or are inherent noise

Success criteria:
- Actionable insights that directly inform which experiments to prioritize

---

### E11. Product — Web Dashboard & Live Prediction

Build user-facing tools for visualizing predictions and running real-time inference.

Tasks:
- [ ] Build a lightweight web dashboard for model results and experiment comparison
- [ ] Prototype real-time prediction pipeline for live race streaming inference
- [ ] Add scenario-based prediction tooling ("What if driver X pits on lap 10?")
- [ ] Visualize rollout predictions vs actual during a race
- [ ] Add stint strategy optimizer (optimal pit window prediction)

Success criteria:
- Functional dashboard accessible via browser
- Sub-second inference latency for real-time use
- Scenario predictions that help inform strategy decisions


### E12. New season adaptation

The 2026 season just started, how can the model be finetuned with data from the first event (first GPs, sprints, practices) to adapt to the new season dynamics?

Tasks:
- [ ] Collect and preprocess data from the first few events of the 2026 season
- [ ] Finetune the best model from 2025 on this new data
- [ ] Evaluate performance on the latest events, comparing to the 2025 baseline
- [ ] Analyze results

Success criteria:
- Improved performance on 2026 events compared to 2025 baseline
- Model adapts to new season dynamics without overfitting