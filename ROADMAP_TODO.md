# F1 Predictor Roadmap and Detailed TODO

This file tracks the next experimentation steps and development axes beyond the current README summary. It is meant to provide a more detailed and structured plan for improving the model and repository over time.

## Current Baselines (Reference)

- Best known run: `results/step2_compound_0.01/`
  - MAE: 23.05 ms
  - RMSE: 40.97 ms
  - Median AE: 15.68 ms
  - Error < 50 ms: 90.65%

### Priority 0: Representative Run for Comparison
Hypothesis:
- Running the current best configs across multiple random seeds will produce a stable, statistically-significant baseline and expose variance; this makes future comparisons fair and reproducible.

Tasks:
- [ ] Implement `scripts/run_seeds.py` to launch N-seed experiments (default N=3)
- [ ] Select canonical config(s): `step2_compound_0.01` and `step6b_150ep_aug0.20` (or other top candidates)
- [ ] Create a leaderboard to aggrecate and compare runs
- [ ] Run canonical configs to add to leaderboard, and commit configs files and results to repository.

## Priority 1: High-Impact Modeling Experiments

### P1.1 Loss weighting and auxiliary task balancing

Hypothesis:
- Better balancing of lap/pit/compound losses can reduce large-error tails.

Tasks:
- [ ] Grid search `compound_loss_weight` in `{0.005, 0.01, 0.02, 0.05}`.
- [ ] Grid search `pit_loss_weight` in `{0.0, 1e-4, 5e-4, 1e-3}`.
- [ ] Compare dynamic aux scaling on vs off.
- [ ] Evaluate Huber lap loss with deltas `{0.05, 0.1, 0.2}`.

Metrics to watch:
- MAE, RMSE, q95/q99 absolute error, error < 50 ms, mean bias.

### P1.2 Scheduled sampling and teacher forcing schedule

Hypothesis:
- Better schedule can improve autoregressive stability and reduce drift.

Tasks:
- [ ] Compare linear vs exponential teacher forcing decay.
- [ ] Sweep start/end TF ratios and decay duration.
- [ ] Add a "hold then decay" schedule option.

Success criteria:
- Lower autoregressive error accumulation on long stints/races.

### P1.3 Context window and sequence strategy

Hypothesis:
- Context window too short misses race state; too long adds noise.

Tasks:
- [ ] Evaluate `context_window` in `{5, 10, 15, 20}`.
- [ ] Compare full-race mode vs bounded horizon mode.
- [ ] Add analysis by lap index to detect where performance drops.

Success criteria:
- Improved late-race MAE/RMSE without worsening early-race accuracy.

## Priority 2: Data and Feature Engineering Axis

### P2.1 Feature ablation study

- [ ] Run systematic ablation sets:
  - weather off
  - compound features off
  - position/track status off
  - driver/team embeddings off
- [ ] Document feature importance by delta in MAE/RMSE.

Success criteria:
- Clear signal of highest-value features and removable noise features.

### P2.2 Augmentation policy refinement

- [ ] Sweep `augment_prob` in `{0.0, 0.05, 0.1, 0.2, 0.3}` with 3 seeds each.
- [ ] Track if augmentation helps robustness or only hurts clean-data MAE.
- [ ] Introduce feature-specific augmentation (not uniform noise).

### P2.3 Pretraining data expansion

- [ ] Add pretraining experiments on FP sessions, qualifying sessions, and sprint races.
- [ ] Evaluate whether pretraining improves race-session generalization.
- [ ] Explicitly test whether pretraining helps auxiliary pit-head learning.

Success criteria:
- Consistent gain on held-out race sessions without harming calibration or stability.

## Priority 3: Architecture Axis


### P3.1 Transformer prototype (controlled scope)

- [ ] Build minimal transformer encoder baseline for lap regression.
- [ ] Keep same input pipeline and metrics for fair comparison.
- [ ] Start with small model size to validate training stability. And increase size if stable.

Risks:
- Overfitting and higher compute cost.

Success criteria:
- Must beat current RNN baseline on RMSE or q95/q99, not just MAE.

### P3.2 Uncertainty prediction

- [ ] Add probabilistic head (mean + variance).
- [ ] Use uncertainty for "confidence-aware" strategy outputs.

## Priority 4: Evaluation and Analysis Axis

### P4.1 Hard-case diagnostics

- [ ] Add dedicated report for top 5% largest errors.
- [ ] Slice errors by:
  - circuit
  - driver
  - compound
  - pit-adjacent laps
  - weather regime
- [ ] Track signed bias by slice.

## Priority 5: Product Axis
### P5.1 Product-oriented experiments

- [ ] Build a lightweight web dashboard
- [ ] Prototype a real-time prediction pipeline for race-like streaming inference.
- [ ] Add scenario-based prediction tooling (for example, "What if I pit on lap 10?").

