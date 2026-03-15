# F1 Predictor Roadmap and Detailed TODO

This file tracks the next experimentation steps and development axes beyond the current README summary. It is meant to provide a more detailed and structured plan for improving the model and repository over time. All experiments are run using the `code/launch_seed_experiments.py` launcher, and results are tracked in `leaderboard.md` for easy comparison. The roadmap is organized by priority and axis of improvement, with specific hypotheses, tasks, and success criteria for each experiment.

## Priority 1: High-Impact Modeling Experiments


### P1.1 Rollout training (multi-step rollout loss)

Status:
- Rollout training is now implemented in the training pipeline and is activated automatically when running with `--autoregressive`.
- This is now the top active Priority 1 experiment axis because the implementation exists and the remaining work is empirical tuning.

Hypothesis:
- Multi-step rollout training (explicit rollout loss on predicted sequences) reduces compounding error and drift during long stints compared to single-step teacher-forced training.

Tasks:
- [x] Implement `AutoregressiveRolloutDataset` usage across runs and validate batching.
- [ ] Sweep `rollout_steps` in `{3, 5, 10}` with 3 seeds each.
- [ ] Sweep `rollout_weight` in `{0.1, 0.5, 1.0, 2.0}` to find stability sweet spot.
- [ ] Test `rollout_start_epoch` at `{0, 5, 10}` to assess warm-up vs immediate rollout.

Success criteria:
- Lower autoregressive error accumulation over long stints (reduced slope of error vs lap index) and improved late-race MAE/RMSE without destabilizing early-race predictions.


### P1.2 Scheduled sampling and teacher forcing schedule

Hypothesis:
- Better schedule can improve autoregressive stability and reduce drift.

Tasks:

- [x] Add a metric for autoregressive error accumulation on long stints/races.
- [ ] Compare linear vs exponential teacher forcing decay.
- [ ] Sweep end TF ratios and decay duration. 
- [ ] Validate `hold_then_decay` end-to-end and tune `teacher_forcing_hold_epochs`.
- [ ] Test several total epoch counts.

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


### P1.4 Loss weighting and auxiliary task balancing

Hypothesis:
- Better balancing of lap/pit/compound losses can reduce large-error tails.

Tasks:
- [ ] Grid search `compound_loss_weight` in `{0.005, 0.01, 0.02, 0.05}`.
- [ ] Grid search `pit_loss_weight` in `{0.0, 1e-4, 5e-4, 1e-3}`.
- [ ] Compare dynamic aux scaling on vs off.
- [ ] Evaluate Huber lap loss with deltas `{0.05, 0.1, 0.2}`.

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

