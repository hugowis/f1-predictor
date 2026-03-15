# F1 Predictor Roadmap and Detailed TODO

This file tracks the next experimentation steps and development axes beyond the current README summary. It is meant to provide a more detailed and structured plan for improving the model and repository over time. All experiments are run using the `code/launch_seed_experiments.py` launcher, and results are tracked in `leaderboard.md` for easy comparison. The roadmap is organized by priority and axis of improvement, with specific hypotheses, tasks, and success criteria for each experiment.

## Priority 1: High-Impact Modeling Experiments

### P1.1 Loss weighting and auxiliary task balancing

Hypothesis:
- Better balancing of lap/pit/compound losses can reduce large-error tails.

Tasks:
- [x] Grid search `compound_loss_weight` in `{0.005, 0.01, 0.02, 0.05}`.
- [x] Grid search `pit_loss_weight` in `{0.0, 1e-4, 5e-4, 1e-3}`.
- [x] Compare dynamic aux scaling on vs off.
- [x] Evaluate Huber lap loss with deltas `{0.05, 0.1, 0.2}`.

Compound-loss Result:
- Best setting remains `compound_loss_weight=0.01` (best-seed metrics: MAE `25.08` ms, RMSE `49.43` ms, Median AE `13.00` ms, Error < 50 ms `89.25%`).
- Higher weights (`0.02`, `0.05`) degraded performance; `0.005` underperformed and showed higher seed variance.

Pit-loss sweep Result:
- Best setting from the pit-weight grid search: `pit_loss_weight=1e-3` (best-seed metrics: MAE `21.66` ms, RMSE `41.24` ms, Median AE `12.11` ms, Error < 50 ms `91.14%`).
- `5e-4` showed high seed variance (one good seed, two poor); `1e-4` underperformed relative to baseline `0.0`.

Dynamic aux scaling Result:
- Dynamic aux scaling ON performed best (best-seed metrics: MAE 21.66 ms, RMSE 41.24 ms, Median AE 12.11 ms, %<50ms 91.14%).
- Dynamic aux scaling OFF showed high seed variance and unstable runs (best seed MAE 26.70 ms but two seeds had very large errors / poor validation), indicating the automatic balancing helps stability for this setup.

Huber-loss Result:
- Tested Huber lap loss with delta=0.05 (three seeds). Best seed (seed 456) produced MAE 37.31 ms, RMSE 69.64 ms, Median AE 18.91 ms, %<50ms 86.63 — substantially worse than the MSE baseline (MAE ~25.08 ms). Based on these runs, we will keep MSE as the primary lap loss.

### P1.2 Scheduled sampling and teacher forcing schedule

Hypothesis:
- Better schedule can improve autoregressive stability and reduce drift.

Tasks:

- [x] Add a metric for autoregressive error accumulation on long stints/races.
- [ ] Compare linear vs exponential teacher forcing decay.
- [ ] Sweep end TF ratios and decay duration. 
- [ ] Add a "hold then decay" schedule option.  epochs
- [ ] Test several epochs numbers.

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

