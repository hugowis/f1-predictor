# Project Leaderboard

This file collects concise, comparable summaries of completed multi-seed experiments.
Keep one row per experiment (best-performing seed shown). Update after running new sweeps.

Last updated: 2026-03-08

| Experiment | Run folder | Batch size | Seeds | Epochs | Best seed | MAE (ms) | RMSE (ms) | Median AE (ms) | % <50ms | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| phase1_gru_multiseed | results/phase1_gru_multiseed | 128 | 42, 123, 789 | 100 | 123 | 33.20 | 62.22 | 18.13 | 83.59 | Phase 1 stint-based, default seq2seq GRU encoder |
| phase1_lstm_multiseed | results/phase1_lstm_multiseed | 128 | 42, 123, 789 | 100 | 789 | 59.71 | 94.99 | 36.84 | 63.66 | Phase 1 stint-based, LSTM encoder run |
| phase2_ar_multiseed | results/phase2_ar_multiseed | 128 | 42, 123, 789 | 150 | 42 | 25.08 | 49.43 | 13.00 | 89.25 | Phase 2 autoregressive baseline, p_aug=0.2, compound_loss=0.01, pit_loss=0.0 |
| comp_w0.005 | results/comp_w0.005 | 128 | 42, 456, 789 | 150 | 456 | 40.46 | 80.46 | 18.11 | 80.64 | Phase 2 AR sweep: compound_loss_weight=0.005 |
| comp_w0.01 | results/comp_w0.01 | 128 | 42, 456, 789 | 150 | 42 | 25.08 | 49.43 | 13.00 | 89.25 | Phase 2 AR sweep: compound_loss_weight=0.01 (best in compound sweep) |
| comp_w0.02 | results/comp_w0.02 | 128 | 42, 456, 789 | 150 | 42 | 40.42 | 74.62 | 24.07 | 78.84 | Phase 2 AR sweep: compound_loss_weight=0.02 (high seed variance) |
| comp_w0.05 | results/comp_w0.05 | 128 | 42, 456, 789 | 150 | 456 | 62.75 | 120.57 | 36.97 | 62.84 | Phase 2 AR sweep: compound_loss_weight=0.05 |
| pit_w0.0 | results/pit_w0.0 | 128 | 42, 456, 789 | 150 | 42 | 25.08 | 49.43 | 13.00 | 89.25 | Phase 2 AR sweep: pit_loss_weight=0.0 |
| pit_w1e-4 | results/pit_w1e-4 | 128 | 42, 456, 789 | 150 | 789 | 28.94 | 51.20 | 16.86 | 86.87 | Phase 2 AR sweep: pit_loss_weight=1e-4 |
| pit_w5e-4 | results/pit_w5e-4 | 128 | 42, 456, 789 | 150 | 789 | 26.33 | 48.82 | 15.82 | 89.17 | Phase 2 AR sweep: pit_loss_weight=5e-4 (high seed variance) |
| pit_w1e-3 | results/pit_w1e-3 | 128 | 42, 456, 789 | 150 | 789 | 21.66 | 41.24 | 12.11 | 91.14 | Phase 2 AR sweep: pit_loss_weight=1e-3 (best overall) |
| dyn_aux_on | results/dyn_aux_on | 128 | 42, 456, 789 | 150 | 789 | 21.66 | 41.24 | 12.11 | 91.14 | Phase 2 AR: dynamic auxiliary scaling ON (best seed 789) |
| dyn_aux_off | results/dyn_aux_off | 128 | 42, 456, 789 | 150 | 789 | 26.70 | 55.75 | 12.27 | 89.25 | Phase 2 AR: dynamic auxiliary scaling OFF (high seed variance; two unstable seeds) |
| huber_d0.05 | results/huber_d0.05 | 128 | 42, 456, 789 | 150 | 456 | 37.31 | 69.64 | 18.91 | 86.63 | Phase 2 AR: Huber lap loss (delta=0.05) — underperformed vs MSE; keeping MSE |
