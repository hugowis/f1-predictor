# Project Leaderboard

This file collects concise, comparable summaries of completed multi-seed experiments.
Keep one row per experiment (best-performing seed shown). Update after running new sweeps.

Last updated: 2026-03-07

| Experiment | Run folder | Batch size | Seeds | Epochs | Best seed | MAE (ms) | RMSE (ms) | Median AE (ms) | % <50ms | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| phase1_gru_multiseed | results/phase1_gru_multiseed | 128 | 42, 123, 789 | 100 | 123 | 33.20 | 62.22 | 18.13 | 83.59 | Phase 1 stint-based, default seq2seq GRU encoder |
| phase1_lstm_multiseed | results/phase1_lstm_multiseed | 128 | 42, 123, 789 | 100 | 789 | 59.71 | 94.99 | 36.84 | 63.66 | Phase 1 stint-based, LSTM encoder run |
| phase2_ar_multiseed | results/phase2_ar_multiseed | 128 | 42, 123, 789 | 150 | 42 | 25.08 | 49.43 | 13.00 | 89.25 | Phase 2 autoregressive, current best setting p_aug=0.2, compound_loss=0.01, pit_loss=0.0|
