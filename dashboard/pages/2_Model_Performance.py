"""
Page 2 — Model Performance Deep-Dive

Select an experiment and a seed to explore:
  - Overview metrics (MAE, RMSE, Median AE, Stability, Stint MAE)
  - Training curves
  - Error analysis (breakdown by bucket + per entity)
  - Rollout horizon error + drift
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard.utils.data_loader import (
    scan_experiments,
    get_seed_dirs,
    get_seed_labels,
    load_training_history,
    load_eval_metrics,
    load_rollout_metrics,
    load_entity_csvs,
)
from dashboard.utils.charts import (
    plot_training_curves,
    plot_loss_components,
    plot_error_breakdown,
    plot_entity_bar,
    plot_rollout_horizon,
    plot_rollout_drift,
)

st.set_page_config(page_title="Model Performance · F1 Predictor", layout="wide")
st.title("Model Performance")

results_root = Path(st.session_state.get("results_root", _REPO_ROOT / "results"))

# ---------------------------------------------------------------------------
# Experiment + seed selection (sidebar)
# ---------------------------------------------------------------------------
with st.spinner("Scanning experiments…"):
    experiments = scan_experiments(results_root)

if not experiments:
    st.warning(f"No experiments found in `{results_root}`.")
    st.stop()

exp_names = [e["name"] for e in experiments]

with st.sidebar:
    st.subheader("Select run")
    selected_exp_name = st.selectbox("Experiment", exp_names)
    exp_dict = next(e for e in experiments if e["name"] == selected_exp_name)

    seed_dirs = get_seed_dirs(exp_dict)
    seed_labels = get_seed_labels(exp_dict)

    if len(seed_dirs) > 1:
        selected_seed_label = st.selectbox("Seed", seed_labels)
        seed_idx = seed_labels.index(selected_seed_label)
    else:
        seed_idx = 0

    seed_dir = seed_dirs[seed_idx] if seed_dirs else None

if seed_dir is None:
    st.error("No seed directory found for this experiment.")
    st.stop()

# ---------------------------------------------------------------------------
# Load data for selected seed
# ---------------------------------------------------------------------------
history = load_training_history(seed_dir)
eval_metrics = load_eval_metrics(seed_dir)
rollout_ms = load_rollout_metrics(seed_dir)
entity_csvs = load_entity_csvs(seed_dir)

ms = eval_metrics.get("metrics_denormalized_ms", {})
error_bd = eval_metrics.get("error_breakdown", {})

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_training, tab_errors, tab_rollout = st.tabs(
    ["Overview", "Training", "Error Analysis", "Rollout"]
)

# ---- Tab: Overview --------------------------------------------------------
with tab_overview:
    st.subheader(f"Test-set metrics  —  {selected_exp_name} / {seed_labels[seed_idx]}")

    if not ms:
        st.info("No evaluation_results.json found for this run.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Next-lap MAE", f"{ms.get('mae_ms', 0):.2f} ms")
        c2.metric("RMSE", f"{ms.get('rmse_ms', 0):.2f} ms")
        c3.metric("Median AE", f"{ms.get('median_ae_ms', 0):.2f} ms")

        if rollout_ms:
            stint = rollout_ms.get("stint_total_time_mae", 0) / 1000
            stab = rollout_ms.get("stability_ratio", 0)
            c4.metric("Stint MAE", f"{stint:.1f} s")
            c5.metric("Stability ratio", f"{stab:.3f}")

        # Error percentiles
        st.markdown("#### Error percentiles (ms)")
        pcts = {
            "Q25": ms.get("q25_ae_ms"),
            "Q50 (median)": ms.get("q50_ae_ms") or ms.get("median_ae_ms"),
            "Q75": ms.get("q75_ae_ms"),
            "Q95": ms.get("q95_ae_ms"),
            "Q99": ms.get("q99_ae_ms"),
        }
        pct_df = pd.DataFrame(
            [{"Percentile": k, "Error (ms)": round(v, 2)} for k, v in pcts.items() if v is not None]
        )
        st.dataframe(pct_df, hide_index=True, use_container_width=False)

# ---- Tab: Training --------------------------------------------------------
with tab_training:
    if history.empty:
        st.info("No history.json found for this run.")
    else:
        st.plotly_chart(plot_training_curves(history), use_container_width=True)

        comp_fig = plot_loss_components(history)
        if comp_fig:
            st.plotly_chart(comp_fig, use_container_width=True)

        with st.expander("Raw history data"):
            st.dataframe(history, use_container_width=True)

# ---- Tab: Error Analysis --------------------------------------------------
with tab_errors:
    if error_bd:
        st.plotly_chart(plot_error_breakdown(error_bd), use_container_width=True)
    else:
        st.info("No error breakdown data available.")

    st.markdown("---")

    entity_tabs = st.tabs(["Driver", "Circuit", "Team", "Compound"])
    entity_keys = [
        ("driver_level", "driver_code"),
        ("circuit_level", "circuit_code"),
        ("team_level", "team_code"),
        ("compound_level", "compound"),
    ]

    for et, (csv_key, entity_col) in zip(entity_tabs, entity_keys):
        with et:
            df = entity_csvs.get(csv_key, pd.DataFrame())
            if df.empty:
                st.info(f"No {csv_key}.csv found.")
                continue
            # Normalise column names (some runs use different names)
            if entity_col not in df.columns:
                first_str_col = df.select_dtypes("object").columns
                entity_col = first_str_col[0] if len(first_str_col) else df.columns[0]
            st.plotly_chart(
                plot_entity_bar(df, entity_col),
                use_container_width=True,
            )
            with st.expander("Raw data"):
                st.dataframe(df, use_container_width=True, hide_index=True)

# ---- Tab: Rollout ---------------------------------------------------------
with tab_rollout:
    if not rollout_ms:
        st.info("No rollout_evaluation.json found for this run.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Stint MAE", f"{rollout_ms.get('stint_total_time_mae', 0)/1000:.1f} s")
        c2.metric("Stability ratio", f"{rollout_ms.get('stability_ratio', 0):.3f}")
        c3.metric("Sequences", rollout_ms.get("num_sequences", "—"))

        st.plotly_chart(plot_rollout_horizon(rollout_ms), use_container_width=True)
        st.plotly_chart(plot_rollout_drift(rollout_ms), use_container_width=True)
