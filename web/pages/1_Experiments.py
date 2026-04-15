"""
Page 1 — Experiments Leaderboard

Scans the results/ directory and shows a sortable comparison table of all
discovered experiments.  Select multiple experiments to visualise them side
by side in a bar chart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from web.utils.data_loader import scan_experiments, load_leaderboard, get_seed_dirs
from web.utils.charts import plot_experiments_bar

st.set_page_config(page_title="Experiments · F1 Predictor", layout="wide")
st.title("Experiments Leaderboard")

# ---------------------------------------------------------------------------
# Load experiments
# ---------------------------------------------------------------------------
results_root = Path(st.session_state.get("results_root", _REPO_ROOT / "results"))

with st.spinner("Scanning experiments…"):
    experiments = scan_experiments(results_root)

if not experiments:
    st.warning(f"No experiments found in `{results_root}`.  Check the path in Settings.")
    st.stop()

# ---------------------------------------------------------------------------
# Build summary DataFrame
# ---------------------------------------------------------------------------
rows = []
for exp in experiments:
    rows.append(
        {
            "Name": exp["name"],
            "Seeds": exp["n_seeds"],
            "Next-lap MAE (ms)": round(exp["mean_mae_ms"], 2) if exp["mean_mae_ms"] else None,
            "Best MAE (ms)": round(exp["best_mae_ms"], 2) if exp["best_mae_ms"] else None,
            "Stint MAE (ms)": (
                round(exp["mean_stint_mae_ms"] / 1000, 1)
                if exp["mean_stint_mae_ms"] else None
            ),
            "Stability": round(exp["mean_stability"], 3) if exp["mean_stability"] else None,
            "Last modified": exp["timestamp"],
            "_path": exp["path"],
        }
    )

summary_df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
col_search, col_sort = st.columns([3, 1])
with col_search:
    query = st.text_input("Filter by name", placeholder="e.g. E2 or scheduled")
with col_sort:
    sort_col = st.selectbox(
        "Sort by",
        ["Stint MAE (ms)", "Next-lap MAE (ms)", "Stability", "Best MAE (ms)", "Name"],
    )

filtered = summary_df.copy()
if query:
    filtered = filtered[filtered["Name"].str.contains(query, case=False, na=False)]
filtered = filtered.sort_values(sort_col, ascending=(sort_col != "Name"), na_position="last")

# ---------------------------------------------------------------------------
# Display table
# ---------------------------------------------------------------------------
display_cols = [c for c in filtered.columns if not c.startswith("_")]
st.dataframe(
    filtered[display_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Stint MAE (ms)": st.column_config.NumberColumn(format="%.1f"),
        "Next-lap MAE (ms)": st.column_config.NumberColumn(format="%.2f"),
        "Best MAE (ms)": st.column_config.NumberColumn(format="%.2f"),
        "Stability": st.column_config.NumberColumn(format="%.3f"),
    },
)

st.caption(f"Showing {len(filtered)} of {len(summary_df)} experiments.  Stint MAE in seconds.")

# ---------------------------------------------------------------------------
# Multi-select bar chart comparison
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Visual comparison")

exp_names = filtered["Name"].tolist()
selected = st.multiselect(
    "Select experiments to compare",
    options=exp_names,
    default=exp_names[:min(10, len(exp_names))],
)

if selected:
    chart_df = filtered[filtered["Name"].isin(selected)]
    metric_choice = st.radio(
        "Metric",
        ["Stint MAE (ms)", "Next-lap MAE (ms)", "Stability", "Best MAE (ms)"],
        horizontal=True,
    )
    chart_df_valid = chart_df.dropna(subset=[metric_choice])
    if not chart_df_valid.empty:
        fig = plot_experiments_bar(chart_df_valid, metric=metric_choice)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No data for **{metric_choice}** in the selected experiments.")

# ---------------------------------------------------------------------------
# Per-seed breakdown expander
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Per-seed details")

exp_for_detail = st.selectbox(
    "Expand experiment",
    options=[e["name"] for e in experiments],
    index=0,
)

if exp_for_detail:
    exp_dict = next(e for e in experiments if e["name"] == exp_for_detail)
    lb_df = load_leaderboard(Path(exp_dict["path"]))

    if lb_df.empty:
        # Single-seed run — show what we know from the scan
        st.json({k: v for k, v in exp_dict.items() if not k.startswith("_") and k != "seed_dirs"})
    else:
        display_lb = lb_df[[c for c in lb_df.columns if c not in ("output_dir", "log_path")]].copy()
        st.dataframe(display_lb, use_container_width=True, hide_index=True)
