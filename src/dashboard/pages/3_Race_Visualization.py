"""
Page 3 — Race Visualization

On-demand rollout inference for a specific driver + circuit.

Workflow:
1. Select experiment → seed → device
2. Click "Run Rollout" — loads the checkpoint, runs inference over all test
   sequences (cached after first run, ~30 s on CPU).
3. Choose circuit → driver → year to display the predicted vs actual chart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dashboard.utils.data_loader import scan_experiments, get_seed_dirs, get_seed_labels
from dashboard.utils.inference import (
    run_rollout_sequences,
    resolve_checkpoint_and_config,
    get_test_years_from_config,
)
from dashboard.utils.charts import plot_race_rollout

st.set_page_config(page_title="Race Visualization · F1 Predictor", layout="wide")
st.title("Race Visualization")

results_root = Path(st.session_state.get("results_root", _REPO_ROOT / "results"))
data_root = st.session_state.get("data_root", str(_REPO_ROOT / "data"))

# ---------------------------------------------------------------------------
# Experiment + seed + device selection (sidebar)
# ---------------------------------------------------------------------------
with st.spinner("Scanning experiments…"):
    experiments = scan_experiments(results_root)

if not experiments:
    st.warning(f"No experiments found in `{results_root}`.")
    st.stop()

with st.sidebar:
    st.subheader("Select model")

    exp_names = [e["name"] for e in experiments]
    selected_exp_name = st.selectbox("Experiment", exp_names)
    exp_dict = next(e for e in experiments if e["name"] == selected_exp_name)

    seed_dirs = get_seed_dirs(exp_dict)
    seed_labels = get_seed_labels(exp_dict)
    selected_seed_label = st.selectbox("Seed", seed_labels)
    seed_idx = seed_labels.index(selected_seed_label)
    seed_dir = seed_dirs[seed_idx]

    device = st.selectbox(
        "Device",
        ["cpu", "cuda"],
        index=0 if st.session_state.get("device", "cpu") == "cpu" else 1,
    )

    run_button = st.button("Run Rollout", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption(
        "The rollout runs once per model and is cached.  "
        "Switching to a different seed or experiment triggers a re-run."
    )

# ---------------------------------------------------------------------------
# Resolve checkpoint path
# ---------------------------------------------------------------------------
ckpt_path, cfg_path = resolve_checkpoint_and_config(seed_dir)

if ckpt_path is None:
    st.error(
        f"No `best_model.pt` found under `{seed_dir}`.  "
        "Make sure the experiment has finished training."
    )
    st.stop()

test_years = tuple(get_test_years_from_config(str(cfg_path) if cfg_path else ""))

# ---------------------------------------------------------------------------
# Run / load rollout (cached)
# ---------------------------------------------------------------------------
sequences: list[dict] = []

if run_button or st.session_state.get("rollout_loaded_key") == str(ckpt_path):
    # Store a flag so the rollout stays loaded when the user navigates away
    # and back without clicking the button again.
    st.session_state["rollout_loaded_key"] = str(ckpt_path)
    try:
        sequences = run_rollout_sequences(
            checkpoint_path=str(ckpt_path),
            config_path=str(cfg_path) if cfg_path else "",
            test_years=test_years,
            device=device,
            data_root=data_root,
        )
    except Exception as exc:
        st.error(f"Rollout failed: {exc}")
        st.stop()
else:
    st.info(
        "Select an experiment and seed above, then click **Run Rollout** to load predictions."
    )
    st.stop()

if not sequences:
    st.warning("No sequences returned from the rollout.  Check that test data exists.")
    st.stop()

st.success(f"Loaded {len(sequences)} sequences from `{exp_dict['name']} / {selected_seed_label}`.")

# ---------------------------------------------------------------------------
# Selectors: circuit → driver → year
# ---------------------------------------------------------------------------
circuits = sorted({s["circuit_name"] for s in sequences})
selected_circuit = st.selectbox("Circuit", circuits)

drivers_for_circuit = sorted(
    {s["driver_name"] for s in sequences if s["circuit_name"] == selected_circuit}
)
selected_driver = st.selectbox("Driver", drivers_for_circuit)

years_for_combo = sorted(
    {
        s["year"]
        for s in sequences
        if s["circuit_name"] == selected_circuit and s["driver_name"] == selected_driver
    }
)
selected_year = st.selectbox("Year", years_for_combo)

# ---------------------------------------------------------------------------
# Filter to the selected sequence
# ---------------------------------------------------------------------------
matching = [
    s
    for s in sequences
    if s["circuit_name"] == selected_circuit
    and s["driver_name"] == selected_driver
    and s["year"] == selected_year
]

if not matching:
    st.warning("No rollout data found for this combination.")
    st.stop()

seq = matching[0]

# ---------------------------------------------------------------------------
# Main chart
# ---------------------------------------------------------------------------
fig = plot_race_rollout(seq)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Summary stats below the chart
# ---------------------------------------------------------------------------
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

pred_ms = seq["predicted_ms"]
act_ms = seq["actual_ms"]

if pred_ms and act_ms:
    errors = [abs(p - a) for p, a in zip(pred_ms, act_ms)]
    mae = sum(errors) / len(errors) / 1000.0
    max_err = max(errors) / 1000.0
    # Cumulative drift at last step
    signed = [p - a for p, a in zip(pred_ms, act_ms)]
    cum_drift = sum(signed) / 1000.0

    col1.metric("Context laps", len(seq["context_actual_ms"]))
    col2.metric("Predicted laps", len(pred_ms))
    col3.metric("Rollout MAE", f"{mae:.3f} s")
    col4.metric("Cumulative drift", f"{cum_drift:+.2f} s")

    # Per-lap error table
    with st.expander("Per-lap data"):
        rows = []
        for i, (lap_n, p, a) in enumerate(
            zip(seq["lap_numbers"][len(seq["context_actual_ms"]):], pred_ms, act_ms)
        ):
            err = (p - a) / 1000.0
            rows.append(
                {
                    "Lap": lap_n,
                    "Predicted (s)": round(p / 1000, 3),
                    "Actual (s)": round(a / 1000, 3),
                    "Error (s)": round(err, 3),
                    "Cum. drift (s)": round(sum(p2 - a2 for p2, a2 in zip(pred_ms[: i + 1], act_ms[: i + 1])) / 1000, 3),
                }
            )
        lap_df = pd.DataFrame(rows)
        st.dataframe(
            lap_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Error (s)": st.column_config.NumberColumn(format="%.3f"),
                "Cum. drift (s)": st.column_config.NumberColumn(format="%+.3f"),
            },
        )
