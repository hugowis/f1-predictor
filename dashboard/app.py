"""
F1 Predictor — Streamlit Dashboard

Entry point.  Launch with:
    streamlit run dashboard/app.py

from the repo root (f1-predictor/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so that dashboard.utils imports work
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="F1 Predictor",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global path defaults stored in session state so all pages can read them
# ---------------------------------------------------------------------------
if "results_root" not in st.session_state:
    st.session_state.results_root = str(_REPO_ROOT / "results")
if "data_root" not in st.session_state:
    st.session_state.data_root = str(_REPO_ROOT / "data")
if "device" not in st.session_state:
    st.session_state.device = "cpu"

# ---------------------------------------------------------------------------
# Sidebar: path & device settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏎 F1 Predictor")
    st.markdown("---")

    st.subheader("Settings")
    st.session_state.results_root = st.text_input(
        "Results directory",
        value=st.session_state.results_root,
        help="Absolute or relative path to the results/ folder",
    )
    st.session_state.data_root = st.text_input(
        "Data directory",
        value=st.session_state.data_root,
        help="Absolute or relative path to the data/ folder (vocabs, clean_data)",
    )
    st.session_state.device = st.selectbox(
        "Inference device",
        options=["cpu", "cuda"],
        index=0,
        help="Device used for on-demand rollout inference (Page 3)",
    )

    st.markdown("---")
    st.caption("Navigate using the **Pages** menu above.")

# ---------------------------------------------------------------------------
# Home page content
# ---------------------------------------------------------------------------
st.title("F1 Lap Time Predictor — Dashboard")

st.markdown(
    """
    Welcome!  Use the navigation menu on the left to explore your experiments.

    | Page | Description |
    |------|-------------|
    | **1 · Experiments** | Compare all experiments side-by-side — MAE, Stint MAE, Stability |
    | **2 · Model Performance** | Deep-dive into one model: training curves, error breakdown, per-driver / circuit analysis |
    | **3 · Race Visualization** | Run on-demand rollout for any driver + circuit and see predicted vs actual lap times |

    ---
    **Current settings**
    """
)

col1, col2, col3 = st.columns(3)
col1.metric("Results root", st.session_state.results_root)
col2.metric("Data root", st.session_state.data_root)
col3.metric("Inference device", st.session_state.device)

# Quick sanity check
results_path = Path(st.session_state.results_root)
if not results_path.exists():
    st.warning(f"Results directory not found: `{results_path}`.  Update the path in the sidebar.")
else:
    from dashboard.utils.data_loader import scan_experiments
    exps = scan_experiments(results_path)
    st.success(f"Found **{len(exps)}** experiment(s) in `{results_path}`.")
