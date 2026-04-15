"""
Plotly figure builders for the F1 Predictor dashboard.

Every function returns a ``plotly.graph_objects.Figure`` that can be
passed directly to ``st.plotly_chart(..., use_container_width=True)``.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Shared colour palette (F1-inspired: red / silver / dark)
_RED = "#E8002D"
_SILVER = "#9FA5AC"
_BLUE = "#0067FF"
_GREEN = "#00D2BE"
_ORANGE = "#FF8000"
_BG = "#1a1a2e"
_GRID = "#2a2a3e"

_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=50, r=20, t=40, b=40),
)


# ---------------------------------------------------------------------------
# Experiments leaderboard
# ---------------------------------------------------------------------------

def plot_experiments_bar(
    df: pd.DataFrame,
    metric: str,
    title: str = "",
) -> go.Figure:
    """Horizontal bar chart comparing experiments on *metric*."""
    df = df.sort_values(metric, ascending=True).dropna(subset=[metric])
    fig = go.Figure(
        go.Bar(
            x=df[metric],
            y=df["Name"] if "Name" in df.columns else df["name"],
            orientation="h",
            marker_color=_RED,
            text=df[metric].round(1),
            textposition="outside",
        )
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=title or metric,
        xaxis_title=metric,
        yaxis_title="",
        height=max(300, 40 * len(df)),
    )
    return fig


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history: pd.DataFrame) -> go.Figure:
    """Line chart of training vs validation loss over epochs."""
    fig = go.Figure()

    if "train_loss" in history.columns:
        fig.add_trace(
            go.Scatter(
                x=history.get("epoch", history.index + 1),
                y=history["train_loss"],
                name="Train loss",
                line=dict(color=_BLUE, width=2),
            )
        )

    if "val_loss" in history.columns:
        fig.add_trace(
            go.Scatter(
                x=history.get("epoch", history.index + 1),
                y=history["val_loss"],
                name="Val loss",
                line=dict(color=_RED, width=2),
            )
        )

    if "val_loss_ema" in history.columns:
        fig.add_trace(
            go.Scatter(
                x=history.get("epoch", history.index + 1),
                y=history["val_loss_ema"],
                name="Val EMA",
                line=dict(color=_ORANGE, width=1.5, dash="dot"),
            )
        )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Training curves",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def plot_loss_components(history: pd.DataFrame) -> Optional[go.Figure]:
    """Stacked area chart of multi-task loss components (lap / pit / compound)."""
    component_cols = [c for c in history.columns if c.startswith("train_") and c != "train_loss"]
    if not component_cols:
        return None

    fig = go.Figure()
    colours = [_BLUE, _ORANGE, _GREEN, _SILVER]
    for col, colour in zip(component_cols, colours):
        label = col.replace("train_", "").replace("_loss", "")
        fig.add_trace(
            go.Scatter(
                x=history.get("epoch", history.index + 1),
                y=history[col],
                name=label,
                stackgroup="one",
                line=dict(color=colour, width=0),
                fillcolor=colour,
            )
        )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Loss components (train)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
    )
    return fig


# ---------------------------------------------------------------------------
# Error breakdown
# ---------------------------------------------------------------------------

def plot_error_breakdown(error_dict: dict) -> go.Figure:
    """Horizontal stacked bar showing % of predictions in each error bucket."""
    labels = {
        "error_0_10ms": "< 10 ms",
        "error_10_50ms": "10–50 ms",
        "error_50_100ms": "50–100 ms",
        "error_100_200ms": "100–200 ms",
        "error_200plus_ms": "> 200 ms",
    }
    colours = [_GREEN, _BLUE, _ORANGE, _RED, "#8B0000"]

    fig = go.Figure()
    for key, label, colour in zip(labels.keys(), labels.values(), colours):
        val = error_dict.get(key, 0)
        fig.add_trace(
            go.Bar(
                name=label,
                x=[val],
                y=["Predictions"],
                orientation="h",
                marker_color=colour,
                text=f"{val:.1f}%",
                textposition="inside",
            )
        )

    fig.update_layout(
        **{**_LAYOUT_DEFAULTS, "margin": dict(l=50, r=20, t=40, b=80)},
        barmode="stack",
        title="Error distribution",
        xaxis_title="% of predictions",
        height=160,
        legend=dict(orientation="h", y=-0.4),
    )
    return fig


# ---------------------------------------------------------------------------
# Entity bar charts (driver / circuit / team / compound)
# ---------------------------------------------------------------------------

def plot_entity_bar(df: pd.DataFrame, entity_col: str, metric: str = "mae") -> go.Figure:
    """Bar chart of MAE (or other metric) per entity (driver / circuit / team)."""
    if df.empty or entity_col not in df.columns:
        return go.Figure()

    # Find the best metric column name
    metric_col = None
    for candidate in (f"{metric}_ms", metric, "mae_ms", "mae"):
        if candidate in df.columns:
            metric_col = candidate
            break
    if metric_col is None:
        return go.Figure()

    df = df.sort_values(metric_col, ascending=True).dropna(subset=[metric_col])
    unit = " (ms)" if "ms" in metric_col else ""

    fig = go.Figure(
        go.Bar(
            x=df[metric_col],
            y=df[entity_col].astype(str),
            orientation="h",
            marker_color=_RED,
            text=df[metric_col].round(1),
            textposition="outside",
        )
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=f"MAE{unit} by {entity_col}",
        xaxis_title=f"MAE{unit}",
        yaxis_title="",
        height=max(300, 30 * len(df)),
        margin=dict(l=100, r=60, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Rollout horizon error
# ---------------------------------------------------------------------------

def plot_rollout_horizon(rollout_ms: dict) -> go.Figure:
    """Line chart of MAE and RMSE vs rollout horizon step."""
    h_mae = rollout_ms.get("horizon_mae", {})
    h_rmse = rollout_ms.get("horizon_rmse", {})

    steps = sorted(h_mae.keys(), key=int)
    mae_vals = [h_mae[s] for s in steps]
    rmse_vals = [h_rmse.get(s, None) for s in steps]
    steps_int = [int(s) for s in steps]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps_int, y=mae_vals, name="MAE",
            line=dict(color=_RED, width=2),
        )
    )
    if any(v is not None for v in rmse_vals):
        fig.add_trace(
            go.Scatter(
                x=steps_int, y=rmse_vals, name="RMSE",
                line=dict(color=_SILVER, width=2, dash="dash"),
            )
        )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Rollout error vs horizon",
        xaxis_title="Horizon step (laps ahead)",
        yaxis_title="Error (ms)",
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def plot_rollout_drift(rollout_ms: dict) -> go.Figure:
    """Line chart of cumulative drift (signed and absolute) vs horizon."""
    h_drift = rollout_ms.get("horizon_mean_drift", {})
    h_abs = rollout_ms.get("horizon_median_abs_drift", {})

    steps = sorted(h_drift.keys(), key=int)
    drift_vals = [h_drift[s] for s in steps]
    abs_vals = [h_abs.get(s, None) for s in steps]
    steps_int = [int(s) for s in steps]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps_int, y=drift_vals, name="Mean drift (signed)",
            line=dict(color=_ORANGE, width=2),
        )
    )
    if any(v is not None for v in abs_vals):
        fig.add_trace(
            go.Scatter(
                x=steps_int, y=abs_vals, name="Median |drift|",
                line=dict(color=_GREEN, width=2, dash="dash"),
            )
        )

    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color=_SILVER, line_width=1)

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Cumulative drift vs horizon",
        xaxis_title="Horizon step",
        yaxis_title="Cumulative error (ms)",
        legend=dict(x=0.01, y=0.99),
    )
    return fig


# ---------------------------------------------------------------------------
# Race rollout visualization (main E11 feature)
# ---------------------------------------------------------------------------

def plot_race_rollout(seq: dict) -> go.Figure:
    """
    Predicted vs actual lap times for a single driver-circuit rollout.

    The context window (ground-truth seed laps) is shown shaded in grey.
    Predicted laps use red, actual laps use silver.
    """
    ctx_ms = seq["context_actual_ms"]
    pred_ms = seq["predicted_ms"]
    act_ms = seq["actual_ms"]
    lap_nums = seq["lap_numbers"]

    n_ctx = len(ctx_ms)
    n_pred = len(pred_ms)

    # Split lap numbers
    ctx_laps = lap_nums[:n_ctx]
    pred_laps = lap_nums[n_ctx: n_ctx + n_pred]

    # Convert ms → seconds for readability
    def to_sec(vals):
        return [v / 1000.0 for v in vals]

    fig = go.Figure()

    # Shaded context region
    if ctx_laps:
        fig.add_vrect(
            x0=ctx_laps[0] - 0.5,
            x1=ctx_laps[-1] + 0.5,
            fillcolor="rgba(100,100,100,0.15)",
            line_width=0,
            annotation_text="Context",
            annotation_position="top left",
            annotation_font_color=_SILVER,
        )

    # Context laps (ground truth, greyed out)
    fig.add_trace(
        go.Scatter(
            x=ctx_laps,
            y=to_sec(ctx_ms),
            name="Context (actual)",
            mode="lines+markers",
            line=dict(color=_SILVER, width=1.5, dash="dot"),
            marker=dict(size=5, color=_SILVER),
            hovertemplate="Lap %{x}<br>Actual: %{y:.3f}s<extra></extra>",
        )
    )

    # Actual lap times in predicted region
    if pred_laps and act_ms:
        fig.add_trace(
            go.Scatter(
                x=pred_laps,
                y=to_sec(act_ms),
                name="Actual",
                mode="lines+markers",
                line=dict(color=_GREEN, width=2),
                marker=dict(size=6, color=_GREEN),
                hovertemplate="Lap %{x}<br>Actual: %{y:.3f}s<extra></extra>",
            )
        )

    # Predicted lap times
    if pred_laps and pred_ms:
        errors_s = [abs(p - a) / 1000.0 for p, a in zip(pred_ms, act_ms)]
        hover_text = [
            f"Lap {l}<br>Predicted: {p/1000:.3f}s<br>Actual: {a/1000:.3f}s<br>Error: {e:.3f}s"
            for l, p, a, e in zip(pred_laps, pred_ms, act_ms, errors_s)
        ]
        fig.add_trace(
            go.Scatter(
                x=pred_laps,
                y=to_sec(pred_ms),
                name="Predicted",
                mode="lines+markers",
                line=dict(color=_RED, width=2),
                marker=dict(size=6, color=_RED),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    driver = seq.get("driver_name", "?")
    circuit = seq.get("circuit_name", "?")
    year = seq.get("year", "?")

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=f"{driver} — {circuit} {year}  |  Rollout predictions vs actuals",
        xaxis_title="Lap",
        yaxis_title="Lap time (s)",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
    )
    return fig
