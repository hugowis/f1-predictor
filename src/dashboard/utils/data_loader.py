"""
Utilities for reading F1 predictor result files from disk.

No model loading here — only JSON/CSV/npz parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Experiment scanning
# ---------------------------------------------------------------------------

def scan_experiments(results_root: Path) -> list[dict]:
    """
    Walk *results_root* and find every directory that is an experiment root.

    An experiment root is any directory that contains **either**:
    - a ``leaderboard.json``  (multi-seed / grid-search run), or
    - a ``history.json``      (single-seed run without a leaderboard wrapper).

    Returns a flat list of dicts, one per discovered root, sorted by path.
    """
    results_root = Path(results_root)
    experiments: list[dict] = []

    for candidate in sorted(results_root.rglob("leaderboard.json")):
        exp_dir = candidate.parent
        entry = _parse_leaderboard_experiment(exp_dir)
        if entry:
            experiments.append(entry)

    # Single-seed runs: have history.json but no leaderboard.json in same dir
    leaderboard_dirs = {Path(e["path"]) for e in experiments}
    for candidate in sorted(results_root.rglob("history.json")):
        run_dir = candidate.parent
        if run_dir not in leaderboard_dirs and not _is_seed_subdir(run_dir):
            entry = _parse_single_seed_experiment(run_dir)
            if entry:
                experiments.append(entry)

    return experiments


def _is_seed_subdir(p: Path) -> bool:
    """Return True if *p* looks like a seed sub-directory (seed_NNN)."""
    return p.name.startswith("seed_") and p.name[5:].isdigit()


def _parse_leaderboard_experiment(exp_dir: Path) -> Optional[dict]:
    lb_path = exp_dir / "leaderboard.json"
    if not lb_path.exists():
        return None
    try:
        with open(lb_path) as f:
            rows = json.load(f)
    except Exception:
        return None

    if not rows:
        return None

    # Aggregate across seeds
    mae_vals = [r["mae_ms"] for r in rows if "mae_ms" in r]
    stint_vals = [r.get("stint_mae_ms") for r in rows if r.get("stint_mae_ms") is not None]
    stab_vals = [r.get("stability_ratio") for r in rows if r.get("stability_ratio") is not None]

    # Discover seed sub-directories
    seed_dirs = sorted(
        [d for d in exp_dir.iterdir() if d.is_dir() and _is_seed_subdir(d)],
        key=lambda d: d.name,
    )

    return {
        "name": exp_dir.name,
        "path": str(exp_dir),
        "type": "multi_seed",
        "n_seeds": len(seed_dirs),
        "seed_dirs": [str(d) for d in seed_dirs],
        "best_mae_ms": min(mae_vals) if mae_vals else None,
        "mean_mae_ms": float(np.mean(mae_vals)) if mae_vals else None,
        "best_stint_mae_ms": min(stint_vals) if stint_vals else None,
        "mean_stint_mae_ms": float(np.mean(stint_vals)) if stint_vals else None,
        "mean_stability": float(np.mean(stab_vals)) if stab_vals else None,
        "timestamp": _dir_mtime(exp_dir),
    }


def _parse_single_seed_experiment(run_dir: Path) -> Optional[dict]:
    eval_dir = run_dir / "evaluation"
    eval_path = eval_dir / "evaluation_results.json"
    if not eval_path.exists():
        return None
    try:
        with open(eval_path) as f:
            data = json.load(f)
    except Exception:
        return None

    mae_ms = data.get("metrics_denormalized_ms", {}).get("mae_ms")
    rollout = _load_rollout_metrics(run_dir)
    stint_mae = rollout.get("stint_total_time_mae") if rollout else None
    stability = rollout.get("stability_ratio") if rollout else None

    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "type": "single_seed",
        "n_seeds": 1,
        "seed_dirs": [str(run_dir)],
        "best_mae_ms": mae_ms,
        "mean_mae_ms": mae_ms,
        "best_stint_mae_ms": stint_mae,
        "mean_stint_mae_ms": stint_mae,
        "mean_stability": stability,
        "timestamp": _dir_mtime(run_dir),
    }


def _dir_mtime(p: Path) -> str:
    try:
        return pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Per-seed loaders
# ---------------------------------------------------------------------------

def load_leaderboard(exp_dir: Path) -> pd.DataFrame:
    """Read leaderboard.json → DataFrame (one row per seed)."""
    lb_path = Path(exp_dir) / "leaderboard.json"
    if not lb_path.exists():
        return pd.DataFrame()
    with open(lb_path) as f:
        rows = json.load(f)
    return pd.DataFrame(rows)


def load_training_history(seed_dir: Path) -> pd.DataFrame:
    """Read history.json → DataFrame with epoch + loss columns."""
    h_path = Path(seed_dir) / "history.json"
    if not h_path.exists():
        return pd.DataFrame()
    with open(h_path) as f:
        data = json.load(f)
    # history.json is either a list of epoch dicts or a dict of lists
    if isinstance(data, list):
        return pd.DataFrame(data)
    # dict of lists: {"train_loss": [...], "val_loss": [...], ...}
    df = pd.DataFrame(data)
    if "epoch" not in df.columns:
        df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def load_eval_metrics(seed_dir: Path) -> dict:
    """Read evaluation_results.json."""
    p = Path(seed_dir) / "evaluation" / "evaluation_results.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _load_rollout_metrics(seed_dir: Path) -> dict:
    p = Path(seed_dir) / "evaluation" / "rollout_evaluation.json"
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    # Prefer ms metrics, fall back to normalized
    return data.get("rollout_metrics_ms") or data.get("rollout_metrics_normalized") or {}


def load_rollout_metrics(seed_dir: Path) -> dict:
    """Read rollout_evaluation.json → metrics dict (ms if available)."""
    return _load_rollout_metrics(seed_dir)


def load_entity_csvs(seed_dir: Path) -> dict[str, pd.DataFrame]:
    """Read driver/circuit/team/compound CSVs → dict of DataFrames."""
    seed_dir = Path(seed_dir)
    result = {}
    for name in ("driver_level", "circuit_level", "team_level", "compound_level"):
        p = seed_dir / f"{name}.csv"
        if p.exists():
            try:
                result[name] = pd.read_csv(p)
            except Exception:
                pass
    return result


def load_predictions(seed_dir: Path) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Load predictions.npz + predictions_metadata.json.

    Returns (predictions_ms, targets_ms, metadata_list).
    """
    eval_dir = Path(seed_dir) / "evaluation"
    npz_path = eval_dir / "predictions.npz"
    meta_path = eval_dir / "predictions_metadata.json"

    preds = targets = np.array([])
    metadata = []

    if npz_path.exists():
        data = np.load(npz_path)
        preds = data["predictions"]
        targets = data["targets"]

    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return preds, targets, metadata


# ---------------------------------------------------------------------------
# Vocabulary / ID decoding
# ---------------------------------------------------------------------------

def load_vocabs(data_root: Path) -> dict[str, dict[int, str]]:
    """
    Load Driver/Circuit/Team/Year vocabularies and invert them to id→name.

    Returns::
        {
            "Driver":  {0: "<UNK>", 6: "ALB", 28: "ALO", ...},
            "Circuit": {0: "<UNK>", 19: "Austin", ...},
            "Team":    {...},
            "Year":    {0: "<UNK>", 1: "2018", ...},
        }
    """
    vocab_dir = Path(data_root) / "vocabs"
    vocabs: dict[str, dict[int, str]] = {}
    for entity in ("Driver", "Circuit", "Team", "Year"):
        p = vocab_dir / f"{entity}.json"
        if p.exists():
            with open(p) as f:
                name_to_id: dict[str, int] = json.load(f)
            vocabs[entity] = {v: k for k, v in name_to_id.items()}
    return vocabs


# ---------------------------------------------------------------------------
# Convenience: resolve seed directories from an experiment entry
# ---------------------------------------------------------------------------

def get_seed_dirs(experiment: dict) -> list[Path]:
    """Return sorted list of seed directories for an experiment dict."""
    return [Path(d) for d in experiment.get("seed_dirs", [])]


def get_seed_labels(experiment: dict) -> list[str]:
    """Return human-readable labels for each seed directory."""
    dirs = get_seed_dirs(experiment)
    if experiment.get("type") == "single_seed":
        return [experiment["name"]]
    return [d.name for d in dirs]
