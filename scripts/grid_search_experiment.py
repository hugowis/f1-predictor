"""Run grid searches over any option accepted by launch_seed_experiments.py.

This script orchestrates repeated invocations of ``scripts/launch_seed_experiments.py``.
Each hyperparameter combination gets its own output directory containing the
launcher's normal multi-seed artifacts. The grid search then writes a master
manifest plus aggregated CSV/JSON summaries at the search root.

Examples
--------
Teacher forcing schedule sweep:

    python scripts/grid_search_experiment.py \
      --search-root results/tf_schedule_grid \
      --grid teacher-forcing-decay=linear,exponential \
      --grid teacher-forcing-hold-epochs=0,10,20,30 \
      --grid teacher-forcing-end=0.5,0.3,0.0 \
      --grid epochs=100,150,200 \
      --phase 2 \
      --autoregressive \
      --seeds 111 222 333 \
      --device cuda

Grid over list-valued launcher arguments:

    python scripts/grid_search_experiment.py \
      --grid seeds=42|123|789,101|202|303 \
      --grid batch-size=64,128 \
      --phase 2

Notes
-----
- Wrapper-specific flags are consumed by this script.
- All remaining arguments are forwarded to ``launch_seed_experiments.py``.
- Use ``true``/``false`` for boolean flags. ``true`` includes the flag,
  ``false`` omits it.
- Use ``none`` or ``null`` to omit an optional flag for one branch.
- Use ``|`` inside a grid value when one option needs multiple CLI tokens,
  such as ``seeds=42|123|789``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_SCRIPT = Path(__file__).resolve().with_name("launch_seed_experiments.py")


@dataclass(frozen=True)
class GridChoice:
    value: Any
    label: str


@dataclass(frozen=True)
class GridDimension:
    flag: str
    param_name: str
    choices: tuple[GridChoice, ...]


@dataclass
class GridRunResult:
    combo_index: int
    combo_name: str
    output_dir: str
    command: str
    status: str
    exit_code: int
    duration_seconds: float
    completed_runs: int
    failed_runs: int
    mean_mae_ms: float | None = None
    std_mae_ms: float | None = None
    best_mae_ms: float | None = None
    mean_rmse_ms: float | None = None
    best_rmse_ms: float | None = None
    mean_median_ae_ms: float | None = None
    mean_error_lt_50ms_pct: float | None = None
    best_error_lt_50ms_pct: float | None = None
    best_seed: int | None = None


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Grid search wrapper for launch_seed_experiments.py")
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        help=(
            "Grid dimension in the form flag=v1,v2,... . Flag may be written with or "
            "without leading dashes; multi-token values can use | separators."
        ),
    )
    parser.add_argument(
        "--grid-file",
        type=Path,
        help="Optional JSON file containing a mapping from flag names to value lists.",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        help="Root directory for all grid-search outputs. Defaults to results/grid_search_<timestamp>",
    )
    parser.add_argument(
        "--combo-name-template",
        default="{index:03d}_{params}",
        help="Template for per-combination directory names. Supports {index}, {params}, and {hash}.",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        help="Optional hard cap on the number of generated combinations.",
    )
    parser.add_argument(
        "--dry-run-grid",
        action="store_true",
        help="Print the generated launcher commands without executing them.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted grid search. --search-root must point to an "
            "existing search directory whose grid_manifest.json will be reloaded."
        ),
    )

    args, launcher_args = parser.parse_known_args()
    _validate_args(parser, args, launcher_args)
    return args, launcher_args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace, launcher_args: Sequence[str]) -> None:
    if args.resume:
        if args.search_root is None:
            parser.error("--resume requires --search-root pointing to an existing search directory.")
        manifest_path = args.search_root / "grid_manifest.json"
        if not manifest_path.exists():
            parser.error(f"Cannot resume: {manifest_path} not found.")
    elif not args.grid and args.grid_file is None:
        parser.error("Provide at least one --grid or a --grid-file.")

    if args.max_combinations is not None and args.max_combinations < 1:
        parser.error("--max-combinations must be >= 1")

    if "--output-root" in launcher_args:
        parser.error("Do not pass --output-root directly; use --search-root for the grid wrapper.")


def _default_search_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / f"grid_search_{timestamp}"


def _normalize_flag_name(flag: str) -> str:
    normalized = flag.strip()
    if not normalized:
        raise ValueError("Grid flag name cannot be empty")
    if not normalized.startswith("--"):
        normalized = f"--{normalized.lstrip('-')}"
    return normalized


def _flag_to_param_name(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def _parse_scalar_token(token: str) -> Any:
    lowered = token.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
            return int(token)
        return float(token)
    except ValueError:
        return token


def _parse_choice_token(raw_choice: str) -> GridChoice:
    token = raw_choice.strip()
    if not token:
        raise ValueError("Grid values cannot be empty")
    if "|" in token:
        parts = [part.strip() for part in token.split("|")]
        if any(not part for part in parts):
            raise ValueError(f"Invalid multi-token grid choice: {raw_choice}")
        parsed = [_parse_scalar_token(part) for part in parts]
        return GridChoice(value=parsed, label=token)
    return GridChoice(value=_parse_scalar_token(token), label=token)


def _parse_inline_grid(spec: str) -> GridDimension:
    if "=" not in spec:
        raise ValueError(f"Grid spec must look like flag=v1,v2,... ; got: {spec}")
    raw_flag, raw_values = spec.split("=", 1)
    flag = _normalize_flag_name(raw_flag)
    value_tokens = [chunk.strip() for chunk in raw_values.split(",")]
    if not value_tokens or any(token == "" for token in value_tokens):
        raise ValueError(f"Grid spec has an empty value: {spec}")
    choices = tuple(_parse_choice_token(token) for token in value_tokens)
    return GridDimension(flag=flag, param_name=_flag_to_param_name(flag), choices=choices)


def _coerce_json_choice(value: Any) -> GridChoice:
    if isinstance(value, list):
        label = "|".join(str(item) for item in value)
        return GridChoice(value=value, label=label)
    return GridChoice(value=value, label="none" if value is None else str(value))


def _load_grid_file(path: Path) -> list[GridDimension]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Grid file must contain a JSON object mapping flags to value lists")

    dimensions = []
    for raw_flag, raw_values in payload.items():
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"Grid file entry for {raw_flag!r} must be a non-empty list")
        flag = _normalize_flag_name(raw_flag)
        choices = tuple(_coerce_json_choice(value) for value in raw_values)
        dimensions.append(GridDimension(flag=flag, param_name=_flag_to_param_name(flag), choices=choices))
    return dimensions


def _load_grid_dimensions(args: argparse.Namespace) -> list[GridDimension]:
    dimensions = []
    if args.grid_file is not None:
        dimensions.extend(_load_grid_file(args.grid_file))
    dimensions.extend(_parse_inline_grid(spec) for spec in args.grid)

    seen_flags = set()
    for dimension in dimensions:
        if dimension.flag == "--output-root":
            raise ValueError("Grid search cannot vary --output-root; each combination manages its own output directory.")
        if dimension.flag in seen_flags:
            raise ValueError(f"Duplicate grid dimension for {dimension.flag}")
        seen_flags.add(dimension.flag)
    return dimensions


def _value_to_cli_tokens(flag: str, value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if value is True:
        return [flag]
    if isinstance(value, list):
        return [flag, *[str(item) for item in value]]
    return [flag, str(value)]


def _slugify(text: str) -> str:
    slug_chars = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            slug_chars.append(char)
        else:
            slug_chars.append("_")
    slug = "".join(slug_chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "combo"


def _build_combo_name(template: str, combo_index: int, labels: dict[str, str]) -> str:
    params_label = "__".join(f"{key}-{labels[key]}" for key in sorted(labels))
    params_slug = _slugify(params_label)
    digest = hashlib.sha1(params_label.encode("utf-8")).hexdigest()[:8]
    if len(params_slug) > 140:
        params_slug = f"{params_slug[:120]}_{digest}"
    combo_name = template.format(index=combo_index, params=params_slug, hash=digest)
    combo_name = _slugify(combo_name)
    if combo_name in {".", "..", ""}:
        raise ValueError("combo-name-template produced an invalid directory name")
    return combo_name


def _generate_combinations(dimensions: Sequence[GridDimension]) -> list[dict[str, GridChoice]]:
    products = itertools.product(*(dimension.choices for dimension in dimensions))
    combinations = []
    for choices in products:
        combinations.append({dimension.param_name: choice for dimension, choice in zip(dimensions, choices)})
    return combinations


def _build_launcher_command(
    launcher_args: Sequence[str],
    dimensions: Sequence[GridDimension],
    selected_choices: dict[str, GridChoice],
    output_dir: Path,
) -> list[str]:
    command = [sys.executable, str(LAUNCHER_SCRIPT), *launcher_args]
    for dimension in dimensions:
        choice = selected_choices[dimension.param_name]
        command.extend(_value_to_cli_tokens(dimension.flag, choice.value))
    command.extend(["--output-root", str(output_dir)])
    return command


def _read_launcher_summary(output_dir: Path) -> list[dict[str, Any]]:
    leaderboard_path = output_dir / "leaderboard.json"
    if not leaderboard_path.exists():
        return []
    with open(leaderboard_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else []


def _metric_stats(rows: Sequence[dict[str, Any]], metric_name: str) -> tuple[float | None, float | None, float | None]:
    values = [float(row[metric_name]) for row in rows if row.get(metric_name) is not None]
    if not values:
        return None, None, None
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    best_value = min(values)
    return mean_value, variance ** 0.5, best_value


def _max_metric(rows: Sequence[dict[str, Any]], metric_name: str) -> float | None:
    values = [float(row[metric_name]) for row in rows if row.get(metric_name) is not None]
    return max(values) if values else None


def _best_seed(rows: Sequence[dict[str, Any]]) -> int | None:
    valid_rows = [row for row in rows if row.get("mae_ms") is not None and row.get("seed") is not None]
    if not valid_rows:
        return None
    best_row = min(valid_rows, key=lambda row: float(row["mae_ms"]))
    return int(best_row["seed"])


def _summarize_combo(
    combo_index: int,
    combo_name: str,
    output_dir: Path,
    command: Sequence[str],
    exit_code: int,
    duration_seconds: float,
    labels: dict[str, str],
) -> dict[str, Any]:
    rows = _read_launcher_summary(output_dir)
    completed_rows = [row for row in rows if row.get("exit_code") == 0]
    failed_rows = [row for row in rows if row.get("exit_code") != 0]

    mean_mae_ms, std_mae_ms, best_mae_ms = _metric_stats(completed_rows, "mae_ms")
    mean_rmse_ms, _, best_rmse_ms = _metric_stats(completed_rows, "rmse_ms")
    mean_median_ae_ms, _, _ = _metric_stats(completed_rows, "median_ae_ms")
    mean_error_lt_50ms_pct, _, _ = _metric_stats(completed_rows, "error_lt_50ms_pct")
    best_error_lt_50ms_pct = _max_metric(completed_rows, "error_lt_50ms_pct")

    result = GridRunResult(
        combo_index=combo_index,
        combo_name=combo_name,
        output_dir=str(output_dir),
        command=subprocess.list2cmdline(list(command)),
        status="completed" if exit_code == 0 else "failed",
        exit_code=exit_code,
        duration_seconds=duration_seconds,
        completed_runs=len(completed_rows),
        failed_runs=len(failed_rows),
        mean_mae_ms=mean_mae_ms,
        std_mae_ms=std_mae_ms,
        best_mae_ms=best_mae_ms,
        mean_rmse_ms=mean_rmse_ms,
        best_rmse_ms=best_rmse_ms,
        mean_median_ae_ms=mean_median_ae_ms,
        mean_error_lt_50ms_pct=mean_error_lt_50ms_pct,
        best_error_lt_50ms_pct=best_error_lt_50ms_pct,
        best_seed=_best_seed(completed_rows),
    )
    record = asdict(result)
    for param_name, label in labels.items():
        record[param_name] = label
    return record


def _write_manifest(
    search_root: Path,
    dimensions: Sequence[GridDimension],
    launcher_args: Sequence[str],
    combo_plan: Sequence[dict[str, Any]],
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "search_root": str(search_root),
        "launcher_args": list(launcher_args),
        "grid": {
            dimension.param_name: [choice.label for choice in dimension.choices]
            for dimension in dimensions
        },
        "combinations": combo_plan,
    }
    with open(search_root / "grid_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _write_summary(search_root: Path, records: Sequence[dict[str, Any]], dimensions: Sequence[GridDimension]) -> None:
    sorted_records = sorted(
        records,
        key=lambda row: (
            row.get("mean_mae_ms") is None,
            row.get("mean_mae_ms") if row.get("mean_mae_ms") is not None else float("inf"),
            row["combo_index"],
        ),
    )

    with open(search_root / "grid_search_results.json", "w", encoding="utf-8") as handle:
        json.dump(sorted_records, handle, indent=2)

    fieldnames = [
        "combo_index",
        "combo_name",
        *[dimension.param_name for dimension in dimensions],
        "status",
        "exit_code",
        "completed_runs",
        "failed_runs",
        "mean_mae_ms",
        "std_mae_ms",
        "best_mae_ms",
        "mean_rmse_ms",
        "best_rmse_ms",
        "mean_median_ae_ms",
        "mean_error_lt_50ms_pct",
        "best_error_lt_50ms_pct",
        "best_seed",
        "duration_seconds",
        "output_dir",
        "command",
    ]

    with open(search_root / "grid_search_results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted_records:
            writer.writerow({field: record.get(field) for field in fieldnames})


def _print_plan(search_root: Path, combo_plan: Sequence[dict[str, Any]], dry_run: bool) -> None:
    print(f"Search root: {search_root}")
    print(f"Total combinations: {len(combo_plan)}")
    for item in combo_plan:
        print(f"combo={item['combo_name']}: {subprocess.list2cmdline(item['command'])}")
    if dry_run:
        print("\nDry run only; no commands were executed.")


def _print_results(search_root: Path, records: Sequence[dict[str, Any]]) -> None:
    print("\nGrid search summary")
    print("=" * 80)
    for record in records:
        mae_text = (
            f"mean_mae_ms={record['mean_mae_ms']:.2f}" if record.get("mean_mae_ms") is not None else "mean_mae_ms=n/a"
        )
        print(
            f"combo={record['combo_name']} status={record['status']} exit_code={record['exit_code']} "
            f"completed_runs={record['completed_runs']} {mae_text}"
        )
    print(f"\nSummary CSV: {search_root / 'grid_search_results.csv'}")
    print(f"Summary JSON: {search_root / 'grid_search_results.json'}")


def _is_combo_complete(output_dir: Path) -> bool:
    """Check whether a combination has already finished successfully."""
    try:
        rows = _read_launcher_summary(output_dir)
        if not rows:
            return False
        return any(row.get("exit_code") == 0 for row in rows)
    except Exception:
        return False


def _rebuild_record_from_existing(item: dict[str, Any]) -> dict[str, Any]:
    """Build a summary record for an already-completed combo without re-running it."""
    return _summarize_combo(
        combo_index=item["combo_index"],
        combo_name=item["combo_name"],
        output_dir=Path(item["output_dir"]),
        command=item["command"],
        exit_code=0,
        duration_seconds=0.0,
        labels=item["params"],
    )


def _load_manifest(search_root: Path) -> dict[str, Any]:
    """Load an existing grid_manifest.json."""
    with open(search_root / "grid_manifest.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def _dimensions_from_manifest(grid_dict: dict[str, list[str]]) -> list[GridDimension]:
    """Rebuild GridDimension objects from a manifest's ``grid`` mapping."""
    dimensions = []
    for param_name, labels in grid_dict.items():
        flag = f"--{param_name.replace('_', '-')}"
        choices = tuple(GridChoice(value=label, label=label) for label in labels)
        dimensions.append(GridDimension(flag=flag, param_name=param_name, choices=choices))
    return dimensions


def _run_combo(command: Sequence[str], combo_name: str, output_dir: Path) -> tuple[int, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "grid_search_launcher.log"

    started_at = time.time()
    with open(log_path, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"Command: {subprocess.list2cmdline(list(command))}\n\n")
        log_handle.flush()
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    duration = time.time() - started_at
    print(f"Finished combo={combo_name} exit_code={completed.returncode} duration={duration:.1f}s")
    return completed.returncode, duration


def main() -> int:
    args, launcher_args = parse_args()

    # ------------------------------------------------------------------
    # Resume path: reload plan from existing manifest
    # ------------------------------------------------------------------
    if args.resume:
        search_root = args.search_root.resolve()
        manifest = _load_manifest(search_root)
        dimensions = _dimensions_from_manifest(manifest["grid"])
        combo_plan = manifest["combinations"]
        print(f"Resuming grid search from {search_root}")
    else:
        dimensions = _load_grid_dimensions(args)
        combinations = _generate_combinations(dimensions)

        if args.max_combinations is not None and len(combinations) > args.max_combinations:
            raise SystemExit(
                f"Generated {len(combinations)} combinations, which exceeds --max-combinations={args.max_combinations}."
            )

        search_root = (args.search_root or _default_search_root()).resolve()
        search_root.mkdir(parents=True, exist_ok=True)

        combo_plan = []
        for combo_index, selected_choices in enumerate(combinations, start=1):
            labels = {param_name: choice.label for param_name, choice in selected_choices.items()}
            combo_name = _build_combo_name(args.combo_name_template, combo_index, labels)
            output_dir = search_root / combo_name
            command = _build_launcher_command(launcher_args, dimensions, selected_choices, output_dir)
            combo_plan.append(
                {
                    "combo_index": combo_index,
                    "combo_name": combo_name,
                    "output_dir": str(output_dir),
                    "params": labels,
                    "command": command,
                }
            )

        _write_manifest(search_root, dimensions, launcher_args, combo_plan)

    _print_plan(search_root, combo_plan, args.dry_run_grid)

    if args.dry_run_grid:
        return 0

    records = []
    skipped = 0
    for item in combo_plan:
        output_dir = Path(item["output_dir"])

        # Skip already-completed combinations
        if _is_combo_complete(output_dir):
            skipped += 1
            record = _rebuild_record_from_existing(item)
            records.append(record)
            print(f"Skipping completed combo={item['combo_name']}")
            with open(search_root / "grid_search.log", "a", encoding="utf-8") as log_handle:
                log_handle.write(
                    f"[{datetime.now().isoformat(timespec='seconds')}] SKIP {item['combo_name']} (already complete)\n"
                )
            continue

        with open(search_root / "grid_search.log", "a", encoding="utf-8") as log_handle:
            log_handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] START {item['combo_name']}\n")
            log_handle.write(f"Command: {subprocess.list2cmdline(item['command'])}\n")

        exit_code, duration = _run_combo(
            command=item["command"],
            combo_name=item["combo_name"],
            output_dir=output_dir,
        )
        labels = item["params"]
        record = _summarize_combo(
            combo_index=item["combo_index"],
            combo_name=item["combo_name"],
            output_dir=output_dir,
            command=item["command"],
            exit_code=exit_code,
            duration_seconds=duration,
            labels=labels,
        )
        records.append(record)
        with open(search_root / "grid_search.log", "a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"[{datetime.now().isoformat(timespec='seconds')}] END {item['combo_name']} "
                f"status={record['status']} exit_code={record['exit_code']}\n"
            )

        # Incremental summary so partial results survive interruption
        _write_summary(search_root, records, dimensions)

    if skipped:
        print(f"\nSkipped {skipped} already-completed combination(s)")

    _write_summary(search_root, records, dimensions)
    _print_results(search_root, records)
    return 1 if any(record["exit_code"] != 0 for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())