"""Launch repeated training runs across multiple random seeds.

This script wraps ``code/train.py`` so each seed runs in its own process,
which allows either sequential or parallel execution while keeping the
existing training, evaluation, and analysis pipeline unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = Path(__file__).resolve().with_name("train.py")


@dataclass
class RunResult:
    seed: int
    output_dir: str
    log_path: str
    status: str
    exit_code: int
    duration_seconds: float
    mae_ms: float | None = None
    rmse_ms: float | None = None
    median_ae_ms: float | None = None
    error_lt_50ms_pct: float | None = None
    best_val_loss: float | None = None


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch multiple training/evaluation runs across seeds"
    )
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--config", type=Path, help="Base config JSON for train.py")
    source_group.add_argument("--phase", type=int, choices=[1, 2], help="Phase preset to pass to train.py")
    parser.add_argument("--autoregressive", action="store_true", help="Use autoregressive training mode")
    parser.add_argument("--device", default="cuda", help="Device to pass to train.py")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[111, 222, 333],
        help="Explicit seed list to run",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Optional batch size override applied to every run",
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "parallel"],
        default="sequential",
        help="Whether to execute runs one after another or concurrently",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        help="Maximum number of concurrent runs when --mode parallel",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Root directory under which one subdirectory per seed is created",
    )
    parser.add_argument(
        "--run-name-template",
        default="seed_{seed}",
        help="Template for per-seed directory names; supports {seed}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed without running them",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop scheduling additional sequential runs after the first failure",
    )

    args, passthrough = parser.parse_known_args()
    _validate_args(parser, args, passthrough)
    return args, passthrough


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace, passthrough: Sequence[str]) -> None:
    if args.config is None and args.phase is None and not args.autoregressive:
        parser.error("Provide one of --config, --phase, or --autoregressive.")

    if args.max_parallel is not None and args.max_parallel < 1:
        parser.error("--max-parallel must be >= 1")

    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    disallowed = {"--seed", "--output", "--batch-size"}
    conflicts = [token for token in passthrough if token in disallowed]
    if conflicts:
        parser.error(
            "Do not pass {} directly; use the launcher's dedicated flags instead.".format(
                ", ".join(sorted(set(conflicts)))
            )
        )


def _default_output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / f"seed_sweep_{timestamp}"


def _resolve_workers(args: argparse.Namespace) -> int:
    if args.mode == "sequential":
        return 1
    if args.max_parallel is not None:
        return args.max_parallel
    return len(args.seeds)


def _build_train_command(
    args: argparse.Namespace,
    passthrough: Sequence[str],
    seed: int,
    output_dir: Path,
) -> list[str]:
    command = [sys.executable, str(TRAIN_SCRIPT)]

    if args.config is not None:
        command.extend(["--config", str(args.config)])
    elif args.phase is not None:
        command.extend(["--phase", str(args.phase)])

    if args.autoregressive:
        command.append("--autoregressive")

    if args.device:
        command.extend(["--device", args.device])

    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])

    command.extend(["--seed", str(seed), "--output", str(output_dir)])
    command.extend(passthrough)
    return command


def _safe_run_name(template: str, seed: int) -> str:
    run_name = template.format(seed=seed)
    if not run_name or run_name in {".", ".."}:
        raise ValueError("run-name-template produced an invalid directory name")
    return run_name


def _read_best_val_loss(output_dir: Path) -> float | None:
    history_path = output_dir / "history.json"
    if not history_path.exists():
        return None

    with open(history_path, "r", encoding="utf-8") as handle:
        history = json.load(handle)

    val_loss = history.get("val_loss") or []
    return float(min(val_loss)) if val_loss else None


def _read_metrics(output_dir: Path) -> dict[str, float | None]:
    results_path = output_dir / "evaluation" / "evaluation_results.json"
    if not results_path.exists():
        return {
            "mae_ms": None,
            "rmse_ms": None,
            "median_ae_ms": None,
            "error_lt_50ms_pct": None,
        }

    with open(results_path, "r", encoding="utf-8") as handle:
        results = json.load(handle)

    metrics_ms = results.get("metrics_denormalized_ms", {})
    error_breakdown = results.get("error_breakdown", {})
    error_lt_50ms_pct = None
    if "error_0_10ms" in error_breakdown and "error_10_50ms" in error_breakdown:
        error_lt_50ms_pct = float(error_breakdown["error_0_10ms"] + error_breakdown["error_10_50ms"])

    return {
        "mae_ms": _maybe_float(metrics_ms.get("mae_ms")),
        "rmse_ms": _maybe_float(metrics_ms.get("rmse_ms")),
        "median_ae_ms": _maybe_float(metrics_ms.get("median_ae_ms")),
        "error_lt_50ms_pct": error_lt_50ms_pct,
    }


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _run_single(command: Sequence[str], seed: int, output_dir: Path) -> RunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "launcher_train.log"
    started_at = time.time()

    with open(log_path, "w", encoding="utf-8") as log_handle:
        log_handle.write("Command: {}\n\n".format(subprocess.list2cmdline(list(command))))
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
    status = "completed" if completed.returncode == 0 else "failed"
    metrics = _read_metrics(output_dir) if completed.returncode == 0 else {}
    best_val_loss = _read_best_val_loss(output_dir) if completed.returncode == 0 else None

    return RunResult(
        seed=seed,
        output_dir=str(output_dir),
        log_path=str(log_path),
        status=status,
        exit_code=completed.returncode,
        duration_seconds=duration,
        best_val_loss=best_val_loss,
        **metrics,
    )


def _write_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    passthrough: Sequence[str],
    output_root: Path,
    commands: dict[int, list[str]],
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "launcher_args": vars(args),
        "passthrough_args": list(passthrough),
        "commands": {str(seed): subprocess.list2cmdline(command) for seed, command in commands.items()},
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)


def _write_summary(output_root: Path, results: Sequence[RunResult]) -> None:
    summary_json = output_root / "leaderboard.json"
    summary_csv = output_root / "leaderboard.csv"

    sorted_results = sorted(
        results,
        key=lambda result: (
            result.mae_ms is None,
            result.mae_ms if result.mae_ms is not None else float("inf"),
            result.seed,
        ),
    )

    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump([asdict(result) for result in sorted_results], handle, indent=2)

    fieldnames = list(asdict(sorted_results[0]).keys()) if sorted_results else list(asdict(RunResult(0, "", "", "", 0, 0.0)).keys())
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted_results:
            writer.writerow(asdict(result))


def _print_plan(commands: dict[int, list[str]], output_root: Path, workers: int) -> None:
    print(f"Output root: {output_root}")
    print(f"Concurrent runs: {workers}")
    for seed, command in commands.items():
        print(f"seed={seed}: {subprocess.list2cmdline(command)}")


def _print_results(results: Sequence[RunResult], output_root: Path) -> None:
    print("\nRun summary")
    print("=" * 80)
    for result in sorted(results, key=lambda item: item.seed):
        metric_text = (
            f"mae_ms={result.mae_ms:.2f}, rmse_ms={result.rmse_ms:.2f}"
            if result.mae_ms is not None and result.rmse_ms is not None
            else "metrics unavailable"
        )
        print(
            f"seed={result.seed} status={result.status} exit_code={result.exit_code} "
            f"duration={result.duration_seconds:.1f}s {metric_text}"
        )

    print(f"\nLeaderboard files: {output_root / 'leaderboard.csv'}")
    print(f"Leaderboard JSON: {output_root / 'leaderboard.json'}")


def main() -> int:
    args, passthrough = parse_args()
    output_root = (args.output_root or _default_output_root()).resolve()
    workers = _resolve_workers(args)

    commands: dict[int, list[str]] = {}
    for seed in args.seeds:
        run_name = _safe_run_name(args.run_name_template, seed)
        run_output_dir = output_root / run_name
        commands[seed] = _build_train_command(args, passthrough, seed, run_output_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(output_root / "launch_manifest.json", args, passthrough, output_root, commands)
    _print_plan(commands, output_root, workers)

    if args.dry_run:
        return 0

    results: list[RunResult] = []

    if args.mode == "sequential":
        for seed in args.seeds:
            result = _run_single(commands[seed], seed, output_root / _safe_run_name(args.run_name_template, seed))
            results.append(result)
            print(f"Finished seed={seed} with status={result.status} exit_code={result.exit_code}")
            if args.fail_fast and result.exit_code != 0:
                break
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_seed = {
                executor.submit(
                    _run_single,
                    commands[seed],
                    seed,
                    output_root / _safe_run_name(args.run_name_template, seed),
                ): seed
                for seed in args.seeds
            }
            for future in as_completed(future_to_seed):
                result = future.result()
                results.append(result)
                print(f"Finished seed={result.seed} with status={result.status} exit_code={result.exit_code}")

    _write_summary(output_root, results)
    _print_results(results, output_root)
    return 1 if any(result.exit_code != 0 for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())