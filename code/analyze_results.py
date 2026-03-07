"""
Post-training analysis and visualization: combines evaluation results,
loss/history plotting, and an error analysis report. This script reads
the outputs produced by `code/evaluate.py` and `code/train.py` and
writes human-readable reports and figures into `results/phase1/`.
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")


def analyze_results(run: str = 'phase1', results_dir: Path = None):
    """Analyze results for the given run folder under `results/`.

    Parameters
    ----------
    run : str
        Name of the run folder inside `results/` (default: 'phase1')
    results_dir : Path, optional
        If provided, use this path as the run results directory (overrides `run`).
    """
    print("=" * 80)
    print("POST-TRAINING ANALYSIS & VISUALIZATION")
    print("=" * 80)

    # Results directory
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results" / run
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading evaluation results...")
    eval_dir = results_dir / "evaluation"
    eval_results = {}
    if eval_dir.exists() and (eval_dir / "evaluation_results.json").exists():
        with open(eval_dir / "evaluation_results.json") as f:
            eval_results = json.load(f)
    else:
        print(f"Warning: evaluation_results.json not found in {eval_dir}. Run code/evaluate.py first.")

    metrics_denorm = eval_results.get('metrics_denormalized_ms', {})
    error_breakdown = eval_results.get('error_breakdown', {})

    print("\n[2/3] Creating visualizations...")

    diagnostics = {}

    # 2.1 Loss curves (if history exists)
    history_path = results_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        epochs = range(1, len(history.get('train_loss', [])) + 1)
        train_losses = history.get('train_loss', [])
        train_lap_losses = history.get('train_lap_loss', [])
        train_pit_losses = history.get('train_pit_loss', [])
        train_compound_losses = history.get('train_compound_loss', [])
        val_losses = history.get('val_loss', [])
        val_loss_ema = history.get('val_loss_ema', [])

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2.5)
        axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2.5)
        if val_loss_ema and len(val_loss_ema) == len(val_losses):
            axes[0].plot(epochs, val_loss_ema, 'm--', label='Validation Loss EMA', linewidth=1.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(epochs, train_losses, 'b-', label='Train Loss', linewidth=2.5)
        axes[1].semilogy(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2.5)
        if val_loss_ema and len(val_loss_ema) == len(val_losses):
            axes[1].semilogy(epochs, val_loss_ema, 'm--', label='Validation Loss EMA', linewidth=1.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (log)')
        axes[1].set_title('Training & Validation Loss (Log Scale)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(results_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved loss_curves.png")

        # Optional training loss components for multi-task runs
        if train_lap_losses and (train_pit_losses or train_compound_losses):
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes[0].plot(epochs, train_lap_losses, label='Train Lap Loss', linewidth=2.2)
            if train_pit_losses:
                axes[0].plot(epochs, train_pit_losses, label='Train Pit Loss', linewidth=1.8)
            if train_compound_losses:
                axes[0].plot(epochs, train_compound_losses, label='Train Compound Loss', linewidth=1.8)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss Components')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(epochs, train_lap_losses, label='Train Lap Loss', linewidth=2.2)
            axes[1].plot(epochs, val_losses, label='Validation Lap Loss', linewidth=2.2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Like-for-like Lap Train vs Val')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(results_dir / 'loss_components.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved loss_components.png")

        if train_losses and val_losses:
            best_idx = int(np.argmin(val_losses))
            diagnostics['training_dynamics'] = {
                'epochs_trained': int(len(val_losses)),
                'best_epoch': int(best_idx + 1),
                'best_val_loss': float(val_losses[best_idx]),
                'last_val_loss': float(val_losses[-1]),
                'last_train_loss': float(train_losses[-1]),
                'overfit_ratio_last_to_best_val': float(val_losses[-1] / val_losses[best_idx]) if val_losses[best_idx] != 0 else float('nan'),
                'gap_last_train_minus_val': float(train_losses[-1] - val_losses[-1]),
            }
            if train_lap_losses and len(train_lap_losses) == len(val_losses):
                diagnostics['training_dynamics']['last_train_lap_loss'] = float(train_lap_losses[-1])
                diagnostics['training_dynamics']['gap_last_train_lap_minus_val'] = float(train_lap_losses[-1] - val_losses[-1])
                diagnostics['training_dynamics']['gap_best_train_lap_minus_val'] = float(train_lap_losses[best_idx] - val_losses[best_idx])
    else:
        print("No history.json found — skipping loss curves.")

    # 2.2 Error breakdown & metrics plots (use eval_results if available)
    if error_breakdown:
        # Convert keys/values into ordered lists for plotting
        labels = [k.replace('error_', '').replace('_', ' ').replace('ms','ms') for k in error_breakdown.keys()]
        values = [v for v in error_breakdown.values()]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#c0392b']
        axes[0].pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, textprops={'fontsize':10})
        axes[0].set_title('Test Set Error Distribution')

        axes[1].bar(labels, values, color=colors, edgecolor='black')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_title('Errors by Range')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(results_dir / 'error_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved error_breakdown.png")

        # Metrics summary bar
        if metrics_denorm:
            names = ['MAE (ms)', 'RMSE (ms)', 'Median AE (ms)']
            vals = [metrics_denorm.get('mae_ms', 0), metrics_denorm.get('rmse_ms', 0), metrics_denorm.get('median_ae_ms', 0)]
            fig, ax = plt.subplots(figsize=(8,4))
            bars = ax.bar(names, vals, color=['#3498db','#e74c3c','#2ecc71'], edgecolor='black')
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2., v * 1.02, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
            ax.set_title('Test Metrics (Denormalized)')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(results_dir / 'test_metrics_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved test_metrics_summary.png")
    else:
        print("No evaluation error breakdown found — skipping error plots.")

    # 2.3 Group-level analyses (driver, circuit, stint phase, compound) if per-sample predictions exist
    preds_path = eval_dir / 'predictions.npz'
    meta_path = eval_dir / 'predictions_metadata.json'
    if preds_path.exists():
        try:
            data = np.load(preds_path, allow_pickle=True)
            pred_arr = data['predictions']
            targ_arr = data['targets']

            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta_list = json.load(f)
            else:
                meta_list = []

            # Ensure alignment
            L = min(len(pred_arr), len(targ_arr), len(meta_list)) if len(meta_list) > 0 else min(len(pred_arr), len(targ_arr))
            pred_arr = np.asarray(pred_arr).reshape(-1)[:L]
            targ_arr = np.asarray(targ_arr).reshape(-1)[:L]
            if len(meta_list) > 0:
                meta_list = meta_list[:L]
            else:
                # build minimal metadata if none present
                meta_list = [{'idx': int(i)} for i in range(L)]

            df = pd.DataFrame(meta_list)

            # Map driver/circuit integer IDs back to short codes using vocabs if available
            try:
                vocabs_dir = Path(__file__).parent.parent / 'data' / 'vocabs'
                driver_map = {}
                circuit_map = {}
                driver_vocab_path = vocabs_dir / 'Driver.json'
                circuit_vocab_path = vocabs_dir / 'Circuit.json'

                if driver_vocab_path.exists():
                    with open(driver_vocab_path, 'r', encoding='utf-8') as f:
                        dv = json.load(f)
                    # inverse mapping: int -> code
                    driver_map = {int(v): k for k, v in dv.items()}

                if circuit_vocab_path.exists():
                    with open(circuit_vocab_path, 'r', encoding='utf-8') as f:
                        cv = json.load(f)
                    circuit_map = {int(v): k for k, v in cv.items()}
                # optional team vocab
                team_vocab_path = vocabs_dir / 'Team.json'
                team_map = {}
                if team_vocab_path.exists():
                    with open(team_vocab_path, 'r', encoding='utf-8') as f:
                        tv = json.load(f)
                    team_map = {int(v): k for k, v in tv.items()}

                if 'driver' in df.columns and driver_map:
                    df['driver_code'] = df['driver'].apply(lambda x: driver_map.get(int(x), str(x)))
                else:
                    df['driver_code'] = df['driver'].astype(str)

                if 'circuit' in df.columns and circuit_map:
                    df['circuit_code'] = df['circuit'].apply(lambda x: circuit_map.get(int(x), str(x)))
                else:
                    df['circuit_code'] = df['circuit'].astype(str)
                if 'team' in df.columns and team_map:
                    df['team_code'] = df['team'].apply(lambda x: team_map.get(int(x), str(x)) if pd.notna(x) else 'unknown')
                elif 'team' in df.columns:
                    df['team_code'] = df['team'].astype(str)
            except Exception:
                df['driver_code'] = df['driver'].astype(str) if 'driver' in df.columns else ''
                df['circuit_code'] = df['circuit'].astype(str) if 'circuit' in df.columns else ''
            df['prediction'] = pred_arr
            df['target'] = targ_arr
            df['error_abs'] = np.abs(df['prediction'] - df['target'])
            df['error_signed'] = df['prediction'] - df['target']

            # Driver-level
            try:
                driver_stats = df.groupby('driver_code').agg(
                    mae=('error_abs', 'mean'),
                    rmse=('error_signed', lambda x: float(np.sqrt(np.mean(np.square(x))))),
                    median_ae=('error_abs', 'median'),
                    count=('error_abs', 'size'),
                )
                driver_stats.to_csv(results_dir / 'driver_level.csv')

                top = driver_stats.sort_values('count', ascending=False).head(20)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(top.index.astype(str), top['mae'])
                ax.set_title('Driver MAE (top 20 by samples)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(results_dir / 'driver_level_mae.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception:
                print('Driver-level analysis skipped (missing driver metadata)')

            # Circuit-level
            try:
                circuit_stats = df.groupby('circuit_code').agg(
                    mae=('error_abs', 'mean'),
                    rmse=('error_signed', lambda x: float(np.sqrt(np.mean(np.square(x))))),
                    median_ae=('error_abs', 'median'),
                    count=('error_abs', 'size'),
                )
                circuit_stats.to_csv(results_dir / 'circuit_level.csv')

                topc = circuit_stats.sort_values('count', ascending=False).head(20)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(topc.index.astype(str), topc['mae'])
                ax.set_title('Circuit MAE (top 20 by samples)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(results_dir / 'circuit_level_mae.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception:
                print('Circuit-level analysis skipped (missing circuit metadata)')

            # Stint-phase analysis removed by request

            # Team-level analysis
            try:
                if 'team_code' in df.columns:
                    team_stats = df.groupby('team_code').agg(
                        mae=('error_abs', 'mean'),
                        rmse=('error_signed', lambda x: float(np.sqrt(np.mean(np.square(x))))),
                        median_ae=('error_abs', 'median'),
                        count=('error_abs', 'size'),
                    )
                    team_stats.to_csv(results_dir / 'team_level.csv')

                    topt = team_stats.sort_values('count', ascending=False).head(20)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.bar(topt.index.astype(str), topt['mae'])
                    ax.set_title('Team MAE (top 20 by samples)')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(results_dir / 'team_level_mae.png', dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    print('Skipping team-level analysis (missing team metadata)')
            except Exception:
                print('Team-level analysis failed')

            # Compound-specific analysis
            if 'compound' in df.columns:
                try:
                    comp_stats = df.groupby('compound').agg(
                        mae=('error_abs', 'mean'),
                        rmse=('error_signed', lambda x: float(np.sqrt(np.mean(np.square(x))))),
                        median_ae=('error_abs', 'median'),
                        count=('error_abs', 'size'),
                    ).sort_values('count', ascending=False)
                    comp_stats.to_csv(results_dir / 'compound_level.csv')

                    topcomp = comp_stats.head(20)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.bar(topcomp.index.astype(str), topcomp['mae'])
                    ax.set_title('Compound MAE (top 20)')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(results_dir / 'compound_level_mae.png', dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception:
                    print('Compound analysis failed')

                print(f"Saved driver/circuit/phase/compound analyses (CSV + plots) to {results_dir}")
        except Exception as e:
            print(f"Failed to run group analyses: {e}")

    print("\n[3/3] Writing analysis report...")

    # Optional comparison with strong baseline if available
    baseline_eval_path = Path(__file__).parent.parent / 'results' / 'experiment_100epochs_gru_run1' / 'evaluation' / 'evaluation_results.json'
    baseline_comparison = {}
    if baseline_eval_path.exists() and metrics_denorm:
        try:
            with open(baseline_eval_path, 'r', encoding='utf-8') as f:
                baseline_eval = json.load(f)
            baseline_denorm = baseline_eval.get('metrics_denormalized_ms', {})
            if baseline_denorm:
                baseline_comparison = {
                    'mae_ms_phase2': float(metrics_denorm.get('mae_ms', float('nan'))),
                    'mae_ms_baseline': float(baseline_denorm.get('mae_ms', float('nan'))),
                    'mae_delta_ms': float(metrics_denorm.get('mae_ms', float('nan')) - baseline_denorm.get('mae_ms', float('nan'))),
                    'rmse_ms_phase2': float(metrics_denorm.get('rmse_ms', float('nan'))),
                    'rmse_ms_baseline': float(baseline_denorm.get('rmse_ms', float('nan'))),
                    'rmse_delta_ms': float(metrics_denorm.get('rmse_ms', float('nan')) - baseline_denorm.get('rmse_ms', float('nan'))),
                    'median_ae_ms_phase2': float(metrics_denorm.get('median_ae_ms', float('nan'))),
                    'median_ae_ms_baseline': float(baseline_denorm.get('median_ae_ms', float('nan'))),
                    'median_ae_delta_ms': float(metrics_denorm.get('median_ae_ms', float('nan')) - baseline_denorm.get('median_ae_ms', float('nan'))),
                }
        except Exception:
            baseline_comparison = {}

    if diagnostics:
        diagnostics['baseline_comparison'] = baseline_comparison
        diagnostics_path = results_dir / 'phase_diagnostics.json'
        with open(diagnostics_path, 'w', encoding='utf-8') as f:
            json.dump(diagnostics, f, indent=2)
        print(f"Saved phase_diagnostics.json")

    mae = metrics_denorm.get('mae_ms', float('nan'))
    rmse = metrics_denorm.get('rmse_ms', float('nan'))
    median = metrics_denorm.get('median_ae_ms', float('nan'))

    analysis_report = f"""
{'='*80}
F1 LAP TIME PREDICTION - ERROR ANALYSIS REPORT
{'='*80}

DENORMALIZED TEST METRICS
{'-'*80}
Mean Absolute Error (MAE):     {mae:.2f} ms
Root Mean Square Error:        {rmse:.2f} ms
Median Absolute Error:         {median:.2f} ms

ERROR DISTRIBUTION BY RANGE
{'-'*80}
{''.join([f"{k}: {v:.2f}%\n" for k, v in error_breakdown.items()])}

TRAINING DYNAMICS
{'-'*80}
epochs_trained: {diagnostics.get('training_dynamics', {}).get('epochs_trained', 'n/a')}
best_epoch: {diagnostics.get('training_dynamics', {}).get('best_epoch', 'n/a')}
best_val_loss: {diagnostics.get('training_dynamics', {}).get('best_val_loss', float('nan')):.6f}
last_val_loss: {diagnostics.get('training_dynamics', {}).get('last_val_loss', float('nan')):.6f}
overfit_ratio_last_to_best_val: {diagnostics.get('training_dynamics', {}).get('overfit_ratio_last_to_best_val', float('nan')):.3f}
last_train_total_loss: {diagnostics.get('training_dynamics', {}).get('last_train_loss', float('nan')):.6f}
last_train_lap_loss: {diagnostics.get('training_dynamics', {}).get('last_train_lap_loss', float('nan')):.6f}
gap_last_train_total_minus_val: {diagnostics.get('training_dynamics', {}).get('gap_last_train_minus_val', float('nan')):.6f}
gap_last_train_lap_minus_val: {diagnostics.get('training_dynamics', {}).get('gap_last_train_lap_minus_val', float('nan')):.6f}

"""

    with open(results_dir / 'error_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(analysis_report)

    print(analysis_report)
    print(f"\nError analysis report saved: {results_dir / 'error_analysis_report.txt'}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)



def main():
    parser = argparse.ArgumentParser(description="Analyze training/evaluation results for a given run folder under `results/`.")
    parser.add_argument('-r', '--run', default='phase1', help='Run folder inside results/ to analyze (default: phase1)')
    parser.add_argument('--results-dir', type=Path, help='Optional explicit results directory to analyze')
    args = parser.parse_args()

    if args.results_dir:
        analyze_results(run=args.run, results_dir=args.results_dir)
    else:
        analyze_results(run=args.run)


if __name__ == "__main__":
    main()