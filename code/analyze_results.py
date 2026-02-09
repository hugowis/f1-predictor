"""
Post-training analysis and visualization: combines evaluation results,
loss/history plotting, and an error analysis report. This script reads
the outputs produced by `code/evaluate.py` and `code/train.py` and
writes human-readable reports and figures into `results/phase1/`.
"""

import argparse
import json
import numpy as np
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

    # 2.1 Loss curves (if history exists)
    history_path = results_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        epochs = range(1, len(history.get('train_loss', [])) + 1)
        train_losses = history.get('train_loss', [])
        val_losses = history.get('val_loss', [])

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2.5)
        axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2.5)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(epochs, train_losses, 'b-', label='Train Loss', linewidth=2.5)
        axes[1].semilogy(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2.5)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (log)')
        axes[1].set_title('Training & Validation Loss (Log Scale)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(results_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved loss_curves.png")
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
        print(f"✓ Saved error_breakdown.png")

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
            print(f"✓ Saved test_metrics_summary.png")
    else:
        print("No evaluation error breakdown found — skipping error plots.")

    print("\n[3/3] Writing analysis report...")

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

{'='*80}
"""

    with open(results_dir / 'error_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(analysis_report)

    print(analysis_report)
    print(f"\n✓ Error analysis report saved: {results_dir / 'error_analysis_report.txt'}")

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