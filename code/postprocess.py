"""
Post-processing utilities: run evaluation and analysis after training.

This module provides a helper to run model evaluation and the analysis
report automatically. It wraps the existing `evaluate.py` and
`analyze_results.py` scripts/modules so they can be invoked programmatically
after training completes. The individual scripts remain runnable on their
own.
"""
from pathlib import Path
import importlib
import logging

logger = logging.getLogger(__name__)


def run_postprocessing(checkpoint_path: Path, config_path: Path = None, device: str = 'cpu', output_dir: Path = None):
    """Run evaluation and analysis for a trained checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the model checkpoint (best_model.pt)
    config_path : Path, optional
        Optional config file path used for evaluation
    device : str
        Device to use for evaluation (e.g., 'cuda' or 'cpu')
    output_dir : Path, optional
        Base output directory used by training (used to locate results)
    """
    # Import evaluation module (this lives in the same directory when called
    # from code/train.py). We import lazily so this module is cheap to import.
    try:
        import evaluate as eval_mod
    except Exception:
        # Fallback: try importing by relative package name
        eval_mod = importlib.import_module("evaluate")

    # Run evaluation
    logger.info(f"Running evaluation for checkpoint: {checkpoint_path}")
    eval_output_dir = None
    try:
        metrics = eval_mod.evaluate(
            checkpoint_path=Path(checkpoint_path),
            test_years=None if config_path is None else None,
            config_path=config_path,
            device=device,
            output_dir=output_dir,
        )
        logger.info("Evaluation complete.")
    except Exception as e:
        logger.exception("Evaluation failed:")
        raise

    # Run analysis script (reads evaluation results and writes analysis)
    # The analysis module contains top-level code that executes on import,
    # so reload it to ensure it runs in this process.
    try:
        analysis_mod = importlib.import_module("analyze_results")
        importlib.reload(analysis_mod)
        logger.info("Analysis executed.")
    except Exception:
        # Some environments may not execute the analysis module on import
        # (for example if it expects to be a script). As a fallback, try
        # to call a 'run' or 'main' if present.
        try:
            analysis_mod = importlib.import_module("analyze_results")
            if hasattr(analysis_mod, 'run'):
                analysis_mod.run()
        except Exception:
            logger.exception("Analysis execution failed.")
            raise

    return True
