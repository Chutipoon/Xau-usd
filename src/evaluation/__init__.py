"""Evaluation: ablation study and backtesting."""

def run_ablation_study(*args, **kwargs):
    from .ablation import run_ablation_study as _run_ablation_study
    return _run_ablation_study(*args, **kwargs)

__all__ = ["run_ablation_study"]
