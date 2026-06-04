"""Evaluation: ablation study and backtesting."""

__all__ = ["run_ablation_study", "WalkForwardBacktester", "monte_carlo_simulation"]

def __getattr__(name):
    if name == "run_ablation_study":
        from .ablation import run_ablation_study
        return run_ablation_study
    if name == "WalkForwardBacktester":
        from .backtester import WalkForwardBacktester
        return WalkForwardBacktester
    if name == "monte_carlo_simulation":
        from .backtester import monte_carlo_simulation
        return monte_carlo_simulation
    raise AttributeError(f"module {__name__} has no attribute {name}")
