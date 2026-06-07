import pandas as pd
import numpy as np
from arch import arch_model
from typing import Dict, Any

class RegimeGARCH:
    def __init__(self, frequency: str = 'H1'):
        self.frequency = frequency
        self.models = {}  # {regime_id: arch.arch_model fitted}

    def fit_all(self, returns: pd.Series, regimes: np.ndarray):
        # Fit separate GARCH(1,1) for each regime
        unique_regimes = np.unique(regimes)
        for rid in unique_regimes:
            mask = (regimes == rid)
            if mask.sum() < 20: # Minimum samples to fit GARCH
                print(f"Warning: Regime {rid} has too few samples ({mask.sum()}). Skipping.")
                continue

            # Use arch library: arch_model(returns[mask], vol='GARCH', p=1, q=1)
            # rescale=True helps with convergence for small returns
            model = arch_model(returns[mask], vol='GARCH', p=1, q=1, rescale=True)
            try:
                res = model.fit(disp='off')
                self.models[rid] = res
            except Exception as e:
                print(f"Error fitting GARCH for regime {rid}: {e}")

    def forecast_vol(self, regime_id: int, horizon: int = 1) -> float:
        # Returns annualized vol forecast for given regime
        if regime_id not in self.models:
            # Fallback if no model for regime (e.g. mean vol of other regimes or global)
            return 0.15 # Default 15% vol

        res = self.models[regime_id]
        forecast = res.forecast(horizon=horizon, reindex=False)
        # Bug Fix: arch library's forecast().variance returns values on the RESCALED scale
        # if rescale=True was used and reindex=False or certain versions of arch.
        # We must manually scale back by dividing by scale^2.
        var_forecast = forecast.variance.values[-1, -1]
        if hasattr(res, 'scale') and res.scale != 1.0:
            var_forecast = var_forecast / (res.scale ** 2)

        # Annualize based on data frequency
        # H1: 252 * 23 (XAU/USD trades 23h/day)
        # D1: 252
        if self.frequency == 'H1':
            ann_factor = 252 * 23
        else:
            ann_factor = 252

        ann_vol = np.sqrt(var_forecast * ann_factor)
        return float(ann_vol)

    def position_size(self, regime_id: int, target_vol: float = 0.10) -> float:
        # Returns position multiplier: target_vol / forecast_vol
        # Capped at 2.0, minimum 0.1
        forecast = self.forecast_vol(regime_id)
        if forecast <= 0:
            return 0.1

        size = target_vol / forecast
        return float(np.clip(size, 0.1, 2.0))
