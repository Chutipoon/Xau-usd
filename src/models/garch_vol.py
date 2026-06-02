import pandas as pd
import numpy as np
from arch import arch_model
from typing import Dict, Any

class RegimeGARCH:
    def __init__(self):
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
        forecast = res.forecast(horizon=horizon)
        # The arch library's forecast() method automatically returns values on the
        # original data scale, even when internal rescaling is applied.
        var_forecast = forecast.variance.values[-1, -1]

        # Annualize (assuming hourly data, 252 days * 24 hours)
        ann_vol = np.sqrt(var_forecast * 252 * 24)
        return float(ann_vol)

    def position_size(self, regime_id: int, target_vol: float = 0.10) -> float:
        # Returns position multiplier: target_vol / forecast_vol
        # Capped at 2.0, minimum 0.1
        forecast = self.forecast_vol(regime_id)
        if forecast <= 0:
            return 0.1

        size = target_vol / forecast
        return float(np.clip(size, 0.1, 2.0))
