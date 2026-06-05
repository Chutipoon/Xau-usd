import numpy as np
import pandas as pd
from arch import arch_model
import sys

def verify_garch_scale():
    # Create synthetic data with small returns to trigger rescaling
    np.random.seed(42)
    # returns with 0.001 std (very small)
    original_returns = np.random.normal(0, 0.001, 1000)

    # Fit with rescale=True
    model = arch_model(original_returns, vol='GARCH', p=1, q=1, rescale=True)
    res = model.fit(disp='off')

    print(f"Internal scale factor: {res.scale}")

    # Forecast
    forecast = res.forecast(horizon=1)

    # Method 1: variance from forecast.variance (directly)
    var_forecast = forecast.variance.values[-1, -1]

    # Empirical variance of the end of the series
    empirical_var = np.var(original_returns[-100:])

    print(f"Forecasted variance: {var_forecast}")
    print(f"Empirical variance (last 100): {empirical_var}")

    # Check if manual rescaling is needed
    manual_var = var_forecast / (res.scale ** 2)
    print(f"Manual rescaled variance: {manual_var}")

    if abs(manual_var - empirical_var) < abs(var_forecast - empirical_var):
        print("ALERT: Manual rescaling (division by scale^2) seems CORRECT.")
    else:
        print("ALERT: Original forecast seems CLOSER to empirical (but wait...).")

if __name__ == "__main__":
    verify_garch_scale()
