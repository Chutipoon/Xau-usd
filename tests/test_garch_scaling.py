import numpy as np
import pandas as pd
from src.models.garch_vol import RegimeGARCH

def test_garch_scaling_correction():
    # 1. Create synthetic data with small returns
    np.random.seed(42)
    std = 0.001
    returns = pd.Series(np.random.normal(0, std, 1000))
    regimes = np.zeros(1000, dtype=int)

    garch = RegimeGARCH()
    garch.fit_all(returns, regimes)

    # Check if rescaling was triggered
    res = garch.models[0]
    print(f"Scale: {res.scale}")

    # Get forecast vol
    # For H1, ann_factor = 252*24 = 6048
    # expected_var ~= 1e-6
    # expected_ann_vol ~= sqrt(1e-6 * 6048) ~= sqrt(0.006048) ~= 0.077 (7.7%)

    ann_vol = garch.forecast_vol(0, frequency='H1')
    print(f"Annualized Vol: {ann_vol}")

    # If uncorrected, it would be sqrt(1.0 * 6048) ~= 77.7 (7770%)

    assert 0.05 < ann_vol < 0.15, f"Annualized vol {ann_vol} out of expected range (0.05, 0.15)"
    print("SUCCESS: Scaling correction verified.")

if __name__ == "__main__":
    test_garch_scaling_correction()
