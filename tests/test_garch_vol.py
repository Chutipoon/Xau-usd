import pytest
import numpy as np
import pandas as pd
from src.models.garch_vol import RegimeGARCH

def test_position_size_capping():
    garch = RegimeGARCH()
    # Mock forecast_vol by monkeypatching
    garch.forecast_vol = lambda rid: 0.02 # 2% vol -> size = 0.1/0.02 = 5.0 -> cap 2.0
    assert garch.position_size(0) == 2.0

    garch.forecast_vol = lambda rid: 2.0 # 200% vol -> size = 0.1/2.0 = 0.05 -> cap 0.1
    assert garch.position_size(0) == 0.1

    garch.forecast_vol = lambda rid: 0.1 # 10% vol -> size = 0.1/0.1 = 1.0
    assert garch.position_size(0) == 1.0

def test_garch_fit_all():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    # 4 regimes
    regimes = np.random.randint(0, 4, 1000)

    garch = RegimeGARCH()
    garch.fit_all(returns, regimes)

    # Check if models were fitted for all regimes (assuming random distribution gave > 20 samples per regime)
    assert len(garch.models) == 4
    for rid in range(4):
        assert rid in garch.models
        vol = garch.forecast_vol(rid)
        assert vol > 0
