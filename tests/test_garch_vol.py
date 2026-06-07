import pytest
import pandas as pd
import numpy as np
from src.models.garch_vol import RegimeGARCH

def test_garch_fit_logic():
    returns = pd.Series(np.random.normal(0, 0.01, 100))
    regimes = np.array([0]*50 + [1]*50)

    garch = RegimeGARCH()
    garch.fit_all(returns, regimes)

    # Check if models are created for both regimes
    assert 0 in garch.models
    assert 1 in garch.models

def test_forecast_vol_calculation():
    returns = pd.Series(np.random.normal(0, 0.01, 100))
    regimes = np.full(100, 0)

    garch = RegimeGARCH(frequency='H1')
    garch.fit_all(returns, regimes)

    vol = garch.forecast_vol(0)
    assert vol > 0
    # Annualized hourly vol for 1% daily should be around 0.15 * sqrt(24) if daily vol is 1%
    # But here we just check it's a float
    assert isinstance(vol, float)

def test_position_size_capping():
    garch = RegimeGARCH(frequency='H1')

    # Mock forecast_vol by monkeypatching
    garch.forecast_vol = lambda rid: 0.02 # 2% vol -> size = 0.1/0.02 = 5.0 -> cap 2.0
    assert garch.position_size(0) == 2.0

    garch.forecast_vol = lambda rid: 2.0 # 200% vol -> size = 0.1/2.0 = 0.05 -> cap 0.1
    assert garch.position_size(0) == 0.1

    garch.forecast_vol = lambda rid: 0.1 # 10% vol -> size = 0.1/0.1 = 1.0
    assert garch.position_size(0) == 1.0
