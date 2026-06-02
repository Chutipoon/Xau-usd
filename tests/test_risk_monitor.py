import pytest
import pandas as pd
import numpy as np
from src.execution.risk_monitor import check_drawdown, check_correlation

def test_drawdown_halt():
    # 5% drawdown threshold
    equity = pd.Series([100, 105, 99, 98, 97, 96, 94])
    # Peak 105, Min 94 -> (94-105)/105 = -0.104 (10.4% drawdown)
    res = check_drawdown(equity)
    assert bool(res['halt_required']) is True
    assert res['drawdown_30d'] > 0.05

    equity_safe = pd.Series([100, 101, 100, 99, 98])
    # Peak 101, Min 98 -> (98-101)/101 = -0.029 (2.9% drawdown)
    res = check_drawdown(equity_safe)
    assert bool(res['halt_required']) is False
    assert res['drawdown_30d'] < 0.05

def test_correlation_breach():
    # 0.3 correlation threshold
    np.random.seed(42)
    # Highly correlated instruments
    returns = pd.DataFrame({
        'GOLD': np.random.normal(0, 0.01, 100),
        'SILVER': np.random.normal(0, 0.01, 100)
    })
    returns['SILVER'] = returns['GOLD'] * 0.9 + returns['SILVER'] * 0.1 # Corr ~ 0.9

    positions = {'GOLD': 1.0, 'SILVER': 1.0}
    res = check_correlation(positions, returns)
    assert bool(res['breach']) is True
    assert res['max_correlation'] > 0.3

    # Uncorrelated instruments
    returns_uncorr = pd.DataFrame({
        'GOLD': np.random.normal(0, 0.01, 100),
        'CRYPTO': np.random.normal(0, 0.01, 100)
    })
    positions_uncorr = {'GOLD': 1.0, 'CRYPTO': 1.0}
    res = check_correlation(positions_uncorr, returns_uncorr)
    assert bool(res['breach']) is False
    assert res['max_correlation'] < 0.3

def test_risk_monitor_edge_cases():
    # Empty/Single data
    assert bool(check_drawdown(pd.Series([100]))['halt_required']) is False
    assert bool(check_correlation({'GOLD': 1.0}, pd.DataFrame())['breach']) is False
