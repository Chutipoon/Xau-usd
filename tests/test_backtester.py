import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.evaluation.backtester import WalkForwardBacktester, monte_carlo_simulation

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    returns = pd.Series(np.random.normal(0.0001, 0.01, size=500), index=dates)
    signals = pd.Series(np.random.uniform(-20, 20, size=500), index=dates)
    return signals, returns

def test_perfect_signals():
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    returns = pd.Series(np.random.normal(0, 0.01, size=100), index=dates)
    # Perfect signal: same sign as return, max magnitude
    signals = pd.Series(np.where(returns >= 0, 20.0, -20.0), index=dates)

    bt = WalkForwardBacktester(n_windows=1, test_months=3, overlap_weeks=0)
    results = bt.run(signals, returns)

    assert results['sharpe_ratio'] > 5.0

def test_random_signals():
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    returns = pd.Series(np.random.normal(0, 0.01, size=1000), index=dates)
    signals = pd.Series(np.random.uniform(-20, 20, size=1000), index=dates)

    bt = WalkForwardBacktester(n_windows=1, test_months=30, overlap_weeks=0)
    results = bt.run(signals, returns)

    # Random signals should have Sharpe near 0
    assert abs(results['sharpe_ratio']) < 2.0

def test_stress_test_coverage():
    dates = pd.date_range(start='2000-01-01', end='2023-01-01', freq='D')
    returns = pd.Series(np.random.normal(0, 0.01, size=len(dates)), index=dates)
    signals = pd.Series(np.random.uniform(-20, 20, size=len(dates)), index=dates)

    stress_periods = {
        '2008_crisis': ('2008-09-01', '2009-03-31'),
        '2020_covid': ('2020-02-01', '2020-05-31'),
        '2022_energy': ('2022-01-01', '2022-12-31'),
    }

    bt = WalkForwardBacktester()
    results = bt.stress_test(signals, returns, stress_periods)

    assert set(results.keys()) == set(stress_periods.keys())
    for period in stress_periods:
        assert 'sharpe_ratio' in results[period]
        assert 'max_drawdown' in results[period]

def test_monte_carlo_paths():
    returns = pd.Series(np.random.normal(0, 0.01, size=100))
    results = monte_carlo_simulation(returns, n_paths=1000)

    assert results['paths'].shape[0] == 1000
    assert len(results['paths'][0]) == 100
    assert 'p5_max_dd' in results
    assert 'p95_sharpe' in results
    assert 'risk_of_ruin_pct' in results

def test_window_count():
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    returns = pd.Series(np.random.normal(0, 0.01, size=1000), index=dates)
    signals = pd.Series(np.random.uniform(-20, 20, size=1000), index=dates)

    bt = WalkForwardBacktester(n_windows=12, test_months=1, overlap_weeks=2)
    results = bt.run(signals, returns)

    assert len(results['window_results']) == 12

def test_max_drawdown_positive_fraction():
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    # Force a losing strategy
    returns = pd.Series(np.random.normal(0.01, 0.001, size=100), index=dates)
    signals = pd.Series(-20.0, index=dates)

    bt = WalkForwardBacktester(n_windows=1, test_months=3, overlap_weeks=0)
    results = bt.run(signals, returns)

    assert results['max_drawdown'] >= 0.0
    assert results['max_drawdown'] <= 1.0
