import pytest
import numpy as np
from src.execution.regime_signal_bridge import RegimeSignalBridge

def test_translate_range():
    bridge = RegimeSignalBridge()
    # Test extreme positive
    post = np.array([1.0, 0, 0, 0]) # Bull
    res = bridge.translate(post, 1.0, 2.0) # direction 1.0 * mult 1.0 * size 2.0 * 20 = 40 -> clip 20
    assert res == 20.0

    # Test extreme negative
    res = bridge.translate(post, 0.0, 2.0) # direction -1.0 * 1.0 * 2.0 * 20 = -40 -> clip -20
    assert res == -20.0

def test_high_vol_reduction():
    bridge = RegimeSignalBridge()
    post = np.array([0, 0, 1.0, 0]) # High-Vol Choppy (mult 0.3)
    res = bridge.translate(post, 1.0, 1.0) # 1.0 * 0.3 * 1.0 * 20 = 6.0
    assert res == 6.0

def test_trending_bull_positive():
    bridge = RegimeSignalBridge()
    post = np.array([1.0, 0, 0, 0]) # Trending Bull (mult 1.0)
    res = bridge.translate(post, 0.9, 1.0) # 0.8 * 1.0 * 1.0 * 20 = 16.0
    assert res > 0
    assert res == 16.0

def test_trending_bear_negative():
    bridge = RegimeSignalBridge()
    post = np.array([0, 1.0, 0, 0]) # Trending Bear (mult 0.6)
    res = bridge.translate(post, 0.1, 1.0) # -0.8 * 0.6 * 1.0 * 20 = -9.6
    assert res < 0
    assert pytest.approx(res) == -9.6

def test_low_vol_standard():
    bridge = RegimeSignalBridge()
    post = np.array([0, 0, 0, 1.0]) # Low-Vol Range (mult 0.8)
    res = bridge.translate(post, 0.5, 1.0) # 0 * 0.8 * 1.0 * 20 = 0
    assert res == 0.0

def test_get_position_weight():
    bridge = RegimeSignalBridge()
    # Equal probability
    post = np.array([0.25, 0.25, 0.25, 0.25])
    # Weighted avg: 0.25*(1.0 + 0.6 + 0.3 + 0.8) = 0.25 * 2.7 = 0.675
    res = bridge.get_position_weight(post)
    assert res == pytest.approx(0.675)

def test_dominant_regime_logic():
    bridge = RegimeSignalBridge()
    post = np.array([0.4, 0.1, 0.1, 0.4]) # argmax will take first if tied, let's make it clear
    post = np.array([0.3, 0.4, 0.2, 0.1]) # Regime 1 is dominant
    res = bridge.translate(post, 1.0, 1.0) # 1.0 * 0.6 * 1.0 * 20 = 12.0
    assert res == 12.0

def test_clipping_logic():
    bridge = RegimeSignalBridge()
    post = np.array([1.0, 0, 0, 0])
    # Size 10.0 is unrealistic but tests clipping
    res = bridge.translate(post, 1.0, 10.0) # 1.0 * 1.0 * 10.0 * 20 = 200 -> clip 20
    assert res == 20.0

def test_zero_mult_edge_case():
    bridge = RegimeSignalBridge()
    post = np.array([1.0, 0, 0, 0])
    # Size 0 or mult 0
    res = bridge.translate(post, 1.0, 0.0)
    assert res == 0.0
