import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from src.models.hmm_regime import RegimeHMM, validate_hmm

@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    n_samples = 1000
    n_features = 6

    # Generate 4 distinct regimes
    data = []
    # State 0: Trending Bull (High return, low vol)
    data.append(np.random.normal(loc=[0.5, 0.0, 0.2, 0.1, 100, 0], scale=0.1, size=(250, n_features)))
    # State 1: Trending Bear (Low return, low vol)
    data.append(np.random.normal(loc=[-0.5, 0.0, 0.2, -0.1, -100, 0], scale=0.1, size=(250, n_features)))
    # State 2: High-Vol Choppy (Zero return, high vol)
    data.append(np.random.normal(loc=[0.0, 1.0, 1.0, 0.0, 0, 2.0], scale=0.5, size=(250, n_features)))
    # State 3: Low-Vol Range (Zero return, low vol)
    data.append(np.random.normal(loc=[0.0, -1.0, 0.1, 0.0, 0, -1.0], scale=0.05, size=(250, n_features)))

    X = np.vstack(data)
    columns = ['returns', 'log_volume', 'realized_vol', 'yield_spread', 'cot_net_long', 'event_spike_zscore']
    df = pd.DataFrame(X, columns=columns)
    return df

def test_fit_convergence(synthetic_data):
    model = RegimeHMM(n_components=4, n_iter=100)
    model.fit(synthetic_data)
    assert model.is_fitted
    assert model.converged

def test_predict_proba_sums_to_one(synthetic_data):
    model = RegimeHMM(n_components=4, n_iter=100)
    model.fit(synthetic_data)
    proba = model.predict_proba(synthetic_data)

    assert proba.shape == (len(synthetic_data), 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

def test_save_load_round_trip(synthetic_data):
    model = RegimeHMM(n_components=4, n_iter=100)
    model.fit(synthetic_data)
    preds_before = model.predict(synthetic_data)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        model.save(tmp.name)
        loaded_model = RegimeHMM.load(tmp.name)
        os.unlink(tmp.name)

    preds_after = loaded_model.predict(synthetic_data)
    np.testing.assert_array_equal(preds_before, preds_after)
    assert loaded_model.is_fitted

def test_validate_hmm(synthetic_data):
    model = RegimeHMM(n_components=4, n_iter=100)
    model.fit(synthetic_data)
    metrics = validate_hmm(model, synthetic_data)

    assert 'state_distribution' in metrics
    assert 'log_likelihood' in metrics
    assert 'avg_state_duration_days' in metrics
    assert len(metrics['state_distribution']) == 4
    assert sum(metrics['state_distribution'].values()) == pytest.approx(1.0)

def test_nan_handling():
    df = pd.DataFrame({
        'returns': [np.nan, 0.1, 0.2, 0.3],
        'log_volume': [0.1, np.nan, 0.2, 0.3],
        'realized_vol': [0.1, 0.2, 0.3, 0.4],
        'yield_spread': [0.1, 0.2, 0.3, 0.4],
        'cot_net_long': [0.1, 0.2, 0.3, 0.4],
        'event_spike_zscore': [0.1, 0.2, 0.3, 0.4]
    })

    model = RegimeHMM(n_components=2, n_iter=10)
    # This should fit without error after ffill and dropna
    model.fit(df)
    assert model.is_fitted

    # Check that it correctly preprocessed
    # Row 0: returns is NaN, ffill won't help if it's the first row. dropna will remove it.
    # Row 1: log_volume is NaN, ffill will make it 0.1.
    # So we should have 3 rows after preprocessing.
    X = model.preprocess(df)
    assert len(X) == 3
