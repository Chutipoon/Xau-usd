import pytest
import numpy as np
import pandas as pd
from src.evaluation.ablation import run_ablation_study

def test_ablation_decision_logic(mocker):
    # Mock calculate_sharpe or the whole LSTM training for speed
    # We want to test the decision logic specifically

    # Need synthetic data
    n_samples = 200
    returns = np.random.normal(0, 0.01, n_samples)
    features_with = np.random.randn(n_samples, 20, 18)
    features_without = np.random.randn(n_samples, 20, 12)
    targets = (np.random.rand(n_samples) > 0.5).astype(float)

    # Mock trainer.train and trainer.predict to return fixed sharpe ratios
    # We can't easily mock trainer because it's instantiated inside run_ablation_study
    # Let's mock calculate_sharpe instead

    # Mock required inputs
    returns_series = pd.Series(returns)
    hmm_features_df = pd.DataFrame(np.random.randn(n_samples, 6))

    # Mock components used in walk-forward loop
    mocker.patch('src.evaluation.ablation.LSTMTrainer')
    mock_hmm_cls = mocker.patch('src.evaluation.ablation.RegimeHMM')
    mock_hmm = mock_hmm_cls.return_value
    mock_hmm.predict.side_effect = lambda X: np.zeros(len(X))

    # We want delta > 5% -> 'keep'
    # avg_with = 1.1, avg_without = 1.0 -> delta = 10%
    mock_sharpe = mocker.patch('src.evaluation.ablation.calculate_sharpe')
    mock_sharpe.side_effect = [1.1, 1.0] * 5 # with, without for 5 folds

    results = run_ablation_study(returns, features_with, features_without, targets, returns_series, hmm_features_df)
    assert results['decision'] == 'keep'
    assert results['sharpe_delta_pct'] == pytest.approx(10.0)

    # We want delta < 5% -> 'weight_zero'
    # avg_with = 1.02, avg_without = 1.0 -> delta = 2%
    mock_sharpe.side_effect = [1.02, 1.0] * 5
    results = run_ablation_study(returns, features_with, features_without, targets, returns_series, hmm_features_df)
    assert results['decision'] == 'weight_zero'
    assert results['sharpe_delta_pct'] == pytest.approx(2.0)
