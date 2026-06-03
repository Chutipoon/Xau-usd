import pytest
import numpy as np
import pandas as pd
from src.evaluation.ablation import run_ablation_study

def test_walk_forward_expanding_window(mocker):
    # Setup mock data
    n_samples = 600
    returns = np.random.normal(0, 0.01, n_samples)
    features_with = np.random.randn(n_samples, 20, 18)
    features_without = np.random.randn(n_samples, 20, 12)
    targets = (np.random.rand(n_samples) > 0.5).astype(float)

    # Mock trainer to avoid slow training
    mock_trainer_cls = mocker.patch('src.evaluation.ablation.LSTMTrainer')
    mock_trainer = mock_trainer_cls.return_value

    # Mock return values for predict
    mock_trainer.predict.return_value = np.random.rand(24, 1)

    # Mock required inputs
    hmm_features_df = pd.DataFrame(np.random.randn(n_samples, 6))
    returns_series = pd.Series(returns)

    # Mock HMM to return valid length predictions
    mock_hmm_cls = mocker.patch('src.evaluation.ablation.RegimeHMM')
    mock_hmm = mock_hmm_cls.return_value
    # It's called twice per fold (train, test), but only train is used for fit
    # We need to return appropriate shapes for predict
    mock_hmm.predict.side_effect = lambda X: np.zeros(len(X))

    results = run_ablation_study(returns, features_with, features_without, targets, returns_series, hmm_features_df)

    assert mock_trainer.train.call_count == 10

    # Fold 1: train 0-20% of 600 = 120 samples
    args, _ = mock_trainer.train.call_args_list[0]
    assert len(args[0]) == 120

    # Fold 5: train 0-96% of 600 = 576 samples
    args, _ = mock_trainer.train.call_args_list[8]
    assert len(args[0]) == 576

def test_ablation_with_garch(mocker):
    n_samples = 120
    returns = np.random.normal(0, 0.01, n_samples)
    features_with = np.random.randn(n_samples, 20, 18)
    features_without = np.random.randn(n_samples, 20, 12)
    targets = (np.random.rand(n_samples) > 0.5).astype(float)
    hmm_features_df = pd.DataFrame(np.random.randn(n_samples, 6))
    returns_series = pd.Series(returns)

    # Mock GARCH
    mock_garch_cls = mocker.patch('src.evaluation.ablation.RegimeGARCH')
    mock_garch = mock_garch_cls.return_value
    mock_garch.position_size.return_value = 1.0

    # Mock LSTM
    mocker.patch('src.evaluation.ablation.LSTMTrainer')
    mocker.patch('src.evaluation.ablation.calculate_sharpe', return_value=1.0)

    # Mock HMM
    mock_hmm_cls = mocker.patch('src.evaluation.ablation.RegimeHMM')
    mock_hmm = mock_hmm_cls.return_value
    mock_hmm.predict.side_effect = lambda X: np.zeros(len(X))

    results = run_ablation_study(returns, features_with, features_without, targets, returns_series, hmm_features_df)

    # GARCH fit_all should be called once per fold (5 times)
    assert mock_garch.fit_all.call_count == 5

    # Total test samples across folds for n_samples=120:
    # F1: 20-24% (4 samples)
    # F2: 40-44% (4)
    # F3: 60-64% (4)
    # F4: 80-84% (4)
    # F5: 96-100% (5)
    # Total = 21
    assert mock_garch.position_size.call_count == 21
