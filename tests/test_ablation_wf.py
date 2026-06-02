import pytest
import numpy as np
import pandas as pd
from src.evaluation.ablation import run_ablation_study

def test_walk_forward_expanding_window(mocker):
    # Setup mock data
    n_samples = 600 # Should give 6 chunks of 100
    returns = np.random.normal(0, 0.01, n_samples)
    features_with = np.random.randn(n_samples, 20, 18)
    features_without = np.random.randn(n_samples, 20, 12)
    targets = (np.random.rand(n_samples) > 0.5).astype(float)

    # Mock trainer to avoid slow training
    # We want to verify that train_end and test_end are correct
    mock_trainer_cls = mocker.patch('src.evaluation.ablation.LSTMTrainer')
    mock_trainer = mock_trainer_cls.return_value
    mock_trainer.predict.return_value = np.random.rand(100, 1) # matches chunk size

    # Capture calls to trainer.train
    # It should be called 10 times (2 models * 5 folds)

    results = run_ablation_study(returns, features_with, features_without, targets)

    assert mock_trainer.train.call_count == 10

    # Check first fold training data size (chunk 1)
    args, kwargs = mock_trainer.train.call_args_list[0]
    # first call is Fold 1, Model With
    assert len(args[0]) == 100 # train_end = 1 * 100

    # Check last fold training data size (chunks 1-5)
    args, kwargs = mock_trainer.train.call_args_list[8]
    # 8th call is Fold 5, Model With
    assert len(args[0]) == 500 # train_end = 5 * 100

def test_ablation_with_garch(mocker):
    n_samples = 120
    returns = np.random.normal(0, 0.01, n_samples)
    features_with = np.random.randn(n_samples, 20, 18)
    features_without = np.random.randn(n_samples, 20, 12)
    targets = (np.random.rand(n_samples) > 0.5).astype(float)
    regimes = np.random.randint(0, 4, n_samples)
    returns_series = pd.Series(returns)

    # Mock GARCH
    mock_garch_cls = mocker.patch('src.evaluation.ablation.RegimeGARCH')
    mock_garch = mock_garch_cls.return_value
    mock_garch.position_size.return_value = 1.0

    # Mock LSTM
    mocker.patch('src.evaluation.ablation.LSTMTrainer')
    mocker.patch('src.evaluation.ablation.calculate_sharpe', return_value=1.0)

    results = run_ablation_study(returns, features_with, features_without, targets, regimes, returns_series)

    # GARCH fit_all should be called once per fold (5 times)
    assert mock_garch.fit_all.call_count == 5
    # GARCH position_size should be called for each sample in test sets (5 * 20 = 100 times)
    assert mock_garch.position_size.call_count == 100
