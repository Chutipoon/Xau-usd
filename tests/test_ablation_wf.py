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

    # Mock return values for predict
    # Fold 1: train 20% (120), test 4% (24)
    # Fold 2: train 40% (240), test 4% (24)
    # Fold 3: train 60% (360), test 4% (24)
    # Fold 4: train 80% (480), test 4% (24)
    # Fold 5: train 96% (576), test 4% (24)
    mock_trainer.predict.return_value = np.random.rand(24, 1)

    # Mock required inputs
    regimes = np.zeros(n_samples)
    returns_series = pd.Series(returns)

    results = run_ablation_study(returns, features_with, features_without, targets, regimes, returns_series)

    assert mock_trainer.train.call_count == 10

    # Check first fold training data size (0-20% of 600 = 120)
    args, kwargs = mock_trainer.train.call_args_list[0]
    # first call is Fold 1, Model With
    assert len(args[0]) == 120

    # Check last fold training data size (0-96% of 600 = 576)
    args, kwargs = mock_trainer.train.call_args_list[8]
    # 8th call is Fold 5, Model With
    assert len(args[0]) == 576

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

    # GARCH position_size should be called for each sample in test sets
    # n_samples = 120
    # Fold 1: train 20% (24), test 4% (24.8 -> 4, no 20-24% is 4.8 -> 4)
    # Actually let's calculate based on code:
    # fold_configs = [(0.20, 0.24), (0.40, 0.44), (0.60, 0.64), (0.80, 0.84), (0.96, 1.00)]
    # for 120 samples:
    # F1: 24, 28 -> test size 4
    # F2: 48, 52 -> test size 4
    # F3: 72, 76 -> test size 4
    # F4: 96, 100 -> test size 4
    # F5: 115, 120 -> test size 5
    # Total test samples: 4+4+4+4+5 = 21
    assert mock_garch.position_size.call_count == 21
