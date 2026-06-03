import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler
from src.models.lstm_signal import LSTMSignalModel, LSTMTrainer
from src.models.garch_vol import RegimeGARCH

def calculate_sharpe(returns: np.ndarray, predictions: np.ndarray, position_sizes: np.ndarray = None) -> float:
    # simple strategy: long if prob > 0.5, else short
    direction = (predictions.flatten() > 0.5).astype(float) * 2 - 1

    if position_sizes is not None:
        signals = direction * position_sizes.flatten()
    else:
        signals = direction

    # Assuming predictions are for next step return
    # Alignment: prediction[i] is for return[i]
    strategy_returns = signals * returns

    if len(strategy_returns) < 2:
        return 0.0

    avg_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)

    if std_ret == 0:
        return 0.0

    # Annualized Sharpe (assuming hourly returns, 252 days * 24 hours)
    return float((avg_ret / std_ret) * np.sqrt(252 * 24))

def run_ablation_study(returns: np.ndarray,
                        features_with_gdelt: np.ndarray,
                        features_without_gdelt: np.ndarray,
                        targets: np.ndarray,
                        regimes: np.ndarray,
                        returns_series: pd.Series) -> Dict[str, Any]:
    """
    True Expanding-Window Walk-Forward:
    Fold 1: train 0-20%, test 20-24%
    Fold 2: train 0-40%, test 40-44%
    Fold 3: train 0-60%, test 60-64%
    Fold 4: train 0-80%, test 80-84%
    Fold 5: train 0-96%, test 96-100%
    """
    n_samples = len(returns)
    fold_configs = [
        (0.20, 0.24),
        (0.40, 0.44),
        (0.60, 0.64),
        (0.80, 0.84),
        (0.96, 1.00)
    ]

    fold_results = []
    sharpes_with = []
    sharpes_without = []

    # Sequence length from LSTM model requirements
    seq_len = 20

    for i, (train_pct, test_pct) in enumerate(fold_configs):
        train_end = int(n_samples * train_pct)
        test_end = int(n_samples * test_pct)

        # Data Leakage Fix: Scaler must fit ONLY on train set of each fold
        scaler_with = StandardScaler()
        # features are shape (samples, seq_len, features)
        # We need to reshape to fit scaler then reshape back
        s_with, sl_with, f_with = features_with_gdelt.shape
        X_train_with_raw = features_with_gdelt[:train_end].reshape(-1, f_with)
        scaler_with.fit(X_train_with_raw)

        X_train_with = scaler_with.transform(X_train_with_raw).reshape(-1, sl_with, f_with)
        X_test_with = scaler_with.transform(features_with_gdelt[train_end:test_end].reshape(-1, f_with)).reshape(-1, sl_with, f_with)

        scaler_without = StandardScaler()
        s_wo, sl_wo, f_wo = features_without_gdelt.shape
        X_train_without_raw = features_without_gdelt[:train_end].reshape(-1, f_wo)
        scaler_without.fit(X_train_without_raw)

        X_train_without = scaler_without.transform(X_train_without_raw).reshape(-1, sl_wo, f_wo)
        X_test_without = scaler_without.transform(features_without_gdelt[train_end:test_end].reshape(-1, f_wo)).reshape(-1, sl_wo, f_wo)

        y_train = targets[:train_end]
        y_test = targets[train_end:test_end]
        ret_test = returns[train_end:test_end]

        # Position sizing via GARCH
        garch = RegimeGARCH()
        garch.fit_all(returns_series.iloc[:train_end], regimes[:train_end])
        fold_regimes = regimes[train_end:test_end]
        pos_sizes = np.array([garch.position_size(r) for r in fold_regimes])

        # Train LSTM with GDELT (50 epochs + early stopping)
        model_with = LSTMSignalModel(input_size=f_with, sequence_length=sl_with)
        trainer_with = LSTMTrainer(model_with, batch_size=32)
        trainer_with.scaler = scaler_with
        trainer_with.train(X_train_with, y_train, X_val=X_test_with, y_val=y_test,
                          epochs=50, early_stopping_patience=10)
        preds_with = trainer_with.predict(X_test_with)
        sharpe_with = calculate_sharpe(ret_test, preds_with, pos_sizes)

        # Train LSTM without GDELT
        model_without = LSTMSignalModel(input_size=f_wo, sequence_length=sl_wo)
        trainer_without = LSTMTrainer(model_without, batch_size=32)
        trainer_without.scaler = scaler_without
        trainer_without.train(X_train_without, y_train, X_val=X_test_without, y_val=y_test,
                             epochs=50, early_stopping_patience=10)
        preds_without = trainer_without.predict(X_test_without)
        sharpe_without = calculate_sharpe(ret_test, preds_without, pos_sizes)

        sharpes_with.append(sharpe_with)
        sharpes_without.append(sharpe_without)

        fold_results.append({
            'fold': i,
            'sharpe_with': sharpe_with,
            'sharpe_without': sharpe_without
        })

    avg_sharpe_with = np.mean(sharpes_with)
    avg_sharpe_without = np.mean(sharpes_without)

    delta_pct = 0.0
    if avg_sharpe_without != 0:
        delta_pct = (avg_sharpe_with - avg_sharpe_without) / abs(avg_sharpe_without) * 100
    elif avg_sharpe_with > 0:
        delta_pct = 100.0

    decision = 'keep' if delta_pct > 5.0 else 'weight_zero'

    return {
        'sharpe_with_gdelt': float(avg_sharpe_with),
        'sharpe_without_gdelt': float(avg_sharpe_without),
        'sharpe_delta_pct': float(delta_pct),
        'decision': decision,
        'fold_results': fold_results
    }
