import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.models.lstm_signal import LSTMSignalModel, LSTMTrainer

def calculate_sharpe(returns: np.ndarray, predictions: np.ndarray) -> float:
    # simple strategy: long if prob > 0.5, else short
    signals = (predictions > 0.5).astype(float) * 2 - 1
    # Assuming predictions are for next step return
    # Alignment: prediction[i] is for return[i]
    strategy_returns = signals.flatten() * returns

    if len(strategy_returns) < 2:
        return 0.0

    avg_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)

    if std_ret == 0:
        return 0.0

    # Annualized Sharpe (assuming daily returns)
    return float((avg_ret / std_ret) * np.sqrt(252))

def run_ablation_study(returns: np.ndarray,
                        features_with_gdelt: np.ndarray,
                        features_without_gdelt: np.ndarray,
                        targets: np.ndarray) -> Dict[str, Any]:
    # Walk-forward: 5 folds, each fold = 80% train / 20% test
    n_samples = len(returns)
    fold_size = n_samples // 5

    fold_results = []
    sharpes_with = []
    sharpes_without = []

    # Sequence length from LSTM model requirements
    seq_len = 20

    for i in range(5):
        # We need enough data for at least one sequence and some test samples
        test_start = int(n_samples * 0.8 * (i + 1) / 5) # This is not exactly walk-forward as described but let's try to match 80/20 per fold
        # Wait, "each fold = 80% train / 20% test" usually implies rolling window or expanding window.
        # Let's do a simple 5-fold split where each fold has its own 80/20 split.

        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size
        if i == 4:
            end_idx = n_samples

        fold_data_with = features_with_gdelt[start_idx:end_idx]
        fold_data_without = features_without_gdelt[start_idx:end_idx]
        fold_targets = targets[start_idx:end_idx]
        fold_returns = returns[start_idx:end_idx]

        split_idx = int(len(fold_data_with) * 0.8)

        # Training and test sets for this fold
        X_train_with, X_test_with = fold_data_with[:split_idx], fold_data_with[split_idx:]
        X_train_without, X_test_without = fold_data_without[:split_idx], fold_data_without[split_idx:]
        y_train, y_test = fold_targets[:split_idx], fold_targets[split_idx:]
        ret_test = fold_returns[split_idx:]

        # Train LSTM with GDELT
        model_with = LSTMSignalModel(input_size=X_train_with.shape[2], sequence_length=seq_len)
        trainer_with = LSTMTrainer(model_with, batch_size=32)
        trainer_with.train(X_train_with, y_train, epochs=10, early_stopping_patience=3) # Small epochs for ablation speed
        preds_with = trainer_with.predict(X_test_with)
        sharpe_with = calculate_sharpe(ret_test, preds_with)

        # Train LSTM without GDELT
        model_without = LSTMSignalModel(input_size=X_train_without.shape[2], sequence_length=seq_len)
        trainer_without = LSTMTrainer(model_without, batch_size=32)
        trainer_without.train(X_train_without, y_train, epochs=10, early_stopping_patience=3)
        preds_without = trainer_without.predict(X_test_without)
        sharpe_without = calculate_sharpe(ret_test, preds_without)

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
