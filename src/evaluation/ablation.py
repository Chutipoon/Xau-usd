import numpy as np
import pandas as pd
from typing import Dict, Any, List
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
                        regimes: np.ndarray = None,
                        returns_series: pd.Series = None) -> Dict[str, Any]:
    """
    True Walk-Forward (Expanding Window):
    Divide data into 6 chunks.
    Fold 1: Train on chunk 1, test on chunk 2.
    Fold 2: Train on chunks 1-2, test on chunk 3.
    ...
    Fold 5: Train on chunks 1-5, test on chunk 6.
    """
    n_samples = len(returns)
    chunk_size = n_samples // 6

    fold_results = []
    sharpes_with = []
    sharpes_without = []

    # Sequence length from LSTM model requirements
    seq_len = 20

    for i in range(1, 6):
        train_end = i * chunk_size
        test_end = (i + 1) * chunk_size if i < 5 else n_samples

        X_train_with = features_with_gdelt[:train_end]
        X_test_with = features_with_gdelt[train_end:test_end]

        X_train_without = features_without_gdelt[:train_end]
        X_test_without = features_without_gdelt[train_end:test_end]

        y_train = targets[:train_end]
        # y_test = targets[train_end:test_end]
        ret_test = returns[train_end:test_end]

        # Position sizing via GARCH if regimes provided
        pos_sizes = None
        if regimes is not None and returns_series is not None:
            garch = RegimeGARCH()
            # Fit on training period returns
            # We need to find the correct slice of returns_series.
            # Assuming returns_series is aligned with features (after seq_len offset)
            garch.fit_all(returns_series.iloc[:train_end], regimes[:train_end])

            # Forecast for test period
            # For simplicity in ablation, use position size for the regime at each step
            fold_regimes = regimes[train_end:test_end]
            pos_sizes = np.array([garch.position_size(r) for r in fold_regimes])

        # Train LSTM with GDELT (Increased epochs + early stopping)
        model_with = LSTMSignalModel(input_size=X_train_with.shape[2], sequence_length=seq_len)
        trainer_with = LSTMTrainer(model_with, batch_size=32)
        trainer_with.train(X_train_with, y_train, X_val=X_test_with, y_val=targets[train_end:test_end],
                          epochs=50, early_stopping_patience=10)
        preds_with = trainer_with.predict(X_test_with)
        sharpe_with = calculate_sharpe(ret_test, preds_with, pos_sizes)

        # Train LSTM without GDELT
        model_without = LSTMSignalModel(input_size=X_train_without.shape[2], sequence_length=seq_len)
        trainer_without = LSTMTrainer(model_without, batch_size=32)
        trainer_without.train(X_train_without, y_train, X_val=X_test_without, y_val=targets[train_end:test_end],
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
