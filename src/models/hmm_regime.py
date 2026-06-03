import pandas as pd
import numpy as np
import joblib
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional

class RegimeHMM:
    STATES = {0: 'trending_bull', 1: 'trending_bear',
              2: 'high_vol_choppy', 3: 'low_vol_range'}

    def __init__(self, n_components: int = 4, covariance_type: str = 'full', n_iter: int = 100):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def preprocess(self, feature_df: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        # Handle NaN: forward-fill then drop leading NaN rows
        df = feature_df.ffill().dropna()

        if df.empty:
            raise ValueError("Feature DataFrame is empty after NaN handling.")

        # Scale features
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(df)
        else:
            scaled_data = self.scaler.transform(df)

        return scaled_data

    def fit(self, feature_df: pd.DataFrame) -> 'RegimeHMM':
        # feature_df columns: [returns, log_volume, realized_vol,
        #                       yield_spread, cot_net_long, event_spike_zscore]
        X = self.preprocess(feature_df, fit_scaler=True)
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        # Returns Viterbi-decoded state sequence
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")
        X = self.preprocess(feature_df, fit_scaler=False)
        return self.model.predict(X)

    def predict_proba(self, feature_df: pd.DataFrame) -> np.ndarray:
        # Returns posterior probability matrix (n_samples, 4)
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict_proba.")
        X = self.preprocess(feature_df, fit_scaler=False)
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray) -> float:
        # Wrapper for internal model score
        return float(self.model.score(X))

    @property
    def converged(self) -> bool:
        # Check if the model has converged
        return self.model.monitor_.converged

    def save(self, path: str):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'n_iter': self.n_iter,
            'is_fitted': self.is_fitted
        }, path)

    @classmethod
    def load(cls, path: str) -> 'RegimeHMM':
        data = joblib.load(path)
        instance = cls(
            n_components=data['n_components'],
            covariance_type=data['covariance_type'],
            n_iter=data['n_iter']
        )
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.is_fitted = data['is_fitted']
        return instance

def validate_hmm(model: RegimeHMM, feature_df: pd.DataFrame) -> Dict[str, Any]:
    # Handle NaN for validation
    df = feature_df.ffill().dropna()
    X = model.scaler.transform(df)

    # Predict states
    states = model.model.predict(X)

    # State Distribution
    unique, counts = np.unique(states, return_counts=True)
    state_dist = {int(u): float(c) / len(states) for u, c in zip(unique, counts)}

    # Log Likelihood
    log_likelihood = model.model.score(X)

    # Average State Duration
    durations = []
    if len(states) > 0:
        current_state = states[0]
        current_duration = 0
        state_durations = {i: [] for i in range(model.n_components)}

        for state in states:
            if state == current_state:
                current_duration += 1
            else:
                state_durations[current_state].append(current_duration)
                current_state = state
                current_duration = 1
        state_durations[current_state].append(current_duration)

        avg_state_duration = {
            int(k): np.mean(v) if v else 0.0
            for k, v in state_durations.items()
        }
    else:
        avg_state_duration = {i: 0.0 for i in range(model.n_components)}

    return {
        'state_distribution': state_dist,
        'log_likelihood': float(log_likelihood),
        'avg_state_duration_days': avg_state_duration
    }
