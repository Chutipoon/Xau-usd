import argparse
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.hmm_regime import RegimeHMM, validate_hmm

def calculate_aic_bic(model, X):
    log_likelihood = model.model.score(X)
    n_features = X.shape[1]
    n_components = model.n_components

    # Number of parameters in GaussianHMM with full covariance
    # Transition matrix: n_components * (n_components - 1)
    # Initial probabilities: n_components - 1
    # Means: n_components * n_features
    # Covariances (full): n_components * n_features * (n_features + 1) / 2

    k = n_components * (n_components - 1) + (n_components - 1) + \
        n_components * n_features + \
        n_components * n_features * (n_features + 1) / 2

    n = X.shape[0]

    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    return aic, bic

def main():
    parser = argparse.ArgumentParser(description='Train HMM Regime Model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to features parquet file')
    parser.add_argument('--output', type=str, required=True, help='Path to save trained model')

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist.")
        return

    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)

    print("Training HMM model...")
    model = RegimeHMM(n_components=4, covariance_type='full', n_iter=100)
    model.fit(df)

    print(f"Saving model to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)

    # Preprocess for AIC/BIC calculation
    X = model.preprocess(df, fit_scaler=False)
    aic, bic = calculate_aic_bic(model, X)
    metrics = validate_hmm(model, df)

    print("\n--- Model Metrics ---")
    print(f"Log-Likelihood: {metrics['log_likelihood']:.4f}")
    print(f"AIC: {aic:.4f}")
    print(f"BIC: {bic:.4f}")

    print("\n--- State Distribution ---")
    for state, pct in metrics['state_distribution'].items():
        state_name = RegimeHMM.STATES.get(state, "Unknown")
        print(f"State {state} ({state_name}): {pct*100:.2f}%")

    print("\n--- Avg State Duration (Days) ---")
    for state, duration in metrics['avg_state_duration_days'].items():
        print(f"State {state}: {duration:.2f} days")

if __name__ == "__main__":
    main()
