import argparse
import pandas as pd
import numpy as np
import os
import sys
import json
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.ablation import run_ablation_study
from src.models.lstm_signal import FEATURE_COLS, GDELT_FEATURES

def create_sequences(data, target, returns, sequence_length):
    xs = []
    ys = []
    rs = []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = target[i + sequence_length]
        r = returns[i + sequence_length]
        xs.append(x)
        ys.append(y)
        rs.append(r)
    return np.array(xs), np.array(ys), np.array(rs)

def main():
    parser = argparse.ArgumentParser(description='Run GDELT Ablation Study')
    parser.add_argument('--data-path', type=str, required=True, help='Path to features parquet file')
    parser.add_argument('--output', type=str, default='reports/ablation_report.json', help='Path to save results')

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist.")
        return

    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    df = df.ffill().dropna()

    features_with = FEATURE_COLS
    features_without = [f for f in FEATURE_COLS if f not in GDELT_FEATURES]

    if 'target' not in df.columns:
        # Create synthetic target if not exists
        ret_col = 'returns_1h' if 'returns_1h' in df.columns else 'returns' if 'returns' in df.columns else None
        if not ret_col:
            # try to find any return column
            ret_cols = [c for c in df.columns if 'return' in c.lower()]
            ret_col = ret_cols[0] if ret_cols else None

        if ret_col:
            df['target'] = (df[ret_col].shift(-1) > 0).astype(float)
            df['actual_return'] = df[ret_col].shift(-1)
        else:
            print("Error: Could not find return column for target generation.")
            return

    df = df.dropna()

    X_with_raw = df[features_with].values
    X_without_raw = df[features_without].values
    y_raw = df['target'].values
    ret_raw = df['actual_return'].values

    # Scale
    scaler_with = StandardScaler()
    X_with_scaled = scaler_with.fit_transform(X_with_raw)

    scaler_without = StandardScaler()
    X_without_scaled = scaler_without.fit_transform(X_without_raw)

    # Create sequences
    seq_len = 20
    X_with_seq, y_seq, ret_seq = create_sequences(X_with_scaled, y_raw, ret_raw, seq_len)
    X_without_seq, _, _ = create_sequences(X_without_scaled, y_raw, ret_raw, seq_len)

    print("Running ablation study (5 folds walk-forward)...")
    results = run_ablation_study(ret_seq, X_with_seq, X_without_seq, y_seq)

    print("\n--- Ablation Study Report ---")
    print(f"Avg Sharpe (With GDELT): {results['sharpe_with_gdelt']:.4f}")
    print(f"Avg Sharpe (Without GDELT): {results['sharpe_without_gdelt']:.4f}")
    print(f"Sharpe Delta: {results['sharpe_delta_pct']:.2f}%")
    print(f"Decision Recommendation: {results['decision'].upper()}")

    print("\nFold Results:")
    for fold in results['fold_results']:
        print(f"Fold {fold['fold']}: With={fold['sharpe_with']:.4f}, Without={fold['sharpe_without']:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
