import argparse
import pandas as pd
import numpy as np
import os
import sys
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.garch_vol import RegimeGARCH

def main():
    parser = argparse.ArgumentParser(description='Train GARCH Regime Model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to features parquet file')
    parser.add_argument('--output', type=str, required=True, help='Path to save trained model')

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist.")
        return

    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)

    # We need returns and regimes
    # If regimes are not in the parquet, we might need to use HMM to predict them first
    # For this training script, we assume regimes are provided or we use a default
    if 'regime' not in df.columns:
        print("Warning: 'regime' column not found. Deriving regimes from HMM if available...")
        # Placeholder for HMM prediction logic if needed
        # For now, we'll just assume they exist or use a single regime if missing
        df['regime'] = 0

    print("Training GARCH models per regime...")
    garch = RegimeGARCH()
    garch.fit_all(df['returns_1h'], df['regime'].values)

    print(f"Saving model to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(garch, args.output)

    print("GARCH training complete.")

if __name__ == "__main__":
    main()
