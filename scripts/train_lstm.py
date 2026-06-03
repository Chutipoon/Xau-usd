import argparse
import pandas as pd
import numpy as np
import torch
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.lstm_signal import LSTMSignalModel, LSTMTrainer, FEATURE_COLS, GDELT_FEATURES

def create_sequences(data, target, sequence_length):
    xs = []
    ys = []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = target[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def permutation_importance(trainer, X_val, y_val, features):
    baseline_outputs = trainer.predict(X_val)
    baseline_loss = np.mean((baseline_outputs - y_val.reshape(-1, 1))**2)

    importances = {}
    for i, feature in enumerate(features):
        X_val_permuted = X_val.copy()
        # Permute the feature across all time steps in all sequences
        for seq in range(X_val_permuted.shape[0]):
            np.random.shuffle(X_val_permuted[seq, :, i])

        permuted_outputs = trainer.predict(X_val_permuted)
        permuted_loss = np.mean((permuted_outputs - y_val.reshape(-1, 1))**2)
        importances[feature] = permuted_loss - baseline_loss

    return importances

def main():
    parser = argparse.ArgumentParser(description='Train LSTM Signal Model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to features parquet file')
    parser.add_argument('--output', type=str, required=True, help='Path to save trained model')
    parser.add_argument('--gdelt-features', type=lambda x: (str(x).lower() == 'true'), default=True, help='Include GDELT features')
    parser.add_argument('--sequence-length', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist.")
        return

    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)

    features = FEATURE_COLS
    if not args.gdelt_features:
        print("Excluding GDELT features...")
        features = [f for f in FEATURE_COLS if f not in GDELT_FEATURES]

    # Preprocessing
    df = df.ffill().dropna()

    X_raw = df[features].values
    # Assuming the target is 'target_move_up' or similar, let's look for a boolean target
    # For now, let's assume the user has a 'target' column or we create a synthetic one for the script to be runnable
    if 'target' not in df.columns:
        print("Warning: 'target' column not found. Creating synthetic binary target based on returns.")
        # Try to find a return column to derive target
        ret_col = 'returns_1h' if 'returns_1h' in df.columns else features[0]
        df['target'] = (df[ret_col].shift(-1) > 0).astype(float)
        df = df.dropna()
        X_raw = df[features].values

    y_raw = df['target'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_raw, args.sequence_length)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    print(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples...")

    model = LSTMSignalModel(input_size=len(features), sequence_length=args.sequence_length)
    trainer = LSTMTrainer(model, batch_size=args.batch_size)
    trainer.scaler = scaler # Assign scaler to trainer for persistence

    results = trainer.train(X_train, y_train, X_val, y_val, epochs=args.epochs)

    print(f"\nFinal Val Loss: {results['val_loss'][-1]:.4f}")
    print(f"Best Epoch: {results['best_epoch'] + 1}")

    print("\nFeature Importance (Permutation):")
    importances = permutation_importance(trainer, X_val, y_val, features)
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for feature, imp in sorted_importances:
        print(f"{feature}: {imp:.6f}")

    print(f"\nSaving model to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    trainer.save(args.output)

if __name__ == "__main__":
    main()
