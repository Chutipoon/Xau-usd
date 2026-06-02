import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

PRICE_FEATURES = ['returns_1h', 'returns_4h', 'returns_24h',
                  'realized_vol_7d', 'rsi_14', 'macd_signal',
                  'bb_position', 'atr_14', 'volume_zscore',
                  'yield_spread', 'cot_net_long', 'dxy_return']

GDELT_FEATURES = ['tone_7d_avg', 'tone_30d_avg', 'event_spike_zscore',
                  'tone_price_divergence', 'article_count_zscore', 'tone_momentum']

FEATURE_COLS = PRICE_FEATURES + GDELT_FEATURES # len=18

class LSTMSignalModel(nn.Module):
    def __init__(self, input_size: int = 18, hidden_sizes: List[int] = [64, 32, 16],
                 dropout: float = 0.2, sequence_length: int = 20):
        super(LSTMSignalModel, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.sequence_length = sequence_length

        # 3-layer LSTM
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.dropout3 = nn.Dropout(dropout)

        # Final Linear layer
        self.fc = nn.Linear(hidden_sizes[2], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, sequence_length, input_size)

        # LSTM layers
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        out, _ = self.lstm3(out)
        out = self.dropout3(out)

        # Take the last time step
        out = out[:, -1, :]

        # Final output
        out = self.fc(out)
        out = self.sigmoid(out)

        return out

class LSTMTrainer:
    def __init__(self, model: LSTMSignalModel, learning_rate: float = 0.001, batch_size: int = 32):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, early_stopping_patience: int = 10) -> Dict[str, Any]:

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        best_model_state = None

        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).view(-1, 1)

        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.FloatTensor(y_val).view(-1, 1)
        else:
            X_val_t, y_val_t = None, None

        for epoch in range(epochs):
            self.model.train()
            epoch_train_losses = []

            # Batch training
            permutation = torch.randperm(X_train_t.size(0))
            for i in range(0, X_train_t.size(0), self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_x, batch_y = X_train_t[indices], y_train_t[indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_train_losses.append(loss.item())

            avg_train_loss = np.mean(epoch_train_losses)
            train_losses.append(avg_train_loss)

            # Validation
            if X_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
                    val_loss = self.criterion(val_outputs, y_val_t).item()
                    val_losses.append(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        patience_counter = 0
                        best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    else:
                        patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if (epoch + 1) % 10 == 0:
                val_str = f", Val Loss: {val_losses[-1]:.4f}" if val_losses else ""
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}{val_str}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'best_epoch': best_epoch
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.FloatTensor(X)
        with torch.no_grad():
            outputs = self.model(X_t)
        return outputs.numpy()

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size': self.model.input_size,
            'hidden_sizes': self.model.hidden_sizes,
            'sequence_length': self.model.sequence_length
        }, path)

    @classmethod
    def load(cls, path: str) -> 'LSTMTrainer':
        checkpoint = torch.load(path)
        model = LSTMSignalModel(
            input_size=checkpoint['input_size'],
            hidden_sizes=checkpoint['hidden_sizes'],
            sequence_length=checkpoint['sequence_length']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer = cls(model)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return trainer
