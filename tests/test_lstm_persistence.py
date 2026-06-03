import pytest
import numpy as np
import os
import tempfile
from sklearn.preprocessing import StandardScaler
from src.models.lstm_signal import LSTMSignalModel, LSTMTrainer

def test_lstm_persistence_with_scaler():
    input_size = 18
    sequence_length = 20
    model = LSTMSignalModel(input_size=input_size, sequence_length=sequence_length)
    trainer = LSTMTrainer(model)

    # Create and fit a scaler
    scaler = StandardScaler()
    dummy_data = np.random.randn(100, input_size)
    scaler.fit(dummy_data)
    trainer.scaler = scaler

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        trainer.save(tmp.name)
        loaded_trainer = LSTMTrainer.load(tmp.name)
        os.unlink(tmp.name)

    assert loaded_trainer.scaler is not None
    assert isinstance(loaded_trainer.scaler, StandardScaler)
    np.testing.assert_array_almost_equal(loaded_trainer.scaler.mean_, scaler.mean_)
    np.testing.assert_array_almost_equal(loaded_trainer.scaler.scale_, scaler.scale_)

def test_lstm_trainer_init_scaler_none():
    model = LSTMSignalModel()
    trainer = LSTMTrainer(model)
    assert trainer.scaler is None
