import pytest
import torch
import numpy as np
import os
import tempfile
from src.models.lstm_signal import LSTMSignalModel, LSTMTrainer

def test_forward_pass():
    input_size = 18
    sequence_length = 20
    batch_size = 2
    model = LSTMSignalModel(input_size=input_size, sequence_length=sequence_length)

    # input shape: (batch, sequence_length, input_size)
    x = torch.randn(batch_size, sequence_length, input_size)
    output = model(x)

    # output shape: (batch, 1)
    assert output.shape == (batch_size, 1)

def test_output_range():
    input_size = 18
    sequence_length = 20
    model = LSTMSignalModel(input_size=input_size, sequence_length=sequence_length)

    x = torch.randn(10, sequence_length, input_size)
    output = model(x)

    assert torch.all(output >= 0)
    assert torch.all(output <= 1)

def test_save_load_round_trip():
    input_size = 18
    sequence_length = 20
    model = LSTMSignalModel(input_size=input_size, sequence_length=sequence_length)
    trainer = LSTMTrainer(model)

    x = np.random.randn(5, sequence_length, input_size).astype(np.float32)
    preds_before = trainer.predict(x)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        trainer.save(tmp.name)
        loaded_trainer = LSTMTrainer.load(tmp.name)
        os.unlink(tmp.name)

    preds_after = loaded_trainer.predict(x)
    np.testing.assert_array_almost_equal(preds_before, preds_after, decimal=5)
    assert loaded_trainer.model.input_size == input_size

def test_train_synthetic_data():
    input_size = 18
    sequence_length = 20
    n_samples = 100

    X = np.random.randn(n_samples, sequence_length, input_size).astype(np.float32)
    # y is probability of up move, so binary target
    y = (np.random.rand(n_samples) > 0.5).astype(np.float32)

    model = LSTMSignalModel(input_size=input_size, sequence_length=sequence_length)
    trainer = LSTMTrainer(model)

    # Run small number of epochs for test
    history = trainer.train(X, y, X_val=X, y_val=y, epochs=5, early_stopping_patience=2)

    assert 'train_loss' in history
    assert 'val_loss' in history
    assert len(history['train_loss']) > 0
    assert history['best_epoch'] >= 0
