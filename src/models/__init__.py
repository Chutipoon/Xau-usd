"""ML models: HMM regime, LSTM signal, GARCH volatility."""

from .hmm_regime import RegimeHMM
from .garch_vol import RegimeGARCH

def LSTMSignalModel(*args, **kwargs):
    from .lstm_signal import LSTMSignalModel as _LSTMSignalModel
    return _LSTMSignalModel(*args, **kwargs)

def LSTMTrainer(*args, **kwargs):
    from .lstm_signal import LSTMTrainer as _LSTMTrainer
    return _LSTMTrainer(*args, **kwargs)

__all__ = ["RegimeHMM", "LSTMSignalModel", "LSTMTrainer", "RegimeGARCH"]
