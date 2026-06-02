import pandas as pd
import numpy as np
from typing import Dict, Any

def check_drawdown(equity_series: pd.Series, window_days: int = 30) -> Dict[str, Any]:
    """
    Returns: {'drawdown_30d': float, 'halt_required': bool}
    halt_required = True if drawdown_30d > 0.05
    """
    if len(equity_series) < 2:
        return {'drawdown_30d': 0.0, 'halt_required': False}

    # Use rolling window
    recent_equity = equity_series.tail(window_days)
    peak = recent_equity.expanding().max()
    drawdown = (recent_equity - peak) / peak
    max_drawdown = abs(drawdown.min())

    return {
        'drawdown_30d': float(max_drawdown),
        'halt_required': max_drawdown > 0.05
    }

def check_correlation(positions: Dict[str, float], returns: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns: {'max_correlation': float, 'breach': bool}
    breach = True if max_rolling_corr > 0.3
    """
    if len(positions) < 2:
        return {'max_correlation': 0.0, 'breach': False}

    # Get instruments in positions
    instruments = list(positions.keys())
    if not all(inst in returns.columns for inst in instruments):
        # Filter for existing instruments
        instruments = [inst for inst in instruments if inst in returns.columns]
        if len(instruments) < 2:
            return {'max_correlation': 0.0, 'breach': False}

    # Calculate correlation matrix for relevant returns
    corr_matrix = returns[instruments].corr()

    # Get max correlation (excluding diagonal)
    mask = np.ones(corr_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    max_corr = corr_matrix.where(mask).max().max()

    if np.isnan(max_corr):
        max_corr = 0.0

    return {
        'max_correlation': float(max_corr),
        'breach': max_corr > 0.3
    }
