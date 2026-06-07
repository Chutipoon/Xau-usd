import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class WalkForwardBacktester:
    def __init__(self, n_windows: int = 12, test_months: int = 1, overlap_weeks: int = 2):
        """
        Args:
            n_windows: Number of walk-forward windows (default 12)
            test_months: Test period duration in months (default 1)
            overlap_weeks: Overlap between windows in weeks (default 2)
        """
        self.n_windows = n_windows
        self.test_months = test_months
        self.overlap_weeks = overlap_weeks

    def run(self, signals: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """
        Run walk-forward backtest.

        Args:
            signals: bridge_forecast values (-20 to +20) indexed by timestamp
            returns: actual XAU/USD log returns indexed by timestamp

        Returns:
            Summary metrics and per-window results.
        """
        # Ensure alignment
        data = pd.DataFrame({'signal': signals, 'return': returns}).dropna()
        if data.empty:
            return {}

        # Strategy return: signal scaled to [-1, 1] * log returns
        # Scaling bridge_forecast to [-1, 1] range
        data['strategy_return'] = (data['signal'] / 20.0) * data['return']

        start_date = data.index.min()
        window_results = []

        # Correct walk-forward: windows advance by test duration, adjusted for overlap
        # 1-month test, 2-week overlap -> advance by 2 weeks
        window_duration = timedelta(days=30 * self.test_months)
        step_delta = window_duration - timedelta(weeks=self.overlap_weeks)

        for i in range(self.n_windows):
            window_start = start_date + i * step_delta
            window_end = window_start + window_duration

            window_data = data[(data.index >= window_start) & (data.index < window_end)]

            if window_data.empty:
                continue

            # Log cumulative equity: exp(cumsum(log_returns))
            cum_log_ret = window_data['strategy_return'].cumsum()
            equity_curve = np.exp(cum_log_ret)

            metrics = self._compute_metrics(window_data['strategy_return'], equity_curve)
            metrics['start'] = window_start
            metrics['end'] = window_end
            metrics['equity_curve'] = equity_curve.tolist()
            window_results.append(metrics)

        if not window_results:
            return {}

        if len(window_results) < self.n_windows:
            logger.warning(f"Backtest produced only {len(window_results)} windows, but {self.n_windows} were requested. Data might be too short.")

        summary = {
            'sharpe_ratio': np.mean([w['sharpe_ratio'] for w in window_results]),
            'max_drawdown': np.mean([w['max_drawdown'] for w in window_results]),
            'calmar_ratio': np.mean([w['calmar_ratio'] for w in window_results]),
            'win_rate': np.mean([w['win_rate'] for w in window_results]),
            'avg_trade_return': np.mean([w['avg_trade_return'] for w in window_results]),
            'n_trades': int(np.sum([w['n_trades'] for w in window_results])),
            'window_results': window_results
        }

        return summary

    def _compute_metrics(self, strategy_returns: pd.Series, equity_curve: pd.Series) -> Dict[str, Any]:
        """
        Compute metrics from returns and equity curve.
        """
        if strategy_returns.empty:
            return {
                'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'calmar_ratio': 0.0,
                'win_rate': 0.0, 'avg_trade_return': 0.0, 'n_trades': 0
            }

        # Annualization factor
        diffs = strategy_returns.index.to_series().diff().dt.total_seconds().dropna()
        if not diffs.empty:
            median_diff = diffs.median()
            if median_diff <= 4000: # Hourly
                ann_factor = np.sqrt(252 * 24)
            else: # Daily
                ann_factor = np.sqrt(252)
        else:
            ann_factor = np.sqrt(252)

        # For log returns, Sharpe = mean / std
        avg_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        sharpe = (avg_ret / std_ret * ann_factor) if std_ret > 0 else 0.0

        # Max Drawdown from equity curve (Peak-to-Trough)
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = abs(drawdown.min())

        # Calmar = Total Return / Max Drawdown
        total_return = equity_curve.iloc[-1] - 1
        calmar = (total_return / max_dd) if max_dd > 0 else 0.0

        trades = strategy_returns[strategy_returns != 0]
        n_trades = len(trades)
        win_rate = (trades > 0).mean() if n_trades > 0 else 0.0
        avg_trade_return = trades.mean() if n_trades > 0 else 0.0

        return {
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'calmar_ratio': float(calmar),
            'win_rate': float(win_rate),
            'avg_trade_return': float(avg_trade_return),
            'n_trades': int(n_trades)
        }

    def stress_test(self, signals: pd.Series, returns: pd.Series, periods: Dict[str, tuple]) -> Dict[str, Any]:
        """
        Stress test on historical crisis periods.
        """
        results = {}
        data = pd.DataFrame({'signal': signals, 'return': returns}).dropna()
        data['strategy_return'] = (data['signal'] / 20.0) * data['return']

        for name, (start, end) in periods.items():
            period_data = data[(data.index >= start) & (data.index <= end)]
            if period_data.empty:
                results[name] = {
                    'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'calmar_ratio': 0.0,
                    'win_rate': 0.0, 'avg_trade_return': 0.0, 'n_trades': 0
                }
                continue

            equity_curve = np.exp(period_data['strategy_return'].cumsum())
            results[name] = self._compute_metrics(period_data['strategy_return'], equity_curve)

        return results

def monte_carlo_simulation(returns: pd.Series, n_paths: int = 1000) -> Dict[str, Any]:
    """
    Bootstrap resample returns and simulate equity curves using block bootstrap.
    """
    if returns.empty:
        return {}

    n_days = len(returns)
    block_size = 10
    n_blocks = int(np.ceil(n_days / block_size))

    paths = np.zeros((n_paths, n_days))
    sharpes = []
    max_dds = []

    # Pre-calculate blocks
    blocks = [returns.iloc[i:i+block_size].values for i in range(n_days - block_size + 1)]
    if not blocks:
        blocks = [returns.values]
        block_size = n_days

    for p in range(n_paths):
        sampled_indices = np.random.choice(len(blocks), size=n_blocks, replace=True)
        path_returns = np.concatenate([blocks[i] for i in sampled_indices])[:n_days]

        # Use log returns cumulative sum
        equity_curve = np.exp(np.cumsum(path_returns))
        paths[p, :] = equity_curve

        avg_ret = np.mean(path_returns)
        std_ret = np.std(path_returns)
        sharpe = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
        sharpes.append(sharpe)

        running_max = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - running_max) / running_max
        max_dds.append(abs(np.min(dd)))

    sharpes = np.array(sharpes)
    max_dds = np.array(max_dds)

    return {
        'p5_max_dd': float(np.percentile(max_dds, 5)),
        'p95_sharpe': float(np.percentile(sharpes, 95)),
        'risk_of_ruin_pct': float((max_dds > 0.20).mean() * 100),
        'paths': paths
    }
