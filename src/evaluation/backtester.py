import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import timedelta

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

        # Strategy return: signal scaled to [-1, 1] * returns
        data['strategy_return'] = (data['signal'] / 20.0) * data['return']

        start_date = data.index.min()
        window_results = []

        # We need to define the windows.
        # Shift each window by (test_months - overlap_weeks_in_months) ?
        # The prompt says "12 windows, 1-month test, 2-week overlap".
        # This usually means window i+1 starts 2 weeks before window i ends.
        # Window duration is 1 month (approx 30 days or 4 weeks).
        # Overlap is 2 weeks. So step is 2 weeks.

        step_delta = timedelta(weeks=self.overlap_weeks)
        window_duration = timedelta(days=30 * self.test_months)

        for i in range(self.n_windows):
            window_start = start_date + i * step_delta
            window_end = window_start + window_duration

            window_data = data[(data.index >= window_start) & (data.index < window_end)]

            if window_data.empty:
                continue

            equity_curve = (1 + window_data['strategy_return']).cumprod()
            metrics = self._compute_metrics(window_data['strategy_return'], equity_curve)
            metrics['start'] = window_start
            metrics['end'] = window_end
            metrics['equity_curve'] = equity_curve.tolist()
            window_results.append(metrics)

        if not window_results:
            return {}

        # Aggregate metrics (mean of windows or overall equity curve?)
        # Typically walk-forward summary is the average of window metrics
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
        # Infer frequency
        diffs = strategy_returns.index.to_series().diff().dt.total_seconds().dropna()
        if not diffs.empty:
            median_diff = diffs.median()
            # 1 hour = 3600s, 1 day = 86400s
            if median_diff <= 4000: # Hourly
                ann_factor = np.sqrt(252 * 24)
            else: # Daily or lower
                ann_factor = np.sqrt(252)
        else:
            ann_factor = np.sqrt(252)

        avg_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        sharpe = (avg_ret / std_ret * ann_factor) if std_ret > 0 else 0.0

        # Max Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = abs(drawdown.min())

        calmar = (avg_ret * (ann_factor**2) / max_dd) if max_dd > 0 else 0.0

        # Trades: A trade is defined as a non-zero position
        # For simple metrics, we can treat each period as a trade or
        # count flips in signal. But prompt asks for "win_rate" and "avg_trade_return".
        # Let's define trades by contiguous non-zero strategy returns.

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

            equity_curve = (1 + period_data['strategy_return']).cumprod()
            results[name] = self._compute_metrics(period_data['strategy_return'], equity_curve)

        return results

def monte_carlo_simulation(returns: pd.Series, n_paths: int = 1000) -> Dict[str, Any]:
    """
    Bootstrap resample returns and simulate equity curves using block bootstrap.
    """
    if returns.empty:
        return {}

    n_days = len(returns)
    # Block size: let's use 10 periods
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
        # Sample blocks with replacement
        sampled_indices = np.random.choice(len(blocks), size=n_blocks, replace=True)
        path_returns = np.concatenate([blocks[i] for i in sampled_indices])[:n_days]

        equity_curve = np.cumprod(1 + path_returns)
        paths[p, :] = equity_curve

        # Calc metrics for this path
        avg_ret = np.mean(path_returns)
        std_ret = np.std(path_returns)

        # Annualization for MC (assume daily as per prompt "returns: daily log returns")
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
