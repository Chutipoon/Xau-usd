#!/usr/bin/env python3
"""
Walk-forward backtest runner with reporting.

Usage:
    python scripts/run_backtest.py \
        --signals-path signals.parquet \
        --returns-path returns.parquet \
        --output-dir reports/
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from src.evaluation.backtester import WalkForwardBacktester, monte_carlo_simulation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--signals-path', required=True, help='Parquet file with bridge_forecast')
    parser.add_argument('--returns-path', required=True, help='Parquet file with returns')
    parser.add_argument('--output-dir', default='reports/', help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    signals_df = pd.read_parquet(args.signals_path)
    returns_df = pd.read_parquet(args.returns_path)

    signals = signals_df['bridge_forecast']
    returns = returns_df['returns']

    # Run walk-forward
    bt = WalkForwardBacktester(n_windows=12, test_months=1, overlap_weeks=2)
    wf_results = bt.run(signals, returns)

    # Stress test
    stress_periods = {
        '2008_crisis': ('2008-09-01', '2009-03-31'),
        '2020_covid': ('2020-02-01', '2020-05-31'),
        '2022_energy': ('2022-01-01', '2022-12-31'),
    }
    stress_results = bt.stress_test(signals, returns, stress_periods)

    # Monte Carlo
    mc_results = monte_carlo_simulation(returns, n_paths=1000)

    # Save summary
    summary = {
        'walk_forward': {k: v for k, v in wf_results.items() if k != 'window_results'},
        'stress_test': stress_results,
        'monte_carlo': {
            'p5_max_dd': mc_results['p5_max_dd'],
            'p95_sharpe': mc_results['p95_sharpe'],
            'risk_of_ruin_pct': mc_results['risk_of_ruin_pct'],
        }
    }

    with open(output_dir / 'backtest_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Plot equity curves
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Window results
    window_results = wf_results.get('window_results', [])
    for i in range(4):
        ax = axes[i // 2, i % 2]
        if i < len(window_results):
            window = window_results[i]
            ax.plot(window.get('equity_curve', []))
            ax.set_title(f"Window {i+1}: Sharpe={window.get('sharpe_ratio', 0):.2f}")
        else:
            ax.set_title(f"Window {i+1}: No data")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curves.png', dpi=150)

    print(f"✅ Backtest complete")
    print(f"   Sharpe: {wf_results.get('sharpe_ratio', 0):.2f}")
    print(f"   Max DD: {wf_results.get('max_drawdown', 0):.1%}")
    print(f"   Risk of Ruin: {mc_results['risk_of_ruin_pct']:.1f}%")
    print(f"   Reports saved to {output_dir}")

if __name__ == '__main__':
    main()
