# XAU/USD Trading System — Deployment Guide

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ with TimescaleDB 2.x extension installed
- Grafana 9.x (optional, for monitoring)
- Apache Airflow 2.8+ (optional, for orchestration)
- pysystemtrade installed: `pip install git+https://github.com/robcarver17/pysystemtrade`
- Paper or live trading account (OANDA, Interactive Brokers, or equivalent)

## Environment Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export TIMESCALE_URL="postgresql://user:password@localhost:5432/xauusd"
export PYSYSTEMTRADE_DB="/path/to/pysystemtrade/data"
export OANDA_API_KEY="your_oanda_paper_key"
export OANDA_ACCOUNT_ID="your_account_id"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."  # optional, for alerts
```

### 3. Create TimescaleDB and schema

```bash
psql -U postgres -c "CREATE DATABASE xauusd;"
psql -U postgres -d xauusd -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
psql -U postgres -d xauusd < db/schema.sql
psql -U postgres -d xauusd < db/migrations/002_add_signals_table.sql
```

## Data Ingestion Setup

### 1. Fetch historical data (first run)

```bash
# Dukascopy OHLCV (1 year)
python -c "
from src.data.dukascopy_fetcher import fetch_and_store
import psycopg2
import os
from datetime import datetime, timedelta

db = psycopg2.connect(os.getenv('TIMESCALE_URL'))
fetch_and_store('XAUUSD',
                datetime.now() - timedelta(days=365),
                datetime.now(),
                '1h', db)
db.close()
"

# FRED macro series
python -c "
from src.data.fred_fetcher import fetch_and_store_fred
import psycopg2
import os
from datetime import datetime, timedelta

db = psycopg2.connect(os.getenv('TIMESCALE_URL'))
fetch_and_store_fred(['real_yield_10y', 'dxy_index', 'vix'],
                      datetime.now() - timedelta(days=365),
                      datetime.now(), db)
db.close()
"

# COT weekly
python scripts/sync_cot.py --year-from 2020 --year-to 2024
```

### 2. Start Airflow (if using orchestration)

```bash
# Initialize Airflow DB
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init

# Start scheduler in background
airflow scheduler &

# Start webserver (optional)
airflow webserver &

# Verify DAGs loaded
airflow dags list
```

Airflow will run 3 DAGs on schedule:
- `dag_data_ingestion`: 6am weekdays (fetch OHLCV, FRED, GDELT, COT)
- `dag_feature_pipeline`: 6:30am (compute features)
- `dag_signal_generation`: 7am (HMM, LSTM, GARCH, bridge forecast)

## Model Training (One-time setup)

### 1. Train HMM (4-state regime detector)

```bash
python scripts/train_hmm.py \
  --data-path data/features.parquet \
  --output models/hmm.pkl
```

Output: `models/hmm.pkl` + regime statistics

### 2. Train LSTM (signal model)

```bash
python scripts/train_lstm.py \
  --data-path data/features.parquet \
  --output models/lstm.pt \
  --gdelt-features true
```

Output: `models/lstm.pt` + feature importance

### 3. Run ablation study

```bash
python scripts/run_ablation.py \
  --signals-path signals.parquet \
  --output-path reports/ablation_report.json
```

Checks: Does GDELT improve Sharpe? (→ keep or weight-zero decision)

### 4. Run backtest

```bash
python scripts/run_backtest.py \
  --signals-path signals.parquet \
  --returns-path returns.parquet \
  --output-dir reports/
```

Output: `backtest_summary.json` + `equity_curves.png`

**Pass criteria:**
- Sharpe > 2.0
- Max Drawdown < 8%
- Stress test: survives 2008, 2020, 2022 periods

## Live System Start

### 1. Verify data freshness

```bash
python scripts/check_health.py
```

Expected output:
```
✅ OHLCV: updated 5 min ago
✅ FRED: updated 1 day ago
✅ COT: updated 3 days ago
✅ GDELT: updated 15 min ago
✅ Signals: updated 2 min ago
```

### 2. Start watchdog (continuous monitoring)

```bash
python src/execution/watchdog_service.py &
```

Watchdog monitors:
- Drawdown (halt at 5%/30d)
- Data lag (alert if > 2h)
- Signal freshness (alert if > 24h)
- Model outputs (NaN check)
- GDELT health (fallback to COT+macro)

### 3. Configure pysystemtrade

Edit `config/system.yaml`:
```yaml
instruments:
  - GOLD
instrument_weights:
  GOLD: 1.0
risk_overlay:
  max_portfolio_leverage: 2.0
  max_correlation_risk: 0.3
  drawdown_fraction: 0.05
position_limits:
  GOLD:
    max_position: 10
    min_position: -10
```

Start pysystemtrade system:
```bash
python -c "
from pysystemtrade.systems.system_builder import SystemBuilder
from sysdata.config.configdata import Config
config = Config('config/system.yaml')
system = SystemBuilder(config=config).get_system()
print(f'Instruments: {system.get_instrument_list()}')
"
```

### 4. Set up Grafana monitoring (optional)

1. Import dashboard: `monitoring/grafana_dashboard.json`
2. Configure TimescaleDB datasource (default: `PostgreSQL`)
3. Set up alert channels (Slack, email)
4. Enable alert rules from `monitoring/alerts.yaml`

Dashboard shows:
- Data freshness (Panel 1)
- Current regime (Panel 2)
- GDELT event spike (Panel 3)
- Bridge forecast gauge (Panel 4)
- LSTM signal time series (Panel 5)
- System alerts & procedures (Panel 6)

## Emergency Procedures

### Manual Emergency Stop

```bash
python scripts/emergency_stop.py --reason "manual operator halt"
```

This will:
1. Zero all signals in DB
2. Close all open positions
3. Write audit log with timestamp, reason, positions_closed
4. Print confirmation to stderr

**Execution time: < 1 second (tested)**

### Post-Stop Checklist

```
[ ] Confirm all positions FLAT in broker UI
[ ] Check audit log: tail logs/emergency_stop.log
[ ] Review trigger reason
[ ] Contact ops lead before resuming
```

### System Recovery

```bash
# 1. Investigate root cause (logs, DB queries)
tail -50 logs/*.log
psql -d xauusd -c "SELECT * FROM emergency_stop_log ORDER BY timestamp DESC LIMIT 5;"

# 2. Fix issue (retrain if model drift, restart if DAG hung, etc.)

# 3. Restart system
python scripts/check_health.py          # Verify data ok
python src/execution/watchdog_service.py &      # Resume monitoring
# Airflow/pysystemtrade resume automatically
```

### GDELT Failover (if gdelt-doc-api down)

Automatically triggers fallback: COT + FRED macro only
- High-Vol Choppy regime detection degrades ~15% accuracy
- System continues trading with reduced signal sophistication
- Monitor GDELT recovery: `check_gdelt_health()` in watchdog

## Troubleshooting

### Issue: Dukascopy fetcher HTTP 404/500

```bash
# Validate endpoint manually
curl "https://datafeed.dukascopy.com/datafeed/XAUUSD/2024/01/01/1h_candles.bi5" -I

# If down, use Yahoo fallback temporarily
python -c "
import yfinance as yf
data = yf.download('GC=F', start='2024-01-01', interval='1h')
"
```

### Issue: FRED API rate limit (429)

```bash
# OpenBB caches requests; wait 1 hour or use yfinance fallback
python -c "
import yfinance as yf
dxy = yf.download('DXY=F', start='2024-01-01')
"
```

### Issue: Airflow DAG hung

```bash
airflow dags trigger dag_signal_generation
airflow tasks list dag_data_ingestion
airflow logs dag_data_ingestion
```

### Issue: pysystemtrade position not updated

```bash
# Check broker connection
python -c "from pysystemtrade.systems.basesystem import System; print(System().positions())"

# Verify bridge_forecast is flowing to external_forecast
psql -d xauusd -c "SELECT * FROM signals ORDER BY timestamp DESC LIMIT 1;"
```

### Issue: High latency (LSTM inference > 100ms)

```bash
# Check system resources
top  # or Task Manager on Windows

# Profile LSTM
python -c "
import torch
import time
from src.models.lstm_signal import LSTMSignalModel

model = LSTMSignalModel()
model.load_state_dict(torch.load('models/lstm.pt'))
model.eval()

x = torch.randn(1, 20, 18)
start = time.time()
with torch.no_grad():
    out = model(x)
print(f'Inference: {(time.time()-start)*1000:.1f}ms')
"
```

## Monitoring & Logging

Logs directory: `logs/`
- `emergency_stop.log` — emergency stop audit trail
- `airflow/` — Airflow scheduler + task logs
- `watchdog.log` — health check logs

Query signals/positions/alerts:
```sql
-- Latest signals
SELECT * FROM signals ORDER BY timestamp DESC LIMIT 10;

-- Emergency stop history
SELECT * FROM emergency_stop_log ORDER BY timestamp DESC LIMIT 10;

-- GDELT features
SELECT ts, event_spike_zscore, tone_7d_avg FROM gdelt_features
  ORDER BY ts DESC LIMIT 10;

-- Regime transitions
SELECT timestamp, hmm_regime FROM signals
  WHERE timestamp > NOW() - INTERVAL '7 days'
  ORDER BY timestamp;
```

## Deployment Checklist

- [ ] Environment variables set
- [ ] TimescaleDB created + schema loaded
- [ ] 1yr historical data ingested
- [ ] HMM, LSTM trained + backtested (Sharpe > 2.0, DD < 8%)
- [ ] Ablation study passed (GDELT keep/zero decision made)
- [ ] pysystemtrade config.yaml created
- [ ] Watchdog health check passes
- [ ] Grafana dashboard imports + panels show data
- [ ] Emergency stop tested (< 1 second)
- [ ] Team trained on emergency procedures
- [ ] Paper trading started (60-day minimum)
- [ ] Slippage vs backtest ±1% validation
- [ ] Only then: 10% live → scale up

## Support

- **Docs:** See `README.md` + inline code comments
- **Issues:** GitHub Issues (tag: #infra, #urgent)
- **On-call:** Ops lead contact in team Slack
