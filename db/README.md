# Database Schema — XAU/USD Trading System

**Engine:** PostgreSQL 14+ + TimescaleDB 2.x  
**Schema version:** 1.0

## Quick Start

```bash
# Apply schema (idempotent — safe to re-run)
psql $DATABASE_URL -f db/schema.sql

# Reset (DEV ONLY — destroys all data)
psql $DATABASE_URL -f db/teardown.sql
```

---

## Tables

### `ohlcv_xauusd` — Price Data (Hypertable)
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | TIMESTAMPTZ PK | Bar open time (UTC) |
| `open_price` | DOUBLE PRECISION | Open price (USD/oz) |
| `high` | DOUBLE PRECISION | High price |
| `low` | DOUBLE PRECISION | Low price |
| `close_price` | DOUBLE PRECISION | Close price |
| `volume` | DOUBLE PRECISION | Volume (contracts) |
| `source` | VARCHAR(20) | `dukascopy` \| `yahoo` \| `manual` |

**Chunk interval:** 1 month  
**Source:** `src/data/dukascopy_fetcher.py`  
**Constraints:** close BETWEEN 500–5000, high ≥ low, all prices > 0

---

### `cot_xauusd` — CFTC COT Report (Weekly)
| Column | Type | Description |
|--------|------|-------------|
| `week_date` | DATE PK | Report publication date (Tuesday) |
| `net_long` | BIGINT | NonComm long − NonComm short |
| `net_short` | BIGINT | Commercial short − Commercial long |
| `noncomm_net` | BIGINT | Non-Commercial net (speculator position) |
| `comm_net` | BIGINT | Commercial net (hedger position) |

**Source:** `src/data/cot_fetcher.py` via `cftc-cot` library  
**Commodity code:** 088691 (COMEX Gold futures)  
**Update:** Every Friday after 3:30pm ET

---

### `gdelt_features` — GDELT Sentiment Features (Hypertable)
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | TIMESTAMPTZ PK | Feature computation time (UTC) |
| `tone_7d_avg` | DOUBLE PRECISION | 7-day rolling mean of GDELT tone score |
| `tone_30d_avg` | DOUBLE PRECISION | 30-day rolling mean of GDELT tone score |
| `event_spike_zscore` | DOUBLE PRECISION | Z-score of article count vs 30d baseline |
| `tone_price_divergence` | DOUBLE PRECISION | tone_7d_avg × (−1 × price_return_24h) |
| `article_count` | INTEGER | Raw article count for this hour |

**Chunk interval:** 1 week  
**Source:** `src/data/gdelt_fetcher.py` via `gdelt-doc-api` (15-min lag)  
**Note:** event_spike_zscore > 2.0 = potential High-Vol Choppy regime trigger

---

### `macro_fred` — FRED Macroeconomic Series (Long Format)
| Column | Type | Description |
|--------|------|-------------|
| `obs_date` | DATE PK | Observation date |
| `series_id` | VARCHAR(20) PK | FRED series code |
| `obs_value` | DOUBLE PRECISION | Observation value (NULL = no data that day) |
| `updated_at` | TIMESTAMPTZ | Last upsert time |

**Source:** `src/data/fred_fetcher.py` via OpenBB SDK  
**Key series:**
| series_id | Description |
|-----------|-------------|
| `DFII10` | 10Y Real Yield |
| `DFII2` | 2Y Real Yield |
| `T10YIE` | 10Y Breakeven Inflation |
| `DTWEXBGS` | DXY (Broad Dollar Index) |
| `VIXCLS` | VIX |
| `FEDFUNDS` | Fed Funds Rate |

---

### `signals` — Trading Signals Output (Hypertable)
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | TIMESTAMPTZ PK | Signal generation time (UTC) |
| `hmm_regime` | SMALLINT | Dominant regime 0–3 |
| `hmm_posterior` | JSONB | Posterior probabilities `[p0,p1,p2,p3]` |
| `lstm_signal` | DOUBLE PRECISION | Direction probability 0.0–1.0 |
| `garch_vol` | DOUBLE PRECISION | Annualized vol forecast |
| `bridge_forecast` | DOUBLE PRECISION | pysystemtrade forecast scale −20 to +20 |

**Chunk interval:** 1 month  
**Source:** `src/execution/regime_signal_bridge.py`  
**Consumed by:** pysystemtrade external forecast mechanism

---

## Views

| View | Description |
|------|-------------|
| `v_latest_signal` | Most recent row from `signals` |
| `v_latest_gdelt` | Most recent row from `gdelt_features` |
| `v_data_freshness` | MAX(timestamp) per feed — used by Grafana + watchdog |

---

## Data Freshness SLA

| Feed | Expected lag | Alert threshold |
|------|-------------|-----------------|
| ohlcv | < 5 min (live) | > 2 hours |
| gdelt | 15 min (API lag) | > 1 hour |
| cot | Weekly (Fri) | > 8 days |
| macro_fred | Daily (6am) | > 2 days |
| signals | < 15 min | > 8 hours |

---

## Indexes

| Table | Index | Purpose |
|-------|-------|---------|
| ohlcv_xauusd | timestamp DESC | Time-range queries (primary) |
| ohlcv_xauusd | (source, timestamp DESC) | Filter by data source |
| gdelt_features | timestamp DESC | Latest feature lookup |
| macro_fred | (series_id, date DESC) | Per-series time-range queries |
| signals | (hmm_regime, timestamp DESC) | Filter signals by regime |