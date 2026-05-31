-- ============================================================
-- XAU/USD Systematic Trading — TimescaleDB Schema v1.0
-- Idempotent (safe to re-run). Requires: TimescaleDB 2.x
-- ============================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================
-- TABLE 1: ohlcv_xauusd — Price bars (Hypertable, 1mo chunks)
-- ============================================================
CREATE TABLE IF NOT EXISTS ohlcv_xauusd (
    ts           TIMESTAMPTZ      NOT NULL,
    open_price   DOUBLE PRECISION NOT NULL,
    high_price   DOUBLE PRECISION NOT NULL,
    low_price    DOUBLE PRECISION NOT NULL,
    close_price  DOUBLE PRECISION NOT NULL,
    volume       DOUBLE PRECISION NOT NULL DEFAULT 0,
    source       VARCHAR(20)      NOT NULL DEFAULT 'dukascopy',
    CONSTRAINT chk_ohlcv_prices_positive
        CHECK (open_price > 0 AND high_price > 0 AND low_price > 0 AND close_price > 0),
    CONSTRAINT chk_ohlcv_high_gte_low
        CHECK (high_price >= low_price),
    CONSTRAINT chk_ohlcv_price_range
        CHECK (close_price BETWEEN 500 AND 5000),
    CONSTRAINT chk_ohlcv_source
        CHECK (source IN ('dukascopy', 'yahoo', 'manual'))
);

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'ohlcv_xauusd'
    ) THEN
        PERFORM create_hypertable(
            'ohlcv_xauusd', 'ts',
            chunk_time_interval => INTERVAL '1 month'
        );
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_ohlcv_ts_desc ON ohlcv_xauusd (ts DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_source ON ohlcv_xauusd (source, ts DESC);

-- ============================================================
-- TABLE 2: cot_xauusd — CFTC COT weekly (COMEX Gold 088691)
-- ============================================================
CREATE TABLE IF NOT EXISTS cot_xauusd (
    week_date   DATE   NOT NULL,
    net_long    BIGINT NOT NULL DEFAULT 0,
    net_short   BIGINT NOT NULL DEFAULT 0,
    noncomm_net BIGINT NOT NULL DEFAULT 0,
    comm_net    BIGINT NOT NULL DEFAULT 0,
    CONSTRAINT chk_cot_date_not_future CHECK (week_date <= CURRENT_DATE),
    CONSTRAINT chk_cot_net_range CHECK (net_long BETWEEN -500000 AND 500000),
    PRIMARY KEY (week_date)
);

CREATE INDEX IF NOT EXISTS idx_cot_week_date ON cot_xauusd (week_date DESC);

-- ============================================================
-- TABLE 3: gdelt_features — GDELT sentiment features (Hypertable)
-- Updated every 15 min via gdelt-doc-api
-- ============================================================
CREATE TABLE IF NOT EXISTS gdelt_features (
    ts                     TIMESTAMPTZ      NOT NULL,
    tone_7d_avg            DOUBLE PRECISION,
    tone_30d_avg           DOUBLE PRECISION,
    event_spike_zscore     DOUBLE PRECISION,
    tone_price_divergence  DOUBLE PRECISION,
    article_count          INTEGER          DEFAULT 0,
    CONSTRAINT chk_gdelt_zscore CHECK (event_spike_zscore BETWEEN -10 AND 10)
);

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'gdelt_features'
    ) THEN
        PERFORM create_hypertable(
            'gdelt_features', 'ts',
            chunk_time_interval => INTERVAL '1 week'
        );
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_gdelt_ts_desc ON gdelt_features (ts DESC);

-- ============================================================
-- TABLE 4: macro_fred — FRED series, long format
-- PK: (obs_date, series_id) — one row per date+series
-- ============================================================
CREATE TABLE IF NOT EXISTS macro_fred (
    obs_date   DATE             NOT NULL,
    series_id  VARCHAR(20)      NOT NULL,
    obs_value  DOUBLE PRECISION,
    updated_at TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (obs_date, series_id)
);

CREATE INDEX IF NOT EXISTS idx_macro_series_date
    ON macro_fred (series_id, obs_date DESC);

-- ============================================================
-- ============================================================
-- VIEWS
-- ============================================================

CREATE OR REPLACE VIEW v_latest_gdelt AS
SELECT
    ts,
    tone_7d_avg,
    tone_30d_avg,
    event_spike_zscore,
    tone_price_divergence,
    article_count
FROM gdelt_features
ORDER BY ts DESC
LIMIT 1;

-- Data freshness: used by Grafana + watchdog
CREATE OR REPLACE VIEW v_data_freshness AS
SELECT 'ohlcv' AS feed, MAX(ts) AS latest_ts FROM ohlcv_xauusd
UNION ALL
SELECT 'gdelt' AS feed, MAX(ts) AS latest_ts FROM gdelt_features
UNION ALL
SELECT 'cot' AS feed, MAX(week_date::TIMESTAMPTZ) AS latest_ts FROM cot_xauusd;