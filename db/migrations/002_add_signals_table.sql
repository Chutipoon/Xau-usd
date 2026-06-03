CREATE TABLE IF NOT EXISTS signals (
    timestamp TIMESTAMPTZ NOT NULL,
    hmm_regime INTEGER,
    hmm_posterior JSONB,
    lstm_signal DOUBLE PRECISION,
    garch_vol DOUBLE PRECISION,
    bridge_forecast DOUBLE PRECISION,
    PRIMARY KEY (timestamp)
);
