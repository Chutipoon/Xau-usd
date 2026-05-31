-- ============================================================
-- XAU/USD Trading System — Teardown Script
-- WARNING: Destroys ALL data. Dev/reset use only.
-- Run: psql $DATABASE_URL -f db/teardown.sql
-- ============================================================

-- Drop views first (depend on tables)
DROP VIEW IF EXISTS v_data_freshness CASCADE;
DROP VIEW IF EXISTS v_latest_gdelt    CASCADE;

-- Drop tables (hypertables auto-dropped with CASCADE)
DROP TABLE IF EXISTS gdelt_features CASCADE;
DROP TABLE IF EXISTS ohlcv_xauusd   CASCADE;
DROP TABLE IF EXISTS macro_fred     CASCADE;
DROP TABLE IF EXISTS cot_xauusd     CASCADE;