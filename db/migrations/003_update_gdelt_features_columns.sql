-- Migration: Add article_count_zscore and tone_momentum to gdelt_features
-- Date: 2025-01-20

ALTER TABLE gdelt_features ADD COLUMN IF NOT EXISTS article_count_zscore DOUBLE PRECISION;
ALTER TABLE gdelt_features ADD COLUMN IF NOT EXISTS tone_momentum DOUBLE PRECISION;
