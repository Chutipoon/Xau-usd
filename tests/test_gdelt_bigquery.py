import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.data.gdelt_bigquery import fetch_gdelt_bigquery, compute_features_from_aggregated, fetch_and_store_gdelt_historical

class TestGdeltBigQuery(unittest.TestCase):

    @patch('src.data.gdelt_bigquery.bigquery.Client')
    def test_fetch_gdelt_bigquery_success(self, mock_client):
        # Setup mock
        mock_instance = mock_client.return_value

        # Mock dry run
        mock_dry_run_job = MagicMock()
        mock_dry_run_job.total_bytes_processed = 100 * 1024**3 # 100 GB
        mock_instance.query.side_effect = [mock_dry_run_job, MagicMock()]

        # Mock actual query result
        mock_query_job = MagicMock()
        mock_instance.query.side_effect = [mock_dry_run_job, mock_query_job]
        mock_df = pd.DataFrame({
            'ts': [datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc), datetime(2023, 1, 1, 1, 0, tzinfo=timezone.utc)],
            'tone': [0.5, -0.2],
            'article_count': [10, 20]
        })
        mock_query_job.to_dataframe.return_value = mock_df

        df = fetch_gdelt_bigquery('2023-01-01', '2023-01-01')

        self.assertFalse(df.empty)
        self.assertEqual(len(df), 2)
        self.assertIn('ts', df.columns)
        self.assertIn('tone', df.columns)
        self.assertIn('article_count', df.columns)

    @patch('src.data.gdelt_bigquery.bigquery.Client')
    def test_fetch_gdelt_bigquery_quota_exceeded(self, mock_client):
        # Setup mock
        mock_instance = mock_client.return_value

        # Mock dry run exceeding quota
        mock_dry_run_job = MagicMock()
        mock_dry_run_job.total_bytes_processed = 900 * 1024**3 # 900 GB
        mock_instance.query.return_value = mock_dry_run_job

        df = fetch_gdelt_bigquery('2023-01-01', '2023-01-01', max_gb_per_month=800)

        self.assertTrue(df.empty)

    def test_compute_features_from_aggregated(self):
        # Create dummy aggregated data
        ts = pd.date_range(start='2023-01-01', periods=1000, freq='1h', tz='UTC')
        df = pd.DataFrame({
            'ts': ts,
            'tone': np.random.randn(1000),
            'article_count': np.random.randint(0, 50, 1000)
        })

        # Dummy price series
        price_series = pd.Series(np.linspace(1800, 1900, 1000), index=ts)

        features = compute_features_from_aggregated(df, price_series)

        self.assertEqual(len(features), 1000)
        self.assertIn('tone_7d_avg', features.columns)
        self.assertIn('tone_30d_avg', features.columns)
        self.assertIn('event_spike_zscore', features.columns)
        self.assertIn('tone_price_divergence', features.columns)
        self.assertIn('article_count', features.columns)

        # Check rolling calculations (first few might be NaN or partial depending on min_periods)
        self.assertFalse(features['tone_7d_avg'].isna().all())
        self.assertFalse(features['event_spike_zscore'].isna().all())

    @patch('src.data.gdelt_bigquery.fetch_gdelt_bigquery')
    @patch('psycopg2.connect')
    def test_fetch_and_store_gdelt_historical(self, mock_connect, mock_fetch_bq):
        # Mock BigQuery fetch
        ts = pd.date_range(start='2023-01-01', periods=24, freq='1h', tz='UTC')
        mock_fetch_bq.return_value = pd.DataFrame({
            'ts': ts,
            'tone': [0.1] * 24,
            'article_count': [5] * 24
        })

        # Mock DB connection and cursor
        mock_conn = MagicMock()
        mock_cur = mock_conn.cursor.return_value.__enter__.return_value

        # Mock price data fetch from DB
        mock_cur.fetchall.return_value = [(t, 1800.0) for t in ts]

        fetch_and_store_gdelt_historical('2023-01-01', '2023-01-01', mock_conn)

        # Verify that execute was called (for inserting features)
        self.assertTrue(mock_cur.execute.called)
        # Verify commit was called
        self.assertTrue(mock_conn.commit.called)

if __name__ == '__main__':
    unittest.main()
