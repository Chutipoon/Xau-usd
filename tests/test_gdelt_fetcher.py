import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime, timedelta
from src.data.gdelt_fetcher import fetch_gdelt_docs, compute_gdelt_features, fetch_and_store_gdelt

def test_fetch_gdelt_docs_structure():
    mock_articles = pd.DataFrame({
        'seendate': ['20240101010000', '20240101020000'],
        'title': ['Gold Rises', 'Inflation Data'],
        'url': ['http://url1', 'http://url2'],
        'tone': [2.5, -1.0],
        'domain': ['site1.com', 'site2.com']
    })

    with patch('src.data.gdelt_fetcher.GdeltDoc') as mock_gd_class:
        mock_instance = mock_gd_class.return_value
        mock_instance.article_search.return_value = mock_articles

        df = fetch_gdelt_docs(hours_back=2)

        assert not df.empty
        assert 'datetime' in df.columns
        assert list(df.columns) == ['datetime', 'title', 'url', 'tone', 'domain']

def test_compute_gdelt_features():
    # Create sample news data for 2 days
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(48)]
    df = pd.DataFrame({
        'datetime': dates,
        'tone': [1.0] * 48,
        'url': [f'url{i}' for i in range(48)]
    })

    # price series for 48 hours
    price_series = pd.Series([2000.0] * 48, index=pd.to_datetime(dates, utc=True))

    features = compute_gdelt_features(df, price_series)

    assert not features.empty
    assert 'tone_7d_avg' in features.columns
    assert 'event_spike_zscore' in features.columns
    assert features.iloc[-1]['tone_7d_avg'] == 1.0
    assert features.iloc[-1]['article_count'] == 1

def test_fetch_gdelt_docs_timeout_fallback():
    with patch('src.data.gdelt_fetcher.GdeltDoc') as mock_gd_class:
        mock_instance = mock_gd_class.return_value
        # Simulate timeout/error
        mock_instance.article_search.side_effect = Exception("Timeout after 30s")

        df = fetch_gdelt_docs(hours_back=24)
        assert df.empty

def test_fetch_and_store_gdelt_fallback():
    # If fetch returns empty, store should return without error and not call DB
    with patch('src.data.gdelt_fetcher.fetch_gdelt_docs', return_value=pd.DataFrame()):
        mock_db_conn = MagicMock()

        fetch_and_store_gdelt(24, pd.Series(), mock_db_conn)

        assert not mock_db_conn.cursor.called
