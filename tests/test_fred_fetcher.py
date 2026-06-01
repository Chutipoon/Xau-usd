import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime
from src.data.fred_fetcher import fetch_fred_series, fetch_and_store_fred

# Helper to mock OpenBB OBB object
class MockOBB:
    def __init__(self, df):
        self._df = df
    def to_df(self):
        return self._df

def test_fetch_fred_series_basic():
    mock_df = pd.DataFrame({'value': [100.0, 101.0]}, index=pd.to_datetime(['2024-01-01', '2024-01-02']))

    # We need to mock the entire path to avoid OpenBB initialization or real calls
    with patch('src.data.fred_fetcher.obb') as mock_obb_module:
        mock_obb_module.economy.fred_series.return_value = MockOBB(mock_df)

        df = fetch_fred_series(['real_yield_10y'], '2024-01-01', '2024-01-02')

        assert not df.empty
        assert 'real_yield_10y' in df.columns
        assert len(df) == 2

def test_fetch_fred_series_forward_fill():
    # One value missing
    mock_df = pd.DataFrame({'value': [100.0, None, 102.0]},
                          index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']))

    with patch('src.data.fred_fetcher.obb') as mock_obb_module:
        mock_obb_module.economy.fred_series.return_value = MockOBB(mock_df)

        df = fetch_fred_series(['real_yield_10y'], '2024-01-01', '2024-01-03')

        assert df.loc['2024-01-02', 'real_yield_10y'] == 100.0 # Forward-filled

def test_fetch_fred_series_yfinance_fallback():
    # Simulate OpenBB failure due to missing credentials
    with patch('src.data.fred_fetcher.obb') as mock_obb_module:
        mock_obb_module.economy.fred_series.side_effect = Exception("missing credentials")

        with patch('src.data.fred_fetcher.yf.download') as mock_yf:
            mock_yf_df = pd.DataFrame({'Close': [2000.0]}, index=pd.to_datetime(['2024-01-01']))
            mock_yf.return_value = mock_yf_df

            df = fetch_fred_series(['gold_etf_gld'], '2024-01-01', '2024-01-01')

            assert not df.empty
            assert 'gold_etf_gld' in df.columns
            assert df.iloc[0, 0] == 2000.0

def test_fetch_fred_series_invalid_key():
    with pytest.raises(ValueError):
        fetch_fred_series(['invalid_key'], '2024-01-01', '2024-01-01')

def test_fetch_and_store_fred_calls_db():
    mock_df = pd.DataFrame({'real_yield_10y': [1.5]}, index=pd.to_datetime(['2024-01-01']))

    with patch('src.data.fred_fetcher.fetch_fred_series', return_value=mock_df):
        mock_db_conn = MagicMock()
        mock_cursor = mock_db_conn.cursor.return_value.__enter__.return_value

        fetch_and_store_fred(['real_yield_10y'], '2024-01-01', '2024-01-01', mock_db_conn)

        assert mock_cursor.execute.called
        # Verify the SQL matches our implementation
        args, _ = mock_cursor.execute.call_args
        assert "INSERT INTO macro_fred" in args[0]
        assert mock_db_conn.commit.called
