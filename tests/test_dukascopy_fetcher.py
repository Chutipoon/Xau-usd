import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pandas as pd
import struct
import lzma
from src.data.dukascopy_fetcher import fetch_ohlcv, fetch_and_store

def test_fetch_ohlcv_structure_and_dtypes():
    # Prepare mock bi5 data
    # format: !i5f (int32, 5x float32)
    # One candle at 0 offset, with prices around 2000
    mock_record = struct.pack('>i5f', 0, 2000.0, 2010.0, 1990.0, 2005.0, 100.0)
    compressed_data = lzma.compress(mock_record)

    with patch('src.data.dukascopy_fetcher.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = compressed_data
        mock_get.return_value = mock_response

        start_date = datetime(2024, 1, 1, 0, 0)
        end_date = datetime(2024, 1, 1, 1, 0)
        df = fetch_ohlcv('XAUUSD', start_date, end_date, '1h')

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert df['timestamp'].dt.tz is not None # UTC aware
        assert df['close'].dtype == 'float32' or df['close'].dtype == 'float64'

def test_fetch_ohlcv_date_range():
    # Mock data for two days
    mock_record = struct.pack('>i5f', 0, 2000.0, 2010.0, 1990.0, 2005.0, 100.0)
    compressed_data = lzma.compress(mock_record)

    with patch('src.data.dukascopy_fetcher.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = compressed_data
        mock_get.return_value = mock_response

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        df = fetch_ohlcv('XAUUSD', start_date, end_date, '1d')

        # We expect 2 requests (one per day)
        assert mock_get.call_count == 2
        assert len(df) == 2

def test_fetch_ohlcv_validation():
    # Mock data with invalid close price
    mock_record = struct.pack('>i5f', 0, 2000.0, 2010.0, 1990.0, 100.0, 100.0) # close=100 is invalid
    compressed_data = lzma.compress(mock_record)

    with patch('src.data.dukascopy_fetcher.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = compressed_data
        mock_get.return_value = mock_response

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 1)
        df = fetch_ohlcv('XAUUSD', start_date, end_date, '1h')

        assert df.empty

def test_fetch_and_store_calls_db():
    mock_record = struct.pack('>i5f', 0, 2000.0, 2010.0, 1990.0, 2005.0, 100.0)
    compressed_data = lzma.compress(mock_record)

    with patch('src.data.dukascopy_fetcher.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = compressed_data
        mock_get.return_value = mock_response

        mock_db_conn = MagicMock()
        mock_cursor = mock_db_conn.cursor.return_value.__enter__.return_value

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 1)
        fetch_and_store('XAUUSD', start_date, end_date, '1h', mock_db_conn)

        assert mock_cursor.execute.called
        # Check if INSERT statement was called
        args, _ = mock_cursor.execute.call_args
        assert "INSERT INTO ohlcv_xauusd" in args[0]
        assert mock_db_conn.commit.called
