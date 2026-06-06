import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import date, timedelta
from src.data.cot_fetcher import fetch_cot_gold, validate_cot, fetch_and_store_cot

def test_fetch_cot_gold_structure():
    # Mock COT response
    mock_df = pd.DataFrame({
        'CFTC_Commodity_Code': ['088691', '088691'],
        'Report_Date_as_YYYY_MM_DD': ['2024-01-02', '2024-01-09'],
        'NonCommercial_Positions_Long_All': [100000, 110000],
        'NonCommercial_Positions_Short_All': [20000, 25000],
        'Commercial_Positions_Long_All': [50000, 45000],
        'Commercial_Positions_Short_All': [130000, 140000]
    })

    with patch('src.data.cot_fetcher.cot_download_year') as mock_download:
        mock_download.return_value = mock_df

        df = fetch_cot_gold(2024, 2024)

        assert not df.empty
        assert len(df) == 2
        assert list(df.columns) == ['week_date', 'net_long', 'net_short', 'noncomm_net', 'comm_net']
        assert df.iloc[0]['net_long'] == 80000 # 100000 - 20000
        assert df.iloc[0]['net_short'] == 80000 # 130000 - 50000

def test_validate_cot_valid():
    df = pd.DataFrame({
        'week_date': [date(2024, 1, 2), date(2024, 1, 9)],
        'net_long': [100000, 100000],
        'net_short': [50000, 50000],
        'noncomm_net': [100000, 100000],
        'comm_net': [-50000, -50000]
    })
    assert validate_cot(df) is True

def test_validate_cot_gaps():
    df = pd.DataFrame({
        'week_date': [date(2024, 1, 2), date(2024, 1, 20)], # 18 day gap
        'net_long': [100000, 100000],
        'net_short': [50000, 50000],
        'noncomm_net': [100000, 100000],
        'comm_net': [-50000, -50000]
    })
    # Gaps > 14 days should return False per blueprint
    assert validate_cot(df) is False

def test_validate_cot_range():
    df = pd.DataFrame({
        'week_date': [date(2024, 1, 2)],
        'net_long': [400000], # Outside [-300000, 300000]
        'net_short': [50000],
        'noncomm_net': [400000],
        'comm_net': [-50000]
    })
    assert validate_cot(df) is False

def test_fetch_and_store_cot_calls_db():
    mock_df = pd.DataFrame({
        'week_date': [date(2024, 1, 2)],
        'net_long': [100000],
        'net_short': [50000],
        'noncomm_net': [100000],
        'comm_net': [-50000]
    })

    with patch('src.data.cot_fetcher.fetch_cot_gold', return_value=mock_df):
        mock_db_conn = MagicMock()
        mock_cursor = mock_db_conn.cursor.return_value.__enter__.return_value
        mock_cursor.fetchone.return_value = [True] # inserted=True

        fetch_and_store_cot(2024, 2024, mock_db_conn)

        assert mock_cursor.execute.called
        assert "INSERT INTO cot_xauusd" in mock_cursor.execute.call_args[0][0]
        assert mock_db_conn.commit.called
