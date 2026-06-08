import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from src.data.feature_engineering import assemble_feature_matrix

def test_temporal_alignment():
    mock_conn = MagicMock()

    # 1. OHLCV data: Hourly from Tuesday May 7 to Saturday May 11, 2024
    dates = pd.date_range(start="2024-05-07 00:00", end="2024-05-11 23:00", freq="h", tz=timezone.utc)
    ohlcv_df = pd.DataFrame({
        'ts': dates,
        'open_price': 2300.0,
        'high_price': 2310.0,
        'low_price': 2290.0,
        'close_price': 2305.0,
        'volume': 1000
    })

    # 2. GDELT data: Minimal columns to avoid errors
    gdelt_df = pd.DataFrame({
        'ts': [dates[0]],
        'tone_7d_avg': [0.5],
        'tone_30d_avg': [0.4],
        'event_spike_zscore': [1.2],
        'tone_price_divergence': [0.1],
        'article_count': [10]
    })

    # 3. COT data: Tuesday May 7
    cot_df = pd.DataFrame({
        'week_date': [datetime(2024, 5, 7, tzinfo=timezone.utc)],
        'net_long': [150000]
    })

    # 4. FRED data: obs_date May 7
    fred_df = pd.DataFrame({
        'obs_date': [
            datetime(2024, 5, 7, tzinfo=timezone.utc),
            datetime(2024, 5, 7, tzinfo=timezone.utc),
            datetime(2024, 5, 7, tzinfo=timezone.utc)
        ],
        'series_id': ['DFII10', 'DFII2', 'DTWEXBGS'],
        'obs_value': [2.0, 1.5, 105.0]
    })

    def mock_read_sql(sql, conn):
        if "ohlcv_xauusd" in sql:
            return ohlcv_df
        if "gdelt_features" in sql:
            return gdelt_df
        if "cot_xauusd" in sql:
            return cot_df
        if "macro_fred" in sql:
            return fred_df
        return pd.DataFrame()

    with patch('pandas.read_sql', side_effect=mock_read_sql):
        # Patch dropna to keep NaNs for verification
        with patch('pandas.DataFrame.dropna', autospec=True) as mock_dropna:
            mock_dropna.side_effect = lambda self, *args, **kwargs: self
            df = assemble_feature_matrix(mock_conn)

    # Ensure index is datetime and UTC
    assert isinstance(df.index, pd.DatetimeIndex)

    # Criteria 1: Wednesday 00:00 UTC, cot_net_long must be NaN (not yet published)
    wed_00 = pd.Timestamp("2024-05-08 00:00", tz=timezone.utc)
    assert np.isnan(df.loc[wed_00, 'cot_net_long']), f"COT should be NaN on Wednesday {wed_00}, but got {df.loc[wed_00, 'cot_net_long']}"

    # Criteria 2: On the following Friday 21:00 UTC, cot_net_long must be populated
    fri_21 = pd.Timestamp("2024-05-10 21:00", tz=timezone.utc)
    assert df.loc[fri_21, 'cot_net_long'] == 150000, f"COT should be 150000 on Friday 21:00, but got {df.loc[fri_21, 'cot_net_long']}"

    # Criteria 3: On obs_date 01:00 UTC, yield_spread must be NaN (not yet published)
    # obs_date is 2024-05-07
    tue_01 = pd.Timestamp("2024-05-07 01:00", tz=timezone.utc)
    assert np.isnan(df.loc[tue_01, 'yield_spread']), f"Yield spread should be NaN on Tuesday 01:00, but got {df.loc[tue_01, 'yield_spread']}"

    # Criteria 4: On obs_date+1 10:00 UTC, yield_spread must be populated
    wed_10 = pd.Timestamp("2024-05-08 10:00", tz=timezone.utc)
    assert df.loc[wed_10, 'yield_spread'] == 0.5, f"Yield spread should be 0.5 on Wednesday 10:00, but got {df.loc[wed_10, 'yield_spread']}"

if __name__ == "__main__":
    pytest.main([__file__])
