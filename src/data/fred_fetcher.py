import pandas as pd
try:
    from openbb import obb
except ImportError:
    obb = None
import yfinance as yf
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

FRED_SERIES = {
    'real_yield_10y': 'DFII10',
    'real_yield_2y':  'DFII2',
    'breakeven_10y':  'T10YIE',
    'dxy_index':      'DTWEXBGS',
    'vix':            'VIXCLS',
    'fed_funds_rate': 'FEDFUNDS',
    'cpi_yoy':        'CPIAUCSL',
    'gold_etf_gld':   'GLD',
    'spx':            'SP500',
    'us_10y_yield':   'DGS10',
}

def fetch_fred_series(series_keys: list[str], start_date, end_date) -> pd.DataFrame:
    """
    Fetches FRED macro data using OpenBB SDK with fallback to yfinance.
    """
    valid_keys = []
    for key in series_keys:
        if key not in FRED_SERIES:
            raise ValueError(f"Series key '{key}' not in FRED_SERIES dictionary.")
        valid_keys.append(key)

    all_data = {}

    for key in valid_keys:
        fred_id = FRED_SERIES[key]
        data = None

        # Try OpenBB
        if obb is not None:
            try:
                res = obb.economy.fred_series(symbol=fred_id, start_date=start_date, end_date=end_date)
                # The mock in tests should return an object with to_df()
                if hasattr(res, 'to_df'):
                    data = res.to_df()
                elif isinstance(res, pd.DataFrame):
                    data = res

                if data is not None and not data.empty:
                    data = data.iloc[:, 0]
                else:
                    data = None
            except Exception as e:
                logger.warning(f"OpenBB failed for {key} ({fred_id}): {e}")

        # Fallback to yfinance for GLD/SP500 or if OpenBB failed
        if (data is None or data.empty) and key in ['gold_etf_gld', 'spx']:
            yf_map = {'gold_etf_gld': 'GLD', 'spx': '^GSPC'}
            try:
                ticker = yf_map[key]
                yf_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not yf_data.empty:
                    data = yf_data['Close']
            except Exception as e:
                logger.error(f"yfinance fallback failed for {key}: {e}")

        if data is not None:
            all_data[key] = data

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Forward-fill missing values (max 5 business days)
    df = df.ffill(limit=5)

    return df

def fetch_and_store_fred(series_keys, start_date, end_date, db_conn):
    """
    Calls fetch_fred_series() and upserts into macro_fred table.
    """
    df = fetch_fred_series(series_keys, start_date, end_date)

    if df.empty:
        logger.warning(f"No macro data fetched for {series_keys}")
        return

    # Transform wide to long format
    df_long = df.reset_index().melt(id_vars='index', var_name='key', value_name='obs_value')
    df_long.rename(columns={'index': 'obs_date'}, inplace=True)

    with db_conn.cursor() as cur:
        for _, row in df_long.iterrows():
            series_id = FRED_SERIES[row['key']]
            if pd.isna(row['obs_value']):
                continue

            cur.execute("""
                INSERT INTO macro_fred (obs_date, series_id, obs_value, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (obs_date, series_id)
                DO UPDATE SET obs_value = EXCLUDED.obs_value, updated_at = NOW()
            """, (row['obs_date'], series_id, row['obs_value']))

        db_conn.commit()

    logger.info(f"Successfully stored FRED macro data for {series_keys}")
