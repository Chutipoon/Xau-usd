import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates RSI, MACD, Bollinger Bands, ATR, and Volume Z-score.
    Input df must have: open_price, high_price, low_price, close_price, volume.
    Index should be datetime.
    """
    df = df.copy().sort_index()

    # Returns
    df['returns_1h'] = df['close_price'].pct_change()
    df['returns_4h'] = df['close_price'].pct_change(4)
    df['returns_24h'] = df['close_price'].pct_change(24)

    # Realized Vol (7d = 168h)
    df['realized_vol_7d'] = df['returns_1h'].rolling(window=168).std() * np.sqrt(168)

    # RSI (14h)
    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close_price'].ewm(span=12, adjust=False).mean()
    ema26 = df['close_price'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20h)
    df['bb_mid'] = df['close_price'].rolling(window=20).mean()
    df['bb_std'] = df['close_price'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
    df['bb_position'] = (df['close_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 1e-9)

    # ATR (14h)
    high_low = df['high_price'] - df['low_price']
    high_close = (df['high_price'] - df['close_price'].shift()).abs()
    low_close = (df['low_price'] - df['close_price'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr_14'] = true_range.rolling(window=14).mean()

    # Volume Z-score (24h)
    vol_mean = df['volume'].rolling(window=24).mean()
    vol_std = df['volume'].rolling(window=24).std().replace(0, 1)
    df['volume_zscore'] = (df['volume'] - vol_mean) / vol_std

    return df

def assemble_feature_matrix(db_conn) -> pd.DataFrame:
    """
    Joins OHLCV, GDELT, COT, and FRED data.
    """
    # 1. Fetch OHLCV
    ohlcv = pd.read_sql("SELECT ts, open_price, high_price, low_price, close_price, volume FROM ohlcv_xauusd ORDER BY ts", db_conn)
    ohlcv['ts'] = pd.to_datetime(ohlcv['ts'], utc=True)
    ohlcv = ohlcv.set_index('ts')

    # Calculate tech indicators
    features = calculate_technical_indicators(ohlcv)

    # 2. Fetch GDELT
    gdelt = pd.read_sql("SELECT ts, tone_7d_avg, tone_30d_avg, event_spike_zscore, tone_price_divergence, article_count FROM gdelt_features ORDER BY ts", db_conn)
    gdelt['ts'] = pd.to_datetime(gdelt['ts'], utc=True)
    gdelt = gdelt.set_index('ts')

    # GDELT specific LSTM features
    gdelt['article_count_zscore'] = (gdelt['article_count'] - gdelt['article_count'].rolling(window=720).mean()) / gdelt['article_count'].rolling(window=720).std().replace(0, 1)
    gdelt['tone_momentum'] = gdelt['tone_7d_avg'].diff(24) # 24h change

    # Join GDELT
    features = features.join(gdelt, how='left')

    # Fill NaNs in GDELT columns with 0 before dropna() to prevent losing all data if GDELT is sparse
    GDELT_COLS = ['tone_7d_avg', 'tone_30d_avg', 'event_spike_zscore', 'tone_price_divergence', 'article_count_zscore', 'tone_momentum']
    features[GDELT_COLS] = features[GDELT_COLS].fillna(0)

    # 3. Fetch COT
    cot = pd.read_sql("SELECT week_date, net_long FROM cot_xauusd ORDER BY week_date", db_conn)
    cot['week_date'] = pd.to_datetime(cot['week_date'], utc=True)
    cot = cot.set_index('week_date')

    # Reindex COT to hourly (using method='pad' to handle alignment)
    cot_hourly = cot.reindex(features.index, method='pad')
    features['cot_net_long'] = cot_hourly['net_long']

    # 4. Fetch FRED
    fred = pd.read_sql("SELECT obs_date, series_id, obs_value FROM macro_fred", db_conn)
    fred['obs_date'] = pd.to_datetime(fred['obs_date'], utc=True)

    # Pivot FRED
    fred_pivot = fred.pivot(index='obs_date', columns='series_id', values='obs_value')

    # Yield Spread (Real 10Y - Real 2Y)
    if 'DFII10' in fred_pivot.columns and 'DFII2' in fred_pivot.columns:
        fred_pivot['yield_spread'] = fred_pivot['DFII10'] - fred_pivot['DFII2']
    elif 'DFII10' in fred_pivot.columns:
        fred_pivot['yield_spread'] = fred_pivot['DFII10'] # Fallback
    else:
        fred_pivot['yield_spread'] = 0

    # DXY
    if 'DTWEXBGS' in fred_pivot.columns:
        fred_pivot['dxy_return'] = fred_pivot['DTWEXBGS'].pct_change()
    else:
        fred_pivot['dxy_return'] = 0

    # Reindex FRED to hourly
    fred_hourly = fred_pivot[['yield_spread', 'dxy_return']].reindex(features.index, method='pad')
    features = features.join(fred_hourly, how='left')

    # HMM Specific features
    features['returns'] = features['returns_1h']
    features['log_volume'] = np.log(features['volume'].replace(0, 1))
    features['realized_vol'] = features['realized_vol_7d'] # Using 7d as proxy

    # Final clean up
    return features.ffill().dropna()
