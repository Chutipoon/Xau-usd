import lzma
import struct
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_ohlcv(symbol, start_date, end_date, timeframe):
    """
    Fetches OHLCV data from Dukascopy.

    symbol: 'XAUUSD'
    start_date, end_date: datetime objects
    timeframe: '1m', '1h', '1d'
    """
    if symbol != 'XAUUSD':
        raise ValueError("Only XAUUSD is supported for now.")

    # Dukascopy symbols for URL
    # XAUUSD -> XAUUSD

    # Timeframe mapping for URL
    tf_map = {
        '1m': '1m',
        '1h': '1h',
        '1d': '1d'
    }

    if timeframe not in tf_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    tf_url = tf_map[timeframe]

    all_data = []

    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date_limit = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

    while current_date <= end_date_limit:
        url = f"https://datafeed.dukascopy.com/datafeed/{symbol}/{current_date.year}/{current_date.month-1:02d}/{current_date.day:02d}/{tf_url}_candles.bi5"

        data = _fetch_with_retry(url)
        if data:
            try:
                decompressed = lzma.decompress(data)
                parsed = _parse_bi5(decompressed, current_date, timeframe)
                all_data.extend(parsed)
            except Exception as e:
                logger.error(f"Error parsing data for {current_date}: {e}")

        current_date += timedelta(days=1)

    if not all_data:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

    # Validation
    df = df[(df['close'] >= 500) & (df['close'] <= 5000)]

    # Filter by date range
    df = df[(df['timestamp'] >= pd.Timestamp(start_date, tz='UTC')) &
            (df['timestamp'] <= pd.Timestamp(end_date, tz='UTC'))]

    return df.reset_index(drop=True)

def _fetch_with_retry(url, retries=3, backoff=2):
    for i in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.content
            elif response.status_code == 404:
                logger.warning(f"Data not found: {url}")
                return None
            elif 500 <= response.status_code < 600:
                logger.error(f"Server error {response.status_code} for {url}. Retrying...")
            else:
                logger.error(f"HTTP error {response.status_code} for {url}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}. Retrying...")

        time.sleep(backoff ** (i + 1))
    return None

def _parse_bi5(data, date, timeframe):
    """
    Parsed .bi5 binary data.
    Structure: 4-byte int (seconds from start of day/period) + 4x uint32 + 1 float
    Dukascopy encodes prices as integers (price * point_multiplier).
    XAUUSD point = 0.001 (3 decimal places).
    """
    struct_fmt = '>i4If' # timestamp(i32), O/H/L/C(4xu32), vol(f32)
    struct_size = struct.calcsize(struct_fmt)
    records = []

    base_ts = int(date.timestamp())

    for i in range(0, len(data), struct_size):
        if i + struct_size > len(data):
            break

        try:
            time_offset, op, hi, lo, cl, vol = struct.unpack(struct_fmt, data[i:i+struct_size])
            # XAUUSD point = 0.001
            records.append({
                'timestamp': base_ts + time_offset,
                'open': op / 1000.0,
                'high': hi / 1000.0,
                'low': lo / 1000.0,
                'close': cl / 1000.0,
                'volume': vol
            })
        except Exception:
            continue

    return records

def fetch_and_store(symbol, start_date, end_date, timeframe, db_conn):
    """
    Calls fetch_ohlcv() and upserts into ohlcv_xauusd table.
    """
    df = fetch_ohlcv(symbol, start_date, end_date, timeframe)

    if df.empty:
        logger.warning(f"No data fetched for {symbol} from {start_date} to {end_date}")
        return

    with db_conn.cursor() as cur:
        # Progress logging per month
        last_logged_month = None

        for _, row in df.iterrows():
            current_month = row['timestamp'].month
            if current_month != last_logged_month:
                logger.info(f"Storing data for {row['timestamp'].strftime('%Y-%m')}")
                last_logged_month = current_month
                db_conn.commit() # Commit monthly

            cur.execute("""
                INSERT INTO ohlcv_xauusd (ts, open_price, high_price, low_price, close_price, volume, source)
                VALUES (%s, %s, %s, %s, %s, %s, 'dukascopy')
                ON CONFLICT (ts) DO NOTHING
            """, (
                row['timestamp'],
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume']
            ))

        db_conn.commit()

    logger.info(f"Successfully stored {len(df)} rows for {symbol}")
