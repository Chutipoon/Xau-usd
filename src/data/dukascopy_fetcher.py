import lzma
import struct
import requests
import pandas as pd
import psycopg2.extras
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_ohlcv(symbol: str, start_date: datetime, end_date: datetime, timeframe: str) -> pd.DataFrame:
    """
    Fetches OHLCV data from Dukascopy public HTTP endpoint.

    Args:
        symbol:     'XAUUSD'
        start_date: datetime (UTC)
        end_date:   datetime (UTC)
        timeframe:  '1m' | '1h' | '1d'

    Returns:
        DataFrame with columns: [timestamp, open, high, low, close, volume]
        timestamp is UTC-aware pd.Timestamp.
    """
    if symbol != "XAUUSD":
        raise ValueError("Only XAUUSD is supported.")

    tf_map = {"1m": "1m", "1h": "1h", "1d": "1d"}
    if timeframe not in tf_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Use one of {list(tf_map)}")

    tf_url = tf_map[timeframe]
    all_data = []

    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date_limit = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

    while current_date <= end_date_limit:
        # FIX: Dukascopy uses 0-indexed months (Jan=00 … Dec=11)
        month_idx = current_date.month - 1
        url = (
            f"https://datafeed.dukascopy.com/datafeed/{symbol}/"
            f"{current_date.year}/{month_idx:02d}/{current_date.day:02d}/{tf_url}_candles.bi5"
        )

        raw = _fetch_with_retry(url)
        if raw:
            try:
                decompressed = lzma.decompress(raw)
                parsed = _parse_bi5(decompressed, current_date)
                all_data.extend(parsed)
            except Exception as e:
                logger.error(f"Error parsing data for {current_date.date()}: {e}")

        current_date += timedelta(days=1)

    if not all_data:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    # Validate price range for XAUUSD
    df = df[(df["close"] >= 500) & (df["close"] <= 5000)].copy()

    # FIX: use tz_localize instead of deprecated pd.Timestamp(dt, tz=...)
    start_ts = pd.Timestamp(start_date).tz_localize("UTC")
    end_ts = pd.Timestamp(end_date).tz_localize("UTC")
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]

    return df.reset_index(drop=True)


def _fetch_with_retry(url: str, retries: int = 3, backoff: int = 2):
    """GET with exponential backoff on 5xx. Returns bytes or None."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.content
            elif response.status_code == 404:
                logger.warning(f"Data not found (404): {url}")
                return None
            elif 500 <= response.status_code < 600:
                logger.error(f"Server error {response.status_code} for {url}. Retry {attempt+1}/{retries}…")
            else:
                logger.error(f"HTTP {response.status_code} for {url}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}. Retry {attempt+1}/{retries}…")

        time.sleep(backoff ** (attempt + 1))

    return None


def _parse_bi5(data: bytes, date: datetime) -> list[dict]:
    """
    Parse Dukascopy .bi5 binary candle data.

    Struct per candle (big-endian):
        i  — seconds offset from midnight of `date`
        4I — open, high, low, close  (uint32, price * 1000 for XAUUSD)
        f  — volume (float32)
    """
    struct_fmt = ">i4If"
    struct_size = struct.calcsize(struct_fmt)  # 24 bytes
    base_ts = int(date.replace(tzinfo=None).timestamp())  # naive UTC midnight

    records = []
    for offset in range(0, len(data) - struct_size + 1, struct_size):
        try:
            time_offset, op, hi, lo, cl, vol = struct.unpack(
                struct_fmt, data[offset : offset + struct_size]
            )
            records.append(
                {
                    "timestamp": base_ts + time_offset,
                    "open":  op  / 1000.0,
                    "high":  hi  / 1000.0,
                    "low":   lo  / 1000.0,
                    "close": cl  / 1000.0,
                    "volume": vol,
                }
            )
        except struct.error:
            continue

    return records


def fetch_and_store(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str,
    db_conn,
) -> None:
    """
    Fetch OHLCV data and upsert into ohlcv_xauusd.
    Uses execute_values() for batch inserts (10-50x faster than row-by-row).
    Column names match schema.sql: timestamp, open, high, low, close, volume, source.
    """
    df = fetch_ohlcv(symbol, start_date, end_date, timeframe)

    if df.empty:
        logger.warning(f"No data fetched for {symbol} {start_date} – {end_date}")
        return

    # FIX: column names now match schema.sql exactly (ts, open_price, ...)
    # FIX: use execute_values for batch insert instead of iterrows + execute
    # FIX: commit happens AFTER execute, not before
    rows = [
        (
            row["timestamp"].to_pydatetime(),
            row["open"],
            row["high"],
            row["low"],
            row["close"],
            row["volume"],
            "dukascopy",
        )
        for _, row in df.iterrows()
    ]

    sql = """
        INSERT INTO ohlcv_xauusd (ts, open_price, high_price, low_price, close_price, volume, source)
        VALUES %s
        ON CONFLICT (ts) DO NOTHING
    """

    with db_conn.cursor() as cur:
        # Log progress per month before the batch
        months = df["timestamp"].dt.to_period("M").unique()
        for m in months:
            logger.info(f"Storing data for {m}")

        psycopg2.extras.execute_values(cur, sql, rows, page_size=1000)

    db_conn.commit()  # FIX: single commit after all rows are inserted
    logger.info(f"Stored {len(rows)} rows for {symbol}")
