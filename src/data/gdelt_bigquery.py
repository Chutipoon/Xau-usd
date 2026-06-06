import os
import logging
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
from datetime import datetime, timedelta
import psycopg2.extras

logger = logging.getLogger(__name__)

XAU_THEMES = [
    'gold', 'XAUUSD', 'inflation', 'federal reserve', 'interest rate',
    'geopolitical', 'war', 'central bank', 'safe haven', 'dollar'
]

def fetch_gdelt_bigquery(start_date: str, end_date: str, max_gb_per_month: float = 800.0) -> pd.DataFrame:
    """
    Queries BigQuery GDELT public dataset for XAU-related events.
    Aggregates by hour.
    """
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    client = bigquery.Client(project=project_id)

    # Convert dates to YYYYMMDD for wildcard table filtering
    start_fmt = start_date.replace('-', '')
    end_fmt = end_date.replace('-', '')

    # Construct the query with hourly aggregation
    # Using REGEXP_CONTAINS with word boundaries for more robust matching
    theme_regex = "|".join([fr"\b{theme.replace(' ', '[ -]')}\b" for theme in XAU_THEMES])

    query = f"""
        SELECT
            TIMESTAMP_TRUNC(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATEADDED AS STRING)), HOUR) as ts,
            AVG(AvgTone) as tone,
            COUNT(DISTINCT SOURCEURL) as article_count
        FROM
            `bigquery-public-data.gdeltv2.events_*`
        WHERE
            _TABLE_SUFFIX BETWEEN '{start_fmt}' AND '{end_fmt}'
            AND REGEXP_CONTAINS(LOWER(SOURCEURL), r'{theme_regex.lower()}')
        GROUP BY 1
        ORDER BY 1
    """

    try:
        # Dry run to estimate quota usage
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query_job = client.query(query, job_config=job_config)
        gb_scanned = query_job.total_bytes_processed / (1024**3)

        logger.info(f"BigQuery Dry Run: {gb_scanned:.2f} GB estimated.")

        # Check against monthly quota
        if gb_scanned > max_gb_per_month:
            logger.error(f"Query exceeds monthly quota: {gb_scanned:.2f}GB > {max_gb_per_month}GB")
            return pd.DataFrame()

        if gb_scanned > 700:
            logger.warning(f"BigQuery usage: {gb_scanned:.2f}GB approaching {max_gb_per_month}GB limit this month")

        # Execute the actual query
        job_config = bigquery.QueryJobConfig(dry_run=False)
        df = client.query(query, job_config=job_config).to_dataframe()

        if df.empty:
            logger.warning("BigQuery returned no results.")
            return pd.DataFrame(columns=['ts', 'tone', 'article_count'])

        df['ts'] = pd.to_datetime(df['ts'], utc=True)
        return df

    except GoogleAPIError as e:
        logger.error(f"BigQuery error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching from BigQuery: {e}")
        return pd.DataFrame()

def compute_features_from_aggregated(df: pd.DataFrame, price_series: pd.Series) -> pd.DataFrame:
    """
    Computes features from hourly aggregated GDELT data.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.set_index('ts').sort_index()

    # Ensure we have all hours in the range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1h', tz='UTC')
    df = df.reindex(full_range).fillna({'tone': 0, 'article_count': 0})

    features = pd.DataFrame(index=df.index)
    features['article_count'] = df['article_count']

    # 1. tone_7d_avg: rolling 7-day mean (168 hours)
    features['tone_7d_avg'] = df['tone'].rolling(window=168, min_periods=168).mean()

    # 2. tone_30d_avg: rolling 30-day mean (720 hours)
    features['tone_30d_avg'] = df['tone'].rolling(window=720, min_periods=720).mean()

    # 3. event_spike_zscore: (count - mean30d) / std30d
    count_mean_30d = features['article_count'].rolling(window=720, min_periods=720).mean()
    count_std_30d = features['article_count'].rolling(window=720, min_periods=720).std().replace(0, 1)
    zscore = (features['article_count'] - count_mean_30d) / count_std_30d
    features['event_spike_zscore'] = np.clip(zscore, -15, 15)

    # 4. tone_price_divergence = tone_7d_avg * (-1 * price_returns_24h)
    price_returns_24h = price_series.pct_change(24)
    aligned_returns = price_returns_24h.reindex(features.index).ffill()
    features['tone_price_divergence'] = features['tone_7d_avg'] * (-1 * aligned_returns)

    return features

def fetch_and_store_gdelt_historical(start_date: str, end_date: str, db_conn):
    """
    Main entry point for historical backfill.
    """
    df_gdelt = fetch_gdelt_bigquery(start_date, end_date)

    if df_gdelt.empty:
        logger.error("No GDELT data fetched. Historical backfill aborted.")
        return

    # Fetch Price series
    with db_conn.cursor() as cur:
        cur.execute("SELECT ts, close_price FROM ohlcv_xauusd WHERE ts >= %s AND ts <= %s", (df_gdelt['ts'].min(), df_gdelt['ts'].max()))
        price_data = cur.fetchall()

    if not price_data:
        logger.warning("No price data found for the given range. tone_price_divergence will be NaN.")
        price_series = pd.Series()
    else:
        price_df = pd.DataFrame(price_data, columns=['ts', 'close_price'])
        price_df['ts'] = pd.to_datetime(price_df['ts'], utc=True)
        price_series = price_df.set_index('ts')['close_price'].sort_index()

    features = compute_features_from_aggregated(df_gdelt, price_series)

    # Store in DB using batch insert
    rows_to_insert = []
    for ts, row in features.iterrows():
        # Insert if any of the key features are non-NaN
        if not (row[['tone_7d_avg', 'tone_30d_avg', 'event_spike_zscore']].isna().all()):
            rows_to_insert.append((
                ts,
                row['tone_7d_avg'],
                row['tone_30d_avg'],
                row['event_spike_zscore'],
                row['tone_price_divergence'],
                int(row['article_count'])
            ))

    if rows_to_insert:
        sql = """
            INSERT INTO gdelt_features (ts, tone_7d_avg, tone_30d_avg, event_spike_zscore, tone_price_divergence, article_count)
            VALUES %s
            ON CONFLICT (ts) DO UPDATE SET
                tone_7d_avg = EXCLUDED.tone_7d_avg,
                tone_30d_avg = EXCLUDED.tone_30d_avg,
                event_spike_zscore = EXCLUDED.event_spike_zscore,
                tone_price_divergence = EXCLUDED.tone_price_divergence,
                article_count = EXCLUDED.article_count
        """
        with db_conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, rows_to_insert, page_size=1000)
            db_conn.commit()
        logger.info(f"Successfully backfilled {len(rows_to_insert)} hours of GDELT features.")
    else:
        logger.warning("No valid features to insert.")
