import os
import logging
import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
from datetime import datetime, timedelta

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
    # We use SOURCEURL to filter for keywords as a proxy for themes in the events table
    theme_filters = " OR ".join([f"LOWER(SOURCEURL) LIKE '%{theme.replace(' ', '%')}%'" for theme in XAU_THEMES])

    query = f"""
        SELECT
            TIMESTAMP_TRUNC(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATEADDED AS STRING)), HOUR) as ts,
            AVG(AvgTone) as tone,
            COUNT(DISTINCT SOURCEURL) as article_count
        FROM
            `bigquery-public-data.gdeltv2.events_*`
        WHERE
            _TABLE_SUFFIX BETWEEN '{start_fmt}' AND '{end_fmt}'
            AND ({theme_filters})
        GROUP BY 1
        ORDER BY 1
    """

    try:
        # Dry run to estimate quota usage
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query_job = client.query(query, job_config=job_config)
        gb_scanned = query_job.total_bytes_processed / (1024**3)

        logger.info(f"BigQuery Dry Run: {gb_scanned:.2f} GB estimated.")

        # Check against monthly quota (Note: actual tracking requires persistence,
        # here we just check if THIS query exceeds the monthly cap or a reasonable threshold)
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

    # Ensure we have all hours in the range to avoid rolling window issues
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1h', tz='UTC')
    df = df.reindex(full_range).fillna({'tone': 0, 'article_count': 0})

    features = pd.DataFrame(index=df.index)
    features['article_count'] = df['article_count']

    # 1. tone_7d_avg: rolling 7-day mean (168 hours)
    features['tone_7d_avg'] = df['tone'].rolling(window=168, min_periods=1).mean()

    # 2. tone_30d_avg: rolling 30-day mean (720 hours)
    features['tone_30d_avg'] = df['tone'].rolling(window=720, min_periods=1).mean()

    # 3. event_spike_zscore: (count - mean30d) / std30d
    count_mean_30d = features['article_count'].rolling(window=720, min_periods=1).mean()
    count_std_30d = features['article_count'].rolling(window=720, min_periods=1).std().replace(0, 1)
    features['event_spike_zscore'] = (features['article_count'] - count_mean_30d) / count_std_30d

    # 4. tone_price_divergence = tone_7d_avg * (-1 * price_returns_24h)
    price_returns_24h = price_series.pct_change(24)
    aligned_returns = price_returns_24h.reindex(features.index).ffill()
    features['tone_price_divergence'] = features['tone_7d_avg'] * (-1 * aligned_returns)

    return features

def fetch_and_store_gdelt_historical(start_date: str, end_date: str, db_conn):
    """
    Main entry point for historical backfill.
    """
    # 1. Fetch from BigQuery
    df_gdelt = fetch_gdelt_bigquery(start_date, end_date)

    if df_gdelt.empty:
        logger.error("No GDELT data fetched. Historical backfill aborted.")
        return

    # 2. Fetch Price series for divergence calculation
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

    # 3. Compute features
    features = compute_features_from_aggregated(df_gdelt, price_series)

    # 4. Store in DB
    with db_conn.cursor() as cur:
        rows_inserted = 0
        for ts, row in features.iterrows():
            if pd.isna(row['tone_7d_avg']) and pd.isna(row['tone_30d_avg']):
                continue

            cur.execute("""
                INSERT INTO gdelt_features (ts, tone_7d_avg, tone_30d_avg, event_spike_zscore, tone_price_divergence, article_count)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (ts) DO UPDATE SET
                    tone_7d_avg = EXCLUDED.tone_7d_avg,
                    tone_30d_avg = EXCLUDED.tone_30d_avg,
                    event_spike_zscore = EXCLUDED.event_spike_zscore,
                    tone_price_divergence = EXCLUDED.tone_price_divergence,
                    article_count = EXCLUDED.article_count
            """, (
                ts,
                row['tone_7d_avg'],
                row['tone_30d_avg'],
                row['event_spike_zscore'],
                row['tone_price_divergence'],
                int(row['article_count'])
            ))
            rows_inserted += 1

        db_conn.commit()

    logger.info(f"Successfully backfilled {rows_inserted} hours of GDELT features.")
