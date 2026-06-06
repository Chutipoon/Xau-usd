import os
import logging
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
from datetime import datetime, timedelta
import psycopg2.extras

logger = logging.getLogger(__name__)

# GKG Themes related to XAU/USD
XAU_GKG_THEMES = [
    'ECON_GOLD', 'ECON_INFLATION', 'ECON_CENTRALBANK_FEDERAL_RESERVE',
    'ECON_INTEREST_RATES', 'WAR', 'CONFLICT', 'ECON_CURRENCY', 'ECON_MONETARYPOLICY'
]

# Keywords for URL matching as a backup
XAU_KEYWORDS = [
    'gold', 'xauusd', 'inflation', 'federal reserve', 'interest rate',
    'geopolitical', 'central bank', 'safe haven', 'dollar'
]

def fetch_gdelt_bigquery(start_date: str, end_date: str, max_gb_per_query: float = 50.0) -> pd.DataFrame:
    """
    Queries BigQuery GDELT GKG partitioned table for XAU-related content.
    Aggregates by hour.
    """
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    client = bigquery.Client(project=project_id)

    # Use _PARTITIONTIME for efficient partition pruning
    theme_filter = "|".join(XAU_GKG_THEMES)
    keyword_regex = "|".join([fr"\b{kw.replace(' ', '[ -]')}\b" for kw in XAU_KEYWORDS])

    # GKG query using Themes and DocumentIdentifier (URL)
    # V2Tone is comma-separated, first element is Average Tone
    query = f"""
        SELECT
            TIMESTAMP_TRUNC(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATEADDED AS STRING)), HOUR) as ts,
            AVG(SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) as tone,
            COUNT(DISTINCT DocumentIdentifier) as article_count
        FROM
            `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE
            _PARTITIONTIME BETWEEN TIMESTAMP(@start_date) AND TIMESTAMP(@end_date)
            AND (
                REGEXP_CONTAINS(Themes, r'{theme_filter}')
                OR REGEXP_CONTAINS(LOWER(DocumentIdentifier), r'{keyword_regex.lower()}')
            )
        GROUP BY 1
        ORDER BY 1
    """

    query_params = [
        bigquery.ScalarQueryParameter("start_date", "STRING", start_date),
        bigquery.ScalarQueryParameter("end_date", "STRING", end_date),
    ]

    try:
        # Use maximum_bytes_billed to enforce quota safely
        max_bytes = int(max_gb_per_query * 1024**3)
        job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=max_bytes,
            query_parameters=query_params
        )

        logger.info(f"Executing BigQuery GKG fetch from {start_date} to {end_date} (Max {max_gb_per_query}GB)")

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
    # Use a conservative max_gb_per_query since GKG is large
    df_gdelt = fetch_gdelt_bigquery(start_date, end_date, max_gb_per_query=100.0)

    if df_gdelt.empty:
        logger.error("No GDELT data fetched. Historical backfill aborted.")
        return

    # Fetch Price series using column names matching schema.sql (ts, close_price)
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
        # Schema includes article_count as per db/schema.sql
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
