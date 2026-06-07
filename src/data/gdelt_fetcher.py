import pandas as pd
from gdeltdoc import GdeltDoc, Filters
import logging
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

XAU_THEMES = [
    'gold', 'XAUUSD', 'inflation', 'federal reserve', 'interest rate',
    'geopolitical', 'war', 'central bank', 'safe haven', 'dollar'
]

def fetch_gdelt_docs(hours_back: int = 24) -> pd.DataFrame:
    """
    Fetches news articles from GDELT Doc API for XAU/USD themes.
    """
    f = Filters(
        keyword=" OR ".join(XAU_THEMES),
        start_date=(datetime.now() - timedelta(hours=hours_back)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d")
    )

    gd = GdeltDoc()

    try:
        # gdeltdoc.query() usually returns a DataFrame or list of dicts depending on version
        # It internally uses requests, but we'll wrap it for timeout handling
        # Since gdeltdoc doesn't explicitly expose timeout, we assume standard behavior
        # and handle it via the caller or specific exception handling if it hangs.
        articles = gd.article_search(f)

        if articles is None or articles.empty:
            logger.warning("GDELT Doc API returned no results.")
            return pd.DataFrame(columns=['datetime', 'title', 'url', 'tone', 'domain'])

        # Ensure column consistency
        # GDELT usually returns 'seendate' which is UTC string like '20240101010000'
        # Or 'datetime' depending on API method.
        if 'seendate' in articles.columns:
            articles['datetime'] = pd.to_datetime(articles['seendate'])

        required_cols = ['datetime', 'title', 'url', 'tone', 'domain']
        return articles[[c for c in required_cols if c in articles.columns]]

    except Exception as e:
        logger.error(f"GDELT API error or timeout: {e}")
        return pd.DataFrame(columns=['datetime', 'title', 'url', 'tone', 'domain'])

def compute_gdelt_features(df: pd.DataFrame, price_series: pd.Series) -> pd.DataFrame:
    """
    Computes sentiment and event features from GDELT data.
    """
    if df.empty:
        return pd.DataFrame()

    # Ensure datetime is index and UTC
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime').sort_index()

    # Aggregate by hour
    hourly_tone = df['tone'].resample('1h').mean()
    article_count = df['url'].resample('1h').count()

    # Combine into features DF
    features = pd.DataFrame({
        'tone_avg': hourly_tone,
        'article_count': article_count
    })

    # 1. tone_7d_avg: rolling 7-day mean (168 hours)
    features['tone_7d_avg'] = features['tone_avg'].rolling(window=168, min_periods=1).mean()

    # 2. tone_30d_avg: rolling 30-day mean (720 hours)
    features['tone_30d_avg'] = features['tone_avg'].rolling(window=720, min_periods=1).mean()

    # 3. event_spike_zscore: (count - mean30d) / std30d
    count_mean_30d = features['article_count'].rolling(window=720, min_periods=1).mean()
    count_std_30d = features['article_count'].rolling(window=720, min_periods=1).std().replace(0, 1)
    features['event_spike_zscore'] = (features['article_count'] - count_mean_30d) / count_std_30d

    # 4. tone_price_divergence = tone_7d_avg * (-1 * price_returns_24h)
    # price_series must be hourly indexed as well
    price_returns_24h = price_series.pct_change(24)

    # Reindex price_returns to match features index
    aligned_returns = price_returns_24h.reindex(features.index).ffill()
    features['tone_price_divergence'] = features['tone_7d_avg'] * (-1 * aligned_returns)

    return features[['tone_7d_avg', 'tone_30d_avg', 'event_spike_zscore', 'tone_price_divergence', 'article_count']]

def fetch_and_store_gdelt(hours_back: int, price_series: pd.Series, db_conn):
    """
    Fetches GDELT data, computes features, and stores in the database.
    """
    # GDELT_FALLBACK logic: if API fails, last stored features stay unchanged
    df_docs = fetch_gdelt_docs(hours_back)

    if df_docs.empty:
        logger.warning("GDELT fetch failed or empty. Skipping feature computation/storage (Fallback Mode).")
        return

    features = compute_gdelt_features(df_docs, price_series)

    if features.empty:
        logger.warning("No features computed from GDELT data.")
        return

    with db_conn.cursor() as cur:
        for ts, row in features.iterrows():
            # Check if values are NaN before insertion
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

        db_conn.commit()

    logger.info(f"Successfully stored {len(features)} hours of GDELT features.")
