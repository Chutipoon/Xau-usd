import argparse
import os
import sys
import logging
import psycopg2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.gdelt_bigquery import fetch_and_store_gdelt_historical

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Fetch and store historical GDELT data from BigQuery')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    try:
        conn = psycopg2.connect(db_url)
        logger.info(f"Starting historical backfill from {args.start_date} to {args.end_date}...")
        fetch_and_store_gdelt_historical(args.start_date, args.end_date, conn)
        conn.close()
        logger.info("Historical backfill completed successfully.")
    except Exception as e:
        logger.error(f"Historical backfill failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
