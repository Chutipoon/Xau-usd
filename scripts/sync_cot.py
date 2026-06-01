import argparse
import psycopg2
import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.cot_fetcher import fetch_and_store_cot, fetch_cot_gold, validate_cot

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Sync CFTC COT Gold data')
    parser.add_argument('--year-from', type=int, required=True, help='Start year')
    parser.add_argument('--year-to', type=int, required=True, help='End year')

    args = parser.parse_args()

    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    try:
        # 1. Fetch for validation report
        logger.info(f"Fetching COT data for {args.year_from}-{args.year_to} for validation...")
        df = fetch_cot_gold(args.year_from, args.year_to)

        print("\n=== COT VALIDATION REPORT ===")
        print(f"Total Rows: {len(df)}")
        if not df.empty:
            print(f"Date Range: {df['week_date'].min()} to {df['week_date'].max()}")
            is_valid = validate_cot(df)
            print(f"Validation Result: {'PASS' if is_valid else 'FAIL'}")
        else:
            print("No data found.")
        print("==============================\n")

        # 2. Sync to DB
        conn = psycopg2.connect(db_url)
        fetch_and_store_cot(args.year_from, args.year_to, conn)
        conn.close()

    except Exception as e:
        logger.error(f"Sync failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
