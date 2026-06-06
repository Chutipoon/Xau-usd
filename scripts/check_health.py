#!/usr/bin/env python3
"""
Health check script verifying lag for OHLCV, FRED, COT, GDELT, and signals.
"""

import os
import sys
import psycopg2
from datetime import datetime

def check_lag(cursor, table, column='timestamp'):
    try:
        cursor.execute(f"SELECT EXTRACT(EPOCH FROM (NOW() - MAX({column}))) FROM {table}")
        result = cursor.fetchone()
        return result[0] if result and result[0] is not None else None
    except Exception as e:
        print(f"Error checking {table}: {e}")
        return None

def format_lag(seconds):
    if seconds is None:
        return "No data"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.0f} min"
    if seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    return f"{seconds/86400:.1f} days"

def main():
    db_url = os.getenv('TIMESCALE_URL', 'postgresql://localhost/xauusd')
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)

    checks = [
        ('OHLCV', 'ohlcv_xauusd', 'ts', 3600),          # 1 hour
        ('FRED', 'macro_fred', 'obs_date', 172800),    # 2 days
        ('COT', 'cot_xauusd', 'week_date', 691200),     # 8 days
        ('GDELT', 'gdelt_features', 'ts', 3600),        # 1 hour
        ('Signals', 'signals', 'timestamp', 3600)      # 1 hour
    ]

    all_ok = True
    print(f"System Health Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    for name, table, col, threshold in checks:
        lag = check_lag(cursor, table, col)
        if lag is None:
            print(f"❌ {name}: No data found in {table}")
            all_ok = False
        elif lag > threshold:
            print(f"❌ {name}: stale ({format_lag(lag)} ago)")
            all_ok = False
        else:
            print(f"✅ {name}: updated {format_lag(lag)} ago")

    cursor.close()
    conn.close()

    if not all_ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
