"""Watchdog service entry point."""
import os
import time
import psycopg2
from src.execution.emergency_stop import Watchdog, emergency_stop, create_emergency_stop_table

def run():
    db_url = os.getenv('TIMESCALE_URL', 'postgresql://localhost/xauusd')
    conn = psycopg2.connect(db_url)
    create_emergency_stop_table(conn)
    watchdog = Watchdog(conn)

    print("Watchdog service started...")
    while True:
        reason = watchdog.should_stop()
        if reason:
            print(f"STOP CONDITION DETECTED: {reason}")
            emergency_stop(reason, conn)
            break
        time.sleep(300)

if __name__ == "__main__":
    run()
