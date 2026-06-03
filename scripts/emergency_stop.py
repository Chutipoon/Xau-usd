#!/usr/bin/env python3
"""
Emergency stop CLI tool.

Usage:
    python scripts/emergency_stop.py --reason "manual halt"
    python scripts/emergency_stop.py --reason "drawdown exceeded"
"""

import argparse
import psycopg2
import os
from src.execution.emergency_stop import emergency_stop, create_emergency_stop_table

def main():
    parser = argparse.ArgumentParser(description='Emergency stop trading system')
    parser.add_argument('--reason', required=True, help='Reason for emergency stop')
    args = parser.parse_args()

    # Connect to DB
    db_url = os.getenv('TIMESCALE_URL', 'postgresql://localhost/xauusd')
    db_conn = psycopg2.connect(db_url)

    try:
        create_emergency_stop_table(db_conn)
        result = emergency_stop(args.reason, db_conn, pst_system=None)

        print("\n" + "="*60)
        print("✅ EMERGENCY STOP EXECUTED")
        print("="*60)
        print(f"Status: {result['status']}")
        print(f"Positions closed: {result['positions_closed']}")
        print(f"Execution time: {result['elapsed_ms']:.1f}ms")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Reason: {result['reason']}")
        print("="*60)
        print("\n⚠️  POST-STOP CHECKLIST:")
        print("  [ ] Confirm all positions FLAT in broker UI")
        print("  [ ] Check audit log: tail -20 logs/emergency_stop.log")
        print("  [ ] Review cause before system restart")
        print("  [ ] Contact ops lead before resuming trading")
        print("\n")

    except Exception as e:
        print(f"❌ EMERGENCY STOP FAILED: {e}")
        exit(1)
    finally:
        db_conn.close()

if __name__ == '__main__':
    main()
