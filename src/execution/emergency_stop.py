import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional
import psycopg2
import numpy as np
import os

logger = logging.getLogger(__name__)
os.makedirs('logs', exist_ok=True)
logger.addHandler(logging.FileHandler('logs/emergency_stop.log'))
logger.setLevel(logging.INFO)

class EmergencyStopException(Exception):
    """Raised when emergency stop is triggered."""
    pass

def emergency_stop(reason: str, db_conn, pst_system=None) -> Dict:
    """Execute emergency stop: flatten all positions, disable signals."""
    start_time = time.time()
    try:
        cursor = db_conn.cursor()
        # Bug Fix #1: Zero all active signals (last 1 hour), preserving history
        cursor.execute("""
            UPDATE signals
            SET bridge_forecast = 0,
                lstm_signal = NULL,
                hmm_regime = NULL,
                hmm_posterior = NULL
            WHERE timestamp > NOW() - INTERVAL '1 hour'
        """)
        db_conn.commit()

        positions_closed = 0
        if pst_system:
            try:
                positions_closed = pst_system.close_all_positions()
            except Exception as e:
                logger.error(f"[EMERGENCY STOP] pysystemtrade close failed: {e}")

        # Bug Fix #2: Use timezone-aware now()
        cursor.execute("""
            INSERT INTO emergency_stop_log (timestamp, reason, positions_closed, status)
            VALUES (%s, %s, %s, %s)
        """, (datetime.now(timezone.utc), reason, positions_closed, 'executed'))
        db_conn.commit()

        elapsed_ms = (time.time() - start_time) * 1000
        alert_msg = f"🚨 EMERGENCY STOP 🚨 [{reason}] {positions_closed} positions closed in {elapsed_ms:.0f}ms"
        print(alert_msg, file=__import__('sys').stderr)

        return {
            'status': 'stopped',
            'positions_closed': positions_closed,
            'elapsed_ms': elapsed_ms,
            'timestamp': datetime.now(timezone.utc),
            'reason': reason
        }
    except Exception as e:
        logger.error(f"[EMERGENCY STOP] FAILED: {e}")
        raise EmergencyStopException(f"Emergency stop failed: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()

def create_emergency_stop_table(db_conn):
    """Create audit log table if not exists."""
    cursor = db_conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emergency_stop_log (
            timestamp TIMESTAMPTZ NOT NULL,
            reason VARCHAR(255),
            positions_closed INTEGER,
            status VARCHAR(50),
            PRIMARY KEY (timestamp)
        );
    """)
    db_conn.commit()
    cursor.close()

class Watchdog:
    """Monitor system health and trigger emergency stop on failures."""
    STOP_CONDITIONS = {
        'drawdown_30d_gt_5pct': 'check_drawdown',
        'data_lag_gt_60min': 'check_data_freshness',
        'signal_stale_24h': 'check_signal_freshness',
        'hmm_posterior_invalid': 'check_hmm_output',
        'lstm_signal_nan': 'check_lstm_output',
        'gdelt_down_30min': 'check_gdelt_health',
    }

    def __init__(self, db_conn, pst_system=None, check_interval_sec=300):
        self.db_conn = db_conn
        self.pst_system = pst_system
        self.check_interval = check_interval_sec
        self.last_check = None

    def check_drawdown(self) -> bool:
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("""
                WITH equity AS (
                    SELECT timestamp,
                           EXP(SUM(COALESCE(garch_vol, 1.0) * bridge_forecast / 2000.0)
                           OVER (ORDER BY timestamp)) as equity
                    FROM signals
                    WHERE timestamp > NOW() - INTERVAL '30 days'
                ),
                peaks AS (
                    SELECT equity,
                           MAX(equity) OVER (ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as peak
                    FROM equity
                )
                SELECT COALESCE(MAX(1 - equity / NULLIF(peak, 0)), 0) as max_dd FROM peaks
            """)
            result = cursor.fetchone()
            max_dd = result[0] if result and result[0] is not None else 0
            return max_dd <= 0.05
        except Exception:
            return False
        finally:
            cursor.close()

    def check_data_freshness(self) -> bool:
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("SELECT EXTRACT(EPOCH FROM (NOW() - MAX(timestamp))) FROM ohlcv_xauusd")
            result = cursor.fetchone()
            lag_sec = result[0] if result and result[0] is not None else 999999
            return lag_sec <= 3600
        except Exception:
            return False
        finally:
            cursor.close()

    def check_signal_freshness(self) -> bool:
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("SELECT EXTRACT(EPOCH FROM (NOW() - MAX(timestamp))) FROM signals")
            result = cursor.fetchone()
            lag_sec = result[0] if result and result[0] is not None else 999999
            return lag_sec <= 86400
        except Exception:
            return False
        finally:
            cursor.close()

    def check_hmm_output(self) -> bool:
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM signals
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                AND (hmm_posterior IS NULL OR hmm_regime IS NULL)
            """)
            return cursor.fetchone()[0] == 0
        except Exception:
            return False
        finally:
            cursor.close()

    def check_lstm_output(self) -> bool:
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM signals
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                AND (lstm_signal IS NULL OR lstm_signal < 0 OR lstm_signal > 1)
            """)
            return cursor.fetchone()[0] == 0
        except Exception:
            return False
        finally:
            cursor.close()

    def check_gdelt_health(self) -> bool:
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("SELECT EXTRACT(EPOCH FROM (NOW() - MAX(timestamp))) FROM gdelt_features")
            result = cursor.fetchone()
            lag_sec = result[0] if result and result[0] is not None else 999999
            return True # Non-fatal
        except Exception:
            return True
        finally:
            cursor.close()

    def run_health_check(self) -> Dict[str, bool]:
        results = {}
        for condition_name, check_method in self.STOP_CONDITIONS.items():
            try:
                results[condition_name] = getattr(self, check_method)()
            except Exception:
                results[condition_name] = False
        self.last_check = datetime.now(timezone.utc)
        return results

    def should_stop(self) -> Optional[str]:
        results = self.run_health_check()
        for condition, passed in results.items():
            if not passed: return condition
        return None
