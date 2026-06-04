import logging
import time
from datetime import datetime
from typing import Dict, Optional
import psycopg2
import numpy as np

logger = logging.getLogger(__name__)
# Ensure logs directory exists
import os
os.makedirs('logs', exist_ok=True)
logger.addHandler(logging.FileHandler('logs/emergency_stop.log'))
logger.setLevel(logging.INFO)


class EmergencyStopException(Exception):
    """Raised when emergency stop is triggered."""
    pass


def emergency_stop(reason: str, db_conn, pst_system=None) -> Dict:
    """
    Execute emergency stop: flatten all positions, disable signals.
    """
    start_time = time.time()

    try:
        # 1. Set all signals to 0
        cursor = db_conn.cursor()
        cursor.execute("""
            UPDATE signals
            SET bridge_forecast = 0,
                lstm_signal = NULL,
                hmm_regime = NULL,
                hmm_posterior = NULL
            WHERE timestamp > NOW() - INTERVAL '1 hour'
        """)
        db_conn.commit()
        logger.warning(f"[EMERGENCY STOP] Signals zeroed: {cursor.rowcount} rows updated")

        # 2. Close positions in pysystemtrade (if available)
        positions_closed = 0
        if pst_system:
            try:
                positions_closed = pst_system.close_all_positions()
                logger.warning(f"[EMERGENCY STOP] Positions closed: {positions_closed}")
            except Exception as e:
                logger.error(f"[EMERGENCY STOP] pysystemtrade close failed: {e}")

        # 3. Write audit log
        cursor.execute("""
            INSERT INTO emergency_stop_log (timestamp, reason, positions_closed, status)
            VALUES (%s, %s, %s, %s)
        """, (datetime.utcnow(), reason, positions_closed, 'executed'))
        db_conn.commit()
        logger.warning(f"[EMERGENCY STOP] Audit log written")

        # 4. Alert to stderr (for monitoring)
        elapsed_ms = (time.time() - start_time) * 1000
        alert_msg = f"🚨 EMERGENCY STOP 🚨 [{reason}] {positions_closed} positions closed in {elapsed_ms:.0f}ms"
        print(alert_msg, file=__import__('sys').stderr)
        logger.warning(alert_msg)

        if elapsed_ms > 1000:
            logger.error(f"[EMERGENCY STOP] SLOW! Took {elapsed_ms:.0f}ms (target <1000ms)")

        return {
            'status': 'stopped',
            'positions_closed': positions_closed,
            'elapsed_ms': elapsed_ms,
            'timestamp': datetime.utcnow(),
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
        """
        Args:
            db_conn: TimescaleDB connection
            pst_system: pysystemtrade system (optional)
            check_interval_sec: Health check every N seconds (default 300)
        """
        self.db_conn = db_conn
        self.pst_system = pst_system
        self.check_interval = check_interval_sec
        self.last_check = None

    def check_drawdown(self) -> bool:
        """Check 30-day rolling drawdown."""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("""
                WITH equity AS (
                    -- Start at 1.0 equity
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
            if max_dd > 0.05:
                logger.error(f"[WATCHDOG] Drawdown {max_dd:.1%} > 5% threshold")
                return False
            return True
        except Exception as e:
            logger.error(f"Drawdown check error: {e}")
            return False
        finally:
            cursor.close()

    def check_data_freshness(self) -> bool:
        """Check OHLCV data lag."""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("""
                SELECT EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))
                FROM ohlcv_xauusd
            """)
            result = cursor.fetchone()
            lag_sec = result[0] if result and result[0] is not None else 999999
            if lag_sec > 3600:  # > 60 min
                logger.error(f"[WATCHDOG] Data lag {lag_sec}s > 3600s threshold")
                return False
            return True
        except Exception as e:
            logger.error(f"Data freshness check error: {e}")
            return False
        finally:
            cursor.close()

    def check_signal_freshness(self) -> bool:
        """Check bridge_forecast updated."""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("""
                SELECT EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))
                FROM signals
            """)
            result = cursor.fetchone()
            lag_sec = result[0] if result and result[0] is not None else 999999
            if lag_sec > 86400:  # > 24 hours
                logger.error(f"[WATCHDOG] Signals stale {lag_sec}s > 86400s threshold")
                return False
            return True
        except Exception as e:
            logger.error(f"Signal freshness check error: {e}")
            return False
        finally:
            cursor.close()

    def check_hmm_output(self) -> bool:
        """Validate HMM posterior is not NaN/NULL."""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM signals
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                AND (hmm_posterior IS NULL OR hmm_regime IS NULL)
            """)
            invalid_count = cursor.fetchone()[0]
            if invalid_count > 0:
                logger.error(f"[WATCHDOG] {invalid_count} signals with invalid HMM output")
                return False
            return True
        except Exception as e:
            logger.error(f"HMM output check error: {e}")
            return False
        finally:
            cursor.close()

    def check_lstm_output(self) -> bool:
        """Validate LSTM signal is not NaN."""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM signals
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                AND (lstm_signal IS NULL OR lstm_signal < 0 OR lstm_signal > 1)
            """)
            invalid_count = cursor.fetchone()[0]
            if invalid_count > 0:
                logger.error(f"[WATCHDOG] {invalid_count} signals with invalid LSTM output")
                return False
            return True
        except Exception as e:
            logger.error(f"LSTM output check error: {e}")
            return False
        finally:
            cursor.close()

    def check_gdelt_health(self) -> bool:
        """Check GDELT data availability."""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("""
                SELECT EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))
                FROM gdelt_features
            """)
            result = cursor.fetchone()
            lag_sec = result[0] if result and result[0] is not None else 999999
            if lag_sec > 1800:  # > 30 min
                logger.warning(f"[WATCHDOG] GDELT stale {lag_sec}s (fallback to COT+macro)")
                return True  # Non-fatal, fallback available
            return True
        except Exception as e:
            logger.error(f"GDELT health check error: {e}")
            return True # Non-fatal
        finally:
            cursor.close()

    def run_health_check(self) -> Dict[str, bool]:
        """Run all health checks, return status dict."""
        results = {}
        for condition_name, check_method in self.STOP_CONDITIONS.items():
            try:
                results[condition_name] = getattr(self, check_method)()
            except Exception as e:
                logger.error(f"[WATCHDOG] Check {condition_name} failed: {e}")
                results[condition_name] = False

        self.last_check = datetime.utcnow()
        return results

    def should_stop(self) -> Optional[str]:
        """Check if any stop condition is true. Return reason or None."""
        results = self.run_health_check()

        for condition, passed in results.items():
            if not passed:
                return condition

        return None
