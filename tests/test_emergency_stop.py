import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import time
from src.execution.emergency_stop import emergency_stop, Watchdog, create_emergency_stop_table

@pytest.fixture
def mock_db():
    conn = MagicMock()
    cursor = conn.cursor.return_value
    return conn, cursor

def test_emergency_stop_timing(mock_db):
    conn, cursor = mock_db

    start = time.perf_counter()
    result = emergency_stop("test stop", conn)
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000
    assert elapsed_ms < 1000
    assert result['status'] == 'stopped'

def test_emergency_stop_audit_log(mock_db):
    conn, cursor = mock_db

    emergency_stop("drawdown exceeded", conn)

    # Check if INSERT into emergency_stop_log was called
    calls = cursor.execute.call_args_list
    audit_call = [c for c in calls if "INSERT INTO emergency_stop_log" in c[0][0]]
    assert len(audit_call) > 0
    args = audit_call[0][0][1] # (timestamp, reason, positions_closed, status)
    assert args[1] == "drawdown exceeded"

def test_watchdog_drawdown_trigger(mock_db):
    conn, cursor = mock_db
    # Mock SQL response for drawdown > 5%
    cursor.fetchone.return_value = [0.06]

    watchdog = Watchdog(conn)
    assert watchdog.check_drawdown() is False

def test_watchdog_data_lag_trigger(mock_db):
    conn, cursor = mock_db
    # Mock SQL response for lag > 3600s
    cursor.fetchone.return_value = [3700]

    watchdog = Watchdog(conn)
    assert watchdog.check_data_freshness() is False

def test_watchdog_signal_stale_trigger(mock_db):
    conn, cursor = mock_db
    # Mock SQL response for lag > 86400s
    cursor.fetchone.return_value = [90000]

    watchdog = Watchdog(conn)
    assert watchdog.check_signal_freshness() is False

def test_watchdog_hmm_invalid_trigger(mock_db):
    conn, cursor = mock_db
    # Mock SQL response for 5 invalid signals
    cursor.fetchone.return_value = [5]

    watchdog = Watchdog(conn)
    assert watchdog.check_hmm_output() is False

def test_watchdog_lstm_invalid_trigger(mock_db):
    conn, cursor = mock_db
    # Mock SQL response for 3 invalid signals
    cursor.fetchone.return_value = [3]

    watchdog = Watchdog(conn)
    assert watchdog.check_lstm_output() is False

def test_watchdog_gdelt_failover(mock_db):
    conn, cursor = mock_db
    # Mock SQL response for GDELT lag > 1800s
    cursor.fetchone.return_value = [2000]

    watchdog = Watchdog(conn)
    # GDELT stale should return True (non-fatal)
    assert watchdog.check_gdelt_health() is True

def test_watchdog_all_pass(mock_db):
    conn, cursor = mock_db
    # Drawdown ok
    # Data lag ok
    # Signal lag ok
    # HMM count 0
    # LSTM count 0
    # GDELT lag ok
    cursor.fetchone.side_effect = [[0.01], [100], [100], [0], [0], [100]]

    watchdog = Watchdog(conn)
    assert watchdog.should_stop() is None
