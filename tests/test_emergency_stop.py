import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone
import time
from src.execution.emergency_stop import emergency_stop, Watchdog

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
    assert (end - start) * 1000 < 1000
    assert result['status'] == 'stopped'

def test_emergency_stop_audit_log(mock_db):
    conn, cursor = mock_db
    emergency_stop("drawdown exceeded", conn)
    calls = cursor.execute.call_args_list
    assert any("INSERT INTO emergency_stop_log" in c[0][0] for c in calls)

def test_watchdog_drawdown_trigger(mock_db):
    conn, cursor = mock_db
    cursor.fetchone.return_value = [0.06]
    watchdog = Watchdog(conn)
    assert watchdog.check_drawdown() is False

def test_watchdog_all_pass(mock_db):
    conn, cursor = mock_db
    cursor.fetchone.side_effect = [[0.01], [100], [100], [0], [0], [100]]
    watchdog = Watchdog(conn)
    assert watchdog.should_stop() is None
