"""Execution layer: signal bridge and risk monitoring."""

from .regime_signal_bridge import RegimeSignalBridge
from .risk_monitor import check_drawdown, check_correlation
from .emergency_stop import emergency_stop, Watchdog

__all__ = ["RegimeSignalBridge", "check_drawdown", "check_correlation", "emergency_stop", "Watchdog"]
