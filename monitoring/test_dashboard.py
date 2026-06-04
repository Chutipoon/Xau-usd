#!/usr/bin/env python3
"""Validate Grafana dashboard JSON structure."""

import json
import sys
from pathlib import Path

def validate_dashboard(path: str) -> bool:
    """Validate dashboard JSON structure."""

    with open(path) as f:
        dashboard = json.load(f)

    # Check required fields
    assert 'title' in dashboard, "Missing title"
    assert 'uid' in dashboard, "Missing uid"
    assert 'panels' in dashboard, "Missing panels array"

    # Check panel count (should be 6)
    panels = dashboard['panels']
    assert len(panels) >= 6, f"Expected >= 6 panels, got {len(panels)}"

    # Check panel types
    panel_types = [p.get('type') for p in panels]
    assert 'stat' in panel_types, "Missing stat panel (data freshness)"
    assert 'gauge' in panel_types, "Missing gauge panel (bridge forecast)"
    assert 'timeseries' in panel_types, "Missing timeseries panel (GDELT/LSTM)"
    assert 'text' in panel_types, "Missing text panel (alerts)"

    print("✅ Dashboard JSON valid")
    print(f"   Title: {dashboard['title']}")
    print(f"   UID: {dashboard['uid']}")
    print(f"   Panels: {len(panels)}")
    return True

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'monitoring/grafana_dashboard.json'
    validate_dashboard(path)
