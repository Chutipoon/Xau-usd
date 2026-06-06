import pytest
import json
import yaml
from pathlib import Path

def test_dashboard_json_validity():
    path = Path('monitoring/grafana_dashboard.json')
    assert path.exists()

    with open(path) as f:
        dashboard = json.load(f)

    assert dashboard['title'] == "XAU/USD Trading System"
    assert dashboard['uid'] == "xauusd-dashboard"
    assert 'panels' in dashboard

    panels = dashboard['panels']
    assert len(panels) == 6

    # Check TimescaleDB datasource
    for panel in panels:
        if 'datasource' in panel and panel['datasource'] is not None:
            if isinstance(panel['datasource'], dict):
                if panel['datasource'].get('type') == 'postgres':
                    assert panel['datasource'].get('uid') == 'TimescaleDB'

def test_alerts_yaml_validity():
    path = Path('monitoring/alerts.yaml')
    assert path.exists()

    with open(path) as f:
        alerts = yaml.safe_load(f)

    assert 'groups' in alerts
    group = alerts['groups'][0]
    assert group['name'] == "xauusd_trading_alerts"
    assert len(group['rules']) == 6

    assert 'notification_channels' in alerts
    assert len(alerts['notification_channels']) == 2

def test_dashboard_panels_content():
    with open('monitoring/grafana_dashboard.json') as f:
        dashboard = json.load(f)

    panel_titles = [p['title'] for p in dashboard['panels']]
    assert "Panel 1: Data Freshness (seconds lag)" in panel_titles
    assert "Panel 2: Current Regime" in panel_titles
    assert "Panel 3: GDELT Event Spike (7 days)" in panel_titles
    assert "Panel 4: Bridge Forecast (range -20 to +20)" in panel_titles
    assert "Panel 5: LSTM Signal (last 24h)" in panel_titles
    assert "Panel 6: System Alerts & Procedures" in panel_titles
