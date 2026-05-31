import pytest
import re

def test_schema_has_hypertable_calls():
    with open('db/schema.sql') as f:
        sql = f.read()
    # Using regex to handle potential whitespace differences
    assert re.search(r"create_hypertable\(\s*'ohlcv_xauusd'", sql)
    assert re.search(r"create_hypertable\(\s*'gdelt_features'", sql)

def test_reserved_words_not_used():
    with open('db/schema.sql') as f:
        sql = f.read().lower()
    # ต้องไม่มี column ชื่อ timestamp, open, close, value, date
    # We use word boundaries \b to ensure we match the exact word and not substrings like open_price
    reserved_words = ['timestamp', 'open', 'close', 'value', 'date']
    for word in reserved_words:
        # This pattern looks for the reserved word as a column name (usually followed by a space and type)
        pattern = rf"\b{word}\s+(double|timestamptz|date|bigint|integer|varchar|jsonb)"
        assert not re.search(pattern, sql), f"Reserved word '{word}' used as column name in schema.sql"

def test_idempotent_keywords():
    with open('db/schema.sql') as f:
        sql = f.read()
    tables = ['ohlcv_xauusd','cot_xauusd', 'gdelt_features','macro_fred','signals']
    for t in tables:
        assert f'CREATE TABLE IF NOT EXISTS {t}' in sql

def test_teardown_drops_all_tables():
    with open('db/teardown.sql') as f:
        sql = f.read()
    assert 'DROP TABLE IF EXISTS' in sql
    tables = ['ohlcv_xauusd','cot_xauusd', 'gdelt_features','macro_fred','signals']
    for t in tables:
        assert re.search(rf'DROP TABLE IF EXISTS\s+{t}', sql)

def test_readme_mentions_all_tables():
    with open('db/README.md') as f:
        md = f.read()
    for table in ['ohlcv_xauusd','cot_xauusd', 'gdelt_features','macro_fred','signals']:
        assert table in md, f"README missing: {table}"
