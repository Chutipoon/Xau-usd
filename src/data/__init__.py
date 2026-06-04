"""Data ingestion layer: OHLCV, FRED, COT, GDELT."""

from .dukascopy_fetcher import fetch_ohlcv, fetch_and_store
from .fred_fetcher import fetch_fred_series, fetch_and_store_fred
from .cot_fetcher import fetch_cot_gold, fetch_and_store_cot, validate_cot
from .gdelt_fetcher import fetch_gdelt_docs, compute_gdelt_features, fetch_and_store_gdelt

__all__ = [
    "fetch_ohlcv", "fetch_and_store",
    "fetch_fred_series", "fetch_and_store_fred",
    "fetch_cot_gold", "fetch_and_store_cot", "validate_cot",
    "fetch_gdelt_docs", "compute_gdelt_features", "fetch_and_store_gdelt",
]
