"""
Tests for data/data_loader.py

Covers load_market_data() including happy-path, schema validation,
symbol guard, and source-routing error handling.
"""

from __future__ import annotations

import pytest
import pandas as pd

from config.settings import SUPPORTED_SYMBOL, TIMEFRAMES
from data.data_loader import load_market_data
from data.market_data import REQUIRED_COLUMNS, validate_candle_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def multi_tf_data() -> dict[str, pd.DataFrame]:
    """Load the full multi-timeframe placeholder dataset once per test module."""
    return load_market_data(SUPPORTED_SYMBOL)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_dict(multi_tf_data: dict) -> None:
    """load_market_data must return a dictionary."""
    assert isinstance(multi_tf_data, dict)


def test_all_timeframes_present(multi_tf_data: dict) -> None:
    """The returned dict must contain a key for every configured timeframe."""
    assert set(multi_tf_data.keys()) == set(TIMEFRAMES)


def test_all_values_are_dataframes(multi_tf_data: dict) -> None:
    """Every value in the returned dict must be a pandas DataFrame."""
    for tf, df in multi_tf_data.items():
        assert isinstance(df, pd.DataFrame), f"Expected DataFrame for timeframe '{tf}'"


# ---------------------------------------------------------------------------
# Schema validation per timeframe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
def test_dataframe_has_required_columns(timeframe: str, multi_tf_data: dict) -> None:
    """Each timeframe DataFrame must contain all canonical candle columns."""
    df = multi_tf_data[timeframe]
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"[{timeframe}] Missing column: {col}"


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
def test_dataframe_passes_schema_validation(timeframe: str, multi_tf_data: dict) -> None:
    """Each timeframe DataFrame must pass full validate_candle_schema without raising."""
    validate_candle_schema(multi_tf_data[timeframe])


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
def test_dataframe_row_count(timeframe: str, multi_tf_data: dict) -> None:
    """Each timeframe DataFrame must contain exactly 300 candles."""
    assert len(multi_tf_data[timeframe]) == 300


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
def test_timestamps_utc(timeframe: str, multi_tf_data: dict) -> None:
    """Timestamps in each timeframe DataFrame must be UTC-aware."""
    df = multi_tf_data[timeframe]
    assert str(df["timestamp"].dt.tz) == "UTC"


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
def test_no_nan_values(timeframe: str, multi_tf_data: dict) -> None:
    """No NaN values are permitted in numeric price columns."""
    df = multi_tf_data[timeframe]
    price_cols = ["open", "high", "low", "close", "volume"]
    for col in price_cols:
        assert not df[col].isna().any(), f"[{timeframe}] NaN in column '{col}'"


# ---------------------------------------------------------------------------
# Symbol validation
# ---------------------------------------------------------------------------


def test_invalid_symbol_raises_value_error() -> None:
    """load_market_data must raise ValueError when passed an unsupported symbol."""
    with pytest.raises(ValueError, match="Unsupported symbol"):
        load_market_data("EURUSD")


def test_empty_symbol_raises_value_error() -> None:
    """load_market_data must raise ValueError when passed an empty string as symbol."""
    with pytest.raises(ValueError):
        load_market_data("")


# ---------------------------------------------------------------------------
# Source routing
# ---------------------------------------------------------------------------


def test_explicit_placeholder_source_works() -> None:
    """Explicitly passing source='placeholder' must return valid data."""
    data = load_market_data(SUPPORTED_SYMBOL, source="placeholder")
    assert set(data.keys()) == set(TIMEFRAMES)


def test_csv_source_raises_not_implemented() -> None:
    """source='csv' must raise NotImplementedError until implemented."""
    with pytest.raises(NotImplementedError):
        load_market_data(SUPPORTED_SYMBOL, source="csv")


def test_api_source_raises_value_error() -> None:
    """source='api' is no longer accepted — replaced by 'oanda'. Must raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported data source"):
        load_market_data(SUPPORTED_SYMBOL, source="api")


def test_oanda_source_calls_fetch_oanda_candles() -> None:
    """source='oanda' must delegate to fetch_oanda_candles() for each timeframe."""
    import pandas as pd
    from unittest.mock import patch, MagicMock
    from data.market_data import REQUIRED_COLUMNS, validate_candle_schema

    # Build a minimal valid canonical DataFrame that passes schema validation
    def _make_df(symbol: str, tf: str) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=1, freq="5min", tz="UTC"),
            "symbol":    ["XAUUSD"],
            "timeframe": [tf],
            "open":      [2000.0],
            "high":      [2005.0],
            "low":       [1999.0],
            "close":     [2003.0],
            "volume":    [1000.0],
        })

    with patch("data.data_loader.fetch_oanda_candles", side_effect=_make_df) as mock_fetch:
        result = load_market_data(SUPPORTED_SYMBOL, source="oanda")

    assert mock_fetch.call_count == len(TIMEFRAMES)
    assert set(result.keys()) == set(TIMEFRAMES)


def test_unknown_source_raises() -> None:
    """An unrecognised source must raise ValueError or NotImplementedError."""
    with pytest.raises((ValueError, NotImplementedError)):
        load_market_data(SUPPORTED_SYMBOL, source="websocket")
