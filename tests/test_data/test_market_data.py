"""
Tests for data/market_data.py

Covers fetch_placeholder_candles() and validate_candle_schema().
"""

from __future__ import annotations

import pytest
import pandas as pd

from config.settings import TIMEFRAME_FREQ
from data.market_data import (
    REQUIRED_COLUMNS,
    fetch_placeholder_candles,
    validate_candle_schema,
)

SYMBOL = "XAUUSD"
NUMERIC_COLS = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=list(TIMEFRAME_FREQ.keys()))
def candles(request: pytest.FixtureRequest) -> pd.DataFrame:
    """Parametrised fixture that yields a candle DataFrame for every timeframe."""
    return fetch_placeholder_candles(SYMBOL, request.param)


# ---------------------------------------------------------------------------
# fetch_placeholder_candles — return type and schema
# ---------------------------------------------------------------------------


def test_returns_dataframe(candles: pd.DataFrame) -> None:
    """fetch_placeholder_candles must return a pandas DataFrame."""
    assert isinstance(candles, pd.DataFrame)


def test_required_columns_present(candles: pd.DataFrame) -> None:
    """All canonical candle columns must be present in the returned DataFrame."""
    for col in REQUIRED_COLUMNS:
        assert col in candles.columns, f"Missing column: {col}"


def test_candle_count(candles: pd.DataFrame) -> None:
    """DataFrame must contain exactly 300 candles."""
    assert len(candles) == 300


def test_symbol_column_value(candles: pd.DataFrame) -> None:
    """Every row in the 'symbol' column must equal the requested symbol."""
    assert (candles["symbol"] == SYMBOL).all()


def test_timeframe_column_value(request: pytest.FixtureRequest, candles: pd.DataFrame) -> None:
    """Every row in the 'timeframe' column must equal the requested timeframe."""
    expected_tf = request.node.callspec.params["candles"]
    assert (candles["timeframe"] == expected_tf).all()


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def test_timestamps_are_utc(candles: pd.DataFrame) -> None:
    """Timestamps must be timezone-aware and set to UTC."""
    assert candles["timestamp"].dt.tz is not None
    assert str(candles["timestamp"].dt.tz) == "UTC"


def test_timestamps_strictly_increasing(candles: pd.DataFrame) -> None:
    """Timestamps must be sorted in strictly ascending order."""
    assert candles["timestamp"].is_monotonic_increasing


def test_timestamps_uniform_spacing(candles: pd.DataFrame) -> None:
    """All consecutive timestamp deltas must be identical (uniform candle spacing)."""
    deltas = candles["timestamp"].diff().dropna().unique()
    assert len(deltas) == 1, f"Non-uniform spacing detected: {deltas}"


# ---------------------------------------------------------------------------
# Numeric columns — no NaN
# ---------------------------------------------------------------------------


def test_no_nan_in_numeric_columns(candles: pd.DataFrame) -> None:
    """No NaN values are permitted in any numeric price or volume column."""
    for col in NUMERIC_COLS:
        assert not candles[col].isna().any(), f"NaN found in column '{col}'"


# ---------------------------------------------------------------------------
# OHLC integrity
# ---------------------------------------------------------------------------


def test_high_gte_open(candles: pd.DataFrame) -> None:
    """high must be greater than or equal to open for every candle."""
    assert (candles["high"] >= candles["open"]).all()


def test_high_gte_close(candles: pd.DataFrame) -> None:
    """high must be greater than or equal to close for every candle."""
    assert (candles["high"] >= candles["close"]).all()


def test_low_lte_open(candles: pd.DataFrame) -> None:
    """low must be less than or equal to open for every candle."""
    assert (candles["low"] <= candles["open"]).all()


def test_low_lte_close(candles: pd.DataFrame) -> None:
    """low must be less than or equal to close for every candle."""
    assert (candles["low"] <= candles["close"]).all()


def test_high_gte_low(candles: pd.DataFrame) -> None:
    """high must be greater than or equal to low for every candle."""
    assert (candles["high"] >= candles["low"]).all()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_output() -> None:
    """Two calls with the same arguments must return identical DataFrames."""
    df1 = fetch_placeholder_candles(SYMBOL, "H1")
    df2 = fetch_placeholder_candles(SYMBOL, "H1")
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_timeframe_raises_value_error() -> None:
    """An unsupported timeframe string must raise ValueError."""
    with pytest.raises(ValueError, match="Unrecognised timeframe"):
        fetch_placeholder_candles(SYMBOL, "W1")


# ---------------------------------------------------------------------------
# validate_candle_schema — error paths
# ---------------------------------------------------------------------------


def test_validate_missing_column_raises() -> None:
    """validate_candle_schema must raise ValueError when a required column is absent."""
    df = fetch_placeholder_candles(SYMBOL, "M15").drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_candle_schema(df)


def test_validate_nan_raises() -> None:
    """validate_candle_schema must raise ValueError when a numeric column contains NaN."""
    df = fetch_placeholder_candles(SYMBOL, "H4").copy()
    df.loc[0, "close"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_candle_schema(df)


def test_validate_unsorted_timestamps_raises() -> None:
    """validate_candle_schema must raise ValueError when timestamps are not ascending."""
    df = fetch_placeholder_candles(SYMBOL, "M5").iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="ascending"):
        validate_candle_schema(df)


def test_validate_timezone_naive_raises() -> None:
    """validate_candle_schema must raise ValueError when timestamps are timezone-naive."""
    df = fetch_placeholder_candles(SYMBOL, "D1").copy()
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_candle_schema(df)


def test_validate_ohlc_integrity_raises() -> None:
    """validate_candle_schema must raise ValueError for invalid OHLC relationships."""
    df = fetch_placeholder_candles(SYMBOL, "H1").copy()
    df.loc[0, "high"] = df.loc[0, "low"] - 1.0  # force high < low
    with pytest.raises(ValueError, match="OHLC"):
        validate_candle_schema(df)
