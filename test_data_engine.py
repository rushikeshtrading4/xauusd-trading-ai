"""
Temporary smoke-test script for the Market Data Engine.

Run from the project root:

    python test_data_engine.py

This is NOT part of the test suite (tests/).  Delete once satisfied.
"""

from __future__ import annotations

import pandas as pd

from data.data_loader import load_market_data
from data.market_data import (
    REQUIRED_COLUMNS,
    TIMEFRAME_FREQ,
    fetch_placeholder_candles,
    validate_candle_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(label: str, condition: bool) -> None:
    print(f"  [{PASS if condition else FAIL}] {label}")
    if not condition:
        raise AssertionError(f"Check failed: {label}")


# ---------------------------------------------------------------------------
# 1. load_market_data — happy path
# ---------------------------------------------------------------------------

print("\n--- 1. load_market_data (placeholder) ---")

data = load_market_data("XAUUSD")

check("Returns dict", isinstance(data, dict))
check("All timeframes present", set(data.keys()) == set(TIMEFRAME_FREQ.keys()))

for tf, df in data.items():
    check(f"{tf}: is DataFrame", isinstance(df, pd.DataFrame))
    check(f"{tf}: 300 rows", len(df) == 300)
    check(f"{tf}: all required columns present", all(c in df.columns for c in REQUIRED_COLUMNS))
    check(f"{tf}: no NaN values", not df.isna().any().any())
    check(f"{tf}: timestamps are UTC", str(df["timestamp"].dt.tz) == "UTC")
    check(f"{tf}: timestamps sorted ascending", df["timestamp"].is_monotonic_increasing)
    spacings = df["timestamp"].diff().dropna().unique()
    check(f"{tf}: uniform timestamp spacing (1 unique delta)", len(spacings) == 1)

# ---------------------------------------------------------------------------
# 2. load_market_data — source parameter
# ---------------------------------------------------------------------------

print("\n--- 2. load_market_data source routing ---")

data2 = load_market_data("XAUUSD", source="placeholder")
check("explicit source='placeholder' works", len(data2) == 6)

for not_impl_source in ("csv", "api"):
    try:
        load_market_data("XAUUSD", source=not_impl_source)
        check(f"source='{not_impl_source}' raises NotImplementedError", False)
    except NotImplementedError:
        check(f"source='{not_impl_source}' raises NotImplementedError", True)

try:
    load_market_data("XAUUSD", source="websocket")
    check("unknown source raises ValueError", False)
except ValueError:
    check("unknown source raises ValueError", True)

# ---------------------------------------------------------------------------
# 3. load_market_data — wrong symbol
# ---------------------------------------------------------------------------

print("\n--- 3. Symbol validation ---")

try:
    load_market_data("EURUSD")
    check("EURUSD raises ValueError", False)
except ValueError:
    check("EURUSD raises ValueError", True)

# ---------------------------------------------------------------------------
# 4. validate_candle_schema — error paths
# ---------------------------------------------------------------------------

print("\n--- 4. validate_candle_schema error paths ---")

# Missing columns
try:
    validate_candle_schema(pd.DataFrame({"timestamp": [], "open": []}))
    check("Missing columns raises ValueError", False)
except ValueError:
    check("Missing columns raises ValueError", True)

# NaN value
df_nan = fetch_placeholder_candles("XAUUSD", "H1").copy()
df_nan.loc[0, "close"] = float("nan")
try:
    validate_candle_schema(df_nan)
    check("NaN value raises ValueError", False)
except ValueError:
    check("NaN value raises ValueError", True)

# Unsorted timestamps
df_unsorted = fetch_placeholder_candles("XAUUSD", "M5").iloc[::-1].reset_index(drop=True)
try:
    validate_candle_schema(df_unsorted)
    check("Unsorted timestamps raises ValueError", False)
except ValueError:
    check("Unsorted timestamps raises ValueError", True)

# Timezone-naive timestamps
df_naive = fetch_placeholder_candles("XAUUSD", "D1").copy()
df_naive["timestamp"] = df_naive["timestamp"].dt.tz_localize(None)
try:
    validate_candle_schema(df_naive)
    check("Timezone-naive timestamps raises ValueError", False)
except ValueError:
    check("Timezone-naive timestamps raises ValueError", True)

# Invalid OHLC relationships
df_ohlc = fetch_placeholder_candles("XAUUSD", "H4").copy()
df_ohlc.loc[0, "high"] = df_ohlc.loc[0, "low"] - 1.0  # high < low
try:
    validate_candle_schema(df_ohlc)
    check("Invalid OHLC raises ValueError", False)
except ValueError:
    check("Invalid OHLC raises ValueError", True)

# ---------------------------------------------------------------------------
# 5. fetch_placeholder_candles — unknown timeframe
# ---------------------------------------------------------------------------

print("\n--- 5. fetch_placeholder_candles unknown timeframe ---")

try:
    fetch_placeholder_candles("XAUUSD", "W1")
    check("Unknown timeframe raises ValueError", False)
except ValueError:
    check("Unknown timeframe raises ValueError", True)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n\033[92mAll checks passed.\033[0m\n")
