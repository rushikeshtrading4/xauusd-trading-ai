"""
Market data handling module.

Responsible for candle data operations: placeholder generation and
schema validation. Real API / CSV ingestion will replace the placeholder
functions in a later phase.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64tz_dtype  # noqa: F401 kept for compat

from config.settings import TIMEFRAME_FREQ

# Timeframe frequency aliases — 'D' replaces deprecated lowercase 'd'.
_FREQ_ALIASES: dict[str, str] = {
    k: v.replace("1d", "1D") for k, v in TIMEFRAME_FREQ.items()
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: list[str] = [
    "timestamp",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

_PLACEHOLDER_CANDLE_COUNT: int = 300

# Realistic mid-price seed for XAUUSD placeholder data (USD per troy oz).
_XAUUSD_BASE_PRICE: float = 2_650.00


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def fetch_placeholder_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    """Generate synthetic placeholder candles for *symbol* at *timeframe*.

    This stub produces a small, schema-compliant DataFrame so that the rest
    of the pipeline can be developed and tested without a live data feed.
    It will be replaced by real API or CSV-based ingestion in a later phase.

    Args:
        symbol:    Trading instrument (e.g. ``"XAUUSD"``).
        timeframe: Candle timeframe key (e.g. ``"H1"``, ``"M15"``).

    Returns:
        :class:`pandas.DataFrame` with :data:`REQUIRED_COLUMNS` and
        timezone-aware UTC timestamps sorted in ascending order.

    Raises:
        ValueError: If *timeframe* is not one of the supported keys.
    """
    if timeframe not in TIMEFRAME_FREQ:
        raise ValueError(
            f"Unrecognised timeframe '{timeframe}'. "
            f"Supported timeframes: {list(TIMEFRAME_FREQ.keys())}"
        )

    # Align end time to the current UTC minute so boundaries are clean.
    end = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)

    # Generate exactly _PLACEHOLDER_CANDLE_COUNT evenly-spaced UTC timestamps
    # using the correct frequency for this timeframe.
    timestamps = pd.date_range(
        end=end,
        periods=_PLACEHOLDER_CANDLE_COUNT,
        freq=_FREQ_ALIASES[timeframe],
        tz="UTC",
    )

    # Deterministic per (symbol, timeframe) so tests are reproducible.
    seed = abs(hash(f"{symbol}{timeframe}")) % (2 ** 32)
    rng = np.random.default_rng(seed=seed)

    # Simulate a realistic price walk around the XAUUSD base price.
    price_changes = rng.normal(loc=0.0, scale=2.5, size=_PLACEHOLDER_CANDLE_COUNT)
    closes  = np.round(_XAUUSD_BASE_PRICE + np.cumsum(price_changes), 2)
    opens   = np.round(closes + rng.normal(0.0, 2.0, size=_PLACEHOLDER_CANDLE_COUNT), 2)
    spreads = np.round(np.abs(rng.normal(0.0, 3.0, size=_PLACEHOLDER_CANDLE_COUNT)), 2)
    highs   = np.round(np.maximum(opens, closes) + spreads, 2)
    lows    = np.round(np.minimum(opens, closes) - spreads, 2)
    volumes = np.round(np.abs(rng.normal(1_000.0, 200.0, size=_PLACEHOLDER_CANDLE_COUNT)), 2)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol":    symbol,
            "timeframe": timeframe,
            "open":      opens,
            "high":      highs,
            "low":       lows,
            "close":     closes,
            "volume":    volumes,
        },
        columns=REQUIRED_COLUMNS,
    )

    return df


def validate_candle_schema(df: pd.DataFrame) -> None:
    """Validate that *df* conforms to the canonical candle schema.

    Validation rules
    ----------------
    1. All :data:`REQUIRED_COLUMNS` must be present.
    2. No ``NaN`` values may exist in any column.
    3. The ``timestamp`` column must be sorted in strictly ascending order.
    4. Timestamps must be timezone-aware and expressed in UTC (offset = 0).

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: With a descriptive message for the first rule violated.
    """
    # 1. Required columns present.
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Candle DataFrame is missing required columns: {missing}"
        )

    # 2. No NaN values in numeric price columns.
    _PRICE_COLS: list[str] = ["open", "high", "low", "close", "volume"]
    nan_cols = [c for c in _PRICE_COLS if c in df.columns and df[c].isna().any()]
    if nan_cols:
        raise ValueError(
            f"Candle DataFrame contains NaN values in numeric column(s): {nan_cols}"
        )

    # 3. Timestamps sorted ascending.
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Candle DataFrame 'timestamp' column is not sorted in ascending order."
        )

    # 4. Timezone-aware UTC timestamps.
    if not isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype):
        raise ValueError(
            "Candle DataFrame 'timestamp' column must be timezone-aware (UTC). "
            "Found timezone-naive timestamps."
        )

    # Individual pd.Timestamp objects expose utcoffset(); UTC must be offset 0.
    sample_offset = df["timestamp"].iloc[0].utcoffset()
    if sample_offset != timedelta(0):
        sample_tz = df["timestamp"].dt.tz
        raise ValueError(
            f"Candle DataFrame 'timestamp' column timezone must be UTC. "
            f"Found timezone: '{sample_tz}'"
        )

    # 5. OHLC integrity — high must be the highest value, low the lowest.
    violations = (
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"]  > df["open"]) |
        (df["low"]  > df["close"]) |
        (df["high"] < df["low"])
    )
    if violations.any():
        count = int(violations.sum())
        raise ValueError(
            f"Candle DataFrame contains {count} row(s) with invalid OHLC relationships. "
            "Required: high >= open, high >= close, low <= open, low <= close, high >= low."
        )

