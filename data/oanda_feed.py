"""
OANDA data feed — converts raw OANDA candles to the canonical DataFrame schema.

This module is the bridge between OandaClient (raw HTTP) and data_loader.py
(canonical DataFrames). It owns three concerns:

1. Calling OandaClient.fetch_candles() with the right instrument + granularity
2. Converting OANDA's JSON candle format to the canonical OHLCV schema
3. Filtering incomplete candles and validating the result

Public API
──────────
fetch_oanda_candles(symbol, timeframe, count, client) → pd.DataFrame

The client parameter is optional — if not provided a new OandaClient()
is created using environment variables. This design makes the function
fully testable by injecting a mock client.

Environment variables
─────────────────────
OANDA_API_KEY       — required
OANDA_ACCOUNT_ID    — required for streaming, optional for candle fetch
OANDA_ENVIRONMENT   — "practice" (default) or "live"
"""

from __future__ import annotations

import os

import pandas as pd

from data.oanda_client import (
    OandaClient,
    TIMEFRAME_TO_GRANULARITY,
    XAUUSD_INSTRUMENT,
    DEFAULT_CANDLE_COUNT,
)
from config.settings import PLACEHOLDER_CANDLE_COUNT


def fetch_oanda_candles(
    symbol:    str,
    timeframe: str,
    count:     int | None = None,
    client:    OandaClient | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV candles from OANDA and return a canonical DataFrame.

    Parameters
    ----------
    symbol : str
        Trading symbol. Only "XAUUSD" is supported; mapped to "XAU_USD".
    timeframe : str
        One of "M1", "M5", "M15", "H1", "H4", "D1".
    count : int | None
        Number of candles to fetch. Defaults to PLACEHOLDER_CANDLE_COUNT
        (300) so it matches the placeholder data size exactly.
    client : OandaClient | None
        Injected client for testing. If None, creates a new OandaClient()
        from environment variables.

    Returns
    -------
    pd.DataFrame
        Canonical OHLCV DataFrame with columns:
        timestamp, symbol, timeframe, open, high, low, close, volume.
        Timestamps are UTC-aware, sorted ascending, no NaN values.

    Raises
    ------
    ValueError
        If symbol is unsupported or timeframe is not recognised.
    OandaAPIError / OandaAuthError
        Propagated from OandaClient on HTTP errors.
    """
    # ── Validate inputs ────────────────────────────────────────────────
    if symbol != "XAUUSD":
        raise ValueError(
            f"fetch_oanda_candles(): unsupported symbol {symbol!r}. "
            f"Only 'XAUUSD' is supported."
        )
    if timeframe not in TIMEFRAME_TO_GRANULARITY:
        raise ValueError(
            f"fetch_oanda_candles(): unrecognised timeframe {timeframe!r}. "
            f"Expected one of {sorted(TIMEFRAME_TO_GRANULARITY)}."
        )

    # ── Resolve defaults ───────────────────────────────────────────────
    candle_count = count if count is not None else PLACEHOLDER_CANDLE_COUNT
    granularity  = TIMEFRAME_TO_GRANULARITY[timeframe]
    instrument   = XAUUSD_INSTRUMENT

    # ── Fetch raw candles ──────────────────────────────────────────────
    if client is None:
        env = os.environ.get("OANDA_ENVIRONMENT", "practice")
        client = OandaClient(environment=env)

    raw_candles = client.fetch_candles(instrument, granularity, candle_count)

    if not raw_candles:
        raise ValueError(
            f"fetch_oanda_candles(): OANDA returned 0 complete candles "
            f"for {symbol} {timeframe}."
        )

    # ── Convert to canonical DataFrame ────────────────────────────────
    return _candles_to_dataframe(raw_candles, symbol, timeframe)


def _candles_to_dataframe(
    candles:   list[dict],
    symbol:    str,
    timeframe: str,
) -> pd.DataFrame:
    """Convert a list of raw OANDA candle dicts to a canonical DataFrame.

    OANDA candle dict shape::

        {
          "time":     "2024-01-01T09:00:00.000000000Z",
          "mid":      {"o": "2000.12", "h": "2005.45",
                       "l": "1999.78", "c": "2003.21"},
          "volume":   1234,
          "complete": True
        }

    Output column order matches REQUIRED_COLUMNS in market_data.py:
        timestamp, symbol, timeframe, open, high, low, close, volume
    """
    rows = []
    for c in candles:
        mid = c.get("mid", {})
        ts  = pd.Timestamp(c["time"])
        # Ensure UTC-aware: tz_convert if already aware, tz_localize if naive
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        rows.append({
            "timestamp": ts,
            "symbol":    symbol,
            "timeframe": timeframe,
            "open":      float(mid["o"]),
            "high":      float(mid["h"]),
            "low":       float(mid["l"]),
            "close":     float(mid["c"]),
            "volume":    float(c.get("volume", 0)),
        })

    df = pd.DataFrame(rows, columns=[
        "timestamp", "symbol", "timeframe",
        "open", "high", "low", "close", "volume",
    ])

    # Sort ascending (OANDA returns oldest-first but ensure it)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
