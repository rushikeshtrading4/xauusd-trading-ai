"""
Data loader and orchestration module.

Main entry point for market data ingestion. Assembles the standardised
multi-timeframe container that the rest of the pipeline consumes.

Extensibility
-------------
The ``source`` parameter of :func:`load_market_data` controls which backend
is used to fetch candle data.  Currently only ``"placeholder"`` is active;
the following sources are planned for future phases:

- ``"csv"``  — load historical candles from local CSV files.
- ``"api"``  — fetch live / historical data from a broker or data-provider API.

Adding a new source requires:

1. Implementing a ``fetch_<source>_candles(symbol, timeframe)`` function in
   :mod:`data.market_data` that returns a DataFrame conforming to the
   canonical candle schema.
2. Adding a corresponding ``elif source == "<source>"`` branch inside
   :func:`load_market_data`.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SUPPORTED_SYMBOL, TIMEFRAMES
from data.market_data import fetch_placeholder_candles, validate_candle_schema


def load_market_data(
    symbol: str,
    source: str = "placeholder",
) -> dict[str, pd.DataFrame]:
    """Load and validate multi-timeframe candle data for *symbol*.

    Orchestrates data retrieval across all required timeframes defined in
    :mod:`config.settings`, validates each DataFrame against the canonical
    candle schema, and returns a unified container keyed by timeframe.

    Args:
        symbol: Trading instrument to load.  Only ``"XAUUSD"`` is currently
            supported.
        source: Backend to use for candle retrieval.  Accepted values:

            - ``"placeholder"`` *(default)* — deterministic synthetic data,
              suitable for development and unit-testing without a live feed.
            - ``"csv"`` *(not yet implemented)* — historical data from local
              CSV files.
            - ``"api"`` *(not yet implemented)* — live or historical data from
              a broker / data-provider API.

    Returns:
        Dictionary mapping each timeframe string to a validated
        :class:`pandas.DataFrame` with the canonical candle schema::

            {
                "D1":  DataFrame,
                "H4":  DataFrame,
                "H1":  DataFrame,
                "M15": DataFrame,
                "M5":  DataFrame,
                "M1":  DataFrame,
            }

    Raises:
        ValueError: If *symbol* is not supported, or if any timeframe
            DataFrame fails schema validation.
        NotImplementedError: If *source* is recognised but not yet
            implemented (``"csv"``, ``"api"``), or unknown entirely.
    """
    if symbol != SUPPORTED_SYMBOL:
        raise ValueError(
            f"Unsupported symbol '{symbol}'. "
            f"This system supports '{SUPPORTED_SYMBOL}' only."
        )

    multi_timeframe_data: dict[str, pd.DataFrame] = {}

    for timeframe in TIMEFRAMES:
        if source == "placeholder":
            df = fetch_placeholder_candles(symbol, timeframe)
        elif source == "csv":
            raise NotImplementedError("CSV data source not implemented yet.")
        elif source == "api":
            raise NotImplementedError("API data source not implemented yet.")
        else:
            raise ValueError(
                f"Unsupported data source '{source}'. "
                f"Supported sources: 'placeholder', 'csv', 'api'."
            )

        validate_candle_schema(df)
        multi_timeframe_data[timeframe] = df

    return multi_timeframe_data


