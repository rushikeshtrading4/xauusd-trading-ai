"""
Centralised configuration for the XAUUSD AI trading system.

All strategy parameters, symbol definitions, and timeframe lists live here
so that the rest of the codebase has a single source of truth.
"""

# ---------------------------------------------------------------------------
# Symbol
# ---------------------------------------------------------------------------

SUPPORTED_SYMBOL: str = "XAUUSD"

# ---------------------------------------------------------------------------
# Timeframes
# ---------------------------------------------------------------------------

# Ordered from highest to lowest resolution — matches the multi-timeframe
# model described in docs/TRADING_STRATEGY.md:
#   Bias   → D1 / H4 / H1
#   Levels → D1 / H4 / H1 / M15
#   Entry  → M5
#   Precision → M1

TIMEFRAMES: list[str] = ["D1", "H4", "H1", "M15", "M5", "M1"]

# ---------------------------------------------------------------------------
# Timeframe frequency aliases (pandas date_range / resample compatible)
# ---------------------------------------------------------------------------

# Maps each timeframe key to its pandas frequency string.  This is the
# single source of truth for candle spacing across the entire codebase.
TIMEFRAME_FREQ: dict[str, str] = {
    "M1":  "1min",
    "M5":  "5min",
    "M15": "15min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1d",
}
