"""
Equal highs and equal lows detection module.

Identifies swing points that cluster at nearly the same price level,
forming a *liquidity pool* — an area where stop orders accumulate and
where institutional price action frequently originates.

Two consecutive swing highs (or lows) are considered "equal" when their
distance is within 0.1 × ATR of each other.  This tolerance is adaptive:
it tightens automatically during low-volatility consolidation and widens
during trending, volatile sessions — keeping the detection proportional
to current market conditions.

Pipeline position::

    swing_points → market_structure → liquidity_equal_levels → signal_engine

Public API::

    from analysis.liquidity_equal_levels import detect_equal_levels
    df_with_levels = detect_equal_levels(df)
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Required input columns
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: list[str] = [
    "timestamp", "open", "high", "low", "close", "volume",
    "ATR", "swing_high", "swing_low",
]

# ---------------------------------------------------------------------------
# Equal-level tolerance and buffer size
# ---------------------------------------------------------------------------

# Two swing points are considered equal when their price distance is within
# this fraction of the current ATR.  0.10 is appropriate for XAUUSD intraday
# work: tight enough to avoid merging structurally distinct levels, yet wide
# enough to capture consolidation clusters where price stacks candle tops or
# bottoms within a fraction of a dollar.
_EQUAL_LEVEL_ATR_FACTOR = 0.10

# In Gold intraday price action, equal-level clusters routinely form across
# 2-3 swings separated by an intervening swing at a different level.
# A buffer of 3 ensures those non-adjacent clusters are detected without
# reaching back so far that structurally unrelated swing highs/lows from
# a different market phase are incorrectly merged.  The fixed size keeps
# the per-row comparison cost O(1), so the overall algorithm remains O(N).
_SWING_BUFFER_SIZE = 3


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def detect_equal_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Detect equal highs and equal lows between consecutive swing points.

    Scans the DataFrame in a single forward pass (O(N)).  For each new swing
    high/low the function computes an ATR-proportional tolerance and compares
    the current level against the most recently seen swing of the same type.
    When the distance falls within the tolerance the level is marked as equal.

    Equal highs signal a *bearish* liquidity pool (buy-side liquidity above
    resting highs).  Equal lows signal a *bullish* liquidity pool (sell-side
    liquidity below resting lows).

    Args:
        df: Swing-annotated OHLCV DataFrame.  Must contain all columns listed
            in :data:`_REQUIRED_COLUMNS`.  The ``swing_high`` and
            ``swing_low`` columns must be boolean.

    Returns:
        Copy of *df* with two additional boolean columns:

        * ``equal_high`` — ``True`` when this swing high equals the previous
          swing high within ATR tolerance.
        * ``equal_low``  — ``True`` when this swing low equals the previous
          swing low within ATR tolerance.

    Raises:
        ValueError: If any required column is missing or the DataFrame is empty.
    """
    _validate_input(df)

    out = df.copy().reset_index(drop=True)
    n   = len(out)

    # Pre-extract numpy arrays — avoids per-row Python attribute lookups.
    highs    = out["high"].to_numpy(dtype=float)
    lows     = out["low"].to_numpy(dtype=float)
    atrs     = out["ATR"].to_numpy(dtype=float)
    sw_highs = out["swing_high"].to_numpy(dtype=bool)
    sw_lows  = out["swing_low"].to_numpy(dtype=bool)

    # Output arrays — default False
    equal_highs = np.zeros(n, dtype=bool)
    equal_lows  = np.zeros(n, dtype=bool)

    # Rolling buffers hold the last _SWING_BUFFER_SIZE swing levels.
    # Using a fixed-size deque keeps each comparison O(1) — the inner loop
    # runs at most _SWING_BUFFER_SIZE (3) iterations regardless of N.
    swing_high_buffer: deque[float] = deque(maxlen=_SWING_BUFFER_SIZE)
    swing_low_buffer:  deque[float] = deque(maxlen=_SWING_BUFFER_SIZE)

    for i in range(n):
        atr = atrs[i]

        # ATR safety guard: NaN or non-positive ATR collapses tolerance to
        # zero, meaning only exact matches qualify.  This prevents broken
        # comparisons during the warm-up period of ATR calculation.
        if np.isnan(atr) or atr <= 0:
            tolerance = 0.0
        else:
            tolerance = _EQUAL_LEVEL_ATR_FACTOR * atr

        # ---- Equal high check ----
        if sw_highs[i]:
            current_high = highs[i]
            # Check current swing against all levels in the buffer.
            # Any match within tolerance marks this as an equal high.
            if any(abs(current_high - prev) <= tolerance for prev in swing_high_buffer):
                equal_highs[i] = True
            swing_high_buffer.append(current_high)

        # ---- Equal low check ----
        if sw_lows[i]:
            current_low = lows[i]
            if any(abs(current_low - prev) <= tolerance for prev in swing_low_buffer):
                equal_lows[i] = True
            swing_low_buffer.append(current_low)

    out["equal_high"] = equal_highs
    out["equal_low"]  = equal_lows

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_input(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` for missing columns or an empty DataFrame.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: On missing columns or zero rows.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"detect_equal_levels(): missing required column(s): {missing}"
        )
    if len(df) == 0:
        raise ValueError(
            "detect_equal_levels(): DataFrame must not be empty."
        )
