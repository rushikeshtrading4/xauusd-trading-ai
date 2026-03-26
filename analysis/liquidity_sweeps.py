"""
Liquidity sweep detection module.

A *liquidity sweep* (also called a stop hunt) occurs when price wicks beyond
a known liquidity pool level but **closes back inside** it within the same
candle.  This wick-and-close pattern is the signature of institutional
accumulation or distribution: market makers drive price into a cluster of
resting stop orders, fill them, and then reverse.

Wick vs close logic
-------------------
The breach is determined by the **wick** (``high`` for buy-side raids,
``low`` for sell-side raids).  The rejection is confirmed by the **close**
returning inside the level.  A close *outside* the level is a breakout or a
CHOCH — a structurally different event handled by ``market_structure.py``.

Only the most recent liquidity level is tracked
------------------------------------------------
The active level is updated each time a new pool row is encountered and
immediately supersedes any older level.  This mirrors how institutional
participants target the freshest, most densely populated cluster of stops
rather than stale levels that have already been approached.

Stop-hunt sequence modelled here::

    1. ``liquidity_pool_high / low`` established by ``liquidity.py``
    2. Active level updated to the pool candle's high / low
    3. Subsequent candle wicks past the level → stops triggered
    4. Close snaps back inside → reversal intent confirmed
    5. ``liquidity_sweep_high / low = True`` marks the sweep candle

Pipeline position::

    liquidity → liquidity_sweeps → signal_engine

Public API::

    from analysis.liquidity_sweeps import detect_liquidity_sweeps
    df_with_sweeps = detect_liquidity_sweeps(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Required input columns
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: list[str] = [
    "high", "low", "close",
    "liquidity_pool_high", "liquidity_pool_low",
]


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def detect_liquidity_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    """Detect wick-based liquidity sweeps in a single forward pass (O(N)).

    For each row the function first checks whether a new liquidity pool starts
    (updating the active level if so), then tests the wick-breach + close-
    rejection conditions.

    Args:
        df: DataFrame containing all columns listed in
            :data:`_REQUIRED_COLUMNS`.  Typically the output of
            :func:`analysis.liquidity.detect_liquidity_pools`.  Any
            additional columns are preserved unchanged in the output.

    Returns:
        A copy of *df* with two additional boolean columns:

        * ``liquidity_sweep_high`` — ``True`` on the candle that wicks above
          the active buy-side pool level and closes back below it.
        * ``liquidity_sweep_low``  — ``True`` on the candle that wicks below
          the active sell-side pool level and closes back above it.

    Raises:
        ValueError: If any required column is missing or the DataFrame is
            empty.
    """
    _validate_input(df)

    out = df.copy().reset_index(drop=True)
    n   = len(out)

    highs      = out["high"].to_numpy(dtype=float)
    lows       = out["low"].to_numpy(dtype=float)
    closes     = out["close"].to_numpy(dtype=float)
    pool_highs = out["liquidity_pool_high"].to_numpy(dtype=bool)
    pool_lows  = out["liquidity_pool_low"].to_numpy(dtype=bool)

    sweep_high = np.zeros(n, dtype=bool)
    sweep_low  = np.zeros(n, dtype=bool)

    active_high_level: float | None = None
    active_low_level:  float | None = None

    for i in range(n):

        # ---- Update active levels from pool rows ----
        # Pool rows are processed before the sweep check so that a candle
        # which both establishes a pool AND immediately sweeps it in the same
        # bar cannot trigger a self-referential sweep (its own high == the
        # level it just set, so high > level is always False).
        if pool_highs[i]:
            active_high_level = highs[i]

        if pool_lows[i]:
            active_low_level = lows[i]

        # ---- Bearish sweep: wick above buy-side pool, close back inside ----
        if active_high_level is not None:
            if highs[i] > active_high_level and closes[i] < active_high_level:
                sweep_high[i] = True

        # ---- Bullish sweep: wick below sell-side pool, close back inside ----
        if active_low_level is not None:
            if lows[i] < active_low_level and closes[i] > active_low_level:
                sweep_low[i] = True

    out["liquidity_sweep_high"] = sweep_high
    out["liquidity_sweep_low"]  = sweep_low

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_input(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` for missing columns or an empty DataFrame."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"detect_liquidity_sweeps(): missing required column(s): {missing}"
        )
    if len(df) == 0:
        raise ValueError("detect_liquidity_sweeps(): DataFrame must not be empty.")
