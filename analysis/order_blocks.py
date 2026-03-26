"""
Order block detection module.

An *order block* (OB) is the last opposite-direction candle before a
displacement move that follows a liquidity sweep.  It represents the price
zone where institutional participants placed their orders; when price revisits
that zone it is the highest-probability entry point in the strategy.

Sweep → displacement → order block sequence
--------------------------------------------
The detection pipeline requires three consecutive events:

1. **Liquidity sweep** — price raids a stop cluster and closes back inside
   (detected by ``liquidity_sweeps.py``).  This is the trigger that arms
   the OB search.

2. **Displacement candle** — the first candle after the sweep whose body
   (``|close − open|``) exceeds ``_DISPLACEMENT_ATR_FACTOR × ATR``.  Only
   strongly-bodied candles qualify: weak candles indicate indecision, not
   institutional intent.  A bearish displacement follows a high sweep; a
   bullish displacement follows a low sweep.

3. **Order block** — the last candle of the opposite direction that appeared
   between the sweep and the displacement.  For a bearish OB (sell setup) this
   is the last *bullish* candle before the bearish impulse; for a bullish OB
   (buy setup) it is the last *bearish* candle before the bullish impulse.
   The OB captures the candle where institutions quietly absorbed supply
   (bearish OB) or demand (bullish OB) before driving price aggressively.

Why displacement is required
-----------------------------
Without a displacement filter any large candle or random move could be
misread as institutional intent.  Requiring the body to exceed 1.5 × ATR
ensures only genuine impulse moves — driven by real order flow — qualify.
This keeps the OB signal high-conviction and reduces noise.

Why the *last* opposite candle
--------------------------------
Institutions typically place their final absorbing orders in the candle
immediately before the impulse.  Earlier candles in the same direction are
already inside the move; the edge is at the boundary.  Using the last
opposite candle maximises entry precision and minimises stop size.

Pipeline position::

    liquidity_sweeps → order_blocks → signal_engine

Public API::

    from analysis.order_blocks import detect_order_blocks
    df_with_obs = detect_order_blocks(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Required input columns
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: list[str] = [
    "open", "high", "low", "close", "ATR",
    "liquidity_sweep_high", "liquidity_sweep_low",
]

# ---------------------------------------------------------------------------
# Displacement filter
# ---------------------------------------------------------------------------

# A displacement candle must have a body larger than this multiple of ATR.
# 1.5 × ATR is the institutional threshold for Gold intraday — strong enough
# to confirm genuine order flow, loose enough to capture fast-moving sessions.
_DISPLACEMENT_ATR_FACTOR = 1.5


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Detect bullish and bearish order blocks from sweep + displacement data.

    Makes a single forward pass (O(N)).  Two independent state machines run in
    parallel — one for each sweep direction:

    * **Bearish OB state machine** — armed by ``liquidity_sweep_high``.  Tracks
      the most recent bullish candle while waiting for a bearish displacement.
      When displacement arrives, the tracked candle is marked as a bearish OB.

    * **Bullish OB state machine** — armed by ``liquidity_sweep_low``.  Tracks
      the most recent bearish candle while waiting for a bullish displacement.
      When displacement arrives, the tracked candle is marked as a bullish OB.

    A new sweep on the same side resets the corresponding state machine,
    ensuring only the most recent sweep drives OB discovery.

    Args:
        df: DataFrame containing all columns listed in
            :data:`_REQUIRED_COLUMNS`.  Typically the output of
            :func:`analysis.liquidity_sweeps.detect_liquidity_sweeps`.  Any
            additional columns are preserved unchanged in the output.

    Returns:
        A copy of *df* with four additional columns:

        * ``bullish_order_block``  — ``True`` on the candle identified as a
          bullish OB (last bearish candle before bullish displacement after
          a low sweep).
        * ``bearish_order_block``  — ``True`` on the candle identified as a
          bearish OB (last bullish candle before bearish displacement after
          a high sweep).
        * ``ob_high``              — High of the OB candle (``NaN`` elsewhere).
        * ``ob_low``               — Low of the OB candle (``NaN`` elsewhere).

    Raises:
        ValueError: If any required column is missing or the DataFrame is
            empty.
    """
    _validate_input(df)

    out = df.copy().reset_index(drop=True)
    n   = len(out)

    opens        = out["open"].to_numpy(dtype=float)
    highs        = out["high"].to_numpy(dtype=float)
    lows         = out["low"].to_numpy(dtype=float)
    closes       = out["close"].to_numpy(dtype=float)
    atrs         = out["ATR"].to_numpy(dtype=float)
    sw_highs     = out["liquidity_sweep_high"].to_numpy(dtype=bool)
    sw_lows      = out["liquidity_sweep_low"].to_numpy(dtype=bool)

    bullish_ob = np.zeros(n, dtype=bool)
    bearish_ob = np.zeros(n, dtype=bool)
    ob_high    = np.full(n, np.nan, dtype=float)
    ob_low     = np.full(n, np.nan, dtype=float)

    # ---- Bearish OB state (armed by high sweep) ----
    pending_bearish         = False   # waiting for bearish displacement
    last_bullish_idx: int | None = None  # most recent bullish candle since arm

    # ---- Bullish OB state (armed by low sweep) ----
    pending_bullish         = False
    last_bearish_idx: int | None = None

    for i in range(n):

        # ------------------------------------------------------------------
        # 1. Arm / re-arm state machines on new sweeps (resets tracking)
        # ------------------------------------------------------------------
        if sw_highs[i]:
            pending_bearish  = True
            last_bullish_idx = None

        if sw_lows[i]:
            pending_bullish  = True
            last_bearish_idx = None

        # ------------------------------------------------------------------
        # 2. Candle classification and body size
        # ------------------------------------------------------------------
        is_bullish = closes[i] > opens[i]
        is_bearish = closes[i] < opens[i]
        body = abs(closes[i] - opens[i])

        atr = atrs[i]
        valid_atr   = not np.isnan(atr) and atr > 0
        is_disp     = valid_atr and body > _DISPLACEMENT_ATR_FACTOR * atr

        # ------------------------------------------------------------------
        # 3. Bearish OB search
        # ------------------------------------------------------------------
        if pending_bearish:

            # Track the most recent bullish candle (OB candidate)
            if is_bullish:
                last_bullish_idx = i

            # Bearish displacement found?
            if is_bearish and is_disp:
                if last_bullish_idx is not None:
                    j = last_bullish_idx
                    bearish_ob[j] = True
                    ob_high[j]    = highs[j]
                    ob_low[j]     = lows[j]
                pending_bearish  = False
                last_bullish_idx = None

        # ------------------------------------------------------------------
        # 4. Bullish OB search
        # ------------------------------------------------------------------
        if pending_bullish:

            # Track the most recent bearish candle (OB candidate)
            if is_bearish:
                last_bearish_idx = i

            # Bullish displacement found?
            if is_bullish and is_disp:
                if last_bearish_idx is not None:
                    j = last_bearish_idx
                    bullish_ob[j] = True
                    ob_high[j]    = highs[j]
                    ob_low[j]     = lows[j]
                pending_bullish  = False
                last_bearish_idx = None

    out["bullish_order_block"] = bullish_ob
    out["bearish_order_block"] = bearish_ob
    out["ob_high"]             = ob_high
    out["ob_low"]              = ob_low

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_input(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` for missing columns or an empty DataFrame."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"detect_order_blocks(): missing required column(s): {missing}"
        )
    if len(df) == 0:
        raise ValueError("detect_order_blocks(): DataFrame must not be empty.")
