"""
Fair value gap (FVG) detection module.

A fair value gap is the price imbalance left between candle 1 (c1) and
candle 3 (c3) after candle 2 (c2) produces a strong institutional
displacement.  The unmitigated gap zone represents an area the market is
likely to revisit, making it a high-probability entry area when aligned
with the prevailing trend.

Three-candle structure
-----------------------
  c1  c2  c3

  Bullish FVG: c1.high < c3.low  (gap above c1, below c3)
  Bearish FVG: c1.low  > c3.high (gap below c1, above c3)

Filters applied
---------------
1. **Displacement** — c2 body (``|close − open|``) must exceed
   ``1.5 × ATR``.  Weak candles indicate indecision, not institutional
   order flow.

2. **Gap size** — the gap must be at least ``0.5 × ATR``.  Smaller gaps
   are micro-imbalances with no strategic significance.

3. **Trend alignment** — bullish FVGs are only marked in BULLISH or
   TRANSITION trend states; bearish FVGs in BEARISH or TRANSITION.

4. **Mitigation** — after detection, any FVG whose zone is subsequently
   entered by price is invalidated:

   * Bullish (gap above c1, below c3): invalidated when any future candle
     has ``low <= fvg_high`` (price enters the top of the gap).
   * Bearish (gap below c1, above c3): invalidated when any future candle
     has ``high >= fvg_low`` (price enters the bottom of the gap).

   The mitigation pass is O(N) via pre-computed suffix min/max arrays
   built with ``np.minimum.accumulate`` / ``np.maximum.accumulate``.

Pipeline position::

    market_structure → fair_value_gaps → signal_engine

Public API::

    from analysis.fair_value_gaps import detect_fair_value_gaps
    df_with_fvgs = detect_fair_value_gaps(df)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Required input columns
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: list[str] = [
    "open", "high", "low", "close", "ATR", "trend_state",
    "displacement_bullish", "displacement_bearish",
]

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

# c2 body must exceed this multiple of ATR to confirm institutional displacement.
_DISPLACEMENT_ATR_FACTOR: float = 1.5

# The gap between c1 and c3 must be at least this multiple of ATR to be
# significant.  Smaller gaps are noise.
_MIN_GAP_ATR_FACTOR: float = 0.5

# ATR cap: values above this are clamped to prevent unreasonable SL/sizing.
_ATR_CAP: float = 100.0

# ATR floor: values below this signal warm-up / stale feed artifacts.
_ATR_MIN: float = 0.1

# ---------------------------------------------------------------------------
# Trend state sets (aligned with market_structure constants)
# ---------------------------------------------------------------------------

_BULLISH_STATES: frozenset[str] = frozenset({"BULLISH", "TRANSITION"})
_BEARISH_STATES: frozenset[str] = frozenset({"BEARISH", "TRANSITION"})

# Maximum candle age for an unmitigated FVG (intraday optimized).
MAX_FVG_AGE: int = 20


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def detect_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Detect bullish and bearish fair value gaps from OHLC + structure data.

    Runs a single forward pass (O(N)) for gap detection followed by an O(N)
    mitigation pass using suffix min/max arrays.  Only unmitigated FVGs appear
    in the output.

    Args:
        df: DataFrame containing all columns listed in
            :data:`_REQUIRED_COLUMNS`.  Typically the output of
            :func:`analysis.market_structure.detect_market_structure`.  Any
            additional columns are preserved unchanged in the output.

    Returns:
        A copy of *df* with four additional boolean/float columns:

        * ``fvg_bullish`` — ``True`` on c3 where an active bullish FVG was
          detected (``False`` elsewhere).
        * ``fvg_bearish`` — ``True`` on c3 where an active bearish FVG was
          detected (``False`` elsewhere).
        * ``fvg_low``     — Lower bound of the FVG zone (``NaN`` elsewhere).
        * ``fvg_high``    — Upper bound of the FVG zone (``NaN`` elsewhere).
        * ``fvg_strength`` — Quality score ``min(gap / atr / 3.0, 1.0)``; 0.0
          for every row that is not an active FVG.

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
    trend_states = out["trend_state"].to_numpy(dtype=object)

    fvg_bullish   = np.zeros(n, dtype=bool)
    fvg_bearish   = np.zeros(n, dtype=bool)
    fvg_low       = np.full(n, np.nan, dtype=float)
    fvg_high      = np.full(n, np.nan, dtype=float)
    fvg_strength  = np.zeros(n, dtype=float)
    fvg_created_at = np.full(n, -1, dtype=int)

    disp_bull_col = out["displacement_bullish"].to_numpy(dtype=bool)
    disp_bear_col = out["displacement_bearish"].to_numpy(dtype=bool)

    # ------------------------------------------------------------------
    # Detection pass
    # i is c3; i-1 is c2 (displacement candle); i-2 is c1
    # ------------------------------------------------------------------
    for i in range(2, n):

        # ATR from c2 — must be valid before any other check
        atr = _validate_atr(atrs[i - 1])
        if atr is None:
            continue

        # Displacement filter: c2 body must exceed 1.5 × ATR
        body = abs(closes[i - 1] - opens[i - 1])
        if body <= _DISPLACEMENT_ATR_FACTOR * atr:
            continue

        # Displacement dependency: c2 must carry a confirmed displacement mark
        if not disp_bull_col[i - 1] and not disp_bear_col[i - 1]:
            continue

        c1_high = highs[i - 2]
        c1_low  = lows[i - 2]
        c3_high = highs[i]
        c3_low  = lows[i]
        trend   = trend_states[i]   # current candle's confirmed structure state

        # ------------------------------------------------------------------
        # Bullish FVG: gap above c1, below c3
        # ------------------------------------------------------------------
        if c1_high < c3_low:
            gap = c3_low - c1_high
            if gap < _MIN_GAP_ATR_FACTOR * atr:
                continue
            if trend not in _BULLISH_STATES:
                continue
            fvg_bullish[i]    = True
            fvg_low[i]        = c1_high
            fvg_high[i]       = c3_low
            fvg_strength[i]   = min(gap / atr / 3.0, 1.0)
            fvg_created_at[i] = i

        # ------------------------------------------------------------------
        # Bearish FVG: gap below c1, above c3
        # ------------------------------------------------------------------
        elif c1_low > c3_high:
            gap = c1_low - c3_high
            if gap < _MIN_GAP_ATR_FACTOR * atr:
                continue
            if trend not in _BEARISH_STATES:
                continue
            fvg_bearish[i]    = True
            fvg_low[i]        = c3_high
            fvg_high[i]       = c1_low
            fvg_strength[i]   = min(gap / atr / 3.0, 1.0)
            fvg_created_at[i] = i

    # ------------------------------------------------------------------
    # Age filter — invalidate FVGs older than MAX_FVG_AGE candles.
    # Age is computed as (current_index - fvg_created_at[index]) so the
    # result is deterministic per candle and correct in a streaming context.
    # ------------------------------------------------------------------
    current_index = n - 1

    bull_age_idx = np.where(fvg_bullish)[0]
    aged_bull = bull_age_idx[(current_index - fvg_created_at[bull_age_idx]) > MAX_FVG_AGE]
    fvg_bullish[aged_bull]  = False
    fvg_low[aged_bull]      = np.nan
    fvg_high[aged_bull]     = np.nan
    fvg_strength[aged_bull] = 0.0

    bear_age_idx = np.where(fvg_bearish)[0]
    aged_bear = bear_age_idx[(current_index - fvg_created_at[bear_age_idx]) > MAX_FVG_AGE]
    fvg_bearish[aged_bear]  = False
    fvg_low[aged_bear]      = np.nan
    fvg_high[aged_bear]     = np.nan
    fvg_strength[aged_bear] = 0.0

    # ------------------------------------------------------------------
    # Mitigation pass — O(N) using suffix min/max arrays
    #
    # suffix_min_excl[i] = min(lows[i+1 : n])
    # suffix_max_excl[i] = max(highs[i+1 : n])
    #
    # Bullish FVG at i is mitigated if suffix_min_excl[i] <= fvg_high[i]
    #   (a future candle's low enters the top of the gap).
    # Bearish FVG at i is mitigated if suffix_max_excl[i] >= fvg_low[i]
    #   (a future candle's high enters the bottom of the gap).
    # ------------------------------------------------------------------
    if n > 1:
        # suffix inclusive: suffix_min_incl[i] = min(lows[i:n])
        suffix_min_incl = np.minimum.accumulate(lows[::-1])[::-1]
        suffix_max_incl = np.maximum.accumulate(highs[::-1])[::-1]

        # Shift right by 1 to get exclusive (future-only) suffix arrays
        suffix_min_excl = np.full(n, math.inf,  dtype=float)
        suffix_max_excl = np.full(n, -math.inf, dtype=float)
        suffix_min_excl[:-1] = suffix_min_incl[1:]
        suffix_max_excl[:-1] = suffix_max_incl[1:]

        # --- Invalidate mitigated bullish FVGs ---
        bull_idx = np.where(fvg_bullish)[0]
        mitigated_bull = bull_idx[suffix_min_excl[bull_idx] <= fvg_high[bull_idx]]
        fvg_bullish[mitigated_bull]  = False
        fvg_low[mitigated_bull]      = np.nan
        fvg_high[mitigated_bull]     = np.nan
        fvg_strength[mitigated_bull] = 0.0

        # --- Invalidate mitigated bearish FVGs ---
        bear_idx = np.where(fvg_bearish)[0]
        mitigated_bear = bear_idx[suffix_max_excl[bear_idx] >= fvg_low[bear_idx]]
        fvg_bearish[mitigated_bear]  = False
        fvg_low[mitigated_bear]      = np.nan
        fvg_high[mitigated_bear]     = np.nan
        fvg_strength[mitigated_bear] = 0.0

    out["fvg_bullish"]  = fvg_bullish
    out["fvg_bearish"]  = fvg_bearish
    out["fvg_low"]      = fvg_low
    out["fvg_high"]     = fvg_high
    out["fvg_strength"] = fvg_strength

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_atr(atr: float) -> float | None:
    """Return a usable ATR value, or ``None`` if the value is unusable.

    Rules (identical to the shared contract across signal and risk modules):

    * NaN   → ``None``
    * ≤ 0   → ``None``
    * < 0.1 → ``None``  (warm-up / stale feed artifact)
    * > 100 → capped to 100.0
    """
    if math.isnan(atr):
        return None
    if atr <= 0:
        return None
    if atr < _ATR_MIN:
        return None
    if atr > _ATR_CAP:
        return _ATR_CAP
    return atr


def _validate_input(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` for missing columns or an empty DataFrame."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"detect_fair_value_gaps(): missing required column(s): {missing}"
        )
    if len(df) == 0:
        raise ValueError("detect_fair_value_gaps(): DataFrame must not be empty.")
