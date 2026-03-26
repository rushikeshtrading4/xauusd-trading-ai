"""
Displacement detection module.

A *displacement* is a sequence of consecutive same-direction candles that
collectively exhibit institutional-grade momentum — strong bodies, a large
net price move, and minimal wick rejection.  It is the engine behind order
blocks and fair value gaps: without a genuine displacement there is no
institutional fingerprint to trade from.

Multi-candle window model
--------------------------
Rather than checking a single candle, this module examines a rolling window of
``window`` consecutive candles (default 3).  A displacement is marked on the
*last* candle of the qualifying window.  Using three candles matches common
institutional descriptions:

  "Three consecutive bullish/bearish candles with expanding bodies and no
   meaningful wick rejection signal smart-money accumulation/distribution."

Five sequential filters
-----------------------
All five must pass; the first failure skips the entire window:

1. **ATR validation** — the reference ATR (last candle) must be ≥ 0.1 and
   finite.
2. **Direction uniformity** — every candle in the window must close in the
   same direction (all bullish or all bearish).
3. **Body strength** — mean body ``> 0.8 × ATR``.  Weak candles are noise.
4. **Total move** — net price travel ``> 2.0 × ATR``.  Ensures the sequence
   covers meaningful distance, not just three tiny candles.
5. **Wick filter** — for every candle the dominant wick must not exceed the
   body.  Wicks larger than the body signal rejection and disqualify the move.

Strength score
--------------
A 0–1 score blends three components:

  raw = (avg_body / atr) × 0.4 + (move / atr) × 0.4 + (window / 3) × 0.2

The raw score is directly bounded via ``min(raw / 5.0, 1.0)``, giving a
session-invariant quality metric.  Windows that failed keep a score of 0.0.

Pipeline position::

    liquidity_sweeps → displacement → order_blocks → signal_engine

Public API::

    from analysis.displacement import detect_displacement
    df_with_disp = detect_displacement(df)
    df_with_disp = detect_displacement(df, window=5)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Required input columns
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: list[str] = [
    "open", "high", "low", "close", "ATR",
]

# ---------------------------------------------------------------------------
# Filter thresholds (mirrors values used in order_blocks / fair_value_gaps)
# ---------------------------------------------------------------------------

# Mean body of window candles must exceed this multiple of ATR.
_BODY_ATR_FACTOR: float = 0.8

# Net move across the window must exceed this multiple of ATR.
_MOVE_ATR_FACTOR: float = 2.0

# ATR boundaries (consistent with shared _validate_atr contract)
_ATR_MIN: float  = 0.1
_ATR_CAP: float  = 100.0


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def detect_displacement(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Detect bullish and bearish institutional displacement sequences.

    Scans a rolling window of *window* consecutive candles.  A displacement
    is marked on the last candle of each qualifying window after all five
    filters pass.  After detection, each qualifying window is assigned a
    bounded strength ``min(raw / 5.0, 1.0)``.

    Args:
        df:     DataFrame containing all columns listed in
                :data:`_REQUIRED_COLUMNS`.  Additional columns are preserved.
        window: Number of consecutive candles that must all qualify.
                Must be ≥ 1.  Defaults to 3.

    Returns:
        A copy of *df* with three additional columns:

        * ``displacement_bullish``  — ``True`` on the last candle of a
          qualifying bullish displacement window.
        * ``displacement_bearish``  — ``True`` on the last candle of a
          qualifying bearish displacement window.
        * ``displacement_strength`` — Normalised 0–1 quality score; 0.0 for
          every row that is not a displacement.

    Raises:
        ValueError: If any required column is missing, the DataFrame is empty,
            or *window* < 1.
    """
    _validate_input(df, window)

    out = df.copy().reset_index(drop=True)
    n   = len(out)

    opens  = out["open"].to_numpy(dtype=float)
    highs  = out["high"].to_numpy(dtype=float)
    lows   = out["low"].to_numpy(dtype=float)
    closes = out["close"].to_numpy(dtype=float)
    atrs   = out["ATR"].to_numpy(dtype=float)

    disp_bull     = np.zeros(n, dtype=bool)
    disp_bear     = np.zeros(n, dtype=bool)
    disp_strength = np.zeros(n, dtype=float)

    # Pre-compute per-candle quantities used in the inner checks.
    bodies      = np.abs(closes - opens)          # body size
    upper_wicks = highs - np.maximum(opens, closes)
    lower_wicks = np.minimum(opens, closes) - lows
    dom_wicks   = np.maximum(upper_wicks, lower_wicks)  # dominant wick per candle

    # -----------------------------------------------------------------------
    # Detection pass — i is the *last* candle index of the window [i-window, i)
    # In the spec this is written as df[i-window:i], i.e. the window ends at
    # index i-1.  We mark the result on index i-1 (last candle in slice).
    # -----------------------------------------------------------------------
    for i in range(window, n + 1):
        start = i - window       # inclusive
        end   = i                # exclusive  →  candles: start … end-1
        last  = end - 1          # index of last candle in window

        # ------------------------------------------------------------------
        # STEP 0 — ATR validation (use last candle's ATR as reference)
        # ------------------------------------------------------------------
        atr = _validate_atr(atrs[last])
        if atr is None:
            continue

        # ------------------------------------------------------------------
        # STEP 1 — Direction uniformity
        # ------------------------------------------------------------------
        win_closes = closes[start:end]
        win_opens  = opens[start:end]
        is_bull_candle = win_closes > win_opens
        is_bear_candle = win_closes < win_opens

        bullish = bool(np.all(is_bull_candle))
        bearish = bool(np.all(is_bear_candle))

        if not bullish and not bearish:
            continue

        # ------------------------------------------------------------------
        # STEP 2 — Body strength
        # ------------------------------------------------------------------
        win_bodies  = bodies[start:end]
        avg_body    = float(np.mean(win_bodies))

        if avg_body <= _BODY_ATR_FACTOR * atr:
            continue

        # ------------------------------------------------------------------
        # STEP 3 — Total move
        # ------------------------------------------------------------------
        first_open = float(opens[start])
        last_close = float(closes[last])

        if bullish:
            move = last_close - first_open
        else:
            move = first_open - last_close

        if move <= _MOVE_ATR_FACTOR * atr:
            continue

        # ------------------------------------------------------------------
        # STEP 4 — Wick filter (vectorised)
        # ------------------------------------------------------------------
        win_bodies_chk = win_bodies
        win_dom_wicks  = dom_wicks[start:end]
        wick_ratio = win_dom_wicks / (win_bodies_chk + 1e-9)

        if np.any(wick_ratio > 0.8):
            continue

        # ------------------------------------------------------------------
        # STEP 5 — Mark displacement on last candle of window
        # ------------------------------------------------------------------
        if bullish:
            disp_bull[last] = True
        else:
            disp_bear[last] = True

        # ------------------------------------------------------------------
        # STEP 6 — Strength score (directly bounded)
        # ------------------------------------------------------------------
        raw = (
            (avg_body / atr) * 0.4
            + (move    / atr) * 0.4
            + min(window / 3.0, 1.0) * 0.2
        )
        # A candle can be the last of multiple overlapping windows; keep the
        # highest score encountered for that index.
        strength = min(raw / 5.0, 1.0)
        if strength > disp_strength[last]:
            disp_strength[last] = strength

    out["displacement_bullish"]  = disp_bull
    out["displacement_bearish"]  = disp_bear
    out["displacement_strength"] = disp_strength

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_atr(atr: float) -> float | None:
    """Return a usable ATR value, or ``None`` if the value is unusable.

    Contract is identical to the shared rule used across signal, risk, and
    analysis modules:

    * NaN   → ``None``
    * ≤ 0   → ``None``
    * < 0.1 → ``None``
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


def _validate_input(df: pd.DataFrame, window: int) -> None:
    """Raise ``ValueError`` for missing columns, empty DataFrame, or bad window."""
    if window < 1:
        raise ValueError(
            f"detect_displacement(): window must be ≥ 1, got {window}."
        )
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"detect_displacement(): missing required column(s): {missing}"
        )
    if len(df) == 0:
        raise ValueError("detect_displacement(): DataFrame must not be empty.")
