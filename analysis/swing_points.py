"""
Swing point detection module.

Identifies market swing highs and swing lows using a two-stage algorithm:

  Stage 1 — Window pivot detection (window = 5 candles)
      A candle at index i is a *pivot high* when its high is the maximum of
      the five-candle window [i-2, i+2].  Symmetrically, it is a *pivot low*
      when its low is the minimum of that window.

  Stage 2 — ATR displacement confirmation
      Raw pivots are promoted to confirmed swings only when the subsequent
      price movement validates that sufficient momentum existed:

      Swing High: (close[i] - close[i+1]) >= ATR[i] * 0.5
          Price must drop at least half an ATR after the high.

      Swing Low:  (close[i+1] - close[i]) >= ATR[i] * 0.5
          Price must rise at least half an ATR after the low.

Edge handling:
    The first two and last two candles are excluded because the five-candle
    window cannot be fully formed at the boundaries.

Usage::

    from analysis.swing_points import detect_swings
    df_with_swings = detect_swings(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINDOW: int = 5          # Total pivot window width (centred on candle i)
_HALF_WIN: int = 2        # Half-window = (_WINDOW - 1) // 2
_ATR_FACTOR: float = 0.5  # Minimum ATR multiple required for displacement

_REQUIRED_COLUMNS: list[str] = [
    "timestamp", "open", "high", "low", "close", "volume", "ATR",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_swings(df: pd.DataFrame) -> pd.DataFrame:
    """Detect swing highs and swing lows in *df* using pivot + ATR confirmation.

    The function applies a two-stage algorithm:

    **Stage 1 — Window pivots**

    For each candle index ``i`` (where ``2 <= i <= len(df) - 3``):

    * *Pivot High* if ``high[i] == max(high[i-2 : i+3])``
    * *Pivot Low*  if ``low[i]  == min(low[i-2  : i+3])``

    **Stage 2 — Displacement filter**

    A pivot high at ``i`` becomes a *confirmed swing high* only when::

        (close[i] - close[i+1]) >= ATR[i] * 0.5

    A pivot low at ``i`` becomes a *confirmed swing low* only when::

        (close[i+1] - close[i]) >= ATR[i] * 0.5

    The first two and last two rows are always ``False`` because the
    five-candle window cannot be centred there.

    Args:
        df: OHLCV DataFrame that must include an ``ATR`` column.
            Required columns: ``timestamp``, ``open``, ``high``, ``low``,
            ``close``, ``volume``, ``ATR``.

    Returns:
        A copy of *df* with two additional boolean columns appended:

        * ``swing_high`` — ``True`` where a confirmed swing high exists.
        * ``swing_low``  — ``True`` where a confirmed swing low exists.

    Raises:
        ValueError: If any required column is absent from *df*.
        ValueError: If *df* contains fewer than five rows (minimum needed to
            form a single pivot window).
    """
    _validate_input(df)

    out = df.copy()
    n = len(out)

    high = out["high"].to_numpy()
    low  = out["low"].to_numpy()
    close = out["close"].to_numpy()
    atr   = out["ATR"].to_numpy()

    swing_high = np.zeros(n, dtype=bool)
    swing_low  = np.zeros(n, dtype=bool)

    # Stage 1 + 2 combined: iterate over valid interior indices.
    # _HALF_WIN = 2, so valid range is [2, n-3] inclusive.
    for i in range(_HALF_WIN, n - _HALF_WIN):
        window_high = high[i - _HALF_WIN : i + _HALF_WIN + 1]
        window_low  = low[i  - _HALF_WIN : i + _HALF_WIN + 1]

        # --- Pivot High ---
        if high[i] == window_high.max():
            displacement = close[i] - close[i + 1]
            if displacement >= atr[i] * _ATR_FACTOR:
                swing_high[i] = True

        # --- Pivot Low ---
        if low[i] == window_low.min():
            displacement = close[i + 1] - close[i]
            if displacement >= atr[i] * _ATR_FACTOR:
                swing_low[i] = True

    out["swing_high"] = swing_high
    out["swing_low"]  = swing_low

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_input(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if *df* is missing required columns or is too short.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: On missing columns or insufficient row count.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"detect_swings(): DataFrame is missing required column(s): {missing}"
        )

    min_rows = _WINDOW  # need at least 5 rows to form one window
    if len(df) < min_rows:
        raise ValueError(
            f"detect_swings(): DataFrame has {len(df)} row(s); "
            f"at least {min_rows} are required to compute pivot windows."
        )
