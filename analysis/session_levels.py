"""Session level detection for XAUUSD intraday trading.

Identifies the running and prior-completed high/low of the three main trading
sessions (Asia, London, New York) using a single O(N) forward pass over a
UTC-timestamped OHLCV DataFrame.  No lookahead, no resampling, no groupby.

Session Windows (UTC, half-open [start, end)):
- Asia:    00:00 – 08:00
- London:  07:00 – 16:00
- NY:      12:00 – 21:00
- OFF:     21:00 – 24:00

Overlap Priority (for ``session_label`` and ``in_*`` booleans):
  London > NY > Asia
- 07:00–08:00 candles → LONDON  (in_london=True, in_asia=False)
- 12:00–16:00 candles → LONDON  (in_london=True, in_ny=False)
- 16:00–21:00 candles → NY only

The ``session_*_high/low`` active tracking columns use raw window membership
so Asia, London, and NY can each accumulate their running range independently
of the priority label.

State Machine Rules
-------------------
Each session maintains an independent state via :class:`_SessionState`.

* When the **first candle of a session** is encountered for a given date:
    - Seal the current instance if still marked active
      (handles data gaps that skip the normal window-exit candle).
    - Start a new instance: seed ``high`` / ``low`` from the current candle.
* While **active**: extend running ``high`` (max) and ``low`` (min).
* When a candle **leaves the session window** while active:
    - Seal the instance (update the ``prev_*`` sealed levels).
    - Mark session as inactive.

``prev_*`` columns carry the most recently sealed level and do not change
until the next seal of that session.

Public API::

    from analysis.session_levels import (
        detect_session_levels,
        get_session_liquidity_targets,
        get_current_session_range,
        classify_price_vs_session,
    )
    result = detect_session_levels(df)
"""

from __future__ import annotations

import datetime
import math
from datetime import time
from typing import NamedTuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Exported label / bias constants
# ---------------------------------------------------------------------------

LABEL_ASIA   = "ASIA"
LABEL_LONDON = "LONDON"
LABEL_NY     = "NY"
LABEL_OFF    = "OFF"

BIAS_BULLISH = "BULLISH"
BIAS_BEARISH = "BEARISH"
BIAS_NEUTRAL = "NEUTRAL"

# ---------------------------------------------------------------------------
# Session window definitions (UTC, half-open [start, end))
# ---------------------------------------------------------------------------

_ASIA_START   = time(0,  0)
_ASIA_END     = time(8,  0)
_LONDON_START = time(7,  0)
_LONDON_END   = time(16, 0)
_NY_START     = time(12, 0)
_NY_END       = time(21, 0)

# ---------------------------------------------------------------------------
# Required input columns
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: list[str] = ["timestamp", "open", "high", "low", "close"]

# ---------------------------------------------------------------------------
# Internal per-session state (immutable NamedTuple; update via _replace)
# ---------------------------------------------------------------------------


class _SessionState(NamedTuple):
    """Immutable snapshot of one session's running state."""

    active: bool = False
    high: float = float("nan")
    low: float = float("nan")
    date: datetime.date | None = None
    sealed_high: float = float("nan")
    sealed_low: float = float("nan")


# ---------------------------------------------------------------------------
# Output column name groups
# ---------------------------------------------------------------------------

_ACTIVE_COLS: list[str] = [
    "session_asia_high",   "session_asia_low",
    "session_london_high", "session_london_low",
    "session_ny_high",     "session_ny_low",
]
_PREV_COLS: list[str] = [
    "prev_asia_high",   "prev_asia_low",
    "prev_london_high", "prev_london_low",
    "prev_ny_high",     "prev_ny_low",
]
_BOOL_COLS:  list[str] = ["in_asia", "in_london", "in_ny"]
_LABEL_COLS: list[str] = ["session_label", "session_bias"]

SESSION_OUTPUT_COLUMNS: list[str] = _ACTIVE_COLS + _PREV_COLS + _BOOL_COLS + _LABEL_COLS

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _in_window(t: time, start: time, end: time) -> bool:
    """Return ``True`` when *t* lies in the half-open interval [start, end)."""
    return start <= t < end


def _validate_input(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` for bad input.

    Checks (in order):
    1. *df* must be a ``pd.DataFrame``.
    2. All :data:`_REQUIRED_COLUMNS` must be present (names the first missing).
    3. DataFrame must not be empty.
    4. ``timestamp`` column must be timezone-aware.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"detect_session_levels expects a pd.DataFrame, got {type(df).__name__}."
        )
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}.  Required: {_REQUIRED_COLUMNS}."
        )
    if len(df) == 0:
        raise ValueError(
            "detect_session_levels: DataFrame must not be empty."
        )
    ts_series = pd.to_datetime(df["timestamp"])
    if ts_series.dt.tz is None:
        raise ValueError(
            "timestamp column must be timezone-aware UTC.  "
            "Use df['timestamp'].dt.tz_localize('UTC') to convert naive timestamps."
        )


# ---------------------------------------------------------------------------
# Internal state-machine helpers
# ---------------------------------------------------------------------------


def _seal(state: _SessionState, h: float, lo: float) -> _SessionState:
    """Seal the active instance: copy running h/l into sealed fields, go inactive."""
    return state._replace(sealed_high=h, sealed_low=lo, active=False)


def _new_instance(state: _SessionState, h: float, lo: float,
                  d: datetime.date) -> _SessionState:
    """Start a fresh session instance from the current candle, sealing if needed."""
    if state.active and not math.isnan(state.high):
        # Seal the outgoing instance before starting fresh.
        state = state._replace(sealed_high=state.high, sealed_low=state.low)
    return state._replace(active=True, high=h, low=lo, date=d)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_session_levels(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """Append 17 session-derived columns to a UTC-timestamped OHLCV DataFrame.

    Single O(N) forward pass without lookahead.  Input must be sorted in
    ascending chronological order.  Timestamps must be timezone-aware UTC.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.  Required columns: ``timestamp``, ``open``,
        ``high``, ``low``, ``close``.
    tz : str
        Timezone label (informational, not used for conversion).  Timestamps
        must already be tz-aware before calling this function.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with 17 additional columns.

        Running session levels (``NaN`` when outside the raw session window)::

            session_asia_high    session_asia_low
            session_london_high  session_london_low
            session_ny_high      session_ny_low

        Most recently sealed session levels (``NaN`` until first seal)::

            prev_asia_high    prev_asia_low
            prev_london_high  prev_london_low
            prev_ny_high      prev_ny_low

        Boolean session membership (mutually exclusive, priority-based)::

            in_asia    in_london    in_ny

        Derived fields::

            session_label  — "ASIA" / "LONDON" / "NY" / "OFF"
            session_bias   — "BULLISH" / "BEARISH" / "NEUTRAL"

    Raises
    ------
    ValueError
        If *df* is not a ``pd.DataFrame``, is missing required columns,
        is empty, or has timezone-naive timestamps.
    """
    _validate_input(df)

    out = df.copy().reset_index(drop=True)
    n   = len(out)

    timestamps = pd.to_datetime(out["timestamp"])
    nan        = float("nan")

    # ---------------------------------------------------------------------- #
    # Pre-allocate output arrays                                              #
    # ---------------------------------------------------------------------- #

    sa_hi = np.full(n, nan)
    sa_lo = np.full(n, nan)
    sl_hi = np.full(n, nan)
    sl_lo = np.full(n, nan)
    sn_hi = np.full(n, nan)
    sn_lo = np.full(n, nan)

    pa_hi = np.full(n, nan)
    pa_lo = np.full(n, nan)
    pl_hi = np.full(n, nan)
    pl_lo = np.full(n, nan)
    pn_hi = np.full(n, nan)
    pn_lo = np.full(n, nan)

    in_a = np.zeros(n, dtype=bool)
    in_l = np.zeros(n, dtype=bool)
    in_n = np.zeros(n, dtype=bool)

    labels = np.empty(n, dtype=object)
    biases = np.empty(n, dtype=object)

    # ---------------------------------------------------------------------- #
    # Session state (NamedTuple, updated via _replace / helpers)             #
    # ---------------------------------------------------------------------- #

    asia   = _SessionState()
    london = _SessionState()
    ny     = _SessionState()

    highs = out["high"].to_numpy(dtype=float)
    lows  = out["low"].to_numpy(dtype=float)

    # ---------------------------------------------------------------------- #
    # Forward pass                                                            #
    # ---------------------------------------------------------------------- #

    for i in range(n):
        ts = timestamps.iloc[i]
        t  = ts.time()
        d  = ts.date()
        h  = highs[i]
        lo = lows[i]

        # ------------------------------------------------------------------ #
        # 1. Raw window membership (used by state machines and active cols)   #
        # ------------------------------------------------------------------ #

        is_asia_win   = _in_window(t, _ASIA_START,   _ASIA_END)
        is_london_win = _in_window(t, _LONDON_START, _LONDON_END)
        is_ny_win     = _in_window(t, _NY_START,     _NY_END)

        # ------------------------------------------------------------------ #
        # 2. Asia state machine                                               #
        # ------------------------------------------------------------------ #

        if is_asia_win:
            if not asia.active or d != asia.date:
                asia = _new_instance(asia, h, lo, d)
            else:
                asia = asia._replace(high=max(asia.high, h), low=min(asia.low, lo))
        else:
            if asia.active:
                asia = _seal(asia, asia.high, asia.low)

        if asia.active:
            sa_hi[i] = asia.high
            sa_lo[i] = asia.low
        pa_hi[i] = asia.sealed_high
        pa_lo[i] = asia.sealed_low

        # ------------------------------------------------------------------ #
        # 3. London state machine                                             #
        # ------------------------------------------------------------------ #

        if is_london_win:
            if not london.active or d != london.date:
                london = _new_instance(london, h, lo, d)
            else:
                london = london._replace(
                    high=max(london.high, h), low=min(london.low, lo)
                )
        else:
            if london.active:
                london = _seal(london, london.high, london.low)

        if london.active:
            sl_hi[i] = london.high
            sl_lo[i] = london.low
        pl_hi[i] = london.sealed_high
        pl_lo[i] = london.sealed_low

        # ------------------------------------------------------------------ #
        # 4. NY state machine                                                 #
        # ------------------------------------------------------------------ #

        if is_ny_win:
            if not ny.active or d != ny.date:
                ny = _new_instance(ny, h, lo, d)
            else:
                ny = ny._replace(high=max(ny.high, h), low=min(ny.low, lo))
        else:
            if ny.active:
                ny = _seal(ny, ny.high, ny.low)

        if ny.active:
            sn_hi[i] = ny.high
            sn_lo[i] = ny.low
        pn_hi[i] = ny.sealed_high
        pn_lo[i] = ny.sealed_low

        # ------------------------------------------------------------------ #
        # 5. Priority label  (London > NY > Asia > OFF)                      #
        # ------------------------------------------------------------------ #

        if is_london_win:
            label = LABEL_LONDON
        elif is_ny_win:
            label = LABEL_NY
        elif is_asia_win:
            label = LABEL_ASIA
        else:
            label = LABEL_OFF
        labels[i] = label

        # ------------------------------------------------------------------ #
        # 6. Boolean membership — priority-based, mutually exclusive          #
        # ------------------------------------------------------------------ #

        in_a[i] = (label == LABEL_ASIA)
        in_l[i] = (label == LABEL_LONDON)
        in_n[i] = (label == LABEL_NY)

        # ------------------------------------------------------------------ #
        # 7. Session bias  (London only; requires prior completed Asia)       #
        # ------------------------------------------------------------------ #

        if is_london_win and london.active:
            if not (math.isnan(asia.sealed_high) or math.isnan(asia.sealed_low)):
                prior_asia_mid = (asia.sealed_high + asia.sealed_low) / 2.0
                london_mid     = (london.high + london.low) / 2.0
                if london_mid > prior_asia_mid:
                    biases[i] = BIAS_BULLISH
                elif london_mid < prior_asia_mid:
                    biases[i] = BIAS_BEARISH
                else:
                    biases[i] = BIAS_NEUTRAL
            else:
                biases[i] = BIAS_NEUTRAL
        else:
            biases[i] = BIAS_NEUTRAL

    # ---------------------------------------------------------------------- #
    # Assign output columns                                                   #
    # ---------------------------------------------------------------------- #

    out["session_asia_high"]   = sa_hi
    out["session_asia_low"]    = sa_lo
    out["session_london_high"] = sl_hi
    out["session_london_low"]  = sl_lo
    out["session_ny_high"]     = sn_hi
    out["session_ny_low"]      = sn_lo

    out["prev_asia_high"]   = pa_hi
    out["prev_asia_low"]    = pa_lo
    out["prev_london_high"] = pl_hi
    out["prev_london_low"]  = pl_lo
    out["prev_ny_high"]     = pn_hi
    out["prev_ny_low"]      = pn_lo

    out["in_asia"]   = in_a
    out["in_london"] = in_l
    out["in_ny"]     = in_n

    out["session_label"] = labels
    out["session_bias"]  = biases

    return out


def get_session_liquidity_targets(df: pd.DataFrame) -> dict[str, float]:
    """Read the last row of a :func:`detect_session_levels` output DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the 17 session columns produced by
        :func:`detect_session_levels`.

    Returns
    -------
    dict
        Keys: ``prev_asia_high``, ``prev_asia_low``, ``prev_london_high``,
        ``prev_london_low``, ``prev_ny_high``, ``prev_ny_low``,
        ``session_label``, ``session_bias``.

    Raises
    ------
    ValueError
        If any expected session output column is absent.
    """
    required = [
        "prev_asia_high", "prev_asia_low",
        "prev_london_high", "prev_london_low",
        "prev_ny_high", "prev_ny_low",
        "session_label", "session_bias",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"get_session_liquidity_targets: missing columns {missing}.  "
            "Pass the output of detect_session_levels()."
        )
    row = df.iloc[-1]
    return {
        "prev_asia_high":   float(row["prev_asia_high"]),
        "prev_asia_low":    float(row["prev_asia_low"]),
        "prev_london_high": float(row["prev_london_high"]),
        "prev_london_low":  float(row["prev_london_low"]),
        "prev_ny_high":     float(row["prev_ny_high"]),
        "prev_ny_low":      float(row["prev_ny_low"]),
        "session_label":    str(row["session_label"]),
        "session_bias":     str(row["session_bias"]),
    }


def get_current_session_range(df: pd.DataFrame) -> dict[str, float | str]:
    """Return the active session label, running high, and running low from the
    last row of a :func:`detect_session_levels` DataFrame.

    During an OFF candle both ``high`` and ``low`` are NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`detect_session_levels`.

    Returns
    -------
    dict
        Keys: ``label`` (str), ``high`` (float), ``low`` (float).
    """
    row   = df.iloc[-1]
    label = str(row["session_label"])

    _active_col_map: dict[str, tuple[str, str]] = {
        LABEL_ASIA:   ("session_asia_high",   "session_asia_low"),
        LABEL_LONDON: ("session_london_high", "session_london_low"),
        LABEL_NY:     ("session_ny_high",     "session_ny_low"),
    }

    if label in _active_col_map:
        hi_col, lo_col = _active_col_map[label]
        return {
            "label": label,
            "high":  float(row[hi_col]),
            "low":   float(row[lo_col]),
        }
    return {"label": label, "high": float("nan"), "low": float("nan")}


def classify_price_vs_session(
    price: float,
    session_df: pd.DataFrame,
    session: str = "ASIA",
) -> str:
    """Classify *price* relative to the prior completed session range.

    Parameters
    ----------
    price : float
        The price level to classify.
    session_df : pd.DataFrame
        Output of :func:`detect_session_levels`.
    session : str
        One of ``"ASIA"``, ``"LONDON"``, ``"NY"`` (case-insensitive).

    Returns
    -------
    str
        ``"ABOVE"`` / ``"INSIDE"`` / ``"BELOW"`` / ``"UNKNOWN"``.
        ``"UNKNOWN"`` is returned when no sealed data exists yet (NaN).

    Raises
    ------
    ValueError
        If *session* is not one of the valid values.
    """
    session_upper = session.upper()
    if session_upper not in {"ASIA", "LONDON", "NY"}:
        raise ValueError(
            f"classify_price_vs_session: invalid session {session!r}.  "
            "Must be 'ASIA', 'LONDON', or 'NY'."
        )

    session_lower = session_upper.lower()
    hi_col = f"prev_{session_lower}_high"
    lo_col = f"prev_{session_lower}_low"

    row = session_df.iloc[-1]
    hi  = float(row[hi_col])
    lo  = float(row[lo_col])

    if math.isnan(hi) or math.isnan(lo):
        return "UNKNOWN"
    if price > hi:
        return "ABOVE"
    if price < lo:
        return "BELOW"
    return "INSIDE"

