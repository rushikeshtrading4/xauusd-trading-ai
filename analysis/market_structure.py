"""
Market structure analysis module.

Detects institutional-grade structural events from swing-point data:

  LIQUIDITY_SWEEP  — price wicks beyond a swing level but fails to close there.
  BREAK            — price closes beyond a swing level (intermediate state).
  BOS_CONFIRMED    — Break of Structure confirmed via break → pullback → continuation.
  CHOCH            — Change of Character: protected swing level violated.

The engine also tracks a trend state machine (BULLISH / BEARISH / TRANSITION)
and records dynamic *protected* high / low levels used for CHOCH detection.

Pipeline position::

    swing_points → market_structure → liquidity_detection → order_blocks

Public API::

    from analysis.market_structure import detect_market_structure
    df_with_structure = detect_market_structure(df, timeframe="H1")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public string constants — event and state labels
# ---------------------------------------------------------------------------

EVENT_LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"
EVENT_BREAK           = "BREAK"
EVENT_BOS_CONFIRMED   = "BOS_CONFIRMED"
EVENT_CHOCH           = "CHOCH"
EVENT_NONE            = ""

STATE_BULLISH    = "BULLISH"
STATE_BEARISH    = "BEARISH"
STATE_TRANSITION = "TRANSITION"

# ---------------------------------------------------------------------------
# Required input columns
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: list[str] = [
    "timestamp", "open", "high", "low", "close", "volume",
    "ATR", "swing_high", "swing_low",
]

# ---------------------------------------------------------------------------
# BOS confirmation tolerances
# ---------------------------------------------------------------------------

_PULLBACK_ATR_FACTOR      = 0.25   # how close pullback low/high must be to the broken level
_LIQUIDITY_ATR_FACTOR    = 0.10   # how close two highs/lows must be to form an equal level
# Sweeps are only valid when the probing wick occurs *near* the swing level.
# Without this filter a candle that trades far through a level and snaps back
# (e.g. a news spike) would incorrectly register as a liquidity sweep.
SWEEP_PROXIMITY_ATR_FACTOR = 1.5


# ---------------------------------------------------------------------------
# Internal state container
# ---------------------------------------------------------------------------


@dataclass
class _StructureState:
    """Mutable state carried forward through the row-by-row loop."""

    trend: str = STATE_TRANSITION

    # Last confirmed swing levels
    last_swing_high: Optional[float] = None
    last_swing_low:  Optional[float] = None

    # Protected levels (for CHOCH detection)
    protected_high: Optional[float] = None
    protected_low:  Optional[float] = None

    # BOS confirmation tracking
    # When a BREAK is registered, we store details for the three-step check.
    pending_break_direction: Optional[str] = None   # "BULL" | "BEAR"
    pending_break_level:     Optional[float] = None
    pending_break_high:      Optional[float] = None  # high of the break candle
    pending_break_low:       Optional[float] = None  # low  of the break candle
    in_pullback: bool = False


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def detect_market_structure(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Detect market structure events from swing-point annotated OHLC data.

    Processes rows sequentially to maintain a deterministic state machine.
    Each row is examined in order (oldest → newest); the state accumulated
    from earlier rows drives the classification of later ones.

    Algorithm overview
    ------------------
    For each candle the engine checks in priority order:

    1. **CHOCH** — did price close beyond the protected level?  If so, emit
       CHOCH and move trend to TRANSITION regardless of current state.
    2. **LIQUIDITY_SWEEP** — did price wick beyond the last known swing level
       but close back inside it?  Emit sweep; trend does not change.
    3. **BREAK** — did price close beyond a swing level?  Register a pending
       break and emit BREAK.  Trend does not change yet.
    4. **BOS_CONFIRMED** — if a BREAK is pending, evaluate whether the current
       candle is in the pullback phase or the continuation phase.  On
       continuation, emit BOS_CONFIRMED and update trend / protected levels.
    5. Update protected levels every time a new confirmed swing is observed.

    Args:
        df:        Swing-annotated OHLCV DataFrame.  Must contain all columns
                   listed in :data:`_REQUIRED_COLUMNS`.
        timeframe: Timeframe identifier (e.g. ``"H1"``, ``"M15"``).
                   Stored in the output for downstream consumers.

    Returns:
        Copy of *df* with four additional string columns:

        * ``event_type``    — one of the ``EVENT_*`` constants or ``""``.
        * ``trend_state``   — one of ``"BULLISH"``, ``"BEARISH"``, ``"TRANSITION"``.
        * ``protected_high``— float (or ``NaN``) protected high level at that row.
        * ``protected_low`` — float (or ``NaN``) protected low level at that row.

    Raises:
        ValueError: If required columns are missing or *df* is empty.
    """
    _validate_input(df)

    out = df.copy().reset_index(drop=True)
    n   = len(out)

    # Pre-extract numpy arrays for speed — avoids repeated .loc / .iloc calls.
    highs      = out["high"].to_numpy(dtype=float)
    lows       = out["low"].to_numpy(dtype=float)
    closes     = out["close"].to_numpy(dtype=float)
    atrs       = out["ATR"].to_numpy(dtype=float)
    sw_highs   = out["swing_high"].to_numpy(dtype=bool)
    sw_lows    = out["swing_low"].to_numpy(dtype=bool)

    # Output arrays
    event_types     = np.full(n, EVENT_NONE,      dtype=object)
    trend_states    = np.full(n, STATE_TRANSITION, dtype=object)
    protected_highs = np.full(n, np.nan,           dtype=float)
    protected_lows  = np.full(n, np.nan,           dtype=float)

    state = _StructureState()

    for i in range(n):
        h      = highs[i]
        lo     = lows[i]
        cl     = closes[i]
        atr    = atrs[i]
        is_sh  = sw_highs[i]
        is_sl  = sw_lows[i]

        # ATR safety guard — when indicator values are not yet stabilised
        # (e.g. the first ATR_period rows), ATR may be NaN or zero.  Using
        # such values in tolerance calculations would produce NaN comparisons
        # or division-by-zero.  Fall back to a near-zero sentinel so the
        # logic remains correct and thresholds effectively collapse to zero.
        if np.isnan(atr) or atr <= 0:
            atr = 1e-6

        event = EVENT_NONE

        # ------------------------------------------------------------------
        # 0. Update latest swing levels from confirmed swing points
        # ------------------------------------------------------------------
        if is_sh:
            state.last_swing_high = h
            # In a bullish trend, a new swing high with a preceding higher low
            # updates the protected low (the HL that led to this HH).
            if state.trend == STATE_BULLISH and state.last_swing_low is not None:
                state.protected_low = state.last_swing_low
            # Also track the most recent swing high as the protected_high so
            # downstream consumers always see the current significant ceiling.
            if state.trend == STATE_BULLISH:
                state.protected_high = h

        if is_sl:
            state.last_swing_low = lo
            # In a bearish trend, a new swing low with a preceding lower high
            # updates the protected high.
            if state.trend == STATE_BEARISH and state.last_swing_high is not None:
                state.protected_high = state.last_swing_high

        # ------------------------------------------------------------------
        # 1. CHOCH — protected level broken
        # ------------------------------------------------------------------
        if event == EVENT_NONE:
            if (
                state.trend == STATE_BULLISH
                and state.protected_low is not None
                and cl < state.protected_low
            ):
                event = EVENT_CHOCH
                state.trend = STATE_TRANSITION
                state.pending_break_direction = None
                state.in_pullback = False
                # Clear protected levels — they are only meaningful in the
                # context of the trend that just ended.  After CHOCH the
                # engine is in TRANSITION and fresh structure must be built.
                state.protected_high = None
                state.protected_low  = None

            elif (
                state.trend == STATE_BEARISH
                and state.protected_high is not None
                and cl > state.protected_high
            ):
                event = EVENT_CHOCH
                state.trend = STATE_TRANSITION
                state.pending_break_direction = None
                state.in_pullback = False
                state.protected_high = None
                state.protected_low  = None

        # ------------------------------------------------------------------
        # 2. BOS_CONFIRMED — three-step check (break already pending)
        # ------------------------------------------------------------------
        if event == EVENT_NONE and state.pending_break_direction is not None:
            direction = state.pending_break_direction
            level     = state.pending_break_level
            tolerance = _PULLBACK_ATR_FACTOR * atr

            if not state.in_pullback:
                # Waiting for pullback
                if direction == "BULL" and lo >= level - tolerance:
                    state.in_pullback = True
                elif direction == "BEAR" and h <= level + tolerance:
                    state.in_pullback = True
                else:
                    # Price moved away from level without a valid pullback —
                    # reset pending break to avoid stale confirmation.
                    if direction == "BULL" and lo < level - tolerance * 4:
                        state.pending_break_direction = None
                    elif direction == "BEAR" and h > level + tolerance * 4:
                        state.pending_break_direction = None

            else:
                # In pullback — waiting for continuation impulse
                break_high = state.pending_break_high
                break_low  = state.pending_break_low

                if direction == "BULL" and h > max(break_high, level):
                    event = EVENT_BOS_CONFIRMED
                    state.trend = STATE_BULLISH
                    state.pending_break_direction = None
                    state.in_pullback = False
                    # The swing high at the break level becomes the new protected high
                    state.protected_high = level

                elif direction == "BEAR" and lo < min(break_low, level):
                    event = EVENT_BOS_CONFIRMED
                    state.trend = STATE_BEARISH
                    state.pending_break_direction = None
                    state.in_pullback = False
                    state.protected_low = level

        # ------------------------------------------------------------------
        # 3. LIQUIDITY_SWEEP (only when no more significant event found yet)
        # ------------------------------------------------------------------
        if event == EVENT_NONE and state.last_swing_high is not None:
            ref_sh = state.last_swing_high
            # Proximity filter: the probing wick must stay within
            # SWEEP_PROXIMITY_ATR_FACTOR × ATR of the swing level.  This
            # prevents far-reaching spikes (e.g. news wicks) from being
            # misclassified as liquidity sweeps.
            if (
                h > ref_sh
                and cl <= ref_sh
                and abs(h - ref_sh) <= SWEEP_PROXIMITY_ATR_FACTOR * atr
            ):
                event = EVENT_LIQUIDITY_SWEEP   # bearish sweep above swing high

        if event == EVENT_NONE and state.last_swing_low is not None:
            ref_sl = state.last_swing_low
            if (
                lo < ref_sl
                and cl >= ref_sl
                and abs(lo - ref_sl) <= SWEEP_PROXIMITY_ATR_FACTOR * atr
            ):
                event = EVENT_LIQUIDITY_SWEEP   # bullish sweep below swing low

        # ------------------------------------------------------------------
        # 4. BREAK — close beyond known swing level (start BOS sequence)
        # ------------------------------------------------------------------
        if event == EVENT_NONE:
            if state.last_swing_high is not None and cl > state.last_swing_high:
                if state.pending_break_direction != "BULL":   # avoid duplicate
                    event = EVENT_BREAK
                    state.pending_break_direction = "BULL"
                    state.pending_break_level     = state.last_swing_high
                    state.pending_break_high      = h
                    state.pending_break_low       = lo
                    state.in_pullback             = False

            elif state.last_swing_low is not None and cl < state.last_swing_low:
                if state.pending_break_direction != "BEAR":
                    event = EVENT_BREAK
                    state.pending_break_direction = "BEAR"
                    state.pending_break_level     = state.last_swing_low
                    state.pending_break_high      = h
                    state.pending_break_low       = lo
                    state.in_pullback             = False

        # ------------------------------------------------------------------
        # 5. Write output for this row
        # ------------------------------------------------------------------
        event_types[i]     = event
        trend_states[i]    = state.trend
        protected_highs[i] = state.protected_high if state.protected_high is not None else np.nan
        protected_lows[i]  = state.protected_low  if state.protected_low  is not None else np.nan

    out["event_type"]     = event_types
    out["trend_state"]    = trend_states
    out["protected_high"] = protected_highs
    out["protected_low"]  = protected_lows

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
            f"detect_market_structure(): missing required column(s): {missing}"
        )
    if len(df) == 0:
        raise ValueError(
            "detect_market_structure(): DataFrame must not be empty."
        )

