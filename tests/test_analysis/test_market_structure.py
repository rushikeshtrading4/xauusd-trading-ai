"""
Comprehensive pytest test suite for analysis/market_structure.py.

Tests are organised into 14 groups:

  1.  Input Validation
  2.  No Structure Events
  3.  Liquidity Sweep Detection
  4.  Break Detection
  5.  Bullish BOS Confirmation
  6.  Bearish BOS Confirmation
  7.  CHOCH Detection
  8.  Trend Persistence
  9.  False Breakout (stale break invalidation)
  10. Consecutive Sweeps
  11. ATR Safety Guard
  12. Event Priority Order
  13. Output Schema
  14. Deterministic Behaviour

All scenarios are driven by synthetic OHLC DataFrames built from precise
candle specifications so that every assertion is deterministic and grounded
in the algorithm's documented behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.market_structure import (
    SWEEP_PROXIMITY_ATR_FACTOR,
    EVENT_BOS_CONFIRMED,
    EVENT_BREAK,
    EVENT_CHOCH,
    EVENT_LIQUIDITY_SWEEP,
    EVENT_NONE,
    STATE_BEARISH,
    STATE_BULLISH,
    STATE_TRANSITION,
    _PULLBACK_ATR_FACTOR,
    _REQUIRED_COLUMNS,
    detect_market_structure,
)

# ---------------------------------------------------------------------------
# Module-level constants used in tolerance calculations
# ---------------------------------------------------------------------------

_ATR = 2.0
_TOLERANCE = _PULLBACK_ATR_FACTOR * _ATR          # 0.5  (pullback window)
_PROXIMITY = SWEEP_PROXIMITY_ATR_FACTOR * _ATR    # 3.0  (sweep wick window)

# ---------------------------------------------------------------------------
# DataFrame construction helpers
# ---------------------------------------------------------------------------


def _ts(n: int) -> pd.DatetimeIndex:
    """Return n hourly UTC timestamps starting 2026-01-01."""
    return pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")


def _row(
    close: float,
    *,
    atr: float = _ATR,
    high: float | None = None,
    low: float | None = None,
    swing_high: bool = False,
    swing_low: bool = False,
) -> dict:
    """Return a single candle dict.

    high and low default to close ± 1.0 when not supplied explicitly.
    open is set equal to close (doji body) for simplicity.
    """
    return {
        "open":       close,
        "high":       high if high is not None else close + 1.0,
        "low":        low  if low  is not None else close - 1.0,
        "close":      close,
        "volume":     1000.0,
        "ATR":        atr,
        "swing_high": swing_high,
        "swing_low":  swing_low,
    }


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Assemble candle dicts into a timestamped DataFrame."""
    df = pd.DataFrame(rows)
    df.insert(0, "timestamp", _ts(len(df)))
    return df


def _flat_df(n: int = 10, close: float = 100.0) -> pd.DataFrame:
    """Return a perfectly flat price DataFrame with no swing markers."""
    return _make_df([_row(close) for _ in range(n)])


# ---------------------------------------------------------------------------
# Reusable scenario fixtures (shared across multiple groups)
# ---------------------------------------------------------------------------


def _bullish_bos_df() -> pd.DataFrame:
    """Minimal DataFrame that produces a complete bullish BOS sequence.

    Row 0: swing_low  → last_swing_low = 79   (close=80, default low=79)
    Row 1: swing_high → last_swing_high = 101  (close=100, default high=101)
    Row 2: BREAK      → close=102 > 101;
                        break_level=101, break_high=103, break_low=101
    Row 3: pullback   → low=101 ≥ level-tolerance=100.5 → in_pullback=True
    Row 4: BOS_CONF   → high=104 > max(103,101)=103 → BOS_CONFIRMED;
                        trend=BULLISH, protected_high=101
    """
    return _make_df([
        _row(80,  swing_low=True),            # row 0: low=79
        _row(100, swing_high=True),           # row 1: high=101
        _row(102, high=103, low=101),         # row 2: BREAK
        _row(101, high=102, low=101),         # row 3: pullback  (lo=101 ≥ 100.5)
        _row(103, high=104, low=102),         # row 4: BOS_CONFIRMED (h=104 > 103)
    ])


def _bearish_bos_df() -> pd.DataFrame:
    """Minimal DataFrame that produces a complete bearish BOS sequence.

    Row 0: swing_high → last_swing_high = 121 (close=120, default high=121)
    Row 1: swing_low  → last_swing_low = 99   (close=100, default low=99)
    Row 2: BREAK      → close=98 < 99;
                        break_level=99, break_high=100, break_low=97
    Row 3: pullback   → high=99 ≤ level+tolerance=99.5 → in_pullback=True
    Row 4: BOS_CONF   → low=96 < min(97,99)=97 → BOS_CONFIRMED;
                        trend=BEARISH, protected_low=99
    """
    return _make_df([
        _row(120, swing_high=True),           # row 0: high=121
        _row(100, swing_low=True),            # row 1: low=99
        _row(98,  high=100, low=97),          # row 2: BREAK
        _row(99,  high=99,  low=98),          # row 3: pullback (h=99 ≤ 99.5)
        _row(96,  high=97,  low=96),          # row 4: BOS_CONFIRMED (lo=96 < 97)
    ])


def _bullish_choch_df() -> pd.DataFrame:
    """DataFrame that ends with a bullish CHOCH event.

    Rows 0-4: bullish BOS → trend=BULLISH, protected_high=101, last_swing_low=79
    Row 5:    new swing_high while BULLISH → protected_low set to 79
    Row 6:    close=78 < protected_low=79 → CHOCH; trend→TRANSITION
    """
    return _make_df([
        _row(80,  swing_low=True),                         # row 0: low=79
        _row(100, swing_high=True),                        # row 1: high=101
        _row(102, high=103, low=101),                      # row 2: BREAK
        _row(101, high=102, low=101),                      # row 3: pullback
        _row(103, high=104, low=102),                      # row 4: BOS_CONFIRMED → BULLISH
        _row(105, high=106, low=104, swing_high=True),     # row 5: protected_low=79
        _row(78,  high=79,  low=77),                       # row 6: CHOCH
    ])


def _bearish_choch_df() -> pd.DataFrame:
    """DataFrame that ends with a bearish CHOCH event.

    Rows 0-4: bearish BOS → trend=BEARISH, protected_low=99, last_swing_high=121
    Row 5:    new swing_low while BEARISH → protected_high set to 121
    Row 6:    close=122 > protected_high=121 → CHOCH; trend→TRANSITION
    """
    return _make_df([
        _row(120, swing_high=True),                        # row 0: high=121
        _row(100, swing_low=True),                         # row 1: low=99
        _row(98,  high=100, low=97),                       # row 2: BREAK
        _row(99,  high=99,  low=98),                       # row 3: pullback
        _row(96,  high=97,  low=96),                       # row 4: BOS_CONFIRMED → BEARISH
        _row(95,  high=96,  low=94, swing_low=True),       # row 5: protected_high=121
        _row(122, high=123, low=121),                      # row 6: CHOCH
    ])


# ===========================================================================
# GROUP 1 — INPUT VALIDATION
# ===========================================================================


class TestInputValidation:
    """detect_market_structure() raises ValueError on invalid inputs."""

    @pytest.mark.parametrize("missing_col", _REQUIRED_COLUMNS)
    def test_missing_column_raises_value_error(self, missing_col: str) -> None:
        """Each individually missing required column must raise ValueError."""
        df = _flat_df(5).drop(columns=[missing_col])
        with pytest.raises(ValueError, match="missing required column"):
            detect_market_structure(df, "H1")

    def test_empty_dataframe_raises_value_error(self) -> None:
        """A zero-row DataFrame (correct schema) must raise ValueError."""
        df = _flat_df(5).iloc[0:0]   # empty but schema intact
        with pytest.raises(ValueError, match="must not be empty"):
            detect_market_structure(df, "H1")


# ===========================================================================
# GROUP 2 — NO STRUCTURE EVENTS
# ===========================================================================


class TestNoStructureEvents:
    """Flat or gently trending price with no swing markers produces no events."""

    def test_all_event_types_are_empty_string(self) -> None:
        """Every event_type value must be the empty-string sentinel."""
        result = detect_market_structure(_flat_df(10), "H1")
        assert (result["event_type"] == EVENT_NONE).all()

    def test_trend_state_stays_transition(self) -> None:
        """Without a BOS or CHOCH the trend state must remain TRANSITION."""
        result = detect_market_structure(_flat_df(10), "H1")
        assert (result["trend_state"] == STATE_TRANSITION).all()

    def test_no_events_on_smooth_uptrend_without_swing_markers(self) -> None:
        """A gentle uptrend without swing markers → no reference levels set,
        so neither BREAK nor SWEEP can fire."""
        rows = [_row(100.0 + i * 0.5) for i in range(15)]
        result = detect_market_structure(_make_df(rows), "H1")
        assert (result["event_type"] == EVENT_NONE).all()


# ===========================================================================
# GROUP 3 — LIQUIDITY SWEEP DETECTION
# ===========================================================================


class TestLiquiditySweepDetection:
    """Wick-beyond-and-close-back produces LIQUIDITY_SWEEP."""

    def test_bearish_sweep_above_swing_high(self) -> None:
        """High wicks above the swing level; close returns below → SWEEP.

        Row 0: swing_high marks high=101 → last_swing_high=101
        Row 1: high=102 > 101, close=99 ≤ 101, abs(102-101)=1 ≤ proximity=3 → SWEEP
        """
        df = _make_df([
            _row(100, swing_high=True),          # high=101
            _row(99,  high=102, low=98),         # wick above, close back below
        ])
        result = detect_market_structure(df, "H1")
        assert result.loc[1, "event_type"] == EVENT_LIQUIDITY_SWEEP

    def test_bullish_sweep_below_swing_low(self) -> None:
        """Low wicks below the swing level; close returns above → SWEEP.

        Row 0: swing_low marks low=99 → last_swing_low=99
        Row 1: low=98 < 99, close=101 ≥ 99, abs(98-99)=1 ≤ 3 → SWEEP
        """
        df = _make_df([
            _row(100, swing_low=True),           # low=99
            _row(101, high=102, low=98),         # wick below, close above
        ])
        result = detect_market_structure(df, "H1")
        assert result.loc[1, "event_type"] == EVENT_LIQUIDITY_SWEEP

    def test_far_spike_above_swing_high_blocked_by_proximity_filter(self) -> None:
        """Spike far above swing level (wick > 1.5×ATR) must NOT produce a sweep.

        ATR=2, proximity=3.  Wick of 10 pts (high=111 vs swing=101): blocked.
        This is the exact scenario from the improvement spec.
        """
        df = _make_df([
            _row(100, swing_high=True),          # high=101
            _row(99,  high=111, low=98),         # far spike (abs=10 > 3) → no sweep
        ])
        result = detect_market_structure(df, "H1")
        assert result.loc[1, "event_type"] != EVENT_LIQUIDITY_SWEEP

    def test_far_spike_below_swing_low_blocked_by_proximity_filter(self) -> None:
        """Spike far below swing level (wick > 1.5×ATR) must NOT produce a sweep."""
        df = _make_df([
            _row(100, swing_low=True),           # low=99
            _row(101, high=102, low=89),         # far spike down (abs=10 > 3) → no sweep
        ])
        result = detect_market_structure(df, "H1")
        assert result.loc[1, "event_type"] != EVENT_LIQUIDITY_SWEEP

    def test_sweep_does_not_change_trend_state(self) -> None:
        """A liquidity sweep must leave trend_state unchanged (still TRANSITION)."""
        df = _make_df([
            _row(100, swing_high=True),
            _row(99,  high=102, low=98),
        ])
        result = detect_market_structure(df, "H1")
        assert result.loc[1, "event_type"] == EVENT_LIQUIDITY_SWEEP
        assert result.loc[1, "trend_state"] == STATE_TRANSITION

    def test_sweep_at_proximity_boundary_is_allowed(self) -> None:
        """Wick distance exactly equal to 1.5×ATR (boundary) must trigger a sweep.

        ATR=2 → proximity=3.0.  high=104 vs swing=101: abs=3.0 == 3.0 → SWEEP (≤ not <).
        """
        df = _make_df([
            _row(100, swing_high=True),          # high=101
            _row(99,  high=104, low=98),         # abs(104-101)=3.0 == boundary
        ])
        result = detect_market_structure(df, "H1")
        assert result.loc[1, "event_type"] == EVENT_LIQUIDITY_SWEEP


# ===========================================================================
# GROUP 4 — BREAK DETECTION
# ===========================================================================


class TestBreakDetection:
    """Close beyond the last known swing level registers EVENT_BREAK."""

    def test_bullish_break_detected(self) -> None:
        """Close above last swing high → BREAK.

        last_swing_high=101 (row 0 high); row 1 close=102 > 101 → BREAK.
        """
        df = _make_df([
            _row(100, swing_high=True),          # high=101
            _row(102, high=103, low=101),        # close=102 > 101
        ])
        result = detect_market_structure(df, "H1")
        assert result.loc[1, "event_type"] == EVENT_BREAK

    def test_bearish_break_detected(self) -> None:
        """Close below last swing low → BREAK.

        last_swing_low=99 (row 0 low); row 1 close=98 < 99 → BREAK.
        """
        df = _make_df([
            _row(100, swing_low=True),           # low=99
            _row(98,  high=99,  low=97),         # close=98 < 99
        ])
        result = detect_market_structure(df, "H1")
        assert result.loc[1, "event_type"] == EVENT_BREAK

    def test_break_trend_state_remains_transition(self) -> None:
        """A naked BREAK must not advance trend_state to BULLISH or BEARISH."""
        df = _make_df([
            _row(100, swing_high=True),
            _row(102, high=103, low=101),
        ])
        result = detect_market_structure(df, "H1")
        assert result.loc[1, "trend_state"] == STATE_TRANSITION

    def test_no_break_without_prior_swing_marker(self) -> None:
        """Without an established swing reference, no BREAK can fire."""
        # No swing_high or swing_low markers → last_swing_high/low are None
        df = _make_df([
            _row(100),
            _row(105),
        ])
        result = detect_market_structure(df, "H1")
        assert (result["event_type"] == EVENT_NONE).all()


# ===========================================================================
# GROUP 5 — BULLISH BOS CONFIRMATION
# ===========================================================================


class TestBullishBOS:
    """Full three-step bullish BOS: break → pullback → continuation."""

    def test_break_event_emitted_on_break_candle(self) -> None:
        """BREAK must appear on the candle that closes above the swing high."""
        result = detect_market_structure(_bullish_bos_df(), "H1")
        assert result.loc[2, "event_type"] == EVENT_BREAK

    def test_bos_confirmed_event_emitted_on_continuation_candle(self) -> None:
        """BOS_CONFIRMED must appear on the continuation candle (row 4)."""
        result = detect_market_structure(_bullish_bos_df(), "H1")
        assert result.loc[4, "event_type"] == EVENT_BOS_CONFIRMED

    def test_trend_becomes_bullish_after_bos(self) -> None:
        """trend_state must become BULLISH at the BOS_CONFIRMED row."""
        result = detect_market_structure(_bullish_bos_df(), "H1")
        assert result.loc[4, "trend_state"] == STATE_BULLISH

    def test_trend_remains_transition_before_bos(self) -> None:
        """Trend must stay TRANSITION on both the BREAK and pullback candles."""
        result = detect_market_structure(_bullish_bos_df(), "H1")
        assert result.loc[2, "trend_state"] == STATE_TRANSITION
        assert result.loc[3, "trend_state"] == STATE_TRANSITION

    def test_protected_high_set_to_break_level(self) -> None:
        """protected_high must be set to the swing high that was broken (101)."""
        result = detect_market_structure(_bullish_bos_df(), "H1")
        assert result.loc[4, "protected_high"] == 101.0

    def test_bos_confirmed_not_on_break_or_pullback_candle(self) -> None:
        """BOS_CONFIRMED must NOT appear before the continuation impulse."""
        result = detect_market_structure(_bullish_bos_df(), "H1")
        assert result.loc[2, "event_type"] != EVENT_BOS_CONFIRMED
        assert result.loc[3, "event_type"] != EVENT_BOS_CONFIRMED


# ===========================================================================
# GROUP 6 — BEARISH BOS CONFIRMATION
# ===========================================================================


class TestBearishBOS:
    """Full three-step bearish BOS: break → pullback → continuation."""

    def test_break_event_emitted_on_break_candle(self) -> None:
        """BREAK must appear on the candle that closes below the swing low."""
        result = detect_market_structure(_bearish_bos_df(), "H1")
        assert result.loc[2, "event_type"] == EVENT_BREAK

    def test_bos_confirmed_event_emitted_on_continuation_candle(self) -> None:
        """BOS_CONFIRMED must appear on the continuation candle (row 4)."""
        result = detect_market_structure(_bearish_bos_df(), "H1")
        assert result.loc[4, "event_type"] == EVENT_BOS_CONFIRMED

    def test_trend_becomes_bearish_after_bos(self) -> None:
        """trend_state must become BEARISH at the BOS_CONFIRMED row."""
        result = detect_market_structure(_bearish_bos_df(), "H1")
        assert result.loc[4, "trend_state"] == STATE_BEARISH

    def test_protected_low_set_to_break_level(self) -> None:
        """protected_low must be set to the swing low that was broken (99)."""
        result = detect_market_structure(_bearish_bos_df(), "H1")
        assert result.loc[4, "protected_low"] == 99.0

    def test_trend_remains_transition_before_bos(self) -> None:
        """Trend must stay TRANSITION on the BREAK and pullback candles."""
        result = detect_market_structure(_bearish_bos_df(), "H1")
        assert result.loc[2, "trend_state"] == STATE_TRANSITION
        assert result.loc[3, "trend_state"] == STATE_TRANSITION


# ===========================================================================
# GROUP 7 — CHOCH DETECTION
# ===========================================================================


class TestCHOCHDetection:
    """CHOCH fires when a protected level is violated in the confirmed trend."""

    def test_bullish_choch_event_emitted(self) -> None:
        """Close below protected_low in a bullish trend → CHOCH on row 6."""
        result = detect_market_structure(_bullish_choch_df(), "H1")
        assert result.loc[6, "event_type"] == EVENT_CHOCH

    def test_bullish_choch_trend_reverts_to_transition(self) -> None:
        """After a bullish CHOCH the trend must revert to TRANSITION."""
        result = detect_market_structure(_bullish_choch_df(), "H1")
        assert result.loc[6, "trend_state"] == STATE_TRANSITION

    def test_bearish_choch_event_emitted(self) -> None:
        """Close above protected_high in a bearish trend → CHOCH on row 6."""
        result = detect_market_structure(_bearish_choch_df(), "H1")
        assert result.loc[6, "event_type"] == EVENT_CHOCH

    def test_bearish_choch_trend_reverts_to_transition(self) -> None:
        """After a bearish CHOCH the trend must revert to TRANSITION."""
        result = detect_market_structure(_bearish_choch_df(), "H1")
        assert result.loc[6, "trend_state"] == STATE_TRANSITION

    def test_no_choch_without_established_protected_level(self) -> None:
        """CHOCH cannot fire before a protected level has been set via BOS."""
        # A flat DataFrame never enters a confirmed trend, so protected_low/high
        # are never defined — CHOCH is structurally impossible.
        result = detect_market_structure(_flat_df(10), "H1")
        assert EVENT_CHOCH not in result["event_type"].values


# ===========================================================================
# GROUP 8 — TREND PERSISTENCE
# ===========================================================================


class TestTrendPersistence:
    """After a BOS the trend must persist until an opposing CHOCH or BOS fires."""

    def test_bullish_trend_persists_through_neutral_candles(self) -> None:
        """After bullish BOS, 5 neutral candles that break nothing must remain BULLISH."""
        rows_bos = [
            _row(80,  swing_low=True),
            _row(100, swing_high=True),
            _row(102, high=103, low=101),
            _row(101, high=102, low=101),
            _row(103, high=104, low=102),
        ]
        # Neutral candles: price oscillates, doesn't break protected levels
        neutral = [_row(102) for _ in range(5)]
        df = _make_df(rows_bos + neutral)
        result = detect_market_structure(df, "H1")
        for i in range(4, 10):
            assert result.loc[i, "trend_state"] == STATE_BULLISH, \
                f"row {i} expected BULLISH, got {result.loc[i, 'trend_state']}"

    def test_bearish_trend_persists_through_neutral_candles(self) -> None:
        """After bearish BOS, 5 neutral candles must remain BEARISH."""
        rows_bos = [
            _row(120, swing_high=True),
            _row(100, swing_low=True),
            _row(98,  high=100, low=97),
            _row(99,  high=99,  low=98),
            _row(96,  high=97,  low=96),
        ]
        neutral = [_row(97) for _ in range(5)]
        df = _make_df(rows_bos + neutral)
        result = detect_market_structure(df, "H1")
        for i in range(4, 10):
            assert result.loc[i, "trend_state"] == STATE_BEARISH, \
                f"row {i} expected BEARISH, got {result.loc[i, 'trend_state']}"


# ===========================================================================
# GROUP 9 — FALSE BREAKOUT (stale break invalidation)
# ===========================================================================


class TestFalseBreakout:
    """A deep pullback (> 4× tolerance) resets the pending BOS sequence."""

    def test_deep_pullback_invalidates_bullish_bos(self) -> None:
        """
        Sequence: BREAK → deep pullback (lo < level - 4×tolerance) → reset.

        With level=101 and tolerance=0.5, the reset threshold is lo < 99.
        Row 2 (lo=96) triggers the reset; BOS_CONFIRMED must never appear.
        """
        df = _make_df([
            _row(100, swing_high=True),           # high=101 → last_swing_high=101
            _row(102, high=103, low=101),         # BREAK; break_level=101
            _row(96,  high=98,  low=96),          # deep pullback lo=96 < 99 → reset
            _row(104, high=105, low=103),         # continuation attempt (no pending)
        ])
        result = detect_market_structure(df, "H1")
        assert EVENT_BOS_CONFIRMED not in result["event_type"].values

    def test_deep_pullback_invalidates_bearish_bos(self) -> None:
        """
        Bear break followed by deep upward spike resets the pending break.

        With level=99 and tolerance=0.5, the reset threshold is h > 101.
        Row 2 (h=104) exceeds 101 → reset; BOS_CONFIRMED must never appear.
        """
        df = _make_df([
            _row(100, swing_low=True),            # low=99 → last_swing_low=99
            _row(98,  high=100, low=97),          # BREAK; break_level=99
            _row(103, high=104, low=102),         # deep rally h=104 > 101 → reset
            _row(96,  high=97,  low=96),          # continuation attempt (no pending)
        ])
        result = detect_market_structure(df, "H1")
        assert EVENT_BOS_CONFIRMED not in result["event_type"].values

    def test_valid_pullback_does_not_invalidate_bos(self) -> None:
        """A pullback within the tolerance window must complete the BOS sequence."""
        result = detect_market_structure(_bullish_bos_df(), "H1")
        assert result.loc[4, "event_type"] == EVENT_BOS_CONFIRMED


# ===========================================================================
# GROUP 10 — CONSECUTIVE SWEEPS
# ===========================================================================


class TestConsecutiveSweeps:
    """Multiple candles can each produce a LIQUIDITY_SWEEP against the same level."""

    def test_three_consecutive_bearish_sweeps_detected(self) -> None:
        """Three candles that all wick above the same swing high → 3 SWEEP events.

        last_swing_high = 101 (row 0).
        Rows 1-3 each have high > 101, close ≤ 101, wick within proximity=3.
        """
        df = _make_df([
            _row(100, swing_high=True),            # high=101
            _row(99,  high=102,   low=98),         # SWEEP: abs(102-101)=1 ≤ 3
            _row(99,  high=101.5, low=98),         # SWEEP: abs(0.5) ≤ 3
            _row(99,  high=103,   low=98),         # SWEEP: abs(2) ≤ 3
        ])
        result = detect_market_structure(df, "H1")
        sweep_count = (result["event_type"] == EVENT_LIQUIDITY_SWEEP).sum()
        assert sweep_count == 3

    def test_consecutive_sweeps_do_not_change_trend(self) -> None:
        """No number of liquidity sweeps alone must advance trend_state."""
        df = _make_df([
            _row(100, swing_high=True),
            _row(99,  high=102,   low=98),
            _row(99,  high=101.5, low=98),
            _row(99,  high=103,   low=98),
        ])
        result = detect_market_structure(df, "H1")
        assert (result["trend_state"] == STATE_TRANSITION).all()


# ===========================================================================
# GROUP 11 — ATR SAFETY GUARD
# ===========================================================================


class TestATRSafetyGuard:
    """NaN or zero ATR must not cause exceptions or corrupt the output."""

    def test_nan_atr_does_not_raise(self) -> None:
        """Rows with ATR=NaN must not raise any exception."""
        rows = [_row(100, atr=float("nan")) for _ in range(5)]
        result = detect_market_structure(_make_df(rows), "H1")
        assert len(result) == 5

    def test_zero_atr_does_not_raise(self) -> None:
        """Rows with ATR=0 must not raise any exception."""
        rows = [_row(100, atr=0.0) for _ in range(5)]
        result = detect_market_structure(_make_df(rows), "H1")
        assert len(result) == 5

    def test_output_still_produced_for_mixed_nan_and_valid_atr(self) -> None:
        """Mixed NaN/valid ATR rows must all produce valid string output columns."""
        rows = (
            [_row(100, atr=float("nan"))] * 3
            + [_row(100, atr=2.0)] * 7
        )
        result = detect_market_structure(_make_df(rows), "H1")
        assert len(result) == 10
        # event_type and trend_state must be non-null strings, not NaN
        assert result["event_type"].notna().all()
        assert result["trend_state"].notna().all()

    def test_events_still_detected_on_candles_after_nan_rows(self) -> None:
        """A BREAK on a candle with normal ATR must be detected even when
        earlier rows had NaN/zero ATR and triggered the safety guard."""
        df = _make_df([
            _row(100, atr=float("nan")),              # ATR guard fires
            _row(100, atr=0.0),                       # ATR guard fires
            _row(100, atr=2.0, swing_high=True),      # high=101 → last_swing_high=101
            _row(102, atr=2.0, high=103, low=101),    # BREAK above 101
        ])
        result = detect_market_structure(df, "H1")
        assert result.loc[3, "event_type"] == EVENT_BREAK


# ===========================================================================
# GROUP 12 — EVENT PRIORITY ORDER
# ===========================================================================


class TestEventPriority:
    """Priority: CHOCH > BOS_CONFIRMED > LIQUIDITY_SWEEP > BREAK."""

    def test_choch_overrides_bos_confirmed_on_same_candle(self) -> None:
        """When both CHOCH and BOS_CONFIRMED qualify for the same candle,
        CHOCH (step 1) must win over BOS_CONFIRMED (step 2).

        Setup:
          - Rows 0-4:  bullish BOS → trend=BULLISH, protected_high=101
          - Row 5:     swing_high in BULLISH state → protected_low=79
          - Rows 6-7:  new BREAK above 106, pullback → in_pullback=True
          - Row 8:     high=109 > max(108,106)=108 (would confirm BOS)
                       AND close=78 < protected_low=79 (triggers CHOCH)
                       → CHOCH must win.
        """
        rows = [
            _row(80,  swing_low=True),                        # 0: low=79
            _row(100, swing_high=True),                       # 1: high=101
            _row(102, high=103, low=101),                     # 2: BREAK (bull 1)
            _row(101, high=102, low=101),                     # 3: pullback
            _row(103, high=104, low=102),                     # 4: BOS_CONFIRMED → BULLISH
            _row(105, high=106, low=104, swing_high=True),    # 5: protected_low=79
            _row(107, high=108, low=106),                     # 6: BREAK (bull 2)
            _row(106, high=107, low=106),                     # 7: pullback (lo=106≥105.5)
            # Row 8: qualifies for BOS (h=109>108) AND CHOCH (cl=78<79)
            _row(78,  high=109, low=77),
        ]
        df = _make_df(rows)
        result = detect_market_structure(df, "H1")
        assert result.loc[8, "event_type"] == EVENT_CHOCH
        assert result.loc[8, "trend_state"] == STATE_TRANSITION

    def test_liquidity_sweep_fires_when_close_returns_below_swing_high(self) -> None:
        """When price wicks above a swing level and closes back BELOW it,
        LIQUIDITY_SWEEP fires (not BREAK), because close ≤ swing_high.
        BREAK requires close > swing_high; they are mutually exclusive by
        close value — this confirms SWEEP fires at priority 3."""
        df = _make_df([
            _row(100, swing_high=True),          # high=101
            _row(99,  high=102, low=98),         # wick above, close=99 ≤ 101 → SWEEP
        ])
        result = detect_market_structure(df, "H1")
        assert result.loc[1, "event_type"] == EVENT_LIQUIDITY_SWEEP

    def test_break_fires_when_close_exceeds_swing_high_not_sweep(self) -> None:
        """When close > swing_high, BREAK fires at priority 4.
        SWEEP cannot fire because it requires close ≤ swing_high."""
        df = _make_df([
            _row(100, swing_high=True),          # high=101
            _row(102, high=103, low=101),        # close=102 > 101 → BREAK, not SWEEP
        ])
        result = detect_market_structure(df, "H1")
        assert result.loc[1, "event_type"] == EVENT_BREAK
        assert result.loc[1, "event_type"] != EVENT_LIQUIDITY_SWEEP


# ===========================================================================
# GROUP 13 — OUTPUT SCHEMA
# ===========================================================================


class TestOutputSchema:
    """The returned DataFrame must have the correct structure."""

    def test_event_type_column_present(self) -> None:
        result = detect_market_structure(_flat_df(5), "H1")
        assert "event_type" in result.columns

    def test_trend_state_column_present(self) -> None:
        result = detect_market_structure(_flat_df(5), "H1")
        assert "trend_state" in result.columns

    def test_protected_high_column_present(self) -> None:
        result = detect_market_structure(_flat_df(5), "H1")
        assert "protected_high" in result.columns

    def test_protected_low_column_present(self) -> None:
        result = detect_market_structure(_flat_df(5), "H1")
        assert "protected_low" in result.columns

    def test_row_count_matches_input(self) -> None:
        """Output row count must equal input row count for various sizes."""
        for n in (1, 10, 50):
            result = detect_market_structure(_flat_df(n), "H1")
            assert len(result) == n, f"Expected {n} rows, got {len(result)}"

    def test_output_is_a_new_dataframe_not_the_input(self) -> None:
        """The returned object must be a distinct DataFrame (different identity)."""
        df = _flat_df(5)
        result = detect_market_structure(df, "H1")
        assert result is not df

    def test_input_dataframe_does_not_gain_new_columns(self) -> None:
        """The original input DataFrame must not be mutated by the function."""
        df = _flat_df(5)
        original_cols = set(df.columns)
        detect_market_structure(df, "H1")
        assert set(df.columns) == original_cols

    @pytest.mark.parametrize("timeframe", ["M1", "M5", "M15", "H1", "H4", "D1"])
    def test_all_standard_timeframes_accepted(self, timeframe: str) -> None:
        """The timeframe argument is accepted without error for every standard value."""
        result = detect_market_structure(_flat_df(5), timeframe)
        assert len(result) == 5


# ===========================================================================
# GROUP 14 — DETERMINISTIC BEHAVIOUR
# ===========================================================================


class TestDeterministicBehaviour:
    """Two calls on identical input must produce byte-for-byte identical output."""

    def test_repeated_calls_produce_identical_output_for_bos(self) -> None:
        """Full bullish BOS scenario must be identical across two runs."""
        df = _bullish_bos_df()
        r1 = detect_market_structure(df, "H1")
        r2 = detect_market_structure(df, "H1")
        pd.testing.assert_frame_equal(r1, r2)

    def test_repeated_calls_produce_identical_output_for_choch(self) -> None:
        """Full CHOCH scenario must be identical across two runs."""
        df = _bullish_choch_df()
        r1 = detect_market_structure(df, "H1")
        r2 = detect_market_structure(df, "H1")
        pd.testing.assert_frame_equal(r1, r2)


# ===========================================================================
# GROUP 15 — PROTECTED LEVEL MANAGEMENT
# ===========================================================================


class TestProtectedLevelManagement:
    """Validate protected_high and protected_low lifecycle: set, update, reset."""

    # -----------------------------------------------------------------------
    # Test 1 — Protected levels reset after CHOCH
    # -----------------------------------------------------------------------

    def test_protected_levels_are_nan_on_choch_candle(self) -> None:
        """After a CHOCH event both protected levels must be NaN on that candle.

        Sequence (uses _bullish_choch_df()):
          Rows 0-4 : bullish BOS → trend=BULLISH, protected_high=101
          Row 5    : swing_high while BULLISH
                       → protected_low = last_swing_low = 79
                       → protected_high updated to 106 (most recent HH)
          Row 6    : close=78 < protected_low=79 → CHOCH
                       → protected_high and protected_low must BOTH reset to NaN

        Once trend reverts to TRANSITION the previous protected levels are no
        longer meaningful; fresh structure must be built to re-establish them.
        """
        result = detect_market_structure(_bullish_choch_df(), "H1")
        choch_row = 6

        # Confirm the event really is CHOCH and trend really reverted
        assert result.loc[choch_row, "event_type"] == EVENT_CHOCH
        assert result.loc[choch_row, "trend_state"] == STATE_TRANSITION

        # Both protected levels must be NaN on the CHOCH candle itself
        assert np.isnan(result.loc[choch_row, "protected_high"]), \
            "protected_high must be NaN after CHOCH"
        assert np.isnan(result.loc[choch_row, "protected_low"]), \
            "protected_low must be NaN after CHOCH"

    def test_protected_levels_were_active_before_choch(self) -> None:
        """Confirm protected levels are non-NaN on the candle immediately before
        the CHOCH row, proving the reset is specific to the CHOCH event itself."""
        result = detect_market_structure(_bullish_choch_df(), "H1")
        # Row 5 is the swing_high in BULLISH that sets protected_low=79
        assert result.loc[5, "protected_low"] == 79.0, \
            "protected_low must be active on the pre-CHOCH swing candle"
        assert not np.isnan(result.loc[5, "protected_high"]), \
            "protected_high must be active on the pre-CHOCH swing candle"

    def test_bearish_choch_also_resets_protected_levels(self) -> None:
        """Reset must work symmetrically for a bearish CHOCH.

        Sequence (uses _bearish_choch_df()):
          Rows 0-4 : bearish BOS → trend=BEARISH, protected_low=99
          Row 5    : swing_low while BEARISH → protected_high = last_swing_high=121
          Row 6    : close=122 > protected_high=121 → CHOCH → both levels reset to NaN
        """
        result = detect_market_structure(_bearish_choch_df(), "H1")
        choch_row = 6
        assert result.loc[choch_row, "event_type"] == EVENT_CHOCH
        assert np.isnan(result.loc[choch_row, "protected_high"]), \
            "protected_high must be NaN after bearish CHOCH"
        assert np.isnan(result.loc[choch_row, "protected_low"]), \
            "protected_low must be NaN after bearish CHOCH"

    # -----------------------------------------------------------------------
    # Test 2 — New swing_high updates protected_high in a bullish trend
    # -----------------------------------------------------------------------

    def test_protected_high_updates_to_new_swing_high_in_bullish_trend(self) -> None:
        """In a confirmed BULLISH trend, each new swing_high must advance
        protected_high to the current significant ceiling.

        Rows 0-4 : bullish BOS → trend=BULLISH, protected_high=101 (break level)
        Row 5    : new swing_high (high=108) while BULLISH
                     → protected_high must update from 101 → 108
        """
        df = _make_df([
            _row(80,  swing_low=True),                       # 0: low=79
            _row(100, swing_high=True),                      # 1: high=101
            _row(102, high=103, low=101),                    # 2: BREAK
            _row(101, high=102, low=101),                    # 3: pullback
            _row(103, high=104, low=102),                    # 4: BOS_CONFIRMED → protected_high=101
            _row(103, high=108, low=102, swing_high=True),   # 5: new HH → protected_high→108
        ])
        result = detect_market_structure(df, "H1")

        # Initial protected_high set by BOS confirmation
        assert result.loc[4, "protected_high"] == 101.0, \
            "protected_high must equal break level immediately after BOS"

        # Must advance to the new swing_high high value
        assert result.loc[5, "protected_high"] == 108.0, \
            "protected_high must update to the new swing_high in BULLISH trend"

        # Trend must remain BULLISH
        assert result.loc[5, "trend_state"] == STATE_BULLISH

    def test_protected_low_updates_alongside_protected_high_on_swing_high(self) -> None:
        """When a swing_high occurs in BULLISH trend, protected_low also updates
        to the most recent swing_low (the higher-low that preceded the new HH).

        Row 0 : swing_low (low=79) → last_swing_low = 79
        Rows 1-4 : BOS → BULLISH
        Row 5 : swing_low (low=102) → last_swing_low = 102
        Row 6 : swing_high while BULLISH → protected_low must become 102
        """
        df = _make_df([
            _row(80,  swing_low=True),                       # 0: low=79
            _row(100, swing_high=True),                      # 1: high=101
            _row(102, high=103, low=101),                    # 2: BREAK
            _row(101, high=102, low=101),                    # 3: pullback
            _row(103, high=104, low=102),                    # 4: BOS_CONFIRMED → BULLISH
            _row(104, high=105, low=102, swing_low=True),    # 5: new HL (low=102)
            _row(108, high=111, low=107, swing_high=True),   # 6: new HH → protected_low=102
        ])
        result = detect_market_structure(df, "H1")
        assert result.loc[6, "protected_low"] == 102.0, \
            "protected_low must update to last swing_low when new swing_high appears"
        assert result.loc[6, "trend_state"] == STATE_BULLISH

    # -----------------------------------------------------------------------
    # Test 3 — Latest swing overrides older swing as the reference level
    # -----------------------------------------------------------------------

    def test_latest_swing_high_overrides_earlier_one_for_break_detection(self) -> None:
        """When two swing_high markers appear, BREAK must use the most recent
        level, not the earlier one.

        Row 0 : swing_high → last_swing_high = 101
        Row 1 : higher swing_high → last_swing_high = 106 (overrides 101)
        Row 2 : close=104 — above old level (101) but below new level (106)
                  → must NOT produce BREAK (confirms old level is discarded)
        Row 3 : close=107 — above new level (106)
                  → BREAK fires (confirms new level is used)
        """
        df = _make_df([
            _row(100, swing_high=True),                     # 0: high=101 → first swing
            _row(105, high=106, low=104, swing_high=True),  # 1: high=106 → overrides
            _row(104, high=105, low=103),                   # 2: 104 > 101 but 104 < 106
            _row(107, high=108, low=106),                   # 3: 107 > 106 → BREAK
        ])
        result = detect_market_structure(df, "H1")

        assert result.loc[2, "event_type"] == EVENT_NONE, \
            "Close above old swing but below new swing must not produce BREAK"
        assert result.loc[3, "event_type"] == EVENT_BREAK, \
            "Close above the most recent swing_high must produce BREAK"

    def test_latest_swing_low_overrides_earlier_one_for_break_detection(self) -> None:
        """Bearish mirror: the most recent swing_low is used for BREAK detection.

        Row 0 : swing_low → last_swing_low = 99
        Row 1 : lower swing_low → last_swing_low = 94 (overrides 99)
        Row 2 : close=96 — below old level (99) but above new level (94)
                  → must NOT produce BREAK
        Row 3 : close=93 — below new level (94) → BREAK fires
        """
        df = _make_df([
            _row(100, swing_low=True),                      # 0: low=99 → first swing
            _row(95,  high=96, low=94, swing_low=True),     # 1: low=94 → overrides
            _row(96,  high=97, low=95),                     # 2: 96 < 99 but 96 > 94
            _row(93,  high=94, low=92),                     # 3: 93 < 94 → BREAK
        ])
        result = detect_market_structure(df, "H1")

        assert result.loc[2, "event_type"] == EVENT_NONE, \
            "Close below old swing but above new swing must not produce BREAK"
        assert result.loc[3, "event_type"] == EVENT_BREAK, \
            "Close below the most recent swing_low must produce BREAK"

    # -----------------------------------------------------------------------
    # Test 4 — Multiple swings maintain correct protected levels
    # -----------------------------------------------------------------------

    def test_multiple_swing_highs_advance_protected_levels_correctly(self) -> None:
        """In a sustained BULLISH trend with multiple new swing_highs,
        protected_high and protected_low must track each new swing correctly
        while trend_state remains BULLISH throughout.

        Sequence:
          Rows 0-4  : bullish BOS → BULLISH, protected_high=101
          Row 5     : swing_high (high=106) → protected_high=106, protected_low=79
          Row 6     : swing_low  (low=103)  → last_swing_low=103  (protected unchanged)
          Row 7     : swing_high (high=111) → protected_high=111, protected_low=103
          Row 8     : neutral candle → BULLISH persists, levels unchanged
        """
        df = _make_df([
            _row(80,  swing_low=True),                        # 0: low=79
            _row(100, swing_high=True),                       # 1: high=101
            _row(102, high=103, low=101),                     # 2: BREAK
            _row(101, high=102, low=101),                     # 3: pullback
            _row(103, high=104, low=102),                     # 4: BOS_CONFIRMED → BULLISH
            _row(103, high=106, low=102, swing_high=True),    # 5: 1st swing_high→ PH=106
            _row(104, high=105, low=103, swing_low=True),     # 6: swing_low (low=103)
            _row(108, high=111, low=107, swing_high=True),    # 7: 2nd swing_high → PH=111
            _row(109, high=110, low=108),                     # 8: neutral
        ])
        result = detect_market_structure(df, "H1")

        # Trend stays BULLISH across all post-BOS rows
        for i in range(4, 9):
            assert result.loc[i, "trend_state"] == STATE_BULLISH, \
                f"row {i}: expected BULLISH, got {result.loc[i, 'trend_state']}"

        # After first swing_high (row 5): PH advances, PL set from last swing_low=79
        assert result.loc[5, "protected_high"] == 106.0
        assert result.loc[5, "protected_low"]  == 79.0

        # After second swing_high (row 7): PH advances again, PL updates to 103
        assert result.loc[7, "protected_high"] == 111.0
        assert result.loc[7, "protected_low"]  == 103.0

        # Neutral candle (row 8): protected levels unchanged
        assert result.loc[8, "protected_high"] == 111.0
        assert result.loc[8, "protected_low"]  == 103.0
