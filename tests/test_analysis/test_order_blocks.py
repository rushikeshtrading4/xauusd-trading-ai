"""Comprehensive test suite for analysis/order_blocks.py — detect_order_blocks().

ATR = 10.0 throughout → displacement threshold = 1.5 × 10 = 15.0 body size.

Bearish OB sequence:
    liquidity_sweep_high=True  →  (scan forward)  →  last bullish candle
                               →  bearish displacement (body > 15)
                               →  mark last bullish as bearish_order_block

Bullish OB sequence:
    liquidity_sweep_low=True   →  (scan forward)  →  last bearish candle
                               →  bullish displacement (body > 15)
                               →  mark last bearish as bullish_order_block

Group overview
--------------
 1. TestInputValidation            (6)  — missing columns, empty df
 2. TestOutputSchema               (6)  — column names, dtypes, NaN defaults
 3. TestNoSweep                    (2)  — no sweep → no OB ever
 4. TestSweepNoDisplacement        (2)  — sweep but no strong candle → no OB
 5. TestSweepNoOppositeCandle      (2)  — sweep + displacement but no prior opp. candle
 6. TestBearishOBBasic             (4)  — high sweep → bullish → bearish disp → OB
 7. TestBullishOBBasic             (4)  — low sweep → bearish → bullish disp → OB
 8. TestDisplacementATRFilter      (4)  — body ≤ threshold not displacement
 9. TestLastOppositeCandleUsed     (3)  — multiple candidates → last one chosen
10. TestSweepResetsState           (4)  — new sweep supersedes old pending state
11. TestMultipleOBs                (3)  — multiple sweep+displacement pairs
12. TestNaNATR                     (2)  — NaN ATR → no displacement possible
13. TestIndependentHighLow         (3)  — bearish/bullish OB states are independent
14. TestOBHighLowValues            (3)  — ob_high/ob_low are H/L of OB candle
15. TestDfNotMutated               (1)  — original df unchanged
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.order_blocks import detect_order_blocks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = pd.Timestamp("2024-01-01 09:00:00")
_STEP    = pd.Timedelta(minutes=5)
_ATR     = 10.0   # displacement threshold = 1.5 × 10 = 15.0


def _row(
    i: int,
    open_:     float = 1900.0,
    high:      float = 1910.0,
    low:       float = 1890.0,
    close:     float = 1900.0,    # doji by default (neither bull nor bear)
    atr:       float = _ATR,
    sw_high:   bool  = False,
    sw_low:    bool  = False,
) -> dict:
    return {
        "timestamp":            _BASE_TS + _STEP * i,
        "open":                 open_,
        "high":                 high,
        "low":                  low,
        "close":                close,
        "ATR":                  atr,
        "liquidity_sweep_high": sw_high,
        "liquidity_sweep_low":  sw_low,
    }


def _bullish(i: int, body: float = 5.0, atr: float = _ATR, **kw) -> dict:
    """Bullish candle with given body size (close - open = body)."""
    open_ = kw.pop("open_", 1900.0)
    close = open_ + body
    high  = kw.pop("high", close + 2.0)
    low   = kw.pop("low",  open_ - 2.0)
    return _row(i, open_=open_, high=high, low=low, close=close, atr=atr, **kw)


def _bearish(i: int, body: float = 5.0, atr: float = _ATR, **kw) -> dict:
    """Bearish candle with given body size (open - close = body)."""
    open_ = kw.pop("open_", 1910.0)
    close = open_ - body
    high  = kw.pop("high", open_ + 2.0)
    low   = kw.pop("low",  close - 2.0)
    return _row(i, open_=open_, high=high, low=low, close=close, atr=atr, **kw)


def _strong_bearish(i: int, atr: float = _ATR, **kw) -> dict:
    """Bearish displacement candle — body = 1.6 × ATR (> threshold)."""
    return _bearish(i, body=atr * 1.6, atr=atr, **kw)


def _strong_bullish(i: int, atr: float = _ATR, **kw) -> dict:
    """Bullish displacement candle — body = 1.6 × ATR (> threshold)."""
    return _bullish(i, body=atr * 1.6, atr=atr, **kw)


def _weak_bearish(i: int, atr: float = _ATR, **kw) -> dict:
    """Bearish non-displacement candle — body = 0.5 × ATR (< threshold)."""
    return _bearish(i, body=atr * 0.5, atr=atr, **kw)


def _weak_bullish(i: int, atr: float = _ATR, **kw) -> dict:
    """Bullish non-displacement candle — body = 0.5 × ATR (< threshold)."""
    return _bullish(i, body=atr * 0.5, atr=atr, **kw)


def _doji(i: int, atr: float = _ATR) -> dict:
    """Doji — close == open, neither bullish nor bearish."""
    return _row(i, open_=1905.0, high=1910.0, low=1900.0, close=1905.0, atr=atr)


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_raises_on_missing_open(self):
        df = _df([_row(0)]).drop(columns=["open"])
        with pytest.raises(ValueError, match="open"):
            detect_order_blocks(df)

    def test_raises_on_missing_atr(self):
        df = _df([_row(0)]).drop(columns=["ATR"])
        with pytest.raises(ValueError, match="ATR"):
            detect_order_blocks(df)

    def test_raises_on_missing_sweep_high(self):
        df = _df([_row(0)]).drop(columns=["liquidity_sweep_high"])
        with pytest.raises(ValueError, match="liquidity_sweep_high"):
            detect_order_blocks(df)

    def test_raises_on_missing_sweep_low(self):
        df = _df([_row(0)]).drop(columns=["liquidity_sweep_low"])
        with pytest.raises(ValueError, match="liquidity_sweep_low"):
            detect_order_blocks(df)

    def test_raises_on_empty_dataframe(self):
        df = _df([_row(0)]).iloc[0:0]
        with pytest.raises(ValueError, match="empty"):
            detect_order_blocks(df)

    def test_error_lists_all_missing(self):
        df = _df([_row(0)]).drop(columns=["ATR", "liquidity_sweep_high"])
        with pytest.raises(ValueError) as exc:
            detect_order_blocks(df)
        assert "ATR" in str(exc.value)
        assert "liquidity_sweep_high" in str(exc.value)


# ---------------------------------------------------------------------------
# 2. Output Schema
# ---------------------------------------------------------------------------

class TestOutputSchema:

    def _result(self):
        return detect_order_blocks(_df([_row(0)]))

    def test_has_bullish_order_block(self):
        assert "bullish_order_block" in self._result().columns

    def test_has_bearish_order_block(self):
        assert "bearish_order_block" in self._result().columns

    def test_has_ob_high(self):
        assert "ob_high" in self._result().columns

    def test_has_ob_low(self):
        assert "ob_low" in self._result().columns

    def test_ob_columns_are_bool(self):
        r = self._result()
        assert r["bullish_order_block"].dtype == bool
        assert r["bearish_order_block"].dtype == bool

    def test_ob_price_columns_default_nan(self):
        r = self._result()
        assert np.isnan(r["ob_high"].iloc[0])
        assert np.isnan(r["ob_low"].iloc[0])


# ---------------------------------------------------------------------------
# 3. No Sweep → No OB
# ---------------------------------------------------------------------------

class TestNoSweep:

    def test_no_bearish_ob_without_high_sweep(self):
        rows = [
            _bullish(0),
            _strong_bearish(1),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bearish_order_block"].sum() == 0

    def test_no_bullish_ob_without_low_sweep(self):
        rows = [
            _bearish(0),
            _strong_bullish(1),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bullish_order_block"].sum() == 0


# ---------------------------------------------------------------------------
# 4. Sweep But No Displacement
# ---------------------------------------------------------------------------

class TestSweepNoDisplacement:

    def test_high_sweep_weak_candles_no_bearish_ob(self):
        rows = [
            _row(0, sw_high=True),
            _bullish(1),
            _weak_bearish(2),   # body < 1.5×ATR → not displacement
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bearish_order_block"].sum() == 0

    def test_low_sweep_weak_candles_no_bullish_ob(self):
        rows = [
            _row(0, sw_low=True),
            _bearish(1),
            _weak_bullish(2),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bullish_order_block"].sum() == 0


# ---------------------------------------------------------------------------
# 5. Sweep + Displacement But No Opposite Candle
# ---------------------------------------------------------------------------

class TestSweepNoOppositeCandle:

    def test_bearish_ob_needs_prior_bullish_candle(self):
        # Sweep, then straight to bearish displacement with no bullish in between.
        rows = [
            _row(0, sw_high=True),
            _strong_bearish(1),   # displacement, but no bullish candle tracked
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bearish_order_block"].sum() == 0

    def test_bullish_ob_needs_prior_bearish_candle(self):
        rows = [
            _row(0, sw_low=True),
            _strong_bullish(1),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bullish_order_block"].sum() == 0


# ---------------------------------------------------------------------------
# 6. Bearish OB Basic
# ---------------------------------------------------------------------------

class TestBearishOBBasic:
    """High sweep → bullish candle → bearish displacement → bearish OB."""

    def _basic(self):
        return [
            _row(0, sw_high=True),          # arm
            _bullish(1),                     # OB candidate (row 1)
            _strong_bearish(2),              # displacement
        ]

    def test_bearish_ob_marked_on_correct_row(self):
        r = detect_order_blocks(_df(self._basic()))
        assert r["bearish_order_block"].iloc[1] is np.bool_(True)

    def test_bearish_ob_not_on_sweep_row(self):
        r = detect_order_blocks(_df(self._basic()))
        assert r["bearish_order_block"].iloc[0] is np.bool_(False)

    def test_bearish_ob_not_on_displacement_row(self):
        r = detect_order_blocks(_df(self._basic()))
        assert r["bearish_order_block"].iloc[2] is np.bool_(False)

    def test_bearish_ob_count_is_one(self):
        r = detect_order_blocks(_df(self._basic()))
        assert r["bearish_order_block"].sum() == 1


# ---------------------------------------------------------------------------
# 7. Bullish OB Basic
# ---------------------------------------------------------------------------

class TestBullishOBBasic:
    """Low sweep → bearish candle → bullish displacement → bullish OB."""

    def _basic(self):
        return [
            _row(0, sw_low=True),
            _bearish(1),
            _strong_bullish(2),
        ]

    def test_bullish_ob_marked_on_correct_row(self):
        r = detect_order_blocks(_df(self._basic()))
        assert r["bullish_order_block"].iloc[1] is np.bool_(True)

    def test_bullish_ob_not_on_sweep_row(self):
        r = detect_order_blocks(_df(self._basic()))
        assert r["bullish_order_block"].iloc[0] is np.bool_(False)

    def test_bullish_ob_not_on_displacement_row(self):
        r = detect_order_blocks(_df(self._basic()))
        assert r["bullish_order_block"].iloc[2] is np.bool_(False)

    def test_bullish_ob_count_is_one(self):
        r = detect_order_blocks(_df(self._basic()))
        assert r["bullish_order_block"].sum() == 1


# ---------------------------------------------------------------------------
# 8. Displacement ATR Filter
# ---------------------------------------------------------------------------

class TestDisplacementATRFilter:

    def test_body_just_below_threshold_not_displacement(self):
        # body = 1.5 × ATR exactly → NOT > threshold (strict)
        rows = [
            _row(0, sw_high=True),
            _bullish(1),
            _bearish(2, body=_ATR * 1.5),   # == threshold, not >
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bearish_order_block"].sum() == 0

    def test_body_just_above_threshold_is_displacement(self):
        rows = [
            _row(0, sw_high=True),
            _bullish(1),
            _bearish(2, body=_ATR * 1.5 + 0.01),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bearish_order_block"].sum() == 1

    def test_weak_bullish_not_displacement_for_bullish_ob(self):
        rows = [
            _row(0, sw_low=True),
            _bearish(1),
            _bullish(2, body=_ATR * 1.5),   # == threshold, not >
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bullish_order_block"].sum() == 0

    def test_strong_bullish_is_displacement_for_bullish_ob(self):
        rows = [
            _row(0, sw_low=True),
            _bearish(1),
            _bullish(2, body=_ATR * 1.5 + 0.01),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bullish_order_block"].sum() == 1


# ---------------------------------------------------------------------------
# 9. Last Opposite Candle Used
# ---------------------------------------------------------------------------

class TestLastOppositeCandleUsed:
    """When multiple opposite candles appear, the last one is the OB."""

    def test_last_bullish_before_displacement_is_ob(self):
        rows = [
            _row(0, sw_high=True),
            _bullish(1),              # first candidate
            _bullish(2),              # second candidate  ← this should be OB
            _strong_bearish(3),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bearish_order_block"].iloc[1] is np.bool_(False)
        assert r["bearish_order_block"].iloc[2] is np.bool_(True)

    def test_last_bearish_before_displacement_is_ob(self):
        rows = [
            _row(0, sw_low=True),
            _bearish(1),
            _bearish(2),              # ← this should be OB
            _strong_bullish(3),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bullish_order_block"].iloc[1] is np.bool_(False)
        assert r["bullish_order_block"].iloc[2] is np.bool_(True)

    def test_doji_between_candidates_does_not_clear_tracking(self):
        # Doji (close == open) is neither bullish nor bearish; tracking unchanged.
        rows = [
            _row(0, sw_high=True),
            _bullish(1),
            _doji(2),                 # should leave last_bullish_idx = 1
            _strong_bearish(3),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bearish_order_block"].iloc[1] is np.bool_(True)


# ---------------------------------------------------------------------------
# 10. Sweep Resets State
# ---------------------------------------------------------------------------

class TestSweepResetsState:

    def test_second_high_sweep_resets_bullish_tracking(self):
        """
        Sequence:
            Row 0: sweep_high=True  → pending; last_bullish = None
            Row 1: bullish          → last_bullish = 1
            Row 2: sweep_high=True  → reset; last_bullish = None
            Row 3: strong bearish   → displacement; last_bullish = None → no OB
        """
        rows = [
            _row(0, sw_high=True),
            _bullish(1),
            _row(2, sw_high=True),   # re-arm: clears tracked candle
            _strong_bearish(3),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bearish_order_block"].sum() == 0

    def test_second_sweep_then_new_ob_formed(self):
        """After reset, a new bullish candle tracked → new OB at row 3."""
        rows = [
            _row(0, sw_high=True),
            _bullish(1),
            _row(2, sw_high=True),   # reset
            _bullish(3),             # new candidate
            _strong_bearish(4),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bearish_order_block"].iloc[1] is np.bool_(False)
        assert r["bearish_order_block"].iloc[3] is np.bool_(True)

    def test_low_sweep_reset_clears_bearish_tracking(self):
        rows = [
            _row(0, sw_low=True),
            _bearish(1),
            _row(2, sw_low=True),    # reset
            _strong_bullish(3),      # no bearish tracked → no OB
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bullish_order_block"].sum() == 0

    def test_state_machine_disarmed_after_displacement(self):
        """After OB is found the state machine disarms. Additional candles do not
        produce another OB from the same sweep."""
        rows = [
            _row(0, sw_high=True),
            _bullish(1),
            _strong_bearish(2),      # displacement → OB at row 1; disarmed
            _bullish(3),             # would be candidate but state is off
            _strong_bearish(4),      # another displacement — ignored (no pending)
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bearish_order_block"].sum() == 1
        assert r["bearish_order_block"].iloc[1] is np.bool_(True)


# ---------------------------------------------------------------------------
# 11. Multiple OBs
# ---------------------------------------------------------------------------

class TestMultipleOBs:

    def test_two_bearish_obs_from_two_sweeps(self):
        rows = [
            _row(0, sw_high=True),
            _bullish(1),
            _strong_bearish(2),      # OB at row 1
            _row(3, sw_high=True),   # new sweep
            _bullish(4),
            _strong_bearish(5),      # OB at row 4
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bearish_order_block"].sum() == 2
        assert r["bearish_order_block"].iloc[1] is np.bool_(True)
        assert r["bearish_order_block"].iloc[4] is np.bool_(True)

    def test_two_bullish_obs_from_two_sweeps(self):
        rows = [
            _row(0, sw_low=True),
            _bearish(1),
            _strong_bullish(2),
            _row(3, sw_low=True),
            _bearish(4),
            _strong_bullish(5),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bullish_order_block"].sum() == 2

    def test_bearish_and_bullish_ob_in_same_df(self):
        rows = [
            _row(0, sw_high=True),
            _bullish(1),
            _strong_bearish(2),      # bearish OB at row 1
            _row(3, sw_low=True),
            _bearish(4),
            _strong_bullish(5),      # bullish OB at row 4
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bearish_order_block"].iloc[1] is np.bool_(True)
        assert r["bullish_order_block"].iloc[4] is np.bool_(True)


# ---------------------------------------------------------------------------
# 12. NaN ATR
# ---------------------------------------------------------------------------

class TestNaNATR:

    def test_nan_atr_prevents_displacement_bearish(self):
        rows = [
            _row(0, sw_high=True),
            _bullish(1, atr=float("nan")),
            _bearish(2, body=100.0, atr=float("nan")),  # large body but NaN ATR
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bearish_order_block"].sum() == 0

    def test_nan_atr_prevents_displacement_bullish(self):
        rows = [
            _row(0, sw_low=True),
            _bearish(1, atr=float("nan")),
            _bullish(2, body=100.0, atr=float("nan")),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bullish_order_block"].sum() == 0


# ---------------------------------------------------------------------------
# 13. Independent High/Low Tracking
# ---------------------------------------------------------------------------

class TestIndependentHighLow:

    def test_low_events_do_not_affect_bearish_ob_search(self):
        """sw_low and sw_high are tracked by separate state machines."""
        rows = [
            _row(0, sw_high=True),
            _row(1, sw_low=True),    # arms bullish state; bearish state unaffected
            _bullish(2),
            _strong_bearish(3),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bearish_order_block"].iloc[2] is np.bool_(True)

    def test_high_events_do_not_affect_bullish_ob_search(self):
        rows = [
            _row(0, sw_low=True),
            _row(1, sw_high=True),
            _bearish(2),
            _strong_bullish(3),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bullish_order_block"].iloc[2] is np.bool_(True)

    def test_simultaneous_sweep_produces_both_obs(self):
        """A row with both sw_high=True and sw_low=True arms both machines."""
        rows = [
            _row(0, sw_high=True, sw_low=True),   # arms both
            _bullish(1),                            # tracked for bearish OB
            _bearish(2),                            # tracked for bullish OB
            _strong_bearish(3),                     # bearish displacement → OB at row 1
            _row(4, sw_low=True),                   # re-arm bullish (resets tracking)
            _bearish(5),
            _strong_bullish(6),                     # bullish displacement → OB at row 5
        ]
        r = detect_order_blocks(_df(rows))
        assert r["bearish_order_block"].iloc[1] is np.bool_(True)
        assert r["bullish_order_block"].iloc[5] is np.bool_(True)


# ---------------------------------------------------------------------------
# 14. OB High / Low Values
# ---------------------------------------------------------------------------

class TestOBHighLowValues:

    def test_bearish_ob_high_low_match_ob_candle(self):
        rows = [
            _row(0, sw_high=True),
            _row(1, open_=1900.0, high=1915.0, low=1897.0, close=1908.0),  # bullish
            _strong_bearish(2),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["ob_high"].iloc[1] == pytest.approx(1915.0)
        assert r["ob_low"].iloc[1]  == pytest.approx(1897.0)

    def test_bullish_ob_high_low_match_ob_candle(self):
        rows = [
            _row(0, sw_low=True),
            _row(1, open_=1910.0, high=1913.0, low=1893.0, close=1903.0),  # bearish
            _strong_bullish(2),
        ]
        r = detect_order_blocks(_df(rows))
        assert r["ob_high"].iloc[1] == pytest.approx(1913.0)
        assert r["ob_low"].iloc[1]  == pytest.approx(1893.0)

    def test_non_ob_rows_have_nan_prices(self):
        rows = [
            _row(0, sw_high=True),
            _bullish(1),
            _strong_bearish(2),
        ]
        r = detect_order_blocks(_df(rows))
        assert np.isnan(r["ob_high"].iloc[0])
        assert np.isnan(r["ob_high"].iloc[2])
        assert np.isnan(r["ob_low"].iloc[0])
        assert np.isnan(r["ob_low"].iloc[2])


# ---------------------------------------------------------------------------
# 15. Original DataFrame Not Mutated
# ---------------------------------------------------------------------------

class TestDfNotMutated:

    def test_original_df_unchanged(self):
        df = _df([
            _row(0, sw_high=True),
            _bullish(1),
            _strong_bearish(2),
        ])
        original_cols = list(df.columns)
        _ = detect_order_blocks(df)
        assert list(df.columns) == original_cols
        assert "bullish_order_block" not in df.columns
        assert "bearish_order_block" not in df.columns
        assert "ob_high" not in df.columns
        assert "ob_low" not in df.columns
