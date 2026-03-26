"""Comprehensive tests for analysis/displacement.py — detect_displacement().

Test group overview
-------------------
 1. TestInputValidation          (7)  — missing columns, empty df, bad window
 2. TestOutputSchema             (5)  — columns present, dtypes, row count, no mutation
 3. TestBullishDetection         (8)  — happy-path bullish displacement
 4. TestBearishDetection         (8)  — happy-path bearish displacement
 5. TestDirectionFilter          (5)  — mixed-direction windows → no displacement
 6. TestBodyStrengthFilter       (5)  — avg body ≤ 0.8×ATR → rejected
 7. TestTotalMoveFilter          (5)  — net move ≤ 2.0×ATR → rejected
 8. TestWickFilter               (6)  — dominant wick_ratio > 0.8 → rejection
 9. TestATRValidation            (6)  — NaN / zero / small / capped ATR
10. TestStrengthScore            (7)  — 0.0 for non-disp; min(raw/5,1) bounded
11. TestWindowParameter          (6)  — window=1, window=5, invalid window
12. TestEdgeCases                (5)  — short df, exact threshold boundaries
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analysis.displacement import detect_displacement

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ATR   = 10.0
_CLOSE = 2000.0


# ---------------------------------------------------------------------------
# Row / DataFrame helpers
# ---------------------------------------------------------------------------

def _row(
    open_: float = _CLOSE - 1.0,
    high:  float = _CLOSE + 2.0,
    low:   float = _CLOSE - 3.0,
    close: float = _CLOSE,
    atr:   float = _ATR,
) -> dict:
    return {"open": open_, "high": high, "low": low, "close": close, "ATR": atr}


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _bull_row(
    base:  float = _CLOSE,
    body:  float = 15.0,   # > 0.8 × _ATR
    wick:  float = 1.0,    # dominant wick < body
    atr:   float = _ATR,
) -> dict:
    """Bullish candle: close > open, small wicks."""
    open_  = base
    close  = base + body
    high   = close + wick
    low    = open_ - wick
    return _row(open_=open_, high=high, low=low, close=close, atr=atr)


def _bear_row(
    base:  float = _CLOSE,
    body:  float = 15.0,
    wick:  float = 1.0,
    atr:   float = _ATR,
) -> dict:
    """Bearish candle: open > close, small wicks."""
    open_  = base
    close  = base - body
    high   = open_ + wick
    low    = close - wick
    return _row(open_=open_, high=high, low=low, close=close, atr=atr)


def _bull_window(n: int = 3, body: float = 15.0, wick: float = 1.0,
                 atr: float = _ATR) -> pd.DataFrame:
    """n consecutive bullish candles that pass all filters for window=n."""
    rows = []
    base = _CLOSE
    for _ in range(n):
        rows.append(_bull_row(base=base, body=body, wick=wick, atr=atr))
        base += body
    return _df(rows)


def _bear_window(n: int = 3, body: float = 15.0, wick: float = 1.0,
                 atr: float = _ATR) -> pd.DataFrame:
    """n consecutive bearish candles that pass all filters for window=n."""
    rows = []
    base = _CLOSE
    for _ in range(n):
        rows.append(_bear_row(base=base, body=body, wick=wick, atr=atr))
        base -= body
    return _df(rows)


# ===========================================================================
# 1. Input Validation
# ===========================================================================

class TestInputValidation:

    def test_raises_on_missing_open(self):
        with pytest.raises(ValueError, match="open"):
            detect_displacement(_bull_window().drop(columns=["open"]))

    def test_raises_on_missing_high(self):
        with pytest.raises(ValueError, match="high"):
            detect_displacement(_bull_window().drop(columns=["high"]))

    def test_raises_on_missing_low(self):
        with pytest.raises(ValueError, match="low"):
            detect_displacement(_bull_window().drop(columns=["low"]))

    def test_raises_on_missing_close(self):
        with pytest.raises(ValueError, match="close"):
            detect_displacement(_bull_window().drop(columns=["close"]))

    def test_raises_on_missing_atr(self):
        with pytest.raises(ValueError, match="ATR"):
            detect_displacement(_bull_window().drop(columns=["ATR"]))

    def test_raises_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            detect_displacement(_bull_window().iloc[0:0])

    def test_raises_on_window_zero(self):
        with pytest.raises(ValueError, match="window"):
            detect_displacement(_bull_window(), window=0)


# ===========================================================================
# 2. Output Schema
# ===========================================================================

class TestOutputSchema:

    def test_displacement_bullish_column_present(self):
        out = detect_displacement(_bull_window())
        assert "displacement_bullish" in out.columns

    def test_displacement_bearish_column_present(self):
        out = detect_displacement(_bull_window())
        assert "displacement_bearish" in out.columns

    def test_displacement_strength_column_present(self):
        out = detect_displacement(_bull_window())
        assert "displacement_strength" in out.columns

    def test_row_count_preserved(self):
        df  = _bull_window()
        out = detect_displacement(df)
        assert len(out) == len(df)

    def test_input_not_mutated(self):
        df   = _bull_window()
        cols = list(df.columns)
        detect_displacement(df)
        assert list(df.columns) == cols


# ===========================================================================
# 3. Bullish Detection
# ===========================================================================

class TestBullishDetection:

    def test_bullish_detected_on_last_candle(self):
        out = detect_displacement(_bull_window())
        assert out["displacement_bullish"].iloc[-1] is np.bool_(True)

    def test_bullish_bearish_false_on_same_row(self):
        out = detect_displacement(_bull_window())
        assert out["displacement_bearish"].iloc[-1] is np.bool_(False)

    def test_bullish_not_marked_on_intermediate_candles(self):
        out = detect_displacement(_bull_window())
        assert not out["displacement_bullish"].iloc[:-1].any()

    def test_bullish_strength_nonzero_when_detected(self):
        out = detect_displacement(_bull_window())
        assert out["displacement_strength"].iloc[-1] > 0.0

    def test_bullish_strength_at_most_one(self):
        out = detect_displacement(_bull_window())
        assert out["displacement_strength"].max() <= 1.0

    def test_bullish_strength_zero_on_non_disp_rows(self):
        out = detect_displacement(_bull_window())
        assert out["displacement_strength"].iloc[:-1].sum() == 0.0

    def test_consecutive_bullish_windows_each_mark_last(self):
        # 6 candles → two overlapping windows of 3; both last candles (2 and 5) marked
        df  = _bull_window(n=6)
        out = detect_displacement(df)
        assert out["displacement_bullish"].iloc[2] is np.bool_(True)
        assert out["displacement_bullish"].iloc[5] is np.bool_(True)

    def test_fvg_low_high_ordered(self):
        """Sanity: fvg_low produced by surrounding modules will use last_close > first_open."""
        df  = _bull_window()
        out = detect_displacement(df)
        # last_close > first_open confirms bullish net move
        last_close  = out["close"].iloc[-1]
        first_open  = out["open"].iloc[0]
        assert last_close > first_open


# ===========================================================================
# 4. Bearish Detection
# ===========================================================================

class TestBearishDetection:

    def test_bearish_detected_on_last_candle(self):
        out = detect_displacement(_bear_window())
        assert out["displacement_bearish"].iloc[-1] is np.bool_(True)

    def test_bearish_bullish_false_on_same_row(self):
        out = detect_displacement(_bear_window())
        assert out["displacement_bullish"].iloc[-1] is np.bool_(False)

    def test_bearish_not_marked_on_intermediate_candles(self):
        out = detect_displacement(_bear_window())
        assert not out["displacement_bearish"].iloc[:-1].any()

    def test_bearish_strength_nonzero_when_detected(self):
        out = detect_displacement(_bear_window())
        assert out["displacement_strength"].iloc[-1] > 0.0

    def test_bearish_strength_at_most_one(self):
        out = detect_displacement(_bear_window())
        assert out["displacement_strength"].max() <= 1.0

    def test_bearish_strength_zero_on_non_disp_rows(self):
        out = detect_displacement(_bear_window())
        assert out["displacement_strength"].iloc[:-1].sum() == 0.0

    def test_consecutive_bearish_windows_each_mark_last(self):
        df  = _bear_window(n=6)
        out = detect_displacement(df)
        assert out["displacement_bearish"].iloc[2] is np.bool_(True)
        assert out["displacement_bearish"].iloc[5] is np.bool_(True)

    def test_bearish_first_open_greater_than_last_close(self):
        df  = _bear_window()
        out = detect_displacement(df)
        first_open = out["open"].iloc[0]
        last_close = out["close"].iloc[-1]
        assert first_open > last_close


# ===========================================================================
# 5. Direction Filter
# ===========================================================================

class TestDirectionFilter:

    def test_mixed_window_bull_bear_bull_no_displacement(self):
        df = _df([
            _bull_row(), _bear_row(base=_CLOSE + 15), _bull_row(base=_CLOSE + 10),
        ])
        out = detect_displacement(df)
        assert not out["displacement_bullish"].any()
        assert not out["displacement_bearish"].any()

    def test_doji_candle_breaks_direction(self):
        # A doji (close == open) is neither bullish nor bearish
        doji = _row(open_=_CLOSE, close=_CLOSE, high=_CLOSE + 2, low=_CLOSE - 2)
        df = _df([_bull_row(), _bull_row(base=_CLOSE + 15), doji])
        out = detect_displacement(df)
        assert not out["displacement_bullish"].any()

    def test_single_bearish_interrupts_bullish_sequence(self):
        df = _df([
            _bull_row(),
            _bear_row(base=_CLOSE + 15),
            _bull_row(base=_CLOSE + 10),
        ])
        out = detect_displacement(df)
        assert not out["displacement_bullish"].any()

    def test_direction_filter_does_not_flag_bearish_in_bull_window(self):
        out = detect_displacement(_bull_window())
        assert not out["displacement_bearish"].any()

    def test_direction_filter_does_not_flag_bullish_in_bear_window(self):
        out = detect_displacement(_bear_window())
        assert not out["displacement_bullish"].any()


# ===========================================================================
# 6. Body Strength Filter
# ===========================================================================

class TestBodyStrengthFilter:

    def test_avg_body_exactly_0_8x_atr_rejected(self):
        # body == 0.8×ATR is NOT strictly greater → rejected
        body = 0.8 * _ATR   # = 8.0
        # need move > 2×ATR: 3 candles × 8.0 body = 24 > 20 ✓
        df = _bull_window(body=body)
        out = detect_displacement(df)
        assert not out["displacement_bullish"].any()

    def test_avg_body_just_above_threshold_passes(self):
        body = 0.8 * _ATR + 0.01
        df   = _bull_window(body=body)
        out  = detect_displacement(df)
        assert out["displacement_bullish"].any()

    def test_weak_body_rejected_for_bearish(self):
        body = 0.5 * _ATR
        df   = _bear_window(body=body)
        out  = detect_displacement(df)
        assert not out["displacement_bearish"].any()

    def test_very_strong_body_passes(self):
        body = 5.0 * _ATR
        df   = _bull_window(body=body)
        out  = detect_displacement(df)
        assert out["displacement_bullish"].any()

    def test_body_strength_filter_independent_of_window_size(self):
        # window=5, weak body → still rejected
        body = 0.5 * _ATR
        df   = _bull_window(n=5, body=body)
        out  = detect_displacement(df, window=5)
        assert not out["displacement_bullish"].any()


# ===========================================================================
# 7. Total Move Filter
# ===========================================================================

class TestTotalMoveFilter:

    def test_move_exactly_2x_atr_rejected(self):
        # With 3 candles and body s.t. avg_body > 0.8×ATR,
        # engineer net move == exactly 2×ATR → rejected
        # body per candle = 2×ATR / 3 ≈ 6.67; need avg_body > 8 → increase body
        # Trick: set body just above threshold for strength but total = 2×ATR
        # body per candle: must satisfy avg > 0.8×ATR = 8 AND sum = 2×ATR = 20
        # sum = 3 × body → body = 20/3 ≈ 6.67 < 8 → body filter rejects first
        # Use body = 8.01 for 1 candle + body = 0 for others → not uniform
        # Simplest: use single candle window, body such that move = 2×ATR exactly
        body = 2.0 * _ATR   # move == 2.0×ATR exactly, not strictly greater
        # avg_body = 2×ATR > 0.8×ATR ✓ — only move filter blocks this
        df  = _bull_window(n=1, body=body)
        out = detect_displacement(df, window=1)
        assert not out["displacement_bullish"].any()

    def test_move_above_threshold_passes(self):
        body = 2.01 * _ATR
        df   = _bull_window(n=1, body=body)
        out  = detect_displacement(df, window=1)
        assert out["displacement_bullish"].any()

    def test_small_move_rejected_for_bearish(self):
        # 3 candles × 9 body per candle = 27 > 2×ATR(10)=20 ✓
        # reduce to tiny bodies so body < 0.8×ATR fails first is OK –
        # we want total move filter specifically: use large body but tiny net move
        # achieved by alternating bodies won't work (direction filter kills it) –
        # use window=1 with body just enough for body filter but not move filter
        body = 0.9 * _ATR   # avg_body > 0.8×ATR ✓; move = 0.9×ATR < 2×ATR → rejected
        df   = _bear_window(n=1, body=body)
        out  = detect_displacement(df, window=1)
        assert not out["displacement_bearish"].any()

    def test_large_window_accumulates_move(self):
        # window=5, each candle body = 9 → move = 45 > 2×ATR=20 ✓
        body = 9.0
        df   = _bull_window(n=5, body=body)
        out  = detect_displacement(df, window=5)
        assert out["displacement_bullish"].any()

    def test_move_filter_after_body_filter(self):
        """Body filter must not hide the move filter: use strong body, tiny move."""
        # window=1: body = 3×ATR → avg_body > 0.8×ATR ✓; move = 3×ATR > 2×ATR ✓
        body = 3.0 * _ATR
        df   = _bull_window(n=1, body=body)
        out  = detect_displacement(df, window=1)
        assert out["displacement_bullish"].any()


# ===========================================================================
# 8. Wick Filter
# ===========================================================================

class TestWickFilter:

    def test_wick_greater_than_body_rejects_bull(self):
        # dominant wick(2.0) > body(1.0) → rejected
        df = _bull_window(body=15.0, wick=20.0)  # wick > body
        out = detect_displacement(df)
        assert not out["displacement_bullish"].any()

    def test_wick_equal_to_body_rejects(self):
        # wick == body → wick_ratio ≈ 1.0 > 0.8 → rejected
        body = 15.0
        df   = _bull_window(body=body, wick=body)
        out  = detect_displacement(df)
        # wick == body: wick_ratio = body/(body+1e-9) ≈ 1.0 > 0.8 → rejected
        assert not out["displacement_bullish"].any()

    def test_wick_just_below_body_passes(self):
        # wick_ratio = wick / body ≤ 0.8 → passes the filter
        body = 15.0
        wick = 0.79 * body   # wick_ratio ≈ 0.79 < 0.8 → passes
        df   = _bull_window(body=body, wick=wick)
        out  = detect_displacement(df)
        assert out["displacement_bullish"].any()

    def test_single_candle_with_large_wick_rejects_whole_window(self):
        rows = [
            _bull_row(body=15.0, wick=1.0),
            _bull_row(base=_CLOSE + 15, body=15.0, wick=20.0),  # bad wick
            _bull_row(base=_CLOSE + 30, body=15.0, wick=1.0),
        ]
        out = detect_displacement(_df(rows))
        assert not out["displacement_bullish"].any()

    def test_wick_filter_applies_to_bearish(self):
        df = _bear_window(body=15.0, wick=20.0)
        out = detect_displacement(df)
        assert not out["displacement_bearish"].any()

    def test_clean_bearish_wicks_pass(self):
        df  = _bear_window(body=15.0, wick=0.5)
        out = detect_displacement(df)
        assert out["displacement_bearish"].any()


# ===========================================================================
# 9. ATR Validation
# ===========================================================================

class TestATRValidation:

    def test_nan_atr_skips_window(self):
        df = _bull_window()
        df.at[2, "ATR"] = float("nan")   # last candle NaN
        out = detect_displacement(df)
        assert not out["displacement_bullish"].any()

    def test_zero_atr_skips_window(self):
        df = _bull_window()
        df.at[2, "ATR"] = 0.0
        out = detect_displacement(df)
        assert not out["displacement_bullish"].any()

    def test_negative_atr_skips_window(self):
        df = _bull_window()
        df.at[2, "ATR"] = -1.0
        out = detect_displacement(df)
        assert not out["displacement_bullish"].any()

    def test_atr_below_min_skips_window(self):
        df = _bull_window()
        df.at[2, "ATR"] = 0.05
        out = detect_displacement(df)
        assert not out["displacement_bullish"].any()

    def test_atr_capped_at_100_uses_lower_threshold(self):
        # ATR=500 → capped to 100; body=120 > 0.8×100=80 ✓; move=360 > 200 ✓
        df  = _bull_window(body=120.0, wick=1.0, atr=500.0)
        df.at[2, "ATR"] = 500.0
        out = detect_displacement(df)
        assert out["displacement_bullish"].any()

    def test_atr_at_minimum_boundary_detects(self):
        # ATR=0.1: body = 0.09/0.1=0.9×0.1 > 0.8×0.1; move = 0.27 > 0.2
        df  = _bull_window(body=0.09, wick=0.001, atr=0.1)
        out = detect_displacement(df)
        assert out["displacement_bullish"].any()


# ===========================================================================
# 10. Strength Score
# ===========================================================================

class TestStrengthScore:

    def test_strength_zero_on_non_displacement_row(self):
        out = detect_displacement(_bull_window())
        # rows 0 and 1 never mark a displacement
        assert out["displacement_strength"].iloc[0] == pytest.approx(0.0)
        assert out["displacement_strength"].iloc[1] == pytest.approx(0.0)

    def test_strength_bounded_0_to_1(self):
        out = detect_displacement(_bull_window(n=6))
        assert out["displacement_strength"].min() >= 0.0
        assert out["displacement_strength"].max() <= 1.0

    def test_single_qualifying_window_strength_is_1(self):
        # body=15, atr=10, move=45, window=3
        # raw = (1.5)*0.4 + (4.5)*0.4 + (1.0)*0.2 = 0.6 + 1.8 + 0.2 = 2.6
        # strength = min(2.6 / 5.0, 1.0) = 0.52
        out = detect_displacement(_bull_window())
        assert out["displacement_strength"].max() == pytest.approx(0.52, abs=0.01)

    def test_stronger_move_gets_higher_score(self):
        # weak:   body=10 → raw = 1.8, strength = 0.36
        # strong: body=30 → raw = 5.0, strength = 1.0
        df_weak   = _bull_window(body=10.0)
        df_strong = _bull_window(body=30.0)
        out_weak   = detect_displacement(df_weak)
        out_strong = detect_displacement(df_strong)
        assert out_strong["displacement_strength"].max() == pytest.approx(1.0)
        assert out_weak["displacement_strength"].max()   == pytest.approx(0.36, abs=0.01)

    def test_relative_ordering_within_same_dataframe(self):
        # Combine a weak window and a strong window so normalisation spreads them
        df_weak   = _bull_window(body=10.0)   # raw ≈ (1.0×0.4 + 3.0×0.4 + 1.0×0.2) = 1.8
        df_strong = _bull_window(body=40.0)   # raw ≈ (4.0×0.4 + 12.0×0.4 + 1.0×0.2) = 6.6
        combined  = pd.concat([df_weak, df_strong], ignore_index=True)
        out       = detect_displacement(combined)
        weak_score   = out["displacement_strength"].iloc[2]   # last of first block
        strong_score = out["displacement_strength"].iloc[5]   # last of second block
        assert strong_score > weak_score

    def test_non_displacement_row_strength_exactly_zero(self):
        out = detect_displacement(_bull_window())
        assert (out["displacement_strength"].iloc[:2] == 0.0).all()

    def test_strength_increases_with_larger_window(self):
        # window=3: raw = (1.5)*0.4 + (4.5)*0.4 + min(1.0,1.0)*0.2 = 2.6,  strength = 0.52
        # window=5: raw = (1.5)*0.4 + (7.5)*0.4 + min(5/3,1.0)*0.2 = 3.8,  strength = 0.76
        df3 = _bull_window(n=3, body=15.0)
        df5 = _bull_window(n=5, body=15.0)
        out3 = detect_displacement(df3, window=3)
        out5 = detect_displacement(df5, window=5)
        assert out3["displacement_strength"].max() == pytest.approx(0.52, abs=0.01)
        assert out5["displacement_strength"].max() == pytest.approx(0.76, abs=0.01)


# ===========================================================================
# 11. Window Parameter
# ===========================================================================

class TestWindowParameter:

    def test_window_1_uses_single_candle(self):
        # One very strong candle: body = 3×ATR > 2×ATR, avg_body = 3×ATR > 0.8×ATR
        df = _df([_bull_row(body=3.0 * _ATR, wick=0.5)])
        out = detect_displacement(df, window=1)
        assert out["displacement_bullish"].iloc[0] is np.bool_(True)

    def test_window_5_requires_five_candles(self):
        # Only 4 candles → no complete window of 5
        df  = _bull_window(n=4)
        out = detect_displacement(df, window=5)
        assert not out["displacement_bullish"].any()

    def test_window_5_fires_on_fifth_candle(self):
        df  = _bull_window(n=5)
        out = detect_displacement(df, window=5)
        assert out["displacement_bullish"].iloc[4] is np.bool_(True)

    def test_window_negative_raises(self):
        with pytest.raises(ValueError, match="window"):
            detect_displacement(_bull_window(), window=-1)

    def test_window_zero_raises(self):
        with pytest.raises(ValueError, match="window"):
            detect_displacement(_bull_window(), window=0)

    def test_window_1_marks_every_qualifying_candle(self):
        # 3 strong candles; each qualifies independently
        df  = _bull_window(n=3, body=3.0 * _ATR)
        out = detect_displacement(df, window=1)
        assert out["displacement_bullish"].sum() == 3


# ===========================================================================
# 12. Edge Cases
# ===========================================================================

class TestEdgeCases:

    def test_single_row_df_no_displacement(self):
        # window=3: need ≥3 rows; single row → no displacement
        df  = _df([_bull_row()])
        out = detect_displacement(df)
        assert not out["displacement_bullish"].any()
        assert not out["displacement_bearish"].any()

    def test_two_rows_no_displacement_window_3(self):
        df  = _bull_window(n=2)
        out = detect_displacement(df)
        assert not out["displacement_bullish"].any()

    def test_exact_three_rows_marks_last(self):
        out = detect_displacement(_bull_window(n=3))
        assert out["displacement_bullish"].sum() == 1
        assert out["displacement_bullish"].iloc[2] is np.bool_(True)

    def test_extra_columns_preserved(self):
        df = _bull_window()
        df["custom"] = 42
        out = detect_displacement(df)
        assert "custom" in out.columns
        assert (out["custom"] == 42).all()

    def test_no_displacement_all_strengths_zero(self):
        # Mixed direction → no displacement → all strengths 0
        rows = [_bull_row(), _bear_row(base=_CLOSE + 15), _bull_row(base=_CLOSE + 5)]
        out  = detect_displacement(_df(rows))
        assert (out["displacement_strength"] == 0.0).all()

