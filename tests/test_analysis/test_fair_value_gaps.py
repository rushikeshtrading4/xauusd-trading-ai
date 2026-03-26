"""Comprehensive tests for analysis/fair_value_gaps.py — detect_fair_value_gaps().

Test group overview
-------------------
 1. TestInputValidation         (8)  — missing columns, empty DataFrame
 2. TestOutputSchema            (6)  — columns present, dtypes, no mutation
 3. TestBullishFVGDetection     (8)  — happy-path bullish detection
 4. TestBearishFVGDetection     (8)  — happy-path bearish detection
 5. TestDisplacementFilter      (5)  — bodies at and below 1.5×ATR → no FVG
 6. TestGapSizeFilter           (4)  — gaps below 0.5×ATR → no FVG; equality passes
 7. TestTrendFilter             (6)  — trend alignment gates
 8. TestATRValidation           (6)  — NaN/0/small/large ATR handling
 9. TestMitigation              (8)  — mitigated FVGs are removed
10. TestUnmitigated             (4)  — unmitigated FVGs survive
11. TestEdgeCases               (5)  — short DataFrames, NaN OHLC
12. TestDisplacementDependency  (5)  — displacement_bullish/bearish gates FVG
13. TestAgeFilter               (3)  — FVGs older than MAX_FVG_AGE invalidated
14. TestFVGStrength             (6)  — fvg_strength score bounded to [0, 1]
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analysis.fair_value_gaps import detect_fair_value_gaps

# ---------------------------------------------------------------------------
# Constants matching detection thresholds
# ---------------------------------------------------------------------------

_ATR   = 10.0   # default ATR for test rows
_CLOSE = 2000.0 # neutral reference price


# ---------------------------------------------------------------------------
# DataFrame builder helpers
# ---------------------------------------------------------------------------

def _row(
    open_:                float = _CLOSE - 1.0,
    high:                 float = _CLOSE + 5.0,
    low:                  float = _CLOSE - 5.0,
    close:                float = _CLOSE,
    atr:                  float = _ATR,
    trend_state:          str   = "BULLISH",
    displacement_bullish: bool  = False,
    displacement_bearish: bool  = False,
) -> dict:
    return {
        "open":                 open_,
        "high":                 high,
        "low":                  low,
        "close":                close,
        "ATR":                  atr,
        "trend_state":          trend_state,
        "displacement_bullish": displacement_bullish,
        "displacement_bearish": displacement_bearish,
    }


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _bull_fvg_df(
    *,
    c2_body_mult: float = 2.0,   # c2 body as multiple of ATR (displacement)
    gap_mult:     float = 1.0,   # gap (c3.low - c1.high) as multiple of ATR
    trend:        str   = "BULLISH",
    atr:          float = _ATR,
    c3_trend:     str | None = None,
) -> pd.DataFrame:
    """Three-row DataFrame producing a clean bullish FVG on row 2 (c3).

    c2 is a strong bullish displacement candle.
    Gap = gap_mult * atr between c1.high and c3.low.
    """
    c3_tr = c3_trend if c3_trend is not None else trend
    base     = _CLOSE
    c2_body  = c2_body_mult * atr
    gap      = gap_mult * atr

    # c1: high at base
    c1 = _row(open_=base - 2, high=base, low=base - 8, close=base - 1,
              atr=atr, trend_state=trend)
    # c2: bullish displacement, body = c2_body; open at base-1
    c2_open  = base - 1.0
    c2_close = c2_open + c2_body
    c2 = _row(open_=c2_open, high=c2_close + 2, low=c2_open - 2,
              close=c2_close, atr=atr, trend_state=trend,
              displacement_bullish=True)
    # c3: low at c1.high + gap
    c3_low   = base + gap
    c3_close = c3_low + 3.0
    c3 = _row(open_=c3_low + 1, high=c3_close + 2, low=c3_low,
              close=c3_close, atr=atr, trend_state=c3_tr)
    return _df([c1, c2, c3])


def _bear_fvg_df(
    *,
    c2_body_mult: float = 2.0,
    gap_mult:     float = 1.0,
    trend:        str   = "BEARISH",
    atr:          float = _ATR,
    c3_trend:     str | None = None,
) -> pd.DataFrame:
    """Three-row DataFrame producing a clean bearish FVG on row 2 (c3).

    c2 is a strong bearish displacement candle.
    Gap = gap_mult * atr between c3.high and c1.low.
    """
    c3_tr = c3_trend if c3_trend is not None else trend
    base     = _CLOSE
    c2_body  = c2_body_mult * atr
    gap      = gap_mult * atr

    # c1: low at base
    c1 = _row(open_=base + 2, high=base + 8, low=base, close=base + 1,
              atr=atr, trend_state=trend)
    # c2: bearish displacement, body = c2_body
    c2_open  = base + 1.0
    c2_close = c2_open - c2_body
    c2 = _row(open_=c2_open, high=c2_open + 2, low=c2_close - 2,
              close=c2_close, atr=atr, trend_state=trend,
              displacement_bearish=True)
    # c3: high at c1.low - gap
    c3_high  = base - gap
    c3_close = c3_high - 3.0
    c3 = _row(open_=c3_high - 1, high=c3_high, low=c3_close - 2,
              close=c3_close, atr=atr, trend_state=c3_tr)
    return _df([c1, c2, c3])


# ===========================================================================
# 1. Input Validation
# ===========================================================================

class TestInputValidation:

    def test_raises_on_missing_open(self):
        df = _bull_fvg_df().drop(columns=["open"])
        with pytest.raises(ValueError, match="open"):
            detect_fair_value_gaps(df)

    def test_raises_on_missing_high(self):
        df = _bull_fvg_df().drop(columns=["high"])
        with pytest.raises(ValueError, match="high"):
            detect_fair_value_gaps(df)

    def test_raises_on_missing_low(self):
        df = _bull_fvg_df().drop(columns=["low"])
        with pytest.raises(ValueError, match="low"):
            detect_fair_value_gaps(df)

    def test_raises_on_missing_atr(self):
        df = _bull_fvg_df().drop(columns=["ATR"])
        with pytest.raises(ValueError, match="ATR"):
            detect_fair_value_gaps(df)

    def test_raises_on_missing_trend_state(self):
        df = _bull_fvg_df().drop(columns=["trend_state"])
        with pytest.raises(ValueError, match="trend_state"):
            detect_fair_value_gaps(df)

    def test_raises_on_missing_displacement_bullish(self):
        df = _bull_fvg_df().drop(columns=["displacement_bullish"])
        with pytest.raises(ValueError, match="displacement_bullish"):
            detect_fair_value_gaps(df)

    def test_raises_on_missing_displacement_bearish(self):
        df = _bull_fvg_df().drop(columns=["displacement_bearish"])
        with pytest.raises(ValueError, match="displacement_bearish"):
            detect_fair_value_gaps(df)

    def test_raises_on_empty_dataframe(self):
        df = _bull_fvg_df().iloc[0:0]
        with pytest.raises(ValueError, match="empty"):
            detect_fair_value_gaps(df)


# ===========================================================================
# 2. Output Schema
# ===========================================================================

class TestOutputSchema:

    def test_fvg_bullish_column_present(self):
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert "fvg_bullish" in out.columns

    def test_fvg_bearish_column_present(self):
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert "fvg_bearish" in out.columns

    def test_fvg_low_column_present(self):
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert "fvg_low" in out.columns

    def test_fvg_high_column_present(self):
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert "fvg_high" in out.columns

    def test_input_dataframe_not_mutated(self):
        df  = _bull_fvg_df()
        original_cols = list(df.columns)
        detect_fair_value_gaps(df)
        assert list(df.columns) == original_cols

    def test_fvg_strength_column_present(self):
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert "fvg_strength" in out.columns


# ===========================================================================
# 3. Bullish FVG Detection
# ===========================================================================

class TestBullishFVGDetection:

    def test_bullish_fvg_detected_on_c3(self):
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert out["fvg_bullish"].iloc[2] is np.bool_(True)

    def test_bullish_fvg_bearish_is_false_on_c3(self):
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert out["fvg_bearish"].iloc[2] is np.bool_(False)

    def test_bullish_fvg_low_equals_c1_high(self):
        df  = _bull_fvg_df()
        out = detect_fair_value_gaps(df)
        assert out["fvg_low"].iloc[2] == pytest.approx(df["high"].iloc[0])

    def test_bullish_fvg_high_equals_c3_low(self):
        df  = _bull_fvg_df()
        out = detect_fair_value_gaps(df)
        assert out["fvg_high"].iloc[2] == pytest.approx(df["low"].iloc[2])

    def test_no_fvg_on_c1_or_c2(self):
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert not out["fvg_bullish"].iloc[0]
        assert not out["fvg_bullish"].iloc[1]

    def test_bullish_fvg_with_transition_trend(self):
        out = detect_fair_value_gaps(_bull_fvg_df(c3_trend="TRANSITION"))
        assert out["fvg_bullish"].iloc[2] is np.bool_(True)

    def test_bullish_fvg_low_high_not_nan_when_detected(self):
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert not math.isnan(out["fvg_low"].iloc[2])
        assert not math.isnan(out["fvg_high"].iloc[2])

    def test_non_fvg_rows_have_nan_fvg_low(self):
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert math.isnan(out["fvg_low"].iloc[0])
        assert math.isnan(out["fvg_low"].iloc[1])


# ===========================================================================
# 4. Bearish FVG Detection
# ===========================================================================

class TestBearishFVGDetection:

    def test_bearish_fvg_detected_on_c3(self):
        out = detect_fair_value_gaps(_bear_fvg_df())
        assert out["fvg_bearish"].iloc[2] is np.bool_(True)

    def test_bearish_fvg_bullish_is_false_on_c3(self):
        out = detect_fair_value_gaps(_bear_fvg_df())
        assert out["fvg_bullish"].iloc[2] is np.bool_(False)

    def test_bearish_fvg_low_equals_c3_high(self):
        df  = _bear_fvg_df()
        out = detect_fair_value_gaps(df)
        assert out["fvg_low"].iloc[2] == pytest.approx(df["high"].iloc[2])

    def test_bearish_fvg_high_equals_c1_low(self):
        df  = _bear_fvg_df()
        out = detect_fair_value_gaps(df)
        assert out["fvg_high"].iloc[2] == pytest.approx(df["low"].iloc[0])

    def test_no_fvg_on_c1_or_c2(self):
        out = detect_fair_value_gaps(_bear_fvg_df())
        assert not out["fvg_bearish"].iloc[0]
        assert not out["fvg_bearish"].iloc[1]

    def test_bearish_fvg_with_transition_trend(self):
        out = detect_fair_value_gaps(_bear_fvg_df(c3_trend="TRANSITION"))
        assert out["fvg_bearish"].iloc[2] is np.bool_(True)

    def test_bearish_fvg_low_high_not_nan_when_detected(self):
        out = detect_fair_value_gaps(_bear_fvg_df())
        assert not math.isnan(out["fvg_low"].iloc[2])
        assert not math.isnan(out["fvg_high"].iloc[2])

    def test_non_fvg_rows_have_nan_fvg_high(self):
        out = detect_fair_value_gaps(_bear_fvg_df())
        assert math.isnan(out["fvg_high"].iloc[0])
        assert math.isnan(out["fvg_high"].iloc[1])


# ===========================================================================
# 5. Displacement Filter
# ===========================================================================

class TestDisplacementFilter:

    def test_c2_body_exactly_1_5x_atr_no_fvg(self):
        # body == 1.5×ATR is NOT strictly greater; expect no FVG
        out = detect_fair_value_gaps(_bull_fvg_df(c2_body_mult=1.5))
        assert not out["fvg_bullish"].any()

    def test_c2_body_below_threshold_no_bull_fvg(self):
        out = detect_fair_value_gaps(_bull_fvg_df(c2_body_mult=1.0))
        assert not out["fvg_bullish"].any()

    def test_c2_body_above_threshold_bull_fvg(self):
        out = detect_fair_value_gaps(_bull_fvg_df(c2_body_mult=1.51))
        assert out["fvg_bullish"].any()

    def test_displacement_filter_also_blocks_bear_fvg(self):
        out = detect_fair_value_gaps(_bear_fvg_df(c2_body_mult=1.0))
        assert not out["fvg_bearish"].any()

    def test_strong_displacement_passes_bear_fvg(self):
        out = detect_fair_value_gaps(_bear_fvg_df(c2_body_mult=2.5))
        assert out["fvg_bearish"].any()


# ===========================================================================
# 6. Gap Size Filter
# ===========================================================================

class TestGapSizeFilter:

    def test_gap_exactly_0_5x_atr_produces_fvg(self):
        # spec: "IF gap < 0.5*atr: continue" — equality passes the filter
        out = detect_fair_value_gaps(_bull_fvg_df(gap_mult=0.5))
        assert out["fvg_bullish"].any()

    def test_gap_below_threshold_no_bull_fvg(self):
        out = detect_fair_value_gaps(_bull_fvg_df(gap_mult=0.3))
        assert not out["fvg_bullish"].any()

    def test_gap_above_threshold_bull_fvg(self):
        out = detect_fair_value_gaps(_bull_fvg_df(gap_mult=0.51))
        assert out["fvg_bullish"].any()

    def test_gap_below_threshold_no_bear_fvg(self):
        out = detect_fair_value_gaps(_bear_fvg_df(gap_mult=0.3))
        assert not out["fvg_bearish"].any()


# ===========================================================================
# 7. Trend Filter
# ===========================================================================

class TestTrendFilter:

    def test_bull_fvg_blocked_in_bearish_trend(self):
        out = detect_fair_value_gaps(_bull_fvg_df(c3_trend="BEARISH"))
        assert not out["fvg_bullish"].any()

    def test_bull_fvg_allowed_in_bullish_trend(self):
        out = detect_fair_value_gaps(_bull_fvg_df(c3_trend="BULLISH"))
        assert out["fvg_bullish"].any()

    def test_bull_fvg_allowed_in_transition_trend(self):
        out = detect_fair_value_gaps(_bull_fvg_df(c3_trend="TRANSITION"))
        assert out["fvg_bullish"].any()

    def test_bear_fvg_blocked_in_bullish_trend(self):
        out = detect_fair_value_gaps(_bear_fvg_df(c3_trend="BULLISH"))
        assert not out["fvg_bearish"].any()

    def test_bear_fvg_allowed_in_bearish_trend(self):
        out = detect_fair_value_gaps(_bear_fvg_df(c3_trend="BEARISH"))
        assert out["fvg_bearish"].any()

    def test_bear_fvg_allowed_in_transition_trend(self):
        out = detect_fair_value_gaps(_bear_fvg_df(c3_trend="TRANSITION"))
        assert out["fvg_bearish"].any()


# ===========================================================================
# 8. ATR Validation
# ===========================================================================

class TestATRValidation:

    def test_atr_nan_skips_row(self):
        df  = _bull_fvg_df()
        df.at[1, "ATR"] = float("nan")   # c2 ATR is NaN
        out = detect_fair_value_gaps(df)
        assert not out["fvg_bullish"].any()

    def test_atr_zero_skips_row(self):
        df  = _bull_fvg_df()
        df.at[1, "ATR"] = 0.0
        out = detect_fair_value_gaps(df)
        assert not out["fvg_bullish"].any()

    def test_atr_negative_skips_row(self):
        df  = _bull_fvg_df()
        df.at[1, "ATR"] = -5.0
        out = detect_fair_value_gaps(df)
        assert not out["fvg_bullish"].any()

    def test_atr_below_min_skips_row(self):
        # ATR = 0.05 < 0.1 threshold → skipped
        df  = _bull_fvg_df()
        df.at[1, "ATR"] = 0.05
        out = detect_fair_value_gaps(df)
        assert not out["fvg_bullish"].any()

    def test_atr_capped_at_100_still_detects(self):
        # ATR = 500 → capped to 100; displacement 2×130 > 1.5×100 OK;
        # gap 65 > 0.5×100 OK → FVG detected
        df  = _bull_fvg_df(atr=130.0, c2_body_mult=2.0, gap_mult=0.6)
        # Set c2 ATR to 500 so it gets capped to 100
        df.at[1, "ATR"] = 500.0
        out = detect_fair_value_gaps(df)
        assert out["fvg_bullish"].any()

    def test_atr_at_minimum_boundary_detects(self):
        # ATR = 0.1 is the exact minimum; body 2×0.1 > 1.5×0.1; gap 1×0.1 > 0.5×0.1
        df = _bull_fvg_df(atr=0.1, c2_body_mult=2.0, gap_mult=1.0)
        out = detect_fair_value_gaps(df)
        assert out["fvg_bullish"].any()


# ===========================================================================
# 9. Mitigation
# ===========================================================================

class TestMitigation:

    def test_bullish_fvg_mitigated_when_future_low_enters_gap(self):
        """A subsequent candle whose low <= fvg_high invalidates the FVG."""
        df = _bull_fvg_df()
        # fvg_high = c3.low on row 2
        fvg_high = df["low"].iloc[2]
        # Append c4: low is inside the FVG (touches the top of the gap)
        c4 = _row(open_=fvg_high + 2, high=fvg_high + 4,
                  low=fvg_high - 1.0, close=fvg_high + 1.0)
        out = detect_fair_value_gaps(pd.concat([df, pd.DataFrame([c4])], ignore_index=True))
        assert not out["fvg_bullish"].iloc[2]

    def test_bullish_fvg_not_mitigated_when_future_low_above_gap(self):
        """A subsequent candle above the FVG zone must NOT invalidate it."""
        df = _bull_fvg_df()
        fvg_high = df["low"].iloc[2]
        # Append c4: low is strictly above fvg_high
        c4 = _row(open_=fvg_high + 5, high=fvg_high + 10,
                  low=fvg_high + 1.0, close=fvg_high + 6.0)
        out = detect_fair_value_gaps(pd.concat([df, pd.DataFrame([c4])], ignore_index=True))
        assert out["fvg_bullish"].iloc[2] is np.bool_(True)

    def test_mitigated_bull_fvg_has_nan_low_and_high(self):
        df = _bull_fvg_df()
        fvg_high = df["low"].iloc[2]
        c4 = _row(low=fvg_high - 1.0)
        out = detect_fair_value_gaps(pd.concat([df, pd.DataFrame([c4])], ignore_index=True))
        assert math.isnan(out["fvg_low"].iloc[2])
        assert math.isnan(out["fvg_high"].iloc[2])

    def test_bearish_fvg_mitigated_when_future_high_enters_gap(self):
        """A subsequent candle whose high >= fvg_low invalidates bearing FVG."""
        df = _bear_fvg_df()
        fvg_low = df["high"].iloc[2]
        c4 = _row(open_=fvg_low - 4, high=fvg_low + 1.0,
                  low=fvg_low - 5, close=fvg_low - 2.0)
        out = detect_fair_value_gaps(pd.concat([df, pd.DataFrame([c4])], ignore_index=True))
        assert not out["fvg_bearish"].iloc[2]

    def test_bearish_fvg_not_mitigated_when_future_high_below_gap(self):
        df = _bear_fvg_df()
        fvg_low = df["high"].iloc[2]
        c4 = _row(open_=fvg_low - 10, high=fvg_low - 1.0,
                  low=fvg_low - 12, close=fvg_low - 5.0)
        out = detect_fair_value_gaps(pd.concat([df, pd.DataFrame([c4])], ignore_index=True))
        assert out["fvg_bearish"].iloc[2] is np.bool_(True)

    def test_mitigated_bear_fvg_has_nan_low_and_high(self):
        df = _bear_fvg_df()
        fvg_low = df["high"].iloc[2]
        c4 = _row(high=fvg_low + 1.0)
        out = detect_fair_value_gaps(pd.concat([df, pd.DataFrame([c4])], ignore_index=True))
        assert math.isnan(out["fvg_low"].iloc[2])
        assert math.isnan(out["fvg_high"].iloc[2])

    def test_mitigation_only_affects_correct_fvg_index(self):
        """Mitigating the first FVG must not invalidate the second FVG.

        Layout:
          rows 0-2 : first bullish FVG  (fvg_high ~ 2010)
          row  3   : mitigation candle  (low enters first FVG zone)
          rows 4-6 : second bullish FVG at 3000 – no rows after it → never mitigated
        """
        df_first       = _bull_fvg_df(gap_mult=1.0)
        fvg_high_first = df_first["low"].iloc[2]   # ~ 2010

        c_mitigate = _row(low=fvg_high_first - 1.0, high=fvg_high_first + 2.0,
                          open_=fvg_high_first, close=fvg_high_first + 1.0)

        high_base = 3000.0
        c1b = _row(open_=high_base - 2, high=high_base, low=high_base - 8,
                   close=high_base - 1)
        c2b_open  = high_base - 1.0
        c2b_close = c2b_open + 2.0 * _ATR
        c2b = _row(open_=c2b_open, high=c2b_close + 2, low=c2b_open - 2,
                   close=c2b_close, displacement_bullish=True)
        c3b_low = high_base + 1.0 * _ATR
        c3b = _row(open_=c3b_low + 1, high=c3b_low + 5, low=c3b_low,
                   close=c3b_low + 3)
        df_second = _df([c1b, c2b, c3b])

        combined = pd.concat(
            [df_first, pd.DataFrame([c_mitigate]), df_second],
            ignore_index=True,
        )
        out = detect_fair_value_gaps(combined)

        assert not out["fvg_bullish"].iloc[2]          # first FVG mitigated
        assert out["fvg_bullish"].iloc[6] is np.bool_(True)  # second FVG survives

    def test_no_mitigation_when_no_future_candles(self):
        """Last-row FVG: no future candles; suffix min = +inf → not mitigated."""
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert out["fvg_bullish"].iloc[2] is np.bool_(True)


# ===========================================================================
# 10. Unmitigated FVGs survive
# ===========================================================================

class TestUnmitigated:

    def test_bull_fvg_survives_without_mitigation_candle(self):
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert out["fvg_bullish"].iloc[2] is np.bool_(True)
        assert not math.isnan(out["fvg_low"].iloc[2])
        assert not math.isnan(out["fvg_high"].iloc[2])

    def test_bear_fvg_survives_without_mitigation_candle(self):
        out = detect_fair_value_gaps(_bear_fvg_df())
        assert out["fvg_bearish"].iloc[2] is np.bool_(True)
        assert not math.isnan(out["fvg_low"].iloc[2])
        assert not math.isnan(out["fvg_high"].iloc[2])

    def test_multiple_unmitigated_fvgs_all_preserved(self):
        """Two non-overlapping bullish FVGs at separate price levels both survive.

        The second block is placed at a high_base of 3000 so none of its candles
        have lows that enter the first FVG zone (~2000–2010).
        """
        df_a = _bull_fvg_df()  # first FVG zone around 2000–2010

        # Second block at 3000; all lows >> 2010 so first FVG is never mitigated
        high_base = 3000.0
        c1b = _row(open_=high_base - 2, high=high_base, low=high_base - 8,
                   close=high_base - 1)
        c2b_open  = high_base - 1.0
        c2b_close = c2b_open + 2.0 * _ATR
        c2b = _row(open_=c2b_open, high=c2b_close + 2, low=c2b_open - 2,
                   close=c2b_close, displacement_bullish=True)
        c3b_low   = high_base + 1.0 * _ATR
        c3b = _row(open_=c3b_low + 1, high=c3b_low + 5, low=c3b_low,
                   close=c3b_low + 3)
        df_b = _df([c1b, c2b, c3b])

        combined = pd.concat([df_a, df_b], ignore_index=True)
        out = detect_fair_value_gaps(combined)
        assert out["fvg_bullish"].sum() == 2

    def test_fvg_low_less_than_fvg_high_when_detected(self):
        """Zone bounds are coherent: fvg_low < fvg_high."""
        out = detect_fair_value_gaps(_bull_fvg_df())
        idx = out[out["fvg_bullish"]].index[0]
        assert out["fvg_low"].iloc[idx] < out["fvg_high"].iloc[idx]


# ===========================================================================
# 11. Edge Cases
# ===========================================================================

class TestEdgeCases:

    def test_single_row_no_fvg(self):
        df = _df([_row()])
        out = detect_fair_value_gaps(df)
        assert not out["fvg_bullish"].any()
        assert not out["fvg_bearish"].any()

    def test_two_rows_no_fvg(self):
        df = _df([_row(), _row()])
        out = detect_fair_value_gaps(df)
        assert not out["fvg_bullish"].any()
        assert not out["fvg_bearish"].any()

    def test_row_count_preserved(self):
        df = _bull_fvg_df()
        out = detect_fair_value_gaps(df)
        assert len(out) == len(df)

    def test_existing_columns_preserved(self):
        df  = _bull_fvg_df()
        out = detect_fair_value_gaps(df)
        for col in df.columns:
            pd.testing.assert_series_equal(
                out[col].reset_index(drop=True),
                df[col].reset_index(drop=True),
                check_names=False,
            )

    def test_no_fvg_when_c1_high_equals_c3_low(self):
        """Equality (c1.high == c3.low) is NOT a strictly positive gap → no FVG."""
        df = _bull_fvg_df(gap_mult=1.0)
        # Force c3.low == c1.high by collapsing the gap to zero
        df.at[2, "low"] = df["high"].iloc[0]
        out = detect_fair_value_gaps(df)
        assert not out["fvg_bullish"].any()


# ===========================================================================
# 12. Displacement Dependency
# ===========================================================================

class TestDisplacementDependency:

    def test_bull_fvg_blocked_when_c2_has_no_displacement(self):
        """FVG detection requires c2 to carry a confirmed displacement mark."""
        df = _bull_fvg_df()
        df.at[1, "displacement_bullish"] = False
        df.at[1, "displacement_bearish"] = False
        out = detect_fair_value_gaps(df)
        assert not out["fvg_bullish"].any()

    def test_bull_fvg_detected_when_c2_has_bullish_displacement(self):
        """c2 with displacement_bullish=True → FVG detected as normal."""
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert out["fvg_bullish"].iloc[2] is np.bool_(True)

    def test_bear_fvg_blocked_when_c2_has_no_displacement(self):
        df = _bear_fvg_df()
        df.at[1, "displacement_bullish"] = False
        df.at[1, "displacement_bearish"] = False
        out = detect_fair_value_gaps(df)
        assert not out["fvg_bearish"].any()

    def test_bear_fvg_detected_when_c2_has_bearish_displacement(self):
        out = detect_fair_value_gaps(_bear_fvg_df())
        assert out["fvg_bearish"].iloc[2] is np.bool_(True)

    def test_bull_fvg_also_passes_with_bearish_displacement_on_c2(self):
        """Either displacement direction on c2 is sufficient to pass the gate."""
        df = _bull_fvg_df()
        df.at[1, "displacement_bullish"] = False
        df.at[1, "displacement_bearish"] = True
        out = detect_fair_value_gaps(df)
        assert out["fvg_bullish"].iloc[2] is np.bool_(True)


# ===========================================================================
# 13. Age Filter
# ===========================================================================

class TestAgeFilter:

    def test_fvg_aged_out_past_max_age(self):
        """FVG created at index 2 of a 3-row df; after 21 neutral rows appended
        its age = 21 > MAX_FVG_AGE (20) → invalidated."""
        df = _bull_fvg_df()
        fvg_high = df["low"].iloc[2]
        # Neutral rows: lows well above fvg_high so price never mitigates the gap
        neutral = _row(low=fvg_high + 10.0, high=fvg_high + 20.0,
                       open_=fvg_high + 11.0, close=fvg_high + 15.0)
        big_df = pd.concat([df, pd.DataFrame([neutral] * 21)], ignore_index=True)
        out = detect_fair_value_gaps(big_df)
        assert not out["fvg_bullish"].iloc[2]
        assert math.isnan(out["fvg_low"].iloc[2])

    def test_fvg_exactly_at_max_age_survives(self):
        """Age = MAX_FVG_AGE is NOT strictly greater → FVG survives."""
        df = _bull_fvg_df()
        fvg_high = df["low"].iloc[2]
        neutral = _row(low=fvg_high + 10.0, high=fvg_high + 20.0,
                       open_=fvg_high + 11.0, close=fvg_high + 15.0)
        big_df = pd.concat([df, pd.DataFrame([neutral] * 20)], ignore_index=True)
        out = detect_fair_value_gaps(big_df)
        assert out["fvg_bullish"].iloc[2] is np.bool_(True)

    def test_bear_fvg_aged_out_past_max_age(self):
        """Bearish FVG also ages out after MAX_FVG_AGE candles."""
        df = _bear_fvg_df()
        fvg_low = df["high"].iloc[2]
        neutral = _row(low=fvg_low - 20.0, high=fvg_low - 10.0,
                       open_=fvg_low - 15.0, close=fvg_low - 12.0,
                       trend_state="BEARISH")
        big_df = pd.concat([df, pd.DataFrame([neutral] * 21)], ignore_index=True)
        out = detect_fair_value_gaps(big_df)
        assert not out["fvg_bearish"].iloc[2]


# ===========================================================================
# 14. FVG Strength
# ===========================================================================

class TestFVGStrength:

    def test_bull_fvg_strength_nonzero_when_detected(self):
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert out["fvg_strength"].iloc[2] > 0.0

    def test_bull_fvg_strength_at_most_one(self):
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert out["fvg_strength"].max() <= 1.0

    def test_bull_fvg_strength_formula(self):
        # gap = 1.0 * ATR = 10, atr = 10
        # fvg_strength = min(10 / 10 / 3.0, 1.0) = min(0.333, 1.0) = 0.333
        out = detect_fair_value_gaps(_bull_fvg_df(gap_mult=1.0, atr=10.0))
        assert out["fvg_strength"].iloc[2] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_fvg_strength_capped_at_one(self):
        # gap = 10.0 * ATR → min(10/3, 1.0) = 1.0
        out = detect_fair_value_gaps(_bull_fvg_df(gap_mult=10.0, atr=10.0))
        assert out["fvg_strength"].iloc[2] == pytest.approx(1.0)

    def test_fvg_strength_zero_on_non_fvg_rows(self):
        out = detect_fair_value_gaps(_bull_fvg_df())
        assert out["fvg_strength"].iloc[0] == pytest.approx(0.0)
        assert out["fvg_strength"].iloc[1] == pytest.approx(0.0)

    def test_mitigated_fvg_strength_becomes_zero(self):
        """Mitigating a FVG must also zero its strength."""
        df = _bull_fvg_df()
        fvg_high = df["low"].iloc[2]
        c4 = _row(low=fvg_high - 1.0)
        extended = pd.concat([df, pd.DataFrame([c4])], ignore_index=True)
        out = detect_fair_value_gaps(extended)
        assert out["fvg_strength"].iloc[2] == pytest.approx(0.0)

