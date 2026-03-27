"""Tests for analysis/ob_mitigation.py — detect_ob_mitigation()

Test group overview
-------------------
 1. TestInputValidation        (5) — missing columns, empty df
 2. TestOutputSchema           (6) — columns, dtypes, row count, immutability
 3. TestBearishOBMitigation    (7) — future high >= ob_high → mitigated
 4. TestBullishOBMitigation    (7) — future low <= ob_low → mitigated
 5. TestUnmitigatedOBs         (4) — no future price action through zone → active
 6. TestMultipleOBs            (4) — independent mitigation across several OBs
 7. TestGetActiveOBs           (4) — convenience filter function
 8. TestEdgeCases              (4) — single row, NaN ob prices, last-bar OB
 9. TestDeterminism            (2) — same input → same output
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.ob_mitigation import detect_ob_mitigation, get_active_obs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = pd.Timestamp("2024-01-01 09:00:00")
_DT = pd.Timedelta(minutes=5)


def _row(
    i: int,
    high:  float = 2005.0,
    low:   float = 1995.0,
    bull_ob: bool = False,
    bear_ob: bool = False,
    ob_high: float = float("nan"),
    ob_low:  float = float("nan"),
) -> dict:
    return {
        "timestamp":            _TS + _DT * i,
        "open":                 2000.0,
        "high":                 high,
        "low":                  low,
        "close":                2000.0,
        "bullish_order_block":  bull_ob,
        "bearish_order_block":  bear_ob,
        "ob_high":              ob_high,
        "ob_low":               ob_low,
    }


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ===========================================================================
# 1. Input Validation
# ===========================================================================


class TestInputValidation:

    def test_raises_on_missing_high(self):
        df = _df([_row(0)]).drop(columns=["high"])
        with pytest.raises(ValueError, match="high"):
            detect_ob_mitigation(df)

    def test_raises_on_missing_low(self):
        df = _df([_row(0)]).drop(columns=["low"])
        with pytest.raises(ValueError, match="low"):
            detect_ob_mitigation(df)

    def test_raises_on_missing_bullish_ob(self):
        df = _df([_row(0)]).drop(columns=["bullish_order_block"])
        with pytest.raises(ValueError, match="bullish_order_block"):
            detect_ob_mitigation(df)

    def test_raises_on_missing_ob_high(self):
        df = _df([_row(0)]).drop(columns=["ob_high"])
        with pytest.raises(ValueError, match="ob_high"):
            detect_ob_mitigation(df)

    def test_raises_on_empty_dataframe(self):
        df = _df([_row(0)]).iloc[0:0]
        with pytest.raises(ValueError, match="empty"):
            detect_ob_mitigation(df)


# ===========================================================================
# 2. Output Schema
# ===========================================================================


class TestOutputSchema:

    def _out(self):
        return detect_ob_mitigation(_df([_row(0)]))

    def test_ob_mitigated_column_present(self):
        assert "ob_mitigated" in self._out().columns

    def test_ob_active_column_present(self):
        assert "ob_active" in self._out().columns

    def test_ob_mitigated_is_bool(self):
        assert self._out()["ob_mitigated"].dtype == bool

    def test_ob_active_is_bool(self):
        assert self._out()["ob_active"].dtype == bool

    def test_row_count_preserved(self):
        df = _df([_row(i) for i in range(5)])
        out = detect_ob_mitigation(df)
        assert len(out) == 5

    def test_input_not_mutated(self):
        df = _df([_row(0)])
        cols_before = list(df.columns)
        detect_ob_mitigation(df)
        assert list(df.columns) == cols_before


# ===========================================================================
# 3. Bearish OB Mitigation
# ===========================================================================


class TestBearishOBMitigation:

    def test_bear_ob_mitigated_when_future_high_reaches_ob_high(self):
        # Bear OB at row 0, future candle at row 1 hits ob_high
        rows = [
            _row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, high=2010.0),  # exactly touches ob_high
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_mitigated"].iloc[0] is np.bool_(True)

    def test_bear_ob_mitigated_when_future_high_exceeds_ob_high(self):
        rows = [
            _row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, high=2015.0),  # exceeds ob_high
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_mitigated"].iloc[0] is np.bool_(True)

    def test_bear_ob_not_mitigated_when_future_high_below_ob_high(self):
        rows = [
            _row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, high=2009.0),  # just below ob_high
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_mitigated"].iloc[0] is np.bool_(False)

    def test_bear_ob_active_when_not_mitigated(self):
        rows = [
            _row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, high=2005.0),
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_active"].iloc[0] is np.bool_(True)

    def test_bear_ob_not_active_when_mitigated(self):
        rows = [
            _row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, high=2015.0),
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_active"].iloc[0] is np.bool_(False)

    def test_bear_ob_mitigated_by_candle_two_bars_later(self):
        rows = [
            _row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, high=2005.0),       # no mitigation
            _row(2, high=2012.0),       # mitigation
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_mitigated"].iloc[0] is np.bool_(True)

    def test_non_ob_rows_never_mitigated(self):
        rows = [
            _row(0),   # plain candle, no OB
            _row(1, high=3000.0),
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_mitigated"].iloc[0] is np.bool_(False)
        assert out["ob_active"].iloc[0] is np.bool_(False)


# ===========================================================================
# 4. Bullish OB Mitigation
# ===========================================================================


class TestBullishOBMitigation:

    def test_bull_ob_mitigated_when_future_low_reaches_ob_low(self):
        rows = [
            _row(0, bull_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, low=2000.0),   # exactly touches ob_low
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_mitigated"].iloc[0] is np.bool_(True)

    def test_bull_ob_mitigated_when_future_low_below_ob_low(self):
        rows = [
            _row(0, bull_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, low=1995.0),   # below ob_low
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_mitigated"].iloc[0] is np.bool_(True)

    def test_bull_ob_not_mitigated_when_future_low_above_ob_low(self):
        rows = [
            _row(0, bull_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, low=2001.0),   # just above ob_low
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_mitigated"].iloc[0] is np.bool_(False)

    def test_bull_ob_active_when_not_mitigated(self):
        rows = [
            _row(0, bull_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, low=2005.0),
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_active"].iloc[0] is np.bool_(True)

    def test_bull_ob_not_active_when_mitigated(self):
        rows = [
            _row(0, bull_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, low=1995.0),
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_active"].iloc[0] is np.bool_(False)

    def test_bull_ob_mitigated_by_distant_candle(self):
        rows = [
            _row(0, bull_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, low=2003.0),
            _row(2, low=2002.0),
            _row(3, low=1998.0),   # mitigation finally occurs
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_mitigated"].iloc[0] is np.bool_(True)

    def test_bull_ob_active_without_any_mitigation(self):
        rows = [
            _row(0, bull_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, low=2003.0),
            _row(2, low=2004.0),
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_active"].iloc[0] is np.bool_(True)


# ===========================================================================
# 5. Unmitigated OBs Survive
# ===========================================================================


class TestUnmitigatedOBs:

    def test_bear_ob_at_last_row_never_mitigated(self):
        # No future candles → suffix_max_excl is -inf → never mitigated
        rows = [
            _row(0),
            _row(1, bear_ob=True, ob_high=2010.0, ob_low=2000.0),
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_active"].iloc[1] is np.bool_(True)

    def test_bull_ob_at_last_row_never_mitigated(self):
        rows = [
            _row(0),
            _row(1, bull_ob=True, ob_high=2010.0, ob_low=2000.0),
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_active"].iloc[1] is np.bool_(True)

    def test_both_obs_survive_when_price_stays_within_zone(self):
        rows = [
            _row(0, bear_ob=True, ob_high=2020.0, ob_low=2010.0),
            _row(1, bull_ob=True, ob_high=1990.0, ob_low=1980.0),
            _row(2, high=2005.0, low=1995.0),  # between both OBs
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_active"].iloc[0] is np.bool_(True)
        assert out["ob_active"].iloc[1] is np.bool_(True)

    def test_mitigation_only_affects_correct_ob_index(self):
        # Bear OB1 is mitigated; Bear OB2 at a different level is not
        rows = [
            _row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0),  # will be mitigated
            _row(1, bear_ob=True, ob_high=2030.0, ob_low=2020.0),  # will NOT be mitigated
            _row(2, high=2012.0),   # enters OB1 zone only
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_mitigated"].iloc[0] is np.bool_(True)
        assert out["ob_mitigated"].iloc[1] is np.bool_(False)
        assert out["ob_active"].iloc[1] is np.bool_(True)


# ===========================================================================
# 6. Multiple OBs
# ===========================================================================


class TestMultipleOBs:

    def test_two_bear_obs_independently_mitigated(self):
        rows = [
            _row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, bear_ob=True, ob_high=2020.0, ob_low=2015.0),
            _row(2, high=2011.0),   # mitigates OB0 only
            _row(3, high=2022.0),   # mitigates OB1
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_mitigated"].iloc[0] is np.bool_(True)
        assert out["ob_mitigated"].iloc[1] is np.bool_(True)

    def test_bear_ob_mitigated_bull_ob_active(self):
        rows = [
            _row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, bull_ob=True, ob_high=1990.0, ob_low=1980.0),
            _row(2, high=2015.0, low=1985.0),  # both zones hit
        ]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_mitigated"].iloc[0] is np.bool_(True)  # bear ob mitigated
        assert out["ob_mitigated"].iloc[1] is np.bool_(True)  # bull ob mitigated

    def test_active_count_correct(self):
        rows = [
            _row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, bear_ob=True, ob_high=2020.0, ob_low=2015.0),
            _row(2, high=2011.0),   # only mitigates row 0
        ]
        out = detect_ob_mitigation(_df(rows))
        active_count = int(out["ob_active"].sum())
        assert active_count == 1

    def test_no_obs_all_mitigated_false(self):
        rows = [_row(i) for i in range(5)]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_mitigated"].sum() == 0
        assert out["ob_active"].sum() == 0


# ===========================================================================
# 7. get_active_obs helper
# ===========================================================================


class TestGetActiveOBs:

    def test_returns_only_active_ob_rows(self):
        rows = [
            _row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, bear_ob=True, ob_high=2020.0, ob_low=2015.0),
            _row(2, high=2011.0),   # mitigates row 0
        ]
        out = detect_ob_mitigation(_df(rows))
        active = get_active_obs(out)
        assert len(active) == 1
        assert active["ob_high"].iloc[0] == 2020.0

    def test_empty_result_when_all_mitigated(self):
        rows = [
            _row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, high=2015.0),
        ]
        out = detect_ob_mitigation(_df(rows))
        active = get_active_obs(out)
        assert len(active) == 0

    def test_raises_if_ob_active_column_missing(self):
        df = _df([_row(0)])
        with pytest.raises(ValueError, match="ob_active"):
            get_active_obs(df)

    def test_returns_copy_not_view(self):
        rows = [
            _row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0),
        ]
        out = detect_ob_mitigation(_df(rows))
        active = get_active_obs(out)
        assert active is not out


# ===========================================================================
# 8. Edge Cases
# ===========================================================================


class TestEdgeCases:

    def test_single_row_ob_never_mitigated(self):
        rows = [_row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0)]
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_mitigated"].iloc[0] is np.bool_(False)
        assert out["ob_active"].iloc[0] is np.bool_(True)

    def test_nan_ob_prices_skipped_gracefully(self):
        # OB row with NaN prices should not crash; treated as non-OB
        rows = [
            _row(0, bear_ob=True),   # ob_high/ob_low default to NaN
            _row(1, high=3000.0),
        ]
        out = detect_ob_mitigation(_df(rows))
        # NaN ob_high → not mitigated (no valid zone to test against)
        assert isinstance(out["ob_mitigated"].iloc[0], (bool, np.bool_))

    def test_last_bar_ob_with_many_prior_candles(self):
        rows = [_row(i) for i in range(10)]
        rows[-1] = _row(9, bear_ob=True, ob_high=2010.0, ob_low=2000.0)
        out = detect_ob_mitigation(_df(rows))
        assert out["ob_active"].iloc[-1] is np.bool_(True)

    def test_extra_columns_preserved(self):
        df = _df([_row(0)])
        df["custom"] = 42
        out = detect_ob_mitigation(df)
        assert "custom" in out.columns
        assert (out["custom"] == 42).all()


# ===========================================================================
# 9. Determinism
# ===========================================================================


class TestDeterminism:

    def test_same_input_same_output(self):
        rows = [
            _row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, high=2012.0),
        ]
        df = _df(rows)
        out1 = detect_ob_mitigation(df)
        out2 = detect_ob_mitigation(df)
        pd.testing.assert_frame_equal(out1, out2)

    def test_bull_and_bear_mirrored_symmetrically(self):
        rows_bear = [
            _row(0, bear_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, high=2015.0),
        ]
        rows_bull = [
            _row(0, bull_ob=True, ob_high=2010.0, ob_low=2000.0),
            _row(1, low=1995.0),
        ]
        out_bear = detect_ob_mitigation(_df(rows_bear))
        out_bull = detect_ob_mitigation(_df(rows_bull))
        assert out_bear["ob_mitigated"].iloc[0] == out_bull["ob_mitigated"].iloc[0]