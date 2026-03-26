"""Comprehensive test suite for analysis/liquidity_sweeps.py.

Sweep conditions (per candle):
    sweep_high: high > active_high_level AND close < active_high_level
    sweep_low:  low  < active_low_level  AND close > active_low_level

The active level is updated to the pool candle's high/low whenever
liquidity_pool_high/low == True and persists until superseded by a newer
pool row.

Group overview
--------------
 1. TestInputValidation          (5)  — missing columns, empty df
 2. TestOutputSchema             (4)  — columns, dtypes, copy semantics
 3. TestNoPool                   (2)  — no pool → no sweep ever
 4. TestNoSweepWhenWickExact     (2)  — high == level is not a sweep
 5. TestSweepHighConfirmed       (4)  — wick above, close inside
 6. TestSweepHighRejected        (3)  — close outside = no sweep
 7. TestSweepLowConfirmed        (4)  — wick below, close inside
 8. TestSweepLowRejected         (3)  — close outside = no sweep
 9. TestLevelUpdates             (4)  — newer pool supersedes older level
10. TestPoolRowNotSweepItself    (2)  — pool candle cannot self-sweep
11. TestMultipleSweeps           (2)  — multiple sweeps across sequence
12. TestIndependentHighLow       (2)  — high/low levels independent
13. TestDfNotMutated             (1)  — input unchanged
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.liquidity_sweeps import detect_liquidity_sweeps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = pd.Timestamp("2024-01-01 09:00:00")
_STEP    = pd.Timedelta(minutes=5)


def _row(
    i: int,
    high:      float = 1910.0,
    low:       float = 1895.0,
    close:     float = 1902.0,
    pool_high: bool  = False,
    pool_low:  bool  = False,
) -> dict:
    return {
        "timestamp":           _BASE_TS + _STEP * i,
        "open":                1900.0,
        "high":                high,
        "low":                 low,
        "close":               close,
        "liquidity_pool_high": pool_high,
        "liquidity_pool_low":  pool_low,
    }


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_raises_on_missing_high(self):
        df = _df([_row(0)]).drop(columns=["high"])
        with pytest.raises(ValueError, match="high"):
            detect_liquidity_sweeps(df)

    def test_raises_on_missing_low(self):
        df = _df([_row(0)]).drop(columns=["low"])
        with pytest.raises(ValueError, match="low"):
            detect_liquidity_sweeps(df)

    def test_raises_on_missing_close(self):
        df = _df([_row(0)]).drop(columns=["close"])
        with pytest.raises(ValueError, match="close"):
            detect_liquidity_sweeps(df)

    def test_raises_on_missing_liquidity_pool_high(self):
        df = _df([_row(0)]).drop(columns=["liquidity_pool_high"])
        with pytest.raises(ValueError, match="liquidity_pool_high"):
            detect_liquidity_sweeps(df)

    def test_raises_on_empty_dataframe(self):
        df = _df([_row(0)]).iloc[0:0]
        with pytest.raises(ValueError, match="empty"):
            detect_liquidity_sweeps(df)


# ---------------------------------------------------------------------------
# 2. Output Schema
# ---------------------------------------------------------------------------

class TestOutputSchema:

    def _result(self):
        return detect_liquidity_sweeps(_df([_row(0)]))

    def test_has_liquidity_sweep_high(self):
        assert "liquidity_sweep_high" in self._result().columns

    def test_has_liquidity_sweep_low(self):
        assert "liquidity_sweep_low" in self._result().columns

    def test_sweep_columns_are_bool(self):
        r = self._result()
        assert r["liquidity_sweep_high"].dtype == bool
        assert r["liquidity_sweep_low"].dtype == bool

    def test_returns_copy_not_same_object(self):
        df = _df([_row(0)])
        assert detect_liquidity_sweeps(df) is not df


# ---------------------------------------------------------------------------
# 3. No Pool → No Sweep
# ---------------------------------------------------------------------------

class TestNoPool:

    def test_no_sweep_high_without_any_pool(self):
        # wick wicks way up but no pool was ever set
        rows = [_row(0, high=1950.0, close=1895.0)]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_high"].iloc[0] is np.bool_(False)

    def test_no_sweep_low_without_any_pool(self):
        rows = [_row(0, low=1850.0, close=1910.0)]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_low"].iloc[0] is np.bool_(False)


# ---------------------------------------------------------------------------
# 4. Wick Exactly At Level Is Not A Sweep
# ---------------------------------------------------------------------------

class TestNoSweepWhenWickExact:

    def test_high_equals_level_not_a_sweep(self):
        # Pool sets level=1910; next candle high==1910 exactly → not > level
        rows = [
            _row(0, high=1910.0, pool_high=True),
            _row(1, high=1910.0, close=1895.0),
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_high"].iloc[1] is np.bool_(False)

    def test_low_equals_level_not_a_sweep(self):
        rows = [
            _row(0, low=1895.0, pool_low=True),
            _row(1, low=1895.0, close=1910.0),
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_low"].iloc[1] is np.bool_(False)


# ---------------------------------------------------------------------------
# 5. Sweep High Confirmed
# ---------------------------------------------------------------------------

class TestSweepHighConfirmed:

    def test_basic_sweep_high(self):
        # level=1910; wick to 1912; close at 1905 < 1910 → sweep
        rows = [
            _row(0, high=1910.0, pool_high=True),
            _row(1, high=1912.0, close=1905.0),
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_high"].iloc[1] is np.bool_(True)

    def test_sweep_high_on_correct_row(self):
        rows = [
            _row(0, high=1910.0, pool_high=True),
            _row(1, high=1908.0, close=1905.0),   # wick below level — not a sweep
            _row(2, high=1912.0, close=1905.0),   # sweep
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_high"].iloc[1] is np.bool_(False)
        assert r["liquidity_sweep_high"].iloc[2] is np.bool_(True)

    def test_sweep_high_close_just_below_level(self):
        rows = [
            _row(0, high=1910.00, pool_high=True),
            _row(1, high=1910.01, close=1909.99),
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_high"].iloc[1] is np.bool_(True)

    def test_sweep_high_with_filler_candles_between(self):
        rows = [
            _row(0, high=1910.0, pool_high=True),
            _row(1, high=1908.0, close=1904.0),
            _row(2, high=1908.0, close=1904.0),
            _row(3, high=1912.0, close=1905.0),   # sweep after gap
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_high"].iloc[3] is np.bool_(True)


# ---------------------------------------------------------------------------
# 6. Sweep High Rejected
# ---------------------------------------------------------------------------

class TestSweepHighRejected:

    def test_close_above_level_is_not_sweep(self):
        rows = [
            _row(0, high=1910.0, pool_high=True),
            _row(1, high=1912.0, close=1911.0),   # close > level → breakout
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_high"].iloc[1] is np.bool_(False)

    def test_close_equal_to_level_is_not_sweep(self):
        # close must be strictly < level
        rows = [
            _row(0, high=1910.0, pool_high=True),
            _row(1, high=1912.0, close=1910.0),
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_high"].iloc[1] is np.bool_(False)

    def test_wick_below_level_no_sweep(self):
        rows = [
            _row(0, high=1910.0, pool_high=True),
            _row(1, high=1909.0, close=1905.0),   # never breaches level
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_high"].iloc[1] is np.bool_(False)


# ---------------------------------------------------------------------------
# 7. Sweep Low Confirmed
# ---------------------------------------------------------------------------

class TestSweepLowConfirmed:

    def test_basic_sweep_low(self):
        # level=1895; wick to 1893; close at 1900 > 1895 → sweep
        rows = [
            _row(0, low=1895.0, pool_low=True),
            _row(1, low=1893.0, close=1900.0),
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_low"].iloc[1] is np.bool_(True)

    def test_sweep_low_on_correct_row(self):
        rows = [
            _row(0, low=1895.0, pool_low=True),
            _row(1, low=1896.0, close=1902.0),    # wick above level — not a sweep
            _row(2, low=1893.0, close=1900.0),    # sweep
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_low"].iloc[1] is np.bool_(False)
        assert r["liquidity_sweep_low"].iloc[2] is np.bool_(True)

    def test_sweep_low_close_just_above_level(self):
        rows = [
            _row(0, low=1895.00, pool_low=True),
            _row(1, low=1894.99, close=1895.01),
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_low"].iloc[1] is np.bool_(True)

    def test_sweep_low_with_filler_candles_between(self):
        rows = [
            _row(0, low=1895.0, pool_low=True),
            _row(1, low=1897.0, close=1902.0),
            _row(2, low=1897.0, close=1902.0),
            _row(3, low=1893.0, close=1900.0),
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_low"].iloc[3] is np.bool_(True)


# ---------------------------------------------------------------------------
# 8. Sweep Low Rejected
# ---------------------------------------------------------------------------

class TestSweepLowRejected:

    def test_close_below_level_is_not_sweep(self):
        rows = [
            _row(0, low=1895.0, pool_low=True),
            _row(1, low=1893.0, close=1894.0),    # close < level → breakdown
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_low"].iloc[1] is np.bool_(False)

    def test_close_equal_to_level_is_not_sweep(self):
        rows = [
            _row(0, low=1895.0, pool_low=True),
            _row(1, low=1893.0, close=1895.0),    # close == level, not strictly >
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_low"].iloc[1] is np.bool_(False)

    def test_wick_above_level_no_sweep(self):
        rows = [
            _row(0, low=1895.0, pool_low=True),
            _row(1, low=1896.0, close=1902.0),    # never breaches level
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_low"].iloc[1] is np.bool_(False)


# ---------------------------------------------------------------------------
# 9. Level Updates
# ---------------------------------------------------------------------------

class TestLevelUpdates:

    def test_newer_high_pool_supersedes_older(self):
        # First pool at 1910, second at 1915.
        # Candle with high=1912 breaches 1910 but not 1915 → NOT a sweep.
        rows = [
            _row(0, high=1910.0, pool_high=True),
            _row(1, high=1915.0, pool_high=True),
            _row(2, high=1912.0, close=1905.0),
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_high"].iloc[2] is np.bool_(False)

    def test_updated_high_level_triggers_correct_sweep(self):
        rows = [
            _row(0, high=1910.0, pool_high=True),
            _row(1, high=1915.0, pool_high=True),
            _row(2, high=1916.0, close=1905.0),   # breaches 1915 → sweep
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_high"].iloc[2] is np.bool_(True)

    def test_newer_low_pool_supersedes_older(self):
        # First pool at 1895, second at 1890.
        # Candle with low=1892 breaches 1895 but not 1890 → NOT a sweep.
        rows = [
            _row(0, low=1895.0, pool_low=True),
            _row(1, low=1890.0, pool_low=True),
            _row(2, low=1892.0, close=1900.0),
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_low"].iloc[2] is np.bool_(False)

    def test_updated_low_level_triggers_correct_sweep(self):
        rows = [
            _row(0, low=1895.0, pool_low=True),
            _row(1, low=1890.0, pool_low=True),
            _row(2, low=1889.0, close=1900.0),    # breaches 1890 → sweep
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_low"].iloc[2] is np.bool_(True)


# ---------------------------------------------------------------------------
# 10. Pool Row Cannot Self-Sweep
# ---------------------------------------------------------------------------

class TestPoolRowNotSweepItself:

    def test_pool_high_row_does_not_self_sweep(self):
        # Pool sets active_high_level = highs[i] = 1910.
        # Then checks highs[i] > 1910 → False. No sweep.
        rows = [_row(0, high=1910.0, close=1895.0, pool_high=True)]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_high"].iloc[0] is np.bool_(False)

    def test_pool_low_row_does_not_self_sweep(self):
        rows = [_row(0, low=1895.0, close=1910.0, pool_low=True)]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_low"].iloc[0] is np.bool_(False)


# ---------------------------------------------------------------------------
# 11. Multiple Sweeps
# ---------------------------------------------------------------------------

class TestMultipleSweeps:

    def test_two_separate_high_sweeps(self):
        rows = [
            _row(0, high=1910.0, pool_high=True),
            _row(1, high=1912.0, close=1905.0),   # sweep 1
            _row(2, high=1915.0, pool_high=True), # new pool
            _row(3, high=1917.0, close=1910.0),   # sweep 2
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_high"].iloc[1] is np.bool_(True)
        assert r["liquidity_sweep_high"].iloc[3] is np.bool_(True)

    def test_high_and_low_sweep_same_candle(self):
        # One candle wicks above high pool AND below low pool simultaneously.
        rows = [
            _row(0, high=1910.0, low=1895.0, pool_high=True, pool_low=True),
            _row(1, high=1912.0, low=1893.0, close=1902.0),
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_high"].iloc[1] is np.bool_(True)
        assert r["liquidity_sweep_low"].iloc[1] is np.bool_(True)


# ---------------------------------------------------------------------------
# 12. Independent High/Low Levels
# ---------------------------------------------------------------------------

class TestIndependentHighLow:

    def test_high_pool_does_not_create_low_level(self):
        rows = [
            _row(0, high=1910.0, pool_high=True),
            _row(1, low=1850.0, close=1910.0),    # no low level set → no sweep_low
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_low"].iloc[1] is np.bool_(False)

    def test_low_pool_does_not_create_high_level(self):
        rows = [
            _row(0, low=1895.0, pool_low=True),
            _row(1, high=1950.0, close=1895.0),   # no high level set → no sweep_high
        ]
        r = detect_liquidity_sweeps(_df(rows))
        assert r["liquidity_sweep_high"].iloc[1] is np.bool_(False)


# ---------------------------------------------------------------------------
# 13. Original DataFrame Not Mutated
# ---------------------------------------------------------------------------

class TestDfNotMutated:

    def test_original_df_unchanged(self):
        df = _df([
            _row(0, high=1910.0, pool_high=True),
            _row(1, high=1912.0, close=1905.0),
        ])
        original_cols = list(df.columns)
        _ = detect_liquidity_sweeps(df)
        assert list(df.columns) == original_cols
        assert "liquidity_sweep_high" not in df.columns
        assert "liquidity_sweep_low" not in df.columns
