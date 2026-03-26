"""Comprehensive test suite for analysis/liquidity_equal_levels.py.

ATR = 10.0 throughout most tests → tolerance = 0.10 × 10.0 = 1.0.
This keeps the arithmetic obvious: two swing levels are "equal" when
they are within $1 of each other.

Group overview
--------------
1.  TestInputValidation       (5 tests)  — missing columns, empty df
2.  TestOutputSchema          (4 tests)  — column names, dtypes, copy
3.  TestNoSwings              (2 tests)  — all False when no swings
4.  TestSingleSwing           (3 tests)  — first-ever swing never equal
5.  TestAdjacentEqualHighs    (3 tests)  — within / outside / boundary
6.  TestAdjacentEqualLows     (3 tests)  — within / outside / boundary
7.  TestNonAdjacentCluster    (4 tests)  — KEY new deque behaviour
8.  TestBufferCapacity        (2 tests)  — oldest evicted after 3 swings
9.  TestATRSafetyGuard        (4 tests)  — NaN / zero / negative ATR
10. TestIndependentTracking   (2 tests)  — highs/lows tracked separately
11. TestClusterOfThree        (2 tests)  — all three mark equal
12. TestIndexReset            (1 test)   — output has 0-based index
13. TestDfNotMutated          (1 test)   — original df unchanged
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.liquidity_equal_levels import detect_equal_levels

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASE_TS = pd.Timestamp("2024-01-01 09:00:00")
_STEP    = pd.Timedelta(minutes=5)
_ATR     = 10.0          # tolerance = 0.10 × 10.0 = 1.00


def _row(
    i: int,
    high: float = 1910.0,
    low:  float = 1895.0,
    atr:  float = _ATR,
    sw_high: bool = False,
    sw_low:  bool = False,
) -> dict:
    """Return a single-row dict with all required columns."""
    return {
        "timestamp":  _BASE_TS + _STEP * i,
        "open":       1900.0,
        "high":       high,
        "low":        low,
        "close":      1905.0,
        "volume":     1000,
        "ATR":        atr,
        "swing_high": sw_high,
        "swing_low":  sw_low,
    }


def _filler(i: int) -> dict:
    """Non-swing row — neither swing_high nor swing_low."""
    return _row(i, high=1910.0, low=1895.0)


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """detect_equal_levels() should raise ValueError for bad inputs."""

    def test_raises_on_missing_atr_column(self):
        df = _df([_row(0, sw_high=True)])
        df = df.drop(columns=["ATR"])
        with pytest.raises(ValueError, match="ATR"):
            detect_equal_levels(df)

    def test_raises_on_missing_swing_high_column(self):
        df = _df([_row(0)])
        df = df.drop(columns=["swing_high"])
        with pytest.raises(ValueError, match="swing_high"):
            detect_equal_levels(df)

    def test_raises_on_missing_swing_low_column(self):
        df = _df([_row(0)])
        df = df.drop(columns=["swing_low"])
        with pytest.raises(ValueError, match="swing_low"):
            detect_equal_levels(df)

    def test_raises_on_empty_dataframe(self):
        df = _df([_row(0)]).iloc[0:0]          # zero-row, correct schema
        with pytest.raises(ValueError, match="empty"):
            detect_equal_levels(df)

    def test_error_message_lists_all_missing_columns(self):
        df = _df([_row(0)]).drop(columns=["ATR", "swing_high", "swing_low"])
        with pytest.raises(ValueError) as exc_info:
            detect_equal_levels(df)
        msg = str(exc_info.value)
        assert "ATR" in msg
        assert "swing_high" in msg
        assert "swing_low" in msg


# ---------------------------------------------------------------------------
# 2. Output Schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Returned DataFrame must have the right shape and types."""

    def _result(self):
        return detect_equal_levels(_df([_row(0), _row(1)]))

    def test_output_has_equal_high_column(self):
        assert "equal_high" in self._result().columns

    def test_output_has_equal_low_column(self):
        assert "equal_low" in self._result().columns

    def test_output_preserves_original_columns(self):
        result = self._result()
        for col in ["timestamp", "open", "high", "low", "close", "volume",
                    "ATR", "swing_high", "swing_low"]:
            assert col in result.columns

    def test_equal_columns_are_bool_dtype(self):
        result = self._result()
        assert result["equal_high"].dtype == bool
        assert result["equal_low"].dtype == bool


# ---------------------------------------------------------------------------
# 3. No Swings
# ---------------------------------------------------------------------------

class TestNoSwings:
    """When no rows have swing_high / swing_low the output is all False."""

    def test_no_swing_highs_all_false(self):
        rows = [_filler(i) for i in range(5)]
        result = detect_equal_levels(_df(rows))
        assert result["equal_high"].sum() == 0

    def test_no_swing_lows_all_false(self):
        rows = [_filler(i) for i in range(5)]
        result = detect_equal_levels(_df(rows))
        assert result["equal_low"].sum() == 0


# ---------------------------------------------------------------------------
# 4. Single Swing
# ---------------------------------------------------------------------------

class TestSingleSwing:
    """The very first swing of each type cannot be 'equal' — buffer is empty."""

    def test_first_swing_high_not_equal(self):
        rows = [_row(0, high=1900.0, sw_high=True)]
        result = detect_equal_levels(_df(rows))
        assert result["equal_high"].iloc[0] is np.bool_(False)

    def test_first_swing_low_not_equal(self):
        rows = [_row(0, low=1895.0, sw_low=True)]
        result = detect_equal_levels(_df(rows))
        assert result["equal_low"].iloc[0] is np.bool_(False)

    def test_only_one_swing_high_in_entire_df(self):
        rows = [_filler(i) for i in range(4)]
        rows.append(_row(4, high=1900.0, sw_high=True))
        result = detect_equal_levels(_df(rows))
        assert result["equal_high"].sum() == 0


# ---------------------------------------------------------------------------
# 5. Adjacent Equal Highs
# ---------------------------------------------------------------------------

class TestAdjacentEqualHighs:
    """Two consecutive swing highs — within / outside / at tolerance."""

    def test_adjacent_highs_within_tolerance_are_equal(self):
        # ATR=10 → tolerance=1.0;  |1900.50 - 1900.00| = 0.50 ≤ 1.0
        rows = [
            _row(0, high=1900.00, sw_high=True),
            _row(1, high=1900.50, sw_high=True),
        ]
        result = detect_equal_levels(_df(rows))
        assert result["equal_high"].iloc[0] is np.bool_(False)
        assert result["equal_high"].iloc[1] is np.bool_(True)

    def test_adjacent_highs_outside_tolerance_not_equal(self):
        # |1902.00 - 1900.00| = 2.0  > 1.0
        rows = [
            _row(0, high=1900.00, sw_high=True),
            _row(1, high=1902.00, sw_high=True),
        ]
        result = detect_equal_levels(_df(rows))
        assert result["equal_high"].iloc[1] is np.bool_(False)

    def test_adjacent_highs_exactly_at_tolerance_are_equal(self):
        # |1901.00 - 1900.00| = 1.0 == tolerance=1.0  →  <= True
        rows = [
            _row(0, high=1900.00, sw_high=True),
            _row(1, high=1901.00, sw_high=True),
        ]
        result = detect_equal_levels(_df(rows))
        assert result["equal_high"].iloc[1] is np.bool_(True)


# ---------------------------------------------------------------------------
# 6. Adjacent Equal Lows
# ---------------------------------------------------------------------------

class TestAdjacentEqualLows:
    """Mirrors Group 5 for swing lows."""

    def test_adjacent_lows_within_tolerance_are_equal(self):
        # |1895.40 - 1895.00| = 0.40 ≤ 1.0
        rows = [
            _row(0, low=1895.00, sw_low=True),
            _row(1, low=1895.40, sw_low=True),
        ]
        result = detect_equal_levels(_df(rows))
        assert result["equal_low"].iloc[0] is np.bool_(False)
        assert result["equal_low"].iloc[1] is np.bool_(True)

    def test_adjacent_lows_outside_tolerance_not_equal(self):
        # |1897.50 - 1895.00| = 2.5 > 1.0
        rows = [
            _row(0, low=1895.00, sw_low=True),
            _row(1, low=1897.50, sw_low=True),
        ]
        result = detect_equal_levels(_df(rows))
        assert result["equal_low"].iloc[1] is np.bool_(False)

    def test_adjacent_lows_exactly_at_tolerance_are_equal(self):
        # |1896.00 - 1895.00| = 1.0 == tolerance=1.0
        rows = [
            _row(0, low=1895.00, sw_low=True),
            _row(1, low=1896.00, sw_low=True),
        ]
        result = detect_equal_levels(_df(rows))
        assert result["equal_low"].iloc[1] is np.bool_(True)


# ---------------------------------------------------------------------------
# 7. Non-Adjacent Cluster  (key deque behaviour)
# ---------------------------------------------------------------------------

class TestNonAdjacentCluster:
    """
    Swing A → different swing B → swing C near A.
    Because B is in the buffer alongside A, C should be marked equal to A
    even though A and C are not directly consecutive.

    Sequence (ATR=10, tolerance=1.0):
        Row 0: swing_high=1900.00  (first → not equal; buffer=[1900.00])
        Row 1: filler
        Row 2: swing_high=1905.00  (|5|>1 → not equal; buffer=[1900.00, 1905.00])
        Row 3: filler
        Row 4: swing_high=1900.40  (|0.40|≤1 vs 1900.00 → EQUAL)
    """

    def _build_high_cluster(self):
        return [
            _row(0, high=1900.00, sw_high=True),
            _filler(1),
            _row(2, high=1905.00, sw_high=True),
            _filler(3),
            _row(4, high=1900.40, sw_high=True),
        ]

    def test_non_adjacent_cluster_high_marked_equal(self):
        result = detect_equal_levels(_df(self._build_high_cluster()))
        assert result["equal_high"].iloc[4] is np.bool_(True)

    def test_intervening_high_not_equal(self):
        # Row 2 (1905.00) is NOT near 1900.00 → must not be marked equal
        result = detect_equal_levels(_df(self._build_high_cluster()))
        assert result["equal_high"].iloc[2] is np.bool_(False)

    def test_non_adjacent_cluster_low_marked_equal(self):
        rows = [
            _row(0, low=1895.00, sw_low=True),
            _filler(1),
            _row(2, low=1890.00, sw_low=True),
            _filler(3),
            _row(4, low=1895.60, sw_low=True),   # |0.60| ≤ 1.0
        ]
        result = detect_equal_levels(_df(rows))
        assert result["equal_low"].iloc[4] is np.bool_(True)

    def test_current_high_matches_multiple_buffer_entries(self):
        # A=1900.00, B=1900.20, C=1900.40 — C is within 1.0 of both A and B
        rows = [
            _row(0, high=1900.00, sw_high=True),
            _row(1, high=1900.20, sw_high=True),
            _row(2, high=1900.40, sw_high=True),
        ]
        result = detect_equal_levels(_df(rows))
        # Both rows 1 and 2 should be equal (match something in the buffer)
        assert result["equal_high"].iloc[1] is np.bool_(True)
        assert result["equal_high"].iloc[2] is np.bool_(True)


# ---------------------------------------------------------------------------
# 8. Buffer Capacity
# ---------------------------------------------------------------------------

class TestBufferCapacity:
    """The deque holds at most 3 entries; the oldest is silently dropped."""

    def test_swing_beyond_buffer_depth_is_not_equal(self):
        """
        4 distinct highs fill + overflow the buffer.  A 5th high near the
        evicted FIRST entry must NOT be flagged as equal.

        ATR=10 → tol=1.0.  Step=5 between each swing so no accidental match.
            Row 0: 1900.00 → buffer: [1900.00]
            Row 1: 1905.00 → buffer: [1900.00, 1905.00]
            Row 2: 1910.00 → buffer: [1900.00, 1905.00, 1910.00]
            Row 3: 1915.00 → buffer: [1905.00, 1910.00, 1915.00]  (1900 evicted)
            Row 4: 1900.40 → compare vs {1905, 1910, 1915} → NOT equal
        """
        rows = [_row(i, high=1900.0 + i * 5, sw_high=True) for i in range(4)]
        rows.append(_row(4, high=1900.40, sw_high=True))
        result = detect_equal_levels(_df(rows))
        assert result["equal_high"].iloc[4] is np.bool_(False)

    def test_swing_within_buffer_depth_is_equal(self):
        """
        3 distinct highs fill the buffer exactly.  A 4th high near the FIRST
        entry MUST be flagged as equal (buffer=[1900, 1905, 1910]; 1900 present).

            Row 0: 1900.00 → buffer: [1900.00]
            Row 1: 1905.00 → buffer: [1900.00, 1905.00]
            Row 2: 1910.00 → buffer: [1900.00, 1905.00, 1910.00]
            Row 3: 1900.40 → compare vs {1900, 1905, 1910} → EQUAL (to 1900)
        """
        rows = [_row(i, high=1900.0 + i * 5, sw_high=True) for i in range(3)]
        rows.append(_row(3, high=1900.40, sw_high=True))
        result = detect_equal_levels(_df(rows))
        assert result["equal_high"].iloc[3] is np.bool_(True)


# ---------------------------------------------------------------------------
# 9. ATR Safety Guard
# ---------------------------------------------------------------------------

class TestATRSafetyGuard:
    """NaN / zero / negative ATR collapses tolerance to 0.0."""

    def test_nan_atr_exact_match_is_equal(self):
        rows = [
            _row(0, high=1900.00, atr=float("nan"), sw_high=True),
            _row(1, high=1900.00, atr=float("nan"), sw_high=True),
        ]
        result = detect_equal_levels(_df(rows))
        assert result["equal_high"].iloc[1] is np.bool_(True)

    def test_nan_atr_near_miss_not_equal(self):
        rows = [
            _row(0, high=1900.00, atr=float("nan"), sw_high=True),
            _row(1, high=1900.01, atr=float("nan"), sw_high=True),
        ]
        result = detect_equal_levels(_df(rows))
        assert result["equal_high"].iloc[1] is np.bool_(False)

    def test_zero_atr_exact_match_is_equal(self):
        rows = [
            _row(0, low=1895.00, atr=0.0, sw_low=True),
            _row(1, low=1895.00, atr=0.0, sw_low=True),
        ]
        result = detect_equal_levels(_df(rows))
        assert result["equal_low"].iloc[1] is np.bool_(True)

    def test_negative_atr_near_miss_not_equal(self):
        rows = [
            _row(0, high=1900.00, atr=-5.0, sw_high=True),
            _row(1, high=1900.50, atr=-5.0, sw_high=True),
        ]
        result = detect_equal_levels(_df(rows))
        assert result["equal_high"].iloc[1] is np.bool_(False)


# ---------------------------------------------------------------------------
# 10. Independent Tracking
# ---------------------------------------------------------------------------

class TestIndependentTracking:
    """
    swing_high events feed only the high buffer; swing_low events feed only
    the low buffer.  Cross-contamination must be impossible.
    """

    def test_high_buffer_unaffected_by_lows(self):
        """Interleaved sw_low rows must not prevent sw_high equality detection."""
        rows = [
            _row(0, high=1900.00, low=1895.00, sw_high=True),
            _row(1, high=1910.00, low=1894.00, sw_low=True),   # low event only
            _row(2, high=1900.50, low=1910.00, sw_high=True),  # near 1900.00
        ]
        result = detect_equal_levels(_df(rows))
        assert result["equal_high"].iloc[2] is np.bool_(True)

    def test_low_buffer_unaffected_by_highs(self):
        """Interleaved sw_high rows must not prevent sw_low equality detection."""
        rows = [
            _row(0, high=1900.00, low=1895.00, sw_low=True),
            _row(1, high=1908.00, low=1890.00, sw_high=True),  # high event only
            _row(2, high=1902.00, low=1895.40, sw_low=True),   # near 1895.00
        ]
        result = detect_equal_levels(_df(rows))
        assert result["equal_low"].iloc[2] is np.bool_(True)


# ---------------------------------------------------------------------------
# 11. Cluster of Three
# ---------------------------------------------------------------------------

class TestClusterOfThree:
    """All three equal swings in a tight cluster — the first is not marked."""

    def test_three_equal_highs(self):
        # A=1900.00, B=1900.30, C=1900.60 — all within 1.0 of each other
        rows = [
            _row(0, high=1900.00, sw_high=True),
            _row(1, high=1900.30, sw_high=True),
            _row(2, high=1900.60, sw_high=True),
        ]
        result = detect_equal_levels(_df(rows))
        assert result["equal_high"].iloc[0] is np.bool_(False)
        assert result["equal_high"].iloc[1] is np.bool_(True)
        assert result["equal_high"].iloc[2] is np.bool_(True)

    def test_three_equal_lows(self):
        rows = [
            _row(0, low=1895.00, sw_low=True),
            _row(1, low=1895.30, sw_low=True),
            _row(2, low=1895.60, sw_low=True),
        ]
        result = detect_equal_levels(_df(rows))
        assert result["equal_low"].iloc[0] is np.bool_(False)
        assert result["equal_low"].iloc[1] is np.bool_(True)
        assert result["equal_low"].iloc[2] is np.bool_(True)


# ---------------------------------------------------------------------------
# 12. Index Reset
# ---------------------------------------------------------------------------

class TestIndexReset:
    """Output must have a clean RangeIndex even when input has a custom index."""

    def test_output_index_is_reset(self):
        df = _df([_row(i) for i in range(5)])
        df.index = [10, 20, 30, 40, 50]          # non-standard input index
        result = detect_equal_levels(df)
        expected = pd.RangeIndex(5)
        pd.testing.assert_index_equal(result.index, expected)


# ---------------------------------------------------------------------------
# 13. Original DataFrame Not Mutated
# ---------------------------------------------------------------------------

class TestDfNotMutated:
    """detect_equal_levels() must return a *copy* and leave the input intact."""

    def test_original_df_has_no_equal_columns(self):
        df = _df([_row(0, sw_high=True), _row(1, high=1900.40, sw_high=True)])
        _ = detect_equal_levels(df)
        assert "equal_high" not in df.columns
        assert "equal_low" not in df.columns

