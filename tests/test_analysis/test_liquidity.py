"""Comprehensive test suite for analysis/liquidity.py — detect_liquidity_pools().

Cluster mechanics
-----------------
- A non-equal swing (equal_high/low=False) resets the counter to 1.
- An equal swing (equal_high/low=True) increments the counter.
- Non-swing rows leave both counters unchanged.
- A pool is declared when the counter reaches 2 (MIN_CLUSTER_SIZE).

Group overview
--------------
 1. TestInputValidation          (5)  — missing columns, empty df
 2. TestOutputSchema             (6)  — columns, dtypes, copy semantics
 3. TestNoSwings                 (2)  — all-zero output
 4. TestAnchorNotPool            (3)  — first swing in a cluster is NOT a pool
 5. TestDoubleTop                (4)  — minimum pool (2 equal highs)
 6. TestDoubleBottom             (4)  — minimum pool (2 equal lows)
 7. TestNonAdjacentCluster       (3)  — gaps between equal swings
 8. TestClusterReset             (4)  — non-equal swing resets counter
 9. TestGrowingCluster           (3)  — 3-swing and 4-swing pools
10. TestClusterSizeValues        (4)  — exact size integers recorded
11. TestIndependentTracking      (3)  — high/low counters are independent
12. TestNonSwingRowsIgnored      (3)  — filler rows never touch counters
13. TestDfNotMutated             (1)  — original df unchanged
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.liquidity import detect_liquidity_pools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = pd.Timestamp("2024-01-01 09:00:00")
_STEP    = pd.Timedelta(minutes=5)


def _row(
    i: int,
    sw_high: bool = False,
    sw_low:  bool = False,
    eq_high: bool = False,
    eq_low:  bool = False,
) -> dict:
    """Minimal row — only the four logic columns required by detect_liquidity_pools."""
    return {
        "timestamp":  _BASE_TS + _STEP * i,
        "swing_high": sw_high,
        "swing_low":  sw_low,
        "equal_high": eq_high,
        "equal_low":  eq_low,
    }


def _filler(i: int) -> dict:
    """Non-swing row."""
    return _row(i)


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_raises_on_missing_swing_high(self):
        df = _df([_row(0)]).drop(columns=["swing_high"])
        with pytest.raises(ValueError, match="swing_high"):
            detect_liquidity_pools(df)

    def test_raises_on_missing_swing_low(self):
        df = _df([_row(0)]).drop(columns=["swing_low"])
        with pytest.raises(ValueError, match="swing_low"):
            detect_liquidity_pools(df)

    def test_raises_on_missing_equal_high(self):
        df = _df([_row(0)]).drop(columns=["equal_high"])
        with pytest.raises(ValueError, match="equal_high"):
            detect_liquidity_pools(df)

    def test_raises_on_missing_equal_low(self):
        df = _df([_row(0)]).drop(columns=["equal_low"])
        with pytest.raises(ValueError, match="equal_low"):
            detect_liquidity_pools(df)

    def test_raises_on_empty_dataframe(self):
        df = _df([_row(0)]).iloc[0:0]
        with pytest.raises(ValueError, match="empty"):
            detect_liquidity_pools(df)


# ---------------------------------------------------------------------------
# 2. Output Schema
# ---------------------------------------------------------------------------

class TestOutputSchema:

    def _result(self):
        return detect_liquidity_pools(_df([_row(0), _row(1)]))

    def test_has_liquidity_pool_high(self):
        assert "liquidity_pool_high" in self._result().columns

    def test_has_liquidity_pool_low(self):
        assert "liquidity_pool_low" in self._result().columns

    def test_has_liquidity_cluster_size_high(self):
        assert "liquidity_cluster_size_high" in self._result().columns

    def test_has_liquidity_cluster_size_low(self):
        assert "liquidity_cluster_size_low" in self._result().columns

    def test_pool_columns_are_bool_dtype(self):
        r = self._result()
        assert r["liquidity_pool_high"].dtype == bool
        assert r["liquidity_pool_low"].dtype == bool

    def test_size_columns_are_int_dtype(self):
        r = self._result()
        assert np.issubdtype(r["liquidity_cluster_size_high"].dtype, np.integer)
        assert np.issubdtype(r["liquidity_cluster_size_low"].dtype, np.integer)


# ---------------------------------------------------------------------------
# 3. No Swings
# ---------------------------------------------------------------------------

class TestNoSwings:

    def test_no_pool_highs_when_no_swing_highs(self):
        rows = [_filler(i) for i in range(5)]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_pool_high"].sum() == 0

    def test_no_pool_lows_when_no_swing_lows(self):
        rows = [_filler(i) for i in range(5)]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_pool_low"].sum() == 0


# ---------------------------------------------------------------------------
# 4. Anchor Not a Pool
# ---------------------------------------------------------------------------

class TestAnchorNotPool:
    """The first swing in a cluster (equal_high=False) starts count=1 — NOT a pool."""

    def test_single_non_equal_high_not_pool(self):
        rows = [_row(0, sw_high=True, eq_high=False)]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_pool_high"].iloc[0] is np.bool_(False)

    def test_single_non_equal_low_not_pool(self):
        rows = [_row(0, sw_low=True, eq_low=False)]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_pool_low"].iloc[0] is np.bool_(False)

    def test_anchor_cluster_size_is_zero(self):
        rows = [_row(0, sw_high=True, eq_high=False)]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_cluster_size_high"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 5. Double Top  (minimum pool)
# ---------------------------------------------------------------------------

class TestDoubleTop:

    def _double_top(self):
        return _df([
            _row(0, sw_high=True, eq_high=False),  # anchor  → count=1
            _row(1, sw_high=True, eq_high=True),   # equal   → count=2  → pool
        ])

    def test_second_swing_is_pool(self):
        r = detect_liquidity_pools(self._double_top())
        assert r["liquidity_pool_high"].iloc[1] is np.bool_(True)

    def test_anchor_is_not_pool(self):
        r = detect_liquidity_pools(self._double_top())
        assert r["liquidity_pool_high"].iloc[0] is np.bool_(False)

    def test_cluster_size_at_pool_is_two(self):
        r = detect_liquidity_pools(self._double_top())
        assert r["liquidity_cluster_size_high"].iloc[1] == 2

    def test_cluster_size_at_anchor_is_zero(self):
        r = detect_liquidity_pools(self._double_top())
        assert r["liquidity_cluster_size_high"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 6. Double Bottom  (minimum pool)
# ---------------------------------------------------------------------------

class TestDoubleBottom:

    def _double_bottom(self):
        return _df([
            _row(0, sw_low=True, eq_low=False),
            _row(1, sw_low=True, eq_low=True),
        ])

    def test_second_swing_is_pool(self):
        r = detect_liquidity_pools(self._double_bottom())
        assert r["liquidity_pool_low"].iloc[1] is np.bool_(True)

    def test_anchor_is_not_pool(self):
        r = detect_liquidity_pools(self._double_bottom())
        assert r["liquidity_pool_low"].iloc[0] is np.bool_(False)

    def test_cluster_size_at_pool_is_two(self):
        r = detect_liquidity_pools(self._double_bottom())
        assert r["liquidity_cluster_size_low"].iloc[1] == 2

    def test_cluster_size_at_anchor_is_zero(self):
        r = detect_liquidity_pools(self._double_bottom())
        assert r["liquidity_cluster_size_low"].iloc[0] == 0


# ---------------------------------------------------------------------------
# 7. Non-Adjacent Cluster  (gaps between equal swings)
# ---------------------------------------------------------------------------

class TestNonAdjacentCluster:
    """
    Filler rows between swing events must NOT reset or break the cluster.

    Row 0  sw_high=T  eq_high=F  → count=1
    Row 1  filler                 → count unchanged
    Row 2  filler                 → count unchanged
    Row 3  sw_high=T  eq_high=T  → count=2  → pool!
    """

    def _rows(self):
        return [
            _row(0, sw_high=True, eq_high=False),
            _filler(1),
            _filler(2),
            _row(3, sw_high=True, eq_high=True),
        ]

    def test_pool_formed_across_gap(self):
        r = detect_liquidity_pools(_df(self._rows()))
        assert r["liquidity_pool_high"].iloc[3] is np.bool_(True)

    def test_anchor_not_pool_despite_gap(self):
        r = detect_liquidity_pools(_df(self._rows()))
        assert r["liquidity_pool_high"].iloc[0] is np.bool_(False)

    def test_filler_rows_not_marked(self):
        r = detect_liquidity_pools(_df(self._rows()))
        assert r["liquidity_pool_high"].iloc[1] is np.bool_(False)
        assert r["liquidity_pool_high"].iloc[2] is np.bool_(False)


# ---------------------------------------------------------------------------
# 8. Cluster Reset
# ---------------------------------------------------------------------------

class TestClusterReset:
    """A non-equal swing (eq_high=False) restarts the counter at 1."""

    def test_reset_row_is_not_pool(self):
        rows = [
            _row(0, sw_high=True, eq_high=False),   # count=1
            _row(1, sw_high=True, eq_high=True),    # count=2  → pool
            _row(2, sw_high=True, eq_high=False),   # reset → count=1, NOT pool
        ]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_pool_high"].iloc[2] is np.bool_(False)

    def test_new_cluster_after_reset_forms_pool(self):
        rows = [
            _row(0, sw_high=True, eq_high=False),   # count=1
            _row(1, sw_high=True, eq_high=True),    # count=2  → pool
            _row(2, sw_high=True, eq_high=False),   # reset → count=1
            _row(3, sw_high=True, eq_high=True),    # count=2  → new pool
        ]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_pool_high"].iloc[3] is np.bool_(True)

    def test_reset_clears_cluster_size(self):
        rows = [
            _row(0, sw_high=True, eq_high=False),
            _row(1, sw_high=True, eq_high=True),
            _row(2, sw_high=True, eq_high=False),   # reset
        ]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_cluster_size_high"].iloc[2] == 0

    def test_multiple_resets_independent(self):
        rows = [
            _row(0, sw_high=True, eq_high=False),   # cluster A anchor
            _row(1, sw_high=True, eq_high=True),    # cluster A pool
            _row(2, sw_high=True, eq_high=False),   # cluster B anchor
            _row(3, sw_high=True, eq_high=True),    # cluster B pool
            _row(4, sw_high=True, eq_high=False),   # cluster C anchor
        ]
        r = detect_liquidity_pools(_df(rows))
        pools = r["liquidity_pool_high"].tolist()
        assert pools == [False, True, False, True, False]


# ---------------------------------------------------------------------------
# 9. Growing Cluster
# ---------------------------------------------------------------------------

class TestGrowingCluster:

    def test_triple_top_all_three_marked_at_row_two(self):
        rows = [
            _row(0, sw_high=True, eq_high=False),   # count=1  no pool
            _row(1, sw_high=True, eq_high=True),    # count=2  pool
            _row(2, sw_high=True, eq_high=True),    # count=3  pool
        ]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_pool_high"].iloc[2] is np.bool_(True)

    def test_triple_top_sizes_grow(self):
        rows = [
            _row(0, sw_high=True, eq_high=False),
            _row(1, sw_high=True, eq_high=True),
            _row(2, sw_high=True, eq_high=True),
        ]
        r = detect_liquidity_pools(_df(rows))
        sizes = r["liquidity_cluster_size_high"].tolist()
        assert sizes == [0, 2, 3]

    def test_four_swing_cluster(self):
        rows = [
            _row(0, sw_high=True, eq_high=False),
            _row(1, sw_high=True, eq_high=True),
            _row(2, sw_high=True, eq_high=True),
            _row(3, sw_high=True, eq_high=True),
        ]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_cluster_size_high"].iloc[3] == 4


# ---------------------------------------------------------------------------
# 10. Cluster Size Values
# ---------------------------------------------------------------------------

class TestClusterSizeValues:

    def test_anchor_size_zero(self):
        rows = [_row(0, sw_high=True, eq_high=False)]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_cluster_size_high"].iloc[0] == 0

    def test_second_equal_size_two(self):
        rows = [
            _row(0, sw_high=True, eq_high=False),
            _row(1, sw_high=True, eq_high=True),
        ]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_cluster_size_high"].iloc[1] == 2

    def test_filler_row_size_zero(self):
        rows = [
            _row(0, sw_high=True, eq_high=False),
            _filler(1),
        ]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_cluster_size_high"].iloc[1] == 0

    def test_low_side_size_independent_of_high(self):
        rows = [
            _row(0, sw_high=True, eq_high=False, sw_low=True, eq_low=False),
            _row(1, sw_high=True, eq_high=True,  sw_low=True, eq_low=True),
            _row(2, sw_high=True, eq_high=True),    # high only
        ]
        r = detect_liquidity_pools(_df(rows))
        # High cluster at row 2 = 3; low cluster at row 2 = 0 (no low event)
        assert r["liquidity_cluster_size_high"].iloc[2] == 3
        assert r["liquidity_cluster_size_low"].iloc[2] == 0


# ---------------------------------------------------------------------------
# 11. Independent High/Low Tracking
# ---------------------------------------------------------------------------

class TestIndependentTracking:

    def test_low_events_do_not_reset_high_counter(self):
        rows = [
            _row(0, sw_high=True, eq_high=False),
            _row(1, sw_low=True,  eq_low=False),   # low event only
            _row(2, sw_high=True, eq_high=True),   # should still form high pool
        ]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_pool_high"].iloc[2] is np.bool_(True)

    def test_high_events_do_not_reset_low_counter(self):
        rows = [
            _row(0, sw_low=True,  eq_low=False),
            _row(1, sw_high=True, eq_high=False),  # high event only
            _row(2, sw_low=True,  eq_low=True),    # should still form low pool
        ]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_pool_low"].iloc[2] is np.bool_(True)

    def test_simultaneous_high_and_low_pool(self):
        rows = [
            _row(0, sw_high=True, eq_high=False, sw_low=True, eq_low=False),
            _row(1, sw_high=True, eq_high=True,  sw_low=True, eq_low=True),
        ]
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_pool_high"].iloc[1] is np.bool_(True)
        assert r["liquidity_pool_low"].iloc[1] is np.bool_(True)


# ---------------------------------------------------------------------------
# 12. Non-Swing Rows Ignored
# ---------------------------------------------------------------------------

class TestNonSwingRowsIgnored:

    def test_many_fillers_between_swings_do_not_break_cluster(self):
        rows = (
            [_row(0, sw_high=True, eq_high=False)]
            + [_filler(i) for i in range(1, 20)]
            + [_row(20, sw_high=True, eq_high=True)]
        )
        r = detect_liquidity_pools(_df(rows))
        assert r["liquidity_pool_high"].iloc[20] is np.bool_(True)

    def test_filler_between_two_clusters_does_not_merge_them(self):
        rows = [
            _row(0, sw_high=True, eq_high=False),   # cluster A anchor
            _row(1, sw_high=True, eq_high=True),    # cluster A pool (count=2)
            _filler(2),
            _row(3, sw_high=True, eq_high=False),   # cluster B anchor (reset)
            _row(4, sw_high=True, eq_high=True),    # cluster B pool (count=2)
        ]
        r = detect_liquidity_pools(_df(rows))
        sizes = r["liquidity_cluster_size_high"].tolist()
        assert sizes == [0, 2, 0, 0, 2]

    def test_cluster_size_zero_on_filler_rows(self):
        rows = [_filler(i) for i in range(10)]
        r = detect_liquidity_pools(_df(rows))
        assert (r["liquidity_cluster_size_high"] == 0).all()
        assert (r["liquidity_cluster_size_low"] == 0).all()


# ---------------------------------------------------------------------------
# 13. Original DataFrame Not Mutated
# ---------------------------------------------------------------------------

class TestDfNotMutated:

    def test_original_df_unchanged(self):
        df = _df([
            _row(0, sw_high=True, eq_high=False),
            _row(1, sw_high=True, eq_high=True),
        ])
        original_cols = list(df.columns)
        _ = detect_liquidity_pools(df)
        assert list(df.columns) == original_cols
        assert "liquidity_pool_high" not in df.columns
        assert "liquidity_pool_low" not in df.columns
        assert "liquidity_cluster_size_high" not in df.columns
        assert "liquidity_cluster_size_low" not in df.columns

