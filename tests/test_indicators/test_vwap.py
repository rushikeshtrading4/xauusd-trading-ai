"""Tests for indicators/vwap.py"""

import math
import numpy as np
import pandas as pd
import pytest

from indicators.vwap import compute_vwap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(highs, lows, closes, volumes):
    return pd.DataFrame(
        {"high": highs, "low": lows, "close": closes, "volume": volumes}
    )


def _uniform(n=10, high=105.0, low=95.0, close=100.0, volume=1000.0):
    """All bars identical — VWAP should equal the typical price throughout."""
    return _make_df([high] * n, [low] * n, [close] * n, [volume] * n)


# ---------------------------------------------------------------------------
# 1. Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_missing_high_raises(self):
        df = pd.DataFrame({"low": [1], "close": [1], "volume": [1]})
        with pytest.raises(ValueError, match="missing required column"):
            compute_vwap(df)

    def test_missing_low_raises(self):
        df = pd.DataFrame({"high": [1], "close": [1], "volume": [1]})
        with pytest.raises(ValueError, match="missing required column"):
            compute_vwap(df)

    def test_missing_close_raises(self):
        df = pd.DataFrame({"high": [1], "low": [1], "volume": [1]})
        with pytest.raises(ValueError, match="missing required column"):
            compute_vwap(df)

    def test_missing_volume_raises(self):
        df = pd.DataFrame({"high": [1], "low": [1], "close": [1]})
        with pytest.raises(ValueError, match="missing required column"):
            compute_vwap(df)

    def test_missing_multiple_columns_raises(self):
        df = pd.DataFrame({"close": [1]})
        with pytest.raises(ValueError, match="missing required column"):
            compute_vwap(df)

    def test_extra_columns_are_ignored(self):
        df = _uniform(5)
        df["open"] = 99.0
        result = compute_vwap(df)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# 2. Output Shape & Metadata
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_returns_series(self):
        assert isinstance(compute_vwap(_uniform(10)), pd.Series)

    def test_length_matches_input(self):
        df = _uniform(20)
        assert len(compute_vwap(df)) == 20

    def test_name_is_vwap(self):
        assert compute_vwap(_uniform(10)).name == "VWAP"

    def test_index_aligned_with_df(self):
        df = _uniform(10)
        result = compute_vwap(df)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_dtype_is_float(self):
        result = compute_vwap(_uniform(10))
        assert result.dtype == float


# ---------------------------------------------------------------------------
# 3. Correctness — Uniform Bars
# ---------------------------------------------------------------------------

class TestUniformBars:
    def test_vwap_equals_typical_price_uniform(self):
        # typical_price = (105 + 95 + 100) / 3 = 100.0
        df = _uniform(10, high=105.0, low=95.0, close=100.0, volume=1000.0)
        result = compute_vwap(df)
        expected = (105.0 + 95.0 + 100.0) / 3.0
        np.testing.assert_allclose(result.values, expected)

    def test_vwap_stable_across_bars_with_equal_volume(self):
        df = _uniform(15)
        result = compute_vwap(df)
        # All values should be identical
        assert result.std() == pytest.approx(0.0, abs=1e-10)

    def test_vwap_single_bar(self):
        df = _make_df([110], [90], [100], [500])
        result = compute_vwap(df)
        expected = (110 + 90 + 100) / 3.0
        assert result.iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 4. Correctness — Manual Calculation
# ---------------------------------------------------------------------------

class TestManualCalculation:
    def test_two_bars_manual(self):
        # Bar 1: H=110, L=90, C=100, V=100 → TP=100, TP*V=10000
        # Bar 2: H=120, L=100, C=110, V=200 → TP=110, TP*V=22000
        # VWAP[0] = 10000 / 100 = 100
        # VWAP[1] = (10000+22000) / (100+200) = 32000/300 ≈ 106.667
        df = _make_df([110, 120], [90, 100], [100, 110], [100, 200])
        result = compute_vwap(df)
        assert result.iloc[0] == pytest.approx(100.0)
        assert result.iloc[1] == pytest.approx(32000.0 / 300.0)

    def test_three_bars_accumulation(self):
        # Verify VWAP is cumulative (not rolling)
        highs  = [100, 102, 104]
        lows   = [ 98, 100, 102]
        closes = [ 99, 101, 103]
        vols   = [100, 100, 100]
        df = _make_df(highs, lows, closes, vols)
        result = compute_vwap(df)
        tp = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        for i in range(3):
            cum_tpv = sum(tp[j] * vols[j] for j in range(i + 1))
            cum_v   = sum(vols[j] for j in range(i + 1))
            assert result.iloc[i] == pytest.approx(cum_tpv / cum_v)


# ---------------------------------------------------------------------------
# 5. Cumulative (Not Rolling) Behaviour
# ---------------------------------------------------------------------------

class TestCumulativeBehaviour:
    def test_vwap_is_non_decreasing_on_rising_closes_equal_volumes(self):
        # Rising closes push TP up on each bar, so VWAP must be non-decreasing
        df = _make_df(
            [101 + i for i in range(10)],
            [ 99 + i for i in range(10)],
            [100 + i for i in range(10)],
            [500] * 10,
        )
        result = compute_vwap(df)
        diffs = result.diff().dropna()
        assert (diffs >= -1e-9).all()

    def test_vwap_not_equal_to_close(self):
        # In general VWAP != close (it averages high/low too)
        df = _make_df([110] * 5, [80] * 5, [100] * 5, [1000] * 5)
        result = compute_vwap(df)
        typical = (110 + 80 + 100) / 3.0   # 96.667, not 100
        assert result.iloc[0] == pytest.approx(typical)
        assert typical != 100.0


# ---------------------------------------------------------------------------
# 6. Zero Volume Handling
# ---------------------------------------------------------------------------

class TestZeroVolume:
    def test_first_bar_zero_volume_gives_nan(self):
        df = _make_df([100], [90], [95], [0])
        result = compute_vwap(df)
        assert math.isnan(result.iloc[0])

    def test_later_bar_zero_volume_does_not_corrupt_vwap(self):
        # Bar 1 has volume; bar 2 has zero; VWAP[2] == VWAP[1]
        df = _make_df([100, 105], [90, 95], [95, 100], [1000, 0])
        result = compute_vwap(df)
        assert result.iloc[1] == pytest.approx(result.iloc[0])


# ---------------------------------------------------------------------------
# 7. Higher Volume Pulls VWAP
# ---------------------------------------------------------------------------

class TestVolumeWeighting:
    def test_high_volume_bar_pulls_vwap_towards_its_price(self):
        # Bar 1: TP=100, vol=1
        # Bar 2: TP=200, vol=999
        # VWAP should be close to 200 (dominated by bar 2)
        df = _make_df([105, 205], [95, 195], [100, 200], [1, 999])
        result = compute_vwap(df)
        assert result.iloc[1] > 190.0

    def test_equal_volume_bars_vwap_is_average_of_tp(self):
        df = _make_df([110, 90], [90, 70], [100, 80], [100, 100])
        result = compute_vwap(df)
        tp1 = (110 + 90 + 100) / 3.0
        tp2 = (90 + 70 + 80)   / 3.0
        expected_final = (tp1 + tp2) / 2.0
        assert result.iloc[1] == pytest.approx(expected_final)


# ---------------------------------------------------------------------------
# 8. Immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_original_df_not_modified(self):
        df = _uniform(10)
        original_cols = list(df.columns)
        compute_vwap(df)
        assert list(df.columns) == original_cols

    def test_no_vwap_column_added_to_df(self):
        df = _uniform(10)
        compute_vwap(df)
        assert "VWAP" not in df.columns


# ---------------------------------------------------------------------------
# 9. Custom Index Support
# ---------------------------------------------------------------------------

class TestCustomIndex:
    def test_datetime_index_preserved(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="1h")
        df = _uniform(5)
        df.index = idx
        result = compute_vwap(df)
        pd.testing.assert_index_equal(result.index, idx)

