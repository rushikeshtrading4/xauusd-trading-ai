"""Comprehensive test suite for indicators/atr.py — compute_atr().

Test group overview
-------------------
 1. TestInputValidation       (5)  — missing columns, bad period
 2. TestOutputShape           (4)  — length, name, index alignment, Series type
 3. TestWarmupNaN             (4)  — first (period-1) values are NaN
 4. TestFlatData              (3)  — ATR == 0 when all candles are identical
 5. TestTrueRangeGap          (4)  — gap between sessions increases TR
 6. TestVolatilitySpike       (3)  — ATR rises after a large candle
 7. TestWilderSmoothing       (3)  — formula correctness for small inputs
 8. TestImmutability          (2)  — input df not modified
 9. TestPeriod1               (2)  — period=1 gives TR directly
10. TestCustomPeriod          (2)  — period=3 has shorter warmup
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators.atr import compute_atr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(
    highs:  list[float],
    lows:   list[float],
    closes: list[float],
) -> pd.DataFrame:
    """Build a minimal OHLC-like DataFrame from price lists."""
    assert len(highs) == len(lows) == len(closes)
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


def _flat(n: int = 20, price: float = 2000.0) -> pd.DataFrame:
    """DataFrame of *n* identical candles (zero true range)."""
    return _df([price] * n, [price] * n, [price] * n)


_PERIOD = 14


# ===========================================================================
# 1. Input validation
# ===========================================================================

class TestInputValidation:

    def test_missing_high_raises(self):
        df = _df([1], [1], [1]).drop(columns=["high"])
        with pytest.raises(ValueError, match="high"):
            compute_atr(df)

    def test_missing_low_raises(self):
        df = _df([1], [1], [1]).drop(columns=["low"])
        with pytest.raises(ValueError, match="low"):
            compute_atr(df)

    def test_missing_close_raises(self):
        df = _df([1], [1], [1]).drop(columns=["close"])
        with pytest.raises(ValueError, match="close"):
            compute_atr(df)

    def test_period_zero_raises(self):
        with pytest.raises(ValueError, match="period"):
            compute_atr(_flat(), period=0)

    def test_period_negative_raises(self):
        with pytest.raises(ValueError, match="period"):
            compute_atr(_flat(), period=-5)


# ===========================================================================
# 2. Output shape and metadata
# ===========================================================================

class TestOutputShape:

    def test_returns_series(self):
        assert isinstance(compute_atr(_flat(20)), pd.Series)

    def test_length_matches_input(self):
        df = _flat(30)
        assert len(compute_atr(df)) == len(df)

    def test_series_named_atr(self):
        assert compute_atr(_flat(20)).name == "ATR"

    def test_index_aligned_with_df(self):
        df = _flat(20)
        result = compute_atr(df)
        pd.testing.assert_index_equal(result.index, df.index)


# ===========================================================================
# 3. Warmup NaN
# ===========================================================================

class TestWarmupNaN:

    def test_first_period_minus_1_are_nan(self):
        result = compute_atr(_flat(30), period=_PERIOD)
        assert result.iloc[: _PERIOD - 1].isna().all()

    def test_value_at_period_minus_1_is_nan(self):
        result = compute_atr(_flat(30), period=_PERIOD)
        assert pd.isna(result.iloc[_PERIOD - 2])

    def test_first_valid_at_index_period_minus_1(self):
        # index _PERIOD - 1 is the last NaN; index _PERIOD - 1 + 1 = _PERIOD - 1 should still be NaN
        # first non-NaN is at position period - 1
        result = compute_atr(_flat(30), period=_PERIOD)
        assert not pd.isna(result.iloc[_PERIOD - 1])

    def test_no_nan_after_warmup_on_clean_data(self):
        result = compute_atr(_flat(30), period=_PERIOD)
        assert result.iloc[_PERIOD - 1 :].notna().all()


# ===========================================================================
# 4. Flat data — ATR should be 0
# ===========================================================================

class TestFlatData:

    def test_atr_is_zero_on_flat_candles(self):
        result = compute_atr(_flat(30))
        valid = result.dropna()
        assert (valid == 0.0).all()

    def test_atr_zero_for_period_3_flat(self):
        result = compute_atr(_flat(10), period=3)
        assert result.iloc[2:].eq(0.0).all()

    def test_atr_dtype_is_float(self):
        result = compute_atr(_flat(20))
        assert result.dtype == float


# ===========================================================================
# 5. True Range gap handling
# ===========================================================================

class TestTrueRangeGap:

    def _gap_df(self) -> pd.DataFrame:
        """Candles with a gap: close=2000 then open gap to 2020."""
        # Row 0: normal candle, close=2000
        # Row 1: gap up — low=2015 (> prev close 2000) — TR = |2015-2000| = 15
        highs  = [2005, 2025, 2025, 2025]
        lows   = [1995, 2015, 2015, 2015]
        closes = [2000, 2020, 2020, 2020]
        return _df(highs, lows, closes)

    def test_gap_increases_tr_vs_intrabar(self):
        df = self._gap_df()
        # intra-bar range (high-low) = 10, but |low - prev_close| = 15
        # so TR on row 1 should be 15, not 10
        high  = df["high"]
        low   = df["low"]
        prev  = df["close"].shift(1)
        tr_intrabar = (high - low).iloc[1]
        tr_gap_component = (low - prev).abs().iloc[1]
        assert tr_gap_component > tr_intrabar

    def test_atr_reflects_gap(self):
        # With a gap, ATR on row 1 should be > 0
        df = self._gap_df()
        result = compute_atr(df, period=2)
        assert result.iloc[1] > 0

    def test_no_gap_lower_atr(self):
        # Same intra-bar range but no gap: ATR should be lower than gap version
        highs  = [2005, 2005, 2005, 2005]
        lows   = [1995, 1995, 1995, 1995]
        closes = [2000, 2000, 2000, 2000]
        df_no_gap = _df(highs, lows, closes)
        df_gap    = self._gap_df()
        atr_no_gap = compute_atr(df_no_gap, period=2).iloc[-1]
        atr_gap    = compute_atr(df_gap,    period=2).iloc[-1]
        assert atr_gap > atr_no_gap

    def test_first_row_tr_is_only_intrabar(self):
        """Row 0 has no previous close so TR = high - low only."""
        highs  = [2010]
        lows   = [1990]
        closes = [2000]
        df = _df(highs, lows, closes)
        # period=1 means no warmup NaN
        result = compute_atr(df, period=1)
        assert result.iloc[0] == pytest.approx(20.0)


# ===========================================================================
# 6. Volatility spike raises ATR
# ===========================================================================

class TestVolatilitySpike:

    def test_large_candle_increases_atr(self):
        """ATR after a large candle must exceed ATR before it."""
        # 20 flat candles (range 1), then one large candle (range 100)
        n = 20
        highs  = [2001.0] * n + [2100.0]
        lows   = [1999.0] * n + [1900.0]
        closes = [2000.0] * n + [2000.0]
        df = _df(highs, lows, closes)
        result = compute_atr(df, period=_PERIOD)
        atr_before = result.iloc[n - 1]
        atr_after  = result.iloc[n]
        assert atr_after > atr_before

    def test_atr_decays_after_spike(self):
        """After the spike, ATR gradually decays back toward baseline."""
        n_base  = 20
        n_decay = 10
        highs  = [2001.0] * n_base + [2100.0] + [2001.0] * n_decay
        lows   = [1999.0] * n_base + [1900.0] + [1999.0] * n_decay
        closes = [2000.0] * (n_base + 1 + n_decay)
        df = _df(highs, lows, closes)
        result = compute_atr(df, period=_PERIOD)
        spike_idx = n_base
        assert result.iloc[spike_idx + n_decay] < result.iloc[spike_idx]

    def test_atr_positive_after_spike(self):
        highs  = [2001.0] * 20 + [2100.0]
        lows   = [1999.0] * 20 + [1900.0]
        closes = [2000.0] * 21
        result = compute_atr(_df(highs, lows, closes), period=_PERIOD)
        assert result.iloc[-1] > 0


# ===========================================================================
# 7. Wilder's smoothing correctness
# ===========================================================================

class TestWilderSmoothing:

    def test_period_1_equals_tr(self):
        """period=1 → alpha=1 → ewm simply returns TR itself."""
        highs  = [10, 20, 15]
        lows   = [ 5, 10,  8]
        closes = [ 8, 15, 12]
        df = _df(highs, lows, closes)
        result = compute_atr(df, period=1)
        # TR row0 = 10-5 = 5; row1: max(20-10, |20-8|, |10-8|) = max(10,12,2) = 12
        assert result.iloc[0] == pytest.approx(5.0)
        assert result.iloc[1] == pytest.approx(12.0)

    def test_wilder_second_value_formula(self):
        """Manually verify ATR[1] = (ATR[0] * (p-1) + TR[1]) / p for period=2."""
        highs  = [2010, 2020, 2015]
        lows   = [1990, 2000, 1990]
        closes = [2000, 2010, 2000]
        df = _df(highs, lows, closes)
        result = compute_atr(df, period=2)
        # TR[0] = 20, TR[1] = max(20, |2020-2000|, |2000-2000|) = 20
        # ATR[0] (period=2, first valid at idx 1) = (TR[0]+TR[1])/2 via ewm init = 20
        # ATR[2] = (ATR[1] * 1 + TR[2]) / 2
        # TR[2] = max(2015-1990, |2015-2010|, |1990-2010|) = max(25, 5, 20) = 25
        expected = (result.iloc[1] * 1 + 25.0) / 2
        assert result.iloc[2] == pytest.approx(expected, rel=1e-6)

    def test_atr_is_always_non_negative(self):
        import random
        random.seed(42)
        prices = sorted([random.uniform(1900, 2100) for _ in range(50)])
        closes = prices
        highs  = [c + random.uniform(0, 10) for c in closes]
        lows   = [c - random.uniform(0, 10) for c in closes]
        df = _df(highs, lows, closes)
        result = compute_atr(df, period=5)
        assert (result.dropna() >= 0).all()


# ===========================================================================
# 8. Input immutability
# ===========================================================================

class TestImmutability:

    def test_original_df_not_modified(self):
        df = _flat(20)
        original_cols = set(df.columns)
        original_values = df.copy()
        compute_atr(df)
        assert set(df.columns) == original_cols
        pd.testing.assert_frame_equal(df, original_values)

    def test_no_atr_column_added_to_df(self):
        df = _flat(20)
        compute_atr(df)
        assert "ATR" not in df.columns


# ===========================================================================
# 9. Period = 1
# ===========================================================================

class TestPeriod1:

    def test_no_nan_values(self):
        """period=1 means warmup = 0 NaN values."""
        result = compute_atr(_flat(10), period=1)
        assert result.notna().all()

    def test_all_zeros_on_flat_data(self):
        result = compute_atr(_flat(10), period=1)
        assert (result == 0.0).all()


# ===========================================================================
# 10. Custom period
# ===========================================================================

class TestCustomPeriod:

    def test_period_3_warmup_is_2(self):
        result = compute_atr(_flat(10), period=3)
        assert result.iloc[:2].isna().all()
        assert result.iloc[2:].notna().all()

    def test_shorter_period_reacts_faster_to_spike(self):
        """Shorter period ATR responds more strongly to a single large candle."""
        n = 20
        highs  = [2001.0] * n + [2200.0]
        lows   = [1999.0] * n + [1800.0]
        closes = [2000.0] * (n + 1)
        df = _df(highs, lows, closes)
        atr_short = compute_atr(df, period=3).iloc[-1]
        atr_long  = compute_atr(df, period=14).iloc[-1]
        assert atr_short > atr_long
