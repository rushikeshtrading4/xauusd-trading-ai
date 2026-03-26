"""Tests for indicators/rsi.py"""

import math
import numpy as np
import pandas as pd
import pytest

from indicators.rsi import compute_rsi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(closes):
    return pd.DataFrame({"close": closes})


def _rising(n=30, start=100.0, step=1.0):
    return _make_df([start + i * step for i in range(n)])


def _falling(n=30, start=130.0, step=1.0):
    return _make_df([start - i * step for i in range(n)])


def _flat(n=30, price=100.0):
    return _make_df([price] * n)


# ---------------------------------------------------------------------------
# 1. Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_missing_close_raises(self):
        df = pd.DataFrame({"open": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing required column"):
            compute_rsi(df)

    def test_period_zero_raises(self):
        with pytest.raises(ValueError, match="period must be"):
            compute_rsi(_make_df([1, 2, 3]), period=0)

    def test_period_negative_raises(self):
        with pytest.raises(ValueError, match="period must be"):
            compute_rsi(_make_df([1, 2, 3]), period=-5)

    def test_extra_columns_ignored(self):
        df = _rising(20)
        df["volume"] = 1000
        result = compute_rsi(df, period=5)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# 2. Output Shape & Metadata
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_returns_series(self):
        assert isinstance(compute_rsi(_rising(20), period=5), pd.Series)

    def test_length_matches_input(self):
        df = _rising(30)
        assert len(compute_rsi(df, period=14)) == 30

    def test_name_is_rsi(self):
        assert compute_rsi(_rising(20), period=5).name == "RSI"

    def test_index_aligned_with_df(self):
        df = _rising(20)
        result = compute_rsi(df, period=5)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_dtype_is_float(self):
        result = compute_rsi(_rising(20), period=5)
        assert result.dtype == float


# ---------------------------------------------------------------------------
# 3. Warmup NaN
# ---------------------------------------------------------------------------

class TestWarmupNaN:
    def test_first_period_rows_are_nan(self):
        period = 7
        result = compute_rsi(_rising(30), period=period)
        assert result.iloc[:period].isna().all()

    def test_last_warmup_row_is_nan(self):
        period = 10
        result = compute_rsi(_rising(30), period=period)
        assert math.isnan(result.iloc[period - 1])

    def test_first_valid_row_is_not_nan(self):
        period = 10
        result = compute_rsi(_rising(30), period=period)
        assert not math.isnan(result.iloc[period])

    def test_no_nan_after_warmup(self):
        period = 5
        result = compute_rsi(_rising(30), period=period)
        assert not result.iloc[period:].isna().any()

    def test_default_period_14_warmup(self):
        result = compute_rsi(_rising(40))
        assert result.iloc[:14].isna().all()
        assert not result.iloc[14:].isna().any()


# ---------------------------------------------------------------------------
# 4. Flat Price → RSI undefined / stable
# ---------------------------------------------------------------------------

class TestFlatPrice:
    def test_flat_price_rsi_is_nan_or_50(self):
        # delta is 0; avg_gain = avg_loss = 0 → 0/0 → NaN → we keep as NaN
        result = compute_rsi(_flat(30), period=5)
        valid = result.iloc[5:]
        # All valid values should be NaN (0/0 case) or exactly 50
        assert valid.apply(lambda v: math.isnan(v) or v == pytest.approx(50.0)).all()

    def test_flat_does_not_produce_out_of_range(self):
        result = compute_rsi(_flat(30), period=5)
        valid = result.dropna()
        assert ((valid >= 0) & (valid <= 100)).all()


# ---------------------------------------------------------------------------
# 5. Strictly Rising Prices → RSI should be high (> 50 after warmup)
# ---------------------------------------------------------------------------

class TestRisingTrend:
    def test_rsi_above_50_on_rising_data(self):
        result = compute_rsi(_rising(40), period=5)
        valid = result.dropna()
        assert (valid > 50).all()

    def test_rsi_not_above_100(self):
        result = compute_rsi(_rising(40), period=5)
        assert (result.dropna() <= 100).all()

    def test_rsi_below_0_never(self):
        result = compute_rsi(_rising(40), period=5)
        assert (result.dropna() >= 0).all()


# ---------------------------------------------------------------------------
# 6. Strictly Falling Prices → RSI should be low (< 50 after warmup)
# ---------------------------------------------------------------------------

class TestFallingTrend:
    def test_rsi_below_50_on_falling_data(self):
        result = compute_rsi(_falling(40), period=5)
        valid = result.dropna()
        assert (valid < 50).all()

    def test_rsi_above_0_on_falling_data(self):
        result = compute_rsi(_falling(40), period=5)
        assert (result.dropna() >= 0).all()

    def test_rsi_not_below_0(self):
        result = compute_rsi(_falling(40), period=5)
        assert (result.dropna() >= 0).all()


# ---------------------------------------------------------------------------
# 7. Continuous Rise → RSI → 100
# ---------------------------------------------------------------------------

class TestExtremes:
    def test_non_stop_rally_rsi_approaches_100(self):
        # After many bars of uninterrupted gains, RSI should be very high
        result = compute_rsi(_rising(200, step=1.0), period=14)
        assert result.dropna().iloc[-1] > 99.0

    def test_non_stop_decline_rsi_approaches_0(self):
        result = compute_rsi(_falling(200, step=1.0), period=14)
        assert result.dropna().iloc[-1] < 1.0

    def test_rsi_100_when_only_gains(self):
        # avg_loss = 0, avg_gain > 0 → RSI must be 100
        df = _rising(50, step=2.0)
        result = compute_rsi(df, period=5)
        # After warmup, RSI should saturate at 100
        assert result.dropna().iloc[-1] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 8. RSI Range [0, 100]
# ---------------------------------------------------------------------------

class TestRange:
    def test_mixed_data_within_range(self):
        rng = np.random.default_rng(42)
        closes = 100 + rng.standard_normal(100).cumsum()
        result = compute_rsi(_make_df(closes), period=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_volatile_spike_within_range(self):
        closes = [100.0] * 20 + [200.0] + [100.0] * 20
        result = compute_rsi(_make_df(closes), period=5)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


# ---------------------------------------------------------------------------
# 9. Immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_original_df_not_modified(self):
        df = _rising(20)
        original_cols = list(df.columns)
        compute_rsi(df, period=5)
        assert list(df.columns) == original_cols

    def test_no_rsi_column_added_to_df(self):
        df = _rising(20)
        compute_rsi(df, period=5)
        assert "RSI" not in df.columns


# ---------------------------------------------------------------------------
# 10. Period=1 Edge Case
# ---------------------------------------------------------------------------

class TestPeriod1:
    def test_period_1_no_warmup_nan(self):
        # period=1 → warmup = first 1 row (only row 0) is NaN
        result = compute_rsi(_rising(10), period=1)
        assert math.isnan(result.iloc[0])
        assert not result.iloc[1:].isna().any()

    def test_period_1_rising_gives_100(self):
        result = compute_rsi(_rising(10), period=1)
        # Every candle is a gain → RSI = 100
        assert np.allclose(result.dropna().values, 100.0)

