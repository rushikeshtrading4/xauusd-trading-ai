"""Comprehensive test suite for indicators/ema.py — compute_ema().

Test group overview
-------------------
 1. TestInputValidation    (4)  — missing close, bad period
 2. TestOutputShape        (5)  — Series type, length, name, index alignment
 3. TestWarmupNaN          (4)  — first (period-1) values are NaN
 4. TestFlatPrice          (3)  — EMA == price on constant data
 5. TestPeriod1            (3)  — period=1 → EMA equals close exactly
 6. TestTrendFollowing     (4)  — EMA lags price, follows trend direction
 7. TestSpikeResponse      (3)  — EMA reacts gradually to a price spike
 8. TestNamingConvention   (3)  — name includes period number
 9. TestImmutability       (2)  — input df not modified
10. TestAlphaFormula       (2)  — alpha = 2/(period+1) used correctly
"""

from __future__ import annotations

import pandas as pd
import pytest

from indicators.ema import compute_ema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": closes})


def _flat(n: int = 30, price: float = 2000.0) -> pd.DataFrame:
    return _df([price] * n)


# ===========================================================================
# 1. Input validation
# ===========================================================================

class TestInputValidation:

    def test_missing_close_raises(self):
        with pytest.raises(ValueError, match="close"):
            compute_ema(pd.DataFrame({"open": [1.0]}), period=5)

    def test_period_zero_raises(self):
        with pytest.raises(ValueError, match="period"):
            compute_ema(_flat(), period=0)

    def test_period_negative_raises(self):
        with pytest.raises(ValueError, match="period"):
            compute_ema(_flat(), period=-3)

    def test_empty_df_still_raises_if_no_close(self):
        with pytest.raises(ValueError, match="close"):
            compute_ema(pd.DataFrame(), period=5)


# ===========================================================================
# 2. Output shape and metadata
# ===========================================================================

class TestOutputShape:

    def test_returns_series(self):
        assert isinstance(compute_ema(_flat(), period=5), pd.Series)

    def test_length_matches_input(self):
        df = _flat(40)
        assert len(compute_ema(df, period=10)) == len(df)

    def test_index_aligned_with_df(self):
        df = _flat(20)
        pd.testing.assert_index_equal(compute_ema(df, period=5).index, df.index)

    def test_name_is_ema_period(self):
        result = compute_ema(_flat(), period=20)
        assert result.name == "EMA_20"

    def test_dtype_is_float(self):
        assert compute_ema(_flat(), period=5).dtype == float


# ===========================================================================
# 3. Warmup NaN
# ===========================================================================

class TestWarmupNaN:

    def test_first_period_minus_1_are_nan(self):
        result = compute_ema(_flat(30), period=10)
        assert result.iloc[:9].isna().all()

    def test_last_warmup_position_is_nan(self):
        result = compute_ema(_flat(30), period=10)
        assert pd.isna(result.iloc[8])

    def test_first_valid_position_is_not_nan(self):
        result = compute_ema(_flat(30), period=10)
        assert not pd.isna(result.iloc[9])

    def test_no_nan_after_warmup(self):
        result = compute_ema(_flat(30), period=5)
        assert result.iloc[4:].notna().all()


# ===========================================================================
# 4. Flat price
# ===========================================================================

class TestFlatPrice:

    def test_ema_equals_price_on_flat_data(self):
        result = compute_ema(_flat(30, price=2000.0), period=10)
        valid = result.dropna()
        assert (valid == 2000.0).all()

    def test_flat_price_zero_gives_zero_ema(self):
        result = compute_ema(_flat(20, price=0.0), period=5)
        assert (result.dropna() == 0.0).all()

    def test_flat_ema_stable_after_warmup(self):
        result = compute_ema(_flat(50, price=1500.0), period=14)
        valid = result.dropna()
        assert valid.std() == pytest.approx(0.0, abs=1e-10)


# ===========================================================================
# 5. Period = 1
# ===========================================================================

class TestPeriod1:

    def test_period_1_equals_close(self):
        closes = [1990.0, 2000.0, 2010.0, 1995.0]
        result = compute_ema(_df(closes), period=1)
        assert list(result) == pytest.approx(closes)

    def test_period_1_no_nan(self):
        result = compute_ema(_flat(10), period=1)
        assert result.notna().all()

    def test_period_1_name_correct(self):
        assert compute_ema(_flat(5), period=1).name == "EMA_1"


# ===========================================================================
# 6. Trend following
# ===========================================================================

class TestTrendFollowing:

    def test_ema_below_price_in_uptrend(self):
        """In an uptrend, EMA lags → it should be below the latest close."""
        closes = [float(i) for i in range(1, 51)]   # 1, 2, …, 50
        result = compute_ema(_df(closes), period=10)
        assert result.iloc[-1] < closes[-1]

    def test_ema_above_price_in_downtrend(self):
        closes = [float(50 - i) for i in range(50)]  # 50, 49, …, 1
        result = compute_ema(_df(closes), period=10)
        assert result.iloc[-1] > closes[-1]

    def test_ema_is_monotone_increasing_on_strictly_rising_prices(self):
        closes = [float(i) for i in range(1, 31)]
        result = compute_ema(_df(closes), period=5)
        valid = result.dropna()
        diffs = valid.diff().dropna()
        assert (diffs > 0).all()

    def test_ema_is_monotone_decreasing_on_strictly_falling_prices(self):
        closes = [float(30 - i) for i in range(30)]
        result = compute_ema(_df(closes), period=5)
        valid = result.dropna()
        diffs = valid.diff().dropna()
        assert (diffs < 0).all()


# ===========================================================================
# 7. Spike response
# ===========================================================================

class TestSpikeResponse:

    def test_ema_does_not_jump_to_spike_immediately(self):
        """After a large spike, EMA must be strictly below the spike price."""
        closes = [2000.0] * 20 + [3000.0]   # sudden +1000 spike
        result = compute_ema(_df(closes), period=10)
        assert result.iloc[-1] < 3000.0

    def test_ema_moves_toward_spike(self):
        """EMA after spike must be above the pre-spike level."""
        closes = [2000.0] * 20 + [3000.0]
        result = compute_ema(_df(closes), period=10)
        ema_before = result.iloc[19]
        ema_after  = result.iloc[20]
        assert ema_after > ema_before

    def test_ema_decays_back_after_spike(self):
        """With flat prices after spike, EMA should converge back down."""
        closes = [2000.0] * 20 + [3000.0] + [2000.0] * 30
        result = compute_ema(_df(closes), period=10)
        assert result.iloc[-1] < result.iloc[20]


# ===========================================================================
# 8. Naming convention
# ===========================================================================

class TestNamingConvention:

    def test_period_20_name(self):
        assert compute_ema(_flat(), period=20).name == "EMA_20"

    def test_period_50_name(self):
        assert compute_ema(_flat(60), period=50).name == "EMA_50"

    def test_period_200_name(self):
        assert compute_ema(_flat(210), period=200).name == "EMA_200"


# ===========================================================================
# 9. Input immutability
# ===========================================================================

class TestImmutability:

    def test_original_df_not_modified(self):
        df = _flat(20)
        original = df.copy()
        compute_ema(df, period=5)
        pd.testing.assert_frame_equal(df, original)

    def test_no_ema_column_added_to_df(self):
        df = _flat(20)
        compute_ema(df, period=5)
        assert "EMA_5" not in df.columns


# ===========================================================================
# 10. Alpha formula verification
# ===========================================================================

class TestAlphaFormula:

    def test_period_2_alpha_is_two_thirds(self):
        """period=2 → seed=SMA(first 2); next bar uses alpha=2/3."""
        # SMA seed at index 1: (100 + 200) / 2 = 150
        # EMA[2] = 300 * (2/3) + 150 * (1/3) = 250
        closes = [100.0, 200.0, 300.0]
        result = compute_ema(_df(closes), period=2)
        assert result.iloc[2] == pytest.approx(250.0, rel=1e-9)

    def test_period_3_second_valid_value(self):
        """Verify the recursive formula for period=3 (alpha=0.5) after SMA seed."""
        # SMA seed at index 2: (100 + 200 + 300) / 3 = 200
        # EMA[3] = 400 * 0.5 + 200 * 0.5 = 300
        closes = [100.0, 200.0, 300.0, 400.0]
        result = compute_ema(_df(closes), period=3)
        assert result.iloc[3] == pytest.approx(300.0, rel=1e-9)

    def test_sma_seed_at_period_minus_1(self):
        """The seed value at index period-1 must equal the SMA of the first period closes."""
        closes = [100.0, 110.0, 120.0, 130.0, 140.0]
        period = 3
        result = compute_ema(_df(closes), period=period)
        expected_seed = (100.0 + 110.0 + 120.0) / 3.0
        assert result.iloc[period - 1] == pytest.approx(expected_seed, rel=1e-9)
