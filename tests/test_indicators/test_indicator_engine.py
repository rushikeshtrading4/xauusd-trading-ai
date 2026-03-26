"""Tests for indicators/indicator_engine.py"""

import numpy as np
import pandas as pd
import pytest

from indicators.indicator_engine import compute_indicators


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ohlcv(n=60, base=100.0, step=0.5, vol=1000.0):
    """Generate a simple trending OHLCV DataFrame."""
    closes = [base + i * step for i in range(n)]
    return pd.DataFrame(
        {
            "open":   [c - 0.3 for c in closes],
            "high":   [c + 1.0 for c in closes],
            "low":    [c - 1.0 for c in closes],
            "close":  closes,
            "volume": [vol] * n,
        }
    )


# ---------------------------------------------------------------------------
# 1. Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_missing_high_raises(self):
        df = _ohlcv(60).drop(columns=["high"])
        with pytest.raises(ValueError):
            compute_indicators(df)

    def test_missing_close_raises(self):
        df = _ohlcv(60).drop(columns=["close"])
        with pytest.raises(ValueError):
            compute_indicators(df)

    def test_missing_volume_raises_when_vwap_enabled(self):
        df = _ohlcv(60).drop(columns=["volume"])
        with pytest.raises(ValueError):
            compute_indicators(df, include_vwap=True)

    def test_missing_volume_ok_when_vwap_disabled(self):
        df = _ohlcv(60).drop(columns=["volume"])
        result = compute_indicators(df, include_vwap=False)
        assert "VWAP" not in result.columns

    def test_invalid_atr_period_raises(self):
        with pytest.raises(ValueError):
            compute_indicators(_ohlcv(60), atr_period=0)

    def test_invalid_rsi_period_raises(self):
        with pytest.raises(ValueError):
            compute_indicators(_ohlcv(60), rsi_period=-1)

    def test_invalid_ema_period_raises(self):
        with pytest.raises(ValueError):
            compute_indicators(_ohlcv(60), ema_periods=[0])


# ---------------------------------------------------------------------------
# 2. Return Type & Shape
# ---------------------------------------------------------------------------

class TestReturnShape:
    def test_returns_dataframe(self):
        assert isinstance(compute_indicators(_ohlcv(60)), pd.DataFrame)

    def test_row_count_unchanged(self):
        df = _ohlcv(60)
        result = compute_indicators(df)
        assert len(result) == len(df)

    def test_index_preserved(self):
        df = _ohlcv(60)
        result = compute_indicators(df)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_original_columns_present(self):
        df = _ohlcv(60)
        result = compute_indicators(df)
        for col in df.columns:
            assert col in result.columns


# ---------------------------------------------------------------------------
# 3. Default Indicator Columns
# ---------------------------------------------------------------------------

class TestDefaultColumns:
    def test_atr_column_present(self):
        assert "ATR" in compute_indicators(_ohlcv(60)).columns

    def test_default_ema_columns_present(self):
        result = compute_indicators(_ohlcv(60))
        for p in [20, 50, 200]:
            assert f"EMA_{p}" in result.columns

    def test_rsi_column_present(self):
        assert "RSI" in compute_indicators(_ohlcv(60)).columns

    def test_vwap_column_present_by_default(self):
        assert "VWAP" in compute_indicators(_ohlcv(60)).columns


# ---------------------------------------------------------------------------
# 4. Custom Parameters
# ---------------------------------------------------------------------------

class TestCustomParameters:
    def test_custom_ema_periods(self):
        result = compute_indicators(_ohlcv(60), ema_periods=[10, 30])
        assert "EMA_10" in result.columns
        assert "EMA_30" in result.columns
        # Default EMAs should NOT be present
        assert "EMA_20" not in result.columns
        assert "EMA_50" not in result.columns
        assert "EMA_200" not in result.columns

    def test_empty_ema_periods(self):
        result = compute_indicators(_ohlcv(60), ema_periods=[])
        for p in [20, 50, 200]:
            assert f"EMA_{p}" not in result.columns

    def test_vwap_excluded_when_disabled(self):
        result = compute_indicators(_ohlcv(60), include_vwap=False)
        assert "VWAP" not in result.columns

    def test_custom_atr_period(self):
        result = compute_indicators(_ohlcv(60), atr_period=7)
        assert "ATR" in result.columns

    def test_custom_rsi_period(self):
        result = compute_indicators(_ohlcv(60), rsi_period=9)
        assert "RSI" in result.columns


# ---------------------------------------------------------------------------
# 5. Immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_original_df_not_modified(self):
        df = _ohlcv(60)
        original_cols = list(df.columns)
        compute_indicators(df)
        assert list(df.columns) == original_cols

    def test_returns_copy_not_original(self):
        df = _ohlcv(60)
        result = compute_indicators(df)
        assert result is not df


# ---------------------------------------------------------------------------
# 6. Value Sanity
# ---------------------------------------------------------------------------

class TestValueSanity:
    def test_atr_values_non_negative(self):
        result = compute_indicators(_ohlcv(60))
        valid = result["ATR"].dropna()
        assert (valid >= 0).all()

    def test_rsi_values_in_range(self):
        result = compute_indicators(_ohlcv(60))
        valid = result["RSI"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_ema_20_values_not_all_nan(self):
        result = compute_indicators(_ohlcv(60))
        assert result["EMA_20"].dropna().shape[0] > 0

    def test_vwap_values_positive(self):
        result = compute_indicators(_ohlcv(60))
        valid = result["VWAP"].dropna()
        assert (valid > 0).all()


# ---------------------------------------------------------------------------
# 7. Up-front validation — stage labels in error messages
# ---------------------------------------------------------------------------

class TestUpFrontValidation:
    def test_missing_high_names_atr_stage(self):
        df = _ohlcv(60).drop(columns=["high"])
        with pytest.raises(ValueError, match="ATR"):
            compute_indicators(df)

    def test_missing_low_names_atr_stage(self):
        df = _ohlcv(60).drop(columns=["low"])
        with pytest.raises(ValueError, match="ATR"):
            compute_indicators(df)

    def test_missing_volume_names_vwap_stage(self):
        df = _ohlcv(60).drop(columns=["volume"])
        with pytest.raises(ValueError, match="VWAP"):
            compute_indicators(df, include_vwap=True)

    def test_error_raised_before_any_computation(self):
        """Validation must fire before compute_atr is even called.

        If 'high' is missing, the error must mention 'missing required column'
        and appear before any indicator column is written to the output.
        """
        df = _ohlcv(60).drop(columns=["high"])
        with pytest.raises(ValueError, match="missing required column"):
            compute_indicators(df)

    def test_vwap_missing_col_not_raised_when_disabled(self):
        """No VWAP validation when include_vwap=False, even if volume absent."""
        df = _ohlcv(60).drop(columns=["volume"])
        result = compute_indicators(df, include_vwap=False)  # must not raise
        assert "VWAP" not in result.columns


# ---------------------------------------------------------------------------
# 8. Pipeline execution order
# ---------------------------------------------------------------------------

class TestPipelineOrder:
    def test_atr_column_appears_before_ema_in_result(self):
        """ATR must precede EMA_* in the appended column order."""
        result = compute_indicators(_ohlcv(60))
        cols = list(result.columns)
        assert cols.index("ATR") < cols.index("EMA_20")

    def test_ema_columns_appear_before_rsi(self):
        result = compute_indicators(_ohlcv(60))
        cols = list(result.columns)
        assert cols.index("EMA_20") < cols.index("RSI")

    def test_rsi_column_appears_before_vwap(self):
        result = compute_indicators(_ohlcv(60))
        cols = list(result.columns)
        assert cols.index("RSI") < cols.index("VWAP")

    def test_all_indicator_columns_appended_after_ohlcv(self):
        """All indicator columns follow the original OHLCV columns."""
        df = _ohlcv(60)
        result = compute_indicators(df)
        original_col_count = len(df.columns)
        indicator_cols = list(result.columns)[original_col_count:]
        assert "ATR" in indicator_cols
        assert "RSI" in indicator_cols
        assert "VWAP" in indicator_cols


# ---------------------------------------------------------------------------
# 9. Market behaviour — system-level trading signal validation
# ---------------------------------------------------------------------------

def _trend_ohlcv(n: int, start: float, step: float, vol: float = 1000.0) -> pd.DataFrame:
    """Uniformly trending OHLCV with realistic spread around each close."""
    closes = [start + i * step for i in range(n)]
    spread = abs(step) * 2 if step != 0 else 1.0
    return pd.DataFrame(
        {
            "open":   [c - spread * 0.3 for c in closes],
            "high":   [c + spread        for c in closes],
            "low":    [c - spread        for c in closes],
            "close":  closes,
            "volume": [vol] * n,
        }
    )


class TestMarketBehavior:
    """Validate that the indicator pipeline produces values consistent with
    real trading regimes.  These tests exercise system *behaviour*, not just
    structural correctness."""

    # ------------------------------------------------------------------
    # Test 1 — Strong Bullish Trend
    # Conditions: 300 steadily rising bars (+1 per bar from 1800).
    # Expected:   EMA20 > EMA50 > EMA200, RSI > 55, Close > VWAP.
    # ------------------------------------------------------------------
    def test_strong_bullish_trend(self):
        df = _trend_ohlcv(300, start=1800.0, step=1.0)
        r = compute_indicators(df, ema_periods=[20, 50, 200])
        last = r.iloc[-1]

        assert last["EMA_20"] > last["EMA_50"],  "Bullish: EMA_20 must be above EMA_50"
        assert last["EMA_50"] > last["EMA_200"], "Bullish: EMA_50 must be above EMA_200"
        assert last["RSI"] > 55,                 "Bullish: RSI must be > 55"
        assert last["close"] > last["VWAP"],     "Bullish: close must be above VWAP"

    # ------------------------------------------------------------------
    # Test 2 — Strong Bearish Trend
    # Conditions: 300 steadily falling bars (-1 per bar from 2100).
    # Expected:   EMA20 < EMA50 < EMA200, RSI < 45, Close < VWAP.
    # ------------------------------------------------------------------
    def test_strong_bearish_trend(self):
        df = _trend_ohlcv(300, start=2100.0, step=-1.0)
        r = compute_indicators(df, ema_periods=[20, 50, 200])
        last = r.iloc[-1]

        assert last["EMA_20"] < last["EMA_50"],  "Bearish: EMA_20 must be below EMA_50"
        assert last["EMA_50"] < last["EMA_200"], "Bearish: EMA_50 must be below EMA_200"
        assert last["RSI"] < 45,                 "Bearish: RSI must be < 45"
        assert last["close"] < last["VWAP"],     "Bearish: close must be below VWAP"

    # ------------------------------------------------------------------
    # Test 3 — Sideways / Ranging Market
    # Conditions: 300 bars oscillating ±1 around a fixed midpoint.
    # Expected:   All three EMA lines converge close together;
    #             RSI stays in the 40–60 neutral band.
    # ------------------------------------------------------------------
    def test_sideways_market(self):
        mid = 2000.0
        n = 300
        closes = [mid + (1.0 if i % 2 == 0 else -1.0) for i in range(n)]
        df = pd.DataFrame(
            {
                "open":   [c - 0.5 for c in closes],
                "high":   [c + 1.0 for c in closes],
                "low":    [c - 1.0 for c in closes],
                "close":  closes,
                "volume": [1000.0] * n,
            }
        )
        r = compute_indicators(df, ema_periods=[20, 50, 200])
        last = r.iloc[-1]

        ema_spread = abs(last["EMA_20"] - last["EMA_200"])
        assert ema_spread < 5.0, (
            f"Sideways: EMA lines should be close together, spread={ema_spread:.4f}"
        )
        assert 40 <= last["RSI"] <= 60, (
            f"Sideways: RSI should be in 40–60 neutral band, got {last['RSI']:.2f}"
        )

    # ------------------------------------------------------------------
    # Test 4 — Volatility Spike
    # Conditions: 60 flat bars, then one candle with high-low range 10×
    #             the typical spread.
    # Expected:   ATR rises noticeably after the spike;
    #             EMA does not spike by the same magnitude.
    # ------------------------------------------------------------------
    def test_volatility_spike_raises_atr(self):
        n_flat = 60
        flat_high  = [2001.0] * n_flat
        flat_low   = [1999.0] * n_flat
        flat_close = [2000.0] * n_flat
        flat_vol   = [1000.0] * n_flat

        # Spike candle: range = 200 (vs normal range of 2)
        spike_high  = flat_high  + [2100.0]
        spike_low   = flat_low   + [1900.0]
        spike_close = flat_close + [2000.0]  # closes back at mid
        spike_vol   = flat_vol   + [1000.0]

        df_flat  = pd.DataFrame({"open": flat_close,  "high": flat_high,  "low": flat_low,  "close": flat_close,  "volume": flat_vol})
        df_spike = pd.DataFrame({"open": spike_close, "high": spike_high, "low": spike_low, "close": spike_close, "volume": spike_vol})

        r_flat  = compute_indicators(df_flat,  ema_periods=[20])
        r_spike = compute_indicators(df_spike, ema_periods=[20])

        atr_before = r_flat["ATR"].iloc[-1]
        atr_after  = r_spike["ATR"].iloc[-1]
        ema_before = r_flat["EMA_20"].iloc[-1]
        ema_after  = r_spike["EMA_20"].iloc[-1]

        assert atr_after > atr_before * 2, (
            f"Spike: ATR should at least double; before={atr_before:.4f} after={atr_after:.4f}"
        )
        # EMA reacts far less than ATR — the spike close equals the flat close,
        # so the EMA shift must be much smaller than the ATR change.
        ema_change = abs(ema_after - ema_before)
        atr_change = abs(atr_after - atr_before)
        assert ema_change < atr_change, (
            f"Spike: EMA change ({ema_change:.4f}) must be smaller than ATR change ({atr_change:.4f})"
        )

    # ------------------------------------------------------------------
    # Test 5 — Volume Dominance
    # Conditions: 30 flat bars at price 2000, then one bar at price 2050
    #             with 100× the volume of background bars.
    # Expected:   VWAP shifts meaningfully toward 2050 after the dominant bar;
    #             removing the dominant bar leaves VWAP close to 2000.
    # ------------------------------------------------------------------
    def test_high_volume_bar_pulls_vwap(self):
        bg_price  = 2000.0
        hv_price  = 2050.0
        bg_vol    = 100.0
        hv_vol    = bg_vol * 100   # dominant bar

        n_bg = 30
        highs  = [bg_price + 1] * n_bg + [hv_price + 1]
        lows   = [bg_price - 1] * n_bg + [hv_price - 1]
        closes = [bg_price]      * n_bg + [hv_price]
        vols   = [bg_vol]        * n_bg + [hv_vol]

        df = pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes, "volume": vols})
        r = compute_indicators(df, ema_periods=[20])

        vwap_final = r["VWAP"].iloc[-1]

        # VWAP must be pulled significantly above the background price
        assert vwap_final > bg_price + 10, (
            f"Volume dominance: VWAP ({vwap_final:.2f}) should be pulled well above {bg_price}"
        )
        # But must not overshoot the dominant bar's price
        assert vwap_final <= hv_price + 1, (
            f"Volume dominance: VWAP ({vwap_final:.2f}) should not exceed dominant bar price {hv_price}"
        )

    # ------------------------------------------------------------------
    # Test 6 — Trend with Pullback
    # Conditions: 100-bar uptrend, 10-bar pullback, 50-bar trend resumption.
    # Expected:   EMA structure remains bullish (20 > 50 > 200);
    #             RSI recovers above 50 after the pullback.
    # Validates stability under partial reversals and noise.
    # ------------------------------------------------------------------
    def test_trend_with_pullback_does_not_break_structure(self):
        # Strong uptrend
        closes = [1800 + i for i in range(100)]

        # Pullback (temporary drop)
        closes += [1900 - i * 2 for i in range(10)]

        # Resume trend — extended to 100 bars so total (210) exceeds EMA_200 warmup
        closes += [1880 + i for i in range(100)]

        df = pd.DataFrame({
            "open":   closes,
            "high":   [c + 2 for c in closes],
            "low":    [c - 2 for c in closes],
            "close":  closes,
            "volume": [1000] * len(closes),
        })

        r = compute_indicators(df, ema_periods=[20, 50, 200])
        last = r.iloc[-1]

        # EMA structure must remain bullish after the pullback
        assert last["EMA_20"] > last["EMA_50"], (
            f"Pullback: EMA_20 ({last['EMA_20']:.2f}) must remain above EMA_50 ({last['EMA_50']:.2f})"
        )
        assert last["EMA_50"] > last["EMA_200"], (
            f"Pullback: EMA_50 ({last['EMA_50']:.2f}) must remain above EMA_200 ({last['EMA_200']:.2f})"
        )

        # RSI must recover above 50 once the trend resumes
        assert last["RSI"] > 50, (
            f"Pullback: RSI ({last['RSI']:.2f}) must recover above 50 after trend resumption"
        )

