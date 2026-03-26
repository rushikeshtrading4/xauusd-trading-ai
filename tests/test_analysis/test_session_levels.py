"""Tests for analysis/session_levels.py — detect_session_levels() and helpers.

Test groups
-----------
 1. TestInputValidation          (7)  — bad input raises ValueError
 2. TestOutputSchema             (8)  — columns, dtypes, shape, immutability
 3. TestSessionClassification   (12)  — session_label per UTC hour
 4. TestSessionOverlapRules      (6)  — in_* booleans are mutually exclusive
 5. TestActiveSessionHighLow    (10)  — running max/min tracking
 6. TestSealedPrevLevels         (8)  — prev_* sealed values
 7. TestDayRollover              (5)  — calendar rollover resets instances
 8. TestSessionBiasLogic         (8)  — BULLISH / BEARISH / NEUTRAL
 9. TestMultiDaySequence         (4)  — realistic multi-day sequence
10. TestInputImmutability        (2)  — original df unchanged
11. TestQueryHelpers            (10)  — helper functions
12. TestOFFSession               (4)  — OFF window behaviour
13. TestDeterminism              (2)  — identical inputs → identical outputs

Total: 86 tests
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from analysis.session_levels import (
    classify_price_vs_session,
    detect_session_levels,
    get_current_session_range,
    get_session_liquidity_targets,
    BIAS_BEARISH,
    BIAS_BULLISH,
    BIAS_NEUTRAL,
    LABEL_ASIA,
    LABEL_LONDON,
    LABEL_NY,
    LABEL_OFF,
)

# ---------------------------------------------------------------------------
# Module-level date constants
# ---------------------------------------------------------------------------

_BASE_DATE = "2026-01-05"   # Monday
_DAY2      = "2026-01-06"   # Tuesday

# ---------------------------------------------------------------------------
# Helpers (match spec exactly)
# ---------------------------------------------------------------------------

def _make_ts(date: str, hour: int, minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp(f"{date} {hour:02d}:{minute:02d}:00", tz="UTC")


def _row(
    date: str,
    hour: int,
    minute: int = 0,
    high: float = 2000.0,
    low:  float = 1990.0,
    close: float = 1995.0,
    open_: float = 1993.0,
) -> dict:
    return {
        "timestamp": _make_ts(date, hour, minute),
        "open":  open_,
        "high":  high,
        "low":   low,
        "close": close,
        "volume": 1000.0,
    }


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# Convenience: 17 output column names
_ALL_OUTPUT_COLS = [
    "session_asia_high",   "session_asia_low",
    "session_london_high", "session_london_low",
    "session_ny_high",     "session_ny_low",
    "prev_asia_high",      "prev_asia_low",
    "prev_london_high",    "prev_london_low",
    "prev_ny_high",        "prev_ny_low",
    "in_asia",             "in_london",    "in_ny",
    "session_label",       "session_bias",
]


# ===========================================================================
# 1. Input validation
# ===========================================================================


class TestInputValidation:

    def test_missing_timestamp_raises(self):
        df = _df([_row(_BASE_DATE, 9)]).drop(columns=["timestamp"])
        with pytest.raises(ValueError, match="timestamp"):
            detect_session_levels(df)

    def test_missing_open_raises(self):
        df = _df([_row(_BASE_DATE, 9)]).drop(columns=["open"])
        with pytest.raises(ValueError, match="open"):
            detect_session_levels(df)

    def test_missing_high_raises(self):
        df = _df([_row(_BASE_DATE, 9)]).drop(columns=["high"])
        with pytest.raises(ValueError, match="high"):
            detect_session_levels(df)

    def test_missing_low_raises(self):
        df = _df([_row(_BASE_DATE, 9)]).drop(columns=["low"])
        with pytest.raises(ValueError, match="low"):
            detect_session_levels(df)

    def test_missing_close_raises(self):
        df = _df([_row(_BASE_DATE, 9)]).drop(columns=["close"])
        with pytest.raises(ValueError, match="close"):
            detect_session_levels(df)

    def test_empty_df_raises(self):
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
        with pytest.raises(ValueError, match="empty"):
            detect_session_levels(df)

    def test_timezone_naive_raises(self):
        df = pd.DataFrame([{
            "timestamp": pd.Timestamp("2026-01-05 09:00:00"),  # naive — no tz
            "open": 1993.0, "high": 2000.0, "low": 1990.0,
            "close": 1995.0, "volume": 1000.0,
        }])
        with pytest.raises(ValueError, match="timezone-aware"):
            detect_session_levels(df)


# ===========================================================================
# 2. Output schema
# ===========================================================================


class TestOutputSchema:

    def _result(self) -> pd.DataFrame:
        return detect_session_levels(_df([_row(_BASE_DATE, 9)]))

    def test_all_17_columns_present(self):
        result = self._result()
        for col in _ALL_OUTPUT_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_in_asia_dtype_is_bool(self):
        assert self._result()["in_asia"].dtype == bool

    def test_in_london_dtype_is_bool(self):
        assert self._result()["in_london"].dtype == bool

    def test_in_ny_dtype_is_bool(self):
        assert self._result()["in_ny"].dtype == bool

    def test_session_label_dtype_is_object(self):
        result = self._result()
        assert pd.api.types.is_string_dtype(result["session_label"])

    def test_session_bias_dtype_is_object(self):
        result = self._result()
        assert pd.api.types.is_string_dtype(result["session_bias"])

    def test_row_count_unchanged(self):
        inp = _df([_row(_BASE_DATE, 9), _row(_BASE_DATE, 10)])
        assert len(detect_session_levels(inp)) == len(inp)

    def test_returns_different_object(self):
        inp = _df([_row(_BASE_DATE, 9)])
        assert detect_session_levels(inp) is not inp


# ===========================================================================
# 3. Session classification — one-row df per hour
# ===========================================================================


class TestSessionClassification:

    def _classify(self, hour: int, minute: int = 0) -> str:
        return detect_session_levels(
            _df([_row(_BASE_DATE, hour, minute)])
        )["session_label"].iloc[0]

    def test_hour_0_is_asia(self):
        assert self._classify(0) == LABEL_ASIA

    def test_hour_3_is_asia(self):
        assert self._classify(3) == LABEL_ASIA

    def test_hour_6_is_asia(self):
        assert self._classify(6) == LABEL_ASIA

    def test_hour_7_is_london(self):
        # 07:00 — London beats Asia in priority
        assert self._classify(7) == LABEL_LONDON

    def test_hour_9_is_london(self):
        assert self._classify(9) == LABEL_LONDON

    def test_hour_12_is_london(self):
        # 12:00 — London beats NY in priority
        assert self._classify(12) == LABEL_LONDON

    def test_hour_15_is_london(self):
        assert self._classify(15) == LABEL_LONDON

    def test_hour_16_is_ny(self):
        # 16:00 is NOT in London [07:00, 16:00); NY takes over
        assert self._classify(16) == LABEL_NY

    def test_hour_17_is_ny(self):
        assert self._classify(17) == LABEL_NY

    def test_hour_20_is_ny(self):
        assert self._classify(20) == LABEL_NY

    def test_hour_21_is_off(self):
        assert self._classify(21) == LABEL_OFF

    def test_hour_23_is_off(self):
        assert self._classify(23) == LABEL_OFF


# ===========================================================================
# 4. Session overlap rules — in_* are mutually exclusive
# ===========================================================================


class TestSessionOverlapRules:

    def _r(self, hour: int) -> pd.Series:
        return detect_session_levels(_df([_row(_BASE_DATE, hour)])).iloc[0]

    def test_hour_7_in_london_not_in_asia(self):
        r = self._r(7)
        assert r["in_london"] == True
        assert r["in_asia"] == False

    def test_hour_12_in_london_not_in_ny(self):
        r = self._r(12)
        assert r["in_london"] == True
        assert r["in_ny"] == False

    def test_hour_16_in_ny_not_in_london(self):
        r = self._r(16)
        assert r["in_london"] == False
        assert r["in_ny"] == True

    def test_hour_21_all_false(self):
        r = self._r(21)
        assert r["in_asia"] == False
        assert r["in_london"] == False
        assert r["in_ny"] == False

    def test_at_most_one_membership_true_per_row(self):
        rows = [_row(_BASE_DATE, h) for h in [3, 7, 12, 16, 21]]
        result = detect_session_levels(_df(rows))
        for i in range(len(result)):
            count = (
                int(result["in_asia"].iloc[i])
                + int(result["in_london"].iloc[i])
                + int(result["in_ny"].iloc[i])
            )
            assert count <= 1, f"Row {i}: expected ≤ 1 True membership, got {count}"

    def test_london_label_iff_in_london(self):
        # Spot-check 3 pure London rows
        rows = [_row(_BASE_DATE, 9), _row(_BASE_DATE, 11), _row(_BASE_DATE, 14)]
        result = detect_session_levels(_df(rows))
        for i in range(len(result)):
            assert result["session_label"].iloc[i] == LABEL_LONDON
            assert result["in_london"].iloc[i] == True


# ===========================================================================
# 5. Active session high / low tracking
# ===========================================================================


class TestActiveSessionHighLow:

    def _asia_seq(self) -> pd.DataFrame:
        rows = [
            _row(_BASE_DATE, 1, high=2010.0, low=1990.0),
            _row(_BASE_DATE, 2, high=2020.0, low=1995.0),
            _row(_BASE_DATE, 3, high=2015.0, low=1992.0),
        ]
        return detect_session_levels(_df(rows))

    def test_asia_seeds_high_from_first_candle(self):
        assert self._asia_seq()["session_asia_high"].iloc[0] == pytest.approx(2010.0)

    def test_asia_extends_running_high(self):
        assert self._asia_seq()["session_asia_high"].iloc[1] == pytest.approx(2020.0)

    def test_asia_high_not_overwritten_by_lower_candle(self):
        # Row 2 has a lower high (2015 < 2020); running max stays 2020
        assert self._asia_seq()["session_asia_high"].iloc[2] == pytest.approx(2020.0)

    def test_asia_running_low_is_cumulative_minimum(self):
        # lows: 1990, 1995, 1992 → min = 1990
        assert self._asia_seq()["session_asia_low"].iloc[2] == pytest.approx(1990.0)

    def test_asia_high_nan_on_pure_london_candle(self):
        # 09:00 — Asia window already closed (exits at 08:00)
        rows = [
            _row(_BASE_DATE, 1, high=2010.0, low=1990.0),  # Asia
            _row(_BASE_DATE, 9, high=2000.0, low=1995.0),  # London (Asia inactive)
        ]
        result = detect_session_levels(_df(rows))
        assert math.isnan(result["session_asia_high"].iloc[1])

    def test_london_tracks_candles_independently(self):
        rows = [
            _row(_BASE_DATE, 8,  high=2000.0, low=1990.0),
            _row(_BASE_DATE, 9,  high=2015.0, low=1992.0),
            _row(_BASE_DATE, 10, high=2010.0, low=1988.0),
        ]
        result = detect_session_levels(_df(rows))
        assert result["session_london_high"].iloc[2] == pytest.approx(2015.0)

    def test_ny_tracks_candles_independently(self):
        rows = [
            _row(_BASE_DATE, 16, high=2005.0, low=1995.0),
            _row(_BASE_DATE, 17, high=2020.0, low=1990.0),
            _row(_BASE_DATE, 18, high=2010.0, low=1985.0),
        ]
        result = detect_session_levels(_df(rows))
        assert result["session_ny_high"].iloc[2] == pytest.approx(2020.0)
        assert result["session_ny_low"].iloc[2] == pytest.approx(1985.0)

    def test_active_level_seeded_from_single_candle(self):
        result = detect_session_levels(_df([_row(_BASE_DATE, 1, high=2010.0, low=1990.0)]))
        assert result["session_asia_high"].iloc[0] == pytest.approx(2010.0)
        assert result["session_asia_low"].iloc[0] == pytest.approx(1990.0)

    def test_non_active_session_levels_nan_during_other_session(self):
        # At 03:00 (pure Asia) London and NY active cols must be NaN
        result = detect_session_levels(_df([_row(_BASE_DATE, 3)]))
        assert math.isnan(result["session_london_high"].iloc[0])
        assert math.isnan(result["session_ny_high"].iloc[0])

    def test_all_active_levels_nan_during_off(self):
        result = detect_session_levels(_df([_row(_BASE_DATE, 22)]))
        r = result.iloc[0]
        for col in ["session_asia_high", "session_asia_low",
                    "session_london_high", "session_london_low",
                    "session_ny_high", "session_ny_low"]:
            assert math.isnan(r[col]), f"Expected NaN for {col}"


# ===========================================================================
# 6. Sealed prev_* levels
# ===========================================================================


class TestSealedPrevLevels:

    def _asia_then_london(self) -> pd.DataFrame:
        # Two Asia candles (h=2020 max, l=1980 min), then London (seals Asia)
        rows = [
            _row(_BASE_DATE, 1, high=2020.0, low=1980.0),
            _row(_BASE_DATE, 2, high=2010.0, low=1985.0),
            _row(_BASE_DATE, 9, high=2000.0, low=1995.0),  # London → seals Asia
        ]
        return detect_session_levels(_df(rows))

    def test_prev_asia_nan_before_seal(self):
        result = self._asia_then_london()
        assert math.isnan(result["prev_asia_high"].iloc[0])
        assert math.isnan(result["prev_asia_high"].iloc[1])

    def test_prev_asia_set_after_asia_ends(self):
        result = self._asia_then_london()
        assert not math.isnan(result["prev_asia_high"].iloc[2])

    def test_prev_asia_high_equals_max_candle(self):
        assert self._asia_then_london()["prev_asia_high"].iloc[2] == pytest.approx(2020.0)

    def test_prev_asia_low_equals_min_candle(self):
        assert self._asia_then_london()["prev_asia_low"].iloc[2] == pytest.approx(1980.0)

    def test_prev_london_nan_while_london_active(self):
        rows = [
            _row(_BASE_DATE, 8,  high=2000.0, low=1990.0),
            _row(_BASE_DATE, 10, high=2010.0, low=1985.0),
        ]
        result = detect_session_levels(_df(rows))
        assert math.isnan(result["prev_london_high"].iloc[0])
        assert math.isnan(result["prev_london_high"].iloc[1])

    def test_prev_london_set_on_first_candle_after_end(self):
        rows = [
            _row(_BASE_DATE, 8,  high=2000.0, low=1990.0),
            _row(_BASE_DATE, 15, high=2050.0, low=1980.0),  # extends London
            _row(_BASE_DATE, 16, high=2010.0, low=2000.0),  # London exits → seals
        ]
        result = detect_session_levels(_df(rows))
        assert result["prev_london_high"].iloc[2] == pytest.approx(2050.0)
        assert result["prev_london_low"].iloc[2] == pytest.approx(1980.0)

    def test_prev_values_persist_unchanged_after_seal(self):
        # Asia seals at index 1; prev_asia must be the same at indexes 1, 2, 3
        rows = [
            _row(_BASE_DATE, 1,  high=2020.0, low=1980.0),  # Asia
            _row(_BASE_DATE, 9,  high=2000.0, low=1995.0),  # London (Asia seals)
            _row(_BASE_DATE, 17, high=1995.0, low=1985.0),  # NY
            _row(_BASE_DATE, 22, high=1990.0, low=1980.0),  # OFF
        ]
        result = detect_session_levels(_df(rows))
        for i in range(1, 4):
            assert result["prev_asia_high"].iloc[i] == pytest.approx(2020.0)
            assert result["prev_asia_low"].iloc[i] == pytest.approx(1980.0)

    def test_second_day_asia_seal_updates_prev_asia(self):
        rows = [
            _row(_BASE_DATE, 1, high=2020.0, low=1980.0),  # Day1 Asia
            _row(_BASE_DATE, 9, high=2000.0, low=1995.0),  # Day1 London (Asia seals)
            _row(_DAY2,      1, high=2060.0, low=2030.0),  # Day2 Asia (new instance)
            _row(_DAY2,      9, high=2070.0, low=2025.0),  # Day2 London (Day2 Asia seals)
        ]
        result = detect_session_levels(_df(rows))
        assert result["prev_asia_high"].iloc[3] == pytest.approx(2060.0)
        assert result["prev_asia_low"].iloc[3] == pytest.approx(2030.0)


# ===========================================================================
# 7. Day rollover
# ===========================================================================


class TestDayRollover:

    def test_day2_asia_high_is_day2_candle_only(self):
        rows = [
            _row(_BASE_DATE, 2, high=2030.0, low=2010.0),  # Day1 Asia h=2030
            _row(_BASE_DATE, 4, high=2020.0, low=2005.0),  # Day1 Asia (lower)
            _row(_DAY2,      2, high=2050.0, low=2040.0),  # Day2 seeds fresh
        ]
        result = detect_session_levels(_df(rows))
        assert result["session_asia_high"].iloc[2] == pytest.approx(2050.0)

    def test_day2_asia_low_is_day2_candle_only(self):
        rows = [
            _row(_BASE_DATE, 2, high=2030.0, low=2010.0),
            _row(_DAY2,      2, high=2050.0, low=2040.0),
        ]
        result = detect_session_levels(_df(rows))
        assert result["session_asia_low"].iloc[1] == pytest.approx(2040.0)

    def test_prev_asia_after_day2_start_is_day1_max(self):
        rows = [
            _row(_BASE_DATE, 2, high=2030.0, low=2010.0),
            _row(_BASE_DATE, 4, high=2020.0, low=2005.0),  # Day1 max = 2030
            _row(_DAY2,      2, high=2050.0, low=2040.0),  # triggers Day1 seal
        ]
        result = detect_session_levels(_df(rows))
        assert result["prev_asia_high"].iloc[2] == pytest.approx(2030.0)

    def test_two_london_sessions_produce_independent_prev_levels(self):
        rows = [
            _row(_BASE_DATE, 9,  high=2100.0, low=2080.0),  # Day1 London
            _row(_BASE_DATE, 16, high=2090.0, low=2085.0),  # Day1 exits → seals 2100
            _row(_DAY2,      9,  high=2200.0, low=2180.0),  # Day2 London
            _row(_DAY2,      16, high=2190.0, low=2185.0),  # Day2 exits → seals 2200
        ]
        result = detect_session_levels(_df(rows))
        assert result["prev_london_high"].iloc[3] == pytest.approx(2200.0)

    def test_day_rollover_resets_active_session_same_hour(self):
        rows = [
            _row(_BASE_DATE, 2, high=2010.0, low=1990.0),  # Day1 02:00
            _row(_DAY2,      2, high=2050.0, low=2030.0),  # Day2 02:00 (same hour)
        ]
        result = detect_session_levels(_df(rows))
        assert result["session_asia_high"].iloc[1] == pytest.approx(2050.0)
        assert result["session_asia_low"].iloc[1] == pytest.approx(2030.0)


# ===========================================================================
# 8. Session bias logic
# ===========================================================================


class TestSessionBiasLogic:

    def _asia_then_london(
        self,
        asia_h: float, asia_l: float,
        lon_h:  float, lon_l:  float,
    ) -> pd.DataFrame:
        # Asia (01:00) seeds; London (09:00) seals Asia and opens.
        return detect_session_levels(_df([
            _row(_BASE_DATE, 1, high=asia_h, low=asia_l),
            _row(_BASE_DATE, 9, high=lon_h,  low=lon_l),
        ]))

    def test_asia_candle_bias_is_neutral(self):
        result = detect_session_levels(_df([_row(_BASE_DATE, 3)]))
        assert result["session_bias"].iloc[0] == BIAS_NEUTRAL

    def test_off_candle_bias_is_neutral(self):
        result = detect_session_levels(_df([_row(_BASE_DATE, 22)]))
        assert result["session_bias"].iloc[0] == BIAS_NEUTRAL

    def test_london_no_prior_asia_bias_neutral(self):
        # Data starts at London hour — no prior Asia seal
        result = detect_session_levels(_df([_row(_BASE_DATE, 9)]))
        assert result["session_bias"].iloc[0] == BIAS_NEUTRAL

    def test_london_bullish_when_mid_above_asia_mid(self):
        # Asia mid = (2020+1980)/2 = 2000; London mid = (2030+2010)/2 = 2020 > 2000
        result = self._asia_then_london(2020.0, 1980.0, 2030.0, 2010.0)
        assert result["session_bias"].iloc[1] == BIAS_BULLISH

    def test_london_bearish_when_mid_below_asia_mid(self):
        # Asia mid = 2000; London mid = (1990+1970)/2 = 1980 < 2000
        result = self._asia_then_london(2020.0, 1980.0, 1990.0, 1970.0)
        assert result["session_bias"].iloc[1] == BIAS_BEARISH

    def test_london_neutral_when_mid_equals_asia_mid(self):
        # Asia mid = 2000; London mid = (2010+1990)/2 = 2000
        result = self._asia_then_london(2020.0, 1980.0, 2010.0, 1990.0)
        assert result["session_bias"].iloc[1] == BIAS_NEUTRAL

    def test_ny_candle_after_london_is_neutral(self):
        rows = [
            _row(_BASE_DATE, 1,  high=2020.0, low=1980.0),  # Asia
            _row(_BASE_DATE, 9,  high=2030.0, low=2010.0),  # London
            _row(_BASE_DATE, 17, high=2015.0, low=2005.0),  # NY
        ]
        result = detect_session_levels(_df(rows))
        assert result["session_bias"].iloc[2] == BIAS_NEUTRAL

    def test_bias_changes_intra_london_as_range_extends(self):
        # Asia mid = 2000.  London bar 1: (1990+1970)/2=1980 → BEARISH.
        # London bar 2: running h=max(1990,2050)=2050, l=min(1970,1975)=1970
        #   → mid=(2050+1970)/2=2010 > 2000 → BULLISH.
        rows = [
            _row(_BASE_DATE, 1,  high=2020.0, low=1980.0),  # Asia
            _row(_BASE_DATE, 9,  high=1990.0, low=1970.0),  # London bar 1
            _row(_BASE_DATE, 10, high=2050.0, low=1975.0),  # London bar 2
        ]
        result = detect_session_levels(_df(rows))
        assert result["session_bias"].iloc[1] == BIAS_BEARISH
        assert result["session_bias"].iloc[2] == BIAS_BULLISH


# ===========================================================================
# 9. Multi-day sequence
# ===========================================================================


class TestMultiDaySequence:
    """Realistic two-day sequence: Day1 Asia → London → NY → OFF, Day2 Asia."""

    def _two_day_result(self) -> pd.DataFrame:
        rows = [
            # Day 1 Asia (hours 1-3)
            _row(_BASE_DATE, 1, high=2010.0, low=1990.0),
            _row(_BASE_DATE, 2, high=2020.0, low=1985.0),  # extends high to 2020
            _row(_BASE_DATE, 3, high=2015.0, low=1988.0),
            # Day 1 London (hour 8 seals Asia; hours 8-10)
            _row(_BASE_DATE, 8,  high=2025.0, low=2005.0),  # idx 3: seals Asia
            _row(_BASE_DATE, 9,  high=2030.0, low=2000.0),
            _row(_BASE_DATE, 10, high=2035.0, low=1995.0),  # London h=2035
            # Day 1 NY (hour 17 seals London; hours 17-18)
            _row(_BASE_DATE, 17, high=2028.0, low=2010.0),  # idx 6: seals London
            _row(_BASE_DATE, 18, high=2040.0, low=2005.0),  # NY h=2040
            # Day 1 OFF (hour 22 seals NY)
            _row(_BASE_DATE, 22, high=2015.0, low=2005.0),  # idx 8: seals NY
            # Day 2 Asia
            _row(_DAY2, 1, high=2000.0, low=1975.0),
            _row(_DAY2, 2, high=2010.0, low=1970.0),  # idx 10
        ]
        return detect_session_levels(_df(rows))

    def test_prev_asia_after_first_london_candle(self):
        result = self._two_day_result()
        # idx 3 = first London candle → Asia seals with max h=2020
        assert result["prev_asia_high"].iloc[3] == pytest.approx(2020.0)

    def test_prev_london_after_first_ny_candle(self):
        result = self._two_day_result()
        # idx 6 = first NY candle → London seals with h=2035
        assert result["prev_london_high"].iloc[6] == pytest.approx(2035.0)

    def test_prev_ny_after_off_candle(self):
        result = self._two_day_result()
        # idx 8 = OFF candle → NY seals with h=2040
        assert result["prev_ny_high"].iloc[8] == pytest.approx(2040.0)

    def test_day2_asia_high_from_day2_candles_only(self):
        result = self._two_day_result()
        # idx 10 = Day2 second Asia candle; running max = max(2000, 2010) = 2010
        assert result["session_asia_high"].iloc[10] == pytest.approx(2010.0)


# ===========================================================================
# 10. Input immutability
# ===========================================================================


class TestInputImmutability:

    def test_returns_different_object(self):
        inp = _df([_row(_BASE_DATE, 9)])
        result = detect_session_levels(inp)
        assert result is not inp

    def test_input_has_no_new_columns_after_call(self):
        inp = _df([_row(_BASE_DATE, 9)])
        cols_before = list(inp.columns)
        detect_session_levels(inp)
        assert list(inp.columns) == cols_before


# ===========================================================================
# 11. Query helpers
# ===========================================================================


class TestQueryHelpers:

    def _sealed_df(self) -> pd.DataFrame:
        """Asia seals, then first London candle."""
        return detect_session_levels(_df([
            _row(_BASE_DATE, 1, high=2020.0, low=1980.0),  # Asia
            _row(_BASE_DATE, 9, high=2030.0, low=2010.0),  # London (Asia seals)
        ]))

    def test_targets_returns_all_8_keys(self):
        targets = get_session_liquidity_targets(self._sealed_df())
        expected = {
            "prev_asia_high", "prev_asia_low",
            "prev_london_high", "prev_london_low",
            "prev_ny_high", "prev_ny_low",
            "session_label", "session_bias",
        }
        assert set(targets.keys()) == expected

    def test_targets_returns_nan_when_no_seal(self):
        # Single Asia candle: sealed yet
        result = detect_session_levels(_df([_row(_BASE_DATE, 1)]))
        targets = get_session_liquidity_targets(result)
        assert math.isnan(targets["prev_asia_high"])

    def test_targets_correct_prev_asia_high_after_seal(self):
        targets = get_session_liquidity_targets(self._sealed_df())
        assert targets["prev_asia_high"] == pytest.approx(2020.0)

    def test_targets_raises_if_session_columns_missing(self):
        df = pd.DataFrame({"price": [1.0]})
        with pytest.raises(ValueError):
            get_session_liquidity_targets(df)

    def test_current_range_london_label_high_low(self):
        rows = [
            _row(_BASE_DATE, 1, high=2020.0, low=1980.0),
            _row(_BASE_DATE, 9, high=2030.0, low=2010.0),  # London (last row)
        ]
        result = detect_session_levels(_df(rows))
        rng = get_current_session_range(result)
        assert rng["label"] == LABEL_LONDON
        assert rng["high"] == pytest.approx(result["session_london_high"].iloc[-1])
        assert rng["low"] == pytest.approx(result["session_london_low"].iloc[-1])

    def test_current_range_off_nan_high_and_low(self):
        rows = [
            _row(_BASE_DATE, 1,  high=2020.0, low=1980.0),
            _row(_BASE_DATE, 22, high=2000.0, low=1990.0),  # OFF
        ]
        result = detect_session_levels(_df(rows))
        rng = get_current_session_range(result)
        assert rng["label"] == LABEL_OFF
        assert math.isnan(rng["high"])
        assert math.isnan(rng["low"])

    def test_classify_unknown_before_any_seal(self):
        # Asia active but not yet sealed
        result = detect_session_levels(_df([_row(_BASE_DATE, 1)]))
        assert classify_price_vs_session(2025.0, result, "ASIA") == "UNKNOWN"

    def test_classify_above(self):
        # prev_asia_high=2020 → 2025 > 2020 → ABOVE
        assert classify_price_vs_session(2025.0, self._sealed_df(), "ASIA") == "ABOVE"

    def test_classify_below_and_inside(self):
        # prev_asia_low=1980 → 1975 < 1980 → BELOW; 2000 is inside
        df = self._sealed_df()
        assert classify_price_vs_session(1975.0, df, "ASIA") == "BELOW"
        assert classify_price_vs_session(2000.0, df, "ASIA") == "INSIDE"

    def test_classify_raises_invalid_session_name(self):
        with pytest.raises(ValueError):
            classify_price_vs_session(2000.0, self._sealed_df(), "INVALID")


# ===========================================================================
# 12. OFF session
# ===========================================================================


class TestOFFSession:

    def test_hour_21_is_off_and_all_bools_false(self):
        result = detect_session_levels(_df([_row(_BASE_DATE, 21)]))
        r = result.iloc[0]
        assert r["session_label"] == LABEL_OFF
        assert r["in_asia"] == False
        assert r["in_london"] == False
        assert r["in_ny"] == False

    def test_hour_23_is_off(self):
        result = detect_session_levels(_df([_row(_BASE_DATE, 23)]))
        assert result["session_label"].iloc[0] == LABEL_OFF

    def test_off_candle_bias_is_neutral(self):
        result = detect_session_levels(_df([_row(_BASE_DATE, 22)]))
        assert result["session_bias"].iloc[0] == BIAS_NEUTRAL

    def test_off_all_active_session_levels_nan(self):
        result = detect_session_levels(_df([_row(_BASE_DATE, 22)]))
        r = result.iloc[0]
        for col in ["session_asia_high", "session_asia_low",
                    "session_london_high", "session_london_low",
                    "session_ny_high", "session_ny_low"]:
            assert math.isnan(r[col]), f"Expected NaN for {col} during OFF"


# ===========================================================================
# 13. Determinism
# ===========================================================================


class TestDeterminism:

    def test_two_calls_identical_output(self):
        rows = [
            _row(_BASE_DATE, 1, high=2010.0, low=1990.0),
            _row(_BASE_DATE, 9, high=2020.0, low=2005.0),
        ]
        inp = _df(rows)
        r1 = detect_session_levels(inp)
        r2 = detect_session_levels(inp)
        pd.testing.assert_frame_equal(r1, r2)

    def test_column_order_identical_both_calls(self):
        rows = [_row(_BASE_DATE, 9)]
        r1 = detect_session_levels(_df(rows))
        r2 = detect_session_levels(_df(rows))
        assert list(r1.columns) == list(r2.columns)
