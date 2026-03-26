"""Comprehensive test suite for ai/bias_model.py — calculate_bias().

Weights: D1=0.5, H4=0.3, H1=0.2
Event multipliers: BOS_CONFIRMED=1.0, BREAK=0.6, CHOCH/none=0.0
Score    →  Bias / Strength
> 0.7    →  BULLISH  / STRONG
0.3–0.7  →  BULLISH  / WEAK
-0.3–0.3 →  NEUTRAL  / WEAK
-0.7–-0.3→  BEARISH  / WEAK
< -0.7   →  BEARISH  / STRONG

Override: H1+H4 both BULLISH  + any recent CHOCH + abs(score)>0.2 → BULLISH / WEAK / REVERSAL
          H1+H4 both BEARISH  + any recent CHOCH + abs(score)>0.2 → BEARISH / WEAK / REVERSAL

Group overview
--------------
 1. TestOutputSchema         (5)  — keys and value types
 2. TestScoreCalculation     (8)  — weighted arithmetic with event multipliers
 3. TestBiasStrength         (8)  — threshold boundary tests (thresholds 0.7/0.3)
 4. TestFullyAligned         (4)  — all-BULLISH / all-BEARISH → TREND
 5. TestMixedAlignment       (3)  — mixed trends → CONSOLIDATION
 6. TestCHOCHContext         (4)  — any CHOCH → REVERSAL context
 7. TestOverrideRule         (6)  — H1+H4 aligned + CHOCH override
 8. TestMissingData          (6)  — missing keys / empty df / None graceful
 9. TestLastRowOnly          (3)  — only last row of each df is used
10. TestChochLookback        (3)  — CHOCH must be within last 10 rows
11. TestDeterminism          (2)  — same input → same output always
12. TestNotMutating          (1)  — input dfs are not modified
13. TestEventBasedWeighting  (6)  — event multipliers applied correctly
"""

from __future__ import annotations

import pandas as pd
import pytest

from ai.bias_model import calculate_bias

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = pd.Timestamp("2024-01-01")
_STEP = pd.Timedelta(hours=1)


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _single(trend: str, event: str = "") -> pd.DataFrame:
    """One-row DataFrame for a single timeframe."""
    return _df([{"timestamp": _TS, "trend": trend, "event": event}])


def _multi(trends: list[str], events: list[str] | None = None) -> pd.DataFrame:
    """Multi-row DataFrame; last row carries the given final trend/event."""
    if events is None:
        events = [""] * len(trends)
    rows = [
        {"timestamp": _TS + _STEP * i, "trend": t, "event": e}
        for i, (t, e) in enumerate(zip(trends, events))
    ]
    return _df(rows)


def _mtf(d1: str = "BULLISH", h4: str = "BULLISH", h1: str = "BULLISH",
         d1_ev: str = "", h4_ev: str = "", h1_ev: str = "") -> dict:
    """Build a minimal mtf_data dict from trend + event strings."""
    return {
        "D1": _single(d1, d1_ev),
        "H4": _single(h4, h4_ev),
        "H1": _single(h1, h1_ev),
    }


# ---------------------------------------------------------------------------
# 1. Output Schema
# ---------------------------------------------------------------------------

class TestOutputSchema:

    def _result(self):
        return calculate_bias(_mtf())

    def test_has_bias_key(self):
        assert "bias" in self._result()

    def test_has_strength_key(self):
        assert "strength" in self._result()

    def test_has_score_key(self):
        assert "score" in self._result()

    def test_has_context_key(self):
        assert "context" in self._result()

    def test_bias_valid_values(self):
        assert self._result()["bias"] in ("BULLISH", "BEARISH", "NEUTRAL")


# ---------------------------------------------------------------------------
# 2. Score Calculation
# ---------------------------------------------------------------------------

class TestScoreCalculation:

    def test_all_bullish_score_is_1(self):
        r = calculate_bias(_mtf("BULLISH", "BULLISH", "BULLISH",
                                d1_ev="BOS_CONFIRMED", h4_ev="BOS_CONFIRMED", h1_ev="BOS_CONFIRMED"))
        assert r["score"] == pytest.approx(1.0)

    def test_all_bearish_score_is_minus1(self):
        r = calculate_bias(_mtf("BEARISH", "BEARISH", "BEARISH",
                                d1_ev="BOS_CONFIRMED", h4_ev="BOS_CONFIRMED", h1_ev="BOS_CONFIRMED"))
        assert r["score"] == pytest.approx(-1.0)

    def test_all_transition_score_is_0(self):
        # TRANSITION → TREND_SCORE = 0.0, so multiplier doesn't matter
        r = calculate_bias(_mtf("TRANSITION", "TRANSITION", "TRANSITION"))
        assert r["score"] == pytest.approx(0.0)

    def test_d1_only_bullish_score(self):
        # D1=+1 × 0.5 × 1.0 = 0.5, H4=0, H1=0
        r = calculate_bias(_mtf("BULLISH", "TRANSITION", "TRANSITION", d1_ev="BOS_CONFIRMED"))
        assert r["score"] == pytest.approx(0.5)

    def test_d1_bearish_h4_h1_bullish(self):
        # −0.5 + 0.3 + 0.2 = 0.0
        r = calculate_bias(_mtf("BEARISH", "BULLISH", "BULLISH",
                                d1_ev="BOS_CONFIRMED", h4_ev="BOS_CONFIRMED", h1_ev="BOS_CONFIRMED"))
        assert r["score"] == pytest.approx(0.0)

    def test_h4_only_bearish(self):
        # 0 − 0.3 + 0 = −0.3
        r = calculate_bias(_mtf("TRANSITION", "BEARISH", "TRANSITION", h4_ev="BOS_CONFIRMED"))
        assert r["score"] == pytest.approx(-0.3)

    def test_d1_bullish_h4_bearish_h1_transition(self):
        # 0.5 − 0.3 + 0 = 0.2
        r = calculate_bias(_mtf("BULLISH", "BEARISH", "TRANSITION",
                                d1_ev="BOS_CONFIRMED", h4_ev="BOS_CONFIRMED"))
        assert r["score"] == pytest.approx(0.2)

    def test_d1_bullish_h4_transition_h1_bearish(self):
        # 0.5 + 0 − 0.2 = 0.3
        r = calculate_bias(_mtf("BULLISH", "TRANSITION", "BEARISH",
                                d1_ev="BOS_CONFIRMED", h1_ev="BOS_CONFIRMED"))
        assert r["score"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 3. Bias and Strength Thresholds
# ---------------------------------------------------------------------------

class TestBiasStrength:

    def test_strong_bullish_above_threshold(self):
        # D1+H4 BULLISH BOS_CONFIRMED: 0.5+0.3=0.8 > 0.7 → STRONG BULLISH
        r = calculate_bias(_mtf("BULLISH", "BULLISH", "TRANSITION",
                                d1_ev="BOS_CONFIRMED", h4_ev="BOS_CONFIRMED"))
        assert r["bias"] == "BULLISH"
        assert r["strength"] == "STRONG"

    def test_weak_bullish_between_03_and_07(self):
        # D1=BULLISH(BOS, 0.5), rest TRANSITION → 0.5. 0.3 < 0.5 < 0.7 → WEAK
        r = calculate_bias(_mtf("BULLISH", "TRANSITION", "TRANSITION", d1_ev="BOS_CONFIRMED"))
        assert r["bias"] == "BULLISH"
        assert r["strength"] == "WEAK"

    def test_strong_bearish_below_threshold(self):
        # D1+H4 BEARISH BOS_CONFIRMED: -0.5-0.3=-0.8 < -0.7 → STRONG BEARISH
        r = calculate_bias(_mtf("BEARISH", "BEARISH", "TRANSITION",
                                d1_ev="BOS_CONFIRMED", h4_ev="BOS_CONFIRMED"))
        assert r["bias"] == "BEARISH"
        assert r["strength"] == "STRONG"

    def test_weak_bearish(self):
        # H4+H1 BEARISH BOS_CONFIRMED: -0.3-0.2=-0.5, -0.7<-0.5<-0.3 → WEAK BEARISH
        r = calculate_bias(_mtf("TRANSITION", "BEARISH", "BEARISH",
                                h4_ev="BOS_CONFIRMED", h1_ev="BOS_CONFIRMED"))
        assert r["bias"] == "BEARISH"
        assert r["strength"] == "WEAK"

    def test_neutral_near_zero(self):
        # D1=BEARISH(BOS,-0.5) + H4=BULLISH(BOS,+0.3) + H1=BULLISH(BOS,+0.2) = 0 → NEUTRAL
        r = calculate_bias(_mtf("BEARISH", "BULLISH", "BULLISH",
                                d1_ev="BOS_CONFIRMED", h4_ev="BOS_CONFIRMED", h1_ev="BOS_CONFIRMED"))
        assert r["bias"] == "NEUTRAL"

    def test_neutral_strength_is_weak(self):
        r = calculate_bias(_mtf("TRANSITION", "TRANSITION", "TRANSITION"))
        assert r["strength"] == "WEAK"

    def test_boundary_score_minus05_is_weak_bearish(self):
        # D1=BEARISH(BOS) → score=-0.5. -0.7 < -0.5 < -0.3 → WEAK BEARISH
        r = calculate_bias(_mtf("BEARISH", "TRANSITION", "TRANSITION", d1_ev="BOS_CONFIRMED"))
        assert r["bias"] == "BEARISH"
        assert r["strength"] == "WEAK"

    def test_boundary_score_plus05_is_weak_bullish(self):
        # D1=BULLISH(BOS) → score=0.5. 0.3 < 0.5 < 0.7 → WEAK BULLISH
        r = calculate_bias(_mtf("BULLISH", "TRANSITION", "TRANSITION", d1_ev="BOS_CONFIRMED"))
        assert r["bias"] == "BULLISH"
        assert r["strength"] == "WEAK"


# ---------------------------------------------------------------------------
# 4. Full Alignment → TREND context
# ---------------------------------------------------------------------------

class TestFullyAligned:

    def test_all_bullish_context_trend(self):
        r = calculate_bias(_mtf("BULLISH", "BULLISH", "BULLISH"))
        assert r["context"] == "TREND"

    def test_all_bearish_context_trend(self):
        r = calculate_bias(_mtf("BEARISH", "BEARISH", "BEARISH"))
        assert r["context"] == "TREND"

    def test_all_transition_not_trend(self):
        # All same but TRANSITION → not a clear trend direction
        r = calculate_bias(_mtf("TRANSITION", "TRANSITION", "TRANSITION"))
        assert r["context"] in ("CONSOLIDATION", "REVERSAL")

    def test_two_bullish_one_transition_not_trend(self):
        r = calculate_bias(_mtf("BULLISH", "BULLISH", "TRANSITION"))
        assert r["context"] != "TREND"


# ---------------------------------------------------------------------------
# 5. Mixed Alignment → CONSOLIDATION
# ---------------------------------------------------------------------------

class TestMixedAlignment:

    def test_mixed_no_choch_is_consolidation(self):
        r = calculate_bias(_mtf("BULLISH", "BEARISH", "TRANSITION"))
        assert r["context"] == "CONSOLIDATION"

    def test_bullish_bullish_bearish_is_consolidation(self):
        r = calculate_bias(_mtf("BULLISH", "BULLISH", "BEARISH"))
        assert r["context"] == "CONSOLIDATION"

    def test_bearish_bullish_bullish_is_consolidation(self):
        # Score = 0.0 → NEUTRAL, mixed → CONSOLIDATION
        r = calculate_bias(_mtf("BEARISH", "BULLISH", "BULLISH"))
        assert r["context"] == "CONSOLIDATION"


# ---------------------------------------------------------------------------
# 6. CHOCH Present → REVERSAL context
# ---------------------------------------------------------------------------

class TestCHOCHContext:

    def test_choch_on_h1_gives_reversal(self):
        r = calculate_bias(_mtf("BULLISH", "BULLISH", "BULLISH", h1_ev="CHOCH"))
        assert r["context"] == "REVERSAL"

    def test_choch_on_h4_gives_reversal(self):
        r = calculate_bias(_mtf("BEARISH", "BEARISH", "BEARISH", h4_ev="CHOCH"))
        assert r["context"] == "REVERSAL"

    def test_choch_on_d1_gives_reversal(self):
        r = calculate_bias(_mtf("BULLISH", "BULLISH", "BULLISH", d1_ev="CHOCH"))
        assert r["context"] == "REVERSAL"

    def test_no_choch_aligned_gives_trend_not_reversal(self):
        r = calculate_bias(_mtf("BULLISH", "BULLISH", "BULLISH"))
        assert r["context"] == "TREND"


# ---------------------------------------------------------------------------
# 7. Override Rule
# ---------------------------------------------------------------------------

class TestOverrideRule:

    def test_h1_h4_bullish_plus_choch_forces_bullish(self):
        # D1=BEARISH (no event → 0), H4=BULLISH(BOS → +0.3), H1=BULLISH(CHOCH → 0)
        # score=0.3, abs(0.3)>0.2 → override applies
        r = calculate_bias(_mtf("BEARISH", "BULLISH", "BULLISH",
                                h4_ev="BOS_CONFIRMED", h1_ev="CHOCH"))
        assert r["bias"] == "BULLISH"
        assert r["context"] == "REVERSAL"

    def test_h1_h4_bullish_choch_strength_is_weak(self):
        r = calculate_bias(_mtf("BEARISH", "BULLISH", "BULLISH",
                                h4_ev="BOS_CONFIRMED", h1_ev="CHOCH"))
        assert r["strength"] == "WEAK"

    def test_h1_h4_bearish_plus_choch_forces_bearish(self):
        # D1=BULLISH (no event → 0), H4=BEARISH(BOS → -0.3), H1=BEARISH(CHOCH → 0)
        # score=-0.3, abs(-0.3)>0.2 → override applies
        r = calculate_bias(_mtf("BULLISH", "BEARISH", "BEARISH",
                                h4_ev="BOS_CONFIRMED", h1_ev="CHOCH"))
        assert r["bias"] == "BEARISH"
        assert r["context"] == "REVERSAL"

    def test_h1_h4_bearish_choch_strength_is_weak(self):
        r = calculate_bias(_mtf("BULLISH", "BEARISH", "BEARISH",
                                h4_ev="BOS_CONFIRMED", h1_ev="CHOCH"))
        assert r["strength"] == "WEAK"

    def test_h1_bullish_h4_transition_choch_no_override(self):
        # Override requires BOTH H1 and H4 BULLISH — H4=TRANSITION → no override
        r = calculate_bias(_mtf("BEARISH", "TRANSITION", "BULLISH", h1_ev="CHOCH"))
        # Should still report REVERSAL context but bias follows score
        assert r["context"] == "REVERSAL"
        assert r["bias"] != "BULLISH" or r["strength"] != "STRONG"  # not overridden to STRONG

    def test_choch_on_d1_only_does_not_trigger_h1_h4_override(self):
        # CHOCH on D1 but H1/H4 are not both BULLISH → no forced BULLISH override
        r = calculate_bias(_mtf("BULLISH", "BEARISH", "BEARISH", d1_ev="CHOCH"))
        # context=REVERSAL; bias follows score not override
        assert r["context"] == "REVERSAL"
        assert r["bias"] in ("BEARISH", "NEUTRAL")


# ---------------------------------------------------------------------------
# 8. Missing / Empty Data
# ---------------------------------------------------------------------------

class TestMissingData:

    def test_missing_d1_key_does_not_raise(self):
        data = {"H4": _single("BULLISH", "BOS_CONFIRMED"), "H1": _single("BULLISH", "BOS_CONFIRMED")}
        r = calculate_bias(data)
        # D1 contributes 0; score = 0.3+0.2 = 0.5 → WEAK BULLISH
        assert r["score"] == pytest.approx(0.5)

    def test_missing_h4_key_does_not_raise(self):
        data = {"D1": _single("BULLISH", "BOS_CONFIRMED"), "H1": _single("BULLISH", "BOS_CONFIRMED")}
        r = calculate_bias(data)
        assert r["score"] == pytest.approx(0.7)

    def test_missing_h1_key_does_not_raise(self):
        data = {"D1": _single("BULLISH", "BOS_CONFIRMED"), "H4": _single("BULLISH", "BOS_CONFIRMED")}
        r = calculate_bias(data)
        assert r["score"] == pytest.approx(0.8)

    def test_empty_d1_df_treated_as_transition(self):
        data = {
            "D1": _single("BULLISH", "BOS_CONFIRMED").iloc[0:0],   # empty
            "H4": _single("BULLISH", "BOS_CONFIRMED"),
            "H1": _single("BULLISH", "BOS_CONFIRMED"),
        }
        r = calculate_bias(data)
        assert r["score"] == pytest.approx(0.5)   # 0 + 0.3 + 0.2

    def test_all_missing_returns_neutral(self):
        r = calculate_bias({})
        assert r["bias"] == "NEUTRAL"
        assert r["score"] == pytest.approx(0.0)

    def test_none_value_in_dict_does_not_raise(self):
        data = {"D1": None, "H4": _single("BULLISH", "BOS_CONFIRMED"), "H1": _single("BULLISH", "BOS_CONFIRMED")}
        r = calculate_bias(data)
        assert r["score"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 9. Last Row Only
# ---------------------------------------------------------------------------

class TestLastRowOnly:

    def test_last_row_trend_used_not_first(self):
        # Multi-row df: first rows BEARISH, last row BULLISH with BOS_CONFIRMED
        df = _multi(["BEARISH", "BEARISH", "BULLISH"], ["", "", "BOS_CONFIRMED"])
        r = calculate_bias({"D1": df,
                            "H4": _single("BULLISH", "BOS_CONFIRMED"),
                            "H1": _single("BULLISH", "BOS_CONFIRMED")})
        # Last row (BULLISH, BOS_CONFIRMED): 0.5+0.3+0.2 = 1.0
        assert r["score"] == pytest.approx(1.0)

    def test_last_row_bearish_overrides_early_bullish(self):
        df = _multi(["BULLISH", "BULLISH", "BEARISH"], ["", "", "BOS_CONFIRMED"])
        r = calculate_bias({"D1": df,
                            "H4": _single("BEARISH", "BOS_CONFIRMED"),
                            "H1": _single("BEARISH", "BOS_CONFIRMED")})
        assert r["score"] == pytest.approx(-1.0)

    def test_single_row_df_works(self):
        r = calculate_bias({"D1": _single("BEARISH", "BOS_CONFIRMED"),
                            "H4": _single("BEARISH", "BOS_CONFIRMED"),
                            "H1": _single("BEARISH", "BOS_CONFIRMED")})
        assert r["score"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 10. CHOCH Lookback
# ---------------------------------------------------------------------------

class TestChochLookback:

    def test_choch_within_10_rows_detected(self):
        # 10 rows; CHOCH on row 9 (the 10th row from the end at position -1 in window)
        events = [""] * 9 + ["CHOCH"]
        trends = ["BULLISH"] * 10
        df = _multi(trends, events)
        r = calculate_bias({"D1": _single("BULLISH"), "H4": df, "H1": _single("BULLISH")})
        assert r["context"] == "REVERSAL"

    def test_choch_exactly_at_lookback_boundary(self):
        # 11 rows; CHOCH on row 0 (11 rows ago, outside the 10-row lookback)
        events = ["CHOCH"] + [""] * 10
        trends = ["BULLISH"] * 11
        df = _multi(trends, events)
        r = calculate_bias({"D1": _single("BULLISH"), "H4": df, "H1": _single("BULLISH")})
        # CHOCH is 11 rows from the end → outside lookback → not detected
        assert r["context"] != "REVERSAL"

    def test_choch_within_lookback_row9_from_end(self):
        # 11 rows; CHOCH on row 1 (10 rows from the end — inside lookback)
        events = [""] + ["CHOCH"] + [""] * 9
        trends = ["BULLISH"] * 11
        df = _multi(trends, events)
        r = calculate_bias({"D1": _single("BULLISH"), "H4": df, "H1": _single("BULLISH")})
        assert r["context"] == "REVERSAL"


# ---------------------------------------------------------------------------
# 11. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_input_same_output(self):
        data = _mtf("BULLISH", "BEARISH", "BULLISH")
        r1 = calculate_bias(data)
        r2 = calculate_bias(data)
        assert r1 == r2

    def test_order_independent_of_call_count(self):
        data = _mtf("BEARISH", "BEARISH", "BULLISH", h1_ev="CHOCH")
        results = [calculate_bias(data) for _ in range(5)]
        assert all(r == results[0] for r in results[1:])


# ---------------------------------------------------------------------------
# 12. Input Not Mutated
# ---------------------------------------------------------------------------

class TestNotMutating:

    def test_input_dfs_unchanged(self):
        df_d1 = _single("BULLISH")
        df_h4 = _single("BEARISH")
        df_h1 = _single("BULLISH")
        cols_before = {
            "D1": list(df_d1.columns),
            "H4": list(df_h4.columns),
            "H1": list(df_h1.columns),
        }
        _ = calculate_bias({"D1": df_d1, "H4": df_h4, "H1": df_h1})
        assert list(df_d1.columns) == cols_before["D1"]
        assert list(df_h4.columns) == cols_before["H4"]
        assert list(df_h1.columns) == cols_before["H1"]


# ---------------------------------------------------------------------------
# 13. Event-Based Weighting
# ---------------------------------------------------------------------------

class TestEventBasedWeighting:

    def test_no_event_gives_zero_score_contribution(self):
        # All trends BULLISH but no qualifying events → score=0
        r = calculate_bias(_mtf("BULLISH", "BULLISH", "BULLISH"))
        assert r["score"] == pytest.approx(0.0)

    def test_bos_confirmed_gives_full_multiplier(self):
        # D1=BULLISH, BOS_CONFIRMED → 1.0 × 0.5 × 1.0 = 0.5
        r = calculate_bias(_mtf("BULLISH", "TRANSITION", "TRANSITION", d1_ev="BOS_CONFIRMED"))
        assert r["score"] == pytest.approx(0.5)

    def test_break_gives_reduced_multiplier(self):
        # D1=BULLISH, BREAK → 1.0 × 0.5 × 0.6 = 0.3
        r = calculate_bias(_mtf("BULLISH", "TRANSITION", "TRANSITION", d1_ev="BREAK"))
        assert r["score"] == pytest.approx(0.3)

    def test_choch_gives_zero_score_contribution(self):
        # CHOCH multiplier=0 → no score contribution despite BULLISH trend
        r = calculate_bias(_mtf("BULLISH", "TRANSITION", "TRANSITION", d1_ev="CHOCH"))
        assert r["score"] == pytest.approx(0.0)

    def test_break_on_bearish_gives_reduced_negative_score(self):
        # H4=BEARISH, BREAK → -1.0 × 0.3 × 0.6 = -0.18
        r = calculate_bias(_mtf("TRANSITION", "BEARISH", "TRANSITION", h4_ev="BREAK"))
        assert r["score"] == pytest.approx(-0.18)

    def test_mixed_bos_and_break_accumulate_correctly(self):
        # D1=BULLISH(BOS: +0.5), H4=BEARISH(BREAK: -0.3×0.6=-0.18), H1=no event
        r = calculate_bias(_mtf("BULLISH", "BEARISH", "BULLISH",
                                d1_ev="BOS_CONFIRMED", h4_ev="BREAK"))
        assert r["score"] == pytest.approx(0.5 - 0.18)

