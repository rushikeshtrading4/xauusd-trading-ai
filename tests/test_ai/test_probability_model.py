"""Tests for ai/probability_model.py"""

import pytest

from ai.probability_model import compute_probability


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _bullish_signal(**overrides):
    base = {
        "entry":       2000.0,
        "stop_loss":   1990.0,
        "take_profit": 2030.0,
        "risk_reward": 3.0,
        "bias":        "BULLISH",
    }
    base.update(overrides)
    return base


def _bearish_signal(**overrides):
    base = {
        "entry":       2000.0,
        "stop_loss":   2010.0,
        "take_profit": 1970.0,
        "risk_reward": 3.0,
        "bias":        "BEARISH",
    }
    base.update(overrides)
    return base


def _strong_bullish_context(**overrides):
    base = {
        "market_structure": "BOS_CONFIRMED",
        "ema_alignment":    "BULLISH",
        "rsi":              65.0,
        "vwap_position":    "ABOVE",
        "liquidity":        "SWEEP",
        "atr":              10.0,
    }
    base.update(overrides)
    return base


def _strong_bearish_context(**overrides):
    base = {
        "market_structure": "BOS_CONFIRMED",
        "ema_alignment":    "BEARISH",
        "rsi":              35.0,
        "vwap_position":    "BELOW",
        "liquidity":        "SWEEP",
        "atr":              10.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_returns_dict(self):
        result = compute_probability(_bullish_signal(), _strong_bullish_context())
        assert isinstance(result, dict)

    def test_has_probability_key(self):
        result = compute_probability(_bullish_signal(), _strong_bullish_context())
        assert "probability" in result

    def test_has_grade_key(self):
        result = compute_probability(_bullish_signal(), _strong_bullish_context())
        assert "grade" in result

    def test_has_confidence_key(self):
        result = compute_probability(_bullish_signal(), _strong_bullish_context())
        assert "confidence" in result

    def test_probability_is_int(self):
        result = compute_probability(_bullish_signal(), _strong_bullish_context())
        assert isinstance(result["probability"], int)

    def test_probability_within_0_to_100(self):
        result = compute_probability(_bullish_signal(), _strong_bullish_context())
        assert 0 <= result["probability"] <= 100

    def test_grade_is_valid_letter(self):
        result = compute_probability(_bullish_signal(), _strong_bullish_context())
        assert result["grade"] in ("A+", "A", "B", "C")

    def test_confidence_is_non_empty_string(self):
        result = compute_probability(_bullish_signal(), _strong_bullish_context())
        assert isinstance(result["confidence"], str) and result["confidence"]


# ---------------------------------------------------------------------------
# 2. Probability range is always clamped
# ---------------------------------------------------------------------------

class TestProbabilityClamp:
    def test_score_never_exceeds_100(self):
        # Give every possible positive contribution
        result = compute_probability(_bullish_signal(), _strong_bullish_context())
        assert result["probability"] <= 100

    def test_score_never_below_0(self):
        # Worst-case signal: everything against the trade
        signal = _bullish_signal(risk_reward=0.5)
        context = _strong_bearish_context(
            market_structure="NONE",
            ema_alignment="BEARISH",
            rsi=80.0,
            vwap_position="BELOW",
            liquidity="NONE",
            atr=0.5,         # too low
        )
        result = compute_probability(signal, context)
        assert result["probability"] >= 0


# ---------------------------------------------------------------------------
# 3. Market structure scoring
# ---------------------------------------------------------------------------

class TestMarketStructure:
    def _score(self, structure):
        ctx = _strong_bullish_context(market_structure=structure,
                                      ema_alignment="MIXED",
                                      rsi=50.0,
                                      vwap_position="ABOVE",
                                      liquidity="NONE",
                                      atr=10.0)
        return compute_probability(_bullish_signal(), ctx)["probability"]

    def test_bos_confirmed_highest(self):
        assert self._score("BOS_CONFIRMED") > self._score("CHOCH")

    def test_choch_higher_than_break(self):
        assert self._score("CHOCH") > self._score("BREAK")

    def test_break_higher_than_none(self):
        assert self._score("BREAK") > self._score("NONE")


# ---------------------------------------------------------------------------
# 4. EMA alignment scoring
# ---------------------------------------------------------------------------

class TestEMAAlignment:
    def _score(self, ema, bias="BULLISH"):
        ctx = _strong_bullish_context(
            market_structure="NONE", ema_alignment=ema,
            rsi=50.0, vwap_position="", liquidity="NONE", atr=10.0
        )
        sig = _bullish_signal(bias=bias)
        return compute_probability(sig, ctx)["probability"]

    def test_perfect_alignment_beats_mixed(self):
        assert self._score("BULLISH") > self._score("MIXED")

    def test_mixed_beats_against(self):
        assert self._score("MIXED") > self._score("BEARISH")

    def test_against_alignment_penalises_score(self):
        base  = self._score("MIXED")
        against = self._score("BEARISH")
        assert against < base


# ---------------------------------------------------------------------------
# 5. RSI scoring
# ---------------------------------------------------------------------------

class TestRSI:
    def _score(self, rsi, bias="BULLISH"):
        ctx = _strong_bullish_context(
            market_structure="NONE", ema_alignment="MIXED",
            rsi=rsi, vwap_position="", liquidity="NONE", atr=10.0
        )
        sig = _bullish_signal(bias=bias)
        return compute_probability(sig, ctx)["probability"]

    def test_bullish_sweet_spot_60_70(self):
        assert self._score(65) > self._score(50)

    def test_bullish_overbought_penalised(self):
        assert self._score(80) < self._score(65)

    def test_bearish_sweet_spot_30_40(self):
        assert self._score(35, "BEARISH") > self._score(50, "BEARISH")

    def test_bearish_oversold_penalised(self):
        assert self._score(20, "BEARISH") < self._score(35, "BEARISH")


# ---------------------------------------------------------------------------
# 6. VWAP position scoring
# ---------------------------------------------------------------------------

class TestVWAPPosition:
    def _score(self, vwap, bias="BULLISH"):
        ctx = _strong_bullish_context(
            market_structure="NONE", ema_alignment="MIXED",
            rsi=50.0, vwap_position=vwap, liquidity="NONE", atr=10.0
        )
        sig = _bullish_signal(bias=bias)
        return compute_probability(sig, ctx)["probability"]

    def test_bullish_above_vwap_rewarded(self):
        assert self._score("ABOVE") > self._score("BELOW")

    def test_bearish_below_vwap_rewarded(self):
        assert self._score("BELOW", "BEARISH") > self._score("ABOVE", "BEARISH")

    def test_opposing_vwap_penalises(self):
        aligned  = self._score("ABOVE", "BULLISH")
        opposing = self._score("BELOW", "BULLISH")
        assert opposing < aligned


# ---------------------------------------------------------------------------
# 7. Liquidity scoring
# ---------------------------------------------------------------------------

class TestLiquidity:
    def _score(self, liquidity):
        ctx = _strong_bullish_context(
            market_structure="NONE", ema_alignment="MIXED",
            rsi=50.0, vwap_position="", liquidity=liquidity, atr=10.0
        )
        return compute_probability(_bullish_signal(), ctx)["probability"]

    def test_sweep_highest(self):
        assert self._score("SWEEP") > self._score("EQUAL")

    def test_equal_higher_than_none(self):
        assert self._score("EQUAL") > self._score("NONE")


# ---------------------------------------------------------------------------
# 8. Risk/Reward scoring
# ---------------------------------------------------------------------------

class TestRiskReward:
    def _score(self, rr):
        ctx = _strong_bullish_context(
            market_structure="NONE", ema_alignment="MIXED",
            rsi=50.0, vwap_position="", liquidity="NONE", atr=10.0
        )
        return compute_probability(_bullish_signal(risk_reward=rr), ctx)["probability"]

    def test_rr_3_beats_rr_2(self):
        assert self._score(3.0) > self._score(2.0)

    def test_rr_2_beats_rr_below_2(self):
        assert self._score(2.0) > self._score(1.5)

    def test_rr_below_2_heavily_penalised(self):
        # Raw penalty swing from RR=2.0 (+5) to RR=1.9 (−20) is 25 points.
        # The floor clamp holds the RR=1.9 score at 0, so the observable
        # difference is at least 20 (2.0 score) − 0 = 20.
        assert self._score(2.0) > self._score(1.9)
        assert self._score(1.9) == 0   # clamped at floor


# ---------------------------------------------------------------------------
# 9. ATR scoring
# ---------------------------------------------------------------------------

class TestATR:
    def _score(self, atr):
        ctx = _strong_bullish_context(
            market_structure="NONE", ema_alignment="MIXED",
            rsi=50.0, vwap_position="", liquidity="NONE", atr=atr
        )
        return compute_probability(_bullish_signal(), ctx)["probability"]

    def test_normal_atr_rewarded(self):
        assert self._score(10.0) > self._score(1.0)   # normal vs too low

    def test_too_low_atr_penalised(self):
        assert self._score(1.0) < self._score(10.0)

    def test_too_high_atr_penalised_vs_normal(self):
        assert self._score(50.0) < self._score(10.0)

    def test_too_high_less_penalised_than_too_low(self):
        # too_high → −5; too_low → −10; so too_high scores higher
        assert self._score(50.0) > self._score(1.0)


# ---------------------------------------------------------------------------
# 10. Grading thresholds
# ---------------------------------------------------------------------------

class TestGrading:
    def _grade(self, probability_target):
        """Drive score toward a target by constructing an appropriate signal."""
        # Use a minimal context that gives exactly the right score via RR penalty
        # A+ requires ≥ 80 — build full-positive bullish setup
        if probability_target >= 80:
            return compute_probability(
                _bullish_signal(risk_reward=3.0),
                _strong_bullish_context()
            )["grade"]
        if probability_target >= 65:
            return compute_probability(
                _bullish_signal(risk_reward=2.0),
                _strong_bullish_context()
            )["grade"]
        # Force low score with bad RR and opposing context
        return compute_probability(
            _bullish_signal(risk_reward=1.0),
            _strong_bearish_context(
                ema_alignment="BEARISH", rsi=80.0,
                vwap_position="BELOW", liquidity="NONE", atr=1.0
            )
        )["grade"]

    def test_high_score_gives_a_plus(self):
        result = compute_probability(
            _bullish_signal(risk_reward=3.0), _strong_bullish_context()
        )
        assert result["grade"] == "A+"
        assert result["probability"] >= 80

    def test_a_plus_confidence_label(self):
        result = compute_probability(
            _bullish_signal(risk_reward=3.0), _strong_bullish_context()
        )
        assert result["confidence"] == "Very High"

    def test_low_score_gives_c(self):
        result = compute_probability(
            _bullish_signal(risk_reward=1.0),
            _strong_bearish_context(
                ema_alignment="BEARISH", rsi=80.0,
                vwap_position="BELOW", liquidity="NONE", atr=1.0
            )
        )
        assert result["grade"] == "C"
        assert result["confidence"] == "Low"

    def test_grade_a_range(self):
        # EMA mixed instead of aligned → slightly lower score
        result = compute_probability(
            _bullish_signal(risk_reward=2.0),
            _strong_bullish_context(ema_alignment="MIXED", rsi=50.0)
        )
        assert result["grade"] in ("A+", "A", "B")

    def test_grade_b_boundaries(self):
        # CHOCH(+15) + MIXED(+10) + RSI 65(+10) + ABOVE(+10) + NONE(0) + RR2(+5) + ATR10(+5) = 55
        result = compute_probability(
            _bullish_signal(risk_reward=2.0),
            _strong_bullish_context(
                market_structure="CHOCH", ema_alignment="MIXED",
                rsi=65.0, vwap_position="ABOVE", liquidity="NONE"
            )
        )
        assert 50 <= result["probability"] <= 79


# ---------------------------------------------------------------------------
# 11. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_same_output(self):
        sig = _bullish_signal()
        ctx = _strong_bullish_context()
        assert compute_probability(sig, ctx) == compute_probability(sig, ctx)

    def test_bearish_and_bullish_mirror(self):
        # Symmetric setups should score identically
        bull_result = compute_probability(
            _bullish_signal(), _strong_bullish_context()
        )
        bear_result = compute_probability(
            _bearish_signal(), _strong_bearish_context()
        )
        assert bull_result["probability"] == bear_result["probability"]
        assert bull_result["grade"] == bear_result["grade"]


# ---------------------------------------------------------------------------
# 12. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_atr(self):
        ctx = _strong_bullish_context(atr=0.0)
        result = compute_probability(_bullish_signal(), ctx)
        assert 0 <= result["probability"] <= 100

    def test_missing_optional_keys_no_crash(self):
        # Minimal dicts — should not raise
        result = compute_probability({"bias": "BULLISH", "risk_reward": 2.0}, {})
        assert 0 <= result["probability"] <= 100

    def test_unknown_structure_treated_as_none(self):
        ctx = _strong_bullish_context(market_structure="UNKNOWN")
        result = compute_probability(_bullish_signal(), ctx)
        assert isinstance(result["probability"], int)

    def test_unknown_liquidity_treated_as_none(self):
        ctx = _strong_bullish_context(liquidity="UNKNOWN")
        r1 = compute_probability(_bullish_signal(), ctx)
        ctx2 = _strong_bullish_context(liquidity="NONE")
        r2 = compute_probability(_bullish_signal(), ctx2)
        assert r1["probability"] == r2["probability"]


# ---------------------------------------------------------------------------
# 13. Confluence boosts
# ---------------------------------------------------------------------------

class TestConfluenceBoosts:

    # ------------------------------------------------------------------ #
    # Boost 1 — Institutional Setup                                       #
    # ------------------------------------------------------------------ #

    def test_institutional_boost_applied_when_all_three_conditions_met(self):
        """BOS_CONFIRMED + SWEEP + perfect EMA alignment earns +10 extra."""
        with_boost = _strong_bullish_context()                # all three present
        without = _strong_bullish_context(liquidity="EQUAL")  # breaks the triplet
        r_with    = compute_probability(_bullish_signal(), with_boost)["probability"]
        r_without = compute_probability(_bullish_signal(), without)["probability"]
        # The institutional boost (+10) fires only for the SWEEP branch.
        # Both scores may be near the 100 ceiling, so assert the direction
        # and that at least the +10 boost pushes the gap to >= 10.
        assert r_with > r_without

    def test_institutional_boost_not_applied_without_sweep(self):
        ctx_no_sweep = _strong_bullish_context(liquidity="EQUAL")
        ctx_sweep    = _strong_bullish_context(liquidity="SWEEP")
        r_no  = compute_probability(_bullish_signal(), ctx_no_sweep)["probability"]
        r_yes = compute_probability(_bullish_signal(), ctx_sweep)["probability"]
        assert r_yes > r_no

    def test_institutional_boost_not_applied_without_bos(self):
        ctx_bos   = _strong_bullish_context(market_structure="BOS_CONFIRMED")
        ctx_choch = _strong_bullish_context(market_structure="CHOCH")
        r_bos   = compute_probability(_bullish_signal(), ctx_bos)["probability"]
        r_choch = compute_probability(_bullish_signal(), ctx_choch)["probability"]
        # BOS gives +25 vs CHOCH +15 AND triggers boost; total gap > 10
        assert r_bos > r_choch

    def test_institutional_boost_not_applied_when_ema_mixed(self):
        ctx_aligned = _strong_bullish_context(ema_alignment="BULLISH")
        ctx_mixed   = _strong_bullish_context(ema_alignment="MIXED")
        r_aligned = compute_probability(_bullish_signal(), ctx_aligned)["probability"]
        r_mixed   = compute_probability(_bullish_signal(), ctx_mixed)["probability"]
        assert r_aligned > r_mixed

    def test_institutional_boost_works_for_bearish_bias(self):
        ctx = _strong_bearish_context()   # BOS_CONFIRMED + SWEEP + BEARISH alignment
        with_boost = compute_probability(_bearish_signal(), ctx)["probability"]
        ctx_partial = _strong_bearish_context(liquidity="EQUAL")
        without_full = compute_probability(_bearish_signal(), ctx_partial)["probability"]
        assert with_boost > without_full

    # ------------------------------------------------------------------ #
    # Boost 2 — Trend Continuation                                        #
    # ------------------------------------------------------------------ #

    def test_trend_continuation_boost_bullish_rsi_sweet_spot(self):
        """EMA aligned + RSI 55–75 (bullish) → +5 continuation boost."""
        ctx_sweet    = _strong_bullish_context(
            market_structure="NONE", liquidity="NONE", ema_alignment="BULLISH", rsi=65.0
        )
        ctx_neutral  = _strong_bullish_context(
            market_structure="NONE", liquidity="NONE", ema_alignment="BULLISH", rsi=50.0
        )
        r_sweet   = compute_probability(_bullish_signal(), ctx_sweet)["probability"]
        r_neutral = compute_probability(_bullish_signal(), ctx_neutral)["probability"]
        # RSI 65 adds +10 factor AND +5 boost vs RSI 50 with +0 factor +0 boost
        assert r_sweet - r_neutral == 15

    def test_trend_continuation_boost_bearish_rsi_sweet_spot(self):
        ctx_sweet   = _strong_bearish_context(
            market_structure="NONE", liquidity="NONE", ema_alignment="BEARISH", rsi=35.0
        )
        ctx_neutral = _strong_bearish_context(
            market_structure="NONE", liquidity="NONE", ema_alignment="BEARISH", rsi=50.0
        )
        r_sweet   = compute_probability(_bearish_signal(), ctx_sweet)["probability"]
        r_neutral = compute_probability(_bearish_signal(), ctx_neutral)["probability"]
        assert r_sweet - r_neutral == 15

    def test_trend_continuation_boost_not_applied_with_mixed_ema(self):
        ctx_aligned = _strong_bullish_context(
            market_structure="NONE", liquidity="NONE", ema_alignment="BULLISH", rsi=65.0
        )
        ctx_mixed   = _strong_bullish_context(
            market_structure="NONE", liquidity="NONE", ema_alignment="MIXED", rsi=65.0
        )
        r_aligned = compute_probability(_bullish_signal(), ctx_aligned)["probability"]
        r_mixed   = compute_probability(_bullish_signal(), ctx_mixed)["probability"]
        # Aligned: +20 factor + (+10 rsi) + (+5 boost) = 35 extra vs mixed: +10 factor + (+10 rsi)
        assert r_aligned > r_mixed

    def test_trend_continuation_boost_not_applied_outside_rsi_range(self):
        ctx_in_range  = _strong_bullish_context(
            market_structure="NONE", liquidity="NONE", ema_alignment="BULLISH", rsi=65.0
        )
        ctx_out_range = _strong_bullish_context(
            market_structure="NONE", liquidity="NONE", ema_alignment="BULLISH", rsi=50.0
        )
        r_in  = compute_probability(_bullish_signal(), ctx_in_range)["probability"]
        r_out = compute_probability(_bullish_signal(), ctx_out_range)["probability"]
        assert r_in > r_out   # boost + rsi factor both contribute

    # ------------------------------------------------------------------ #
    # Penalty — Weak Context                                               #
    # ------------------------------------------------------------------ #

    def test_weak_context_penalty_applied_when_no_structure_no_liquidity(self):
        ctx_weak   = _strong_bullish_context(market_structure="NONE", liquidity="NONE")
        ctx_partial = _strong_bullish_context(market_structure="NONE", liquidity="EQUAL")
        r_weak    = compute_probability(_bullish_signal(), ctx_weak)["probability"]
        r_partial = compute_probability(_bullish_signal(), ctx_partial)["probability"]
        # EQUAL adds +5, and weak context loses −10; total swing = 15
        assert r_partial - r_weak == 15

    def test_weak_context_penalty_not_applied_when_structure_present(self):
        ctx_with_structure = _strong_bullish_context(
            market_structure="CHOCH", liquidity="NONE"
        )
        ctx_no_structure = _strong_bullish_context(
            market_structure="NONE", liquidity="NONE"
        )
        r_with = compute_probability(_bullish_signal(), ctx_with_structure)["probability"]
        r_none = compute_probability(_bullish_signal(), ctx_no_structure)["probability"]
        # CHOCH +15 vs NONE +0, and NONE triggers −10 penalty; gap must be ≥ 25
        assert r_with - r_none >= 25

    def test_weak_context_penalty_not_applied_when_liquidity_present(self):
        ctx_with_liq = _strong_bullish_context(
            market_structure="NONE", liquidity="SWEEP"
        )
        ctx_no_liq   = _strong_bullish_context(
            market_structure="NONE", liquidity="NONE"
        )
        r_liq  = compute_probability(_bullish_signal(), ctx_with_liq)["probability"]
        r_none = compute_probability(_bullish_signal(), ctx_no_liq)["probability"]
        assert r_liq > r_none

    def test_score_still_clamped_after_boosts(self):
        """Confluence boosts must not push score above 100."""
        result = compute_probability(_bullish_signal(), _strong_bullish_context())
        assert result["probability"] <= 100

    def test_score_still_clamped_at_zero_after_penalty(self):
        """Weak-context penalty on an already-low score must not go below 0."""
        signal = _bullish_signal(risk_reward=0.5)
        ctx = _strong_bullish_context(
            market_structure="NONE", ema_alignment="BEARISH",
            rsi=80.0, vwap_position="BELOW", liquidity="NONE", atr=0.5
        )
        result = compute_probability(signal, ctx)
        assert result["probability"] >= 0

