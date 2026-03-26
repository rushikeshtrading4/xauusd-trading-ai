"""Tests for execution/potential_setup.py"""

import pytest

from execution.potential_setup import evaluate_potential_setup


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _signal(**overrides):
    base = {"probability": 75, "risk_reward": 3.0, "bias": "BULLISH"}
    base.update(overrides)
    return base


def _reversal_context(**overrides):
    base = {
        "market_structure": "CHOCH",
        "liquidity":        "SWEEP",
        "ema_alignment":    "BULLISH",
        "rsi":              65.0,
        "vwap_position":    "ABOVE",
        "atr":              12.0,
        "near_order_block": True,
        "near_fvg":         False,
        "displacement":     True,
    }
    base.update(overrides)
    return base


def _continuation_context(**overrides):
    base = {
        "market_structure": "BOS_CONFIRMED",
        "liquidity":        "SWEEP",
        "ema_alignment":    "BULLISH",
        "rsi":              65.0,
        "vwap_position":    "ABOVE",
        "atr":              12.0,
        "near_order_block": True,
        "near_fvg":         False,
        "displacement":     True,
    }
    base.update(overrides)
    return base


def _breakout_context(**overrides):
    base = {
        "market_structure": "BREAK",
        "liquidity":        "SWEEP",
        "ema_alignment":    "BULLISH",
        "rsi":              65.0,
        "vwap_position":    "ABOVE",
        "atr":              15.0,
        "near_order_block": True,
        "near_fvg":         False,
        "displacement":     True,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_returns_dict(self):
        result = evaluate_potential_setup(_signal(), _reversal_context())
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = evaluate_potential_setup(_signal(), _reversal_context())
        for key in ("is_valid_setup", "setup_type", "confidence", "waiting_for", "reason",
                    "setup_score"):
            assert key in result

    def test_is_valid_setup_is_bool(self):
        result = evaluate_potential_setup(_signal(), _reversal_context())
        assert isinstance(result["is_valid_setup"], bool)

    def test_reason_is_string(self):
        result = evaluate_potential_setup(_signal(), _reversal_context())
        assert isinstance(result["reason"], str) and result["reason"]

    def test_rejected_result_has_none_fields(self):
        result = evaluate_potential_setup(_signal(probability=10), _reversal_context())
        assert result["is_valid_setup"] is False
        assert result["setup_type"]  is None
        assert result["confidence"]  is None
        assert result["waiting_for"] is None
        assert result["setup_score"] == 0


# ---------------------------------------------------------------------------
# 2. Hard Filter — Probability
# ---------------------------------------------------------------------------

class TestHardFilterProbability:
    def test_probability_below_60_rejected(self):
        result = evaluate_potential_setup(_signal(probability=59), _reversal_context())
        assert result["is_valid_setup"] is False

    def test_probability_exactly_60_passes_filter(self):
        result = evaluate_potential_setup(_signal(probability=60), _reversal_context())
        assert result["is_valid_setup"] is True

    def test_probability_0_rejected(self):
        result = evaluate_potential_setup(_signal(probability=0), _reversal_context())
        assert result["is_valid_setup"] is False

    def test_rejection_reason_mentions_probability(self):
        result = evaluate_potential_setup(_signal(probability=40), _reversal_context())
        assert "probability" in result["reason"].lower()


# ---------------------------------------------------------------------------
# 3. Hard Filter — Risk/Reward
# ---------------------------------------------------------------------------

class TestHardFilterRiskReward:
    def test_rr_below_2_rejected(self):
        result = evaluate_potential_setup(_signal(risk_reward=1.9), _reversal_context())
        assert result["is_valid_setup"] is False

    def test_rr_exactly_2_passes_filter(self):
        result = evaluate_potential_setup(_signal(risk_reward=2.0), _reversal_context())
        assert result["is_valid_setup"] is True

    def test_rr_0_rejected(self):
        result = evaluate_potential_setup(_signal(risk_reward=0.0), _reversal_context())
        assert result["is_valid_setup"] is False

    def test_rejection_reason_mentions_risk_reward(self):
        result = evaluate_potential_setup(_signal(risk_reward=1.0), _reversal_context())
        assert "risk_reward" in result["reason"].lower()


# ---------------------------------------------------------------------------
# 4. Hard Filter — No Structure + No Liquidity
# ---------------------------------------------------------------------------

class TestHardFilterNoContext:
    def test_no_structure_and_no_liquidity_rejected(self):
        ctx = _reversal_context(market_structure="NONE", liquidity="NONE")
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["is_valid_setup"] is False

    def test_no_structure_but_liquidity_present_not_filtered(self):
        ctx = _reversal_context(market_structure="NONE", liquidity="SWEEP")
        # May be rejected by setup classification, but NOT by this hard filter
        result = evaluate_potential_setup(_signal(), ctx)
        # reason must NOT say "no market structure and no liquidity"
        assert "no market structure and no liquidity" not in result["reason"]

    def test_structure_present_but_no_liquidity_not_filtered_by_this_rule(self):
        ctx = _continuation_context(liquidity="NONE")
        result = evaluate_potential_setup(_signal(), ctx)
        assert "no market structure and no liquidity" not in result["reason"]

    def test_rejection_reason_mentions_structure_and_liquidity(self):
        ctx = _reversal_context(market_structure="NONE", liquidity="NONE")
        result = evaluate_potential_setup(_signal(), ctx)
        assert "structure" in result["reason"].lower()
        assert "liquidity" in result["reason"].lower()


# ---------------------------------------------------------------------------
# 5. Hard Filter — ATR
# ---------------------------------------------------------------------------

class TestHardFilterATR:
    def test_atr_below_3_rejected(self):
        result = evaluate_potential_setup(_signal(), _reversal_context(atr=2.9))
        assert result["is_valid_setup"] is False

    def test_atr_exactly_3_passes_filter(self):
        result = evaluate_potential_setup(_signal(), _reversal_context(atr=3.0))
        assert result["is_valid_setup"] is True

    def test_atr_zero_rejected(self):
        result = evaluate_potential_setup(_signal(), _reversal_context(atr=0.0))
        assert result["is_valid_setup"] is False

    def test_rejection_reason_mentions_atr(self):
        result = evaluate_potential_setup(_signal(), _reversal_context(atr=1.0))
        assert "atr" in result["reason"].lower()


# ---------------------------------------------------------------------------
# 6. Setup Type — REVERSAL
# ---------------------------------------------------------------------------

class TestReversalSetup:
    def test_reversal_detected_with_choch(self):
        result = evaluate_potential_setup(_signal(), _reversal_context(market_structure="CHOCH"))
        assert result["is_valid_setup"] is True
        assert result["setup_type"] == "REVERSAL"

    def test_reversal_not_detected_with_break(self):
        # BREAK is no longer a confirmed structure; the confluence gate rejects
        # any setup where structure is in-progress (not yet BOS/CHOCH)
        result = evaluate_potential_setup(_signal(), _reversal_context(market_structure="BREAK", atr=5.0))
        assert result["is_valid_setup"] is False
        assert result.get("setup_type") is None

    def test_reversal_confidence_is_high_when_prob_gte_80(self):
        # probability >= 80 must yield HIGH regardless of EMA alignment
        sig = _signal(probability=80)
        ctx = _reversal_context(ema_alignment="BEARISH")  # against BULLISH bias
        result = evaluate_potential_setup(sig, ctx)
        assert result["confidence"] == "HIGH"

    def test_reversal_waiting_for_choch_confirmation(self):
        result = evaluate_potential_setup(_signal(), _reversal_context())
        assert result["waiting_for"] == "CHOCH_CONFIRMATION"

    def test_reversal_requires_sweep(self):
        ctx = _reversal_context(liquidity="EQUAL")  # no sweep
        result = evaluate_potential_setup(_signal(), ctx)
        assert result.get("setup_type") != "REVERSAL"

    def test_reversal_requires_near_level(self):
        ctx = _reversal_context(near_order_block=False, near_fvg=False)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result.get("setup_type") != "REVERSAL"

    def test_reversal_with_fvg_instead_of_ob(self):
        ctx = _reversal_context(near_order_block=False, near_fvg=True)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["setup_type"] == "REVERSAL"

    def test_reversal_not_triggered_on_bos_confirmed(self):
        ctx = _reversal_context(market_structure="BOS_CONFIRMED")
        result = evaluate_potential_setup(_signal(), ctx)
        assert result.get("setup_type") != "REVERSAL"


# ---------------------------------------------------------------------------
# 7. Setup Type — CONTINUATION
# ---------------------------------------------------------------------------

class TestContinuationSetup:
    def test_continuation_detected(self):
        result = evaluate_potential_setup(_signal(), _continuation_context())
        assert result["is_valid_setup"] is True
        assert result["setup_type"] == "CONTINUATION"

    def test_continuation_confidence_is_high_when_prob_gte_80(self):
        # probability >= 80 must yield HIGH
        result = evaluate_potential_setup(_signal(probability=80), _continuation_context())
        assert result["confidence"] == "HIGH"

    def test_continuation_waiting_for_pullback_confirmation(self):
        result = evaluate_potential_setup(_signal(), _continuation_context())
        assert result["waiting_for"] == "PULLBACK_CONFIRMATION"

    def test_continuation_requires_bos_confirmed(self):
        ctx = _continuation_context(market_structure="CHOCH")
        result = evaluate_potential_setup(_signal(), ctx)
        assert result.get("setup_type") != "CONTINUATION"

    def test_continuation_requires_ema_aligned_to_bias(self):
        ctx = _continuation_context(ema_alignment="BEARISH")  # opposes BULLISH bias
        result = evaluate_potential_setup(_signal(), ctx)
        assert result.get("setup_type") != "CONTINUATION"

    def test_continuation_requires_near_level(self):
        ctx = _continuation_context(near_order_block=False, near_fvg=False)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result.get("setup_type") != "CONTINUATION"

    def test_continuation_with_fvg_only(self):
        ctx = _continuation_context(near_order_block=False, near_fvg=True)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["setup_type"] == "CONTINUATION"

    def test_continuation_works_for_bearish_bias(self):
        sig = _signal(bias="BEARISH")
        ctx = _continuation_context(ema_alignment="BEARISH")
        result = evaluate_potential_setup(sig, ctx)
        assert result["setup_type"] == "CONTINUATION"


# ---------------------------------------------------------------------------
# 8. Setup Type — BREAKOUT
# ---------------------------------------------------------------------------

class TestBreakoutSetup:
    def test_breakout_blocked_at_confluence(self):
        # BREAK is not a confirmed structure (BOS/CHOCH required)
        # so every breakout context is rejected at the strict confluence gate
        result = evaluate_potential_setup(_signal(), _breakout_context())
        assert result["is_valid_setup"] is False
        assert result.get("setup_type") is None

    def test_breakout_confidence_is_none_when_rejected(self):
        result = evaluate_potential_setup(_signal(), _breakout_context())
        assert result["confidence"] is None

    def test_breakout_waiting_for_is_none_when_rejected(self):
        result = evaluate_potential_setup(_signal(), _breakout_context())
        assert result["waiting_for"] is None

    def test_breakout_requires_break_structure(self):
        ctx = _breakout_context(market_structure="BOS_CONFIRMED")
        result = evaluate_potential_setup(_signal(), ctx)
        assert result.get("setup_type") != "BREAKOUT"

    def test_breakout_requires_atr_above_10(self):
        ctx = _breakout_context(atr=9.9)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result.get("setup_type") != "BREAKOUT"

    def test_breakout_atr_exactly_10_not_triggered(self):
        # condition is atr > 10 (strict), so exactly 10 must not qualify
        ctx = _breakout_context(atr=10.0)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result.get("setup_type") != "BREAKOUT"


# ---------------------------------------------------------------------------
# 9. No valid setup type
# ---------------------------------------------------------------------------

class TestNoValidSetupType:
    def test_passes_filters_but_no_pattern_rejected(self):
        # BOS_CONFIRMED + no near level → no continuation; no sweep → no reversal;
        # no BREAK → no breakout
        ctx = _continuation_context(near_order_block=False, near_fvg=False)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["is_valid_setup"] is False

    def test_reason_explains_no_setup_type(self):
        ctx = _continuation_context(near_order_block=False, near_fvg=False)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["reason"] != ""


# ---------------------------------------------------------------------------
# 10. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_same_output(self):
        sig = _signal()
        ctx = _reversal_context()
        assert evaluate_potential_setup(sig, ctx) == evaluate_potential_setup(sig, ctx)

    def test_bullish_and_bearish_symmetric(self):
        bull = evaluate_potential_setup(
            _signal(bias="BULLISH"),
            _continuation_context(ema_alignment="BULLISH"),
        )
        bear = evaluate_potential_setup(
            _signal(bias="BEARISH"),
            _continuation_context(ema_alignment="BEARISH"),
        )
        assert bull["setup_type"] == bear["setup_type"]
        assert bull["confidence"] == bear["confidence"]
        assert bull["waiting_for"] == bear["waiting_for"]


# ---------------------------------------------------------------------------
# 11. Change 1 — Reversal trend filter
# ---------------------------------------------------------------------------

class TestReversalTrendFilter:
    def test_ema_against_bias_retains_high_initial_confidence(self):
        # ema != bias → initial confidence HIGH before prob scaling
        # Use prob=80 so Change 3 confirms HIGH
        sig = _signal(probability=80)
        ctx = _reversal_context(ema_alignment="BEARISH")  # opposes BULLISH bias
        result = evaluate_potential_setup(sig, ctx)
        assert result["setup_type"] == "REVERSAL"
        assert result["confidence"] == "HIGH"

    def test_ema_with_bias_initial_confidence_medium(self):
        # ema == bias → initial confidence MEDIUM (likely pullback)
        # prob=75 < 80 so Change 3 also gives MEDIUM
        sig = _signal(probability=75)
        ctx = _reversal_context(ema_alignment="BULLISH")  # matches BULLISH bias
        result = evaluate_potential_setup(sig, ctx)
        assert result["setup_type"] == "REVERSAL"
        assert result["confidence"] == "MEDIUM"

    def test_high_prob_overrides_ema_with_bias_downgrade(self):
        # even when ema==bias (would downgrade), prob>=80 upgrades to HIGH
        sig = _signal(probability=80)
        ctx = _reversal_context(ema_alignment="BULLISH")  # matches BULLISH bias
        result = evaluate_potential_setup(sig, ctx)
        assert result["setup_type"] == "REVERSAL"
        assert result["confidence"] == "HIGH"

    def test_bearish_reversal_ema_against_bias(self):
        sig = _signal(bias="BEARISH", probability=80)
        ctx = _reversal_context(ema_alignment="BULLISH")  # against BEARISH bias
        result = evaluate_potential_setup(sig, ctx)
        assert result["setup_type"] == "REVERSAL"
        assert result["confidence"] == "HIGH"


# ---------------------------------------------------------------------------
# 12. Change 2 — Breakout liquidity validation
# ---------------------------------------------------------------------------

class TestBreakoutLiquidityFilter:
    def test_breakout_with_liquidity_equal_passes(self):
        # Under strict confluence a SWEEP is required; EQUAL is not a sweep
        # and fails the has_sweep check → setup is rejected
        result = evaluate_potential_setup(_signal(), _breakout_context(liquidity="EQUAL"))
        assert result["is_valid_setup"] is False
        assert result.get("setup_type") != "BREAKOUT"

    def test_breakout_with_liquidity_sweep_still_blocked(self):
        # Even with a sweep, BREAK is not a confirmed structure — confluence rejects
        result = evaluate_potential_setup(_signal(), _breakout_context(liquidity="SWEEP"))
        assert result["is_valid_setup"] is False
        assert result.get("setup_type") != "BREAKOUT"

    def test_breakout_with_liquidity_none_rejected(self):
        # liquidity == NONE → breakout invalid (no institutional catalyst)
        ctx = _breakout_context(liquidity="NONE")
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["is_valid_setup"] is False
        assert result.get("setup_type") != "BREAKOUT"

    def test_breakout_none_liquidity_reason_not_empty(self):
        ctx = _breakout_context(liquidity="NONE")
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["reason"] != ""


# ---------------------------------------------------------------------------
# 13. Change 3 — Probability-based confidence scaling
# ---------------------------------------------------------------------------

class TestProbabilityConfidenceScaling:
    def test_probability_80_gives_high_reversal(self):
        sig = _signal(probability=80)
        result = evaluate_potential_setup(sig, _reversal_context())
        assert result["confidence"] == "HIGH"

    def test_probability_79_gives_medium_reversal(self):
        sig = _signal(probability=79)
        result = evaluate_potential_setup(sig, _reversal_context())
        assert result["confidence"] == "MEDIUM"

    def test_probability_60_gives_medium_continuation(self):
        sig = _signal(probability=60)
        result = evaluate_potential_setup(sig, _continuation_context())
        assert result["confidence"] == "MEDIUM"

    def test_probability_100_gives_high(self):
        sig = _signal(probability=100)
        result = evaluate_potential_setup(sig, _reversal_context())
        assert result["confidence"] == "HIGH"

    def test_probability_80_gives_high_continuation_2(self):
        # Breakout contexts are no longer valid; use continuation as alternative
        sig = _signal(probability=80)
        result = evaluate_potential_setup(sig, _continuation_context())
        assert result["confidence"] == "HIGH"

    def test_probability_75_gives_medium_continuation_2(self):
        sig = _signal(probability=75)
        result = evaluate_potential_setup(sig, _continuation_context())
        assert result["confidence"] == "MEDIUM"

    def test_probability_80_gives_high_continuation(self):
        sig = _signal(probability=80)
        result = evaluate_potential_setup(sig, _continuation_context())
        assert result["confidence"] == "HIGH"


# ---------------------------------------------------------------------------
# 14. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_missing_signal_keys_no_crash(self):
        result = evaluate_potential_setup({}, {})
        assert isinstance(result, dict)
        assert result["is_valid_setup"] is False

    def test_all_filters_pass_but_minimal_context(self):
        sig = _signal(probability=60, risk_reward=2.0)
        ctx = _reversal_context(atr=3.0)
        result = evaluate_potential_setup(sig, ctx)
        assert isinstance(result["is_valid_setup"], bool)

    def test_near_ob_and_near_fvg_both_true(self):
        ctx = _reversal_context(near_order_block=True, near_fvg=True)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["setup_type"] == "REVERSAL"


# ---------------------------------------------------------------------------
# 15. Strict Confluence Gate
# ---------------------------------------------------------------------------

class TestConfluentCheck:

    def test_all_confluence_elements_present_passes(self):
        result = evaluate_potential_setup(_signal(), _reversal_context())
        assert result["is_valid_setup"] is True

    def test_no_sweep_rejected(self):
        ctx = _reversal_context(liquidity="EQUAL")
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["is_valid_setup"] is False
        assert "liquidity sweep" in result["reason"]

    def test_no_displacement_rejected(self):
        ctx = _reversal_context(displacement=False)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["is_valid_setup"] is False
        assert "displacement" in result["reason"]

    def test_no_structure_rejected(self):
        ctx = _reversal_context(market_structure="NONE")
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["is_valid_setup"] is False
        assert "valid structure" in result["reason"]

    def test_no_zone_rejected(self):
        ctx = _reversal_context(near_order_block=False, near_fvg=False)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["is_valid_setup"] is False
        assert "institutional zone" in result["reason"]

    def test_multiple_missing_elements_all_listed_in_reason(self):
        # No sweep, no displacement, no zone
        ctx = _reversal_context(liquidity="EQUAL", displacement=False,
                                near_order_block=False, near_fvg=False)
        result = evaluate_potential_setup(_signal(), ctx)
        assert "liquidity sweep"    in result["reason"]
        assert "displacement"       in result["reason"]
        assert "institutional zone" in result["reason"]

    def test_confluence_gate_fires_before_setup_classification(self):
        # Even though the context would otherwise classify as REVERSAL,
        # the missing displacement stops the flow before classification.
        ctx = _reversal_context(displacement=False)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["setup_type"] is None

    def test_continuation_requires_sweep_under_strict_confluence(self):
        ctx = _continuation_context(liquidity="EQUAL")
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["is_valid_setup"] is False
        assert "liquidity sweep" in result["reason"]


# ---------------------------------------------------------------------------
# 16. Quality Scoring
# ---------------------------------------------------------------------------

class TestQualityScoring:

    def test_valid_setup_has_setup_score_key(self):
        result = evaluate_potential_setup(_signal(), _reversal_context())
        assert "setup_score" in result

    def test_reversal_ob_only_score_is_4(self):
        # sweep(1) + displacement(1) + structure(1) + ob(1) + no-fvg(0) = 4
        ctx = _reversal_context(near_order_block=True, near_fvg=False)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["setup_score"] == 4

    def test_reversal_ob_and_fvg_score_is_5(self):
        # all five elements present → score = 5
        ctx = _reversal_context(near_order_block=True, near_fvg=True)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["setup_score"] == 5

    def test_continuation_ob_only_score_is_4(self):
        ctx = _continuation_context(near_order_block=True, near_fvg=False)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["setup_score"] == 4

    def test_min_score_4_passes_gate(self):
        ctx = _reversal_context(near_order_block=True, near_fvg=False)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["is_valid_setup"] is True
        assert result["setup_score"] >= 4

    def test_rejected_by_confluence_has_score_zero(self):
        # Missing displacement → rejected at confluence before scoring
        ctx = _reversal_context(displacement=False)
        result = evaluate_potential_setup(_signal(), ctx)
        assert result["setup_score"] == 0

    def test_rejected_by_probability_has_score_zero(self):
        result = evaluate_potential_setup(_signal(probability=30), _reversal_context())
        assert result["setup_score"] == 0

