"""Focused tests for Factor 8 (Macro Sentiment) in ai/probability_model.py.

Score reference for test setups used here
------------------------------------------

CHOCH context (market_structure="CHOCH", ema_alignment="BULLISH", rsi=65.0,
               vwap_position="ABOVE", liquidity="SWEEP", atr=10.0)
with BULLISH signal (risk_reward=2.5), no macro factor:

  F1 CHOCH          +15
  F2 EMA aligned    +20
  F3 RSI 65         +10
  F4 VWAP above     +10
  F5 SWEEP          +15
  F6 RR 2.5          +5
  F7 ATR 10          +5
  Boost2 (ema+rsi)   +5
  Base total         85  → Grade A+

BREAK context (market_structure="BREAK", others as CHOCH) with BULLISH:
  F1 BREAK           +5   (vs +15 for CHOCH)
  All others same, no Boost2 hit change
  Base total         75  → Grade A

BOS+NONE context (market_structure="BOS_CONFIRMED", liquidity="NONE"):
  F1 BOS            +25
  F2 EMA            +20
  F3 RSI 65         +10
  F4 VWAP           +10
  F5 NONE            +0
  F6 RR 2.5          +5
  F7 ATR 10          +5
  Boost2 (ema+rsi)   +5
  Base total         80  → Grade A+

Full BOS context (BOS+SWEEP, all factors), pre-clamp = 105, clamped = 100.

Weak context (structure=NONE, ema opposing, rsi>75, vwap opposing,
              liquidity=NONE, atr=10, RR=1.5) → pre-clamp ≤ 0, clamped = 0.
"""

from __future__ import annotations

import pytest

from ai.probability_model import compute_probability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_signal(bias: str = "BULLISH") -> dict:
    return {"bias": bias, "risk_reward": 2.5}


def _base_context(**overrides) -> dict:
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


def _score(signal: dict, ctx: dict) -> int:
    return compute_probability(signal, ctx)["probability"]


def _grade(signal: dict, ctx: dict) -> str:
    return compute_probability(signal, ctx)["grade"]


# ---------------------------------------------------------------------------
# TestMacroFactorAbsent
# ---------------------------------------------------------------------------


class TestMacroFactorAbsent:
    def test_score_unchanged_without_macro_key(self):
        # Full BOS context, no macro → pre-clamp 105, clamped to 100.
        ctx = _base_context()
        assert _score(_base_signal(), ctx) == 100

    def test_no_exception_raised(self):
        ctx = _base_context()
        result = compute_probability(_base_signal(), ctx)
        assert isinstance(result, dict)

    def test_absent_news_blackout_no_penalty(self):
        # Score must be 100 — same as with news_blackout=False explicitly.
        ctx_absent  = _base_context()
        ctx_explicit = _base_context(news_blackout=False)
        assert _score(_base_signal(), ctx_absent) == _score(_base_signal(), ctx_explicit)


# ---------------------------------------------------------------------------
# TestMacroFactorAligned
# ---------------------------------------------------------------------------


class TestMacroFactorAligned:
    def test_bullish_macro_adds_15_to_bullish_trade(self):
        # CHOCH base = 85; with macro aligned = 100.
        ctx_no_macro = _base_context(market_structure="CHOCH")
        ctx_aligned  = _base_context(market_structure="CHOCH",
                                     macro_sentiment="BULLISH")
        base = _score(_base_signal("BULLISH"), ctx_no_macro)
        with_macro = _score(_base_signal("BULLISH"), ctx_aligned)
        assert with_macro - base == 15

    def test_bearish_macro_adds_15_to_bearish_trade(self):
        # Mirror: BEARISH bias + BEARISH macro, CHOCH base = 85 → 100.
        ctx_no_macro = _base_context(
            market_structure="CHOCH",
            ema_alignment="BEARISH",
            rsi=35.0,
            vwap_position="BELOW",
        )
        ctx_aligned = _base_context(
            market_structure="CHOCH",
            ema_alignment="BEARISH",
            rsi=35.0,
            vwap_position="BELOW",
            macro_sentiment="BEARISH",
        )
        base = _score(_base_signal("BEARISH"), ctx_no_macro)
        with_macro = _score(_base_signal("BEARISH"), ctx_aligned)
        assert with_macro - base == 15

    def test_grade_improves_with_aligned_macro(self):
        # BREAK base = 75, Grade A; with aligned macro = 90, Grade A+.
        sig = _base_signal("BULLISH")
        ctx_no_macro = _base_context(market_structure="BREAK")
        ctx_aligned  = _base_context(market_structure="BREAK",
                                     macro_sentiment="BULLISH")
        assert _grade(sig, ctx_no_macro) == "A"
        assert _grade(sig, ctx_aligned) == "A+"


# ---------------------------------------------------------------------------
# TestMacroFactorNeutral
# ---------------------------------------------------------------------------


class TestMacroFactorNeutral:
    def test_neutral_macro_same_score_as_absent(self):
        ctx_absent  = _base_context(market_structure="CHOCH")
        ctx_neutral = _base_context(market_structure="CHOCH",
                                    macro_sentiment="NEUTRAL")
        assert (_score(_base_signal(), ctx_absent)
                == _score(_base_signal(), ctx_neutral))

    def test_neutral_macro_grade_unchanged(self):
        ctx_absent  = _base_context(market_structure="CHOCH")
        ctx_neutral = _base_context(market_structure="CHOCH",
                                    macro_sentiment="NEUTRAL")
        assert (_grade(_base_signal(), ctx_absent)
                == _grade(_base_signal(), ctx_neutral))


# ---------------------------------------------------------------------------
# TestMacroFactorOpposing
# ---------------------------------------------------------------------------


class TestMacroFactorOpposing:
    def test_opposing_macro_subtracts_10(self):
        # CHOCH base = 85; opposing macro → 75. Difference = 10.
        ctx_no_macro = _base_context(market_structure="CHOCH")
        ctx_opposing = _base_context(market_structure="CHOCH",
                                     macro_sentiment="BEARISH")
        base = _score(_base_signal("BULLISH"), ctx_no_macro)
        with_opposing = _score(_base_signal("BULLISH"), ctx_opposing)
        assert base - with_opposing == 10

    def test_bullish_trade_bearish_macro_minus_10(self):
        ctx = _base_context(market_structure="CHOCH", macro_sentiment="BEARISH")
        assert _score(_base_signal("BULLISH"), ctx) == 75

    def test_bearish_trade_bullish_macro_minus_10(self):
        ctx = _base_context(
            market_structure="CHOCH",
            ema_alignment="BEARISH",
            rsi=35.0,
            vwap_position="BELOW",
            macro_sentiment="BULLISH",
        )
        assert _score(_base_signal("BEARISH"), ctx) == 75


# ---------------------------------------------------------------------------
# TestNewsBlackout
# ---------------------------------------------------------------------------


class TestNewsBlackout:
    def test_news_blackout_applies_20_penalty_regardless_of_macro(self):
        # CHOCH base = 85. With blackout → 65 regardless of macro_sentiment.
        ctx_bull = _base_context(market_structure="CHOCH",
                                 news_blackout=True,
                                 macro_sentiment="BULLISH")
        ctx_bear = _base_context(market_structure="CHOCH",
                                 news_blackout=True,
                                 macro_sentiment="BEARISH")
        assert _score(_base_signal("BULLISH"), ctx_bull) == 65
        assert _score(_base_signal("BULLISH"), ctx_bear) == 65

    def test_ap_setup_drops_below_70_with_blackout(self):
        # BOS+NONE base = 80, Grade A+. With blackout → 60 < 70.
        ctx = _base_context(liquidity="NONE", news_blackout=True)
        result = compute_probability(_base_signal("BULLISH"), ctx)
        assert result["probability"] < 70
        assert result["probability"] == 60

    def test_blackout_overrides_aligned_macro(self):
        # CHOCH base = 85. Both blackout AND aligned macro: blackout wins → 65,
        # NOT 85 + 15 = 100.
        ctx_blackout_only = _base_context(market_structure="CHOCH",
                                          news_blackout=True)
        ctx_both = _base_context(market_structure="CHOCH",
                                 news_blackout=True,
                                 macro_sentiment="BULLISH")
        assert _score(_base_signal("BULLISH"), ctx_blackout_only) == 65
        assert _score(_base_signal("BULLISH"), ctx_both) == 65

    def test_false_blackout_with_aligned_macro_gives_plus_15(self):
        # news_blackout=False is same as absent: only the aligned +15 applies.
        ctx_no_blackout = _base_context(market_structure="CHOCH",
                                        news_blackout=False,
                                        macro_sentiment="BULLISH")
        assert _score(_base_signal("BULLISH"), ctx_no_blackout) == 100


# ---------------------------------------------------------------------------
# TestMacroScoreBoundaries
# ---------------------------------------------------------------------------


class TestMacroScoreBoundaries:
    def test_aligned_macro_clamped_to_100(self):
        # Full BOS context pre-clamp = 105; +15 → 120; clamped to 100.
        ctx = _base_context(macro_sentiment="BULLISH")
        assert _score(_base_signal("BULLISH"), ctx) == 100

    def test_opposing_macro_on_weak_setup_clamped_to_0(self):
        # Weak setup pre-clamp ≤ 0; opposing macro subtracts further → floor 0.
        weak_ctx = {
            "market_structure": "NONE",
            "ema_alignment":    "BEARISH",
            "rsi":              80.0,
            "vwap_position":    "BELOW",
            "liquidity":        "NONE",
            "atr":              10.0,
            "macro_sentiment":  "BEARISH",
        }
        weak_signal = {"bias": "BULLISH", "risk_reward": 1.5}
        assert _score(weak_signal, weak_ctx) == 0

    def test_blackout_on_already_low_score_stays_at_0(self):
        # Same weak setup + blackout → still floor 0.
        weak_ctx = {
            "market_structure": "NONE",
            "ema_alignment":    "BEARISH",
            "rsi":              80.0,
            "vwap_position":    "BELOW",
            "liquidity":        "NONE",
            "atr":              10.0,
            "news_blackout":    True,
        }
        weak_signal = {"bias": "BULLISH", "risk_reward": 1.5}
        assert _score(weak_signal, weak_ctx) == 0
