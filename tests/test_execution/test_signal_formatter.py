"""Comprehensive test suite for execution/signal_formatter.py — format_trade_signal().

Test group overview
-------------------
 1. TestNoTradeGuard         (2)  — None signal returns no-trade string
 2. TestOutputIsString       (2)  — return type always str
 3. TestTradeCardFields      (11) — every field present with correct value
 4. TestNumericFormatting    (5)  — 2dp prices, integer %, 1:X ratio
 5. TestRRFormatting         (5)  — integer and fractional RR strings
 6. TestRRPrecision          (2)  — floating-point tolerance in RR check
 7. TestInputImmutability    (1)  — signal dict not mutated
 8. TestDeterminism          (2)  — same input always same output
 9. TestIntegration          (4)  — full pipeline: signal_engine → formatter
"""

from __future__ import annotations

import copy

import pytest

from execution.signal_formatter import format_trade_signal


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sig(
    timeframe:     str   = "M5",
    bias:          str   = "SELL",
    entry:         float = 1990.0,
    stop_loss:     float = 2000.0,
    take_profit:   float = 1970.0,
    risk_reward:   float = 2.0,
    confidence:    float = 80.0,
    mtf_bias:      str   = "BEARISH",
    bias_strength: str   = "STRONG",
    context:       str   = "TREND",
) -> dict:
    return {
        "timeframe":     timeframe,
        "bias":          bias,
        "entry":         entry,
        "stop_loss":     stop_loss,
        "take_profit":   take_profit,
        "risk_reward":   risk_reward,
        "confidence":    confidence,
        "mtf_bias":      mtf_bias,
        "bias_strength": bias_strength,
        "context":       context,
        # extra internal fields that must NOT appear in output
        "atr":           10.0,
        "pair":          "XAUUSD",
    }


# ===========================================================================
# 1. No-trade guard
# ===========================================================================

class TestNoTradeGuard:

    def test_none_signal_returns_no_trade(self):
        assert format_trade_signal(None) == "No trade opportunity"

    def test_no_trade_is_exact_string(self):
        assert format_trade_signal(None) == "No trade opportunity"


# ===========================================================================
# 2. Return type
# ===========================================================================

class TestOutputIsString:

    def test_no_trade_returns_str(self):
        assert isinstance(format_trade_signal(None), str)

    def test_signal_returns_str(self):
        assert isinstance(format_trade_signal(_sig()), str)


# ===========================================================================
# 3. Trade card fields
# ===========================================================================

class TestTradeCardFields:

    @pytest.fixture()
    def card(self) -> str:
        return format_trade_signal(_sig())

    def test_pair_line_present(self, card):
        assert "Pair: XAUUSD" in card

    def test_timeframe_line_present(self, card):
        assert "Timeframe: M5" in card

    def test_bias_line_present(self, card):
        assert "Bias: SELL" in card

    def test_entry_line_present(self, card):
        assert "Entry:" in card

    def test_stop_loss_line_present(self, card):
        assert "Stop Loss:" in card

    def test_take_profit_line_present(self, card):
        assert "Take Profit:" in card

    def test_risk_reward_line_present(self, card):
        assert "Risk Reward:" in card

    def test_confidence_line_present(self, card):
        assert "Confidence:" in card

    def test_htf_bias_line_present(self, card):
        assert "HTF Bias:" in card

    def test_context_line_present(self, card):
        assert "Context:" in card

    def test_invalidation_line_present(self, card):
        assert "Invalidation:" in card


# ===========================================================================
# 4. Numeric formatting
# ===========================================================================

class TestNumericFormatting:

    def test_entry_rounded_to_2dp(self):
        assert "Entry: 1990.12" in format_trade_signal(_sig(entry=1990.123456))

    def test_stop_loss_rounded_to_2dp(self):
        assert "Stop Loss: 2000.79" in format_trade_signal(_sig(stop_loss=2000.789))

    def test_take_profit_rounded_to_2dp(self):
        assert "Take Profit: 1970.56" in format_trade_signal(_sig(take_profit=1970.555))

    def test_confidence_formatted_as_integer_percent(self):
        assert "Confidence: 80%" in format_trade_signal(_sig(confidence=80.0))

    def test_confidence_truncated_not_fractional(self):
        # int(85.7) == 85
        assert "Confidence: 85%" in format_trade_signal(_sig(confidence=85.7))


# ===========================================================================
# 5. RR formatting
# ===========================================================================

class TestRRFormatting:

    def test_integer_rr_has_no_decimal(self):
        card = format_trade_signal(_sig(risk_reward=2.0))
        assert "Risk Reward: 1:2" in card
        assert "1:2.0" not in card

    def test_rr_3_formatted_correctly(self):
        assert "Risk Reward: 1:3" in format_trade_signal(_sig(risk_reward=3.0))

    def test_fractional_rr_preserved(self):
        assert "Risk Reward: 1:2.5" in format_trade_signal(_sig(risk_reward=2.5))

    def test_rr_4_formatted_correctly(self):
        assert "Risk Reward: 1:4" in format_trade_signal(_sig(risk_reward=4.0))

    def test_rr_2_75_preserved(self):
        assert "Risk Reward: 1:2.75" in format_trade_signal(_sig(risk_reward=2.75))


# ===========================================================================
# 6. RR precision — tolerance-based integer detection
# ===========================================================================

class TestRRPrecision:

    def test_rr_float_precision_integer(self):
        """2.0000000001 is within 1e-6 of 2 → should render as '1:2'."""
        card = format_trade_signal(_sig(risk_reward=2.0000000001))
        assert "Risk Reward: 1:2" in card
        assert "1:2.0000000001" not in card

    def test_rr_float_precision_non_integer(self):
        """2.5000000001 is NOT within 1e-6 of any integer → passed through."""
        card = format_trade_signal(_sig(risk_reward=2.5000000001))
        assert "Risk Reward: 1:2.5000000001" in card


# ===========================================================================
# 7. Input immutability
# ===========================================================================

class TestInputImmutability:

    def test_signal_not_mutated(self):
        sig = _sig()
        original = copy.deepcopy(sig)
        format_trade_signal(sig)
        assert sig == original


# ===========================================================================
# 8. Determinism
# ===========================================================================

class TestDeterminism:

    def test_same_input_same_output(self):
        sig = _sig()
        assert format_trade_signal(sig) == format_trade_signal(sig)

    def test_no_trade_always_identical(self):
        assert format_trade_signal(None) == format_trade_signal(None)


# ===========================================================================
# 9. Integration — full pipeline: signal_engine → formatter
# ===========================================================================

class TestIntegration:
    """Format real output produced by the full upstream pipeline."""

    @pytest.fixture()
    def signal(self):
        import pandas as pd
        from ai.signal_engine import generate_trade_signal
        from execution.risk_manager import evaluate_risk

        _TS = pd.Timestamp("2024-01-01 09:00:00")
        _DT = pd.Timedelta(minutes=5)

        def row(i, **kw):
            base = dict(
                timestamp=_TS + _DT * i,
                open=1989.0, high=1995.0, low=1983.0, close=1990.0,
                ATR=10.0, EMA_20=2000.0, RSI=40.0,
                trend="BEARISH", event="",
                liquidity_sweep_high=False, liquidity_sweep_low=False,
                bearish_order_block=False, bullish_order_block=False,
                ob_high=float("nan"), ob_low=float("nan"),
            )
            base.update(kw)
            return base

        df = pd.DataFrame([
            row(0, liquidity_sweep_high=True),
            row(1, bearish_order_block=True, ob_high=1995.0, ob_low=1985.0),
            row(2),
        ])
        raw_signal = generate_trade_signal(df, "M5")
        # Only pass a signal through if risk gate approves it
        if evaluate_risk(raw_signal, 10_000.0, 0, 0.0) is None:
            return None
        return raw_signal

    def test_pipeline_produces_string(self, signal):
        assert isinstance(format_trade_signal(signal), str)

    def test_pipeline_card_contains_pair(self, signal):
        assert "Pair: XAUUSD" in format_trade_signal(signal)

    def test_pipeline_card_contains_bias(self, signal):
        assert "Bias: SELL" in format_trade_signal(signal)

    def test_pipeline_card_not_no_trade(self, signal):
        assert format_trade_signal(signal) != "No trade opportunity"
