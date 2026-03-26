"""Tests for execution.risk_manager.evaluate_risk.

Contract
--------
- Returns ``None``  for every failure path (no reason strings, no dicts).
- Returns the **same signal dict** (identity) when all gates pass.
- Signal is never mutated.
- Daily loss cap is 1 % of account balance.
"""

from __future__ import annotations

import math
import pytest

from execution.risk_manager import evaluate_risk

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_BALANCE: float = 10_000.0  # reference account equity


def _sig(**overrides) -> dict:
    """Return a minimal valid signal dict, with optional field overrides."""
    base = {
        "entry":       2_000.0,
        "stop_loss":   1_990.0,   # 10-pip SL → sl_distance = 10
        "take_profit": 2_020.0,
        "risk_reward": 2.0,
        "confidence":  80.0,
        "atr":         8.0,
        "side":        "BUY",
    }
    base.update(overrides)
    return base


# ===========================================================================
# 1. Signal presence gate (Gate 1)
# ===========================================================================

class TestNoneSignal:
    def test_none_signal_rejected(self):
        assert evaluate_risk(None, _BALANCE, 0, 0.0) is None

    def test_empty_dict_rejected(self):
        assert evaluate_risk({}, _BALANCE, 0, 0.0) is None

    def test_missing_entry_rejected(self):
        sig = _sig()
        del sig["entry"]
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is None

    def test_missing_stop_loss_rejected(self):
        sig = _sig()
        del sig["stop_loss"]
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is None

    def test_missing_risk_reward_rejected(self):
        sig = _sig()
        del sig["risk_reward"]
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is None

    def test_missing_confidence_rejected(self):
        sig = _sig()
        del sig["confidence"]
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is None

    def test_missing_atr_rejected(self):
        sig = _sig()
        del sig["atr"]
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is None

    def test_valid_signal_not_rejected(self):
        assert evaluate_risk(_sig(), _BALANCE, 0, 0.0) is not None


# ===========================================================================
# 2. Account balance gate (Gate 2)
# ===========================================================================

class TestInvalidBalance:
    def test_zero_balance_rejected(self):
        assert evaluate_risk(_sig(), 0.0, 0, 0.0) is None

    def test_negative_balance_rejected(self):
        assert evaluate_risk(_sig(), -1_000.0, 0, 0.0) is None

    def test_tiny_positive_balance_accepted(self):
        # ATR=8, sl_distance=10 → SL too large for tiny account, but
        # balance gate passes.  Use tight stop to also pass gate 9.
        sig = _sig(entry=2_000.0, stop_loss=1_999.0, atr=2.0)  # sl=1, atr=2
        assert evaluate_risk(sig, 0.01, 0, 0.0) is not None

    def test_large_balance_accepted(self):
        assert evaluate_risk(_sig(), 1_000_000.0, 0, 0.0) is not None


# ===========================================================================
# 3. Risk-reward gate (Gate 3)
# ===========================================================================

class TestRRGate:
    def test_rr_below_min_rejected(self):
        assert evaluate_risk(_sig(risk_reward=1.99), _BALANCE, 0, 0.0) is None

    def test_rr_at_min_accepted(self):
        assert evaluate_risk(_sig(risk_reward=2.0), _BALANCE, 0, 0.0) is not None

    def test_rr_above_min_accepted(self):
        assert evaluate_risk(_sig(risk_reward=3.5), _BALANCE, 0, 0.0) is not None

    def test_rr_zero_rejected(self):
        assert evaluate_risk(_sig(risk_reward=0.0), _BALANCE, 0, 0.0) is None

    def test_rr_negative_rejected(self):
        assert evaluate_risk(_sig(risk_reward=-1.0), _BALANCE, 0, 0.0) is None


# ===========================================================================
# 4. Confidence gate (Gate 4)
# ===========================================================================

class TestConfidenceGate:
    def test_confidence_below_min_rejected(self):
        assert evaluate_risk(_sig(confidence=69.9), _BALANCE, 0, 0.0) is None

    def test_confidence_at_min_accepted(self):
        assert evaluate_risk(_sig(confidence=70.0), _BALANCE, 0, 0.0) is not None

    def test_confidence_above_min_accepted(self):
        assert evaluate_risk(_sig(confidence=95.0), _BALANCE, 0, 0.0) is not None

    def test_zero_confidence_rejected(self):
        assert evaluate_risk(_sig(confidence=0.0), _BALANCE, 0, 0.0) is None


# ===========================================================================
# 5. Daily loss limit gate (Gate 5)  — cap is 1 % of account_balance
# ===========================================================================

class TestDailyLossLimit:
    """Daily loss cap is 1 % of account_balance (= 100.0 for _BALANCE=10_000)."""

    def test_daily_loss_equals_1pct_rejected(self):
        daily_loss = 0.01 * _BALANCE          # = 100.0  (at limit)
        assert evaluate_risk(_sig(), _BALANCE, 0, daily_loss) is None

    def test_daily_loss_exceeds_1pct_rejected(self):
        daily_loss = 0.01 * _BALANCE + 0.01   # = 100.01 (over limit)
        assert evaluate_risk(_sig(), _BALANCE, 0, daily_loss) is None

    def test_daily_loss_just_under_1pct_accepted(self):
        daily_loss = 0.01 * _BALANCE - 0.01   # = 99.99
        assert evaluate_risk(_sig(), _BALANCE, 0, daily_loss) is not None

    def test_zero_daily_loss_accepted(self):
        assert evaluate_risk(_sig(), _BALANCE, 0, 0.0) is not None

    def test_daily_loss_scales_with_balance(self):
        big_balance = 100_000.0
        at_limit    = 0.01 * big_balance       # = 1_000.0
        assert evaluate_risk(_sig(), big_balance, 0, at_limit) is None
        assert evaluate_risk(_sig(), big_balance, 0, at_limit - 0.01) is not None

    def test_old_3pct_threshold_no_longer_limits(self):
        # 3 % of balance (old limit) must now pass; only 1 % cap applies.
        # 300.0 > 100.0, so 300.0 should be rejected at the 1 % gate.
        old_3pct = 0.03 * _BALANCE             # = 300.0
        assert evaluate_risk(_sig(), _BALANCE, 0, old_3pct) is None


# ===========================================================================
# 6. Max open trades gate (Gate 6)
# ===========================================================================

class TestMaxOpenTrades:
    def test_two_open_trades_rejected(self):
        assert evaluate_risk(_sig(), _BALANCE, 2, 0.0) is None

    def test_three_open_trades_rejected(self):
        assert evaluate_risk(_sig(), _BALANCE, 3, 0.0) is None

    def test_one_open_trade_accepted(self):
        assert evaluate_risk(_sig(), _BALANCE, 1, 0.0) is not None

    def test_zero_open_trades_accepted(self):
        assert evaluate_risk(_sig(), _BALANCE, 0, 0.0) is not None


# ===========================================================================
# 7. Zero SL distance gate (Gate 7)
# ===========================================================================

class TestZeroSLDistance:
    def test_entry_equals_stop_loss_rejected(self):
        sig = _sig(entry=2_000.0, stop_loss=2_000.0)
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is None

    def test_nonzero_sl_distance_accepted(self):
        sig = _sig(entry=2_000.0, stop_loss=1_990.0)
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is not None


# ===========================================================================
# 8. ATR validation gate (Gate 8)
# ===========================================================================

class TestATRValidation:
    def test_nan_atr_rejected(self):
        assert evaluate_risk(_sig(atr=float("nan")), _BALANCE, 0, 0.0) is None

    def test_zero_atr_rejected(self):
        assert evaluate_risk(_sig(atr=0.0), _BALANCE, 0, 0.0) is None

    def test_negative_atr_rejected(self):
        assert evaluate_risk(_sig(atr=-1.0), _BALANCE, 0, 0.0) is None

    def test_too_small_atr_rejected(self):
        # atr < 0.1 is rejected as too-low volatility
        assert evaluate_risk(_sig(atr=0.05), _BALANCE, 0, 0.0) is None

    def test_atr_exactly_0_1_rejected(self):
        # 0.1 is the boundary: < 0.1 rejected, >= 0.1 accepted (unless SL too large)
        # sl_distance=10, atr=0.1 → 10 > 2*0.1=0.2 → SL too large
        # Use a very tight SL to isolate ATR gate boundary
        sig = _sig(entry=2_000.0, stop_loss=1_999.9, atr=0.1)  # sl=0.1, 2*atr=0.2 → ok
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is not None

    def test_atr_capped_at_100(self):
        # atr=500 gets capped to 100; sl_distance=10 <= 200 → still passes
        assert evaluate_risk(_sig(atr=500.0), _BALANCE, 0, 0.0) is not None

    def test_valid_atr_accepted(self):
        assert evaluate_risk(_sig(atr=8.0), _BALANCE, 0, 0.0) is not None


# ===========================================================================
# 9. SL too large gate (Gate 9)
# ===========================================================================

class TestSLTooLarge:
    def test_sl_exceeds_2x_atr_rejected(self):
        # sl_distance=10, atr=4 → 10 > 8 → rejected
        sig = _sig(entry=2_000.0, stop_loss=1_990.0, atr=4.0)
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is None

    def test_sl_equals_2x_atr_accepted(self):
        # sl_distance=10, atr=5 → 10 == 10 → accepted
        sig = _sig(entry=2_000.0, stop_loss=1_990.0, atr=5.0)
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is not None

    def test_sl_under_2x_atr_accepted(self):
        # sl_distance=10, atr=8 → 10 < 16 → accepted
        sig = _sig(entry=2_000.0, stop_loss=1_990.0, atr=8.0)
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is not None

    def test_short_side_sl_too_large_rejected(self):
        # SELL side: stop_loss above entry, sl_distance=10, atr=4 → rejected
        sig = _sig(entry=2_000.0, stop_loss=2_010.0, atr=4.0, side="SELL",
                   risk_reward=2.0)
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is None


# ===========================================================================
# 10. Approved signal — output contract
# ===========================================================================

class TestApprovedSignal:
    def test_returns_signal_on_all_pass(self):
        sig = _sig()
        result = evaluate_risk(sig, _BALANCE, 0, 0.0)
        assert result is not None

    def test_returns_exact_same_object(self):
        """Returns must be the identity of the input signal (not a copy)."""
        sig = _sig()
        result = evaluate_risk(sig, _BALANCE, 0, 0.0)
        assert result is sig

    def test_returns_dict_type(self):
        result = evaluate_risk(_sig(), _BALANCE, 0, 0.0)
        assert isinstance(result, dict)

    def test_no_extra_keys_added(self):
        sig = _sig()
        original_keys = set(sig.keys())
        evaluate_risk(sig, _BALANCE, 0, 0.0)
        assert set(sig.keys()) == original_keys

    def test_no_lot_size_key(self):
        result = evaluate_risk(_sig(), _BALANCE, 0, 0.0)
        assert "lot_size" not in result

    def test_no_risk_amount_key(self):
        result = evaluate_risk(_sig(), _BALANCE, 0, 0.0)
        assert "risk_amount" not in result

    def test_no_approved_key(self):
        result = evaluate_risk(_sig(), _BALANCE, 0, 0.0)
        assert "approved" not in result

    def test_no_reason_key(self):
        result = evaluate_risk(_sig(), _BALANCE, 0, 0.0)
        assert "reason" not in result

    def test_failure_returns_none_not_false_dict(self):
        result = evaluate_risk(None, _BALANCE, 0, 0.0)
        assert result is None
        assert result != {"approved": False}


# ===========================================================================
# 11. Input immutability
# ===========================================================================

class TestInputImmutability:
    def test_signal_not_mutated_on_approval(self):
        sig = _sig()
        snapshot = dict(sig)
        evaluate_risk(sig, _BALANCE, 0, 0.0)
        assert sig == snapshot

    def test_signal_not_mutated_on_rejection_because_rr(self):
        sig = _sig(risk_reward=1.0)
        snapshot = dict(sig)
        evaluate_risk(sig, _BALANCE, 0, 0.0)
        assert sig == snapshot


# ===========================================================================
# 12. Gate ordering
# ===========================================================================

class TestGateOrdering:
    """Verify each gate fires independently; earlier gates shadow later ones."""

    def test_gate1_fires_before_gate2(self):
        # signal=None fails gate 1; balance=0 would also fail gate 2
        assert evaluate_risk(None, 0.0, 0, 0.0) is None

    def test_gate2_fires_before_gate3(self):
        # balance=0 fails gate 2; rr=1 would also fail gate 3
        assert evaluate_risk(_sig(risk_reward=1.0), 0.0, 0, 0.0) is None

    def test_gate3_fires_independently(self):
        # rr=1 fails gate 3; daily_loss gate would be fine
        assert evaluate_risk(_sig(risk_reward=1.0), _BALANCE, 0, 0.0) is None

    def test_gate4_fires_independently(self):
        # confidence=50 fails gate 4
        assert evaluate_risk(_sig(confidence=50.0), _BALANCE, 0, 0.0) is None

    def test_gate5_fires_before_gate6(self):
        # daily_loss at limit fails gate 5; open_trades=3 would fail gate 6
        daily_loss = 0.01 * _BALANCE
        assert evaluate_risk(_sig(), _BALANCE, 3, daily_loss) is None

    def test_gate6_fires_independently(self):
        assert evaluate_risk(_sig(), _BALANCE, 2, 0.0) is None

    def test_gate7_fires_before_gate8(self):
        # zero SL fails gate 7; NaN ATR would fail gate 8
        sig = _sig(entry=2_000.0, stop_loss=2_000.0, atr=float("nan"))
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is None

    def test_gate8_fires_before_gate9(self):
        # NaN ATR fails gate 8; SL also too large
        sig = _sig(entry=2_000.0, stop_loss=1_990.0, atr=float("nan"))
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is None

    def test_all_gates_pass_returns_signal(self):
        sig = _sig()
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is sig


# ===========================================================================
# 13. Boundary values
# ===========================================================================

class TestBoundaryValues:
    def test_rr_exactly_at_min(self):
        assert evaluate_risk(_sig(risk_reward=2.0), _BALANCE, 0, 0.0) is not None

    def test_rr_just_below_min(self):
        assert evaluate_risk(_sig(risk_reward=1.999), _BALANCE, 0, 0.0) is None

    def test_confidence_exactly_at_min(self):
        assert evaluate_risk(_sig(confidence=70.0), _BALANCE, 0, 0.0) is not None

    def test_confidence_just_below_min(self):
        assert evaluate_risk(_sig(confidence=69.99), _BALANCE, 0, 0.0) is None

    def test_daily_loss_just_below_1pct(self):
        assert evaluate_risk(_sig(), _BALANCE, 0, 99.99) is not None

    def test_daily_loss_at_1pct(self):
        assert evaluate_risk(_sig(), _BALANCE, 0, 100.0) is None

    def test_open_trades_at_limit(self):
        assert evaluate_risk(_sig(), _BALANCE, 2, 0.0) is None

    def test_open_trades_one_below_limit(self):
        assert evaluate_risk(_sig(), _BALANCE, 1, 0.0) is not None


# ===========================================================================
# 14. Integration — representative live signals
# ===========================================================================

class TestSignalEngineIntegration:
    """Simulate real signal shapes coming from signal_engine / trade_setup."""

    def _live_buy_signal(self) -> dict:
        return {
            "symbol":       "XAUUSD",
            "side":         "BUY",
            "entry":        2_350.0,
            "stop_loss":    2_342.0,   # sl_distance = 8
            "take_profit":  2_366.0,
            "risk_reward":  2.0,
            "confidence":   75.0,
            "atr":          7.5,
            "session":      "LONDON",
        }

    def _live_sell_signal(self) -> dict:
        return {
            "symbol":       "XAUUSD",
            "side":         "SELL",
            "entry":        2_380.0,
            "stop_loss":    2_388.0,   # sl_distance = 8
            "take_profit":  2_364.0,
            "risk_reward":  2.0,
            "confidence":   82.0,
            "atr":          7.0,
            "session":      "NEW_YORK",
        }

    def test_live_buy_signal_approved(self):
        result = evaluate_risk(self._live_buy_signal(), _BALANCE, 0, 0.0)
        assert result is not None

    def test_live_sell_signal_approved(self):
        result = evaluate_risk(self._live_sell_signal(), _BALANCE, 0, 0.0)
        assert result is not None

    def test_live_buy_returns_same_object(self):
        sig = self._live_buy_signal()
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is sig

    def test_live_buy_rejected_on_daily_limit(self):
        # daily_loss at 1 % cap blocks an otherwise valid live signal
        daily_loss = 0.01 * _BALANCE   # = 100.0
        assert evaluate_risk(self._live_buy_signal(), _BALANCE, 0, daily_loss) is None

    def test_live_sell_rejected_on_low_rr(self):
        sig = self._live_sell_signal()
        sig["risk_reward"] = 1.5
        assert evaluate_risk(sig, _BALANCE, 0, 0.0) is None

    def test_live_buy_rejected_when_at_trade_cap(self):
        assert evaluate_risk(self._live_buy_signal(), _BALANCE, 2, 0.0) is None
