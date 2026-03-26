"""Tests for execution/trade_setup.py"""

import math
import pytest

from execution.trade_setup import build_trade_setup, MIN_RR


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _signal(**overrides):
    base = {"bias": "BULLISH"}
    base.update(overrides)
    return base


def _context(**overrides):
    """Default: clean BULLISH REVERSAL geometry.

    entry=2000, recent_low=1990, recent_high=2040, atr=10
    atr_buffer  = 5
    stop_loss   = 1990 - 5 = 1985       → risk   = 15
    target      = 2000 + (15 × 2.5)     = 2037.5
    take_profit = min(2037.5, 2040)     = 2037.5 → reward = 37.5
    rr          = 37.5 / 15             = 2.5
    """
    base = {
        "setup_type":  "REVERSAL",
        "atr":         10.0,
        "recent_high": 2040.0,
        "recent_low":  1990.0,
        "entry_price": 2000.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_returns_dict_for_valid_setup(self):
        result = build_trade_setup(_signal(), _context())
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = build_trade_setup(_signal(), _context())
        for key in ("entry", "stop_loss", "take_profit", "risk_reward", "invalidation"):
            assert key in result

    def test_all_values_are_numeric(self):
        result = build_trade_setup(_signal(), _context())
        for key in ("entry", "stop_loss", "take_profit", "risk_reward", "invalidation"):
            assert isinstance(result[key], float)

    def test_invalidation_equals_stop_loss(self):
        result = build_trade_setup(_signal(), _context())
        assert result["invalidation"] == result["stop_loss"]

    def test_price_rounded_to_5_decimals(self):
        ctx = _context(atr=3.333333, recent_low=1990.111111, entry_price=2001.999999)
        result = build_trade_setup(_signal(), ctx)
        for key in ("entry", "stop_loss", "take_profit", "invalidation"):
            assert result[key] == round(result[key], 5)

    def test_rr_rounded_to_2_decimals(self):
        result = build_trade_setup(_signal(), _context())
        assert result["risk_reward"] == round(result["risk_reward"], 2)


# ---------------------------------------------------------------------------
# 2. Entry
# ---------------------------------------------------------------------------

class TestEntry:
    def test_entry_equals_entry_price(self):
        # entry=2015.5, sl=2010-5=2005 (risk=10.5), tp=2060 (reward=44.5), rr≈4.24 ✓
        ctx = _context(entry_price=2015.5, recent_low=2010.0, recent_high=2060.0, atr=10.0)
        result = build_trade_setup(_signal(), ctx)
        assert result["entry"] == 2015.5

    def test_entry_unchanged_for_bearish(self):
        result = build_trade_setup(_signal(bias="BEARISH"), _context(
            bias="BEARISH", recent_high=2050.0, recent_low=1980.0,
            entry_price=2040.0, atr=10.0,
        ))
        assert result["entry"] == 2040.0


# ---------------------------------------------------------------------------
# 3. Stop Loss — BULLISH
# ---------------------------------------------------------------------------

class TestStopLossBullish:
    def test_stop_loss_below_recent_low(self):
        result = build_trade_setup(_signal(), _context())
        assert result["stop_loss"] < _context()["recent_low"]

    def test_stop_loss_equals_recent_low_minus_half_atr(self):
        ctx = _context(recent_low=1990.0, atr=10.0)
        result = build_trade_setup(_signal(), ctx)
        assert result["stop_loss"] == round(1990.0 - 5.0, 5)

    def test_stop_loss_scales_with_atr(self):
        ctx_small = _context(atr=4.0)
        ctx_large = _context(atr=20.0)
        assert (
            build_trade_setup(_signal(), ctx_large)["stop_loss"]
            < build_trade_setup(_signal(), ctx_small)["stop_loss"]
        )

    def test_stop_loss_reversal_type(self):
        ctx = _context(setup_type="REVERSAL", recent_low=1988.0, atr=8.0)
        result = build_trade_setup(_signal(), ctx)
        assert result["stop_loss"] == round(1988.0 - 4.0, 5)

    def test_stop_loss_continuation_type(self):
        ctx = _context(setup_type="CONTINUATION", recent_low=1988.0, atr=8.0)
        result = build_trade_setup(_signal(), ctx)
        assert result["stop_loss"] == round(1988.0 - 4.0, 5)

    def test_stop_loss_breakout_type(self):
        ctx = _context(setup_type="BREAKOUT", recent_low=1988.0, atr=8.0)
        result = build_trade_setup(_signal(), ctx)
        assert result["stop_loss"] == round(1988.0 - 4.0, 5)


# ---------------------------------------------------------------------------
# 4. Stop Loss — BEARISH
# ---------------------------------------------------------------------------

class TestStopLossBearish:
    def _bearish_ctx(self, **overrides):
        base = {
            "setup_type":  "REVERSAL",
            "atr":         10.0,
            "recent_high": 2050.0,
            "recent_low":  1980.0,
            "entry_price": 2040.0,
        }
        base.update(overrides)
        return base

    def test_stop_loss_above_recent_high(self):
        result = build_trade_setup(_signal(bias="BEARISH"), self._bearish_ctx())
        assert result["stop_loss"] > self._bearish_ctx()["recent_high"]

    def test_stop_loss_equals_recent_high_plus_half_atr(self):
        ctx = self._bearish_ctx(recent_high=2050.0, atr=10.0)
        result = build_trade_setup(_signal(bias="BEARISH"), ctx)
        assert result["stop_loss"] == round(2050.0 + 5.0, 5)

    def test_stop_loss_scales_with_atr_bearish(self):
        ctx_small = self._bearish_ctx(atr=4.0)
        ctx_large = self._bearish_ctx(atr=20.0)
        assert (
            build_trade_setup(_signal(bias="BEARISH"), ctx_large)["stop_loss"]
            > build_trade_setup(_signal(bias="BEARISH"), ctx_small)["stop_loss"]
        )


# ---------------------------------------------------------------------------
# 5. Take Profit — capping behaviour
# ---------------------------------------------------------------------------

class TestTakeProfit:
    def test_bullish_tp_capped_at_recent_high_when_target_exceeds(self):
        # target = 2000 + 15*2.5 = 2037.5 > recent_high(2030) → tp = 2030
        ctx = _context(recent_high=2030.0)
        result = build_trade_setup(_signal(), ctx)
        assert result["take_profit"] == 2030.0

    def test_bearish_tp_capped_at_recent_low_when_target_below(self):
        # entry=2040, sl=2055, risk=15, target=2040-37.5=2002.5 < recent_low(2005) → tp=2005
        ctx = {
            "setup_type": "REVERSAL", "atr": 10.0,
            "recent_high": 2050.0, "recent_low": 2005.0, "entry_price": 2040.0,
        }
        result = build_trade_setup(_signal(bias="BEARISH"), ctx)
        assert result["take_profit"] == 2005.0


# ---------------------------------------------------------------------------
# 5b. Dynamic Take Profit
# ---------------------------------------------------------------------------

class TestDynamicTakeProfit:
    """TP = 2.5 × risk from entry, capped at the liquidity level."""

    def test_bullish_tp_equals_dynamic_target_when_below_cap(self):
        # target = 2000 + 15*2.5 = 2037.5 < recent_high(2040) → tp = 2037.5
        result = build_trade_setup(_signal(), _context())
        assert result["take_profit"] == round(2037.5, 5)

    def test_bullish_tp_is_capped_when_target_exceeds_recent_high(self):
        # target = 2000 + 15*2.5 = 2037.5 > recent_high(2035) → tp = 2035
        # reward=35, rr=35/15≈2.33 ≥ 2 so result is valid ✓
        ctx = _context(recent_high=2035.0)
        result = build_trade_setup(_signal(), ctx)
        assert result["take_profit"] == 2035.0

    def test_bullish_tp_never_exceeds_recent_high(self):
        result = build_trade_setup(_signal(), _context())
        assert result["take_profit"] <= _context()["recent_high"]

    def test_bearish_tp_equals_dynamic_target_when_above_cap(self):
        # entry=2040, sl=2055, risk=15, target=2040-37.5=2002.5 > recent_low(1990) → tp=2002.5
        ctx = {
            "setup_type": "REVERSAL", "atr": 10.0,
            "recent_high": 2050.0, "recent_low": 1990.0, "entry_price": 2040.0,
        }
        result = build_trade_setup(_signal(bias="BEARISH"), ctx)
        assert result["take_profit"] == round(2002.5, 5)

    def test_bearish_tp_is_capped_when_target_below_recent_low(self):
        # target=2002.5 < recent_low(2005) → tp=2005
        ctx = {
            "setup_type": "REVERSAL", "atr": 10.0,
            "recent_high": 2050.0, "recent_low": 2005.0, "entry_price": 2040.0,
        }
        result = build_trade_setup(_signal(bias="BEARISH"), ctx)
        assert result["take_profit"] == 2005.0

    def test_bearish_tp_never_below_recent_low(self):
        ctx = {
            "setup_type": "REVERSAL", "atr": 10.0,
            "recent_high": 2050.0, "recent_low": 1980.0, "entry_price": 2040.0,
        }
        result = build_trade_setup(_signal(bias="BEARISH"), ctx)
        if result is not None:
            assert result["take_profit"] >= ctx["recent_low"]

    def test_tp_target_scales_with_risk(self):
        # Larger ATR → larger risk → larger uncapped target
        ctx_small = _context(atr=4.0,  recent_high=2100.0)  # risk≈12, target≈2030
        ctx_large = _context(atr=20.0, recent_high=2100.0)  # risk=20,  target=2050
        small = build_trade_setup(_signal(), ctx_small)
        large = build_trade_setup(_signal(), ctx_large)
        if small and large:
            assert large["take_profit"] > small["take_profit"]


# ---------------------------------------------------------------------------
# 5c. Bad-entry filter (reject when reward ≤ 0.5 × risk)
# ---------------------------------------------------------------------------

class TestBadEntryFilter:
    def test_entry_very_close_to_capped_tp_returns_none(self):
        # entry=2035, recent_high=2036 → tp=2036 (capped), reward=1
        # sl=1990-5=1985, risk=50 → 0.5*risk=25, reward(1) ≤ 25 → None
        ctx = _context(entry_price=2035.0, recent_high=2036.0, recent_low=1990.0, atr=10.0)
        result = build_trade_setup(_signal(), ctx)
        assert result is None

    def test_entry_well_separated_from_tp_passes(self):
        # Default geometry: reward=37.5 >> 0.5*risk=7.5
        result = build_trade_setup(_signal(), _context())
        assert result is not None

    def test_reward_exactly_half_risk_is_rejected(self):
        # entry=2000, sl=1985 (risk=15), 0.5*risk=7.5
        # target=2037.5 > recent_high(2007.5) → tp=2007.5 (capped), reward=7.5 ≤ 7.5 → None
        ctx = _context(recent_high=2007.5)
        result = build_trade_setup(_signal(), ctx)
        assert result is None


# ---------------------------------------------------------------------------
# 6. Risk/Reward calculation
# ---------------------------------------------------------------------------

class TestRiskReward:
    def test_rr_is_reward_over_risk(self):
        # entry=2000, sl=1985 (risk=15), tp=min(2037.5, 2040)=2037.5 (reward=37.5)
        result = build_trade_setup(_signal(), _context())
        expected = round(37.5 / 15.0, 2)
        assert result["risk_reward"] == expected

    def test_rr_minimum_exactly_2_passes(self):
        # Craft geometry so rr == exactly 2.0
        # risk = 10, reward = 20
        # entry=2010, sl=2000 (need sl=entry - risk=2010-10=2000)
        #   recent_low - 0.5*atr = 2000  → recent_low=2005, atr=10 → sl=2005-5=2000  ✓
        # tp = recent_high = 2030  → reward = 2030-2010 = 20  ✓
        ctx = _context(entry_price=2010.0, recent_low=2005.0, recent_high=2030.0, atr=10.0)
        result = build_trade_setup(_signal(), ctx)
        assert result is not None
        assert result["risk_reward"] == 2.0

    def test_rr_just_below_2_returns_none(self):
        # risk=10, reward=19 → rr=1.9 < 2
        # entry=2010, sl=2000 (recent_low=2005, atr=10 → sl=2000)
        # tp=recent_high=2029 → reward=19
        ctx = _context(entry_price=2010.0, recent_low=2005.0, recent_high=2029.0, atr=10.0)
        result = build_trade_setup(_signal(), ctx)
        assert result is None

    def test_rr_zero_risk_returns_none(self):
        # entry == stop_loss: occurs when entry_price == recent_low - 0.5*atr
        # recent_low=2000, atr=0 → sl=2000, and entry=2000
        ctx = _context(entry_price=2000.0, recent_low=2000.0, atr=0.0)
        result = build_trade_setup(_signal(), ctx)
        assert result is None


# ---------------------------------------------------------------------------
# 7. RR filter (returns None)
# ---------------------------------------------------------------------------

class TestRRFilter:
    def test_returns_none_when_rr_below_min(self):
        # TP very close to entry → tiny reward
        ctx = _context(recent_high=2001.0, recent_low=1990.0, entry_price=2000.0, atr=10.0)
        result = build_trade_setup(_signal(), ctx)
        assert result is None

    def test_returns_dict_when_rr_meets_min(self):
        result = build_trade_setup(_signal(), _context())
        assert result is not None


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_same_output(self):
        sig = _signal()
        ctx = _context()
        assert build_trade_setup(sig, ctx) == build_trade_setup(sig, ctx)

    def test_different_bias_different_output(self):
        ctx = _context(
            recent_high=2050.0, recent_low=1980.0,
            entry_price=2000.0, atr=10.0,
        )
        bull = build_trade_setup(_signal(bias="BULLISH"), ctx)
        bear = build_trade_setup(_signal(bias="BEARISH"), ctx)
        if bull is not None and bear is not None:
            assert bull["stop_loss"] != bear["stop_loss"]


# ---------------------------------------------------------------------------
# 9. Setup type symmetry
# ---------------------------------------------------------------------------

class TestSetupTypeSymmetry:
    """SL geometry is identical across all three setup types (by design)."""
    @pytest.mark.parametrize("setup_type", ["REVERSAL", "CONTINUATION", "BREAKOUT"])
    def test_all_setup_types_produce_same_sl_geometry(self, setup_type):
        ctx = _context(setup_type=setup_type)
        result = build_trade_setup(_signal(), ctx)
        assert result is not None
        assert result["stop_loss"] == round(
            _context()["recent_low"] - 0.5 * _context()["atr"], 5
        )


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_missing_keys_returns_none_not_exception(self):
        # Empty dicts → atr=0 → risk=0 → None
        result = build_trade_setup({}, {})
        assert result is None

    def test_large_atr_still_computes(self):
        # atr=200 → buffer=100, sl=1800-100=1700, risk=300
        # recent_high=2700 → reward=700, rr≈2.33 ✓
        ctx = _context(atr=200.0, recent_low=1800.0, recent_high=2700.0, entry_price=2000.0)
        result = build_trade_setup(_signal(), ctx)
        assert result is not None
        assert result["stop_loss"] == round(1800.0 - 100.0, 5)

    def test_high_precision_prices_rounded(self):
        ctx = _context(
            atr=10.123456789,
            recent_low=1990.123456789,
            recent_high=2050.987654321,
            entry_price=2000.555555555,
        )
        result = build_trade_setup(_signal(), ctx)
        if result is not None:
            assert result["stop_loss"] == round(result["stop_loss"], 5)
            assert result["entry"] == round(result["entry"], 5)

    def test_bearish_valid_trade_returns_dict(self):
        ctx = {
            "setup_type": "REVERSAL",
            "atr": 10.0,
            "recent_high": 2060.0,
            "recent_low": 1980.0,
            "entry_price": 2045.0,
        }
        result = build_trade_setup(_signal(bias="BEARISH"), ctx)
        # stop = 2060 + 5 = 2065, risk = 20, tp = 1980, reward = 65, rr = 3.25
        assert result is not None
        assert result["risk_reward"] >= 2.0

    def test_atr_zero_with_nonzero_prices_stop_exactly_at_low(self):
        ctx = _context(atr=0.0, recent_low=1990.0, entry_price=2000.0)
        # risk = |2000 - 1990| = 10  →  valid if reward ≥ 20
        result = build_trade_setup(_signal(), ctx)
        if result is not None:
            assert result["stop_loss"] == 1990.0


# ---------------------------------------------------------------------------
# 11. Entry Confirmation
# ---------------------------------------------------------------------------

class TestEntryConfirmation:
    """Entry confirmation gate: only applied when caller supplies candle/m1 keys."""

    def _ctx_confirmed_buy(self, **overrides):
        """Bullish confirmation: long lower wick, close near high."""
        base = _context()   # valid base geometry
        base.update({
            # candle: low=1990, open=1995, close=2005, high=2010
            # lower_wick = 1995-1990 = 5   body = |2005-1995| = 10
            # But we need lower_wick > body AND close in upper 60%
            # Use: low=1990, open=1992, close=2008, high=2010
            # lower_wick = 1992-1990 = 2; body = 16 → NOT confirmed (wick < body)
            # Better: low=1990, open=2000, close=2008, high=2010 (bullish)
            # lower_wick = 2000-1990 = 10; body = |2008-2000| = 8
            # upper_body = (2008-1990) / (2010-1990) = 18/20 = 0.9 > 0.6 ✓
            # lower_wick(10) > body(8) ✓  → confirmed
            "candle_low":   1990.0,
            "candle_open":  2000.0,
            "candle_close": 2008.0,
            "candle_high":  2010.0,
        })
        base.update(overrides)
        return base

    def _ctx_unconfirmed_buy(self, **overrides):
        """Bullish rejection candle fails: close near low (bearish candle)."""
        base = _context()
        base.update({
            # low=1990, open=2008, close=1995, high=2010 (bearish body)
            # lower_wick = 1995-1990 = 5; body = |1995-2008| = 13 → wick < body → NOT confirmed
            "candle_low":   1990.0,
            "candle_open":  2008.0,
            "candle_close": 1995.0,
            "candle_high":  2010.0,
        })
        base.update(overrides)
        return base

    def test_no_conf_keys_bypasses_gate(self):
        """Without any confirmation keys the gate is skipped entirely."""
        result = build_trade_setup(_signal(), _context())
        assert result is not None

    def test_entry_confirmed_true_when_no_keys(self):
        result = build_trade_setup(_signal(), _context())
        assert result is not None
        assert result["entry_confirmed"] is True

    def test_m1_choch_confirms_entry(self):
        ctx = _context()
        ctx["m1_choch"] = True
        result = build_trade_setup(_signal(), ctx)
        assert result is not None
        assert result["entry_confirmed"] is True

    def test_m1_choch_false_no_candle_returns_none(self):
        """m1_choch=False triggers the gate; without a rejection candle → rejected."""
        ctx = _context()
        ctx["m1_choch"] = False
        # No candle data → candle_range=0 → strong_rejection=False → entry_confirmed=False
        result = build_trade_setup(_signal(), ctx)
        assert result is None

    def test_bullish_rejection_candle_confirms(self):
        result = build_trade_setup(_signal(bias="BULLISH"), self._ctx_confirmed_buy())
        assert result is not None
        assert result["entry_confirmed"] is True

    def test_bullish_no_rejection_candle_returns_none(self):
        result = build_trade_setup(_signal(bias="BULLISH"), self._ctx_unconfirmed_buy())
        assert result is None

    def test_bearish_rejection_candle_confirms(self):
        ctx = {
            "setup_type":   "REVERSAL",
            "atr":          10.0,
            "recent_high":  2060.0,
            "recent_low":   1980.0,
            "entry_price":  2045.0,
            # high=2060, open=2055, close=2042, low=2040
            # upper_wick = 2060-2055 = 5; body = |2042-2055| = 13 → wick < body → NOT confirmed
            # Use: high=2060, open=2050, close=2042, low=2040
            # upper_wick = 2060-2050 = 10; body = |2042-2050| = 8
            # lower_body = (2060-2042) / (2060-2040) = 18/20 = 0.9 > 0.6 ✓  → confirmed
            "candle_high":  2060.0,
            "candle_open":  2050.0,
            "candle_close": 2042.0,
            "candle_low":   2040.0,
        }
        result = build_trade_setup(_signal(bias="BEARISH"), ctx)
        assert result is not None
        assert result["entry_confirmed"] is True

    def test_output_has_entry_confirmed_key(self):
        result = build_trade_setup(_signal(), _context())
        assert "entry_confirmed" in result
