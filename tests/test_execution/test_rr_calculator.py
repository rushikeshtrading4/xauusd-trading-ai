"""Tests for execution/rr_calculator.py — compute_risk_reward()"""

import math
import pytest

from execution.rr_calculator import compute_risk_reward


# ---------------------------------------------------------------------------
# 1. Spec examples (from docstring / requirements)
# ---------------------------------------------------------------------------

class TestSpecExamples:

    def test_buy_example_from_spec(self):
        # entry=2000, sl=1990, tp=2020 → risk=10, reward=20 → rr=2.0
        assert compute_risk_reward(2000, 1990, 2020, "BUY") == 2.0

    def test_sell_example_from_spec(self):
        # entry=2000, sl=2010, tp=1980 → risk=10, reward=20 → rr=2.0
        assert compute_risk_reward(2000, 2010, 1980, "SELL") == 2.0


# ---------------------------------------------------------------------------
# 2. Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:

    def test_returns_float(self):
        result = compute_risk_reward(2000, 1990, 2020, "BUY")
        assert isinstance(result, float)

    def test_result_rounded_to_2_decimal_places(self):
        result = compute_risk_reward(2000, 1990, 2033, "BUY")
        assert result == round(result, 2)

    def test_result_is_positive(self):
        result = compute_risk_reward(2000, 1990, 2020, "BUY")
        assert result > 0.0

    def test_result_is_finite(self):
        result = compute_risk_reward(2000, 1990, 2020, "BUY")
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# 3. Correct RR computation — BUY
# ---------------------------------------------------------------------------

class TestBuyRR:

    def test_rr_1_to_1(self):
        # risk=10, reward=10 → rr=1.0
        assert compute_risk_reward(2000, 1990, 2010, "BUY") == pytest.approx(1.0)

    def test_rr_1_to_2(self):
        assert compute_risk_reward(2000, 1990, 2020, "BUY") == pytest.approx(2.0)

    def test_rr_1_to_3(self):
        assert compute_risk_reward(2000, 1990, 2030, "BUY") == pytest.approx(3.0)

    def test_rr_asymmetric(self):
        # risk=15, reward=37.5 → rr=2.5
        assert compute_risk_reward(2000, 1985, 2037.5, "BUY") == pytest.approx(2.5)

    def test_rr_fractional_distances(self):
        # risk=5, reward=10 → rr=2.0
        assert compute_risk_reward(2005, 2000, 2015, "BUY") == pytest.approx(2.0)

    def test_lowercase_buy_accepted(self):
        assert compute_risk_reward(2000, 1990, 2020, "buy") == pytest.approx(2.0)

    def test_mixed_case_buy_accepted(self):
        assert compute_risk_reward(2000, 1990, 2020, "Buy") == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 4. Correct RR computation — SELL
# ---------------------------------------------------------------------------

class TestSellRR:

    def test_rr_1_to_1(self):
        assert compute_risk_reward(2000, 2010, 1990, "SELL") == pytest.approx(1.0)

    def test_rr_1_to_2(self):
        assert compute_risk_reward(2000, 2010, 1980, "SELL") == pytest.approx(2.0)

    def test_rr_1_to_3(self):
        assert compute_risk_reward(2000, 2010, 1970, "SELL") == pytest.approx(3.0)

    def test_rr_asymmetric(self):
        # risk=15, reward=37.5 → rr=2.5
        assert compute_risk_reward(2000, 2015, 1962.5, "SELL") == pytest.approx(2.5)

    def test_lowercase_sell_accepted(self):
        assert compute_risk_reward(2000, 2010, 1980, "sell") == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 5. Rounding behaviour
# ---------------------------------------------------------------------------

class TestRounding:

    def test_irrational_rr_rounded_to_2dp(self):
        # risk=3, reward=10 → rr=3.333... → 3.33
        assert compute_risk_reward(2000, 1997, 2010, "BUY") == pytest.approx(3.33, abs=0.005)

    def test_rounding_does_not_exceed_2dp(self):
        result = compute_risk_reward(2000, 1997, 2010, "BUY")
        assert result == round(result, 2)

    def test_exactly_2_point_0_not_2_point_00(self):
        result = compute_risk_reward(2000, 1990, 2020, "BUY")
        assert result == 2.0

    def test_sell_rounding_matches_buy_symmetry(self):
        buy_rr  = compute_risk_reward(2000, 1997, 2010, "BUY")
        sell_rr = compute_risk_reward(2000, 2003, 1990, "SELL")
        assert buy_rr == sell_rr


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_inputs_same_output_buy(self):
        args = (2000, 1990, 2020, "BUY")
        assert compute_risk_reward(*args) == compute_risk_reward(*args)

    def test_same_inputs_same_output_sell(self):
        args = (2000, 2010, 1980, "SELL")
        assert compute_risk_reward(*args) == compute_risk_reward(*args)

    def test_integer_inputs_equal_float_inputs(self):
        assert (
            compute_risk_reward(2000, 1990, 2020, "BUY")
            == compute_risk_reward(2000.0, 1990.0, 2020.0, "BUY")
        )


# ---------------------------------------------------------------------------
# 7. Invalid side
# ---------------------------------------------------------------------------

class TestInvalidSide:

    def test_long_raises(self):
        with pytest.raises(ValueError, match="LONG"):
            compute_risk_reward(2000, 1990, 2020, "LONG")

    def test_short_raises(self):
        with pytest.raises(ValueError, match="SHORT"):
            compute_risk_reward(2000, 2010, 1980, "SHORT")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            compute_risk_reward(2000, 1990, 2020, "")

    def test_number_as_side_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            compute_risk_reward(2000, 1990, 2020, 1)

    def test_none_as_side_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            compute_risk_reward(2000, 1990, 2020, None)


# ---------------------------------------------------------------------------
# 8. Non-finite price inputs
# ---------------------------------------------------------------------------

class TestNonFiniteInputs:

    def test_nan_entry_raises(self):
        with pytest.raises(ValueError, match="entry"):
            compute_risk_reward(float("nan"), 1990, 2020, "BUY")

    def test_nan_stop_loss_raises(self):
        with pytest.raises(ValueError, match="stop_loss"):
            compute_risk_reward(2000, float("nan"), 2020, "BUY")

    def test_nan_take_profit_raises(self):
        with pytest.raises(ValueError, match="take_profit"):
            compute_risk_reward(2000, 1990, float("nan"), "BUY")

    def test_inf_entry_raises(self):
        with pytest.raises(ValueError, match="entry"):
            compute_risk_reward(float("inf"), 1990, 2020, "BUY")

    def test_neg_inf_stop_loss_raises(self):
        with pytest.raises(ValueError, match="stop_loss"):
            compute_risk_reward(2000, float("-inf"), 2020, "BUY")

    def test_inf_take_profit_raises(self):
        with pytest.raises(ValueError, match="take_profit"):
            compute_risk_reward(2000, 1990, float("inf"), "BUY")


# ---------------------------------------------------------------------------
# 9. Non-numeric price inputs
# ---------------------------------------------------------------------------

class TestNonNumericInputs:

    def test_string_entry_raises(self):
        with pytest.raises((ValueError, TypeError)):
            compute_risk_reward("2000", 1990, 2020, "BUY")

    def test_none_entry_raises(self):
        with pytest.raises((ValueError, TypeError)):
            compute_risk_reward(None, 1990, 2020, "BUY")

    def test_list_entry_raises(self):
        with pytest.raises((ValueError, TypeError)):
            compute_risk_reward([2000], 1990, 2020, "BUY")


# ---------------------------------------------------------------------------
# 10. Invalid geometry (wrong direction)
# ---------------------------------------------------------------------------

class TestInvalidGeometry:

    def test_buy_sl_above_entry_raises(self):
        # stop_loss > entry → risk < 0
        with pytest.raises(ValueError, match="geometry"):
            compute_risk_reward(2000, 2010, 2020, "BUY")

    def test_buy_tp_below_entry_raises(self):
        # take_profit < entry → reward < 0
        with pytest.raises(ValueError, match="geometry"):
            compute_risk_reward(2000, 1990, 1980, "BUY")

    def test_sell_sl_below_entry_raises(self):
        # stop_loss < entry → risk < 0
        with pytest.raises(ValueError, match="geometry"):
            compute_risk_reward(2000, 1990, 1980, "SELL")

    def test_sell_tp_above_entry_raises(self):
        # take_profit > entry → reward < 0
        with pytest.raises(ValueError, match="geometry"):
            compute_risk_reward(2000, 2010, 2020, "SELL")

    def test_buy_entry_equals_sl_raises(self):
        # risk == 0
        with pytest.raises(ValueError, match="geometry"):
            compute_risk_reward(2000, 2000, 2020, "BUY")

    def test_sell_entry_equals_sl_raises(self):
        # risk == 0
        with pytest.raises(ValueError, match="geometry"):
            compute_risk_reward(2000, 2000, 1980, "SELL")

    def test_buy_entry_equals_tp_raises(self):
        # reward == 0
        with pytest.raises(ValueError, match="geometry"):
            compute_risk_reward(2000, 1990, 2000, "BUY")

    def test_sell_entry_equals_tp_raises(self):
        # reward == 0
        with pytest.raises(ValueError, match="geometry"):
            compute_risk_reward(2000, 2010, 2000, "SELL")


# ---------------------------------------------------------------------------
# 11. Minimum distance filter
# ---------------------------------------------------------------------------

class TestMinimumDistance:

    def test_buy_risk_below_minimum_raises(self):
        # risk = 0.005 < 0.01 — use integer-safe distances to avoid FP drift
        with pytest.raises(ValueError):
            compute_risk_reward(2000.005, 2000.0, 2020.0, "BUY")

    def test_buy_reward_below_minimum_raises(self):
        # reward = 0.005 < 0.01
        with pytest.raises(ValueError):
            compute_risk_reward(2000.0, 1990.0, 2000.005, "BUY")

    def test_sell_risk_below_minimum_raises(self):
        # risk = 0.005 < 0.01
        with pytest.raises(ValueError):
            compute_risk_reward(2000.0, 2000.005, 1990.0, "SELL")

    def test_sell_reward_below_minimum_raises(self):
        # reward = 0.005 < 0.01
        with pytest.raises(ValueError):
            compute_risk_reward(2000.0, 2010.0, 1999.995, "SELL")

    def test_risk_above_minimum_passes(self):
        # risk=0.05, reward=0.10 → rr=2.0 — both well above 0.01
        result = compute_risk_reward(2000.05, 2000.0, 2000.15, "BUY")
        assert result == pytest.approx(2.0)

    def test_reward_above_minimum_passes(self):
        # risk=0.10, reward=0.05 → rr=0.5 — both well above 0.01
        result = compute_risk_reward(2000.10, 2000.0, 2000.15, "BUY")
        assert result == pytest.approx(0.5)
