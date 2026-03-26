"""Tests for execution/position_sizing.py — compute_position_size()"""

import math
import pytest

from execution.position_sizing import compute_position_size


# ---------------------------------------------------------------------------
# 1. Spec example
# ---------------------------------------------------------------------------

class TestSpecExample:

    def test_spec_example(self):
        # account=10,000  risk=0.5%  →  risk_amount=50
        # entry=2000  sl=1990  →  sl_distance=10
        # position = 50 / 10 / 100 = 0.05
        result = compute_position_size(10_000, 2000, 1990, 0.5)
        assert result == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 2. Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:

    def test_returns_float(self):
        result = compute_position_size(10_000, 2000, 1990, 0.5)
        assert isinstance(result, float)

    def test_result_is_positive(self):
        result = compute_position_size(10_000, 2000, 1990, 0.5)
        assert result > 0.0

    def test_result_is_finite(self):
        result = compute_position_size(10_000, 2000, 1990, 0.5)
        assert math.isfinite(result)

    def test_result_is_not_rounded(self):
        # risk_amount = 10 000 * 1/100 = 100;  sl = 30;  100/30/100 = 0.03333...
        result = compute_position_size(10_000, 2000, 1970, 1.0)
        assert result == pytest.approx(100 / 30 / 100)


# ---------------------------------------------------------------------------
# 3. Correct calculation
# ---------------------------------------------------------------------------

class TestCalculation:

    def test_1_percent_risk(self):
        # risk=100, sl=10 → 100/10/100 = 0.1
        assert compute_position_size(10_000, 2000, 1990, 1.0) == pytest.approx(0.1)

    def test_2_percent_risk(self):
        # risk=200, sl=10 → 200/10/100 = 0.2
        assert compute_position_size(10_000, 2000, 1990, 2.0) == pytest.approx(0.2)

    def test_larger_sl_distance_gives_smaller_position(self):
        small_sl = compute_position_size(10_000, 2000, 1990, 1.0)   # sl=10
        large_sl = compute_position_size(10_000, 2000, 1950, 1.0)   # sl=50
        assert large_sl < small_sl

    def test_larger_risk_percent_gives_larger_position(self):
        low  = compute_position_size(10_000, 2000, 1990, 0.5)
        high = compute_position_size(10_000, 2000, 1990, 2.0)
        assert high > low

    def test_larger_account_gives_larger_position(self):
        small = compute_position_size(5_000,  2000, 1990, 1.0)
        large = compute_position_size(10_000, 2000, 1990, 1.0)
        assert large == pytest.approx(2 * small)

    def test_sl_above_entry_same_as_below(self):
        # sl_distance is always abs(entry - stop_loss) — direction agnostic
        below = compute_position_size(10_000, 2000, 1990, 1.0)
        above = compute_position_size(10_000, 2000, 2010, 1.0)
        assert below == pytest.approx(above)

    def test_integer_inputs_accepted(self):
        result = compute_position_size(10_000, 2000, 1990, 1)
        assert result == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_inputs_same_output(self):
        args = (10_000, 2000, 1990, 1.0)
        assert compute_position_size(*args) == compute_position_size(*args)

    def test_float_and_int_inputs_equal(self):
        assert (
            compute_position_size(10_000, 2000, 1990, 1)
            == compute_position_size(10_000.0, 2000.0, 1990.0, 1.0)
        )


# ---------------------------------------------------------------------------
# 5. Invalid risk_percent
# ---------------------------------------------------------------------------

class TestInvalidRiskPercent:

    def test_zero_risk_percent_raises(self):
        with pytest.raises(ValueError, match="risk_percent"):
            compute_position_size(10_000, 2000, 1990, 0.0)

    def test_negative_risk_percent_raises(self):
        with pytest.raises(ValueError, match="risk_percent"):
            compute_position_size(10_000, 2000, 1990, -1.0)

    def test_nan_risk_percent_raises(self):
        with pytest.raises(ValueError, match="risk_percent"):
            compute_position_size(10_000, 2000, 1990, float("nan"))

    def test_inf_risk_percent_raises(self):
        with pytest.raises(ValueError, match="risk_percent"):
            compute_position_size(10_000, 2000, 1990, float("inf"))


# ---------------------------------------------------------------------------
# 6. Non-finite price / balance inputs
# ---------------------------------------------------------------------------

class TestNonFiniteInputs:

    def test_nan_account_balance_raises(self):
        with pytest.raises(ValueError, match="account_balance"):
            compute_position_size(float("nan"), 2000, 1990, 1.0)

    def test_inf_account_balance_raises(self):
        with pytest.raises(ValueError, match="account_balance"):
            compute_position_size(float("inf"), 2000, 1990, 1.0)

    def test_nan_entry_raises(self):
        with pytest.raises(ValueError, match="entry"):
            compute_position_size(10_000, float("nan"), 1990, 1.0)

    def test_nan_stop_loss_raises(self):
        with pytest.raises(ValueError, match="stop_loss"):
            compute_position_size(10_000, 2000, float("nan"), 1.0)

    def test_neg_inf_stop_loss_raises(self):
        with pytest.raises(ValueError, match="stop_loss"):
            compute_position_size(10_000, 2000, float("-inf"), 1.0)


# ---------------------------------------------------------------------------
# 7. Non-numeric inputs
# ---------------------------------------------------------------------------

class TestNonNumericInputs:

    def test_string_balance_raises(self):
        with pytest.raises((ValueError, TypeError)):
            compute_position_size("10000", 2000, 1990, 1.0)

    def test_none_entry_raises(self):
        with pytest.raises((ValueError, TypeError)):
            compute_position_size(10_000, None, 1990, 1.0)

    def test_list_stop_loss_raises(self):
        with pytest.raises((ValueError, TypeError)):
            compute_position_size(10_000, 2000, [1990], 1.0)


# ---------------------------------------------------------------------------
# 8. Invalid geometry (zero / too-small SL distance)
# ---------------------------------------------------------------------------

class TestInvalidGeometry:

    def test_entry_equals_stop_loss_raises(self):
        with pytest.raises(ValueError, match="stop loss distance"):
            compute_position_size(10_000, 2000, 2000, 1.0)

    def test_sl_distance_below_minimum_raises(self):
        # distance = 0.005 < 0.01
        with pytest.raises(ValueError, match="stop loss distance"):
            compute_position_size(10_000, 2000.005, 2000.0, 1.0)

    def test_sl_distance_above_minimum_passes(self):
        # distance = 0.05 — safely above threshold
        result = compute_position_size(10_000, 2000.05, 2000.0, 1.0)
        assert result > 0.0
