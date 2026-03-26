"""Tests for execution/signal_engine.py — generate_signal() and generate_signal_dict()

Strategy
--------
The signal_engine orchestrates 7 analysis stages and 5 execution stages;
end-to-end tests with real data would be expensive and brittle.  Instead
we mock the boundary of each downstream module and test the orchestration
logic directly:

    * Does generate_signal return None at each failure point?
    * Does a full happy-path call reach format_trade_signal and return its output?
    * Is the output deterministic across repeated calls?
    * Are context fields correctly derived (EMA alignment, liquidity, structure)?
    * Does generate_signal_dict return the raw dict (not a string)?

Mocking approach: unittest.mock.patch replaces each imported function with a
controllable MagicMock for the duration of each test.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from execution.signal_engine import (
    _CONFIDENCE_SCORE,
    _EXECUTION_TIMEFRAME,
    _all_finite,
    _safe_float,
    generate_signal,
    generate_signal_dict,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _minimal_df(n: int = 30, *, bullish: bool = True) -> pd.DataFrame:
    """Minimal DataFrame with the columns consumed by generate_signal
    AFTER the full pipeline has already run.  Used when the pipeline
    stages are mocked out."""
    close_base = 2000.0 if bullish else 2050.0
    closes = [close_base + (i * (0.5 if bullish else -0.5)) for i in range(n)]
    highs  = [c + 2.0 for c in closes]
    lows   = [c - 2.0 for c in closes]

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="5min"),
        "open":    [c - 0.5 for c in closes],
        "high":    highs,
        "low":     lows,
        "close":   closes,
        "volume":  [1000.0] * n,
        # Indicator columns (added by compute_indicators)
        "ATR":     [12.0] * n,
        "EMA_20":  [2010.0 if bullish else 2040.0] * n,
        "EMA_50":  [2005.0 if bullish else 2045.0] * n,
        "EMA_200": [2000.0 if bullish else 2050.0] * n,
        "RSI":     [62.0  if bullish else 38.0]    * n,
        "VWAP":    [2008.0 if bullish else 2042.0] * n,
        # Swing columns (detect_swings)
        "swing_high": [False] * n,
        "swing_low":  [False] * n,
        # Equal-level columns (detect_equal_levels)
        "equal_high": [False] * n,
        "equal_low":  [False] * n,
        # Market structure (detect_market_structure)
        "event_type":     [""] * (n - 1) + ["CHOCH"],
        "trend_state":    ["BULLISH" if bullish else "BEARISH"] * n,
        "protected_high": [2060.0] * n,
        "protected_low":  [1990.0] * n,
        # Liquidity columns (detect_liquidity_pools)
        "liquidity_pool_high": [False] * n,
        "liquidity_pool_low":  [False] * n,
        # Liquidity sweep columns (detect_liquidity_sweeps)
        "liquidity_sweep_high": [False] * n,
        "liquidity_sweep_low":  [False] * (n - 1) + [True],
        # Order block columns (detect_order_blocks)
        "bullish_order_block": [False] * (n - 1) + [True],
        "bearish_order_block": [False] * n,
        "ob_high": [float("nan")] * n,
        "ob_low":  [float("nan")] * n,
    })
    return df


_MOCKED_PIPELINE_TARGETS = [
    "execution.signal_engine.compute_indicators",
    "execution.signal_engine.detect_swings",
    "execution.signal_engine.detect_equal_levels",
    "execution.signal_engine.detect_market_structure",
    "execution.signal_engine.detect_liquidity_pools",
    "execution.signal_engine.detect_liquidity_sweeps",
    "execution.signal_engine.detect_order_blocks",
]


def _patch_pipeline(df: pd.DataFrame):
    """Return a list of patch objects that make every analysis stage a no-op
    (returning *df* unchanged)."""
    mocks = []
    for target in _MOCKED_PIPELINE_TARGETS:
        m = patch(target, side_effect=lambda x, *a, **kw: x)
        mocks.append(m)
    return mocks


def _start_pipeline_mocks(df: pd.DataFrame):
    """Activate all pipeline mocks and return the started mock objects."""
    started = []
    for target in _MOCKED_PIPELINE_TARGETS:
        m = patch(target, side_effect=lambda x, *a, **kw: x)
        started.append(m.start())
    return started


class _PipelineMockContext:
    """Context manager that mocks the entire analysis pipeline."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._patches = []

    def __enter__(self):
        for target in _MOCKED_PIPELINE_TARGETS:
            p = patch(target, side_effect=lambda x, *a, **kw: x)
            self._patches.append(p)
            p.start()
        return self

    def __exit__(self, *args):
        for p in self._patches:
            p.stop()


# ---------------------------------------------------------------------------
# 1. Null / empty input
# ---------------------------------------------------------------------------

class TestNullInput:
    def test_none_market_data_key_returns_none(self):
        result = generate_signal({})
        assert result is None

    def test_m5_is_none_returns_none(self):
        result = generate_signal({"M5": None})
        assert result is None

    def test_m5_empty_dataframe_returns_none(self):
        result = generate_signal({"M5": pd.DataFrame()})
        assert result is None

    def test_missing_m5_key_returns_none(self):
        result = generate_signal({"H1": _minimal_df()})
        assert result is None


# ---------------------------------------------------------------------------
# 2. Pipeline failure (ValueError from missing columns)
# ---------------------------------------------------------------------------

class TestPipelineFailure:
    def test_pipeline_valueerror_returns_none(self):
        # A single-row df with no required columns will cause ValueError in
        # compute_indicators.
        df = pd.DataFrame({"close": [2000.0]})
        result = generate_signal({"M5": df})
        assert result is None

    def test_pipeline_keyerror_returns_none(self):
        with patch(
            "execution.signal_engine.compute_indicators",
            side_effect=KeyError("close"),
        ):
            result = generate_signal({"M5": _minimal_df()})
        assert result is None


# ---------------------------------------------------------------------------
# 3. TRANSITION bias → no signal
# ---------------------------------------------------------------------------

class TestTransitionBias:
    def test_transition_trend_state_returns_none(self):
        df = _minimal_df()
        df["trend_state"] = "TRANSITION"
        with _PipelineMockContext(df):
            result = generate_signal({"M5": df})
        assert result is None

    def test_empty_trend_state_returns_none(self):
        df = _minimal_df()
        df["trend_state"] = ""
        with _PipelineMockContext(df):
            result = generate_signal({"M5": df})
        assert result is None


# ---------------------------------------------------------------------------
# 4. Potential setup rejection → no signal
# ---------------------------------------------------------------------------

class TestPotentialSetupRejection:
    def test_invalid_potential_setup_returns_none(self):
        df = _minimal_df()
        with _PipelineMockContext(df):
            with patch(
                "execution.signal_engine.evaluate_potential_setup",
                return_value={"is_valid_setup": False, "reason": "test"},
            ):
                result = generate_signal({"M5": df})
        assert result is None


# ---------------------------------------------------------------------------
# 5. Trade setup failure → no signal
# ---------------------------------------------------------------------------

class TestTradeSetupFailure:
    def test_build_trade_setup_none_returns_none(self):
        df = _minimal_df()
        with _PipelineMockContext(df):
            with patch(
                "execution.signal_engine.evaluate_potential_setup",
                return_value={
                    "is_valid_setup": True,
                    "setup_type":     "REVERSAL",
                    "confidence":     "HIGH",
                    "waiting_for":    "CHOCH_CONFIRMATION",
                    "reason":         "ok",
                },
            ):
                with patch(
                    "execution.signal_engine.build_trade_setup",
                    return_value=None,
                ):
                    result = generate_signal({"M5": df})
        assert result is None


# ---------------------------------------------------------------------------
# 6. Risk manager rejection → no signal
# ---------------------------------------------------------------------------

class TestRiskRejection:
    def test_risk_not_approved_returns_none(self):
        df = _minimal_df()
        _trade = {
            "entry": 2015.0, "stop_loss": 1990.0,
            "take_profit": 2055.0, "risk_reward": 2.5, "invalidation": 1990.0,
        }
        with _PipelineMockContext(df):
            with patch(
                "execution.signal_engine.evaluate_potential_setup",
                return_value={
                    "is_valid_setup": True, "setup_type": "REVERSAL",
                    "confidence": "HIGH", "waiting_for": "CHOCH_CONFIRMATION",
                    "reason": "ok",
                },
            ):
                with patch("execution.signal_engine.build_trade_setup", return_value=_trade):
                    with patch(
                        "execution.signal_engine.evaluate_risk",
                        return_value=None,
                    ):
                        result = generate_signal({"M5": df})
        assert result is None


# ---------------------------------------------------------------------------
# 7. Happy path — full valid signal
# ---------------------------------------------------------------------------

class TestHappyPath:
    def _run_happy_path(self, df: pd.DataFrame) -> str | None:
        _trade = {
            "entry": 2015.0, "stop_loss": 1990.0,
            "take_profit": 2055.0, "risk_reward": 2.5, "invalidation": 1990.0,
        }
        with _PipelineMockContext(df):
            with patch(
                "execution.signal_engine.evaluate_potential_setup",
                return_value={
                    "is_valid_setup": True, "setup_type": "REVERSAL",
                    "confidence": "HIGH", "waiting_for": "CHOCH_CONFIRMATION",
                    "reason": "ok",
                },
            ):
                with patch("execution.signal_engine.build_trade_setup", return_value=_trade):
                    with patch(
                        "execution.signal_engine.evaluate_risk",
                        return_value=_trade,
                    ):
                        return generate_signal({"M5": df})

    def test_happy_path_returns_string(self):
        result = self._run_happy_path(_minimal_df())
        assert isinstance(result, str)

    def test_happy_path_not_no_trade(self):
        result = self._run_happy_path(_minimal_df())
        assert result != "No trade opportunity"

    def test_happy_path_contains_xauusd(self):
        result = self._run_happy_path(_minimal_df())
        assert "XAUUSD" in result

    def test_happy_path_contains_m5(self):
        result = self._run_happy_path(_minimal_df())
        assert "M5" in result

    def test_happy_path_contains_bias(self):
        result = self._run_happy_path(_minimal_df(bullish=True))
        assert "BULLISH" in result

    def test_happy_path_contains_entry(self):
        result = self._run_happy_path(_minimal_df())
        assert "Entry" in result

    def test_happy_path_contains_stop_loss(self):
        result = self._run_happy_path(_minimal_df())
        assert "Stop Loss" in result

    def test_happy_path_contains_take_profit(self):
        result = self._run_happy_path(_minimal_df())
        assert "Take Profit" in result


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_same_output(self):
        df = _minimal_df()
        _trade = {
            "entry": 2015.0, "stop_loss": 1990.0,
            "take_profit": 2055.0, "risk_reward": 2.5, "invalidation": 1990.0,
        }
        with _PipelineMockContext(df):
            with patch(
                "execution.signal_engine.evaluate_potential_setup",
                return_value={
                    "is_valid_setup": True, "setup_type": "REVERSAL",
                    "confidence": "HIGH", "waiting_for": "CHOCH_CONFIRMATION",
                    "reason": "ok",
                },
            ):
                with patch("execution.signal_engine.build_trade_setup", return_value=_trade):
                    with patch(
                        "execution.signal_engine.evaluate_risk",
                        return_value=_trade,
                    ):
                        r1 = generate_signal({"M5": df})
                        r2 = generate_signal({"M5": df})
        assert r1 == r2


# ---------------------------------------------------------------------------
# 9. Context derivation helpers — EMA alignment
# ---------------------------------------------------------------------------

class TestEMAAlignment:
    def test_bullish_ema_stack_detected(self):
        df = _minimal_df(bullish=True)
        # EMA_20=2010 > EMA_50=2005 > EMA_200=2000 → BULLISH alignment
        captured: list[dict] = []

        def _capture_setup(signal, context):
            captured.append(context)
            return {"is_valid_setup": False, "reason": "test"}

        with _PipelineMockContext(df):
            with patch("execution.signal_engine.evaluate_potential_setup", side_effect=_capture_setup):
                generate_signal({"M5": df})

        assert len(captured) == 1
        assert captured[0]["ema_alignment"] == "BULLISH"

    def test_bearish_ema_stack_detected(self):
        df = _minimal_df(bullish=False)
        # EMA_20=2040 < EMA_50=2045 < EMA_200=2050 → BEARISH alignment
        captured: list[dict] = []

        def _capture_setup(signal, context):
            captured.append(context)
            return {"is_valid_setup": False, "reason": "test"}

        with _PipelineMockContext(df):
            with patch("execution.signal_engine.evaluate_potential_setup", side_effect=_capture_setup):
                generate_signal({"M5": df})

        assert len(captured) == 1
        assert captured[0]["ema_alignment"] == "BEARISH"

    def test_mixed_ema_detected(self):
        df = _minimal_df()
        df["EMA_20"]  = 2010.0
        df["EMA_50"]  = 2012.0  # EMA_20 < EMA_50 → not a clean stack
        df["EMA_200"] = 2008.0
        captured: list[dict] = []

        def _capture(signal, context):
            captured.append(context)
            return {"is_valid_setup": False, "reason": "test"}

        with _PipelineMockContext(df):
            with patch("execution.signal_engine.evaluate_potential_setup", side_effect=_capture):
                generate_signal({"M5": df})

        if captured:
            assert captured[0]["ema_alignment"] == "MIXED"


# ---------------------------------------------------------------------------
# 10. Context derivation — liquidity mapping
# ---------------------------------------------------------------------------

class TestLiquidityMapping:
    def _get_context(self, df: pd.DataFrame) -> dict | None:
        captured: list[dict] = []

        def _capture(signal, context):
            captured.append(context)
            return {"is_valid_setup": False, "reason": "test"}

        with _PipelineMockContext(df):
            with patch("execution.signal_engine.evaluate_potential_setup", side_effect=_capture):
                generate_signal({"M5": df})

        return captured[0] if captured else None

    def test_sweep_high_maps_to_sweep(self):
        df = _minimal_df()
        df["liquidity_sweep_high"] = [False] * (len(df) - 1) + [True]
        ctx = self._get_context(df)
        assert ctx is not None
        assert ctx["liquidity"] == "SWEEP"

    def test_sweep_low_maps_to_sweep(self):
        df = _minimal_df()
        df["liquidity_sweep_low"] = [False] * (len(df) - 1) + [True]
        ctx = self._get_context(df)
        assert ctx is not None
        assert ctx["liquidity"] == "SWEEP"

    def test_pool_without_sweep_maps_to_equal(self):
        df = _minimal_df()
        df["liquidity_sweep_high"] = False
        df["liquidity_sweep_low"]  = False
        df["liquidity_pool_high"]  = [False] * (len(df) - 1) + [True]
        ctx = self._get_context(df)
        assert ctx is not None
        assert ctx["liquidity"] == "EQUAL"

    def test_no_liquidity_maps_to_none(self):
        df = _minimal_df()
        df["liquidity_sweep_high"] = False
        df["liquidity_sweep_low"]  = False
        df["liquidity_pool_high"]  = False
        df["liquidity_pool_low"]   = False
        ctx = self._get_context(df)
        # Missing liquidity → potential_setup will reject, but context key is NONE
        if ctx is not None:
            assert ctx["liquidity"] == "NONE"


# ---------------------------------------------------------------------------
# 11. Context derivation — market structure mapping
# ---------------------------------------------------------------------------

class TestStructureMapping:
    def _get_context(self, event_type: str) -> dict | None:
        df = _minimal_df()
        df["event_type"] = event_type
        captured: list[dict] = []

        def _capture(signal, context):
            captured.append(context)
            return {"is_valid_setup": False, "reason": "test"}

        with _PipelineMockContext(df):
            with patch("execution.signal_engine.evaluate_potential_setup", side_effect=_capture):
                generate_signal({"M5": df})

        return captured[0] if captured else None

    def test_bos_confirmed_mapped(self):
        ctx = self._get_context("BOS_CONFIRMED")
        assert ctx is not None and ctx["market_structure"] == "BOS_CONFIRMED"

    def test_choch_mapped(self):
        ctx = self._get_context("CHOCH")
        assert ctx is not None and ctx["market_structure"] == "CHOCH"

    def test_break_mapped(self):
        ctx = self._get_context("BREAK")
        assert ctx is not None and ctx["market_structure"] == "BREAK"

    def test_liquidity_sweep_event_maps_to_none(self):
        # LIQUIDITY_SWEEP is a liquidity event, not a structural one
        ctx = self._get_context("LIQUIDITY_SWEEP")
        assert ctx is not None and ctx["market_structure"] == "NONE"

    def test_empty_event_maps_to_none(self):
        ctx = self._get_context("")
        assert ctx is not None and ctx["market_structure"] == "NONE"


# ---------------------------------------------------------------------------
# 12. Confidence score mapping
# ---------------------------------------------------------------------------

class TestConfidenceScoreMapping:
    def test_high_maps_to_85(self):
        assert _CONFIDENCE_SCORE["HIGH"] == 85.0

    def test_medium_maps_to_70(self):
        assert _CONFIDENCE_SCORE["MEDIUM"] == 70.0

    def test_high_passes_risk_manager_gate(self):
        # Risk manager rejects confidence < 70; HIGH=85 must pass
        assert _CONFIDENCE_SCORE["HIGH"] >= 70.0

    def test_medium_passes_risk_manager_gate(self):
        assert _CONFIDENCE_SCORE["MEDIUM"] >= 70.0


# ---------------------------------------------------------------------------
# 13. Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_safe_float_returns_value(self):
        row = {"ATR": 12.5}
        assert _safe_float(row, "ATR") == 12.5

    def test_safe_float_missing_key_returns_default(self):
        assert math.isnan(_safe_float({}, "MISSING"))

    def test_safe_float_custom_default(self):
        assert _safe_float({}, "X", default=99.0) == 99.0

    def test_all_finite_true(self):
        assert _all_finite(1.0, 2.0, 3.0) is True

    def test_all_finite_with_nan(self):
        assert _all_finite(1.0, float("nan"), 3.0) is False

    def test_all_finite_with_inf(self):
        assert _all_finite(1.0, float("inf"), 3.0) is False

    def test_execution_timeframe_is_m5(self):
        assert _EXECUTION_TIMEFRAME == "M5"


# ---------------------------------------------------------------------------
# 14. ATR safety guard — atr <= 0 returns None
# ---------------------------------------------------------------------------

class TestATRGuard:
    def test_zero_atr_returns_none(self):
        df = _minimal_df()
        df["ATR"] = 0.0
        with _PipelineMockContext(df):
            result = generate_signal({"M5": df})
        assert result is None

    def test_negative_atr_returns_none(self):
        df = _minimal_df()
        df["ATR"] = -1.0
        with _PipelineMockContext(df):
            result = generate_signal({"M5": df})
        assert result is None

    def test_positive_atr_proceeds(self):
        # ATR > 0 should NOT be blocked by the guard; setup rejection is fine
        df = _minimal_df()
        df["ATR"] = 12.0
        captured: list[dict] = []

        def _capture(signal, context):
            captured.append(context)
            return {"is_valid_setup": False, "reason": "test"}

        with _PipelineMockContext(df):
            with patch("execution.signal_engine.evaluate_potential_setup", side_effect=_capture):
                generate_signal({"M5": df})

        assert len(captured) == 1  # made it past the ATR guard


# ---------------------------------------------------------------------------
# 15. Structure-based high/low (protected_high/low used when present)
# ---------------------------------------------------------------------------

class TestStructureHighLow:
    def _get_context(self, df: pd.DataFrame) -> dict | None:
        captured: list[dict] = []

        def _capture(signal, context):
            captured.append(context)
            return {"is_valid_setup": False, "reason": "test"}

        with _PipelineMockContext(df):
            with patch("execution.signal_engine.evaluate_potential_setup", side_effect=_capture):
                generate_signal({"M5": df})

        return captured[0] if captured else None

    def test_protected_high_used_when_present(self):
        df = _minimal_df()
        # protected_high = 2060 is already set in _minimal_df; verify it's used
        ctx = self._get_context(df)
        assert ctx is not None
        assert ctx["recent_high"] == 2060.0

    def test_protected_low_used_when_present(self):
        df = _minimal_df()
        # protected_low = 1990 is already set in _minimal_df; verify it's used
        ctx = self._get_context(df)
        assert ctx is not None
        assert ctx["recent_low"] == 1990.0

    def test_fallback_to_20bar_window_without_protected_high(self):
        df = _minimal_df()
        df["protected_high"] = float("nan")
        ctx = self._get_context(df)
        assert ctx is not None
        # fallback: max of last 20 highs
        expected = float(df["high"].iloc[-20:].max())
        assert ctx["recent_high"] == pytest.approx(expected)

    def test_fallback_to_20bar_window_without_protected_low(self):
        df = _minimal_df()
        df["protected_low"] = float("nan")
        ctx = self._get_context(df)
        assert ctx is not None
        expected = float(df["low"].iloc[-20:].min())
        assert ctx["recent_low"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 16. Order block proximity — price within ATR of OB level
# ---------------------------------------------------------------------------

class TestOBProximity:
    def _get_context(self, df: pd.DataFrame) -> dict | None:
        captured: list[dict] = []

        def _capture(signal, context):
            captured.append(context)
            return {"is_valid_setup": False, "reason": "test"}

        with _PipelineMockContext(df):
            with patch("execution.signal_engine.evaluate_potential_setup", side_effect=_capture):
                generate_signal({"M5": df})

        return captured[0] if captured else None

    def test_bullish_ob_within_atr_is_near(self):
        df = _minimal_df(bullish=True)
        close = float(df["close"].iloc[-1])
        atr   = float(df["ATR"].iloc[-1])
        # Place a valid OB with price inside [ob_low, ob_high]
        df["ob_low"]  = float("nan")
        df["ob_high"] = float("nan")
        df.loc[df.index[-1], "ob_low"]  = close - (atr * 0.5)
        df.loc[df.index[-1], "ob_high"] = close + (atr * 0.5)
        ctx = self._get_context(df)
        assert ctx is not None
        assert ctx["near_order_block"] is True

    def test_bullish_ob_beyond_atr_is_not_near(self):
        df = _minimal_df(bullish=True)
        close = float(df["close"].iloc[-1])
        atr   = float(df["ATR"].iloc[-1])
        # OB entirely below close: price is outside [ob_low, ob_high]
        df["ob_low"]  = float("nan")
        df["ob_high"] = float("nan")
        df.loc[df.index[-1], "ob_low"]  = close - (atr * 5.0)
        df.loc[df.index[-1], "ob_high"] = close - (atr * 3.0)
        ctx = self._get_context(df)
        assert ctx is not None
        assert ctx["near_order_block"] is False

    def test_bearish_ob_within_atr_is_near(self):
        df = _minimal_df(bullish=False)
        close = float(df["close"].iloc[-1])
        atr   = float(df["ATR"].iloc[-1])
        # Place a valid OB with price inside [ob_low, ob_high]
        df["ob_high"] = float("nan")
        df["ob_low"]  = float("nan")
        df.loc[df.index[-1], "ob_high"] = close + (atr * 0.5)
        df.loc[df.index[-1], "ob_low"]  = close - (atr * 0.5)
        ctx = self._get_context(df)
        assert ctx is not None
        assert ctx["near_order_block"] is True

    def test_no_ob_columns_gives_false(self):
        df = _minimal_df(bullish=True)
        df.drop(columns=["ob_low", "ob_high"], inplace=True)
        ctx = self._get_context(df)
        assert ctx is not None
        assert ctx["near_order_block"] is False


# ---------------------------------------------------------------------------
# 17. Real-RR probability recomputation after trade is built
# ---------------------------------------------------------------------------

class TestRealRRProbability:
    def test_compute_probability_called_with_real_rr(self):
        """compute_probability must be called a second time after build_trade_setup
        using trade['risk_reward'], not the approximate RR."""
        df = _minimal_df()
        _trade = {
            "entry": 2015.0, "stop_loss": 1990.0,
            "take_profit": 2055.0, "risk_reward": 3.1, "invalidation": 1990.0,
        }
        real_rr_calls: list[float] = []

        _call_count = [0]

        def _prob(signal, context):
            _call_count[0] += 1
            if _call_count[0] > 1:
                real_rr_calls.append(signal.get("risk_reward"))
            return {"probability": 75, "grade": "B", "confidence": "HIGH"}

        with _PipelineMockContext(df):
            with patch("execution.signal_engine.compute_probability", side_effect=_prob):
                with patch(
                    "execution.signal_engine.evaluate_potential_setup",
                    return_value={
                        "is_valid_setup": True, "setup_type": "REVERSAL",
                        "confidence": "HIGH", "waiting_for": None, "reason": "ok",
                    },
                ):
                    with patch("execution.signal_engine.build_trade_setup", return_value=_trade):
                        with patch(
                            "execution.signal_engine.evaluate_risk",
                            return_value=_trade,
                        ):
                            generate_signal({"M5": df})

        # The second call must use the trade's exact RR
        assert len(real_rr_calls) >= 1
        assert real_rr_calls[0] == pytest.approx(3.1)


# ---------------------------------------------------------------------------
# 18. Robustness — bad ATR / RSI / EMA / OHLC inputs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 19. generate_signal_dict — raw dict contract
# ---------------------------------------------------------------------------

class TestGenerateSignalDict:
    """generate_signal_dict() must return the same pipeline output as
    generate_signal() but as a raw dict rather than a formatted string."""

    _TRADE = {
        "entry": 2015.0, "stop_loss": 1990.0,
        "take_profit": 2055.0, "risk_reward": 2.5, "invalidation": 1990.0,
    }
    _SETUP = {
        "is_valid_setup": True, "setup_type": "REVERSAL",
        "confidence": "HIGH", "waiting_for": None, "reason": "ok",
    }

    def _run_dict(self, df):
        with _PipelineMockContext(df):
            with patch("execution.signal_engine.evaluate_potential_setup",
                       return_value=self._SETUP):
                with patch("execution.signal_engine.build_trade_setup",
                           return_value=self._TRADE):
                    with patch("execution.signal_engine.evaluate_risk",
                               return_value=self._TRADE):
                        return generate_signal_dict({"M5": df})

    def test_returns_dict_not_string(self):
        result = self._run_dict(_minimal_df())
        assert isinstance(result, dict)

    def test_returns_none_on_pipeline_failure(self):
        result = generate_signal_dict({})
        assert result is None

    def test_returns_none_on_none_m5(self):
        result = generate_signal_dict({"M5": None})
        assert result is None

    def test_has_entry_key(self):
        assert "entry" in self._run_dict(_minimal_df())

    def test_has_stop_loss_key(self):
        assert "stop_loss" in self._run_dict(_minimal_df())

    def test_has_take_profit_key(self):
        assert "take_profit" in self._run_dict(_minimal_df())

    def test_has_risk_reward_key(self):
        assert "risk_reward" in self._run_dict(_minimal_df())

    def test_has_bias_key(self):
        assert "bias" in self._run_dict(_minimal_df())

    def test_has_confidence_key(self):
        assert "confidence" in self._run_dict(_minimal_df())

    def test_has_atr_key(self):
        assert "atr" in self._run_dict(_minimal_df())

    def test_has_pair_key(self):
        assert "pair" in self._run_dict(_minimal_df())

    def test_has_timeframe_key(self):
        assert "timeframe" in self._run_dict(_minimal_df())

    def test_has_mtf_bias_key(self):
        assert "mtf_bias" in self._run_dict(_minimal_df())

    def test_has_bias_strength_key(self):
        assert "bias_strength" in self._run_dict(_minimal_df())

    def test_has_context_key(self):
        assert "context" in self._run_dict(_minimal_df())

    def test_generate_signal_wraps_dict(self):
        """generate_signal() must return str iff generate_signal_dict() returns dict."""
        dict_result = self._run_dict(_minimal_df())
        assert isinstance(dict_result, dict)

        with _PipelineMockContext(_minimal_df()):
            with patch("execution.signal_engine.evaluate_potential_setup",
                       return_value=self._SETUP):
                with patch("execution.signal_engine.build_trade_setup",
                           return_value=self._TRADE):
                    with patch("execution.signal_engine.evaluate_risk",
                               return_value=self._TRADE):
                        str_result = generate_signal({"M5": _minimal_df()})
        assert isinstance(str_result, str)

    def test_returns_none_when_risk_rejected(self):
        with _PipelineMockContext(_minimal_df()):
            with patch("execution.signal_engine.evaluate_potential_setup",
                       return_value=self._SETUP):
                with patch("execution.signal_engine.build_trade_setup",
                           return_value=self._TRADE):
                    with patch("execution.signal_engine.evaluate_risk",
                               return_value=None):
                        result = generate_signal_dict({"M5": _minimal_df()})
        assert result is None


class TestRobustness:
    """Verify that invalid or extreme indicator values are caught before
    any downstream module is invoked."""

    def _run(self, df: pd.DataFrame):
        with _PipelineMockContext(df):
            return generate_signal({"M5": df})

    def _captures(self, df: pd.DataFrame) -> list[dict]:
        """Return context dicts captured by evaluate_potential_setup."""
        captured: list[dict] = []

        def _capture(signal, context):
            captured.append(context)
            return {"is_valid_setup": False, "reason": "test"}

        with _PipelineMockContext(df):
            with patch("execution.signal_engine.evaluate_potential_setup",
                       side_effect=_capture):
                generate_signal({"M5": df})
        return captured

    # --- ATR ---

    def test_atr_nan_returns_none(self):
        df = _minimal_df()
        df["ATR"] = float("nan")
        assert self._run(df) is None

    def test_atr_zero_returns_none(self):
        df = _minimal_df()
        df["ATR"] = 0.0
        assert self._run(df) is None

    def test_atr_very_small_returns_none(self):
        # ATR < 0.1 treated as too-low volatility
        df = _minimal_df()
        df["ATR"] = 0.01
        assert self._run(df) is None

    def test_atr_very_large_capped_to_100(self):
        # ATR = 500 must be capped to 100; pipeline continues past ATR guard
        df = _minimal_df()
        df["ATR"] = 500.0
        caps = self._captures(df)
        assert len(caps) == 1
        assert caps[0]["atr"] == pytest.approx(100.0)

    # --- RSI ---

    def test_rsi_nan_returns_none(self):
        df = _minimal_df()
        df["RSI"] = float("nan")
        assert self._run(df) is None

    def test_rsi_below_zero_returns_none(self):
        df = _minimal_df()
        df["RSI"] = -1.0
        assert self._run(df) is None

    def test_rsi_above_100_returns_none(self):
        df = _minimal_df()
        df["RSI"] = 101.0
        assert self._run(df) is None

    # --- EMA ---

    def test_ema_20_nan_returns_none(self):
        df = _minimal_df()
        df["EMA_20"] = float("nan")
        assert self._run(df) is None

    def test_ema_50_nan_returns_none(self):
        df = _minimal_df()
        df["EMA_50"] = float("nan")
        assert self._run(df) is None

    def test_ema_200_nan_returns_none(self):
        df = _minimal_df()
        df["EMA_200"] = float("nan")
        assert self._run(df) is None

    # --- OHLC ---

    def test_invalid_ohlc_high_below_close_returns_none(self):
        df = _minimal_df()
        # Force high below close to violate high >= max(open, close)
        df.loc[df.index[-1], "high"] = float(df["close"].iloc[-1]) - 5.0
        assert self._run(df) is None

    def test_invalid_ohlc_low_above_open_returns_none(self):
        df = _minimal_df()
        # Force low above open to violate low <= min(open, close)
        df.loc[df.index[-1], "low"] = float(df["open"].iloc[-1]) + 5.0
        assert self._run(df) is None

    def test_invalid_ohlc_high_below_low_returns_none(self):
        df = _minimal_df()
        # Force high below low to violate high >= low
        df.loc[df.index[-1], "high"] = float(df["low"].iloc[-1]) - 1.0
        assert self._run(df) is None

    def test_valid_ohlc_and_indicators_proceed(self):
        # Default _minimal_df has valid OHLC and valid indicators;
        # the function must progress past all safety checks.
        df = _minimal_df()
        caps = self._captures(df)
        assert len(caps) == 1  # reached evaluate_potential_setup
