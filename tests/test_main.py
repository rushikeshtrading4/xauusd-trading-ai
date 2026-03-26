"""Tests for main.py — lifecycle, macro helpers, run_cycle, auto-save, summary."""

import datetime
import logging
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Minimal signal dict that satisfies TradeLogger.log_signal() requirements
# ---------------------------------------------------------------------------

_VALID_SIGNAL = {
    "pair":         "XAUUSD",
    "timeframe":    "M5",
    "bias":         "BUY",
    "entry":        2000.0,
    "stop_loss":    1990.0,
    "take_profit":  2020.0,
    "risk_reward":  2.0,
    "confidence":   80.0,
    "invalidation": 1990.0,
    "atr":          10.0,
    "mtf_bias":     "BULLISH",
    "bias_strength":"STRONG",
    "context":      "TREND",
}


# ---------------------------------------------------------------------------
# Helper — import main without executing module-level side-effects
# (logging setup + singleton construction) each time.
# We patch the heavy I/O operations before the first import.
# ---------------------------------------------------------------------------

def _import_main():
    """Return the main module, importing it fresh if not already cached."""
    if "main" not in sys.modules:
        # FileHandler must return a real handler (NullHandler) so that
        # logging.callHandlers() doesn't fail on hdlr.level comparisons.
        with patch("os.makedirs"), \
             patch("logging.FileHandler",
                   side_effect=lambda *a, **kw: logging.NullHandler()), \
             patch("data.macro_data.MacroDataHandler"), \
             patch("execution.trade_logger.TradeLogger"):
            import main  # noqa: PLC0415
    return sys.modules["main"]


# ---------------------------------------------------------------------------
# 1. TestSetupLogging
# ---------------------------------------------------------------------------

class TestSetupLogging(unittest.TestCase):
    """_setup_logging() creates the logs/ directory and does not crash on re-call."""

    def setUp(self):
        self.m = _import_main()
        # Snapshot root handler list so tearDown can restore it.  This prevents
        # re-calling _setup_logging() from leaving extra (or mock) handlers
        # in the root logger that would break tests in other classes.
        self._root_before = list(logging.getLogger().handlers)

    def tearDown(self):
        logging.getLogger().handlers = self._root_before

    def test_creates_logs_directory(self):
        with patch("os.makedirs") as mock_makedirs, \
             patch("logging.FileHandler",
                   side_effect=lambda *a, **kw: logging.NullHandler()):
            self.m._setup_logging()
        mock_makedirs.assert_called_with("logs", exist_ok=True)

    def test_no_crash_on_second_call(self):
        """Calling _setup_logging() twice must not raise."""
        with patch("os.makedirs"), \
             patch("logging.FileHandler",
                   side_effect=lambda *a, **kw: logging.NullHandler()):
            try:
                self.m._setup_logging()
                self.m._setup_logging()
            except Exception as exc:
                self.fail(f"_setup_logging() raised on second call: {exc}")


# ---------------------------------------------------------------------------
# 2. TestUpdateMacroState
# ---------------------------------------------------------------------------

class TestUpdateMacroState(unittest.TestCase):
    """update_macro_state() delegates to _macro_handler and derives gold_sentiment."""

    def setUp(self):
        self.m = _import_main()
        # Replace the module singleton with a fresh mock for each test
        self.mock_handler = MagicMock()
        self.mock_handler.state.gold_sentiment = "BEARISH"
        self._orig = self.m._macro_handler
        self.m._macro_handler = self.mock_handler

    def tearDown(self):
        self.m._macro_handler = self._orig

    def test_calls_update_dxy_and_yields(self):
        self.m.update_macro_state(dxy_trend="RISING", yield_trend="RISING")
        self.mock_handler.update_dxy.assert_called_once_with("RISING")
        self.mock_handler.update_yields.assert_called_once_with("RISING")

    def test_neutral_trends(self):
        self.m.update_macro_state()
        self.mock_handler.update_dxy.assert_called_once_with("NEUTRAL")
        self.mock_handler.update_yields.assert_called_once_with("NEUTRAL")

    def test_falling_trends(self):
        self.m.update_macro_state(dxy_trend="FALLING", yield_trend="FALLING")
        self.mock_handler.update_dxy.assert_called_once_with("FALLING")
        self.mock_handler.update_yields.assert_called_once_with("FALLING")

    def test_gold_sentiment_logged(self):
        self.mock_handler.state.gold_sentiment = "BULLISH"
        # Should not raise even if gold_sentiment is overridden
        self.m.update_macro_state(dxy_trend="FALLING", yield_trend="FALLING")
        self.assertEqual(self.mock_handler.state.gold_sentiment, "BULLISH")


# ---------------------------------------------------------------------------
# 3. TestAddMacroEvent
# ---------------------------------------------------------------------------

class TestAddMacroEvent(unittest.TestCase):
    """add_macro_event() registers the event and increases the event list."""

    def setUp(self):
        self.m = _import_main()
        self.mock_handler = MagicMock()
        self._orig = self.m._macro_handler
        self.m._macro_handler = self.mock_handler

    def tearDown(self):
        self.m._macro_handler = self._orig

    def test_calls_add_event(self):
        dt = datetime.datetime(2025, 1, 15, 13, 30, tzinfo=datetime.timezone.utc)
        self.m.add_macro_event("NFP", dt, impact="HIGH")
        self.mock_handler.add_event.assert_called_once_with("NFP", dt, impact="HIGH")

    def test_default_impact_is_high(self):
        dt = datetime.datetime(2025, 2, 5, 15, 0, tzinfo=datetime.timezone.utc)
        self.m.add_macro_event("CPI", dt)
        _, kwargs = self.mock_handler.add_event.call_args
        self.assertEqual(kwargs.get("impact"), "HIGH")

    def test_low_impact_forwarded(self):
        dt = datetime.datetime(2025, 3, 1, 9, 0, tzinfo=datetime.timezone.utc)
        self.m.add_macro_event("Housing", dt, impact="LOW")
        _, kwargs = self.mock_handler.add_event.call_args
        self.assertEqual(kwargs.get("impact"), "LOW")


# ---------------------------------------------------------------------------
# 4. TestRunCycle
# ---------------------------------------------------------------------------

class TestRunCycle(unittest.TestCase):
    """run_cycle() returns True on a valid signal, False otherwise."""

    def setUp(self):
        self.m = _import_main()
        self._orig_logger = self.m._trade_logger
        self.mock_logger = MagicMock()
        self.m._trade_logger = self.mock_logger

    def tearDown(self):
        self.m._trade_logger = self._orig_logger

    def test_returns_false_when_signal_is_none(self):
        with patch.object(self.m, "load_market_data", return_value=MagicMock()), \
             patch.object(self.m, "generate_signal_dict", return_value=None):
            result = self.m.run_cycle()
        self.assertFalse(result)

    def test_returns_true_on_valid_signal(self):
        with patch.object(self.m, "load_market_data", return_value=MagicMock()), \
             patch.object(self.m, "generate_signal_dict", return_value=_VALID_SIGNAL), \
             patch.object(self.m, "format_trade_signal", return_value="[SIGNAL CARD]"), \
             patch("builtins.print"):
            result = self.m.run_cycle()
        self.assertTrue(result)

    def test_returns_false_on_data_fetch_exception(self):
        with patch.object(self.m, "load_market_data", side_effect=RuntimeError("no data")):
            result = self.m.run_cycle()
        self.assertFalse(result)

    def test_returns_false_on_signal_engine_exception(self):
        with patch.object(self.m, "load_market_data", return_value=MagicMock()), \
             patch.object(self.m, "generate_signal_dict", side_effect=RuntimeError("crash")):
            result = self.m.run_cycle()
        self.assertFalse(result)

    def test_log_signal_called_on_valid_signal(self):
        with patch.object(self.m, "load_market_data", return_value=MagicMock()), \
             patch.object(self.m, "generate_signal_dict", return_value=_VALID_SIGNAL), \
             patch.object(self.m, "format_trade_signal", return_value="[CARD]"), \
             patch("builtins.print"):
            self.m.run_cycle()
        self.mock_logger.log_signal.assert_called_once()

    def test_log_signal_failure_is_non_fatal(self):
        """A logging error must not prevent run_cycle returning True."""
        self.mock_logger.log_signal.side_effect = Exception("db error")
        with patch.object(self.m, "load_market_data", return_value=MagicMock()), \
             patch.object(self.m, "generate_signal_dict", return_value=_VALID_SIGNAL), \
             patch.object(self.m, "format_trade_signal", return_value="[CARD]"), \
             patch("builtins.print"):
            result = self.m.run_cycle()
        self.assertTrue(result)


# ---------------------------------------------------------------------------
# 5. TestMaybeAutosave
# ---------------------------------------------------------------------------

class TestMaybeAutosave(unittest.TestCase):
    """_maybe_autosave() saves only at multiples of AUTOSAVE_EVERY_N_CYCLES."""

    def setUp(self):
        self.m = _import_main()
        self._orig_logger = self.m._trade_logger
        self.mock_tl = MagicMock()
        self.m._trade_logger = self.mock_tl

    def tearDown(self):
        self.m._trade_logger = self._orig_logger

    def test_saves_at_exact_multiple(self):
        n = self.m.AUTOSAVE_EVERY_N_CYCLES
        self.m._maybe_autosave(n)
        self.mock_tl.save.assert_called_once_with(self.m.TRADE_LOG_PATH)

    def test_does_not_save_one_before_multiple(self):
        n = self.m.AUTOSAVE_EVERY_N_CYCLES
        self.m._maybe_autosave(n - 1)
        self.mock_tl.save.assert_not_called()

    def test_saves_at_second_multiple(self):
        n = self.m.AUTOSAVE_EVERY_N_CYCLES
        self.m._maybe_autosave(n * 2)
        self.mock_tl.save.assert_called_once()

    def test_zero_interval_never_saves(self):
        orig = self.m.AUTOSAVE_EVERY_N_CYCLES
        self.m.AUTOSAVE_EVERY_N_CYCLES = 0
        try:
            for i in range(1, 51):
                self.m._maybe_autosave(i)
            self.mock_tl.save.assert_not_called()
        finally:
            self.m.AUTOSAVE_EVERY_N_CYCLES = orig

    def test_save_failure_does_not_raise(self):
        self.mock_tl.save.side_effect = IOError("disk full")
        n = self.m.AUTOSAVE_EVERY_N_CYCLES
        try:
            self.m._maybe_autosave(n)
        except Exception as exc:
            self.fail(f"_maybe_autosave raised on save failure: {exc}")


# ---------------------------------------------------------------------------
# 6. TestMaybePrintSummary
# ---------------------------------------------------------------------------

class TestMaybePrintSummary(unittest.TestCase):
    """_maybe_print_summary() logs metrics when resolved records exist."""

    def setUp(self):
        self.m = _import_main()
        self._orig_logger = self.m._trade_logger
        self.mock_tl = MagicMock()
        self.m._trade_logger = self.mock_tl

    def tearDown(self):
        self.m._trade_logger = self._orig_logger

    def _fake_summary(self):
        s = MagicMock()
        s.total_trades = 5
        s.win_rate = 0.6
        s.expectancy_r = 0.4
        s.max_drawdown_r = -1.2
        s.profit_factor = 1.8
        return s

    def test_logs_summary_when_records_exist(self):
        self.mock_tl.resolved_records.return_value = [MagicMock()]
        n = self.m.SUMMARY_EVERY_N_CYCLES
        with patch.object(self.m, "compute_summary", return_value=self._fake_summary()):
            # Should not raise
            self.m._maybe_print_summary(n)

    def test_skips_when_no_resolved_records(self):
        self.mock_tl.resolved_records.return_value = []
        n = self.m.SUMMARY_EVERY_N_CYCLES
        with patch.object(self.m, "compute_summary") as mock_cs:
            self.m._maybe_print_summary(n)
        mock_cs.assert_not_called()

    def test_does_not_print_on_non_multiple(self):
        self.mock_tl.resolved_records.return_value = [MagicMock()]
        n = self.m.SUMMARY_EVERY_N_CYCLES
        with patch.object(self.m, "compute_summary") as mock_cs:
            self.m._maybe_print_summary(n - 1)
        mock_cs.assert_not_called()

    def test_zero_interval_never_prints(self):
        orig = self.m.SUMMARY_EVERY_N_CYCLES
        self.m.SUMMARY_EVERY_N_CYCLES = 0
        self.mock_tl.resolved_records.return_value = [MagicMock()]
        try:
            with patch.object(self.m, "compute_summary") as mock_cs:
                for i in range(1, 51):
                    self.m._maybe_print_summary(i)
            mock_cs.assert_not_called()
        finally:
            self.m.SUMMARY_EVERY_N_CYCLES = orig


# ---------------------------------------------------------------------------
# 7. TestMain
# ---------------------------------------------------------------------------

class TestMain(unittest.TestCase):
    """main() runs the loop, handles KeyboardInterrupt, and saves on shutdown."""

    def setUp(self):
        self.m = _import_main()
        self._orig_logger = self.m._trade_logger
        self.mock_tl = MagicMock()
        self.mock_tl.all_records.return_value = []
        self.m._trade_logger = self.mock_tl

    def tearDown(self):
        self.m._trade_logger = self._orig_logger

    def test_keyboard_interrupt_causes_clean_shutdown(self):
        """main() must catch KeyboardInterrupt and save the trade log."""
        call_count = 0

        def fake_run_cycle():
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise KeyboardInterrupt

        with patch.object(self.m, "run_cycle", side_effect=fake_run_cycle), \
             patch.object(self.m, "_maybe_autosave"), \
             patch.object(self.m, "_maybe_print_summary"), \
             patch.object(self.m.time, "sleep"):
            self.m.main()  # Must not propagate KeyboardInterrupt

        self.mock_tl.save.assert_called_once_with(self.m.TRADE_LOG_PATH)

    def test_sleep_called_between_cycles(self):
        """time.sleep must be called with CYCLE_INTERVAL_SECONDS."""
        calls = 0

        def fake_run_cycle():
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise KeyboardInterrupt

        with patch.object(self.m, "run_cycle", side_effect=fake_run_cycle), \
             patch.object(self.m, "_maybe_autosave"), \
             patch.object(self.m, "_maybe_print_summary"), \
             patch.object(self.m.time, "sleep") as mock_sleep:
            self.m.main()

        mock_sleep.assert_called_with(self.m.CYCLE_INTERVAL_SECONDS)

    def test_save_failure_on_shutdown_does_not_reraise(self):
        """If the final save fails, main() must still return cleanly."""
        self.mock_tl.save.side_effect = IOError("no space")

        def raise_kb():
            raise KeyboardInterrupt

        with patch.object(self.m, "run_cycle", side_effect=raise_kb), \
             patch.object(self.m, "_maybe_autosave"), \
             patch.object(self.m, "_maybe_print_summary"), \
             patch.object(self.m.time, "sleep"):
            try:
                self.m.main()
            except Exception as exc:
                self.fail(f"main() re-raised after save failure: {exc}")


if __name__ == "__main__":
    unittest.main()
