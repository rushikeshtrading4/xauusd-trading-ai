"""Tests for backtesting/backtest_engine.py — run_backtest().

Strategy: all pipeline calls are patched via
``unittest.mock.patch("backtesting.backtest_engine.generate_signal_dict")``.
This isolates the engine logic completely from upstream complexity and lets
each test control exactly which signals are emitted and when.

Test groups
-----------
 1. TestOutputSchema          (9)  — return dict has all required keys and types
 2. TestNoTrades              (4)  — empty / short df / no signal → zero-trade metrics
 3. TestWinResolution         (4)  — BUY + SELL TP hits recorded as WIN
 4. TestLossResolution        (4)  — BUY + SELL SL hits recorded as LOSS
 5. TestSameCandleConflict    (3)  — same-candle SL+TP → conservative LOSS
 6. TestEquityCompounding     (4)  — equity grows / shrinks correctly each trade
 7. TestMaxDrawdown           (4)  — peak-to-trough fraction computed correctly
 8. TestEquityCurve           (4)  — curve length and values match trade sequence
 9. TestMetricsCalculation    (6)  — win_rate, avg_rr, expectancy arithmetic
10. TestNoLookahead           (3)  — pipeline receives only past candles
11. TestUnresolvedTrades      (3)  — trades that never hit SL/TP are excluded
12. TestDeterminism           (2)  — same input → same output
13. TestTradeLock             (3)  — locked bars skipped; one trade at a time
14. TestEntryPriceGuard       (2)  — signal at last bar produces no trade
"""

from __future__ import annotations

from unittest.mock import patch, call
import pytest
import pandas as pd

from backtesting.backtest_engine import run_backtest

# ---------------------------------------------------------------------------
# Constants matching the engine (imported to keep tests DRY)
# ---------------------------------------------------------------------------
from backtesting.backtest_engine import _START_EQUITY, _RISK_PCT, _WARMUP_BARS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_df(n: int, price: float = 2_000.0) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with *n* identical flat candles."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open":      price,
        "high":      price + 5.0,
        "low":       price - 5.0,
        "close":     price,
        "volume":    1_000.0,
    })


def _buy_signal(entry: float = 2_000.0,
                stop_loss: float = 1_990.0,
                take_profit: float = 2_020.0,
                rr: float = 2.0) -> dict:
    return {
        "entry":       entry,
        "stop_loss":   stop_loss,
        "take_profit": take_profit,
        "risk_reward": rr,
        "bias":        "BULLISH",
    }


def _sell_signal(entry: float = 2_000.0,
                 stop_loss: float = 2_010.0,
                 take_profit: float = 1_980.0,
                 rr: float = 2.0) -> dict:
    return {
        "entry":       entry,
        "stop_loss":   stop_loss,
        "take_profit": take_profit,
        "risk_reward": rr,
        "bias":        "BEARISH",
    }


def _candle(low: float, high: float) -> dict:
    """Build a row dict with the given low/high (other OHLC fields don't matter)."""
    return {"low": low, "high": high, "open": low, "close": high, "volume": 100.0}


def _df_with_resolution(signal_bar: int,
                        signal: dict,
                        resolution_candle: dict,
                        total_bars: int | None = None) -> pd.DataFrame:
    """
    Build a DataFrame where:
    - bar *signal_bar* triggers the given signal (via mock)
    - bar *signal_bar + 1* is the entry candle (entry at its open)
    - bar *signal_bar + 2* is the resolution candle (first bar the engine
      checks for SL/TP, because the resolution loop starts at i+2)
    - remaining bars are flat (won't re-trigger the signal)
    """
    n = total_bars or (signal_bar + 5)
    rows = []
    for k in range(n):
        if k == signal_bar + 2:
            rows.append(resolution_candle)
        else:
            rows.append(_candle(low=1_995.0, high=2_005.0))
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")
    return df


# ---------------------------------------------------------------------------
# Patch context manager helper
# ---------------------------------------------------------------------------

def _patch_signal(side_effect):
    """Patch generate_signal_dict in the engine module."""
    return patch(
        "backtesting.backtest_engine.generate_signal_dict",
        side_effect=side_effect,
    )


# ===========================================================================
# 1. Output schema
# ===========================================================================

class TestOutputSchema:

    def _result(self):
        df = _flat_df(_WARMUP_BARS)
        with _patch_signal(side_effect=lambda _: None):
            return run_backtest(df)

    def test_returns_dict(self):
        assert isinstance(self._result(), dict)

    def test_has_total_trades(self):
        assert "total_trades" in self._result()

    def test_has_wins(self):
        assert "wins" in self._result()

    def test_has_losses(self):
        assert "losses" in self._result()

    def test_has_win_rate(self):
        assert "win_rate" in self._result()

    def test_has_avg_rr(self):
        assert "avg_rr" in self._result()

    def test_has_expectancy(self):
        assert "expectancy" in self._result()

    def test_has_max_drawdown(self):
        assert "max_drawdown" in self._result()

    def test_has_equity_curve(self):
        assert "equity_curve" in self._result()
        assert isinstance(self._result()["equity_curve"], list)


# ===========================================================================
# 2. No trades
# ===========================================================================

class TestNoTrades:

    def test_empty_df_returns_zero_trades(self):
        with _patch_signal(side_effect=lambda _: None):
            result = run_backtest(_flat_df(0))
        assert result["total_trades"] == 0

    def test_df_shorter_than_warmup_returns_zero_trades(self):
        with _patch_signal(side_effect=lambda _: None):
            result = run_backtest(_flat_df(_WARMUP_BARS - 1))
        assert result["total_trades"] == 0

    def test_no_signal_returns_zero_trades(self):
        with _patch_signal(side_effect=lambda _: None):
            result = run_backtest(_flat_df(_WARMUP_BARS + 10))
        assert result["total_trades"] == 0

    def test_no_trades_equity_curve_starts_at_start_equity(self):
        with _patch_signal(side_effect=lambda _: None):
            result = run_backtest(_flat_df(_WARMUP_BARS + 5))
        assert result["equity_curve"] == [_START_EQUITY]

    def test_no_trades_zero_metrics(self):
        with _patch_signal(side_effect=lambda _: None):
            result = run_backtest(_flat_df(_WARMUP_BARS + 5))
        assert result["win_rate"] == 0.0
        assert result["avg_rr"] == 0.0
        assert result["expectancy"] == 0.0


# ===========================================================================
# 3. WIN resolution — BUY and SELL
# ===========================================================================

class TestWinResolution:

    def _run_single(self, signal: dict, resolution: dict) -> dict:
        """Emit *signal* once at bar WARMUP_BARS; resolve on the candle after entry."""
        n = _WARMUP_BARS + 4
        calls = [0]

        def side_effect(_):
            calls[0] += 1
            # Emit signal only on the first call (bar WARMUP_BARS)
            return signal if calls[0] == 1 else None

        rows = [_candle(1_995.0, 2_005.0)] * n
        rows[_WARMUP_BARS + 2] = resolution   # first bar checked by resolution loop (i+2)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=side_effect):
            return run_backtest(df)

    def test_buy_tp_hit_is_win(self):
        sig = _buy_signal(entry=2_000.0, stop_loss=1_990.0, take_profit=2_020.0)
        res = _candle(low=2_010.0, high=2_025.0)  # high >= tp
        result = self._run_single(sig, res)
        assert result["wins"] == 1
        assert result["losses"] == 0

    def test_sell_tp_hit_is_win(self):
        sig = _sell_signal(entry=2_000.0, stop_loss=2_010.0, take_profit=1_980.0)
        res = _candle(low=1_975.0, high=1_999.0)  # low <= tp
        result = self._run_single(sig, res)
        assert result["wins"] == 1
        assert result["losses"] == 0

    def test_win_increments_total_trades(self):
        sig = _buy_signal()
        res = _candle(low=2_010.0, high=2_025.0)
        result = self._run_single(sig, res)
        assert result["total_trades"] == 1

    def test_win_rate_is_one_for_single_win(self):
        sig = _buy_signal()
        res = _candle(low=2_010.0, high=2_025.0)
        result = self._run_single(sig, res)
        assert result["win_rate"] == pytest.approx(1.0)


# ===========================================================================
# 4. LOSS resolution — BUY and SELL
# ===========================================================================

class TestLossResolution:

    def _run_single(self, signal: dict, resolution: dict) -> dict:
        n = _WARMUP_BARS + 4
        calls = [0]

        def side_effect(_):
            calls[0] += 1
            return signal if calls[0] == 1 else None

        rows = [_candle(1_995.0, 2_005.0)] * n
        rows[_WARMUP_BARS + 2] = resolution   # first bar checked by resolution loop (i+2)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=side_effect):
            return run_backtest(df)

    def test_buy_sl_hit_is_loss(self):
        sig = _buy_signal(entry=2_000.0, stop_loss=1_990.0, take_profit=2_020.0)
        res = _candle(low=1_985.0, high=1_998.0)  # low <= sl
        result = self._run_single(sig, res)
        assert result["losses"] == 1
        assert result["wins"] == 0

    def test_sell_sl_hit_is_loss(self):
        sig = _sell_signal(entry=2_000.0, stop_loss=2_010.0, take_profit=1_980.0)
        res = _candle(low=2_001.0, high=2_015.0)  # high >= sl
        result = self._run_single(sig, res)
        assert result["losses"] == 1
        assert result["wins"] == 0

    def test_loss_increments_total_trades(self):
        sig = _buy_signal()
        res = _candle(low=1_985.0, high=1_998.0)
        result = self._run_single(sig, res)
        assert result["total_trades"] == 1

    def test_win_rate_is_zero_for_single_loss(self):
        sig = _buy_signal()
        res = _candle(low=1_985.0, high=1_998.0)
        result = self._run_single(sig, res)
        assert result["win_rate"] == pytest.approx(0.0)


# ===========================================================================
# 5. Same-candle conflict → conservative LOSS
# ===========================================================================

class TestSameCandleConflict:

    def _run_conflict(self, signal: dict) -> dict:
        """Emit signal once; resolution candle hits both SL and TP."""
        n = _WARMUP_BARS + 4
        calls = [0]

        def side_effect(_):
            calls[0] += 1
            return signal if calls[0] == 1 else None

        rows = [_candle(1_995.0, 2_005.0)] * n
        # For BUY: sl=1990, tp=2020 → low=1985 (sl hit), high=2025 (tp hit)
        # For SELL: sl=2010, tp=1980 → high=2015 (sl hit), low=1975 (tp hit)
        # Placed at i+2 (first bar the resolution loop checks)
        if signal["bias"].upper() == "BULLISH":
            rows[_WARMUP_BARS + 2] = _candle(low=1_985.0, high=2_025.0)
        else:
            rows[_WARMUP_BARS + 2] = _candle(low=1_975.0, high=2_015.0)

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=side_effect):
            return run_backtest(df)

    def test_buy_same_candle_is_loss(self):
        result = self._run_conflict(_buy_signal())
        assert result["losses"] == 1
        assert result["wins"] == 0

    def test_sell_same_candle_is_loss(self):
        result = self._run_conflict(_sell_signal())
        assert result["losses"] == 1
        assert result["wins"] == 0

    def test_same_candle_counted_as_one_trade(self):
        result = self._run_conflict(_buy_signal())
        assert result["total_trades"] == 1


# ===========================================================================
# 6. Equity compounding
# ===========================================================================

class TestEquityCompounding:

    def _run_two_trades(self, outcomes: list[str]) -> dict:
        """Simulate exactly two resolved trades with RR=2 for each."""
        n = _WARMUP_BARS + 6
        call_count = [0]
        signals = [_buy_signal(rr=2.0), _buy_signal(rr=2.0)]

        def side_effect(_):
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(signals):
                return signals[idx]
            return None

        rows = [_candle(1_995.0, 2_005.0)] * n
        # resolution candle for trade 1 at bar WARMUP_BARS+2 (i+2 = WARMUP+0+2)
        if outcomes[0] == "WIN":
            rows[_WARMUP_BARS + 2] = _candle(low=2_010.0, high=2_025.0)
        else:
            rows[_WARMUP_BARS + 2] = _candle(low=1_985.0, high=1_998.0)
        # resolution candle for trade 2 at bar WARMUP_BARS+4 (i+2 = WARMUP+2+2)
        if outcomes[1] == "WIN":
            rows[_WARMUP_BARS + 4] = _candle(low=2_010.0, high=2_025.0)
        else:
            rows[_WARMUP_BARS + 4] = _candle(low=1_985.0, high=1_998.0)

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=side_effect):
            return run_backtest(df)

    def test_single_win_increases_equity(self):
        n = _WARMUP_BARS + 4
        calls = [0]

        def se(_):
            calls[0] += 1
            return _buy_signal(rr=2.0) if calls[0] == 1 else None

        rows = [_candle(1_995.0, 2_005.0)] * n
        rows[_WARMUP_BARS + 2] = _candle(low=2_010.0, high=2_025.0)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=se):
            result = run_backtest(df)

        expected = _START_EQUITY + 2.0 * (_RISK_PCT * _START_EQUITY)
        assert result["equity_curve"][-1] == pytest.approx(expected)

    def test_single_loss_decreases_equity(self):
        n = _WARMUP_BARS + 4
        calls = [0]

        def se(_):
            calls[0] += 1
            return _buy_signal(rr=2.0) if calls[0] == 1 else None

        rows = [_candle(1_995.0, 2_005.0)] * n
        rows[_WARMUP_BARS + 2] = _candle(low=1_985.0, high=1_998.0)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=se):
            result = run_backtest(df)

        expected = _START_EQUITY - (_RISK_PCT * _START_EQUITY)
        assert result["equity_curve"][-1] == pytest.approx(expected)

    def test_second_trade_uses_updated_equity(self):
        result = self._run_two_trades(["WIN", "WIN"])
        equity_after_t1 = _START_EQUITY + 2.0 * (_RISK_PCT * _START_EQUITY)
        equity_after_t2 = equity_after_t1 + 2.0 * (_RISK_PCT * equity_after_t1)
        assert result["equity_curve"][-1] == pytest.approx(equity_after_t2)

    def test_win_then_loss_equity(self):
        result = self._run_two_trades(["WIN", "LOSS"])
        equity_after_t1 = _START_EQUITY + 2.0 * (_RISK_PCT * _START_EQUITY)
        equity_after_t2 = equity_after_t1 - (_RISK_PCT * equity_after_t1)
        assert result["equity_curve"][-1] == pytest.approx(equity_after_t2)


# ===========================================================================
# 7. Max drawdown
# ===========================================================================

class TestMaxDrawdown:

    def test_no_drawdown_when_all_wins(self):
        n = _WARMUP_BARS + 4
        calls = [0]

        def se(_):
            calls[0] += 1
            return _buy_signal(rr=2.0) if calls[0] == 1 else None

        rows = [_candle(1_995.0, 2_005.0)] * n
        rows[_WARMUP_BARS + 2] = _candle(low=2_010.0, high=2_025.0)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=se):
            result = run_backtest(df)
        assert result["max_drawdown"] == pytest.approx(0.0)

    def test_single_loss_drawdown(self):
        n = _WARMUP_BARS + 4
        calls = [0]

        def se(_):
            calls[0] += 1
            return _buy_signal(rr=2.0) if calls[0] == 1 else None

        rows = [_candle(1_995.0, 2_005.0)] * n
        rows[_WARMUP_BARS + 2] = _candle(low=1_985.0, high=1_998.0)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=se):
            result = run_backtest(df)
        expected_dd = _RISK_PCT  # loss of exactly 0.5% from peak
        assert result["max_drawdown"] == pytest.approx(expected_dd)

    def test_max_drawdown_is_fraction_not_absolute(self):
        n = _WARMUP_BARS + 4
        calls = [0]

        def se(_):
            calls[0] += 1
            return _buy_signal(rr=2.0) if calls[0] == 1 else None

        rows = [_candle(1_995.0, 2_005.0)] * n
        rows[_WARMUP_BARS + 2] = _candle(low=1_985.0, high=1_998.0)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=se):
            result = run_backtest(df)
        # Must be in [0, 1] — a fraction, not a dollar amount
        assert 0.0 <= result["max_drawdown"] <= 1.0

    def test_max_drawdown_zero_when_no_trades(self):
        with _patch_signal(side_effect=lambda _: None):
            result = run_backtest(_flat_df(_WARMUP_BARS + 5))
        assert result["max_drawdown"] == pytest.approx(0.0)


# ===========================================================================
# 8. Equity curve
# ===========================================================================

class TestEquityCurve:

    def test_curve_starts_with_start_equity(self):
        n = _WARMUP_BARS + 4
        calls = [0]

        def se(_):
            calls[0] += 1
            return _buy_signal(rr=2.0) if calls[0] == 1 else None

        rows = [_candle(1_995.0, 2_005.0)] * n
        rows[_WARMUP_BARS + 2] = _candle(low=2_010.0, high=2_025.0)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=se):
            result = run_backtest(df)
        assert result["equity_curve"][0] == _START_EQUITY

    def test_curve_length_equals_trades_plus_one(self):
        n = _WARMUP_BARS + 4
        calls = [0]

        def se(_):
            calls[0] += 1
            return _buy_signal(rr=2.0) if calls[0] == 1 else None

        rows = [_candle(1_995.0, 2_005.0)] * n
        rows[_WARMUP_BARS + 2] = _candle(low=2_010.0, high=2_025.0)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=se):
            result = run_backtest(df)
        assert len(result["equity_curve"]) == result["total_trades"] + 1

    def test_curve_no_trades_has_one_element(self):
        with _patch_signal(side_effect=lambda _: None):
            result = run_backtest(_flat_df(_WARMUP_BARS + 5))
        assert len(result["equity_curve"]) == 1

    def test_curve_values_are_floats(self):
        n = _WARMUP_BARS + 4
        calls = [0]

        def se(_):
            calls[0] += 1
            return _buy_signal(rr=2.0) if calls[0] == 1 else None

        rows = [_candle(1_995.0, 2_005.0)] * n
        rows[_WARMUP_BARS + 2] = _candle(low=2_010.0, high=2_025.0)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=se):
            result = run_backtest(df)
        assert all(isinstance(v, float) for v in result["equity_curve"])


# ===========================================================================
# 9. Metrics calculation
# ===========================================================================

class TestMetricsCalculation:

    def _run_n_trades(self, outcomes_rrs: list[tuple[str, float]]) -> dict:
        """
        Simulate exactly len(outcomes_rrs) resolved trades.
        outcomes_rrs: list of ("WIN"/"LOSS", rr) tuples.

        How spacing works under the new resolution loop (range(i+2, n)):
          - Signal[k] fires at call 3k, which maps to outer-loop bar
            i = WARMUP + k*4.  One bar (i+1) is locked after each trade
            because next_trade_bar = j = i+2, so i+1 < i+2 is skipped.
          - Resolution candle placed at i+2 = WARMUP + 2 + k*4.
        """
        n = _WARMUP_BARS + len(outcomes_rrs) * 4 + 5
        call_count = [0]

        signals = [_buy_signal(rr=rr) for _, rr in outcomes_rrs]

        def side_effect(_):
            idx = call_count[0]
            call_count[0] += 1
            trade_idx, step = divmod(idx, 3)
            if step == 0 and trade_idx < len(signals):
                return signals[trade_idx]
            return None

        rows = [_candle(1_995.0, 2_005.0)] * n
        for k, (outcome, _) in enumerate(outcomes_rrs):
            res_idx = _WARMUP_BARS + 2 + k * 4   # first bar the resolution loop checks
            if res_idx < n:
                if outcome == "WIN":
                    rows[res_idx] = _candle(low=2_010.0, high=2_025.0)
                else:
                    rows[res_idx] = _candle(low=1_985.0, high=1_998.0)

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=side_effect):
            return run_backtest(df)

    def test_win_rate_two_wins_one_loss(self):
        result = self._run_n_trades([("WIN", 2.0), ("WIN", 2.0), ("LOSS", 2.0)])
        assert result["win_rate"] == pytest.approx(2 / 3)

    def test_avg_rr_mixed(self):
        result = self._run_n_trades([("WIN", 2.0), ("WIN", 3.0)])
        assert result["avg_rr"] == pytest.approx(2.5)

    def test_expectancy_positive(self):
        # 2 wins at rr=2, avg_rr=2, win_rate=1 → expectancy = 1*2 - 0*1 = 2
        result = self._run_n_trades([("WIN", 2.0), ("WIN", 2.0)])
        assert result["expectancy"] == pytest.approx(2.0)

    def test_expectancy_negative(self):
        # 2 losses, win_rate=0, avg_rr=2 → expectancy = 0*2 - 1*1 = -1
        result = self._run_n_trades([("LOSS", 2.0), ("LOSS", 2.0)])
        assert result["expectancy"] == pytest.approx(-1.0)

    def test_wins_plus_losses_equals_total(self):
        result = self._run_n_trades([("WIN", 2.0), ("LOSS", 2.0), ("WIN", 3.0)])
        assert result["wins"] + result["losses"] == result["total_trades"]

    def test_expectancy_formula(self):
        result = self._run_n_trades([("WIN", 3.0), ("LOSS", 2.0)])
        wr = result["win_rate"]
        lr = 1.0 - wr
        expected = (wr * result["avg_rr"]) - (lr * 1.0)
        assert result["expectancy"] == pytest.approx(expected)


# ===========================================================================
# 10. No lookahead
# ===========================================================================

class TestNoLookahead:

    def test_pipeline_receives_only_past_candles(self):
        """generate_signal_dict must be called with df.iloc[:i], not the full df."""
        n = _WARMUP_BARS + 3
        received_lengths = []

        def se(market_data):
            received_lengths.append(len(market_data["M5"]))
            return None

        with _patch_signal(side_effect=se):
            run_backtest(_flat_df(n))

        # Each call receives df[:i] for i in [WARMUP_BARS, ..., n-1]
        for k, length in enumerate(received_lengths):
            expected = _WARMUP_BARS + k
            assert length == expected

    def test_pipeline_called_from_warmup_bar(self):
        """First call happens at index WARMUP_BARS, not before."""
        first_length = [None]

        def se(market_data):
            if first_length[0] is None:
                first_length[0] = len(market_data["M5"])
            return None

        n = _WARMUP_BARS + 5
        with _patch_signal(side_effect=se):
            run_backtest(_flat_df(n))

        assert first_length[0] == _WARMUP_BARS

    def test_pipeline_not_called_for_short_df(self):
        """No calls when df is shorter than warmup."""
        call_count = [0]

        def se(_):
            call_count[0] += 1
            return None

        with _patch_signal(side_effect=se):
            run_backtest(_flat_df(_WARMUP_BARS - 1))

        assert call_count[0] == 0


# ===========================================================================
# 11. Unresolved trades
# ===========================================================================

class TestUnresolvedTrades:

    def test_trade_that_never_hits_sl_or_tp_excluded(self):
        """If no future candle hits SL or TP the trade is skipped (no PnL)."""
        # BUY: sl=1990, tp=2020. All subsequent candles stay between them.
        n = _WARMUP_BARS + 4
        calls = [0]

        def se(_):
            calls[0] += 1
            return _buy_signal(stop_loss=1_990.0, take_profit=2_020.0) if calls[0] == 1 else None

        # All candles: low=1995, high=2005 — neither SL nor TP touched
        rows = [_candle(1_995.0, 2_005.0)] * n
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=se):
            result = run_backtest(df)

        assert result["total_trades"] == 0

    def test_unresolved_trade_does_not_affect_equity(self):
        n = _WARMUP_BARS + 4
        calls = [0]

        def se(_):
            calls[0] += 1
            return _buy_signal(stop_loss=1_990.0, take_profit=2_020.0) if calls[0] == 1 else None

        rows = [_candle(1_995.0, 2_005.0)] * n
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=se):
            result = run_backtest(df)

        assert result["equity_curve"] == [_START_EQUITY]

    def test_unresolved_trade_not_counted_in_wins_or_losses(self):
        n = _WARMUP_BARS + 4
        calls = [0]

        def se(_):
            calls[0] += 1
            return _buy_signal(stop_loss=1_990.0, take_profit=2_020.0) if calls[0] == 1 else None

        rows = [_candle(1_995.0, 2_005.0)] * n
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=se):
            result = run_backtest(df)

        assert result["wins"] == 0
        assert result["losses"] == 0


# ===========================================================================
# 12. Determinism
# ===========================================================================

class TestDeterminism:

    def test_same_input_same_output(self):
        n = _WARMUP_BARS + 4
        signals_1 = [0]
        signals_2 = [0]

        def se1(_):
            signals_1[0] += 1
            return _buy_signal(rr=2.0) if signals_1[0] == 1 else None

        def se2(_):
            signals_2[0] += 1
            return _buy_signal(rr=2.0) if signals_2[0] == 1 else None

        rows = [_candle(1_995.0, 2_005.0)] * n
        rows[_WARMUP_BARS + 2] = _candle(low=2_010.0, high=2_025.0)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=se1):
            r1 = run_backtest(df)
        with _patch_signal(side_effect=se2):
            r2 = run_backtest(df)

        assert r1["total_trades"] == r2["total_trades"]
        assert r1["equity_curve"] == r2["equity_curve"]
        assert r1["win_rate"] == r2["win_rate"]

    def test_no_signal_output_is_stable(self):
        with _patch_signal(side_effect=lambda _: None):
            r1 = run_backtest(_flat_df(_WARMUP_BARS + 5))
        with _patch_signal(side_effect=lambda _: None):
            r2 = run_backtest(_flat_df(_WARMUP_BARS + 5))
        assert r1 == r2


# ===========================================================================
# 13. Trade lock — one trade at a time
# ===========================================================================

class TestTradeLock:
    """Bars inside an active trade's forward scan window must be skipped by
    the outer loop so no second trade can overlap the first."""

    def test_locked_bars_not_sampled_for_signals(self):
        """The mock must be called exactly 3 times:
        i=WARMUP (trade opens), i=WARMUP+3 (unlocked after resolution),
        i=WARMUP+4 (entry guard fires — no trade, but mock still called)."""
        n = _WARMUP_BARS + 5
        call_count = [0]

        def side_effect(_):
            call_count[0] += 1
            return _buy_signal(stop_loss=1_990.0, take_profit=2_020.0, rr=2.0)

        # WARMUP+3 triggers TP → trade resolves at j=WARMUP+3, next_trade_bar=WARMUP+3
        rows = [_candle(1_995.0, 2_005.0)] * n
        rows[_WARMUP_BARS + 3] = _candle(low=2_010.0, high=2_025.0)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=side_effect):
            run_backtest(df)

        assert call_count[0] == 3

    def test_only_first_trade_counted_during_locked_window(self):
        """Bars WARMUP+1 and WARMUP+2 are locked; only the trade opened at
        WARMUP (WIN) should be recorded."""
        n = _WARMUP_BARS + 5

        def side_effect(_):
            return _buy_signal(stop_loss=1_990.0, take_profit=2_020.0, rr=2.0)

        rows = [_candle(1_995.0, 2_005.0)] * n
        rows[_WARMUP_BARS + 3] = _candle(low=2_010.0, high=2_025.0)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=side_effect):
            result = run_backtest(df)

        assert result["total_trades"] == 1
        assert result["wins"] == 1

    def test_new_trade_allowed_after_resolution(self):
        """Bar j (the resolution bar) must NOT be locked — a new trade may
        open on it if a signal fires there.

        Timeline (new resolution loop range(i+2, n)):
          - Trade1: opens at i=WARMUP, entry at WARMUP+1, resolves at j=WARMUP+3
            (rows[WARMUP+3]=WIN).  next_trade_bar=WARMUP+3.
          - Trade2: opens at i=WARMUP+3 (not locked), entry at WARMUP+4,
            resolves at j=WARMUP+5 (rows[WARMUP+5]=WIN).
        """
        n = _WARMUP_BARS + 7

        def side_effect(_):
            return _buy_signal(stop_loss=1_990.0, take_profit=2_020.0, rr=2.0)

        rows = [_candle(1_995.0, 2_005.0)] * n
        rows[_WARMUP_BARS + 3] = _candle(low=2_010.0, high=2_025.0)   # Trade1 WIN at j=WARMUP+3
        rows[_WARMUP_BARS + 5] = _candle(low=2_010.0, high=2_025.0)   # Trade2 WIN at j=WARMUP+5
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=side_effect):
            result = run_backtest(df)

        assert result["wins"] >= 2


# ===========================================================================
# 14. Entry price guard — no trade when signal fires on the last bar
# ===========================================================================

class TestEntryPriceGuard:
    """When a signal fires at bar i and i+1 >= len(df), no entry candle
    exists; the engine must skip the trade entirely."""

    def test_signal_at_last_bar_creates_no_trade(self):
        """n = WARMUP+1: only bar WARMUP exists past warmup.
        i+1 = WARMUP+1 = n → guard must fire; total_trades == 0."""
        n = _WARMUP_BARS + 1
        calls = [0]

        def se(_):
            calls[0] += 1
            return _buy_signal()

        rows = [_candle(1_995.0, 2_005.0)] * n
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=se):
            result = run_backtest(df)

        assert result["total_trades"] == 0
        assert calls[0] == 1   # mock was called; guard fired after

    def test_guard_does_not_fire_when_next_candle_exists(self):
        """n = WARMUP+3: i+1 = WARMUP+1 < n → guard must NOT fire.
        Resolution starts at i+2 = WARMUP+2 (index n-1), so WIN there → 1 trade."""
        n = _WARMUP_BARS + 3
        calls = [0]

        def se(_):
            calls[0] += 1
            return _buy_signal() if calls[0] == 1 else None

        rows = [_candle(1_995.0, 2_005.0)] * n
        rows[_WARMUP_BARS + 2] = _candle(low=2_010.0, high=2_025.0)   # WIN at i+2
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="5min")

        with _patch_signal(side_effect=se):
            result = run_backtest(df)

        assert result["total_trades"] == 1
