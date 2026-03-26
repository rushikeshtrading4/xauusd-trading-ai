"""Tests for execution/performance_analytics.py"""

from __future__ import annotations

import math

import pytest

from execution.trade_logger import TradeRecord, OUTCOME_WIN, OUTCOME_LOSS
from execution.performance_analytics import (
    compute_summary,
    win_rate,
    profit_factor,
    expectancy_r,
    avg_win_r,
    avg_loss_r,
    max_consecutive_wins,
    max_consecutive_losses,
    max_drawdown_r,
    sharpe_r,
    best_trade,
    worst_trade,
    equity_curve_r,
    trades_by_bias,
    trades_by_context,
    trades_by_timeframe,
    PerformanceSummary,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _win(rr: float = 2.0) -> TradeRecord:
    return TradeRecord(
        signal_id="X",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        pair="XAUUSD",
        timeframe="M5",
        bias="BUY",
        entry=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        risk_reward=rr,
        confidence=80.0,
        invalidation=1990.0,
        atr=10.0,
        mtf_bias="BULLISH",
        bias_strength="STRONG",
        context="TREND",
        outcome=OUTCOME_WIN,
        exit_price=2000.0 + 10.0 * rr,
        pnl_r=float(rr),
        notes="",
    )


def _loss() -> TradeRecord:
    return TradeRecord(
        signal_id="Y",
        timestamp_utc="2026-01-01T01:00:00+00:00",
        pair="XAUUSD",
        timeframe="M5",
        bias="BUY",
        entry=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        risk_reward=2.0,
        confidence=75.0,
        invalidation=1990.0,
        atr=10.0,
        mtf_bias="BULLISH",
        bias_strength="WEAK",
        context="TREND",
        outcome=OUTCOME_LOSS,
        exit_price=1990.0,
        pnl_r=-1.0,
        notes="",
    )


def _pending() -> TradeRecord:
    """PENDING record — must be invisible to analytics."""
    from execution.trade_logger import OUTCOME_PENDING
    return TradeRecord(
        signal_id="Z",
        timestamp_utc="2026-01-01T02:00:00+00:00",
        pair="XAUUSD",
        timeframe="M5",
        bias="BUY",
        entry=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        risk_reward=2.0,
        confidence=80.0,
        invalidation=1990.0,
        atr=10.0,
        mtf_bias="BULLISH",
        bias_strength="STRONG",
        context="TREND",
        outcome=OUTCOME_PENDING,
        exit_price=None,
        pnl_r=None,
        notes="",
    )


# ---------------------------------------------------------------------------
# 1. TestEmptyInput
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_win_rate_empty(self):
        assert win_rate([]) == 0.0

    def test_profit_factor_empty(self):
        assert profit_factor([]) == float("inf")

    def test_expectancy_r_empty(self):
        assert expectancy_r([]) == 0.0

    def test_avg_win_r_empty(self):
        assert avg_win_r([]) == 0.0

    def test_avg_loss_r_empty(self):
        assert avg_loss_r([]) == 0.0

    def test_max_consecutive_wins_empty(self):
        assert max_consecutive_wins([]) == 0

    def test_max_consecutive_losses_empty(self):
        assert max_consecutive_losses([]) == 0

    def test_max_drawdown_r_empty(self):
        assert max_drawdown_r([]) == 0.0

    def test_sharpe_r_empty(self):
        assert math.isnan(sharpe_r([]))

    def test_equity_curve_r_empty(self):
        assert equity_curve_r([]) == [0.0]

    def test_best_trade_empty(self):
        assert best_trade([]) is None

    def test_worst_trade_empty(self):
        assert worst_trade([]) is None

    def test_compute_summary_empty(self):
        s = compute_summary([])
        assert s.total_trades == 0
        assert s.wins == 0
        assert s.losses == 0
        assert s.win_rate == 0.0
        assert s.total_r == 0.0
        assert math.isnan(s.sharpe_r)


# ---------------------------------------------------------------------------
# 2. TestWinRate
# ---------------------------------------------------------------------------


class TestWinRate:
    def test_all_losses(self):
        assert win_rate([_loss(), _loss()]) == 0.0

    def test_all_wins(self):
        assert win_rate([_win(), _win()]) == 1.0

    def test_mixed(self):
        assert win_rate([_win(), _win(), _loss()]) == pytest.approx(2 / 3)

    def test_pending_ignored(self):
        assert win_rate([_win(), _pending()]) == 1.0


# ---------------------------------------------------------------------------
# 3. TestProfitFactor
# ---------------------------------------------------------------------------


class TestProfitFactor:
    def test_no_losses_returns_inf(self):
        assert profit_factor([_win(2.0)]) == float("inf")

    def test_correct_ratio(self):
        # 2 wins of 2R each = 4R gross wins; 2 losses of 1R each = 2R gross losses → PF = 2.0
        records = [_win(2.0), _win(2.0), _loss(), _loss()]
        assert profit_factor(records) == pytest.approx(2.0)

    def test_zero_wins(self):
        assert profit_factor([_loss(), _loss()]) == 0.0


# ---------------------------------------------------------------------------
# 4. TestExpectancyR
# ---------------------------------------------------------------------------


class TestExpectancyR:
    def test_single_win(self):
        assert expectancy_r([_win(3.0)]) == pytest.approx(3.0)

    def test_single_loss(self):
        assert expectancy_r([_loss()]) == pytest.approx(-1.0)

    def test_mixed(self):
        # 2.0 + (-1.0) / 2 = 0.5
        assert expectancy_r([_win(2.0), _loss()]) == pytest.approx(0.5)

    def test_all_losses_negative(self):
        assert expectancy_r([_loss(), _loss()]) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 5. TestAvgWinLoss
# ---------------------------------------------------------------------------


class TestAvgWinLoss:
    def test_avg_win_r_correct(self):
        assert avg_win_r([_win(2.0), _win(4.0)]) == pytest.approx(3.0)

    def test_avg_win_r_no_wins(self):
        assert avg_win_r([_loss()]) == 0.0

    def test_avg_loss_r_magnitude(self):
        assert avg_loss_r([_loss(), _loss()]) == pytest.approx(1.0)

    def test_avg_loss_r_no_losses(self):
        assert avg_loss_r([_win()]) == 0.0


# ---------------------------------------------------------------------------
# 6. TestConsecutiveStreaks
# ---------------------------------------------------------------------------


class TestConsecutiveStreaks:
    def test_all_wins_streak(self):
        assert max_consecutive_wins([_win(), _win(), _win()]) == 3

    def test_all_losses_streak(self):
        assert max_consecutive_losses([_loss(), _loss()]) == 2

    def test_interleaved(self):
        records = [_win(), _win(), _loss(), _win(), _win(), _win(), _loss()]
        assert max_consecutive_wins(records) == 3
        assert max_consecutive_losses(records) == 1

    def test_single_loss_streak(self):
        assert max_consecutive_losses([_win(), _loss(), _win()]) == 1

    def test_no_wins_streak(self):
        assert max_consecutive_wins([_loss(), _loss()]) == 0


# ---------------------------------------------------------------------------
# 7. TestMaxDrawdownR
# ---------------------------------------------------------------------------


class TestMaxDrawdownR:
    def test_no_drawdown_all_wins(self):
        assert max_drawdown_r([_win(2.0), _win(2.0)]) == 0.0

    def test_single_loss(self):
        assert max_drawdown_r([_loss()]) == pytest.approx(1.0)

    def test_wlw_pattern(self):
        # equity: 0 → +2 → +1 → +3. Peak=2, trough=1, dd=1.0
        records = [_win(2.0), _loss(), _win(2.0)]
        assert max_drawdown_r(records) == pytest.approx(1.0)

    def test_empty_returns_zero(self):
        assert max_drawdown_r([]) == 0.0


# ---------------------------------------------------------------------------
# 8. TestSharpeR
# ---------------------------------------------------------------------------


class TestSharpeR:
    def test_fewer_than_2_trades_returns_nan(self):
        assert math.isnan(sharpe_r([_win()]))

    def test_all_same_r_returns_nan(self):
        # zero standard deviation → NaN (not division by zero error)
        records = [_win(2.0), _win(2.0), _win(2.0)]
        result = sharpe_r(records)
        assert math.isnan(result)

    def test_mixed_returns_finite_value(self):
        records = [_win(3.0), _loss(), _win(2.0), _loss()]
        result = sharpe_r(records)
        assert math.isfinite(result)

    def test_empty_returns_nan(self):
        assert math.isnan(sharpe_r([]))


# ---------------------------------------------------------------------------
# 9. TestEquityCurveR
# ---------------------------------------------------------------------------


class TestEquityCurveR:
    def test_starts_at_zero(self):
        curve = equity_curve_r([_win(2.0), _loss()])
        assert curve[0] == 0.0

    def test_length_is_trades_plus_one(self):
        records = [_win(), _win(), _loss()]
        curve = equity_curve_r(records)
        assert len(curve) == 4  # 0 + 3 trades

    def test_monotone_for_all_wins(self):
        records = [_win(2.0), _win(3.0), _win(1.5)]
        curve = equity_curve_r(records)
        assert all(curve[i] <= curve[i + 1] for i in range(len(curve) - 1))

    def test_correct_cumulative_values(self):
        curve = equity_curve_r([_win(2.0), _loss()])
        assert curve == pytest.approx([0.0, 2.0, 1.0])

    def test_pending_records_skipped(self):
        curve = equity_curve_r([_win(2.0), _pending()])
        assert curve == pytest.approx([0.0, 2.0])


# ---------------------------------------------------------------------------
# 10. TestGrouping
# ---------------------------------------------------------------------------


class TestGrouping:
    def _sell_win(self) -> TradeRecord:
        r = _win(2.0)
        return TradeRecord(**{**r.to_dict(), "bias": "SELL"})

    def _consolidation_win(self) -> TradeRecord:
        r = _win(2.0)
        return TradeRecord(**{**r.to_dict(), "context": "CONSOLIDATION"})

    def _h1_win(self) -> TradeRecord:
        r = _win(2.0)
        return TradeRecord(**{**r.to_dict(), "timeframe": "H1"})

    def test_trades_by_bias(self):
        records = [_win(), self._sell_win(), _win()]
        grouped = trades_by_bias(records)
        assert len(grouped["BUY"]) == 2
        assert len(grouped["SELL"]) == 1

    def test_trades_by_context(self):
        records = [_win(), self._consolidation_win()]
        grouped = trades_by_context(records)
        assert "TREND" in grouped
        assert "CONSOLIDATION" in grouped

    def test_trades_by_timeframe(self):
        records = [_win(), self._h1_win()]
        grouped = trades_by_timeframe(records)
        assert "M5" in grouped
        assert "H1" in grouped

    def test_pending_excluded_from_grouping(self):
        records = [_win(), _pending()]
        grouped = trades_by_bias(records)
        assert sum(len(v) for v in grouped.values()) == 1


# ---------------------------------------------------------------------------
# 11. TestComputeSummary
# ---------------------------------------------------------------------------


class TestComputeSummary:
    def test_all_fields_populated(self):
        records = [_win(2.0), _win(3.0), _loss(), _loss()]
        s = compute_summary(records)
        assert s.total_trades == 4
        assert s.wins == 2
        assert s.losses == 2
        assert s.win_rate == pytest.approx(0.5)
        assert s.avg_win_r == pytest.approx(2.5)
        assert s.avg_loss_r == pytest.approx(1.0)
        assert s.profit_factor == pytest.approx(2.5)
        assert s.expectancy_r == pytest.approx(0.75)
        assert s.total_r == pytest.approx(3.0)

    def test_zero_trade_safe_defaults(self):
        s = compute_summary([])
        assert s.total_trades == 0
        assert s.profit_factor == 0.0
        assert math.isnan(s.sharpe_r)

    def test_returns_performance_summary_instance(self):
        assert isinstance(compute_summary([_win()]), PerformanceSummary)

    def test_max_consecutive_in_summary(self):
        records = [_win(), _win(), _loss()]
        s = compute_summary(records)
        assert s.max_consecutive_wins == 2
        assert s.max_consecutive_losses == 1

    def test_pending_ignored_in_summary(self):
        records = [_win(), _pending()]
        s = compute_summary(records)
        assert s.total_trades == 1


# ---------------------------------------------------------------------------
# 12. TestBestWorstTrade
# ---------------------------------------------------------------------------


class TestBestWorstTrade:
    def test_best_trade_highest_pnl(self):
        records = [_win(2.0), _win(5.0), _loss()]
        assert best_trade(records).pnl_r == pytest.approx(5.0)

    def test_worst_trade_lowest_pnl(self):
        records = [_win(2.0), _loss(), _win(3.0)]
        assert worst_trade(records).pnl_r == pytest.approx(-1.0)

    def test_best_none_on_empty(self):
        assert best_trade([]) is None

    def test_worst_none_on_empty(self):
        assert worst_trade([]) is None

    def test_pending_excluded(self):
        assert best_trade([_pending()]) is None


# ---------------------------------------------------------------------------
# 13. TestBacktestDictCompat
# ---------------------------------------------------------------------------


class TestBacktestDictCompat:
    """All analytics functions must accept dicts from run_backtest() trade list."""

    def _wdict(self, rr: float = 2.0) -> dict:
        return {
            "result": "WIN",
            "rr":     rr,
            "bias":   "BUY",
            "context": "TREND",
            "timeframe": "M5",
        }

    def _ldict(self) -> dict:
        return {
            "result": "LOSS",
            "rr":     2.0,
            "bias":   "SELL",
            "context": "REVERSAL",
            "timeframe": "M5",
        }

    def test_win_rate_dicts(self):
        assert win_rate([self._wdict(), self._ldict()]) == pytest.approx(0.5)

    def test_profit_factor_dicts(self):
        assert profit_factor([self._wdict(2.0), self._ldict()]) == pytest.approx(2.0)

    def test_expectancy_r_dicts(self):
        # (2.0 + (-1.0)) / 2 = 0.5
        assert expectancy_r([self._wdict(2.0), self._ldict()]) == pytest.approx(0.5)

    def test_equity_curve_r_dicts(self):
        curve = equity_curve_r([self._wdict(2.0), self._ldict()])
        assert curve == pytest.approx([0.0, 2.0, 1.0])

    def test_max_drawdown_r_dicts(self):
        records = [self._wdict(2.0), self._ldict()]
        assert max_drawdown_r(records) == pytest.approx(1.0)

    def test_compute_summary_dicts(self):
        records = [self._wdict(2.0), self._wdict(3.0), self._ldict()]
        s = compute_summary(records)
        assert s.total_trades == 3
        assert s.wins == 2
        assert s.losses == 1

    def test_trades_by_bias_dicts(self):
        records = [self._wdict(), self._ldict()]
        grouped = trades_by_bias(records)
        assert "BUY" in grouped
        assert "SELL" in grouped

    def test_trades_by_context_dicts(self):
        records = [self._wdict(), self._ldict()]
        grouped = trades_by_context(records)
        assert "TREND" in grouped
        assert "REVERSAL" in grouped


# ---------------------------------------------------------------------------
# 14. TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_win_rate(self):
        records = [_win(2.0), _loss(), _win(3.0)]
        assert win_rate(records) == win_rate(records)

    def test_same_input_same_summary(self):
        records = [_win(2.0), _loss()]
        s1 = compute_summary(records)
        s2 = compute_summary(records)
        assert s1 == s2

    def test_equity_curve_deterministic(self):
        records = [_win(2.0), _loss(), _win(1.5)]
        assert equity_curve_r(records) == equity_curve_r(records)
