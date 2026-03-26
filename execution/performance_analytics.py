"""
Performance analytics module for XAUUSD trade records.

Pure functions over a list of TradeRecord objects (or the equivalent
list of dicts from run_backtest()). No external dependencies beyond
the standard library. All functions are deterministic and have no
side effects.

Why R-multiples
───────────────
All P&L is expressed in R (units of risk) rather than currency. This
makes statistics account-size-agnostic and directly comparable across
different risk-per-trade settings. A trade that risks $100 and wins
$200 and a trade that risks $10 and wins $20 both have pnl_r = +2.0.

Function catalogue
──────────────────
compute_summary(records)         → PerformanceSummary dataclass
win_rate(records)                → float
profit_factor(records)           → float
expectancy_r(records)            → float
avg_win_r(records)               → float
avg_loss_r(records)              → float
max_consecutive_wins(records)    → int
max_consecutive_losses(records)  → int
max_drawdown_r(records)          → float   (peak-to-trough in R)
sharpe_r(records, periods=252)   → float   (annualised Sharpe in R-space)
best_trade(records)              → TradeRecord | None
worst_trade(records)             → TradeRecord | None
trades_by_bias(records)          → dict[str, list[TradeRecord]]
trades_by_context(records)       → dict[str, list[TradeRecord]]
trades_by_timeframe(records)     → dict[str, list[TradeRecord]]
equity_curve_r(records)          → list[float]  (cumulative R from 0)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from execution.trade_logger import TradeRecord


# ---------------------------------------------------------------------------
# PerformanceSummary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerformanceSummary:
    """Immutable snapshot of all key performance metrics."""

    total_trades:           int
    wins:                   int
    losses:                 int
    win_rate:               float   # 0.0 – 1.0
    avg_win_r:              float
    avg_loss_r:             float   # expressed as a positive number (magnitude)
    profit_factor:          float   # gross_wins_r / gross_losses_r; inf if no losses
    expectancy_r:           float   # expected R per trade
    max_consecutive_wins:   int
    max_consecutive_losses: int
    max_drawdown_r:         float   # peak-to-trough expressed in R (positive number)
    sharpe_r:               float   # annualised Sharpe in R-space (NaN if < 2 trades)
    total_r:                float   # net cumulative R (sum of all pnl_r)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pnl(r) -> float | None:
    """Extract pnl_r from a TradeRecord or backtest dict."""
    if isinstance(r, dict):
        result = r.get("result")
        rr     = r.get("rr", 0.0)
        if result == "WIN":
            return float(rr)
        elif result == "LOSS":
            return -1.0
        return None
    return r.pnl_r


def _outcome(r) -> str:
    if isinstance(r, dict):
        return r.get("result", "")
    return r.outcome


def _resolved(records: list) -> list:
    """Return only WIN/LOSS records with a valid pnl_r."""
    return [
        r for r in records
        if _outcome(r) in ("WIN", "LOSS") and _pnl(r) is not None
    ]


def _max_streak(resolved: list, target_outcome: str) -> int:
    current = 0
    best    = 0
    for r in resolved:
        if _outcome(r) == target_outcome:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_summary(records: list) -> PerformanceSummary:
    """Compute all performance metrics from a list of TradeRecord or backtest dicts.

    Parameters
    ----------
    records : list
        List of TradeRecord objects or dicts from run_backtest() trades list.
        Pending / cancelled records are silently skipped.

    Returns
    -------
    PerformanceSummary
        All metrics as a frozen dataclass. Zero-safe: returns sensible
        defaults (0.0 / NaN) when there are no resolved trades.
    """
    resolved = _resolved(records)
    total    = len(resolved)

    if total == 0:
        return PerformanceSummary(
            total_trades=0, wins=0, losses=0,
            win_rate=0.0, avg_win_r=0.0, avg_loss_r=0.0,
            profit_factor=0.0, expectancy_r=0.0,
            max_consecutive_wins=0, max_consecutive_losses=0,
            max_drawdown_r=0.0, sharpe_r=float("nan"), total_r=0.0,
        )

    pnls = [_pnl(r) for r in resolved]
    wins_list   = [p for p in pnls if p is not None and p > 0]
    losses_list = [p for p in pnls if p is not None and p < 0]

    n_wins   = len(wins_list)
    n_losses = len(losses_list)

    _win_rate     = n_wins / total
    _avg_win      = sum(wins_list)   / n_wins   if n_wins   else 0.0
    _avg_loss_mag = (
        sum(abs(l) for l in losses_list) / n_losses if n_losses else 0.0
    )

    gross_wins   = sum(wins_list)
    gross_losses = sum(abs(l) for l in losses_list)
    _pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    _exp     = sum(p for p in pnls if p is not None) / total
    _total_r = sum(p for p in pnls if p is not None)

    _sharpe = (
        round(sharpe_r(records), 4) if total >= 2 else float("nan")
    )

    return PerformanceSummary(
        total_trades           = total,
        wins                   = n_wins,
        losses                 = n_losses,
        win_rate               = round(_win_rate, 4),
        avg_win_r              = round(_avg_win, 4),
        avg_loss_r             = round(_avg_loss_mag, 4),
        profit_factor          = round(_pf, 4) if math.isfinite(_pf) else _pf,
        expectancy_r           = round(_exp, 4),
        max_consecutive_wins   = max_consecutive_wins(records),
        max_consecutive_losses = max_consecutive_losses(records),
        max_drawdown_r         = round(max_drawdown_r(records), 4),
        sharpe_r               = _sharpe,
        total_r                = round(_total_r, 4),
    )


def win_rate(records: list) -> float:
    """Return wins / total resolved trades. 0.0 if no trades."""
    resolved = _resolved(records)
    if not resolved:
        return 0.0
    wins = sum(1 for r in resolved if _outcome(r) == "WIN")
    return wins / len(resolved)


def profit_factor(records: list) -> float:
    """Return gross_wins_r / gross_losses_r. Returns inf if no losses."""
    resolved = _resolved(records)
    pnls = [_pnl(r) for r in resolved if _pnl(r) is not None]
    gross_wins   = sum(p for p in pnls if p > 0)
    gross_losses = sum(abs(p) for p in pnls if p < 0)
    return gross_wins / gross_losses if gross_losses > 0 else float("inf")


def expectancy_r(records: list) -> float:
    """Return the average R per trade. 0.0 if no trades."""
    resolved = _resolved(records)
    pnls = [_pnl(r) for r in resolved if _pnl(r) is not None]
    return sum(pnls) / len(pnls) if pnls else 0.0


def avg_win_r(records: list) -> float:
    """Return the average R of winning trades. 0.0 if no wins."""
    resolved = _resolved(records)
    wins = [
        _pnl(r) for r in resolved
        if _outcome(r) == "WIN" and _pnl(r) is not None
    ]
    return sum(wins) / len(wins) if wins else 0.0


def avg_loss_r(records: list) -> float:
    """Return the average magnitude of losing trades in R. 0.0 if no losses."""
    resolved = _resolved(records)
    losses = [
        abs(_pnl(r)) for r in resolved
        if _outcome(r) == "LOSS" and _pnl(r) is not None
    ]
    return sum(losses) / len(losses) if losses else 0.0


def max_consecutive_wins(records: list) -> int:
    """Return the longest consecutive WIN streak."""
    return _max_streak(_resolved(records), "WIN")


def max_consecutive_losses(records: list) -> int:
    """Return the longest consecutive LOSS streak."""
    return _max_streak(_resolved(records), "LOSS")


def max_drawdown_r(records: list) -> float:
    """Return the peak-to-trough drawdown expressed in R (positive number)."""
    curve = equity_curve_r(records)
    if len(curve) < 2:
        return 0.0
    peak   = curve[0]
    max_dd = 0.0
    for val in curve[1:]:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    return max_dd


def sharpe_r(records: list, periods: int = 252) -> float:
    """Return the annualised Sharpe ratio in R-space.

    Uses the per-trade R series as returns. Annualises by √periods.
    Returns NaN when fewer than 2 resolved trades exist or stdev is zero.

    Parameters
    ----------
    records : list
        Trade records.
    periods : int
        Number of trades per year for annualisation. Default 252.
    """
    resolved = _resolved(records)
    pnls = [_pnl(r) for r in resolved if _pnl(r) is not None]
    if len(pnls) < 2:
        return float("nan")
    mean = sum(pnls) / len(pnls)
    std  = statistics.stdev(pnls)
    if std == 0.0:
        return float("nan")
    return (mean / std) * math.sqrt(periods)


def best_trade(records: list):
    """Return the TradeRecord with the highest pnl_r, or None."""
    resolved = _resolved(records)
    if not resolved:
        return None
    return max(resolved, key=lambda r: _pnl(r) or float("-inf"))


def worst_trade(records: list):
    """Return the TradeRecord with the lowest pnl_r, or None."""
    resolved = _resolved(records)
    if not resolved:
        return None
    return min(resolved, key=lambda r: _pnl(r) or float("inf"))


def equity_curve_r(records: list) -> list[float]:
    """Return the cumulative R curve starting from 0.

    The first element is always 0.0 (before any trade).
    """
    resolved = _resolved(records)
    pnls = [_pnl(r) for r in resolved if _pnl(r) is not None]
    curve: list[float] = [0.0]
    running = 0.0
    for p in pnls:
        running += p
        curve.append(round(running, 6))
    return curve


def trades_by_bias(records: list) -> dict[str, list]:
    """Group resolved trades by bias ("BUY" / "SELL")."""
    result: dict[str, list] = {}
    for r in _resolved(records):
        key = r.bias if not isinstance(r, dict) else r.get("bias", "UNKNOWN")
        result.setdefault(key, []).append(r)
    return result


def trades_by_context(records: list) -> dict[str, list]:
    """Group resolved trades by context ("TREND" / "CONSOLIDATION" / "REVERSAL")."""
    result: dict[str, list] = {}
    for r in _resolved(records):
        key = r.context if not isinstance(r, dict) else r.get("context", "UNKNOWN")
        result.setdefault(key, []).append(r)
    return result


def trades_by_timeframe(records: list) -> dict[str, list]:
    """Group resolved trades by timeframe (e.g. "M5")."""
    result: dict[str, list] = {}
    for r in _resolved(records):
        key = r.timeframe if not isinstance(r, dict) else r.get("timeframe", "UNKNOWN")
        result.setdefault(key, []).append(r)
    return result
