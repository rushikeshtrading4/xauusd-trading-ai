"""Historical backtesting engine for the XAUUSD trading system.

Simulates trading performance over a complete historical OHLCV DataFrame by
running the full signal pipeline on a rolling window and resolving each
generated trade against future candles.

Design principles
-----------------
No lookahead bias
    Each iteration feeds only ``df.iloc[:i]`` (candles before bar *i*) into
    the signal pipeline.  The direction of future price is never visible to
    the signal generator.

Conservative trade resolution
    When a single candle simultaneously touches both the stop-loss and the
    take-profit level the trade is counted as a loss.  This matches the
    worst-case intra-bar path assumption used in institutional backtesting.

One trade at a time
    A new trade cannot be opened until the previous one has been resolved.
    The outer loop skips every bar that was already scanned by the active
    trade's inner resolution loop, preventing overlapping positions.

Realistic entry price
    Trades are entered at the *open* of the candle immediately following the
    signal bar, matching real-world execution where the order is placed after
    the signal candle closes.

Compound equity model
    Risk is expressed as a fixed percentage of the *current* equity at the
    moment each trade is entered.  This produces realistic compounding
    behaviour over long data series.

No randomness
    All outcomes are fully deterministic for a given input DataFrame.
"""

from __future__ import annotations

import pandas as pd

from execution.signal_engine import generate_signal_dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_START_EQUITY:   float = 10_000.0   # initial account balance
_RISK_PCT:       float = 0.005      # 0.5 % of equity risked per trade
_WARMUP_BARS:    int   = 50         # minimum bars before first signal attempt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame) -> dict:
    """Simulate historical trading performance on an OHLCV DataFrame.

    The function iterates from bar ``_WARMUP_BARS`` to the end of *df*,
    feeding a growing window of candles into the production signal pipeline.
    For each validated signal it resolves the trade against subsequent candles
    and updates the equity curve.

    Parameters
    ----------
    df : pd.DataFrame
        Historical OHLCV data sorted in ascending chronological order.
        Required columns: ``timestamp``, ``open``, ``high``, ``low``,
        ``close``, ``volume``.

    Returns
    -------
    dict
        ============== =============================================================
        Key            Description
        ============== =============================================================
        total_trades   Number of trades that reached a definitive WIN or LOSS.
        wins           Trades that hit take-profit before stop-loss.
        losses         Trades that hit stop-loss (including same-candle conflicts).
        win_rate       ``wins / total_trades`` (0.0 when no trades).
        avg_rr         Mean risk-reward of all resolved trades (0.0 when no trades).
        expectancy     ``(win_rate * avg_rr) - (loss_rate * 1.0)``.
        max_drawdown   Peak-to-trough equity decline as a fraction (e.g. 0.05 = 5%).
        equity_curve   List of equity snapshots: initial value + one per resolved
                       trade in chronological order.
        ============== =============================================================
    """
    equity: float              = _START_EQUITY
    peak_equity: float         = _START_EQUITY
    max_drawdown: float        = 0.0
    equity_curve: list[float]  = [equity]
    trades: list[dict]         = []
    in_trade: bool             = False
    next_trade_bar: int        = 0   # earliest outer-loop index for a new trade

    n = len(df)

    for i in range(_WARMUP_BARS, n):
        # ------------------------------------------------------------------ #
        # Fix 1 — one trade at a time                                        #
        # Skip bars that belong to the active trade's forward scan window.   #
        # ------------------------------------------------------------------ #
        if i < next_trade_bar:
            continue
        in_trade = False

        # ------------------------------------------------------------------ #
        # Step 1 — feed only historical data (no lookahead)                  #
        # ------------------------------------------------------------------ #
        current_data = df.iloc[:i]
        market_data  = {"M5": current_data}

        signal = generate_signal_dict(market_data)
        if signal is None:
            continue

        # ------------------------------------------------------------------ #
        # Fix 2 — realistic entry price                                      #
        # Enter at the open of the next candle, not the signal's midpoint.   #
        # ------------------------------------------------------------------ #
        if i + 1 >= n:
            continue   # no candle available to enter on

        entry       = float(df.iloc[i + 1]["open"])   # next candle open
        stop_loss   = float(signal["stop_loss"])
        take_profit = float(signal["take_profit"])
        rr          = float(signal["risk_reward"])
        bias        = str(signal["bias"]).upper()   # "BULLISH" or "BEARISH"

        risk_amount = _RISK_PCT * equity
        in_trade    = True   # trade is now open

        # ------------------------------------------------------------------ #
        # Step 3 — simulate trade outcome on future candles                  #
        # Resolution starts at i+2: entry is at i+1 (the open of that       #
        # candle) so the first candle whose high/low can trigger SL/TP is    #
        # the one AFTER entry, preventing same-candle entry+exit.            #
        # ------------------------------------------------------------------ #
        outcome: str | None = None

        for j in range(i + 2, n):
            candle = df.iloc[j]
            low    = float(candle["low"])
            high   = float(candle["high"])

            if bias == "BULLISH":
                sl_hit = low  <= stop_loss
                tp_hit = high >= take_profit
            else:   # BEARISH / SELL
                sl_hit = high >= stop_loss
                tp_hit = low  <= take_profit

            # Conservative: same-candle conflict → LOSS
            if sl_hit and tp_hit:
                outcome = "LOSS"
                break
            if sl_hit:
                outcome = "LOSS"
                break
            if tp_hit:
                outcome = "WIN"
                break

        in_trade = False   # trade resolved (WIN/LOSS) or exhausted

        if outcome is None:
            # Trade never resolved within remaining data — skip, no PnL.
            continue

        next_trade_bar = j   # block bars (i+1)..(j-1) from opening new trades

        # ------------------------------------------------------------------ #
        # Step 4 — update equity                                             #
        # ------------------------------------------------------------------ #
        if outcome == "WIN":
            equity += rr * risk_amount
        else:
            equity -= risk_amount

        # ------------------------------------------------------------------ #
        # Step 5 — update max drawdown                                       #
        # ------------------------------------------------------------------ #
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        equity_curve.append(equity)
        trades.append({"result": outcome, "rr": rr, "equity": equity})

    # -----------------------------------------------------------------------
    # Compute summary metrics
    # -----------------------------------------------------------------------
    total_trades = len(trades)

    if total_trades == 0:
        return {
            "total_trades": 0,
            "wins":         0,
            "losses":       0,
            "win_rate":     0.0,
            "avg_rr":       0.0,
            "expectancy":   0.0,
            "max_drawdown": max_drawdown,
            "equity_curve": equity_curve,
        }

    wins     = sum(1 for t in trades if t["result"] == "WIN")
    losses   = total_trades - wins
    win_rate = wins / total_trades
    loss_rate = 1.0 - win_rate
    avg_rr   = sum(t["rr"] for t in trades) / total_trades
    expectancy = (win_rate * avg_rr) - (loss_rate * 1.0)

    return {
        "total_trades": total_trades,
        "wins":         wins,
        "losses":       losses,
        "win_rate":     win_rate,
        "avg_rr":       avg_rr,
        "expectancy":   expectancy,
        "max_drawdown": max_drawdown,
        "equity_curve": equity_curve,
    }
