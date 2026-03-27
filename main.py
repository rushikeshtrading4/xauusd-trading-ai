"""
XAUUSD AI Trading System
Entry point — orchestrates the full data → signal → log pipeline.

Architecture
────────────
main.py owns only three responsibilities:

1. Lifecycle management  — startup, shutdown, cycle timing, error recovery.
2. Macro state injection — exposes update_macro_state() so an operator
   (or future live feed) can inject DXY/yield trends without restarting.
3. Persistence           — saves the trade log to disk after each resolved
   signal and prints a rolling performance summary every N cycles.

Macro state
───────────
All macro state is managed through the SHARED handler in
execution.signal_engine.  Calling get_macro_handler() returns the same
instance that signal generation reads from, ensuring that events registered
here (e.g. NFP, CPI) actually trigger the news blackout gate during signal
generation.

Data source
───────────
Change source= in load_market_data() to switch between:
  "placeholder" — synthetic data for development/testing (default)
  "oanda"       — live OANDA REST API data
"""

import datetime
import logging
import os
import sys
import time

from config.settings import SUPPORTED_SYMBOL
from data.data_loader import load_market_data
from data.macro_data import TREND_RISING, TREND_FALLING, TREND_NEUTRAL
from execution.signal_engine import (
    generate_signal_dict,
    get_macro_handler,        # ← shared handler; no longer a separate instance
)
from execution.signal_formatter import format_trade_signal
from execution.trade_logger import TradeLogger
from execution.performance_analytics import compute_summary

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CYCLE_INTERVAL_SECONDS: int = 300
AUTOSAVE_EVERY_N_CYCLES: int = 10
TRADE_LOG_PATH: str = "logs/trades.json"
SUMMARY_EVERY_N_CYCLES: int = 20

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    """Create logs/ directory and configure three log streams."""
    os.makedirs("logs", exist_ok=True)

    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(console_handler)

    engine_handler = logging.FileHandler("logs/engine.log", encoding="utf-8")
    engine_handler.setLevel(logging.INFO)
    engine_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(engine_handler)

    signal_handler = logging.FileHandler("logs/signals.log", encoding="utf-8")
    signal_handler.setLevel(logging.INFO)
    signal_handler.setFormatter(logging.Formatter(fmt))
    logging.getLogger("signals").addHandler(signal_handler)
    logging.getLogger("signals").propagate = True

    error_handler = logging.FileHandler("logs/errors.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(error_handler)


_setup_logging()

logger        = logging.getLogger("main")
signal_logger = logging.getLogger("signals")

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_trade_logger = TradeLogger()

# ---------------------------------------------------------------------------
# Public helpers — macro state injection (all routed to shared handler)
# ---------------------------------------------------------------------------


def update_macro_state(
    dxy_trend:   str = TREND_NEUTRAL,
    yield_trend: str = TREND_NEUTRAL,
) -> None:
    """Inject the current DXY and yield trends into the shared macro handler.

    This updates the SAME handler instance used by the signal engine, so
    changes here immediately affect news blackout gating and gold sentiment
    scoring in the next signal generation cycle.

    Parameters
    ----------
    dxy_trend : str
        "RISING", "FALLING", or "NEUTRAL".
    yield_trend : str
        "RISING", "FALLING", or "NEUTRAL".
    """
    handler = get_macro_handler()
    handler.update_dxy(dxy_trend)
    handler.update_yields(yield_trend)
    logger.info(
        "Macro state updated — DXY: %s  Yields: %s  Gold sentiment: %s",
        dxy_trend, yield_trend, handler.state.gold_sentiment,
    )


def add_macro_event(
    name: str,
    scheduled_utc: datetime.datetime,
    impact: str = "HIGH",
) -> None:
    """Register a high-impact event (CPI, NFP, FOMC) in the shared macro handler.

    The signal engine will suppress signals in the ±30-minute window around
    the event's scheduled UTC time.

    Parameters
    ----------
    name : str
        Human-readable event name, e.g. "NFP", "CPI".
    scheduled_utc : datetime.datetime
        Timezone-aware UTC datetime of the release.
    impact : str
        "HIGH" (default) or "LOW".
    """
    handler = get_macro_handler()
    handler.add_event(name, scheduled_utc, impact=impact)
    logger.info(
        "Macro event registered — %s at %s (impact: %s)",
        name, scheduled_utc.isoformat(), impact,
    )


# ---------------------------------------------------------------------------
# Core cycle
# ---------------------------------------------------------------------------


def run_cycle() -> bool:
    """Execute one complete analysis cycle.

    Returns True if a signal was generated and logged; False otherwise.
    """
    # Step 1: Fetch market data for ALL configured timeframes.
    # The execution signal engine uses M5 for entry signals and
    # D1/H4/H1 for multi-timeframe bias computation.
    try:
        market_data = load_market_data(SUPPORTED_SYMBOL, source="placeholder")
    except Exception as exc:
        logger.error("Data fetch failed: %s", exc, exc_info=True)
        return False

    # Step 2: Run full pipeline (MTF bias now wired inside signal engine)
    try:
        signal = generate_signal_dict(market_data)
    except Exception as exc:
        logger.error("Signal engine error: %s", exc, exc_info=True)
        return False

    if signal is None:
        logger.debug("No trade setup found this cycle.")
        return False

    # Step 3: Format and emit
    trade_card = format_trade_signal(signal)
    signal_logger.info("\n%s", trade_card)
    print(f"\n{'=' * 60}\n{trade_card}\n{'=' * 60}\n")

    # Step 4: Log the signal
    try:
        ts = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        _trade_logger.log_signal(signal, timestamp_utc=ts)
        logger.info(
            "Signal logged — %s %s  Entry: %.2f  RR: %.1f  Confidence: %.0f%%  "
            "MTF: %s (%s)  Prob: %d",
            signal.get("bias"),
            signal.get("timeframe"),
            signal.get("entry", 0.0),
            signal.get("risk_reward", 0.0),
            signal.get("confidence", 0.0),
            signal.get("mtf_bias", "?"),
            signal.get("bias_strength", "?"),
            signal.get("probability", 0),
        )
    except Exception as exc:
        logger.warning("Trade log error (non-fatal): %s", exc)

    return True


def _maybe_autosave(cycle_count: int) -> None:
    if AUTOSAVE_EVERY_N_CYCLES <= 0:
        return
    if cycle_count % AUTOSAVE_EVERY_N_CYCLES == 0:
        try:
            _trade_logger.save(TRADE_LOG_PATH)
            logger.info("Trade log saved -> %s", TRADE_LOG_PATH)
        except Exception as exc:
            logger.warning("Trade log save failed: %s", exc)


def _maybe_print_summary(cycle_count: int) -> None:
    if SUMMARY_EVERY_N_CYCLES <= 0:
        return
    if cycle_count % SUMMARY_EVERY_N_CYCLES == 0:
        records = _trade_logger.resolved_records()
        if not records:
            return
        s = compute_summary(records)
        logger.info(
            "Performance summary — Trades: %d  WR: %.1f%%  "
            "Expectancy: %.2fR  MaxDD: %.2fR  PF: %.2f",
            s.total_trades, s.win_rate * 100,
            s.expectancy_r, s.max_drawdown_r, s.profit_factor,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the XAUUSD analysis engine and run the continuous cycle loop."""
    handler = get_macro_handler()
    logger.info("=" * 60)
    logger.info("XAUUSD AI Trading System — starting up")
    logger.info(
        "Symbol: %s  |  Cycle interval: %ds", SUPPORTED_SYMBOL, CYCLE_INTERVAL_SECONDS
    )
    logger.info("Data source: placeholder (synthetic — no live feed)")
    logger.info(
        "Macro state: DXY=%s  Yields=%s  Sentiment=%s",
        handler.state.dxy_trend,
        handler.state.yield_trend,
        handler.state.gold_sentiment,
    )
    logger.info("=" * 60)

    cycle_count = 0

    try:
        while True:
            cycle_count += 1
            logger.info("— Cycle %d —", cycle_count)

            run_cycle()
            _maybe_autosave(cycle_count)
            _maybe_print_summary(cycle_count)

            logger.info(
                "Cycle %d complete. Sleeping %ds...", cycle_count, CYCLE_INTERVAL_SECONDS
            )
            time.sleep(CYCLE_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("Shutdown requested (Ctrl+C). Saving trade log...")
        try:
            _trade_logger.save(TRADE_LOG_PATH)
            logger.info(
                "Trade log saved -> %s  (%d records)",
                TRADE_LOG_PATH, len(_trade_logger.all_records())
            )
        except Exception as exc:
            logger.warning("Final save failed: %s", exc)
        logger.info("XAUUSD AI Trading System — shutdown complete.")


if __name__ == "__main__":
    main()