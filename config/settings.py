"""
Centralised configuration for the XAUUSD AI trading system.

This is the single source of truth for every strategy parameter, threshold,
and tunable constant used anywhere in the system.  Modules carry their own
private copies of constants they need (to avoid circular imports); this file
documents what those values ARE and provides them for the dashboard, UI layer,
and any future live parameter injection.

How to use this file
────────────────────
- Read values from here in the UI / dashboard layer.
- When tuning the strategy, change the value here AND in the corresponding
  module's private constant.  The docstring on each constant names the
  module(s) that must also be updated.
- Never import from settings.py inside analysis, ai, execution, data, or
  indicators modules.  Those modules are self-contained by design.

Parameter sections
──────────────────
1.  Symbol and timeframes           (already present — unchanged)
2.  ATR validation contract         (shared across 6+ modules)
3.  Indicator defaults              (ATR / EMA / RSI / VWAP periods)
4.  Signal quality gates            (min RR, confidence, probability)
5.  Risk management                 (daily loss cap, position limits, SL gate)
6.  Trade geometry                  (SL/TP ATR multiples, price precision)
7.  Displacement detection          (body and move ATR factors)
8.  Backtesting defaults            (equity, risk %, warmup)
9.  XAUUSD instrument specifics     (lot size, placeholder base price)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Symbol and timeframes  (UNCHANGED — preserved exactly)
# ---------------------------------------------------------------------------

SUPPORTED_SYMBOL: str = "XAUUSD"

# Ordered from highest to lowest resolution — matches the multi-timeframe
# model described in docs/TRADING_STRATEGY.md:
#   Bias   → D1 / H4 / H1
#   Levels → D1 / H4 / H1 / M15
#   Entry  → M5
#   Precision → M1

TIMEFRAMES: list[str] = ["D1", "H4", "H1", "M15", "M5", "M1"]

# Maps each timeframe key to its pandas frequency string.  This is the
# single source of truth for candle spacing across the entire codebase.
TIMEFRAME_FREQ: dict[str, str] = {
    "M1":  "1min",
    "M5":  "5min",
    "M15": "15min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1d",
}

# ---------------------------------------------------------------------------
# 2. ATR validation contract
#    Used by: analysis/displacement.py, analysis/fair_value_gaps.py,
#             analysis/order_blocks.py, ai/signal_engine.py,
#             execution/risk_manager.py, execution/signal_engine.py
#
#    Rule: ATR values below ATR_MIN are warm-up / stale-feed artifacts
#    and must be rejected.  Values above ATR_CAP are spike anomalies and
#    are clamped rather than rejected.
# ---------------------------------------------------------------------------

# Minimum usable ATR.  Below this the feed has not warmed up or the market
# is synthetically flat.  Signals generated with ATR below this value would
# have degenerate stop-loss geometry.
ATR_MIN: float = 0.1

# Maximum usable ATR.  Values above this indicate a data spike; they are
# clamped to this ceiling to prevent absurdly wide stops.
ATR_CAP: float = 100.0

# ---------------------------------------------------------------------------
# 3. Indicator defaults
#    Used by: indicators/indicator_engine.py (default arguments)
#
#    These are Wilder's original periods — the gold standard for institutional
#    charting platforms (MetaTrader, TradingView, Bloomberg).
# ---------------------------------------------------------------------------

# Wilder's ATR smoothing period.
ATR_PERIOD: int = 14

# EMA periods computed by default in the indicator pipeline.
# EMA_20 → near-term trend; EMA_50 → medium-term; EMA_200 → macro trend.
EMA_PERIODS: list[int] = [20, 50, 200]

# Wilder's RSI period.
RSI_PERIOD: int = 14

# ---------------------------------------------------------------------------
# 4. Signal quality gates
#    Used by: ai/signal_engine.py, execution/risk_manager.py,
#             execution/potential_setup.py, execution/signal_engine.py
#
#    These three values form the triple gate that every candidate setup
#    must clear before a trade card is emitted.
# ---------------------------------------------------------------------------

# Minimum risk-reward ratio.  Strategy requires 1:2; below this the
# potential reward does not justify the capital at risk.
MIN_RR: float = 2.0

# Minimum signal confidence score (0–100).  Setups scoring below this
# after all confluence factors are applied are suppressed entirely.
MIN_CONFIDENCE: float = 70.0

# Minimum probability score from the probability model (0–100).
# Setups below this lack sufficient institutional confluence.
MIN_PROBABILITY: int = 60

# ---------------------------------------------------------------------------
# 5. Risk management
#    Used by: execution/risk_manager.py, backtesting/backtest_engine.py
# ---------------------------------------------------------------------------

# Maximum daily loss expressed as a fraction of account equity.
# 0.01 = 1 %.  Breaching this cap blocks all new signals for the session.
DAILY_LOSS_LIMIT_PCT: float = 0.01

# Maximum number of simultaneously open positions.
MAX_OPEN_TRADES: int = 2

# Stop-loss distance must not exceed this multiple of ATR.  Stops wider
# than this indicate either a bad entry or an extremely volatile bar that
# should not be traded.
MAX_SL_ATR_MULTIPLE: float = 2.0

# ---------------------------------------------------------------------------
# 6. Trade geometry
#    Used by: ai/signal_engine.py (_SL_ATR_MULTIPLE, _TP_RR_MULTIPLE),
#             execution/trade_setup.py (atr_buffer = 0.5 × ATR),
#             execution/rr_calculator.py (_RR_DECIMALS, _MIN_DISTANCE),
#             execution/position_sizing.py (_XAUUSD_LOT_UNITS, _MIN_SL_DISTANCE)
# ---------------------------------------------------------------------------

# Stop-loss placed this many ATR widths beyond the order-block boundary.
# Provides breathing room beyond normal candle noise.
SL_ATR_MULTIPLE: float = 0.5

# Take-profit target as a multiple of risk in ai/signal_engine.py.
# NOTE: execution/trade_setup.py uses a separate 2.5× dynamic target — these
# are intentionally different (signal engine uses fixed 2×; trade_setup uses
# dynamic 2.5× capped at nearest liquidity level).
TP_RR_MULTIPLE_SIGNAL: float = 2.0    # ai/signal_engine.py
TP_RR_MULTIPLE_TRADE:  float = 2.5    # execution/trade_setup.py

# Decimal precision for output prices (XAUUSD standard: 5 d.p.).
PRICE_DECIMALS: int = 5

# Decimal precision for risk-reward ratio output.
RR_DECIMALS: int = 2

# Minimum meaningful price distance for XAUUSD.  Distances below this
# indicate degenerate geometry (entry == stop, or rounding artefact).
MIN_PRICE_DISTANCE: float = 0.01

# ---------------------------------------------------------------------------
# 7. Displacement detection
#    Used by: analysis/displacement.py (_BODY_ATR_FACTOR, _MOVE_ATR_FACTOR),
#             analysis/fair_value_gaps.py (_DISPLACEMENT_ATR_FACTOR),
#             analysis/order_blocks.py (_DISPLACEMENT_ATR_FACTOR)
#
#    A displacement is a genuine institutional impulse move.  These thresholds
#    separate institutional order flow from retail noise.
# ---------------------------------------------------------------------------

# Minimum mean candle body (as ATR multiple) to qualify as displacement.
# Bodies smaller than this are indecision candles, not institutional intent.
DISPLACEMENT_BODY_ATR_FACTOR: float = 0.8   # analysis/displacement.py

# Minimum net price move across the window (as ATR multiple).
# Ensures the sequence covers meaningful distance, not just several tiny candles.
DISPLACEMENT_MOVE_ATR_FACTOR: float = 2.0   # analysis/displacement.py

# Single-candle displacement threshold (used for OB and FVG detection).
# The candle body must exceed 1.5 × ATR to confirm institutional intent.
DISPLACEMENT_ATR_FACTOR: float = 1.5        # analysis/order_blocks.py,
                                             # analysis/fair_value_gaps.py

# ---------------------------------------------------------------------------
# 8. Backtesting defaults
#    Used by: backtesting/backtest_engine.py
#
#    These values are realistic defaults for a retail account running the
#    strategy.  They are separate from live risk parameters because a
#    backtester may be run with different equity / risk settings.
# ---------------------------------------------------------------------------

# Starting equity for backtest simulations.
BACKTEST_START_EQUITY: float = 10_000.0

# Fraction of current equity risked per trade in the backtest.
# 0.005 = 0.5 %.  Compounds on each trade.
BACKTEST_RISK_PCT: float = 0.005

# Minimum number of candles consumed before the first signal is attempted.
# Ensures all indicators (ATR_14, EMA_200) have fully warmed up.
BACKTEST_WARMUP_BARS: int = 50

# ---------------------------------------------------------------------------
# 9. XAUUSD instrument specifics
# ---------------------------------------------------------------------------

# Standard lot size for XAUUSD: 1 lot = 100 troy ounces.
# Used by execution/position_sizing.py to convert risk amount to lot size.
LOT_UNITS: int = 100

# Synthetic base price used only when generating placeholder candle data
# for development and unit-testing.  Never used in live signal logic.
PLACEHOLDER_BASE_PRICE: float = 2_650.00

# Number of candles generated per timeframe in placeholder mode.
PLACEHOLDER_CANDLE_COUNT: int = 300
