"""Execution-layer signal orchestration engine for XAUUSD.

This module is the final integration layer that wires the full analysis and
execution pipeline together into a single call.  Given a dict of multi-
timeframe OHLCV DataFrames it:

    1. Runs the complete analysis pipeline on the M5 execution timeframe.
    2. Runs analysis on H4 and D1 for multi-timeframe bias calculation.
    3. Extracts a structured market context from the latest candle.
    4. Scores the setup via the probability model.
    5. Applies the potential-setup quality gate.
    6. Builds the precise trade geometry (entry, SL, TP, RR).
    7. Validates the trade against institutional risk controls.
    8. Formats and returns the human-readable trade card.

Returns ``None`` at any gate that is not satisfied (no trade opportunity).
Returns the formatted trade-card string when all pipeline gates pass.

Architecture notes
------------------
* A single MacroDataHandler instance is shared across the module.
  Callers should inject macro state via ``get_macro_handler()`` rather than
  creating their own instances, to avoid the singleton isolation bug that
  previously caused ``main.py`` macro updates to have no effect on signals.

* Multi-timeframe bias is computed from H4 and D1 DataFrames when present in
  ``market_data``. Keys "H4" and "D1" are optional; absent keys degrade
  gracefully to NEUTRAL bias (score=0 contribution).

Pipeline position::

    market_data → execution.signal_engine.generate_signal()   ← this module
        → human-readable trade card / None

No randomness.  All logic is deterministic.
No external network calls.  All inputs arrive via *market_data*.
No partial outputs — the function either returns a complete card or None.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Analysis pipeline imports
# ---------------------------------------------------------------------------
from indicators.indicator_engine import compute_indicators
from analysis.swing_points import detect_swings
from analysis.liquidity_equal_levels import detect_equal_levels
from analysis.market_structure import detect_market_structure
from analysis.liquidity import detect_liquidity_pools
from analysis.liquidity_sweeps import detect_liquidity_sweeps
from analysis.order_blocks import detect_order_blocks
from analysis.displacement import detect_displacement
from analysis.fair_value_gaps import detect_fair_value_gaps

# ---------------------------------------------------------------------------
# Macro data
# ---------------------------------------------------------------------------
from data.macro_data import MacroDataHandler

# ---------------------------------------------------------------------------
# Signal / execution imports
# ---------------------------------------------------------------------------
from ai.bias_model import calculate_bias
from ai.probability_model import compute_probability
from execution.potential_setup import evaluate_potential_setup
from execution.trade_setup import build_trade_setup
from execution.risk_manager import evaluate_risk
from execution.signal_formatter import format_trade_signal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Execution timeframe — all entry signals are derived from M5.
_EXECUTION_TIMEFRAME = "M5"

# Higher timeframes used for bias calculation (order: highest → lowest).
_BIAS_TIMEFRAMES = ["D1", "H4", "H1"]

# Bias → account-equity parameters forwarded to the risk manager.
_DEFAULT_ACCOUNT_BALANCE: float = 10_000.0
_DEFAULT_OPEN_TRADES:     int   = 0
_DEFAULT_DAILY_LOSS:      float = 0.0

# Confidence label (from potential_setup) → numeric score used by the risk manager.
_CONFIDENCE_SCORE: dict[str, float] = {
    "HIGH":   85.0,
    "MEDIUM": 70.0,
}

# OB recency: only order blocks from the last N candles are considered "near".
# This prevents stale OBs from incorrectly flagging near_order_block=True.
_OB_RECENCY_BARS: int = 50

# ---------------------------------------------------------------------------
# Module-level shared macro handler.
#
# IMPORTANT: This is the single source of truth for macro state in the
# execution pipeline.  Use ``get_macro_handler()`` to access it from other
# modules (e.g. main.py) rather than creating new MacroDataHandler instances.
# ---------------------------------------------------------------------------
_macro_handler = MacroDataHandler()


def get_macro_handler() -> MacroDataHandler:
    """Return the shared MacroDataHandler instance used by the signal engine.

    Callers that need to register news events or update DXY/yield trends
    should use this function to obtain the same handler that signal generation
    reads from, avoiding the isolation bug where separate instances have
    independent state.

    Example::

        from execution.signal_engine import get_macro_handler
        handler = get_macro_handler()
        handler.update_dxy("FALLING")
        handler.update_yields("FALLING")
        handler.add_event("NFP", scheduled_utc)
    """
    return _macro_handler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_signal(market_data: dict) -> str | None:
    """Generate a complete, risk-validated XAUUSD trade signal as a string.

    Returns a formatted trade card string when all pipeline gates pass, or
    ``None`` when no tradeable setup exists.  For the raw signal dict use
    :func:`generate_signal_dict`.
    """
    result = generate_signal_dict(market_data)
    if result is None:
        return None
    return format_trade_signal(result)


def generate_signal_dict(market_data: dict) -> dict | None:
    """Run the full pipeline and return the raw signal dict, or ``None``.

    Parameters
    ----------
    market_data : dict
        Keys are timeframe identifiers; values are OHLCV DataFrames.
        Required key: ``"M5"`` (execution timeframe).
        Optional keys: ``"D1"``, ``"H4"``, ``"H1"`` (used for MTF bias).

    Returns
    -------
    dict or None
        Raw signal dict or ``None`` when no tradeable setup exists.
    """
    # ------------------------------------------------------------------ #
    # MACRO BLACKOUT GATE                                                 #
    # ------------------------------------------------------------------ #
    import datetime
    _now = datetime.datetime.now(tz=datetime.timezone.utc)
    if _macro_handler.is_high_impact_window(_now):
        return None

    # ------------------------------------------------------------------ #
    # STEP 1 — Extract M5 execution timeframe data                       #
    # ------------------------------------------------------------------ #
    df = market_data.get(_EXECUTION_TIMEFRAME)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None

    # ------------------------------------------------------------------ #
    # STEP 2 — Run the full analysis pipeline on M5                      #
    # ------------------------------------------------------------------ #
    try:
        df = compute_indicators(df)
        df = detect_swings(df)
        df = detect_equal_levels(df)
        df = detect_market_structure(df, timeframe=_EXECUTION_TIMEFRAME)
        df = detect_liquidity_pools(df)
        df = detect_liquidity_sweeps(df)
        df = detect_order_blocks(df)
        df = detect_displacement(df)
        df = detect_fair_value_gaps(df)
    except (ValueError, KeyError):
        return None

    if df.empty:
        return None

    # ------------------------------------------------------------------ #
    # STEP 3 — Build multi-timeframe bias from H1/H4/D1                  #
    # Each HTF DataFrame is run through the full indicator + structure    #
    # pipeline so that trend_state and event_type are available for the   #
    # bias model.  Missing timeframes degrade gracefully to NEUTRAL.      #
    # ------------------------------------------------------------------ #
    mtf_data: dict[str, pd.DataFrame] = {}
    for tf in _BIAS_TIMEFRAMES:
        htf_df = market_data.get(tf)
        if htf_df is not None and isinstance(htf_df, pd.DataFrame) and not htf_df.empty:
            try:
                htf_df = compute_indicators(htf_df)
                htf_df = detect_swings(htf_df)
                htf_df = detect_equal_levels(htf_df)
                htf_df = detect_market_structure(htf_df, timeframe=tf)
                # Rename to the column names expected by bias_model
                htf_df = htf_df.rename(
                    columns={"trend_state": "trend", "event_type": "event"}
                )
                mtf_data[tf] = htf_df
            except (ValueError, KeyError):
                # Degrade gracefully — this timeframe won't contribute to bias
                pass

    # Calculate weighted MTF bias
    mtf_bias = calculate_bias(mtf_data)

    # ------------------------------------------------------------------ #
    # STEP 4 — Extract latest row                                        #
    # ------------------------------------------------------------------ #
    row = df.iloc[-1]

    # ------------------------------------------------------------------ #
    # STEP 5 — Derive directional bias from trend state                  #
    # ------------------------------------------------------------------ #
    trend_state = str(row.get("trend_state", "TRANSITION")).upper()
    if trend_state not in ("BULLISH", "BEARISH"):
        return None
    bias = trend_state

    # ------------------------------------------------------------------ #
    # STEP 6 — Check MTF hard block                                      #
    # When the dominant HTF bias strongly opposes M5 direction, block.   #
    # ------------------------------------------------------------------ #
    mtf_direction = mtf_bias.get("bias", "NEUTRAL")
    mtf_strength  = mtf_bias.get("strength", "WEAK")
    mtf_context   = mtf_bias.get("context", "CONSOLIDATION")

    # Strong opposing bias blocks the trade (not during reversals)
    if mtf_context != "REVERSAL":
        if bias == "BULLISH" and mtf_direction == "BEARISH" and mtf_strength == "STRONG":
            return None
        if bias == "BEARISH" and mtf_direction == "BULLISH" and mtf_strength == "STRONG":
            return None

    # ------------------------------------------------------------------ #
    # STEP 7 — Build the market context dictionary                       #
    # ------------------------------------------------------------------ #
    raw_event = str(row.get("event_type", "")).upper()
    _STRUCTURE_MAP = {
        "BOS_CONFIRMED": "BOS_CONFIRMED",
        "CHOCH":         "CHOCH",
        "BREAK":         "BREAK",
    }
    market_structure = _STRUCTURE_MAP.get(raw_event, "NONE")

    # Liquidity classification
    has_sweep = (
        bool(row.get("liquidity_sweep_high", False))
        or bool(row.get("liquidity_sweep_low", False))
    )
    has_pool = (
        bool(row.get("liquidity_pool_high", False))
        or bool(row.get("liquidity_pool_low", False))
    )
    if has_sweep:
        liquidity = "SWEEP"
    elif has_pool:
        liquidity = "EQUAL"
    else:
        liquidity = "NONE"

    # OHLC sanity check
    _o = _safe_float(row, "open")
    _h = _safe_float(row, "high")
    _l = _safe_float(row, "low")
    _c = _safe_float(row, "close")
    if not (_all_finite(_o, _h, _l, _c)
            and _h >= max(_o, _c)
            and _l <= min(_o, _c)
            and _h >= _l):
        return None
    close = _c

    # EMA alignment
    ema_20  = _safe_float(row, "EMA_20")
    ema_50  = _safe_float(row, "EMA_50")
    ema_200 = _safe_float(row, "EMA_200")
    if not _all_finite(ema_20, ema_50, ema_200):
        return None
    if ema_20 > ema_50 > ema_200:
        ema_alignment = "BULLISH"
    elif ema_20 < ema_50 < ema_200:
        ema_alignment = "BEARISH"
    else:
        ema_alignment = "MIXED"

    # ATR validation
    atr = _validate_atr(_safe_float(row, "ATR"))
    if atr is None:
        return None

    # RSI validation
    _raw_rsi = _safe_float(row, "RSI")
    if not _all_finite(_raw_rsi) or not (0.0 <= _raw_rsi <= 100.0):
        return None
    rsi = _raw_rsi

    # VWAP position
    vwap = _safe_float(row, "VWAP", default=float("nan"))
    if _all_finite(vwap):
        vwap_position = "ABOVE" if close > vwap else "BELOW"
    else:
        vwap_position = "ABOVE" if bias == "BULLISH" else "BELOW"

    # Price range from protected levels or 20-bar fallback
    _ph = _safe_float(row, "protected_high")
    recent_high = float(_ph if _all_finite(_ph) else df["high"].iloc[-20:].max())
    _pl = _safe_float(row, "protected_low")
    recent_low  = float(_pl if _all_finite(_pl) else df["low"].iloc[-20:].min())

    # Near order block — only consider recent OBs within _OB_RECENCY_BARS
    price = close
    near_order_block = False
    if "ob_high" in df.columns and "ob_low" in df.columns:
        # Restrict to recent candles to avoid stale OB false positives
        recent_df = df.iloc[-_OB_RECENCY_BARS:]
        ob_pairs = recent_df[["ob_high", "ob_low"]].dropna()
        near_order_block = any(
            ob_h > ob_l and ob_l <= price <= ob_h
            for ob_h, ob_l in zip(ob_pairs["ob_high"], ob_pairs["ob_low"])
        )

    macro_ctx = _macro_handler.get_macro_context()

    context: dict = {
        "market_structure": market_structure,
        "liquidity":        liquidity,
        "ema_alignment":    ema_alignment,
        "rsi":              rsi,
        "vwap_position":    vwap_position,
        "atr":              atr,
        "near_order_block": near_order_block,
        "displacement": (
            bool(row.get("displacement_bullish", False))
            or bool(row.get("displacement_bearish", False))
        ),
        "near_fvg": (
            bool(row.get("fvg_bullish", False))
            or bool(row.get("fvg_bearish", False))
        ),
        "recent_high":      recent_high,
        "recent_low":       recent_low,
        "entry_price":      close,
        **macro_ctx,
    }
    context["macro_sentiment"] = context.get("gold_sentiment", "NEUTRAL")

    # ------------------------------------------------------------------ #
    # STEP 8 — Apply MTF bias confidence adjustment to context           #
    # Pass MTF info so probability model and potential_setup can use it. #
    # ------------------------------------------------------------------ #
    context["mtf_bias"]      = mtf_direction
    context["mtf_strength"]  = mtf_strength
    context["mtf_context"]   = mtf_context

    # ------------------------------------------------------------------ #
    # STEP 9 — Probability scoring (approximate RR)                      #
    # ------------------------------------------------------------------ #
    if bias == "BULLISH":
        raw_sl     = recent_low  - (0.5 * atr)
        raw_risk   = abs(close   - raw_sl)
        raw_reward = abs(recent_high - close)
    else:
        raw_sl     = recent_high + (0.5 * atr)
        raw_risk   = abs(raw_sl  - close)
        raw_reward = abs(close   - recent_low)

    approx_rr = (raw_reward / raw_risk) if raw_risk > 0 else 0.0

    # Apply MTF confidence adjustment to approximate probability
    mtf_adj = _compute_mtf_probability_adjustment(bias, mtf_direction, mtf_strength, mtf_context)

    prob_result = compute_probability(
        {"bias": bias, "risk_reward": approx_rr},
        context,
    )
    probability = max(0, min(100, int(prob_result["probability"]) + mtf_adj))

    # ------------------------------------------------------------------ #
    # STEP 10 — Potential setup quality gate                             #
    # ------------------------------------------------------------------ #
    setup_signal = {
        "probability": probability,
        "risk_reward": approx_rr,
        "bias":        bias,
    }
    setup_result = evaluate_potential_setup(setup_signal, context)
    if not setup_result["is_valid_setup"]:
        return None

    # ------------------------------------------------------------------ #
    # STEP 11 — Trade setup (precise entry, SL, TP, RR)                 #
    # ------------------------------------------------------------------ #
    trade = build_trade_setup(
        {"bias": bias},
        {**context, "setup_type": setup_result["setup_type"]},
    )
    if trade is None:
        return None

    # Recompute probability with exact trade RR
    probability = max(0, min(100, int(
        compute_probability(
            {"bias": bias, "risk_reward": trade["risk_reward"]},
            context,
        )["probability"]
    ) + mtf_adj))

    # ------------------------------------------------------------------ #
    # STEP 12 — Numeric confidence for risk manager                      #
    # ------------------------------------------------------------------ #
    confidence_label = setup_result.get("confidence", "MEDIUM")
    confidence_score = _CONFIDENCE_SCORE.get(confidence_label, 70.0)

    # Apply a weak opposing MTF penalty to confidence if applicable
    if mtf_context != "REVERSAL":
        if (bias == "BULLISH" and mtf_direction == "BEARISH") or \
           (bias == "BEARISH" and mtf_direction == "BULLISH"):
            confidence_score = max(0.0, confidence_score - 15.0)
    elif mtf_context != "REVERSAL":
        if (bias == "BULLISH" and mtf_direction == "BULLISH") or \
           (bias == "BEARISH" and mtf_direction == "BEARISH"):
            confidence_score = min(100.0, confidence_score + 10.0)

    trade_signal: dict = {
        **trade,
        "confidence": confidence_score,
        "atr":        atr,
    }

    # ------------------------------------------------------------------ #
    # STEP 13 — Risk manager validation gate                             #
    # ------------------------------------------------------------------ #
    risk = evaluate_risk(
        trade_signal,
        account_balance=_DEFAULT_ACCOUNT_BALANCE,
        open_trades=_DEFAULT_OPEN_TRADES,
        daily_loss=_DEFAULT_DAILY_LOSS,
    )
    if risk is None:
        return None

    # ------------------------------------------------------------------ #
    # STEP 14 — Assemble final output signal                             #
    # ------------------------------------------------------------------ #
    output_signal: dict = {
        **trade_signal,
        "pair":          "XAUUSD",
        "timeframe":     _EXECUTION_TIMEFRAME,
        "bias":          bias,
        "mtf_bias":      mtf_direction,
        "bias_strength": mtf_strength,
        "context":       f"{setup_result.get('setup_type', 'UNKNOWN')} setup | MTF: {mtf_context}",
        "mtf_score":     round(mtf_bias.get("score", 0.0), 3),
        "probability":   probability,
    }
    return output_signal


# ---------------------------------------------------------------------------
# MTF probability adjustment
# ---------------------------------------------------------------------------


def _compute_mtf_probability_adjustment(
    bias: str,
    mtf_direction: str,
    mtf_strength: str,
    mtf_context: str,
) -> int:
    """Compute a probability score adjustment based on MTF alignment.

    Returns an integer delta to add to the raw probability score:

    * Aligned STRONG  → +15  (institutional flow fully behind the trade)
    * Aligned WEAK    → +7   (partial institutional alignment)
    * REVERSAL context → 0   (HTF is mid-transition; no adjustment)
    * Opposing WEAK   → -10  (some counter-flow pressure)
    * Opposing STRONG → trade is already blocked upstream; returns -20
    * NEUTRAL          → 0   (no macro information either way)
    """
    if mtf_context == "REVERSAL":
        return 0

    aligned = (
        (bias == "BULLISH" and mtf_direction == "BULLISH") or
        (bias == "BEARISH" and mtf_direction == "BEARISH")
    )
    opposing = (
        (bias == "BULLISH" and mtf_direction == "BEARISH") or
        (bias == "BEARISH" and mtf_direction == "BULLISH")
    )

    if aligned:
        return 15 if mtf_strength == "STRONG" else 7
    if opposing:
        return -20 if mtf_strength == "STRONG" else -10
    return 0  # NEUTRAL


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_atr(atr: float) -> float | None:
    """Ensure ATR is usable for trading logic."""
    import math
    if math.isnan(atr):
        return None
    if atr <= 0:
        return None
    if atr < 0.1:
        return None
    if atr > 100.0:
        return 100.0
    return atr


def _safe_float(row, key: str, *, default: float = float("nan")) -> float:
    """Extract a float from a row-like object, returning *default* on error."""
    try:
        val = float(row[key])
    except (KeyError, TypeError, ValueError):
        return default
    return val


def _all_finite(*values: float) -> bool:
    """Return True when all *values* are neither NaN nor infinite."""
    import math
    return all(math.isfinite(v) for v in values)