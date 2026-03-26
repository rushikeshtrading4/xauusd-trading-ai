"""Execution-layer signal orchestration engine for XAUUSD.

This module is the final integration layer that wires the full analysis and
execution pipeline together into a single call.  Given a dict of multi-
timeframe OHLCV DataFrames it:

    1. Runs the complete analysis pipeline on the M5 execution timeframe.
    2. Extracts a structured market context from the latest candle.
    3. Scores the setup via the probability model.
    4. Applies the potential-setup quality gate.
    5. Builds the precise trade geometry (entry, SL, TP, RR).
    6. Validates the trade against institutional risk controls.
    7. Formats and returns the human-readable trade card.

Returns ``None`` at any gate that is not satisfied (no trade opportunity).
Returns the formatted trade-card string when all gates pass.

Pipeline position
-----------------
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

# Bias → account-equity parameters forwarded to the risk manager.
# Values match AI_RULES.md: 1% daily loss cap.
_DEFAULT_ACCOUNT_BALANCE: float = 10_000.0
_DEFAULT_OPEN_TRADES:     int   = 0
_DEFAULT_DAILY_LOSS:      float = 0.0

# Confidence label (from potential_setup) → numeric score used by the risk
# manager (gate: confidence < 70) and the signal formatter (displayed as %).
_CONFIDENCE_SCORE: dict[str, float] = {
    "HIGH":   85.0,
    "MEDIUM": 70.0,
}

# Module-level macro handler (stateless default — no events registered).
# In production, inject events from an economic calendar before market open.
_macro_handler = MacroDataHandler()

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

    Identical pipeline to :func:`generate_signal` but skips the final
    formatting step, returning the ``output_signal`` dict directly so
    callers (e.g. backtesting) can access ``entry``, ``stop_loss``,
    ``take_profit``, ``risk_reward``, ``bias``, etc.

    Parameters
    ----------
    market_data : dict
        Keys are timeframe identifiers; values are OHLCV DataFrames.
        The M5 DataFrame must contain at minimum:
        ``timestamp``, ``open``, ``high``, ``low``, ``close``, ``volume``.

    Returns
    -------
    dict or None
        Raw signal dict containing at minimum ``"entry"``, ``"stop_loss"``,
        ``"take_profit"``, ``"risk_reward"``, ``"bias"``, ``"confidence"``
        and ``"atr"``, or ``None`` when no tradeable setup exists.

    Notes
    -----
    The function is intentionally *silent* on rejection.
    """
    # ------------------------------------------------------------------ #
    # MACRO BLACKOUT GATE — Hard block during high-impact news events     #
    # No technical setup, however strong, is valid during CPI/NFP/FOMC.  #
    # ------------------------------------------------------------------ #
    import datetime
    _now = datetime.datetime.now(tz=datetime.timezone.utc)
    if _macro_handler.is_high_impact_window(_now):
        return None

    # ------------------------------------------------------------------ #
    # STEP 1 — Extract M5 execution timeframe data                       #
    # All entry signals are derived from M5 candles.  None or empty       #
    # DataFrames are rejected immediately.                                #
    # ------------------------------------------------------------------ #
    df = market_data.get(_EXECUTION_TIMEFRAME)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None

    # ------------------------------------------------------------------ #
    # STEP 2 — Run the full analysis pipeline                             #
    # Order is mandatory (each stage depends on the previous):           #
    #   indicators → swings → equal_levels → market_structure            #
    #   → liquidity_pools → liquidity_sweeps → order_blocks              #
    # Any ValueError from missing columns is a data-quality failure.     #
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
    # STEP 3 — Extract latest row                                        #
    # ------------------------------------------------------------------ #
    row = df.iloc[-1]

    # ------------------------------------------------------------------ #
    # STEP 4 — Derive directional bias from trend state                  #
    # TRANSITION indicates structural ambiguity; no signal is generated. #
    # ------------------------------------------------------------------ #
    trend_state = str(row.get("trend_state", "TRANSITION")).upper()
    if trend_state not in ("BULLISH", "BEARISH"):
        return None
    bias = trend_state

    # ------------------------------------------------------------------ #
    # STEP 5 — Build the market context dictionary                       #
    # ------------------------------------------------------------------ #

    # Market structure: map the raw event_type to the labels recognised by
    # evaluate_potential_setup.  LIQUIDITY_SWEEP is a liquidity event, not
    # a structural one, so it maps to "NONE" here and is captured below.
    raw_event = str(row.get("event_type", "")).upper()
    _STRUCTURE_MAP = {
        "BOS_CONFIRMED": "BOS_CONFIRMED",
        "CHOCH":         "CHOCH",
        "BREAK":         "BREAK",
    }
    market_structure = _STRUCTURE_MAP.get(raw_event, "NONE")

    # Liquidity: priority is SWEEP > EQUAL > NONE.
    # Sweeps (stop-hunts) are the strongest institutional signal.
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

    # OHLC sanity: high must be >= max(open, close); low must be <= min(open, close).
    # Violations indicate corrupt feed data; no signal is generated.
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

    # EMA alignment: bullish when stacked 20 > 50 > 200, bearish when inverted.
    # All three EMAs must be finite; NaN indicates insufficient warm-up data.
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

    # ATR: validated through _validate_atr (NaN / <=0 / <0.1 -> None; >100 -> cap).
    atr = _validate_atr(_safe_float(row, "ATR"))
    if atr is None:
        return None

    # RSI: must be finite and within [0, 100]; a fallback default would mask
    # abnormal values and allow corrupted data to produce trade signals.
    _raw_rsi = _safe_float(row, "RSI")
    if not _all_finite(_raw_rsi) or not (0.0 <= _raw_rsi <= 100.0):
        return None
    rsi = _raw_rsi

    vwap = _safe_float(row, "VWAP", default=float("nan"))

    # VWAP position: price above VWAP favours longs; below favours shorts.
    # When VWAP is unavailable (NaN) fall back to bias-consistent assumption.
    if _all_finite(vwap):
        vwap_position = "ABOVE" if close > vwap else "BELOW"
    else:
        vwap_position = "ABOVE" if bias == "BULLISH" else "BELOW"

    # Price range: prefer structure levels (protected_high/low) which
    # reflect the last confirmed swing, falling back to the 20-bar window.
    # _safe_float + _all_finite handles the case where the column exists
    # but contains NaN (row.get() returns NaN, not the default, for NaN values).
    _ph = _safe_float(row, "protected_high")
    recent_high = float(_ph if _all_finite(_ph) else df["high"].iloc[-20:].max())
    _pl = _safe_float(row, "protected_low")
    recent_low  = float(_pl if _all_finite(_pl) else df["low"].iloc[-20:].min())

    # Near order block: True when price lies inside a valid OB zone where
    # both levels are non-NaN and ob_high > ob_low (degenerate OBs skipped).
    price = close
    if "ob_high" in df.columns and "ob_low" in df.columns:
        ob_pairs = df[["ob_high", "ob_low"]].dropna()
        near_order_block = any(
            ob_h > ob_l and ob_l <= price <= ob_h
            for ob_h, ob_l in zip(ob_pairs["ob_high"], ob_pairs["ob_low"])
        )
    else:
        near_order_block = False

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
        **_macro_handler.get_macro_context(),
    }
    context["macro_sentiment"] = context.get("gold_sentiment", "NEUTRAL")

    # ------------------------------------------------------------------ #
    # STEP 6 — Probability scoring                                       #
    # compute_probability() requires an R:R estimate.  We approximate it  #
    # from raw context geometry (same SL formula as build_trade_setup)    #
    # to avoid a circular dependency.  The full trade is built later in  #
    # STEP 8 using the same ATR-based SL, so the RR here is consistent.  #
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

    prob_result = compute_probability(
        {"bias": bias, "risk_reward": approx_rr},
        context,
    )
    probability = int(prob_result["probability"])

    # ------------------------------------------------------------------ #
    # STEP 7 — Potential setup quality gate                              #
    # Enforces hard minimums (probability, R:R, ATR) and classifies the  #
    # setup type before the full trade is built.                          #
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
    # STEP 8 — Trade setup (precise entry, SL, TP, RR)                  #
    # ------------------------------------------------------------------ #
    trade = build_trade_setup(
        {"bias": bias},
        {**context, "setup_type": setup_result["setup_type"]},
    )
    if trade is None:
        return None

    # ------------------------------------------------------------------ #
    # STEP 8b — Recompute probability with the exact trade R:R           #
    # The STEP 6 estimate used approximate geometry; now that the         #
    # precise trade is built we replace it with the real risk_reward.    #
    # ------------------------------------------------------------------ #
    probability = int(
        compute_probability(
            {"bias": bias, "risk_reward": trade["risk_reward"]},
            context,
        )["probability"]
    )

    # ------------------------------------------------------------------ #
    # STEP 9 — Numeric confidence for risk manager and formatter         #
    # potential_setup returns "HIGH"/"MEDIUM"; downstream modules require #
    # a numeric score (risk_manager gate: confidence < 70).              #
    # ------------------------------------------------------------------ #
    confidence_label = setup_result.get("confidence", "MEDIUM")
    confidence_score = _CONFIDENCE_SCORE.get(confidence_label, 70.0)

    trade_signal: dict = {
        **trade,
        "confidence": confidence_score,
        "atr":        atr,
    }

    # ------------------------------------------------------------------ #
    # STEP 10 — Risk manager validation gate                             #
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
    # STEP 11 — Format and return the trade card                         #
    # ------------------------------------------------------------------ #
    output_signal: dict = {
        **trade_signal,
        "pair":          "XAUUSD",
        "timeframe":     _EXECUTION_TIMEFRAME,
        "bias":          bias,
        # mtf_bias mirrors M5 bias; multi-timeframe injection is a
        # future enhancement when HTF DataFrames are passed in market_data.
        "mtf_bias":      bias,
        "bias_strength": confidence_label,
        "context":       f"{setup_result.get('setup_type', 'UNKNOWN')} setup",
    }
    return output_signal


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_atr(atr: float) -> float | None:
    """Ensure ATR is usable for trading logic.

    Rules:
    - NaN  -> None
    - <= 0 -> None
    - < 0.1 -> None  (too-low volatility; warm-up artifact or stale feed)
    - > 100 -> cap to 100.0  (spike protection; prevents degenerate geometry)

    Returns:
        Validated ATR float, or None to signal that the candle should be
        skipped without generating a trade signal.
    """
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
    """Extract a float from a row-like object, returning *default* on KeyError
    or when the value is NaN."""
    try:
        val = float(row[key])
    except (KeyError, TypeError, ValueError):
        return default
    return val


def _all_finite(*values: float) -> bool:
    """Return True when all *values* are neither NaN nor infinite."""
    import math
    return all(math.isfinite(v) for v in values)
