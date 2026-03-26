"""
AI signal generation engine — the core decision module.

``generate_trade_signal`` is the single public entry point.  It receives a
fully-enriched OHLC DataFrame (market structure, liquidity sweeps, order
blocks, and indicator columns all present) and returns a structured trade
signal dictionary when all validity conditions are satisfied, or ``None``
when no high-probability setup exists.

Why OB + sweep is the core entry model
---------------------------------------
Institutional traders leave footprints in two predictable ways:

1. **Liquidity sweeps** — engineered moves that hunt retail stop orders placed
   at equal highs / equal lows.  Once those stops are collected the
   institutional position is filled, and price reverses aggressively.

2. **Order blocks** — the last opposite-direction candle before that reversal.
   It represents the price zone where institutions absorbed supply (bearish OB)
   or demand (bullish OB).  When price returns to that zone it is the
   highest-conviction re-entry opportunity in the strategy.

Requiring *both* conditions eliminates the vast majority of false signals:
a sweep without an OB lacks a precise entry zone; an OB without a sweep lacks
confirmation that institutional order flow actually engaged.

How multiple confirmations reduce false signals
------------------------------------------------
Each additional filter independently selects for high-probability setups:

* **Trend filter** (``trend == "BEARISH"`` or CHOCH) — trades only in the
  direction of the dominant structural bias; counter-trend fades are excluded.
* **RSI filter** (< 50 for SELL, > 50 for BUY) — ensures momentum is aligned;
  exhausted moves with diverging momentum are excluded.
* **EMA filter** (``close < EMA`` for SELL, ``close > EMA`` for BUY) — price
  position relative to the moving average confirms medium-term trend direction.
* **Price inside OB** — ensures entry precision; a setup where price has not
  yet returned to the OB is speculative.
* **Minimum RR ≥ 2** — mathematical filter: any setup where the reward does
  not justify the risk is rejected regardless of confluence.
* **Confidence ≥ 70** — a composite gate that quantifies the total strength of
  all confirmations; weak confluence setups are suppressed.

How risk is calculated
-----------------------
::

    entry     = (ob_high + ob_low) / 2          # OB midpoint
    stop_loss = ob_high + 0.5 × ATR             # SELL (above OB + volatility buffer)
                ob_low  − 0.5 × ATR             # BUY  (below OB − volatility buffer)
    risk      = |entry − stop_loss|
    take_profit = entry − 2 × risk              # SELL
                  entry + 2 × risk              # BUY
    risk_reward = |take_profit − entry| / risk  # always ≥ 2

The ATR buffer on the stop loss places it beyond normal candle noise while
remaining within the OB structure.  The 2×risk take-profit targets the next
liquidity pool using a fixed formula — the next-pool column is used when
available; otherwise the formula is the fallback.

Pipeline position::

    order_blocks → signal_engine → risk_manager

Public API::

    from ai.signal_engine import generate_trade_signal
    signal = generate_trade_signal(df, timeframe="M5")

Input ``df`` must contain the columns listed in :data:`_REQUIRED_COLUMNS`.
These are produced by running the full analysis pipeline::

    market_structure → liquidity → liquidity_sweeps → order_blocks
    + indicators (EMA, RSI, ATR)

Note on column naming
---------------------
``market_structure.detect_market_structure()`` outputs ``trend_state`` and
``event_type``.  Rename these to ``trend`` and ``event`` before passing the
DataFrame to ``generate_trade_signal()``::

    df = df.rename(columns={"trend_state": "trend", "event_type": "event"})
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai.bias_model import calculate_bias

# ---------------------------------------------------------------------------
# Required input columns
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: list[str] = [
    "open", "high", "low", "close",
    "EMA_20", "RSI", "ATR",
    "trend", "event",
    "liquidity_sweep_high", "liquidity_sweep_low",
    "bullish_order_block", "bearish_order_block",
    "ob_high", "ob_low",
]

# ---------------------------------------------------------------------------
# Strategy constants
# ---------------------------------------------------------------------------

# Rows scanned backward for sweep + OB discovery.  20 × M5 candles ≈ 100 min.
_LOOKBACK: int = 20

# Minimum acceptable risk-reward ratio — reject the setup below this.
_MIN_RR: float = 2.0

# Minimum confidence score required to emit a signal (per AI_RULES.md).
_MIN_CONFIDENCE: float = 70.0

# ATR multiplier for stop-loss buffer beyond the OB boundary.
_SL_ATR_MULTIPLE: float = 0.5

# Take-profit as a multiple of risk (defines the target RR).
_TP_RR_MULTIPLE: float = 2.0

# Confidence scoring
_CONFIDENCE_BASE: float = 50.0   # starting score
_CONFIDENCE_CAP:  float = 90.0   # maximum allowed

# OB range threshold for "strong displacement" bonus (+10): the OB candle's
# high-low span exceeds this multiple of ATR.  Since OB formation requires
# a displacement of ≥ 1.5 × ATR, setting this to 2.0 selects only the
# unusually forceful institutional moves.
_STRONG_DISP_ATR_FACTOR: float = 2.0

# RSI thresholds for "RSI strong" bonus (+10).
_RSI_STRONG_SELL: float = 45.0   # RSI below this = strong bearish momentum
_RSI_STRONG_BUY:  float = 55.0   # RSI above this = strong bullish momentum

# EMA threshold for "EMA strongly aligned" bonus (+10): the distance between
# close and EMA must exceed this multiple of ATR.
_EMA_STRONG_MULTIPLE: float = 0.5

# "Clean sweep" bonus (+10): sweep occurred within this many rows of the last.
_CLEAN_SWEEP_ROWS: int = 3

# ---------------------------------------------------------------------------
# Multi-timeframe bias adjustment constants
# ---------------------------------------------------------------------------

# A *strong* opposing MTF bias means the dominant institutional flow directly
# contradicts the proposed trade.  Trading into a strong opposing trend risks
# entering right before a larger institutional reversal — the setup is blocked
# entirely rather than penalised, because no confidence adjustment is enough
# to make a counter-strong-trend entry valid.
_MTF_STRONG_OPPOSE_BLOCK: bool = True   # sentinel — strong opposite → None

# A *weak* opposing bias means one or two timeframes are misaligned but the
# dominant trend is not fully confirmed against us.  The trade is allowed but
# confidence is penalised so only high-conviction setups survive the 70 gate.
_MTF_WEAK_OPPOSE_PENALTY: float = 15.0

# When MTF bias *confirms* the trade direction, confidence receives a bonus
# proportional to having institutional flow behind the entry.
_MTF_ALIGNED_BONUS: float = 10.0

# Absolute confidence ceiling after MTF adjustments (allows the aligned bonus
# to push a 90-point setup to 100).
_CONFIDENCE_ABSOLUTE_CAP: float = 100.0

# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def generate_trade_signal(
    df: pd.DataFrame,
    timeframe: str,
    mtf_data: dict | None = None,
) -> dict | None:
    """Generate a XAUUSD trade signal from an enriched OHLC DataFrame.

    Performs a single backward scan through a fixed lookback window — O(N)
    over the window, O(1) amortised for the full history.

    Why strong MTF bias blocks trades
    ------------------------------------
    When the dominant multi-timeframe bias *strongly* opposes the proposed
    entry direction, the broader institutional order flow is actively working
    against the trade.  A bearish order-block sell signal issued while D1 and
    H4 are in strong BULLISH alignment is fading an established trend — even
    a perfect entry geometry cannot overcome that context.  Hard blocking
    protects capital from the highest-risk scenario: a technically valid
    micro-structure setup that points directly into a macro wall.

    Why weak opposing bias allows flexibility
    ------------------------------------------
    Weak opposing bias indicates partial disagreement — perhaps one timeframe
    has shifted while others have not.  Markets frequently develop intraday
    setups that run against the incomplete HTF picture, especially during
    corrective phases.  The confidence penalty (−15) ensures the setup must
    have strong lower-timeframe confluence (score ≥ 70 after the penalty) to
    proceed: weak entries are filtered out while high-conviction ones survive.

    How reversal context enables early entries
    -------------------------------------------
    When ``bias["context"] == "REVERSAL"`` (a CHOCH has been detected recently
    on any timeframe), the previous trend is in the process of being broken.
    In this situation the bias score may still reflect the *old* trend while
    the lower-timeframe structure has already rotated.  Blocking or penalising
    the trade would mean missing precisely the high-probability entries the
    strategy is designed to capture.  Instead, the REVERSAL context bypasses
    both the strong-block and the weak-penalty, relying on the base confidence
    gate (≥ 70) to maintain quality control.

    Args:
        df:        Enriched OHLC DataFrame.  Must contain every column listed
                   in :data:`_REQUIRED_COLUMNS`.  The *last* row represents
                   the current (live) market state.  The function never
                   mutates *df*.
        timeframe: Candle timeframe string (e.g. ``"M5"``, ``"H1"``).
                   Passed through verbatim to the output dictionary.
        mtf_data:  Optional dictionary mapping timeframe labels (``"D1"``,
                   ``"H4"``, ``"H1"``) to enriched DataFrames as expected by
                   :func:`ai.bias_model.calculate_bias`.  If ``None`` or an
                   empty dict is supplied the MTF layer is effectively neutral
                   and existing signal logic is unchanged.

    Returns:
        A signal dictionary with keys::

            pair          – always ``"XAUUSD"``
            timeframe     – the *timeframe* argument
            bias          – ``"BUY"`` or ``"SELL"`` (trade direction)
            entry         – float (OB midpoint)
            stop_loss     – float
            take_profit   – float
            risk_reward   – float (≥ 2.0)
            confidence    – float (0 – 100)
            invalidation  – float (ob_high for SELL, ob_low for BUY)
            mtf_bias      – MTF directional bias (``"BULLISH"`` / ``"BEARISH"`` / ``"NEUTRAL"``)
            bias_strength – MTF bias strength (``"STRONG"`` / ``"WEAK"``)
            context       – structural context (``"TREND"`` / ``"CONSOLIDATION"`` / ``"REVERSAL"``)

        ``None`` if no setup satisfies all validation rules.

    Raises:
        ValueError: If any required column is absent or *df* is empty.
    """
    _validate_input(df)

    # -----------------------------------------------------------------------
    # Extract live values from the last (current) row
    # -----------------------------------------------------------------------
    last = df.iloc[-1]
    close = float(last["close"])
    rsi   = float(last["RSI"])
    ema   = float(last["EMA_20"])
    atr   = float(last["ATR"])
    trend = str(last["trend"])
    event = str(last["event"])

    # Reject immediately if critical indicators are unavailable (warm-up period)
    if np.isnan(close) or np.isnan(rsi) or np.isnan(ema):
        return None
    if not (0.0 <= rsi <= 100.0):
        return None
    atr = _validate_atr(atr)
    if atr is None:
        return None

    # OHLC sanity: high must be the highest value, low the lowest.
    _o = float(last["open"])
    _h = float(last["high"])
    _l = float(last["low"])
    if (np.isnan(_o) or np.isnan(_h) or np.isnan(_l)
            or _h < max(_o, close)
            or _l > min(_o, close)
            or _h < _l):
        return None

    # -----------------------------------------------------------------------
    # Multi-timeframe bias (neutral when mtf_data is not provided)
    # -----------------------------------------------------------------------
    mtf_bias = calculate_bias(mtf_data if mtf_data is not None else {})

    # -----------------------------------------------------------------------
    # Slice the lookback window (last _LOOKBACK rows)
    # -----------------------------------------------------------------------
    n      = len(df)
    start  = max(0, n - _LOOKBACK)
    window = df.iloc[start:].reset_index(drop=True)
    w      = len(window)

    # Pre-extract numpy arrays for fast scanning
    w_open    = window["open"].to_numpy(dtype=float)
    w_close   = window["close"].to_numpy(dtype=float)
    w_atr     = window["ATR"].to_numpy(dtype=float)
    w_sw_high = window["liquidity_sweep_high"].to_numpy(dtype=bool)
    w_sw_low  = window["liquidity_sweep_low"].to_numpy(dtype=bool)
    w_bear_ob = window["bearish_order_block"].to_numpy(dtype=bool)
    w_bull_ob = window["bullish_order_block"].to_numpy(dtype=bool)
    w_ob_high = window["ob_high"].to_numpy(dtype=float)
    w_ob_low  = window["ob_low"].to_numpy(dtype=float)

    # -----------------------------------------------------------------------
    # Check for strong displacement (any recent candle body > 2 × ATR)
    # -----------------------------------------------------------------------
    bodies = np.abs(w_close - w_open)
    strong_w_atr = np.where(np.isnan(w_atr) | (w_atr <= 0), np.inf, w_atr)
    has_strong_displacement = bool(np.any(bodies > _STRONG_DISP_ATR_FACTOR * strong_w_atr))

    # -----------------------------------------------------------------------
    # SELL setup
    # -----------------------------------------------------------------------
    sell = _evaluate_sell(
        close, rsi, ema, atr, trend, event,
        w_sw_high, w_bear_ob, w_ob_high, w_ob_low, w,
    )
    if sell is not None:
        ob_high_v, ob_low_v, sweep_rows_ago = sell
        confidence = _compute_confidence(
            "SELL", ob_high_v, ob_low_v, atr, trend, rsi, close, ema,
            sweep_rows_ago, has_strong_displacement,
        )
        adjusted = _apply_mtf_bias_adjustment("SELL", confidence, mtf_bias)
        if adjusted is not None and adjusted >= _MIN_CONFIDENCE:
            return _build_signal(
                "SELL", ob_high_v, ob_low_v, atr, timeframe, adjusted,
                sweep_rows_ago, mtf_bias,
            )

    # -----------------------------------------------------------------------
    # BUY setup
    # -----------------------------------------------------------------------
    buy = _evaluate_buy(
        close, rsi, ema, atr, trend, event,
        w_sw_low, w_bull_ob, w_ob_high, w_ob_low, w,
    )
    if buy is not None:
        ob_high_v, ob_low_v, sweep_rows_ago = buy
        confidence = _compute_confidence(
            "BUY", ob_high_v, ob_low_v, atr, trend, rsi, close, ema,
            sweep_rows_ago, has_strong_displacement,
        )
        adjusted = _apply_mtf_bias_adjustment("BUY", confidence, mtf_bias)
        if adjusted is not None and adjusted >= _MIN_CONFIDENCE:
            return _build_signal(
                "BUY", ob_high_v, ob_low_v, atr, timeframe, adjusted,
                sweep_rows_ago, mtf_bias,
            )

    return None


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------


def _validate_atr(atr: float) -> float | None:
    """Ensure ATR is usable for trading logic.

    Rules:
    - NaN  -> None
    - <= 0 -> None
    - < 0.1 -> None  (too-low volatility; warm-up artifact or stale feed)
    - > 100 -> cap to 100.0  (spike protection; prevents degenerate geometry)

    Returns:
        Validated ATR float, or None to skip signal generation.
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


def _validate_input(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` for missing columns or an empty DataFrame."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"generate_trade_signal(): missing required column(s): {missing}"
        )
    if len(df) == 0:
        raise ValueError("generate_trade_signal(): DataFrame must not be empty.")


# ---------------------------------------------------------------------------
# Setup evaluators
# ---------------------------------------------------------------------------


def _evaluate_sell(
    close: float,
    rsi:   float,
    ema:   float,
    atr:   float,
    trend: str,
    event: str,
    sw_high:  np.ndarray,
    bear_ob:  np.ndarray,
    ob_high:  np.ndarray,
    ob_low:   np.ndarray,
    w:        int,
) -> tuple[float, float, int] | None:
    """Return ``(ob_high, ob_low, sweep_rows_ago)`` for a valid SELL setup, or
    ``None`` if conditions are not met.

    Gate conditions (all must pass):

    1. ``trend == "BEARISH"`` OR ``event == "CHOCH"`` — structural bias
    2. ``rsi < 50`` — bearish momentum confirmed
    3. ``close < ema`` — price below medium-term trend
    4. A ``liquidity_sweep_high`` exists within the lookback window
    5. A ``bearish_order_block`` exists within the lookback window with the
       current close price inside the OB zone [ob_low, ob_high]
    """
    # --- Gate 1: structural bias ---
    if trend != "BEARISH" and event != "CHOCH":
        return None

    # --- Gate 2: RSI ---
    if rsi >= 50.0:
        return None

    # --- Gate 3: EMA ---
    if close >= ema:
        return None

    # --- Gate 4: recent sweep_high ---
    if not sw_high.any():
        return None

    # --- Gate 5: active bearish OB with price inside zone ---
    ob_h, ob_l, ob_rows_ago = _find_active_ob(bear_ob, ob_high, ob_low, close, w)
    if ob_h is None:
        return None

    # locate most recent sweep_high for confidence scoring
    sweep_rows_ago = _find_recent_sweep(sw_high, w)

    return ob_h, ob_l, sweep_rows_ago  # type: ignore[return-value]


def _evaluate_buy(
    close: float,
    rsi:   float,
    ema:   float,
    atr:   float,
    trend: str,
    event: str,
    sw_low:   np.ndarray,
    bull_ob:  np.ndarray,
    ob_high:  np.ndarray,
    ob_low:   np.ndarray,
    w:        int,
) -> tuple[float, float, int] | None:
    """Return ``(ob_high, ob_low, sweep_rows_ago)`` for a valid BUY setup, or
    ``None`` if conditions are not met.

    Mirror of :func:`_evaluate_sell` with directions reversed:

    1. ``trend == "BULLISH"`` OR ``event == "CHOCH"``
    2. ``rsi > 50`` — bullish momentum
    3. ``close > ema`` — price above medium-term trend
    4. Recent ``liquidity_sweep_low`` exists
    5. Active ``bullish_order_block`` with close inside OB zone
    """
    if trend != "BULLISH" and event != "CHOCH":
        return None
    if rsi <= 50.0:
        return None
    if close <= ema:
        return None
    if not sw_low.any():
        return None

    ob_h, ob_l, _ = _find_active_ob(bull_ob, ob_high, ob_low, close, w)
    if ob_h is None:
        return None

    sweep_rows_ago = _find_recent_sweep(sw_low, w)
    return ob_h, ob_l, sweep_rows_ago  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------------


def _find_active_ob(
    ob_col:   np.ndarray,
    ob_high:  np.ndarray,
    ob_low:   np.ndarray,
    close:    float,
    w:        int,
) -> tuple[float, float, int] | tuple[None, None, None]:
    """Scan backward for the most recent OB row where close is inside [ob_low, ob_high].

    Args:
        ob_col:  Boolean array marking OB candles.
        ob_high: Float array of OB highs (NaN on non-OB rows).
        ob_low:  Float array of OB lows  (NaN on non-OB rows).
        close:   Current close price to test for zone membership.
        w:       Window length (== len(ob_col)).

    Returns:
        ``(ob_high_val, ob_low_val, rows_ago)`` for the nearest qualifying OB,
        or ``(None, None, None)`` if none found.
    """
    for i in range(w - 1, -1, -1):
        if ob_col[i]:
            oh = ob_high[i]
            ol = ob_low[i]
            if not (np.isnan(oh) or np.isnan(ol)) and oh > ol and ol <= close <= oh:
                return oh, ol, (w - 1 - i)
    return None, None, None


def _find_recent_sweep(sweep_col: np.ndarray, w: int) -> int:
    """Return the number of rows ago the most recent sweep occurred.

    Always succeeds (caller has already asserted ``sweep_col.any()``).  If,
    for any reason, no sweep is found, returns ``w`` (oldest possible).
    """
    for i in range(w - 1, -1, -1):
        if sweep_col[i]:
            return w - 1 - i
    return w  # pragma: no cover


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


def _compute_confidence(
    bias:                  str,
    ob_high:               float,
    ob_low:                float,
    atr:                   float,
    trend:                 str,
    rsi:                   float,
    close:                 float,
    ema:                   float,
    sweep_rows_ago:        int,
    has_strong_displacement: bool,
) -> float:
    """Compute the confidence score for a validated setup.

    Scoring model (per ``docs/AI_RULES.md`` and the signal spec):

    =========================================  ======
    Condition                                  Bonus
    =========================================  ======
    Strong displacement (OB range > 2 × ATR)  +10
    Trend fully aligned (not just CHOCH)       +10
    Clean sweep (within last 3 rows)           +10
    RSI strongly directional                   +10
    Price strongly beyond EMA (> 0.5 × ATR)   +10
    =========================================  ======

    Base = 50.  Maximum = 90 (capped).

    Args:
        bias:                    ``"SELL"`` or ``"BUY"``.
        ob_high/ob_low:          OB price boundaries.
        atr:                     Current ATR.
        trend:                   Current trend state string.
        rsi:                     Current RSI value.
        close:                   Current close price.
        ema:                     Current EMA value.
        sweep_rows_ago:          Distance (rows) to the nearest sweep.
        has_strong_displacement: Whether any window candle had body > 2 × ATR.

    Returns:
        Confidence score in ``[50.0, 90.0]``.
    """
    score = _CONFIDENCE_BASE

    # +10 strong displacement — exceptionally forceful institutional candle
    if has_strong_displacement:
        score += 10.0

    # +10 trend fully aligned (structural state, not just a CHOCH transition)
    if bias == "SELL" and trend == "BEARISH":
        score += 10.0
    elif bias == "BUY" and trend == "BULLISH":
        score += 10.0

    # +10 clean sweep — institutional sweep is very recent (within 3 rows)
    if sweep_rows_ago <= _CLEAN_SWEEP_ROWS:
        score += 10.0

    # +10 RSI strongly directional (beyond the 50 gate into 45/55 territory)
    if bias == "SELL" and rsi < _RSI_STRONG_SELL:
        score += 10.0
    elif bias == "BUY" and rsi > _RSI_STRONG_BUY:
        score += 10.0

    # +10 price strongly displaced from EMA (distance > 0.5 × ATR)
    ema_distance = ema - close if bias == "SELL" else close - ema
    if ema_distance > _EMA_STRONG_MULTIPLE * atr:
        score += 10.0

    return min(score, _CONFIDENCE_CAP)


# ---------------------------------------------------------------------------
# MTF bias adjustment
# ---------------------------------------------------------------------------


def _apply_mtf_bias_adjustment(
    trade_dir: str,
    confidence: float,
    mtf_bias: dict,
) -> float | None:
    """Apply multi-timeframe bias rules to the raw confidence score.

    Strict two-state logic (REVERSAL context bypasses all rules):

    * **Context == REVERSAL** — HTF is in transition; skip all MTF penalties
      and bonuses.  The base confidence gate is the sole quality control.
    * **Any opposing bias** — returns ``None`` (hard rejection).
      No counter-trend trades are permitted regardless of HTF bias strength.
    * **Aligned bias** — adds :data:`_MTF_ALIGNED_BONUS`.
    * **Neutral** — no adjustment.

    Result is clamped to ``[0, _CONFIDENCE_ABSOLUTE_CAP]``.

    Args:
        trade_dir:  ``"SELL"`` or ``"BUY"``.
        confidence: Raw confidence score from ``_compute_confidence()``.
        mtf_bias:   Dictionary returned by ``calculate_bias()``.

    Returns:
        Adjusted confidence float, or ``None`` if the trade is hard-blocked.
    """
    mtf_dir      = mtf_bias.get("bias",     "NEUTRAL")
    mtf_strength = mtf_bias.get("strength", "WEAK")  # noqa: F841 — retained for future use
    context      = mtf_bias.get("context",  "CONSOLIDATION")

    # REVERSAL: HTF narrative is mid-flip — no adjustment either way
    if context == "REVERSAL":
        return max(0.0, min(confidence, _CONFIDENCE_ABSOLUTE_CAP))

    # Determine whether MTF aligns or opposes the proposed trade
    if trade_dir == "SELL":
        opposing = mtf_dir == "BULLISH"
        aligned  = mtf_dir == "BEARISH"
    else:  # BUY
        opposing = mtf_dir == "BEARISH"
        aligned  = mtf_dir == "BULLISH"

    if opposing:
        return None   # hard block — no counter-trend trades, no exceptions

    if aligned:
        confidence += _MTF_ALIGNED_BONUS

    return max(0.0, min(confidence, _CONFIDENCE_ABSOLUTE_CAP))


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------


def _build_signal(
    bias:           str,
    ob_high:        float,
    ob_low:         float,
    atr:            float,
    timeframe:      str,
    confidence:     float,
    sweep_rows_ago: int,        # kept for potential future use / logging
    mtf_bias:       dict | None = None,
) -> dict | None:
    """Construct the final signal dictionary.

    Calculates entry, stop loss, take profit, and risk-reward, then validates
    that RR ≥ ``_MIN_RR``.  Returns ``None`` if the geometry is degenerate
    (zero risk distance — extremely unlikely but guarded).

    Args:
        bias:          ``"SELL"`` or ``"BUY"``.
        ob_high:       High of the order-block candle.
        ob_low:        Low of the order-block candle.
        atr:           Current ATR (used for stop-loss buffer).
        timeframe:     Passed through to the output.
        confidence:    Adjusted confidence score (already MTF-filtered).
        sweep_rows_ago: Rows since the triggering sweep (unused in output).
        mtf_bias:      MTF bias dict from ``calculate_bias()`` (optional).

    Returns:
        Signal dict or ``None``.
    """
    entry = (ob_high + ob_low) / 2.0

    if bias == "SELL":
        stop_loss    = ob_high + _SL_ATR_MULTIPLE * atr
        take_profit  = entry - _TP_RR_MULTIPLE * abs(entry - stop_loss)
        invalidation = ob_high
    else:  # BUY
        stop_loss    = ob_low - _SL_ATR_MULTIPLE * atr
        take_profit  = entry + _TP_RR_MULTIPLE * abs(entry - stop_loss)
        invalidation = ob_low

    risk   = abs(entry - stop_loss)
    if risk == 0.0:
        return None  # degenerate geometry — cannot compute RR

    reward = abs(take_profit - entry)
    rr     = reward / risk

    if rr < _MIN_RR:
        return None

    _mtf = mtf_bias or {}
    return {
        "pair":          "XAUUSD",
        "timeframe":     timeframe,
        "bias":          bias,
        "entry":         round(entry, 5),
        "stop_loss":     round(stop_loss, 5),
        "take_profit":   round(take_profit, 5),
        "risk_reward":   round(rr, 2),
        "confidence":    round(confidence, 1),
        "invalidation":  round(invalidation, 5),
        "atr":           round(atr, 5),
        "mtf_bias":      _mtf.get("bias",     "NEUTRAL"),
        "bias_strength": _mtf.get("strength", "WEAK"),
        "context":       _mtf.get("context",  "CONSOLIDATION"),
    }

