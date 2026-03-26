"""Trade setup construction for XAUUSD.

Converts a qualified potential setup into a fully specified trade: entry price,
stop loss, take profit, risk/reward ratio, and invalidation level.

This module sits at the bottom of the execution pipeline:

    potential_setup.evaluate_potential_setup()
        → build_trade_setup()   ← this module
            → signal_formatter.format_signal()

All arithmetic is deterministic and contains no randomness.  Prices are
rounded to 5 decimal places (standard for XAUUSD spot), RR to 2 d.p.

Returns ``None`` rather than raising for three invalid-data conditions:
    * risk == 0             (entry == stop loss — degenerate geometry)
    * reward <= 0.5 * risk  (entry too close to capped TP — bad entry position)
    * rr   < 2              (below minimum 1:2 R:R required by AI_RULES)

No external dependencies.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Public constants (used by tests and downstream modules)
# ---------------------------------------------------------------------------
MIN_RR          = 2.0   # 1:2 minimum from AI_RULES — absolute hard floor
_PRICE_DECIMALS = 5     # XAUUSD standard precision
_RR_DECIMALS    = 2


def build_trade_setup(signal: dict, context: dict) -> dict | None:
    """Build a complete trade setup from a qualified potential setup.

    Parameters
    ----------
    signal : dict
        ``"bias"`` : str  – ``"BULLISH"`` or ``"BEARISH"``
    context : dict
        ``"setup_type"``  : str   – ``"REVERSAL"``, ``"CONTINUATION"``, or
                                     ``"BREAKOUT"``
        ``"atr"``         : float – current ATR value (used for SL buffer)
        ``"recent_high"`` : float – highest relevant price level
        ``"recent_low"``  : float – lowest relevant price level
        ``"entry_price"`` : float – current price or confirmation close

    Returns
    -------
    dict or None
        ``None`` when the setup is geometrically invalid or fails a filter:

        * ``risk == 0``            — entry exactly at stop loss
        * ``reward <= 0.5 * risk`` — entry too close to the capped TP
        * ``rr < 2``               — below AI_RULES minimum 1:2 R:R

        Otherwise:

        .. code-block:: python

            {
                "entry":        float,  # entry_price, 5 d.p.
                "stop_loss":    float,  # SL level, 5 d.p.
                "take_profit":  float,  # TP level, 5 d.p.
                "risk_reward":  float,  # reward / risk, 2 d.p.
                "invalidation": float,  # mirrors stop_loss, 5 d.p.
            }

    Take-Profit design
    ------------------
    TP is computed *dynamically* from the risk distance (target = 2.5 × risk),
    then capped at the nearest institutional liquidity level (``recent_high``
    for longs, ``recent_low`` for shorts).

    * Dynamic target keeps the reward proportional to actual risk taken,
      preventing a mechanical fixed-target from drifting out of alignment
      with the current entry price.
    * The liquidity cap anchors TP at a realistic, structure-defined exit and
      avoids projecting the target beyond the last confirmed price level.
    * Together they guarantee RR consistency while remaining adaptive to
      where in the structure the entry occurs.
    """
    # ------------------------------------------------------------------ #
    # Extract values with safe defaults                                   #
    # ------------------------------------------------------------------ #
    bias        = str(signal.get("bias", "")).upper()
    setup_type  = str(context.get("setup_type", "")).upper()
    atr         = float(context.get("atr",          0.0))
    recent_high = float(context.get("recent_high",  0.0))
    recent_low  = float(context.get("recent_low",   0.0))
    entry_price = float(context.get("entry_price",  0.0))

    # ATR buffer used for all stop-loss placements: half an ATR beyond the
    # swing extreme gives the stop room to breathe without being excessive.
    atr_buffer = 0.5 * atr

    # ------------------------------------------------------------------ #
    # Entry Confirmation (optional — only applied when caller supplies    #
    # confirmation context keys; omitting them bypasses this gate so that #
    # existing callers without candle data are unaffected)                #
    # ------------------------------------------------------------------ #
    _CONF_KEYS = ("m1_choch", "candle_high", "candle_low", "candle_open", "candle_close")
    _has_confirmation_keys = any(k in context for k in _CONF_KEYS)

    if _has_confirmation_keys:
        m1_choch      = bool(context.get("m1_choch",      False))
        candle_high   = float(context.get("candle_high",  0.0))
        candle_low    = float(context.get("candle_low",   0.0))
        candle_open   = float(context.get("candle_open",  0.0))
        candle_close  = float(context.get("candle_close", 0.0))
        candle_range  = candle_high - candle_low

        if candle_range > 0:
            body = abs(candle_close - candle_open)
            if bias == "BULLISH":
                lower_wick          = min(candle_open, candle_close) - candle_low
                upper_body_position = (candle_close - candle_low) / candle_range
                strong_rejection    = (lower_wick > body) and (upper_body_position > 0.6)
            else:  # BEARISH
                upper_wick          = candle_high - max(candle_open, candle_close)
                lower_body_position = (candle_high - candle_close) / candle_range
                strong_rejection    = (upper_wick > body) and (lower_body_position > 0.6)
        else:
            strong_rejection = False

        entry_confirmed = m1_choch or strong_rejection
        if not entry_confirmed:
            return None
    else:
        entry_confirmed = True  # bypass when no confirmation data supplied

    # ------------------------------------------------------------------ #
    # STEP 1 — Entry                                                      #
    # Use the provided entry price directly; no fuzzing or adjustment.    #
    # ------------------------------------------------------------------ #
    entry = entry_price

    # ------------------------------------------------------------------ #
    # STEP 2 — Stop Loss                                                  #
    # Identical geometry for all three setup types: place beyond the      #
    # swing extreme that would invalidate the trade thesis.               #
    # BULLISH: stop below recent_low (long positions invalidated below)   #
    # BEARISH: stop above recent_high (short positions invalidated above) #
    # ------------------------------------------------------------------ #
    if bias == "BULLISH":
        stop_loss = recent_low - atr_buffer
    else:  # BEARISH
        stop_loss = recent_high + atr_buffer

    # ------------------------------------------------------------------ #
    # STEP 3 — Risk                                                       #
    # Computed before TP because the dynamic TP target depends on it.     #
    # ------------------------------------------------------------------ #
    risk = abs(entry - stop_loss)

    # Guard: degenerate geometry — entry exactly at stop loss
    if risk == 0:
        return None

    # ------------------------------------------------------------------ #
    # STEP 4 — Take Profit (dynamic, capped at liquidity level)           #
    # Target is set at 2.5× risk from entry, then capped at the nearest  #
    # institutional liquidity level so the TP is always anchored to a    #
    # realistic price structure rather than projecting into empty air.   #
    # ------------------------------------------------------------------ #
    if bias == "BULLISH":
        target      = entry + (risk * 2.5)
        take_profit = min(target, recent_high)
    else:  # BEARISH
        target      = entry - (risk * 2.5)
        take_profit = max(target, recent_low)

    # ------------------------------------------------------------------ #
    # STEP 5 — Reward and bad-entry filter                                #
    # Reject setups where the entry is so close to the capped TP that    #
    # the effective reward is less than half the risk — such entries are  #
    # either too late or structurally misaligned.                        #
    # ------------------------------------------------------------------ #
    reward = abs(take_profit - entry)

    if reward <= (0.5 * risk):
        return None

    # ------------------------------------------------------------------ #
    # STEP 6 — R:R filter                                                 #
    # Enforces the AI_RULES minimum of 1:2.                               #
    # ------------------------------------------------------------------ #
    rr = reward / risk

    if rr < MIN_RR:
        return None

    # ------------------------------------------------------------------ #
    # STEP 7 — Build and return output                                    #
    # Prices rounded to 5 d.p., RR to 2 d.p.                             #
    # ------------------------------------------------------------------ #
    return {
        "entry":           round(entry,       _PRICE_DECIMALS),
        "stop_loss":       round(stop_loss,   _PRICE_DECIMALS),
        "take_profit":     round(take_profit, _PRICE_DECIMALS),
        "risk_reward":     round(rr,          _RR_DECIMALS),
        "invalidation":    round(stop_loss,   _PRICE_DECIMALS),
        "entry_confirmed": entry_confirmed,
    }
