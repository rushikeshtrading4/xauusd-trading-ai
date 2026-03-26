"""Rule-based probability scoring engine for XAUUSD trade setups.

This module is the core decision engine.  It evaluates a candidate trade
signal against seven independent factors — market structure, EMA alignment,
RSI momentum, VWAP position, liquidity context, risk/reward ratio, and ATR
volatility — and produces a deterministic score, letter grade, and confidence
label.

Scoring summary (max additive contribution shown):
    1. Market structure   +25
    2. EMA alignment      +20
    3. RSI momentum       +10
    4. VWAP position      +10
    5. Liquidity          +15
    6. Risk/Reward        +10
    7. ATR                 +5
    8. Macro sentiment    +15
    Raw ceiling          +110  (clamped to 100; floor clamped to 0)

Grading:
    A+  ≥ 80
    A   65–79
    B   50–64
    C   < 50
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# ATR thresholds (XAUUSD context, USD per ounce per bar on the trading TF)
# ---------------------------------------------------------------------------
_ATR_TOO_LOW  = 3.0    # below this → market is compressed / illiquid
_ATR_TOO_HIGH = 40.0   # above this → excessive volatility; harder to trade


def compute_probability(signal: dict, context: dict) -> dict:
    """Score a trade setup and return probability, grade, and confidence.

    Parameters
    ----------
    signal : dict
        Keys expected:
            ``"entry"``       – float, intended entry price
            ``"stop_loss"``   – float, stop-loss price
            ``"take_profit"`` – float, take-profit price
            ``"risk_reward"`` – float, R:R ratio (e.g. 2.5 means 1:2.5)
            ``"bias"``        – str, ``"BULLISH"`` or ``"BEARISH"``
    context : dict
        Keys expected:
            ``"market_structure"`` – str, one of ``"BOS_CONFIRMED"``,
                                     ``"CHOCH"``, ``"BREAK"``, ``"NONE"``
            ``"ema_alignment"``    – str, ``"BULLISH"``, ``"BEARISH"``,
                                     or ``"MIXED"``
            ``"rsi"``              – float, RSI value (0–100)
            ``"vwap_position"``    – str, ``"ABOVE"`` or ``"BELOW"``
            ``"liquidity"``        – str, ``"SWEEP"``, ``"EQUAL"``,
                                     or ``"NONE"``
            ``"atr"``              – float, current ATR value
            ``"macro_sentiment"``  – str, ``"BULLISH"``, ``"BEARISH"``, or
                                     ``"NEUTRAL"`` (absent or empty → treated
                                     as NEUTRAL, no penalty)
            ``"news_blackout"``    – bool, True when inside a high-impact
                                     event window

    Returns
    -------
    dict
        ``"probability"`` : int   – clamped score 0–100
        ``"grade"``       : str   – letter grade (``"A+"``, ``"A"``,
                                    ``"B"``, ``"C"``)
        ``"confidence"``  : str   – human-readable label
    """
    bias = str(signal.get("bias", "")).upper()
    score = 0

    # ------------------------------------------------------------------ #
    # Factor 1 — Market Structure                                         #
    # BOS_CONFIRMED: clean break + hold → strongest structural signal    #
    # CHOCH: change of character → high probability reversal              #
    # BREAK: partial structural break → weak confirmation                 #
    # ------------------------------------------------------------------ #
    structure = str(context.get("market_structure", "NONE")).upper()
    if structure == "BOS_CONFIRMED":
        score += 25
    elif structure == "CHOCH":
        score += 15
    elif structure == "BREAK":
        score += 5
    # NONE → +0

    # ------------------------------------------------------------------ #
    # Factor 2 — EMA Alignment                                            #
    # Perfect: EMA20/50/200 all stacked in the direction of the trade     #
    # Mixed:   partial alignment → some confirmation                      #
    # Against: EMAs oppose the trade direction → negative weight          #
    # ------------------------------------------------------------------ #
    ema = str(context.get("ema_alignment", "MIXED")).upper()
    if ema == bias:
        # e.g. bias=BULLISH and ema_alignment=BULLISH → perfect stack
        score += 20
    elif ema == "MIXED":
        score += 10
    else:
        # ema alignment is the opposite of the trade bias
        score -= 15

    # ------------------------------------------------------------------ #
    # Factor 3 — RSI Momentum                                             #
    # Bullish: sweet spot 60–70 (momentum without overbought excess)      #
    # Bullish: >75 penalised (overbought, late entry risk)                #
    # Bearish: sweet spot 30–40 (momentum without oversold excess)        #
    # Bearish: <25 penalised (oversold, potential snap-back)              #
    # ------------------------------------------------------------------ #
    rsi = float(context.get("rsi", 50.0))
    if bias == "BULLISH":
        if 60 <= rsi <= 70:
            score += 10
        elif rsi > 75:
            score -= 5
        # 55–59 or 70–75 → +0 (acceptable but not ideal)
    elif bias == "BEARISH":
        if 30 <= rsi <= 40:
            score += 10
        elif rsi < 25:
            score -= 5
        # 25–29 or 40–45 → +0

    # ------------------------------------------------------------------ #
    # Factor 4 — VWAP Position                                            #
    # Price above VWAP → institutional buyers in control → favours LONG   #
    # Price below VWAP → institutional sellers in control → favours SHORT #
    # ------------------------------------------------------------------ #
    vwap = str(context.get("vwap_position", "")).upper()
    vwap_bullish_aligned = (bias == "BULLISH" and vwap == "ABOVE")
    vwap_bearish_aligned = (bias == "BEARISH" and vwap == "BELOW")
    if vwap_bullish_aligned or vwap_bearish_aligned:
        score += 10
    elif vwap in ("ABOVE", "BELOW"):
        # VWAP position is known but opposes the trade
        score -= 10
    # vwap_position absent or empty → +0 (neutral)

    # ------------------------------------------------------------------ #
    # Factor 5 — Liquidity                                                #
    # SWEEP: price took out a liquidity pool and reversed → highest       #
    #        probability setup (institutional hunt-and-reverse)           #
    # EQUAL: equal highs/lows present → moderate setup quality            #
    # NONE:  no detectable liquidity context → no adjustment              #
    # ------------------------------------------------------------------ #
    liquidity = str(context.get("liquidity", "NONE")).upper()
    if liquidity == "SWEEP":
        score += 15
    elif liquidity == "EQUAL":
        score += 5
    # NONE → +0

    # ------------------------------------------------------------------ #
    # Factor 6 — Risk / Reward Ratio                                      #
    # RR ≥ 3 → premium setup; reward greatly exceeds risk                 #
    # RR ≥ 2 → minimum institutional standard (strategy requirement)     #
    # RR < 2 → below threshold → heavily penalised                        #
    # ------------------------------------------------------------------ #
    rr = float(signal.get("risk_reward", 0.0))
    if rr >= 3.0:
        score += 10
    elif rr >= 2.0:
        score += 5
    else:
        score -= 20

    # ------------------------------------------------------------------ #
    # Factor 7 — ATR (volatility)                                         #
    # Normal range → market is liquid and moving → small reward           #
    # Too low  → compressed; breakout/fake-out risk → penalise            #
    # Too high → excessive volatility; spread and SL risk → penalise      #
    # ------------------------------------------------------------------ #
    atr = float(context.get("atr", 0.0))
    if atr <= 0 or atr < _ATR_TOO_LOW:
        score -= 10  # compressed / illiquid market
    elif atr > _ATR_TOO_HIGH:
        score -= 5   # excessive volatility
    else:
        score += 5   # normal, tradeable volatility

    # ------------------------------------------------------------------ #
    # Factor 8 — Macro Sentiment                                          #
    # Gold is directly driven by DXY and treasury yield direction.        #
    # When macro aligns with the trade direction → reward.                #
    # When macro opposes → penalise.                                      #
    # High-impact news window → hard penalise (spreads blow out,          #
    # technical setups are invalidated by the macro catalyst).            #
    # ------------------------------------------------------------------ #
    macro = str(context.get("macro_sentiment", "NEUTRAL")).upper()
    news_blackout = bool(context.get("news_blackout", False))

    if news_blackout:
        score -= 20   # hard penalty — pushes any setup below the 70 gate
    elif macro == bias:
        score += 15   # macro and trade direction fully aligned
    elif macro == "NEUTRAL":
        score += 0    # no macro signal — no reward, no penalty
    else:
        score -= 10   # macro directly opposes the trade direction

    # ------------------------------------------------------------------ #
    # Confluence boosts — applied after individual factors, before clamp  #
    # These reward setups where multiple high-quality signals align,      #
    # reflecting how institutional traders weight confirmation clusters   #
    # rather than treating each signal independently.                     #
    # ------------------------------------------------------------------ #

    # Boost 1 — Institutional Setup                                       #
    # BOS_CONFIRMED + SWEEP + perfect EMA alignment is the textbook       #
    # ICT/SMC entry: structure has shifted, liquidity has been taken, and #
    # the trend is confirmed — the highest-probability setup in the       #
    # strategy.  Reward the confluence beyond the sum of parts.           #
    if (
        structure == "BOS_CONFIRMED"
        and liquidity == "SWEEP"
        and ema == bias
    ):
        score += 10

    # Boost 2 — Trend Continuation                                        #
    # EMA perfectly aligned with the trade AND RSI in the momentum sweet  #
    # spot (not overbought/oversold) signals a healthy trend continuation  #
    # setup — price is moving with institutional flow and has room to run. #
    rsi_continuation = (
        (bias == "BULLISH" and 55 <= rsi <= 75)
        or (bias == "BEARISH" and 25 <= rsi <= 45)
    )
    if ema == bias and rsi_continuation:
        score += 5

    # Penalty — Weak Context                                               #
    # No structure break AND no liquidity context means the setup has no  #
    # institutional catalyst behind it.  Trading into a structureless,    #
    # unliquidated market is counter to the strategy; penalise to keep    #
    # scores below the ≥70 signal-generation threshold.                   #
    if structure == "NONE" and liquidity == "NONE":
        score -= 10

    # ------------------------------------------------------------------ #
    # Normalisation — clamp to [0, 100]                                   #
    # ------------------------------------------------------------------ #
    score = max(0, min(100, score))

    # ------------------------------------------------------------------ #
    # Grading                                                              #
    # ------------------------------------------------------------------ #
    if score >= 80:
        grade = "A+"
        confidence = "Very High"
    elif score >= 65:
        grade = "A"
        confidence = "High"
    elif score >= 50:
        grade = "B"
        confidence = "Moderate"
    else:
        grade = "C"
        confidence = "Low"

    return {
        "probability": score,
        "grade": grade,
        "confidence": confidence,
    }

