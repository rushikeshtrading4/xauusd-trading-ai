"""Pre-trade filtering engine for XAUUSD setup qualification.

This module acts as the quality gate that sits between raw signal scoring and
full trade construction.  It enforces hard minimum thresholds (probability,
R:R, structural context, ATR) and then classifies every passing setup into one
of three recognised setup types: REVERSAL, CONTINUATION, or BREAKOUT.

No external dependencies.  All logic is deterministic.

Pipeline position
-----------------
    probability_model.compute_probability()
        → potential_setup.evaluate_potential_setup()   ← this module
            → trade_setup.build_trade_setup()
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Hard-filter thresholds
# ---------------------------------------------------------------------------
_MIN_PROBABILITY  = 60    # below this the confluence score is too weak to trade
_MIN_RR           = 2.0   # minimum 1:2 R:R — strategy requirement from AI_RULES
_MIN_ATR          = 3.0   # below this the market is compressed / illiquid

# Quality scoring — minimum number of confluence elements required
MIN_SCORE: int = 4  # sweep + displacement + structure + at least one zone


def evaluate_potential_setup(signal: dict, context: dict) -> dict:
    """Determine whether a valid, classifiable setup exists.

    Applies hard filters first (fast-reject path), then classifies the setup
    type and assigns the waiting condition that must be confirmed before entry.

    Parameters
    ----------
    signal : dict
        Keys expected:
            ``"risk_reward"``  – float, R:R ratio
            ``"bias"``         – str, ``"BULLISH"`` or ``"BEARISH"``
            ``"probability"``  – int/float, score from ``compute_probability``
    context : dict
        Keys expected:
            ``"market_structure"`` – str, e.g. ``"BOS_CONFIRMED"``, ``"CHOCH"``,
                                     ``"BREAK"``, ``"NONE"``
            ``"liquidity"``        – str, ``"SWEEP"``, ``"EQUAL"``, ``"NONE"``
            ``"ema_alignment"``    – str, ``"BULLISH"``, ``"BEARISH"``, ``"MIXED"``
            ``"rsi"``              – float
            ``"vwap_position"``    – str, ``"ABOVE"`` or ``"BELOW"``
            ``"atr"``              – float, current ATR value
            ``"displacement"``    – bool, displacement confirmed after sweep
            ``"near_order_block"`` – bool, price inside / near OB
            ``"near_fvg"``         – bool, price inside / near FVG

    Returns
    -------
    dict
        ``"is_valid_setup"`` : bool
        ``"setup_type"``     : str or None
        ``"confidence"``     : str or None    (``"HIGH"`` / ``"MEDIUM"``)
        ``"waiting_for"``    : str or None
        ``"reason"``         : str
        ``"setup_score"``    : int             (0–5; 0 on rejection)
    """
    # ------------------------------------------------------------------ #
    # Extract values with safe defaults                                   #
    # ------------------------------------------------------------------ #
    probability  = float(signal.get("probability",  0))
    rr           = float(signal.get("risk_reward",  0))
    bias         = str(signal.get("bias", "")).upper()

    structure    = str(context.get("market_structure", "NONE")).upper()
    liquidity    = str(context.get("liquidity",        "NONE")).upper()
    ema          = str(context.get("ema_alignment",    "MIXED")).upper()
    atr          = float(context.get("atr", 0))
    near_ob      = bool(context.get("near_order_block", False))
    near_fvg     = bool(context.get("near_fvg",         False))
    displacement = bool(context.get("displacement",     False))

    # Convenience flag: price is near a key institutional level
    near_level = near_ob or near_fvg

    # Granular zone flags for quality scoring
    fvg_present = near_fvg
    ob_present  = near_ob

    # Confluence flags (evaluated once; reused in the gate and classification)
    # has_structure requires a *confirmed* structural shift — BREAK (in-progress)
    # is explicitly excluded; only BOS_CONFIRMED and CHOCH qualify.
    has_sweep        = liquidity == "SWEEP"
    has_displacement = displacement
    has_structure    = structure in ("BOS_CONFIRMED", "CHOCH")
    has_zone         = near_level

    # ------------------------------------------------------------------ #
    # STEP 1 — Hard Filters (fast-reject)                                 #
    # Each check is independent; first failure wins.                      #
    # ------------------------------------------------------------------ #

    # Filter: probability score too low — insufficient confluence
    if probability < _MIN_PROBABILITY:
        return _reject(
            f"Filtered: probability {probability:.0f} below minimum {_MIN_PROBABILITY}"
        )

    # Filter: R:R below institutional minimum — strategy requires ≥ 1:2
    if rr < _MIN_RR:
        return _reject(
            f"Filtered: risk_reward {rr:.2f} below minimum {_MIN_RR}"
        )

    # Filter: no structural AND no liquidity context — structureless market,
    # no institutional catalyst; trading here is counter to strategy rules
    if structure == "NONE" and liquidity == "NONE":
        return _reject(
            "Filtered: no market structure and no liquidity context"
        )

    # Filter: ATR too low — market is compressed; spreads and fake-outs
    # dominate; stop-losses cannot be placed at meaningful distances
    if atr < _MIN_ATR:
        return _reject(
            f"Filtered: ATR {atr:.2f} below minimum {_MIN_ATR} (compressed market)"
        )

    # Filter: strict confluence — all four institutional elements must be
    # present before any setup type is evaluated.  Partial confluence is
    # never enough — no sweep means no stop-hunt, no displacement means no
    # institutional momentum, no structure means no confirmed direction, no
    # zone means no precision entry area.
    valid_confluence = has_sweep and has_displacement and has_structure and has_zone
    if not valid_confluence:
        _missing: list[str] = []
        if not has_sweep:        _missing.append("liquidity sweep")
        if not has_displacement: _missing.append("displacement")
        if not has_structure:    _missing.append("valid structure")
        if not has_zone:         _missing.append("institutional zone")
        return _reject(
            f"Filtered: strict confluence not met — missing: {', '.join(_missing)}"
        )

    # ------------------------------------------------------------------ #
    # QUALITY SCORE — anti-overtrading gate                               #
    # Count every confirmed confluence element.  A setup scoring below   #
    # MIN_SCORE lacks enough institutional backing to justify a trade.   #
    # Elements: liquidity sweep, displacement, confirmed structure,       #
    #           FVG proximity, Order Block proximity.                     #
    # ------------------------------------------------------------------ #
    setup_score = (
        int(has_sweep)
        + int(has_displacement)
        + int(has_structure)
        + int(fvg_present)
        + int(ob_present)
    )
    if setup_score < MIN_SCORE:
        return _reject(
            f"Filtered: quality score {setup_score} below minimum {MIN_SCORE}"
        )

    # ------------------------------------------------------------------ #
    # STEP 2 — Setup Classification                                       #
    # Evaluated in priority order: BREAKOUT > REVERSAL > CONTINUATION.   #
    # Results are captured as local variables so that STEP 3 can apply    #
    # probability-based confidence scaling before returning.              #
    # ------------------------------------------------------------------ #

    # ── BREAKOUT ────────────────────────────────────────────────────────
    # Requires: a structural break is in progress, ATR is elevated
    # (confirming genuine momentum), AND there is an active liquidity
    # interaction.  A structural break without a liquidity catalyst is
    # treated as a false break and excluded.
    # Evaluated first to avoid ambiguity with REVERSAL when BREAK structure
    # is present alongside a sweep and institutional zone.
    if (
        structure == "BREAK"
        and atr > 10
        and liquidity != "NONE"
    ):
        _setup_type  = "BREAKOUT"
        _confidence  = "MEDIUM"
        _waiting_for = "RETEST_CONFIRMATION"
        _reason      = (
            "Breakout setup: structural break in progress with elevated ATR "
            "and liquidity interaction"
        )

    # ── REVERSAL ────────────────────────────────────────────────────────
    # Requires: liquidity has been swept (stop-hunt), a confirmed character
    # shift (CHOCH) is present, AND price is reacting from an institutional
    # level (OB or FVG).  BREAK is excluded — it is structural momentum,
    # not a confirmed reversal signal.
    #
    # Trend-filter: when EMA already agrees with the trade bias the move
    # is more likely a pullback continuation than a genuine reversal —
    # initial confidence is downgraded to MEDIUM.  When EMA opposes the
    # bias the sweep confirms a true character shift — keep HIGH.
    elif (
        liquidity == "SWEEP"
        and structure == "CHOCH"
        and near_level
    ):
        _setup_type  = "REVERSAL"
        _confidence  = "MEDIUM" if ema == bias else "HIGH"
        _waiting_for = "CHOCH_CONFIRMATION"
        _reason      = (
            "Reversal setup: liquidity swept, structural shift confirmed, "
            "price near institutional level"
        )

    # ── CONTINUATION ────────────────────────────────────────────────────
    # Requires: structure has broken cleanly in the direction of the trade
    # (BOS_CONFIRMED), the EMA stack aligns with that bias, AND price is
    # pulling back into an institutional level (OB or FVG) for a re-entry.
    elif (
        structure == "BOS_CONFIRMED"
        and ema == bias
        and near_level
    ):
        _setup_type  = "CONTINUATION"
        _confidence  = "HIGH"
        _waiting_for = "PULLBACK_CONFIRMATION"
        _reason      = (
            "Continuation setup: BOS confirmed, EMA aligned, "
            "price near institutional level"
        )

    # ── No pattern matched ──────────────────────────────────────────────
    # Passed the hard filters but does not fit any recognised setup type.
    else:
        return _reject(
            "Filtered: conditions passed hard filters but no valid setup type detected"
        )

    # ------------------------------------------------------------------ #
    # STEP 3 — Probability-based confidence scaling                       #
    # The probability score from compute_probability() is the final       #
    # authority on confidence.  A score of ≥ 80 confirms strong           #
    # confluence and warrants HIGH confidence.  Anything in the passing   #
    # band (60–79) caps confidence at MEDIUM regardless of setup type.    #
    # Applied after classification so it reflects the full picture.       #
    # ------------------------------------------------------------------ #
    if probability >= 80:
        _confidence = "HIGH"
    else:  # 60 ≤ probability < 80 (hard filter already rejected < 60)
        _confidence = "MEDIUM"

    return _valid(
        setup_type=_setup_type,
        confidence=_confidence,
        waiting_for=_waiting_for,
        reason=_reason,
        setup_score=setup_score,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _reject(reason: str) -> dict:
    return {
        "is_valid_setup": False,
        "setup_type":     None,
        "confidence":     None,
        "waiting_for":    None,
        "reason":         reason,
        "setup_score":    0,
    }


def _valid(
    *,
    setup_type: str,
    confidence: str,
    waiting_for: str,
    reason: str,
    setup_score: int,
) -> dict:
    return {
        "is_valid_setup": True,
        "setup_type":     setup_type,
        "confidence":     confidence,
        "waiting_for":    waiting_for,
        "reason":         reason,
        "setup_score":    setup_score,
    }
