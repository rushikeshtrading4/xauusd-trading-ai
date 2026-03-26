"""
Multi-timeframe bias model — directional edge guard.

``calculate_bias`` aggregates the structural trend across D1, H4, and H1
into a single weighted score that prevents the signal engine from trading
against the dominant higher-timeframe context.

Why weighted bias instead of a simple majority vote
-----------------------------------------------------
Gold (XAUUSD) is driven by multi-day institutional positioning that unfolds
from higher timeframes downward.  A single bearish H1 candle pattern is far
less meaningful when D1 and H4 are both in a confirmed bullish trend.
Weighting (D1 = 0.5, H4 = 0.3, H1 = 0.2) reflects the relative significance
of each timeframe:

* **D1 (50 %)** — the primary directional bias.  Institutional desks
  manage XAUUSD exposure on a multi-day basis; D1 trend is the dominant
  context.
* **H4 (30 %)** — the intermediate swing structure.  This captures
  multi-session order flow and is the first timeframe where institutional
  entries are typically planned.
* **H1 (20 %)** — the near-term execution context.  It refines entry
  timing but carries the least directional weight.

This weighting ensures that a contradicting H1 signal (score − 0.2) can
never flip the bias on its own when D1 and H4 are aligned (max combined
score + 0.8).

Why CHOCH is not a full trend confirmation
-------------------------------------------
A Change of Character (CHOCH) means that the *protected* structural level
was violated and the current trend is compromised — but the new direction has
not yet been confirmed by a BOS sequence.  The system treats CHOCH as an
*early reversal signal* that elevates context to REVERSAL without assigning a
full directional score.  Trading aggressively on a CHOCH alone would mean
fading an established trend before proof of a new one; the override rule
(H1 + H4 BULLISH + recent CHOCH) requires two confirmed timeframes in the new
direction before reversal bias is granted.

How this prevents wrong-direction trades
-----------------------------------------
``generate_trade_signal()`` in ``ai/signal_engine.py`` receives a ``bias``
from this module and may optionally filter signals that oppose the MTF bias.
For example, a BEARISH signal produced by the signal engine while the MTF
bias is BULLISH/STRONG can be suppressed or downgraded.  This layer acts as
the directional conscience of the system — it never generates the entry
itself, but it can veto one.

Pipeline position::

    market_structure (per TF) → calculate_bias → signal_engine (context input)

Public API::

    from ai.bias_model import calculate_bias
    bias_dict = calculate_bias({"D1": df_d1, "H4": df_h4, "H1": df_h1})

Returned keys: ``bias``, ``strength``, ``score``, ``context``.

Column note
-----------
Each DataFrame must contain ``trend`` and ``event`` columns.  These are the
columns produced by ``market_structure.detect_market_structure()`` after the
rename ``trend_state → trend``, ``event_type → event``.  If a key is absent
from *mtf_data* or its DataFrame is empty the corresponding timeframe
contributes 0 (neutral) to the score without raising an error.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Timeframe weights  (must sum to 1.0)
# ---------------------------------------------------------------------------

_WEIGHTS: dict[str, float] = {
    "D1": 0.5,
    "H4": 0.3,
    "H1": 0.2,
}

# ---------------------------------------------------------------------------
# Trend → numeric score mapping
# ---------------------------------------------------------------------------

_TREND_SCORE: dict[str, float] = {
    "BULLISH":    1.0,
    "BEARISH":   -1.0,
    "TRANSITION": 0.0,
}

# ---------------------------------------------------------------------------
# Score thresholds
# ---------------------------------------------------------------------------

_STRONG_THRESHOLD: float = 0.7   # |score| > this → STRONG
_WEAK_THRESHOLD:   float = 0.3   # |score| > this → WEAK (else NEUTRAL)

# ---------------------------------------------------------------------------
# Event → score multiplier
# Only structurally confirmed events (BOS_CONFIRMED, BREAK) contribute to the
# bias score.  CHOCH and absent events are treated as 0 — they are captured
# via the CHOCH override rule, not the weighted score.
# ---------------------------------------------------------------------------

_EVENT_MULTIPLIERS: dict[str, float] = {
    "BOS_CONFIRMED": 1.0,
    "BREAK":         0.6,
}

# ---------------------------------------------------------------------------
# CHOCH lookback — how many rows back to scan for a recent CHOCH event
# ---------------------------------------------------------------------------

_CHOCH_LOOKBACK: int = 10

# ---------------------------------------------------------------------------
# Event string constant (mirrors analysis/market_structure.py)
# ---------------------------------------------------------------------------

_EVENT_CHOCH = "CHOCH"


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def calculate_bias(mtf_data: dict[str, pd.DataFrame]) -> dict:
    """Compute the multi-timeframe directional bias for XAUUSD.

    Reads only the *last row* of each supplied DataFrame, so the function is
    O(K) in the number of timeframes (constant) plus O(L) for the CHOCH
    lookback — effectively O(1) for typical inputs.

    Args:
        mtf_data: Dictionary mapping timeframe labels to enriched DataFrames.
            Recognised keys: ``"D1"``, ``"H4"``, ``"H1"``.  Unknown keys are
            silently ignored.  Missing keys or empty DataFrames are treated as
            TRANSITION (score 0) without raising an error — this ensures the
            system degrades gracefully when a higher timeframe has not yet
            accumulated enough data.

    Returns:
        Dictionary with four keys:

        * ``"bias"``     — ``"BULLISH"``, ``"BEARISH"``, or ``"NEUTRAL"``
        * ``"strength"`` — ``"STRONG"`` or ``"WEAK"``
        * ``"score"``    — weighted float in ``[−1.0, 1.0]``
        * ``"context"``  — ``"TREND"``, ``"CONSOLIDATION"``, or ``"REVERSAL"``

        The result is fully deterministic for any given input.
    """
    # -----------------------------------------------------------------------
    # 1. Extract last-row trend, CHOCH presence, and last event per timeframe
    # -----------------------------------------------------------------------
    tf_trends:  dict[str, str]  = {}   # "D1" → "BULLISH" / "BEARISH" / "TRANSITION"
    tf_choch:   dict[str, bool] = {}   # "D1" → True if CHOCH present in lookback
    tf_events:  dict[str, str]  = {}   # "D1" → last-row event string (normalised)

    for tf in _WEIGHTS:
        trend, has_choch = _extract_tf_state(mtf_data.get(tf))
        tf_trends[tf] = trend
        tf_choch[tf]  = has_choch
        df_tf = mtf_data.get(tf)
        if df_tf is not None and len(df_tf) > 0 and "event" in df_tf.columns:
            tf_events[tf] = str(df_tf.iloc[-1]["event"]).upper()
        else:
            tf_events[tf] = ""

    # -----------------------------------------------------------------------
    # 2. Event-weighted score
    # Only BOS_CONFIRMED (×1.0) and BREAK (×0.6) contribute to the score.
    # CHOCH and no-event timeframes contribute 0; CHOCH is handled via the
    # override rule in Step 4.
    # -----------------------------------------------------------------------
    score: float = 0.0
    for tf, weight in _WEIGHTS.items():
        multiplier = _EVENT_MULTIPLIERS.get(tf_events[tf], 0.0)
        score += _TREND_SCORE.get(tf_trends[tf], 0.0) * weight * multiplier

    # Round to avoid floating-point noise (0.3 + 0.2 arithmetic)
    score = round(score, 10)

    # -----------------------------------------------------------------------
    # 3. Bias and strength from score
    # -----------------------------------------------------------------------
    bias, strength = _score_to_bias_strength(score)

    # -----------------------------------------------------------------------
    # 4. Override: H1 + H4 both BULLISH/BEARISH and recent CHOCH anywhere,
    #    only when the score already carries a non-trivial directional signal.
    #    Requiring abs(score) > 0.2 prevents a spurious CHOCH on a flat,
    #    eventless market from forcing a directional bias.
    # -----------------------------------------------------------------------
    any_choch = any(tf_choch.values())

    if any_choch and abs(score) > 0.2:
        if tf_trends["H1"] == "BULLISH" and tf_trends["H4"] == "BULLISH":
            bias     = "BULLISH"
            strength = "WEAK"   # partial confirmation — D1 may not agree yet
            context  = "REVERSAL"
            return {"bias": bias, "strength": strength, "score": score, "context": context}

        if tf_trends["H1"] == "BEARISH" and tf_trends["H4"] == "BEARISH":
            bias     = "BEARISH"
            strength = "WEAK"
            context  = "REVERSAL"
            return {"bias": bias, "strength": strength, "score": score, "context": context}

    # -----------------------------------------------------------------------
    # 5. Context classification
    # -----------------------------------------------------------------------
    context = _classify_context(tf_trends, any_choch)

    return {
        "bias":     bias,
        "strength": strength,
        "score":    score,
        "context":  context,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_tf_state(df: pd.DataFrame | None) -> tuple[str, bool]:
    """Return ``(trend_str, has_choch)`` for a single timeframe DataFrame.

    If *df* is ``None`` or empty, returns ``("TRANSITION", False)``.

    Args:
        df: Enriched OHLC DataFrame with ``trend`` and ``event`` columns, or
            ``None``.

    Returns:
        ``(trend, has_choch)`` where *trend* is one of ``"BULLISH"``,
        ``"BEARISH"``, ``"TRANSITION"`` and *has_choch* is ``True`` if the
        ``event`` column contains ``"CHOCH"`` within the last
        :data:`_CHOCH_LOOKBACK` rows.
    """
    if df is None or len(df) == 0:
        return "TRANSITION", False

    # Trend: last row
    last = df.iloc[-1]
    trend_raw = str(last["trend"]) if "trend" in df.columns else "TRANSITION"
    trend = trend_raw if trend_raw in _TREND_SCORE else "TRANSITION"

    # CHOCH: scan last _CHOCH_LOOKBACK rows of the event column
    has_choch = False
    if "event" in df.columns:
        window = df["event"].iloc[-_CHOCH_LOOKBACK:]
        has_choch = bool((window == _EVENT_CHOCH).any())

    return trend, has_choch


def _score_to_bias_strength(score: float) -> tuple[str, str]:
    """Map a numeric score to ``(bias, strength)``.

    Thresholds (inclusive on the outer boundary):

    ============  =========  ========
    Score range   Bias       Strength
    ============  =========  ========
    > +0.7        BULLISH    STRONG
    +0.3 to +0.7  BULLISH    WEAK
    −0.3 to +0.3  NEUTRAL    WEAK
    −0.7 to −0.3  BEARISH    WEAK
    < −0.7        BEARISH    STRONG
    ============  =========  ========
    """
    abs_score = abs(score)

    if score > _STRONG_THRESHOLD:
        return "BULLISH", "STRONG"
    elif score > _WEAK_THRESHOLD:
        return "BULLISH", "WEAK"
    elif score < -_STRONG_THRESHOLD:
        return "BEARISH", "STRONG"
    elif score < -_WEAK_THRESHOLD:
        return "BEARISH", "WEAK"
    else:
        return "NEUTRAL", "WEAK"


def _classify_context(
    tf_trends: dict[str, str],
    any_choch: bool,
) -> str:
    """Classify the structural context from timeframe alignment.

    Rules (in priority order):

    1. Any CHOCH present → ``"REVERSAL"``
    2. All three timeframes agree on the same directional state → ``"TREND"``
    3. Mixed signals → ``"CONSOLIDATION"``

    Args:
        tf_trends: Mapping of timeframe → trend string for D1, H4, H1.
        any_choch: Whether any timeframe recently had a CHOCH event.

    Returns:
        One of ``"REVERSAL"``, ``"TREND"``, ``"CONSOLIDATION"``.
    """
    if any_choch:
        return "REVERSAL"

    trends = [tf_trends[tf] for tf in _WEIGHTS]
    if len(set(trends)) == 1 and trends[0] != "TRANSITION":
        return "TREND"

    return "CONSOLIDATION"
