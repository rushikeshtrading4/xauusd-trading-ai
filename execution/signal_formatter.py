"""Human-readable trade instruction formatter.

This is the final user-facing layer of the signal pipeline.  Its sole
responsibility is translating a validated signal dictionary into a clean,
unambiguous plain-text trade card that a trader — or a downstream
notification system — can read instantly.

Why clean output matters
    Internal dictionaries carry implementation artefacts (ATR,
    floating-point noise) that are not useful to the trader and could be
    distracting or misleading.  A single well-defined text format removes
    ambiguity, makes audit logging trivial, and makes it easy to forward the
    output over Telegram, email, or a trading journal without further
    processing.

Why RR formatting matters
    Risk-reward is conventionally expressed as a ratio relative to 1 unit of
    risk, e.g. "1:2" means the potential reward is twice the risk.  Displaying
    a raw float like 2.0 is technically correct but does not match the mental
    model traders use when deciding whether a setup meets their criteria.
    Normalising to the "1:X" form makes the value self-explanatory.

Why invalidation == stop loss
    The stop-loss placement is derived directly from the order-block boundary
    beyond which the trade thesis is structurally invalidated.  Labelling it
    "Invalidation" rather than repeating "Stop Loss" makes the meaning
    explicit: if price reaches this level the original setup is broken and the
    position must be exited, regardless of whether a hard stop was triggered.
"""

from __future__ import annotations


def format_trade_signal(signal: dict | None) -> str:
    """Convert a validated signal dict into a trade instruction string.

    Returns the fixed string ``"No trade opportunity"`` when *signal* is
    ``None``.  Otherwise formats all relevant fields into the canonical
    multi-line trade card.

    The function is purely functional: *signal* is never modified and no
    external state is read or written.

    Args:
        signal: Output dict from ``execution.signal_engine.generate_signal``,
                or ``None``.  Expected keys: ``"timeframe"``, ``"bias"``,
                ``"entry"``, ``"stop_loss"``, ``"take_profit"``,
                ``"risk_reward"``, ``"confidence"``, ``"mtf_bias"``,
                ``"bias_strength"``, ``"context"``.

    Returns:
        A multi-line trade instruction string, or ``"No trade opportunity"``.

    Examples:
        >>> format_trade_signal(None)
        'No trade opportunity'
    """
    # ------------------------------------------------------------------
    # Step 1 — no-trade guard
    # ------------------------------------------------------------------
    if signal is None:
        return "No trade opportunity"

    # ------------------------------------------------------------------
    # Step 2 — extract values
    # ------------------------------------------------------------------
    timeframe  = signal["timeframe"]
    bias       = signal["bias"]

    entry      = signal["entry"]
    sl         = signal["stop_loss"]
    tp         = signal["take_profit"]

    rr         = signal["risk_reward"]
    confidence = signal["confidence"]

    mtf_bias   = signal["mtf_bias"]
    strength   = signal["bias_strength"]
    context    = signal["context"]

    # ------------------------------------------------------------------
    # Step 3 — format values
    # ------------------------------------------------------------------
    entry_fmt      = f"{round(entry, 2):.2f}"
    sl_fmt         = f"{round(sl,    2):.2f}"
    tp_fmt         = f"{round(tp,    2):.2f}"
    confidence_fmt = f"{int(confidence)}%"
    rr_fmt         = _format_rr(rr)

    # ------------------------------------------------------------------
    # Step 4 — build and return the trade card
    # ------------------------------------------------------------------
    return (
        f"Pair: XAUUSD\n"
        f"Timeframe: {timeframe}\n"
        f"\n"
        f"Bias: {bias}\n"
        f"\n"
        f"Entry: {entry_fmt}\n"
        f"Stop Loss: {sl_fmt}\n"
        f"Take Profit: {tp_fmt}\n"
        f"\n"
        f"Risk Reward: {rr_fmt}\n"
        f"Confidence: {confidence_fmt}\n"
        f"\n"
        f"HTF Bias: {mtf_bias} ({strength})\n"
        f"Context: {context}\n"
        f"\n"
        f"Invalidation: {sl_fmt}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_rr(rr: float) -> str:
    """Format a risk-reward float as a "1:X" ratio string.

    Integer multiples (2.0, 3.0) are displayed without a trailing decimal so
    the output matches conventional notation ("1:2" rather than "1:2.0").
    Non-integer multiples are passed through verbatim ("1:2.5").

    Floating-point precision
        Floating-point arithmetic (e.g. ``reward / risk``) can produce values
        like ``2.0000000001`` instead of the exact integer ``2.0``.  A strict
        equality check (``rr == int(rr)``) would fail for such values even
        though they are conceptually whole numbers.  A tolerance-based
        comparison — ``abs(rr - round(rr)) < 1e-9`` — absorbs this noise and
        ensures the output is clean in all practical cases without misclassifying
        genuinely non-integer ratios like ``2.5`` or ``2.75``.

    Args:
        rr: Raw risk-reward ratio (reward / risk).

    Returns:
        Formatted ratio string, e.g. ``"1:2"`` or ``"1:2.5"``.
    """
    if abs(rr - round(rr)) < 1e-6:
        return f"1:{int(round(rr))}"
    return f"1:{rr}"
