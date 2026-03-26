"""Risk validation gate for XAUUSD trade signals.

Pure, deterministic validation layer.  Contains no position-sizing logic,
no trading decisions, and no I/O.  Every gate either passes (returns the
input signal unchanged) or rejects (returns ``None``).

Gate sequence
-------------
1. Signal presence and required keys
2. Account balance > 0
3. Risk-reward ≥ 2.0
4. Confidence ≥ 70
5. Daily loss < 1 % of account balance
6. Open trades < 2
7. SL distance ≠ 0
8. ATR validity  (_validate_atr)
9. SL distance ≤ 2 × ATR
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Strategy constants  (mirror docs/AI_RULES.md and docs/TRADING_STRATEGY.md)
# ---------------------------------------------------------------------------

_DAILY_LOSS_LIMIT_PCT: float = 0.01  # 1 % daily loss cap
_MAX_OPEN_TRADES:      int   = 2     # maximum concurrent open positions
_MIN_RR:               float = 2.0   # minimum risk-reward ratio
_MIN_CONFIDENCE:       float = 70.0  # minimum signal confidence score
_MAX_SL_ATR_MULTIPLE:  float = 2.0   # stop-loss distance must be ≤ 2 × ATR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_risk(
    signal:          dict | None,
    account_balance: float,
    open_trades:     int,
    daily_loss:      float,
) -> dict | None:
    """Validate a trade signal against institutional risk rules.

    Applies nine sequential gates.  The first gate that fails immediately
    returns ``None``.  If all gates pass, the original *signal* dict is
    returned unchanged — no keys are added, removed, or modified.

    Gate sequence
    -------------
    1. **Signal presence** — ``None`` signal or any required key missing.
    2. **Account balance** — ``account_balance ≤ 0``.
    3. **Risk-reward** — ``signal["risk_reward"] < 2.0``.
    4. **Confidence** — ``signal["confidence"] < 70``.
    5. **Daily loss limit** — ``daily_loss ≥ 1 %`` of ``account_balance``.
    6. **Max open trades** — ``open_trades ≥ 2``.
    7. **Zero SL distance** — ``entry == stop_loss``.
    8. **ATR validity** — see ``_validate_atr``.
    9. **Stop-loss size** — ``|entry − stop_loss| > 2 × ATR``.

    Args:
        signal:          Output dict from ``ai.signal_engine.generate_trade_signal``,
                         or ``None``.  Required keys: ``"entry"``, ``"stop_loss"``,
                         ``"risk_reward"``, ``"confidence"``, ``"atr"``.
        account_balance: Current account equity in account currency (> 0).
        open_trades:     Number of currently open positions.
        daily_loss:      Total realised loss recorded so far today (positive
                         number; e.g. ``100.0`` means $100 lost).

    Returns:
        The original *signal* dict (unmodified) if all gates pass, or
        ``None`` if any gate fails.
    """
    # ------------------------------------------------------------------
    # Gate 1 — signal presence and required keys
    # ------------------------------------------------------------------
    if signal is None:
        return None

    for key in ("entry", "stop_loss", "risk_reward", "confidence", "atr"):
        if key not in signal:
            return None

    # ------------------------------------------------------------------
    # Gate 2 — account balance
    # ------------------------------------------------------------------
    if account_balance <= 0:
        return None

    # ------------------------------------------------------------------
    # Gate 3 — risk-reward
    # ------------------------------------------------------------------
    if signal["risk_reward"] < _MIN_RR:
        return None

    # ------------------------------------------------------------------
    # Gate 4 — confidence
    # ------------------------------------------------------------------
    if signal["confidence"] < _MIN_CONFIDENCE:
        return None

    # ------------------------------------------------------------------
    # Gate 5 — daily loss limit (1 % cap)
    # ------------------------------------------------------------------
    if daily_loss >= _DAILY_LOSS_LIMIT_PCT * account_balance:
        return None

    # ------------------------------------------------------------------
    # Gate 6 — maximum concurrent open trades
    # ------------------------------------------------------------------
    if open_trades >= _MAX_OPEN_TRADES:
        return None

    # ------------------------------------------------------------------
    # Gate 7 — stop-loss distance (zero guard)
    # ------------------------------------------------------------------
    sl_distance = abs(float(signal["entry"]) - float(signal["stop_loss"]))

    if sl_distance == 0.0:
        return None

    # ------------------------------------------------------------------
    # Gate 8 — ATR validity
    # ------------------------------------------------------------------
    atr = _validate_atr(float(signal["atr"]))
    if atr is None:
        return None

    # ------------------------------------------------------------------
    # Gate 9 — stop-loss distance vs ATR multiple
    # ------------------------------------------------------------------
    if sl_distance > _MAX_SL_ATR_MULTIPLE * atr:
        return None

    return signal


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_atr(atr: float) -> float | None:
    """Ensure ATR is usable for position sizing and stop-loss validation.

    Rules:
    - NaN  -> None
    - <= 0 -> None
    - < 0.1 -> None  (too-low volatility; warm-up artifact or stale feed)
    - > 100 -> cap to 100.0  (spike protection; keeps SL gate sensible)

    Returns:
        Validated ATR float, or None if the ATR cannot be used.
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
