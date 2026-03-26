"""Risk-reward ratio calculator for XAUUSD trade setups.

Pure, deterministic, dependency-free module.  Contains no trading decisions
and performs no I/O.  All validation is strict — invalid inputs raise
``ValueError`` immediately.

Pipeline position
-----------------
    potential_setup.evaluate_potential_setup()
        → trade_setup.build_trade_setup()
            → rr_calculator.compute_risk_reward()   ← this module
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_DISTANCE: float = 0.01   # minimum meaningful price distance for XAUUSD
_RR_DECIMALS:  int   = 2
_VALID_SIDES          = frozenset(("BUY", "SELL"))


def compute_risk_reward(
    entry:     float,
    stop_loss: float,
    take_profit: float,
    side:      str,
) -> float:
    """Compute the risk-reward ratio for a trade.

    Parameters
    ----------
    entry : float
        Trade entry price.
    stop_loss : float
        Stop-loss level.
    take_profit : float
        Take-profit level.
    side : str
        Trade direction — ``"BUY"`` or ``"SELL"``.

    Returns
    -------
    float
        Risk-reward ratio rounded to 2 decimal places (e.g. ``2.0``).

    Raises
    ------
    ValueError
        * Any price input is non-finite (NaN / ±inf).
        * ``side`` is not ``"BUY"`` or ``"SELL"``.
        * Computed risk or reward is not positive (invalid geometry).
        * Risk or reward is below the minimum distance threshold (noise trade).
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    for name, value in (("entry", entry), ("stop_loss", stop_loss),
                        ("take_profit", take_profit)):
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Invalid input: '{name}' must be a numeric type, "
                f"got {type(value).__name__}"
            )
        if not math.isfinite(float(value)):
            raise ValueError(
                f"Invalid input: '{name}' must be a finite float, got {value!r}"
            )

    entry       = float(entry)
    stop_loss   = float(stop_loss)
    take_profit = float(take_profit)
    side        = str(side).upper()

    if side not in _VALID_SIDES:
        raise ValueError(
            f"Invalid side: '{side}' — must be 'BUY' or 'SELL'"
        )

    # ------------------------------------------------------------------
    # 2. Calculate distances
    # ------------------------------------------------------------------
    if side == "BUY":
        risk   = entry - stop_loss
        reward = take_profit - entry
    else:  # SELL
        risk   = stop_loss - entry
        reward = entry - take_profit

    # ------------------------------------------------------------------
    # 3. Validate geometry
    # ------------------------------------------------------------------
    if risk <= 0 or reward <= 0:
        raise ValueError(
            "Invalid trade geometry: risk and reward must both be positive. "
            f"Got risk={risk:.5f}, reward={reward:.5f} for side={side!r} "
            f"(entry={entry}, stop_loss={stop_loss}, take_profit={take_profit})"
        )

    # ------------------------------------------------------------------
    # 4. Minimum distance filter (noise trade guard)
    # ------------------------------------------------------------------
    if risk < _MIN_DISTANCE:
        raise ValueError(
            f"Invalid trade geometry: risk distance {risk:.5f} is below the "
            f"minimum threshold of {_MIN_DISTANCE} for XAUUSD"
        )

    if reward < _MIN_DISTANCE:
        raise ValueError(
            f"Invalid trade geometry: reward distance {reward:.5f} is below the "
            f"minimum threshold of {_MIN_DISTANCE} for XAUUSD"
        )

    # ------------------------------------------------------------------
    # 5 & 6. Compute and round RR
    # ------------------------------------------------------------------
    rr = reward / risk

    return round(rr, _RR_DECIMALS)
