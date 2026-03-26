"""Position sizing calculator for XAUUSD trade setups.

Pure, deterministic, dependency-free module.  Contains no trading decisions
and performs no I/O.  All validation is strict — invalid inputs raise
``ValueError`` immediately.

Pipeline position
-----------------
    trade_setup.build_trade_setup()
        → position_sizing.compute_position_size()   ← this module
            → signal_formatter.format_signal()

XAUUSD lot convention
---------------------
1 standard lot = 100 oz of gold.  The position size returned is in lots.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_SL_DISTANCE: float = 0.01   # minimum meaningful price distance for XAUUSD
_XAUUSD_LOT_UNITS: int  = 100    # 1 lot = 100 units (oz)


def compute_position_size(
    account_balance: float,
    entry:           float,
    stop_loss:       float,
    risk_percent:    float,
) -> float:
    """Compute the position size in lots for a XAUUSD trade.

    Parameters
    ----------
    account_balance : float
        Total trading account balance in account currency.
    entry : float
        Trade entry price.
    stop_loss : float
        Stop-loss level.
    risk_percent : float
        Percentage of account balance to risk on this trade (e.g. ``0.5``
        means 0.5%).  Must be strictly positive.

    Returns
    -------
    float
        Position size in lots (unrounded).

    Raises
    ------
    ValueError
        * Any input is non-numeric or non-finite (NaN / ±inf).
        * ``risk_percent`` is not strictly positive.
        * ``sl_distance`` is zero or below the minimum threshold.

    Examples
    --------
    >>> compute_position_size(10_000, 2000, 1990, 0.5)
    0.05
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    for name, value in (
        ("account_balance", account_balance),
        ("entry",           entry),
        ("stop_loss",       stop_loss),
        ("risk_percent",    risk_percent),
    ):
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Invalid input: '{name}' must be a numeric type, "
                f"got {type(value).__name__}"
            )
        if not math.isfinite(float(value)):
            raise ValueError(
                f"Invalid input: '{name}' must be a finite float, got {value!r}"
            )

    account_balance = float(account_balance)
    entry           = float(entry)
    stop_loss       = float(stop_loss)
    risk_percent    = float(risk_percent)

    if risk_percent <= 0:
        raise ValueError(
            f"Invalid input: 'risk_percent' must be > 0, got {risk_percent!r}"
        )

    # ------------------------------------------------------------------
    # 2. Calculate risk amount
    # ------------------------------------------------------------------
    risk_amount = account_balance * (risk_percent / 100.0)

    # ------------------------------------------------------------------
    # 3. Calculate stop-loss distance
    # ------------------------------------------------------------------
    sl_distance = abs(entry - stop_loss)

    # ------------------------------------------------------------------
    # 4 & 5. Validate geometry and minimum distance
    # ------------------------------------------------------------------
    if sl_distance <= 0:
        raise ValueError(
            "Invalid stop loss distance: entry and stop_loss must differ"
        )

    if sl_distance < _MIN_SL_DISTANCE:
        raise ValueError(
            f"Invalid stop loss distance: {sl_distance:.5f} is below the "
            f"minimum threshold of {_MIN_SL_DISTANCE} for XAUUSD"
        )

    # ------------------------------------------------------------------
    # 6. Calculate position size
    # position_size = risk_amount / sl_distance / lot_units
    # ------------------------------------------------------------------
    position_size = risk_amount / sl_distance / _XAUUSD_LOT_UNITS

    return position_size
