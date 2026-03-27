"""
Order block mitigation tracking module.

An order block (OB) is *mitigated* once price trades back through it after
its formation.  A mitigated OB has lost its institutional significance — the
liquidity that created it has been consumed — and should no longer be used
as an entry zone.

Mitigation rules
----------------
* **Bearish OB** (last bullish candle before bearish displacement):
  Mitigated when a subsequent candle's HIGH enters or exceeds ``ob_high``.
  Rationale: bears defending the OB have been overpowered by buying pressure.

* **Bullish OB** (last bearish candle before bullish displacement):
  Mitigated when a subsequent candle's LOW enters or falls below ``ob_low``.
  Rationale: bulls defending the OB have been overpowered by selling pressure.

Only the wick (not the close) is used for mitigation detection because
institutional traders consider a wick through the zone as sufficient evidence
that the level has been tested and absorbed.

State transition
----------------
``ob_mitigated`` is written as a boolean column; ``True`` means the OB at
that row index was mitigated at some point after its formation.  The column
is computed in a single forward pass using suffix-array optimisation (O(N)).

This module is designed to run AFTER ``detect_order_blocks()``.

Pipeline position::

    order_blocks → ob_mitigation → signal_engine

Public API::

    from analysis.ob_mitigation import detect_ob_mitigation
    df_with_mitigation = detect_ob_mitigation(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Required input columns
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: list[str] = [
    "high", "low",
    "bullish_order_block", "bearish_order_block",
    "ob_high", "ob_low",
]


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def detect_ob_mitigation(df: pd.DataFrame) -> pd.DataFrame:
    """Mark order blocks as mitigated when price subsequently trades through them.

    Uses suffix min/max arrays to compute mitigation in O(N) time without
    nested loops.

    Args:
        df: DataFrame containing all columns listed in :data:`_REQUIRED_COLUMNS`.
            Typically the output of :func:`analysis.order_blocks.detect_order_blocks`.

    Returns:
        A copy of *df* with two additional boolean columns:

        * ``ob_mitigated``       — ``True`` on rows where the OB (bull or bear)
          was subsequently mitigated by price action.
        * ``ob_active``          — ``True`` on OB rows that have NOT yet been
          mitigated. This is the column to use for entry decisions.

    Raises:
        ValueError: If any required column is missing or the DataFrame is empty.
    """
    _validate_input(df)

    out = df.copy().reset_index(drop=True)
    n   = len(out)

    highs    = out["high"].to_numpy(dtype=float)
    lows     = out["low"].to_numpy(dtype=float)
    bull_ob  = out["bullish_order_block"].to_numpy(dtype=bool)
    bear_ob  = out["bearish_order_block"].to_numpy(dtype=bool)
    ob_highs = out["ob_high"].to_numpy(dtype=float)
    ob_lows  = out["ob_low"].to_numpy(dtype=float)

    ob_mitigated = np.zeros(n, dtype=bool)

    if n > 1:
        # suffix_min_excl[i] = min(lows[i+1 : n])   — future lows only
        # suffix_max_excl[i] = max(highs[i+1 : n])  — future highs only
        suffix_min_incl = np.minimum.accumulate(lows[::-1])[::-1]
        suffix_max_incl = np.maximum.accumulate(highs[::-1])[::-1]

        import math
        suffix_min_excl = np.full(n, math.inf,  dtype=float)
        suffix_max_excl = np.full(n, -math.inf, dtype=float)
        suffix_min_excl[:-1] = suffix_min_incl[1:]
        suffix_max_excl[:-1] = suffix_max_incl[1:]

        # --- Bearish OB mitigated when a future high reaches ob_high ---
        bear_idx = np.where(bear_ob)[0]
        valid_bear = bear_idx[~np.isnan(ob_highs[bear_idx])]
        if len(valid_bear) > 0:
            mitigated_bear = valid_bear[
                suffix_max_excl[valid_bear] >= ob_highs[valid_bear]
            ]
            ob_mitigated[mitigated_bear] = True

        # --- Bullish OB mitigated when a future low reaches ob_low ---
        bull_idx = np.where(bull_ob)[0]
        valid_bull = bull_idx[~np.isnan(ob_lows[bull_idx])]
        if len(valid_bull) > 0:
            mitigated_bull = valid_bull[
                suffix_min_excl[valid_bull] <= ob_lows[valid_bull]
            ]
            ob_mitigated[mitigated_bull] = True

    # An OB is "active" if it is an OB row AND has NOT been mitigated
    is_ob = bull_ob | bear_ob
    ob_active = is_ob & ~ob_mitigated

    out["ob_mitigated"] = ob_mitigated
    out["ob_active"]    = ob_active

    return out


def get_active_obs(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the rows that contain active (unmitigated) order blocks.

    Convenience wrapper for downstream consumers that need to iterate over
    live OB zones without filtering manually.

    Args:
        df: Output of :func:`detect_ob_mitigation`.

    Returns:
        Filtered DataFrame with only active OB rows.

    Raises:
        ValueError: If ``ob_active`` column is not present.
    """
    if "ob_active" not in df.columns:
        raise ValueError(
            "get_active_obs(): 'ob_active' column not found. "
            "Run detect_ob_mitigation() first."
        )
    return df[df["ob_active"]].copy()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_input(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` for missing columns or an empty DataFrame."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"detect_ob_mitigation(): missing required column(s): {missing}"
        )
    if len(df) == 0:
        raise ValueError("detect_ob_mitigation(): DataFrame must not be empty.")