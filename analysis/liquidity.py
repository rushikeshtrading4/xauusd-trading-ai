"""
Liquidity pool detection module.

A *liquidity pool* forms when two or more consecutive swing highs (or lows)
cluster at nearly the same price level.  Stop orders accumulate at these
clusters and make them high-probability targets for institutional sweeps.

Event-based clustering
----------------------
Clusters are counted in *swing-event space*, not in row space.  Non-swing
rows are invisible to the counter — a gap of filler candles between two
equal swing highs does not break the cluster.

Example with gaps::

    Row 0  swing_high=T  equal_high=F  → count=1  (anchor, not a pool yet)
    Row 1  filler                       → count unchanged
    Row 2  swing_high=T  equal_high=T  → count=2  → liquidity_pool_high=True
    Row 3  filler                       → count unchanged
    Row 4  swing_high=T  equal_high=T  → count=3  → liquidity_pool_high=True
    Row 5  swing_high=T  equal_high=F  → count=1  (new cluster starts)

A non-equal swing (``equal_high=False``) resets the counter to 1, anchoring
the next potential cluster at the current swing.

Pipeline position::

    swing_points → market_structure → liquidity_equal_levels → liquidity

Public API::

    from analysis.liquidity import detect_liquidity_pools
    df_with_pools = detect_liquidity_pools(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Required input columns
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: list[str] = [
    "swing_high", "swing_low",
    "equal_high", "equal_low",
]

# ---------------------------------------------------------------------------
# Pool threshold
# ---------------------------------------------------------------------------

# Minimum number of equal consecutive swing events required to declare a
# liquidity pool.  2 captures double-top / double-bottom formations, which
# are the most common and most tradeable liquidity pattern.
_MIN_CLUSTER_SIZE = 2


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def detect_liquidity_pools(df: pd.DataFrame) -> pd.DataFrame:
    """Detect buy-side and sell-side liquidity pools from equal swing data.

    Makes a single forward pass (O(N), no nested loops).  For each swing-high
    event the function either increments the running high-cluster counter (when
    ``equal_high`` is ``True``) or resets it to 1 (new anchor swing).  The
    same logic runs independently for swing-low events.  Non-swing rows do
    **not** touch either counter.

    A row is marked as part of a pool only when its counter reaches
    ``_MIN_CLUSTER_SIZE`` (2) or higher.  Each member of the pool receives the
    full cluster-size count at the time it is written, so the size grows as the
    cluster extends.

    Args:
        df: DataFrame that must already contain the columns listed in
            :data:`_REQUIRED_COLUMNS` (typically the output of
            :func:`analysis.liquidity_equal_levels.detect_equal_levels`).
            All other columns are preserved unchanged in the output.

    Returns:
        A copy of *df* with four additional columns:

        * ``liquidity_pool_high``       — ``True`` when this swing high
          belongs to a qualifying buy-side liquidity pool.
        * ``liquidity_pool_low``        — ``True`` when this swing low
          belongs to a qualifying sell-side liquidity pool.
        * ``liquidity_cluster_size_high`` — Running cluster depth for the
          high side (0 when not in a pool).
        * ``liquidity_cluster_size_low``  — Running cluster depth for the
          low side (0 when not in a pool).

    Raises:
        ValueError: If any required column is missing or the DataFrame is
            empty.
    """
    _validate_input(df)

    out = df.copy().reset_index(drop=True)
    n   = len(out)

    sw_highs = out["swing_high"].to_numpy(dtype=bool)
    sw_lows  = out["swing_low"].to_numpy(dtype=bool)
    eq_highs = out["equal_high"].to_numpy(dtype=bool)
    eq_lows  = out["equal_low"].to_numpy(dtype=bool)

    pool_high  = np.zeros(n, dtype=bool)
    pool_low   = np.zeros(n, dtype=bool)
    size_high  = np.zeros(n, dtype=int)
    size_low   = np.zeros(n, dtype=int)

    high_cluster_count = 0
    low_cluster_count  = 0

    for i in range(n):

        # ---- High side ----
        if sw_highs[i]:
            if eq_highs[i]:
                high_cluster_count += 1
            else:
                high_cluster_count = 1

            if high_cluster_count >= _MIN_CLUSTER_SIZE:
                pool_high[i] = True
                size_high[i] = high_cluster_count

        # ---- Low side ----
        if sw_lows[i]:
            if eq_lows[i]:
                low_cluster_count += 1
            else:
                low_cluster_count = 1

            if low_cluster_count >= _MIN_CLUSTER_SIZE:
                pool_low[i] = True
                size_low[i] = low_cluster_count

    out["liquidity_pool_high"]         = pool_high
    out["liquidity_pool_low"]          = pool_low
    out["liquidity_cluster_size_high"] = size_high
    out["liquidity_cluster_size_low"]  = size_low

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_input(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` for missing columns or an empty DataFrame."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"detect_liquidity_pools(): missing required column(s): {missing}"
        )
    if len(df) == 0:
        raise ValueError("detect_liquidity_pools(): DataFrame must not be empty.")

