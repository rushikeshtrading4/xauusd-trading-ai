"""Average True Range (ATR) indicator using Wilder's smoothing method.

True Range (TR)
    TR measures the full price range of a single candle, accounting for
    overnight gaps and limit moves.  It is defined as the greatest of:

    * ``high − low``                — intra-bar range
    * ``|high − previous_close|``   — gap-up volatility
    * ``|low − previous_close|``    — gap-down volatility

    Using only ``high − low`` would understate volatility when price gaps
    between sessions, which is common in XAUUSD during news events.

Wilder's smoothing vs simple moving average
    A simple rolling mean of TR assigns equal weight to all *n* bars and
    produces a hard step change every time an old bar drops out of the window.
    Wilder's method is an exponential moving average with ``alpha = 1/period``
    (equivalent to a ``2*period − 1`` EMA period in standard notation).  It
    weights recent bars more heavily and transitions smoothly, which avoids
    artificial spikes in ATR when large TR values roll off a fixed window.
    Most professional platforms (MetaTrader, TradingView) use Wilder's ATR,
    so this implementation matches the values traders will observe on their
    charts.

Usage in this system
    * Stop-loss sizing   — SL placed at ``entry ± 0.5 × ATR``
    * Displacement gate  — institutional candle body > ``1.5 × ATR``
    * Volatility filter  — reject signals when ATR is abnormally low
"""

from __future__ import annotations

import pandas as pd

_REQUIRED_COLUMNS = ("high", "low", "close")


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR using Wilder's exponential smoothing.

    The first ``period − 1`` values are ``NaN`` because there is insufficient
    history to form a meaningful average.  All subsequent values are finite
    provided the input prices are finite.

    The original DataFrame is never modified.

    Args:
        df:     OHLC DataFrame with at minimum ``"high"``, ``"low"``, and
                ``"close"`` columns.
        period: Smoothing period (default 14, matching Wilder's original
                specification).

    Returns:
        A ``pd.Series`` named ``"ATR"`` aligned to *df*'s index.

    Raises:
        ValueError: If *df* is missing any required column.
        ValueError: If *period* is less than 1.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"compute_atr(): missing required column(s): {missing}")
    if period < 1:
        raise ValueError(f"compute_atr(): period must be ≥ 1, got {period}")

    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    # Vectorised True Range — no Python loops
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing: alpha = 1/period, adjust=False (recursive formula)
    # The first row has prev_close = NaN → TR = NaN → propagates through ewm
    # so the warmup window is preserved automatically.
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()

    # Force the first (period - 1) values to NaN so callers see a clean
    # warmup boundary that matches standard platform behaviour.
    atr.iloc[: period - 1] = float("nan")

    atr.name = "ATR"
    return atr
