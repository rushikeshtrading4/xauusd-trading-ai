"""Exponential Moving Average (EMA) indicator.

EMA formula
    EMA[i] = close[i] × α  +  EMA[i-1] × (1 − α)

    where  α = 2 / (period + 1)

    The seed value (``EMA[period-1]``) is the SMA of the first ``period``
    closes.  From bar ``period`` onward the formula is applied strictly
    recursively, matching institutional backtesting models (Bloomberg,
    TradeStation, institutional Python quant libraries).

Why EMA is preferred over SMA
    A Simple Moving Average (SMA) assigns equal weight to all bars in its
    window and drops the oldest bar abruptly when it leaves the window, which
    can produce sharp kinks unrelated to actual price action.  The EMA gives
    exponentially more weight to recent prices and never fully discards
    historical data, so the output is smoother and more responsive to recent
    trend changes.  This is why institutional traders and professional
    platforms default to EMA for trend detection.

Importance in trend detection
    The relationship between fast (e.g. EMA20) and slow (e.g. EMA50/EMA200)
    lines tells the system whether price is in a bullish or bearish structural
    phase:

    * EMA20 > EMA50 > EMA200  → bullish trending regime
    * EMA20 < EMA50 < EMA200  → bearish trending regime

    The signal engine also checks whether the current close is above or below
    the EMA to confirm individual bar momentum before generating a signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute an Exponential Moving Average of the closing price.

    The standard alpha = 2 / (period + 1) is used, matching the convention
    of MetaTrader, TradingView, and most professional charting platforms.

    The first ``period − 1`` values are set to ``NaN`` (warmup).  The seed
    value at position ``period − 1`` is the SMA of the first ``period``
    closes.  From bar ``period`` onward the recursive formula is applied
    strictly, ensuring independence from data outside the warmup window and
    matching institutional backtesting models.

    The original DataFrame is never modified.

    Args:
        df:     DataFrame containing at minimum a ``"close"`` column.
        period: EMA period (must be ≥ 1).

    Returns:
        A ``pd.Series`` named ``"EMA_{period}"`` aligned to *df*'s index.

    Raises:
        ValueError: If ``"close"`` column is absent.
        ValueError: If *period* is less than 1.
    """
    if "close" not in df.columns:
        raise ValueError("compute_ema(): missing required column: 'close'")
    if period < 1:
        raise ValueError(f"compute_ema(): period must be ≥ 1, got {period}")

    alpha = 2.0 / (period + 1)
    close = df["close"].to_numpy(dtype=float)
    n = len(close)

    # Use a plain numpy buffer — avoids pandas iloc-enlargement constraints
    values = np.full(n, np.nan)

    if period <= n:
        # Seed: SMA of the first `period` closes
        values[period - 1] = close[:period].mean()
        # Strict recursive EMA from bar `period` onward
        for i in range(period, n):
            values[i] = close[i] * alpha + values[i - 1] * (1 - alpha)
    # else: period > n → all NaN (insufficient data), values already filled

    ema = pd.Series(values, index=df.index, name=f"EMA_{period}", dtype=float)
    return ema
