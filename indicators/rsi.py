"""Relative Strength Index (RSI) indicator using Wilder's smoothing."""

import pandas as pd


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return the RSI of closing prices using Wilder's smoothed avg gain/loss.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a "close" column.
    period : int
        Look-back period (≥ 1).  Default is 14 (Wilder's original).

    Returns
    -------
    pd.Series
        RSI values in the range [0, 100], named ``"RSI"``.
        The first ``period`` values are ``NaN`` (warmup).

    Raises
    ------
    ValueError
        If "close" column is missing or ``period < 1``.
    """
    if "close" not in df.columns:
        raise ValueError("compute_rsi(): missing required column: 'close'")
    if period < 1:
        raise ValueError(f"compute_rsi(): period must be ≥ 1, got {period}")

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # avg_loss == 0 and avg_gain > 0 → RSI = 100 (all gains, no losses)
    # avg_loss == 0 and avg_gain == 0 → keep NaN (flat price, no movement)
    rsi = rsi.where((avg_loss != 0) | (avg_gain == 0), other=100.0)

    # Force warmup window to NaN (first `period` rows have insufficient history)
    rsi.iloc[:period] = float("nan")
    rsi.name = "RSI"
    return rsi
