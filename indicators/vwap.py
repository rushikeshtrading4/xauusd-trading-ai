"""Volume Weighted Average Price (VWAP) indicator."""

import pandas as pd


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Return the cumulative VWAP for the supplied price/volume data.

    VWAP is computed as the cumulative sum of typical-price × volume divided by
    the cumulative sum of volume.  It resets at the beginning of each session
    (i.e. uses the full slice of ``df`` passed in, so callers should subset
    the DataFrame to the desired session window before calling).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``"high"``, ``"low"``, ``"close"``, and ``"volume"``
        columns.

    Returns
    -------
    pd.Series
        VWAP values aligned to ``df.index``, named ``"VWAP"``.
        Where cumulative volume is zero the value is ``NaN``.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    required = {"high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_vwap(): missing required column(s): {sorted(missing)}"
        )

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical_price * df["volume"]

    cum_tp_vol = tp_vol.cumsum()
    cum_vol = df["volume"].cumsum()

    vwap = cum_tp_vol / cum_vol
    vwap.name = "VWAP"
    return vwap
