"""
Tests for analysis/swing_points.py — detect_swings()

Each test builds a small, fully deterministic OHLC + ATR DataFrame so that
expected swing locations are known in advance and assertions are exact.

Fixture helpers
---------------
_make_flat_df   — flat-price candles, no natural pivots.
_make_peak_df   — single clear swing high in the interior.
_make_trough_df — single clear swing low in the interior.
"""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.swing_points import detect_swings

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_ATR = 2.0          # ATR used in all fixtures
_BASE = 100.0       # base close price


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _base_df(highs: list[float], lows: list[float], closes: list[float]) -> pd.DataFrame:
    """Build a minimal schema-compliant DataFrame from explicit H/L/C arrays.

    Opens are set equal to closes; volume and ATR are constant so that only
    the explicitly supplied price levels drive swing detection outcomes.
    """
    n = len(closes)
    assert len(highs) == n and len(lows) == n, "H/L/C arrays must be the same length"
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
            "open":   closes,
            "high":   highs,
            "low":    lows,
            "close":  closes,
            "volume": [1_000.0] * n,
            "ATR":    [_ATR] * n,
        }
    )


def _flat_df(n: int = 10) -> pd.DataFrame:
    """Return a flat candle series — no natural pivots anywhere."""
    closes = [_BASE] * n
    highs  = [_BASE + 0.1] * n
    lows   = [_BASE - 0.1] * n
    return _base_df(highs, lows, closes)


def _peak_df() -> pd.DataFrame:
    """Return a series with one unambiguous swing high at index 4.

    Layout (10 candles, index 0-9):
        Index 4 has the highest high by a large margin.
        After the peak, close drops by > ATR * 0.5 (drop = 5.0 > 1.0).
        Edges (0,1,8,9) are excluded from detection automatically.
    """
    closes = [100, 101, 102, 103, 110, 105, 103, 101, 100, 99]
    highs  = [c + 0.5 for c in closes]
    lows   = [c - 0.5 for c in closes]
    # Boost index-4 high so it dominates its window unambiguously
    highs[4] = 120.0
    return _base_df(highs, lows, closes)


def _trough_df() -> pd.DataFrame:
    """Return a series with one unambiguous swing low at index 5.

    Layout (10 candles, index 0-9):
        Index 5 has the lowest low by a large margin.
        After the trough, close rises by > ATR * 0.5 (rise = 5.0 > 1.0).
        Edges are excluded automatically.
    """
    closes = [100, 101, 102, 101, 100, 90, 95, 98, 100, 101]
    highs  = [c + 0.5 for c in closes]
    lows   = [c - 0.5 for c in closes]
    # Drop index-5 low so it dominates its window unambiguously
    lows[5] = 80.0
    return _base_df(highs, lows, closes)


# ---------------------------------------------------------------------------
# 1. Output structure
# ---------------------------------------------------------------------------


def test_returns_dataframe() -> None:
    """detect_swings must return a pandas DataFrame."""
    result = detect_swings(_flat_df())
    assert isinstance(result, pd.DataFrame)


def test_output_has_swing_high_column() -> None:
    """The returned DataFrame must contain a 'swing_high' boolean column."""
    result = detect_swings(_flat_df())
    assert "swing_high" in result.columns
    assert result["swing_high"].dtype == bool


def test_output_has_swing_low_column() -> None:
    """The returned DataFrame must contain a 'swing_low' boolean column."""
    result = detect_swings(_flat_df())
    assert "swing_low" in result.columns
    assert result["swing_low"].dtype == bool


def test_output_length_equals_input_length() -> None:
    """Output row count must be identical to input row count."""
    df = _flat_df(20)
    result = detect_swings(df)
    assert len(result) == len(df)


def test_does_not_mutate_original_dataframe() -> None:
    """detect_swings must not modify the caller's DataFrame (returns a copy)."""
    df = _flat_df()
    _ = detect_swings(df)
    assert "swing_high" not in df.columns
    assert "swing_low" not in df.columns


def test_original_columns_preserved() -> None:
    """All original columns must still be present in the returned DataFrame."""
    df = _flat_df()
    result = detect_swings(df)
    for col in df.columns:
        assert col in result.columns


# ---------------------------------------------------------------------------
# 2. Swing high detection
# ---------------------------------------------------------------------------


def test_swing_high_detected_at_peak() -> None:
    """A clear local maximum with sufficient post-peak drop must be a swing high."""
    result = detect_swings(_peak_df())
    assert result.loc[4, "swing_high"] is True or result.loc[4, "swing_high"] == True


def test_no_swing_high_on_flat_series() -> None:
    """A flat price series contains no swing highs."""
    result = detect_swings(_flat_df())
    assert not result["swing_high"].any()


def test_swing_high_not_flagged_as_swing_low() -> None:
    """A swing high candle must not simultaneously be marked as a swing low."""
    result = detect_swings(_peak_df())
    swing_high_idx = result.index[result["swing_high"]].tolist()
    for idx in swing_high_idx:
        assert not result.loc[idx, "swing_low"], (
            f"Index {idx} is both swing_high and swing_low"
        )


# ---------------------------------------------------------------------------
# 3. Swing low detection
# ---------------------------------------------------------------------------


def test_swing_low_detected_at_trough() -> None:
    """A clear local minimum with sufficient post-trough rise must be a swing low."""
    result = detect_swings(_trough_df())
    assert result.loc[5, "swing_low"] is True or result.loc[5, "swing_low"] == True


def test_no_swing_low_on_flat_series() -> None:
    """A flat price series contains no swing lows."""
    result = detect_swings(_flat_df())
    assert not result["swing_low"].any()


def test_swing_low_not_flagged_as_swing_high() -> None:
    """A swing low candle must not simultaneously be marked as a swing high."""
    result = detect_swings(_trough_df())
    swing_low_idx = result.index[result["swing_low"]].tolist()
    for idx in swing_low_idx:
        assert not result.loc[idx, "swing_high"], (
            f"Index {idx} is both swing_low and swing_high"
        )


# ---------------------------------------------------------------------------
# 4. Displacement filter
# ---------------------------------------------------------------------------


def test_weak_pivot_high_rejected_by_displacement_filter() -> None:
    """A pivot high followed by a tiny close drop must NOT become a swing high.

    Here the post-peak drop is only 0.01 (< ATR * 0.5 = 1.0), so the
    displacement filter must reject it even though it is a window-local max.
    """
    # Index 2 is the local max; close barely moves after it
    closes = [100.0, 101.0, 110.0, 109.99, 109.98, 109.97, 109.96, 109.95]
    highs  = [c + 0.5 for c in closes]
    lows   = [c - 0.5 for c in closes]
    highs[2] = 120.0   # clear pivot high
    df = _base_df(highs, lows, closes)
    result = detect_swings(df)
    assert not result.loc[2, "swing_high"], (
        "Weak pivot high should be rejected by displacement filter"
    )


def test_weak_pivot_low_rejected_by_displacement_filter() -> None:
    """A pivot low followed by a tiny close rise must NOT become a swing low.

    Here the post-trough rise is only 0.01 (< ATR * 0.5 = 1.0).
    """
    closes = [100.0, 99.0, 90.0, 90.01, 90.02, 90.03, 90.04, 90.05]
    highs  = [c + 0.5 for c in closes]
    lows   = [c - 0.5 for c in closes]
    lows[2] = 80.0    # clear pivot low
    df = _base_df(highs, lows, closes)
    result = detect_swings(df)
    assert not result.loc[2, "swing_low"], (
        "Weak pivot low should be rejected by displacement filter"
    )


# ---------------------------------------------------------------------------
# 5. Edge candles (first 2 / last 2 must never be swings)
# ---------------------------------------------------------------------------


def test_no_swing_at_first_two_candles() -> None:
    """Indices 0 and 1 must never be marked as swing high or low (edge exclusion)."""
    result = detect_swings(_peak_df())
    for idx in (0, 1):
        assert not result.loc[idx, "swing_high"], f"swing_high at edge index {idx}"
        assert not result.loc[idx, "swing_low"],  f"swing_low at edge index {idx}"


def test_no_swing_at_last_two_candles() -> None:
    """The last two indices must never be marked as swing high or low (edge exclusion)."""
    result = detect_swings(_peak_df())
    n = len(result)
    for idx in (n - 2, n - 1):
        assert not result.loc[idx, "swing_high"], f"swing_high at edge index {idx}"
        assert not result.loc[idx, "swing_low"],  f"swing_low at edge index {idx}"


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


def test_deterministic_results() -> None:
    """Calling detect_swings twice with identical input must produce identical output."""
    df = _peak_df()
    result1 = detect_swings(df)
    result2 = detect_swings(df)
    pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# 7. Error handling
# ---------------------------------------------------------------------------


def test_missing_atr_column_raises_value_error() -> None:
    """Omitting the required 'ATR' column must raise ValueError."""
    df = _flat_df().drop(columns=["ATR"])
    with pytest.raises(ValueError, match="ATR"):
        detect_swings(df)


@pytest.mark.parametrize("missing_col", ["high", "low", "close", "timestamp", "volume"])
def test_missing_required_column_raises_value_error(missing_col: str) -> None:
    """Any missing required column must raise ValueError naming that column."""
    df = _flat_df().drop(columns=[missing_col])
    with pytest.raises(ValueError, match=missing_col):
        detect_swings(df)


def test_too_few_rows_raises_value_error() -> None:
    """A DataFrame with fewer than 5 rows must raise ValueError."""
    df = _flat_df(n=4)
    with pytest.raises(ValueError, match="at least 5"):
        detect_swings(df)


def test_exactly_five_rows_does_not_raise() -> None:
    """A DataFrame with exactly 5 rows must not raise (minimum valid input)."""
    df = _flat_df(n=5)
    result = detect_swings(df)
    assert len(result) == 5
