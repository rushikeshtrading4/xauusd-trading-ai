"""Tests for data/oanda_feed.py — fetch_oanda_candles, _candles_to_dataframe."""

import unittest
from unittest.mock import MagicMock, patch, call

import pandas as pd

from data.oanda_feed import fetch_oanda_candles, _candles_to_dataframe
from data.oanda_client import (
    TIMEFRAME_TO_GRANULARITY,
    XAUUSD_INSTRUMENT,
    OandaAuthError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candle(
    time="2024-01-01T09:00:00.000000000Z",
    o="2000.12",
    h="2005.45",
    l="1999.78",
    c="2003.21",
    volume=1000,
    complete=True,
):
    """Minimal valid OANDA candle dict."""
    return {
        "time":     time,
        "mid":      {"o": o, "h": h, "l": l, "c": c},
        "volume":   volume,
        "complete": complete,
    }


def _make_mock_client(candles):
    """Return an OandaClient mock whose fetch_candles returns *candles*."""
    client = MagicMock()
    client.fetch_candles.return_value = candles
    return client


def _make_canonical_df(symbol="XAUUSD", timeframe="M5", n=5):
    """Build a minimal valid canonical DataFrame for load_market_data mocking."""
    timestamps = pd.date_range(
        start="2024-01-01T09:00:00",
        periods=n,
        freq="5min",
        tz="UTC",
    )
    return pd.DataFrame({
        "timestamp": timestamps,
        "symbol":    symbol,
        "timeframe": timeframe,
        "open":      [2000.0 + i for i in range(n)],
        "high":      [2005.0 + i for i in range(n)],
        "low":       [1999.0 + i for i in range(n)],
        "close":     [2003.0 + i for i in range(n)],
        "volume":    [1000.0 + i for i in range(n)],
    })


# ---------------------------------------------------------------------------
# 1. TestFetchOandaCandles
# ---------------------------------------------------------------------------

class TestFetchOandaCandles(unittest.TestCase):
    """fetch_oanda_candles() correct output and error handling."""

    def test_valid_response_produces_canonical_dataframe_columns(self):
        candles = [_candle()]
        client  = _make_mock_client(candles)
        df = fetch_oanda_candles("XAUUSD", "M5", count=1, client=client)
        expected = ["timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume"]
        self.assertEqual(list(df.columns), expected)

    def test_timestamps_are_utc_aware(self):
        candles = [_candle()]
        client  = _make_mock_client(candles)
        df = fetch_oanda_candles("XAUUSD", "H1", count=1, client=client)
        self.assertIsInstance(df["timestamp"].dtype, pd.DatetimeTZDtype)
        self.assertEqual(str(df["timestamp"].dt.tz), "UTC")

    def test_timestamps_sorted_ascending(self):
        candles = [
            _candle(time="2024-01-01T10:00:00.000000000Z"),
            _candle(time="2024-01-01T09:00:00.000000000Z"),
            _candle(time="2024-01-01T11:00:00.000000000Z"),
        ]
        client = _make_mock_client(candles)
        df = fetch_oanda_candles("XAUUSD", "H1", count=3, client=client)
        self.assertTrue(df["timestamp"].is_monotonic_increasing)

    def test_no_nan_values_in_price_columns(self):
        candles = [_candle(), _candle(time="2024-01-01T10:00:00.000000000Z")]
        client  = _make_mock_client(candles)
        df = fetch_oanda_candles("XAUUSD", "M5", count=2, client=client)
        for col in ["open", "high", "low", "close", "volume"]:
            self.assertFalse(df[col].isna().any(), f"NaN found in column '{col}'")

    def test_unsupported_symbol_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            fetch_oanda_candles("EURUSD", "M5", count=1, client=MagicMock())
        self.assertIn("EURUSD", str(ctx.exception))

    def test_unrecognised_timeframe_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            fetch_oanda_candles("XAUUSD", "M3", count=1, client=MagicMock())
        self.assertIn("M3", str(ctx.exception))

    def test_zero_candles_returned_raises_value_error(self):
        client = _make_mock_client([])
        with self.assertRaises(ValueError) as ctx:
            fetch_oanda_candles("XAUUSD", "M5", count=1, client=client)
        self.assertIn("0 complete candles", str(ctx.exception))

    def test_count_parameter_passed_to_fetch_candles(self):
        candles = [_candle()]
        client  = _make_mock_client(candles)
        fetch_oanda_candles("XAUUSD", "M5", count=42, client=client)
        client.fetch_candles.assert_called_once_with(XAUUSD_INSTRUMENT, "M5", 42)

    def test_default_count_uses_placeholder_candle_count(self):
        """When count=None, PLACEHOLDER_CANDLE_COUNT (300) is used."""
        from config.settings import PLACEHOLDER_CANDLE_COUNT
        candles = [_candle()]
        client  = _make_mock_client(candles)
        fetch_oanda_candles("XAUUSD", "M5", client=client)
        _, positional, _ = client.fetch_candles.mock_calls[0]
        self.assertEqual(positional[2], PLACEHOLDER_CANDLE_COUNT)

    def test_correct_granularity_passed_for_h4(self):
        candles = [_candle()]
        client  = _make_mock_client(candles)
        fetch_oanda_candles("XAUUSD", "H4", count=1, client=client)
        client.fetch_candles.assert_called_once_with(XAUUSD_INSTRUMENT, "H4", 1)

    def test_correct_granularity_passed_for_d1(self):
        candles = [_candle()]
        client  = _make_mock_client(candles)
        fetch_oanda_candles("XAUUSD", "D1", count=1, client=client)
        client.fetch_candles.assert_called_once_with(XAUUSD_INSTRUMENT, "D", 1)


# ---------------------------------------------------------------------------
# 2. TestCandlesToDataframe
# ---------------------------------------------------------------------------

class TestCandlesToDataframe(unittest.TestCase):
    """_candles_to_dataframe() correctness for column types and ordering."""

    def test_correct_column_names_and_order(self):
        df = _candles_to_dataframe([_candle()], "XAUUSD", "M5")
        self.assertEqual(
            list(df.columns),
            ["timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume"],
        )

    def test_price_columns_are_float_dtype(self):
        df = _candles_to_dataframe([_candle()], "XAUUSD", "M5")
        for col in ["open", "high", "low", "close", "volume"]:
            self.assertEqual(df[col].dtype, float, f"Column '{col}' not float")

    def test_timestamp_is_datetime_tz_dtype_utc(self):
        df = _candles_to_dataframe([_candle()], "XAUUSD", "M5")
        self.assertIsInstance(df["timestamp"].dtype, pd.DatetimeTZDtype)
        self.assertEqual(str(df["timestamp"].dt.tz), "UTC")

    def test_volume_defaults_to_zero_when_missing(self):
        candle = {
            "time": "2024-01-01T09:00:00.000000000Z",
            "mid":  {"o": "2000.0", "h": "2005.0", "l": "1999.0", "c": "2003.0"},
            "complete": True,
            # "volume" key intentionally absent
        }
        df = _candles_to_dataframe([candle], "XAUUSD", "M5")
        self.assertEqual(df["volume"].iloc[0], 0.0)

    def test_multiple_candles_sorted_by_timestamp_ascending(self):
        candles = [
            _candle(time="2024-01-01T11:00:00.000000000Z"),
            _candle(time="2024-01-01T09:00:00.000000000Z"),
            _candle(time="2024-01-01T10:00:00.000000000Z"),
        ]
        df = _candles_to_dataframe(candles, "XAUUSD", "H1")
        self.assertTrue(df["timestamp"].is_monotonic_increasing)

    def test_single_candle_produces_one_row_dataframe(self):
        df = _candles_to_dataframe([_candle()], "XAUUSD", "M5")
        self.assertEqual(len(df), 1)

    def test_symbol_and_timeframe_columns_match_inputs(self):
        df = _candles_to_dataframe([_candle()], "XAUUSD", "H4")
        self.assertEqual(df["symbol"].iloc[0], "XAUUSD")
        self.assertEqual(df["timeframe"].iloc[0], "H4")

    def test_oanda_string_prices_converted_to_float(self):
        df = _candles_to_dataframe([_candle(o="2000.12", h="2005.45")], "XAUUSD", "M5")
        self.assertAlmostEqual(df["open"].iloc[0], 2000.12)
        self.assertAlmostEqual(df["high"].iloc[0], 2005.45)

    def test_timestamp_with_z_suffix_is_utc_aware(self):
        """OANDA timestamps end in Z — must be parsed as UTC."""
        df = _candles_to_dataframe(
            [_candle(time="2024-06-15T14:30:00.000000000Z")],
            "XAUUSD", "M5",
        )
        ts = df["timestamp"].iloc[0]
        self.assertEqual(ts.utcoffset().total_seconds(), 0)


# ---------------------------------------------------------------------------
# 3. TestTimeframeMapping
# ---------------------------------------------------------------------------

class TestTimeframeMapping(unittest.TestCase):
    """TIMEFRAME_TO_GRANULARITY maps all 6 timeframes to valid granularity strings."""

    def test_all_six_timeframes_present(self):
        expected = {"M1", "M5", "M15", "H1", "H4", "D1"}
        self.assertEqual(set(TIMEFRAME_TO_GRANULARITY.keys()), expected)

    def test_d1_maps_to_capital_d(self):
        self.assertEqual(TIMEFRAME_TO_GRANULARITY["D1"], "D")

    def test_d1_does_not_map_to_d1(self):
        self.assertNotEqual(TIMEFRAME_TO_GRANULARITY["D1"], "D1")

    def test_m5_maps_to_m5(self):
        self.assertEqual(TIMEFRAME_TO_GRANULARITY["M5"], "M5")

    def test_m1_maps_to_m1(self):
        self.assertEqual(TIMEFRAME_TO_GRANULARITY["M1"], "M1")

    def test_m15_maps_to_m15(self):
        self.assertEqual(TIMEFRAME_TO_GRANULARITY["M15"], "M15")

    def test_h1_maps_to_h1(self):
        self.assertEqual(TIMEFRAME_TO_GRANULARITY["H1"], "H1")

    def test_h4_maps_to_h4(self):
        self.assertEqual(TIMEFRAME_TO_GRANULARITY["H4"], "H4")

    def test_all_granularity_strings_are_strings(self):
        for tf, gran in TIMEFRAME_TO_GRANULARITY.items():
            self.assertIsInstance(gran, str, f"Granularity for {tf} is not a str")


# ---------------------------------------------------------------------------
# 4. TestDataLoaderIntegration
# ---------------------------------------------------------------------------

class TestDataLoaderIntegration(unittest.TestCase):
    """load_market_data() routes correctly to fetch_oanda_candles for source='oanda'."""

    def _valid_df(self, timeframe):
        return _make_canonical_df(symbol="XAUUSD", timeframe=timeframe)

    def test_oanda_source_calls_fetch_oanda_candles_for_each_timeframe(self):
        from data.data_loader import load_market_data
        from config.settings import TIMEFRAMES

        with patch("data.data_loader.fetch_oanda_candles") as mock_fetch:
            # Return a valid canonical df for every timeframe call
            mock_fetch.side_effect = lambda sym, tf: self._valid_df(tf)
            result = load_market_data("XAUUSD", source="oanda")

        self.assertEqual(mock_fetch.call_count, len(TIMEFRAMES))
        # Called once per timeframe
        called_timeframes = {c.args[1] for c in mock_fetch.call_args_list}
        self.assertEqual(called_timeframes, set(TIMEFRAMES))

    def test_oanda_source_returns_dict_keyed_by_timeframe(self):
        from data.data_loader import load_market_data
        from config.settings import TIMEFRAMES

        with patch("data.data_loader.fetch_oanda_candles") as mock_fetch:
            mock_fetch.side_effect = lambda sym, tf: self._valid_df(tf)
            result = load_market_data("XAUUSD", source="oanda")

        self.assertEqual(set(result.keys()), set(TIMEFRAMES))

    def test_placeholder_source_does_not_call_fetch_oanda_candles(self):
        from data.data_loader import load_market_data

        with patch("data.data_loader.fetch_oanda_candles") as mock_fetch:
            load_market_data("XAUUSD", source="placeholder")

        mock_fetch.assert_not_called()

    def test_api_source_no_longer_accepted(self):
        """'api' was renamed to 'oanda' — must now raise ValueError."""
        from data.data_loader import load_market_data

        with self.assertRaises(ValueError) as ctx:
            load_market_data("XAUUSD", source="api")
        self.assertIn("api", str(ctx.exception))

    def test_unknown_source_raises_value_error(self):
        from data.data_loader import load_market_data

        with self.assertRaises(ValueError):
            load_market_data("XAUUSD", source="unknown")

    def test_oanda_source_symbol_passed_correctly(self):
        from data.data_loader import load_market_data

        with patch("data.data_loader.fetch_oanda_candles") as mock_fetch:
            mock_fetch.side_effect = lambda sym, tf: self._valid_df(tf)
            load_market_data("XAUUSD", source="oanda")

        # Every call must receive "XAUUSD" as the symbol
        for c in mock_fetch.call_args_list:
            self.assertEqual(c.args[0], "XAUUSD")


if __name__ == "__main__":
    unittest.main()
