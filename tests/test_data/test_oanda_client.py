"""Tests for data/oanda_client.py — OandaClient, exceptions, _raise_for_status."""

import json
import unittest
from unittest.mock import MagicMock, patch, call

from data.oanda_client import (
    OandaClient,
    OandaError,
    OandaAuthError,
    OandaRateLimitError,
    OandaAPIError,
    TIMEFRAME_TO_GRANULARITY,
    XAUUSD_INSTRUMENT,
    _raise_for_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(api_key="test-key", account_id="12345678", environment="practice"):
    """Return an OandaClient without touching environment variables."""
    return OandaClient(api_key=api_key, account_id=account_id, environment=environment)


def _mock_response(status_code=200, json_data=None, text=""):
    """Return a MagicMock simulating a requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def _candle(complete=True):
    """Minimal OANDA candle dict."""
    return {
        "time": "2024-01-01T09:00:00.000000000Z",
        "mid":  {"o": "2000.12", "h": "2005.45", "l": "1999.78", "c": "2003.21"},
        "volume": 1234,
        "complete": complete,
    }


# ---------------------------------------------------------------------------
# 1. TestOandaClientInit
# ---------------------------------------------------------------------------

class TestOandaClientInit(unittest.TestCase):
    """OandaClient.__init__() validates environment and credentials."""

    def test_practice_environment_succeeds(self):
        client = OandaClient(api_key="k", environment="practice")
        self.assertEqual(client.environment, "practice")

    def test_live_environment_succeeds(self):
        client = OandaClient(api_key="k", environment="live")
        self.assertEqual(client.environment, "live")

    def test_invalid_environment_raises_value_error(self):
        with self.assertRaises(ValueError):
            OandaClient(api_key="k", environment="demo")

    def test_missing_api_key_raises_oanda_auth_error(self):
        with patch.dict("os.environ", {}, clear=True):
            # Ensure OANDA_API_KEY is absent
            import os
            os.environ.pop("OANDA_API_KEY", None)
            with self.assertRaises(OandaAuthError):
                OandaClient()

    def test_explicit_api_key_bypasses_env_var(self):
        """Passing api_key= should succeed even with no env var set."""
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("OANDA_API_KEY", None)
            client = OandaClient(api_key="explicit-key")
            self.assertEqual(client._api_key, "explicit-key")

    def test_base_url_practice(self):
        client = OandaClient(api_key="k", environment="practice")
        self.assertEqual(client.base_url, "https://api-fxpractice.oanda.com")

    def test_base_url_live(self):
        client = OandaClient(api_key="k", environment="live")
        self.assertEqual(client.base_url, "https://api-fxtrade.oanda.com")

    def test_api_key_from_env_var(self):
        with patch.dict("os.environ", {"OANDA_API_KEY": "env-key"}):
            client = OandaClient()
            self.assertEqual(client._api_key, "env-key")


# ---------------------------------------------------------------------------
# 2. TestFetchCandles
# ---------------------------------------------------------------------------

class TestFetchCandles(unittest.TestCase):
    """OandaClient.fetch_candles() makes the correct HTTP request."""

    def test_returns_only_complete_candles(self):
        complete   = _candle(complete=True)
        incomplete = _candle(complete=False)
        resp = _mock_response(json_data={"candles": [complete, incomplete]})
        with patch("requests.get", return_value=resp):
            result = _make_client().fetch_candles("XAU_USD", "M5", 300)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["complete"])

    def test_incomplete_candles_filtered_out(self):
        resp = _mock_response(json_data={"candles": [_candle(False), _candle(False)]})
        with patch("requests.get", return_value=resp):
            result = _make_client().fetch_candles("XAU_USD", "M5", 300)
        self.assertEqual(result, [])

    def test_all_complete_candles_returned(self):
        candles = [_candle(True), _candle(True), _candle(True)]
        resp = _mock_response(json_data={"candles": candles})
        with patch("requests.get", return_value=resp):
            result = _make_client().fetch_candles("XAU_USD", "M5", 3)
        self.assertEqual(len(result), 3)

    def test_401_raises_oanda_auth_error(self):
        resp = _mock_response(status_code=401, text="Unauthorized")
        with patch("requests.get", return_value=resp):
            with self.assertRaises(OandaAuthError):
                _make_client().fetch_candles("XAU_USD", "M5")

    def test_429_raises_oanda_rate_limit_error(self):
        resp = _mock_response(status_code=429, text="Too Many Requests")
        with patch("requests.get", return_value=resp):
            with self.assertRaises(OandaRateLimitError):
                _make_client().fetch_candles("XAU_USD", "M5")

    def test_500_raises_oanda_api_error_with_status_code(self):
        resp = _mock_response(status_code=500, text="Internal Server Error")
        with patch("requests.get", return_value=resp):
            with self.assertRaises(OandaAPIError) as ctx:
                _make_client().fetch_candles("XAU_USD", "M5")
        self.assertEqual(ctx.exception.status_code, 500)

    def test_correct_url_constructed(self):
        resp = _mock_response(json_data={"candles": []})
        with patch("requests.get", return_value=resp) as mock_get:
            _make_client().fetch_candles("XAU_USD", "M5", 100)
        call_kwargs = mock_get.call_args
        url = call_kwargs[0][0] if call_kwargs[0] else call_kwargs.kwargs.get("url", "")
        self.assertIn("/v3/instruments/XAU_USD/candles", url)

    def test_correct_params_sent(self):
        resp = _mock_response(json_data={"candles": []})
        with patch("requests.get", return_value=resp) as mock_get:
            _make_client().fetch_candles("XAU_USD", "H1", 50)
        params = mock_get.call_args.kwargs.get("params") or mock_get.call_args[1].get("params")
        self.assertEqual(params["granularity"], "H1")
        self.assertEqual(params["count"], 50)
        self.assertEqual(params["price"], "M")

    def test_empty_candles_key_handled(self):
        """Response without 'candles' key returns empty list."""
        resp = _mock_response(json_data={})
        with patch("requests.get", return_value=resp):
            result = _make_client().fetch_candles("XAU_USD", "M5")
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# 3. TestStreamPrices
# ---------------------------------------------------------------------------

def _make_stream_response(lines):
    """Return a mock context-manager response whose iter_lines yields lines."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = iter(lines)
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestStreamPrices(unittest.TestCase):
    """OandaClient.stream_prices() dispatches callbacks and respects max_ticks."""

    def _price_line(self, instrument="XAU_USD", bid="2000.0", ask="2001.0"):
        return json.dumps({
            "type": "PRICE",
            "instrument": instrument,
            "bids": [{"price": bid, "liquidity": 10000000}],
            "asks": [{"price": ask, "liquidity": 10000000}],
            "time": "2024-01-01T09:00:00Z",
        }).encode()

    def _heartbeat_line(self):
        return json.dumps({"type": "HEARTBEAT", "time": "2024-01-01T09:00:00Z"}).encode()

    def test_no_account_id_raises_oanda_auth_error(self):
        client = OandaClient(api_key="k", account_id="")
        with self.assertRaises(OandaAuthError):
            client.stream_prices(["XAU_USD"], on_price=lambda m: None)

    def test_price_messages_call_on_price_callback(self):
        received = []
        lines = [self._price_line(), self._price_line()]
        mock_resp = _make_stream_response(lines)
        with patch("requests.get", return_value=mock_resp):
            _make_client().stream_prices(
                ["XAU_USD"],
                on_price=received.append,
            )
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0]["type"], "PRICE")

    def test_heartbeat_messages_call_on_heartbeat_callback(self):
        heartbeats = []
        lines = [self._heartbeat_line(), self._heartbeat_line()]
        mock_resp = _make_stream_response(lines)
        with patch("requests.get", return_value=mock_resp):
            _make_client().stream_prices(
                ["XAU_USD"],
                on_price=lambda m: None,
                on_heartbeat=heartbeats.append,
            )
        self.assertEqual(len(heartbeats), 2)

    def test_max_ticks_stops_stream_after_n_ticks(self):
        received = []
        lines = [self._price_line()] * 10
        mock_resp = _make_stream_response(lines)
        with patch("requests.get", return_value=mock_resp):
            _make_client().stream_prices(
                ["XAU_USD"],
                on_price=received.append,
                max_ticks=3,
            )
        self.assertEqual(len(received), 3)

    def test_malformed_json_lines_skipped_without_error(self):
        received = []
        lines = [
            b"not valid json",
            self._price_line(),
            b"{broken",
            self._price_line(),
        ]
        mock_resp = _make_stream_response(lines)
        with patch("requests.get", return_value=mock_resp):
            _make_client().stream_prices(
                ["XAU_USD"],
                on_price=received.append,
            )
        # Only the 2 valid PRICE lines should be received
        self.assertEqual(len(received), 2)

    def test_empty_lines_skipped(self):
        received = []
        lines = [b"", b"", self._price_line()]
        mock_resp = _make_stream_response(lines)
        with patch("requests.get", return_value=mock_resp):
            _make_client().stream_prices(
                ["XAU_USD"],
                on_price=received.append,
            )
        self.assertEqual(len(received), 1)

    def test_heartbeat_without_callback_does_not_raise(self):
        lines = [self._heartbeat_line()]
        mock_resp = _make_stream_response(lines)
        with patch("requests.get", return_value=mock_resp):
            try:
                # on_heartbeat not provided — must not raise
                _make_client().stream_prices(
                    ["XAU_USD"],
                    on_price=lambda m: None,
                )
            except Exception as exc:
                self.fail(f"stream_prices raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# 4. TestPing
# ---------------------------------------------------------------------------

class TestPing(unittest.TestCase):
    """OandaClient.ping() returns True for 200, False for errors."""

    def test_200_returns_true(self):
        resp = _mock_response(status_code=200)
        with patch("requests.get", return_value=resp):
            self.assertTrue(_make_client().ping())

    def test_request_exception_returns_false(self):
        import requests as req
        with patch("requests.get", side_effect=req.RequestException("timeout")):
            self.assertFalse(_make_client().ping())

    def test_401_returns_false(self):
        resp = _mock_response(status_code=401)
        with patch("requests.get", return_value=resp):
            self.assertFalse(_make_client().ping())

    def test_500_returns_false(self):
        resp = _mock_response(status_code=500)
        with patch("requests.get", return_value=resp):
            self.assertFalse(_make_client().ping())


# ---------------------------------------------------------------------------
# 5. TestRaiseForStatus
# ---------------------------------------------------------------------------

class TestRaiseForStatus(unittest.TestCase):
    """_raise_for_status() raises the correct exception for each HTTP status."""

    def _resp(self, status_code, text=""):
        r = MagicMock()
        r.status_code = status_code
        r.text = text
        return r

    def test_200_does_not_raise(self):
        try:
            _raise_for_status(self._resp(200))
        except Exception as exc:
            self.fail(f"_raise_for_status raised on 200: {exc}")

    def test_401_raises_oanda_auth_error(self):
        with self.assertRaises(OandaAuthError):
            _raise_for_status(self._resp(401))

    def test_429_raises_oanda_rate_limit_error(self):
        with self.assertRaises(OandaRateLimitError):
            _raise_for_status(self._resp(429))

    def test_503_raises_oanda_api_error(self):
        with self.assertRaises(OandaAPIError) as ctx:
            _raise_for_status(self._resp(503, "Service Unavailable"))
        self.assertEqual(ctx.exception.status_code, 503)

    def test_oanda_api_error_body_truncated(self):
        long_body = "x" * 500
        with self.assertRaises(OandaAPIError) as ctx:
            _raise_for_status(self._resp(503, long_body))
        self.assertIn("OANDA API error 503", str(ctx.exception))

    def test_oanda_auth_error_is_subclass_of_oanda_error(self):
        with self.assertRaises(OandaError):
            _raise_for_status(self._resp(401))

    def test_oanda_rate_limit_error_is_subclass_of_oanda_error(self):
        with self.assertRaises(OandaError):
            _raise_for_status(self._resp(429))


if __name__ == "__main__":
    unittest.main()
