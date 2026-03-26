"""
OANDA REST API v20 client for XAUUSD data ingestion.

Pure HTTP layer — no pandas, no business logic. Every method returns
raw Python dicts or generators. Conversion to DataFrames is done in
oanda_feed.py.

Environment variables
─────────────────────
OANDA_API_KEY     — Bearer token from your OANDA account
OANDA_ACCOUNT_ID  — your numeric account ID (for streaming)

Never hardcode credentials. The constructor reads env vars by default
but accepts explicit arguments for testing.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Generator

import requests

# OANDA v20 API base URLs
_BASE_URLS: dict[str, str] = {
    "practice": "https://api-fxpractice.oanda.com",
    "live":     "https://api-fxtrade.oanda.com",
}

# Mapping from our timeframe labels to OANDA granularity strings
TIMEFRAME_TO_GRANULARITY: dict[str, str] = {
    "M1":  "M1",
    "M5":  "M5",
    "M15": "M15",
    "H1":  "H1",
    "H4":  "H4",
    "D1":  "D",
}

# OANDA instrument name for XAUUSD
XAUUSD_INSTRUMENT = "XAU_USD"

# Default number of candles to request per timeframe
DEFAULT_CANDLE_COUNT = 300

# Streaming read chunk size in bytes
_STREAM_CHUNK_SIZE = 512


class OandaClient:
    """Thin HTTP wrapper around the OANDA v20 REST API.

    Parameters
    ----------
    api_key : str | None
        Bearer token. If None, reads OANDA_API_KEY from environment.
    account_id : str | None
        Account ID for streaming. If None, reads OANDA_ACCOUNT_ID from
        environment.
    environment : str
        "practice" (default) or "live".

    Raises
    ------
    ValueError
        If environment is not "practice" or "live".
    OandaAuthError
        If api_key is not provided and OANDA_API_KEY is not set.
    """

    def __init__(
        self,
        api_key:     str | None = None,
        account_id:  str | None = None,
        environment: str = "practice",
    ) -> None:
        if environment not in _BASE_URLS:
            raise ValueError(
                f"Invalid environment {environment!r}. "
                f"Expected 'practice' or 'live'."
            )
        self._environment = environment
        self._base_url    = _BASE_URLS[environment]

        resolved_key = api_key or os.environ.get("OANDA_API_KEY", "")
        if not resolved_key:
            raise OandaAuthError(
                "No API key provided. Set OANDA_API_KEY environment variable "
                "or pass api_key= to OandaClient()."
            )
        self._api_key    = resolved_key
        self._account_id = account_id or os.environ.get("OANDA_ACCOUNT_ID", "")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        """The resolved base URL for this environment."""
        return self._base_url

    @property
    def environment(self) -> str:
        """'practice' or 'live'."""
        return self._environment

    @property
    def _headers(self) -> dict[str, str]:
        """Standard OANDA v20 request headers."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }

    # ------------------------------------------------------------------
    # Candle fetching
    # ------------------------------------------------------------------

    def fetch_candles(
        self,
        instrument:  str,
        granularity: str,
        count:       int = DEFAULT_CANDLE_COUNT,
    ) -> list[dict]:
        """Fetch OHLCV candles for an instrument.

        Requests mid-price candles (price=M). Only complete candles are
        returned (incomplete=False is not passed — OANDA returns both;
        we filter complete=True candles in this method).

        Parameters
        ----------
        instrument : str
            OANDA instrument name, e.g. "XAU_USD".
        granularity : str
            OANDA granularity string, e.g. "M5", "H1", "D".
        count : int
            Number of candles to request. Maximum 5000 per OANDA docs.

        Returns
        -------
        list[dict]
            List of raw candle dicts with keys: time, mid (o/h/l/c), volume,
            complete. Only complete candles are included.

        Raises
        ------
        OandaAPIError
            If the HTTP response status is not 200.
        OandaRateLimitError
            If the HTTP response status is 429.
        """
        url    = f"{self._base_url}/v3/instruments/{instrument}/candles"
        params = {
            "granularity": granularity,
            "count":       count,
            "price":       "M",
        }

        response = requests.get(url, headers=self._headers, params=params, timeout=30)
        _raise_for_status(response)

        data    = response.json()
        candles = data.get("candles", [])
        return [c for c in candles if c.get("complete", True)]

    # ------------------------------------------------------------------
    # Price streaming
    # ------------------------------------------------------------------

    def stream_prices(
        self,
        instruments: list[str],
        on_price:    Callable[[dict], None],
        *,
        on_heartbeat: Callable[[dict], None] | None = None,
        max_ticks: int | None = None,
    ) -> None:
        """Stream live prices for a list of instruments.

        Opens a persistent HTTP streaming connection to the OANDA pricing
        stream endpoint. Calls on_price() for every PRICE message and
        optionally calls on_heartbeat() for HEARTBEAT messages.

        Blocks until the connection is closed, an exception is raised, or
        max_ticks price messages have been received.

        Parameters
        ----------
        instruments : list[str]
            OANDA instrument names, e.g. ["XAU_USD"].
        on_price : Callable[[dict], None]
            Called with the parsed price dict for every PRICE tick.
        on_heartbeat : Callable[[dict], None] | None
            Called with the heartbeat dict. Optional.
        max_ticks : int | None
            Stop after this many PRICE ticks. None = run forever.

        Raises
        ------
        OandaAuthError
            If no account_id is configured.
        OandaAPIError
            If the HTTP response status is not 200.
        """
        if not self._account_id:
            raise OandaAuthError(
                "No account ID configured for streaming. Set OANDA_ACCOUNT_ID "
                "environment variable or pass account_id= to OandaClient()."
            )

        url    = f"{self._base_url}/v3/accounts/{self._account_id}/pricing/stream"
        params = {"instruments": ",".join(instruments)}

        tick_count = 0
        with requests.get(
            url,
            headers=self._headers,
            params=params,
            stream=True,
            timeout=30,
        ) as response:
            _raise_for_status(response)
            for line in response.iter_lines(chunk_size=_STREAM_CHUNK_SIZE):
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type", "")
                if msg_type == "PRICE":
                    on_price(msg)
                    tick_count += 1
                    if max_ticks is not None and tick_count >= max_ticks:
                        break
                elif msg_type == "HEARTBEAT" and on_heartbeat is not None:
                    on_heartbeat(msg)

    # ------------------------------------------------------------------
    # Account info (lightweight connectivity check)
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if the API is reachable and the key is valid.

        Makes a lightweight GET /v3/accounts request. Returns False
        (rather than raising) so callers can use this as a health check.
        """
        try:
            url      = f"{self._base_url}/v3/accounts"
            response = requests.get(url, headers=self._headers, timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OandaError(Exception):
    """Base class for OANDA client errors."""


class OandaAuthError(OandaError):
    """Raised when credentials are missing or invalid (HTTP 401)."""


class OandaRateLimitError(OandaError):
    """Raised when OANDA rate-limits the request (HTTP 429)."""


class OandaAPIError(OandaError):
    """Raised for any other non-200 OANDA HTTP response.

    Attributes
    ----------
    status_code : int
    body : str
    """

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body        = body
        super().__init__(f"OANDA API error {status_code}: {body[:200]}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _raise_for_status(response: requests.Response) -> None:
    """Raise the appropriate OandaError for non-200 responses."""
    if response.status_code == 200:
        return
    if response.status_code == 401:
        raise OandaAuthError(
            "Authentication failed (HTTP 401). Check your API key."
        )
    if response.status_code == 429:
        raise OandaRateLimitError(
            "Rate limit exceeded (HTTP 429). Slow down requests."
        )
    raise OandaAPIError(response.status_code, response.text)
