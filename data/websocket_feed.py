"""
WebSocket / streaming price feed for XAUUSD.

Wraps OandaClient.stream_prices() with reconnection logic and a
clean start/stop interface. Runs in a background thread so main.py
can call start() and continue its cycle loop without blocking.

Usage
-----
    feed = OandaPriceFeed()
    feed.start(on_tick=my_callback)   # non-blocking
    # ... main loop runs ...
    feed.stop()

The on_tick callback receives a normalised tick dict:
    {
        "instrument": "XAU_USD",
        "bid":  float,
        "ask":  float,
        "mid":  float,
        "time": str   (ISO-8601 UTC)
    }
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

from data.oanda_client import OandaClient, OandaError, XAUUSD_INSTRUMENT

logger = logging.getLogger(__name__)

# Seconds to wait before reconnecting after a stream error
_RECONNECT_DELAY: int = 5


class OandaPriceFeed:
    """Background streaming price feed using OANDA's pricing stream API.

    Parameters
    ----------
    client : OandaClient | None
        Injected client for testing. If None, creates OandaClient from env vars.
    instruments : list[str]
        OANDA instrument names to stream. Defaults to ["XAU_USD"].
    reconnect_delay : int
        Seconds to wait before reconnecting after a dropped connection.
    """

    def __init__(
        self,
        client:          OandaClient | None = None,
        instruments:     list[str] | None = None,
        reconnect_delay: int = _RECONNECT_DELAY,
    ) -> None:
        self._client          = client
        self._instruments     = instruments or [XAUUSD_INSTRUMENT]
        self._reconnect_delay = reconnect_delay
        self._thread:  threading.Thread | None = None
        self._stop_event = threading.Event()
        self._on_tick:   Callable[[dict], None] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self, on_tick: Callable[[dict], None]) -> None:
        """Start the streaming feed in a background thread.

        Parameters
        ----------
        on_tick : Callable[[dict], None]
            Called for every price tick. Receives a normalised dict with
            keys: instrument, bid, ask, mid, time.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("OandaPriceFeed is already running.")
            return

        self._on_tick = on_tick
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="OandaPriceFeed",
        )
        self._thread.start()
        logger.info("OandaPriceFeed started -- streaming %s", self._instruments)

    def stop(self) -> None:
        """Signal the streaming thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("OandaPriceFeed stopped.")

    @property
    def is_running(self) -> bool:
        """True if the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal streaming loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Reconnecting stream loop. Runs in the background thread."""
        while not self._stop_event.is_set():
            try:
                client = self._client or OandaClient(
                    environment=os.environ.get("OANDA_ENVIRONMENT", "practice")
                )
                logger.info("OandaPriceFeed connecting...")
                client.stream_prices(
                    self._instruments,
                    on_price=self._handle_price,
                    on_heartbeat=self._handle_heartbeat,
                )
            except OandaError as exc:
                if self._stop_event.is_set():
                    break
                logger.warning(
                    "Stream error: %s. Reconnecting in %ds...",
                    exc, self._reconnect_delay,
                )
                time.sleep(self._reconnect_delay)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.error("Unexpected stream error: %s", exc, exc_info=True)
                time.sleep(self._reconnect_delay)

    def _handle_price(self, msg: dict) -> None:
        """Normalise a raw OANDA PRICE message and call on_tick."""
        if self._on_tick is None:
            return
        try:
            bids = msg.get("bids", [{}])
            asks = msg.get("asks", [{}])
            bid  = float(bids[0].get("price", 0)) if bids else 0.0
            ask  = float(asks[0].get("price", 0)) if asks else 0.0
            mid  = round((bid + ask) / 2, 5) if bid and ask else 0.0
            tick = {
                "instrument": msg.get("instrument", ""),
                "bid":  bid,
                "ask":  ask,
                "mid":  mid,
                "time": msg.get("time", ""),
            }
            self._on_tick(tick)
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("Failed to parse price tick: %s", exc)

    def _handle_heartbeat(self, msg: dict) -> None:
        logger.debug("Stream heartbeat: %s", msg.get("time", ""))
