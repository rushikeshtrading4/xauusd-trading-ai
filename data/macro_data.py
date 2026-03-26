"""Macroeconomic data handling module for XAUUSD.

Provides a deterministic, dependency-free macro sentiment state. No live API
calls. No network I/O. The MacroState is populated manually or injected by a
future live feed. All logic is pure Python with no external dependencies
beyond the standard library.

Why this matters for XAUUSD
----------------------------
Gold has three macro drivers that override any technical setup:

1. DXY (US Dollar Index): inversely correlated with Gold.
   Rising DXY → bearish macro bias. Falling DXY → bullish macro bias.
2. US 10Y Treasury Yield: rising yields increase the opportunity cost of
   holding Gold. Rising yields → bearish. Falling yields → bullish.
3. High-impact news events (CPI, NFP, FOMC): cause 20–50 USD instantaneous
   moves. Trading during the 30-minute window around these releases is
   forbidden — no technical setup survives them.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Exported constants
# ---------------------------------------------------------------------------

TREND_RISING  = "RISING"
TREND_FALLING = "FALLING"
TREND_NEUTRAL = "NEUTRAL"

GOLD_BULLISH = "BULLISH"
GOLD_BEARISH = "BEARISH"
GOLD_NEUTRAL = "NEUTRAL"

_VALID_TRENDS = frozenset({TREND_RISING, TREND_FALLING, TREND_NEUTRAL})

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class EventRecord(NamedTuple):
    """An immutable record representing a scheduled high-impact event."""

    name: str
    scheduled_utc: datetime.datetime
    impact: str


@dataclass
class MacroState:
    """Mutable snapshot of the current macro environment for XAUUSD."""

    dxy_trend: str = TREND_NEUTRAL
    yield_trend: str = TREND_NEUTRAL
    gold_sentiment: str = GOLD_NEUTRAL
    active_event: str | None = None
    event_time: datetime.datetime | None = None
    news_blackout: bool = False


# ---------------------------------------------------------------------------
# MacroDataHandler
# ---------------------------------------------------------------------------


class MacroDataHandler:
    """Manages macroeconomic state and high-impact event tracking for XAUUSD.

    Pure Python, no external dependencies. All state is held in
    ``self.state`` (a :class:`MacroState`) and ``self._events``
    (a list of :class:`EventRecord`).
    """

    def __init__(self) -> None:
        self.state = MacroState()
        self._events: list[EventRecord] = []

    # ------------------------------------------------------------------
    # Trend updates
    # ------------------------------------------------------------------

    def update_dxy(self, trend: str) -> None:
        """Update the DXY trend and re-derive gold sentiment.

        Parameters
        ----------
        trend : str
            ``"RISING"``, ``"FALLING"``, or ``"NEUTRAL"`` (case-insensitive).

        Raises
        ------
        ValueError
            If *trend* is not one of the three accepted values.
        """
        normalised = trend.upper()
        if normalised not in _VALID_TRENDS:
            raise ValueError(
                f"Invalid DXY trend {trend!r}. Expected one of "
                f"{sorted(_VALID_TRENDS)}."
            )
        self.state.dxy_trend = normalised
        self.update_gold_sentiment()

    def update_yields(self, trend: str) -> None:
        """Update the US 10Y yield trend and re-derive gold sentiment.

        Parameters
        ----------
        trend : str
            ``"RISING"``, ``"FALLING"``, or ``"NEUTRAL"`` (case-insensitive).

        Raises
        ------
        ValueError
            If *trend* is not one of the three accepted values.
        """
        normalised = trend.upper()
        if normalised not in _VALID_TRENDS:
            raise ValueError(
                f"Invalid yield trend {trend!r}. Expected one of "
                f"{sorted(_VALID_TRENDS)}."
            )
        self.state.yield_trend = normalised
        self.update_gold_sentiment()

    # ------------------------------------------------------------------
    # Sentiment derivation
    # ------------------------------------------------------------------

    def update_gold_sentiment(self) -> None:
        """Derive ``gold_sentiment`` from the current DXY and yield trends.

        Rules (applied in order):

        1. DXY FALLING + Yield FALLING → BULLISH
        2. DXY RISING  + Yield RISING  → BEARISH
        3. DXY FALLING + Yield NEUTRAL → BULLISH
        4. DXY NEUTRAL + Yield FALLING → BULLISH
        5. DXY RISING  + Yield NEUTRAL → BEARISH
        6. DXY NEUTRAL + Yield RISING  → BEARISH
        7. All other combinations      → NEUTRAL
        """
        dxy = self.state.dxy_trend
        yld = self.state.yield_trend

        if dxy == TREND_FALLING and yld == TREND_FALLING:
            self.state.gold_sentiment = GOLD_BULLISH
        elif dxy == TREND_RISING and yld == TREND_RISING:
            self.state.gold_sentiment = GOLD_BEARISH
        elif dxy == TREND_FALLING and yld == TREND_NEUTRAL:
            self.state.gold_sentiment = GOLD_BULLISH
        elif dxy == TREND_NEUTRAL and yld == TREND_FALLING:
            self.state.gold_sentiment = GOLD_BULLISH
        elif dxy == TREND_RISING and yld == TREND_NEUTRAL:
            self.state.gold_sentiment = GOLD_BEARISH
        elif dxy == TREND_NEUTRAL and yld == TREND_RISING:
            self.state.gold_sentiment = GOLD_BEARISH
        else:
            self.state.gold_sentiment = GOLD_NEUTRAL

    # ------------------------------------------------------------------
    # Event registry
    # ------------------------------------------------------------------

    def add_event(
        self,
        name: str,
        scheduled_utc: datetime.datetime,
        impact: str = "HIGH",
    ) -> None:
        """Add an event to the high-impact event registry.

        Parameters
        ----------
        name : str
            Human-readable event name (e.g. ``"NFP"``, ``"CPI"``).
        scheduled_utc : datetime.datetime
            UTC datetime of the scheduled release.
        impact : str
            Impact level. Default is ``"HIGH"``.
        """
        self._events.append(
            EventRecord(name=name, scheduled_utc=scheduled_utc, impact=impact)
        )

    def is_high_impact_window(
        self,
        current_utc: datetime.datetime,
        blackout_minutes: int = 30,
    ) -> bool:
        """Return ``True`` if *current_utc* is within *blackout_minutes* of
        any HIGH-impact event in the registry.

        The window is symmetric and inclusive:
        ``[event_time - blackout_minutes, event_time + blackout_minutes]``.

        Updates ``self.state.news_blackout``, ``self.state.active_event``,
        and ``self.state.event_time`` as a side effect.

        Parameters
        ----------
        current_utc : datetime.datetime
            The time to evaluate.
        blackout_minutes : int
            Half-window in minutes. Default 30.

        Returns
        -------
        bool
            ``True`` when inside a blackout window; ``False`` otherwise.
        """
        if not self._events:
            self.state.news_blackout = False
            self.state.active_event = None
            self.state.event_time = None
            return False

        delta = datetime.timedelta(minutes=blackout_minutes)
        for event in self._events:
            if event.impact != "HIGH":
                continue
            window_start = event.scheduled_utc - delta
            window_end   = event.scheduled_utc + delta
            if window_start <= current_utc <= window_end:
                self.state.news_blackout = True
                self.state.active_event  = event.name
                self.state.event_time    = event.scheduled_utc
                return True

        self.state.news_blackout = False
        self.state.active_event  = None
        future_events = [
            e for e in self._events
            if e.impact == "HIGH" and e.scheduled_utc > current_utc
        ]
        self.state.event_time = (
            min(e.scheduled_utc for e in future_events)
            if future_events else None
        )
        return False

    # ------------------------------------------------------------------
    # Context snapshot
    # ------------------------------------------------------------------

    def get_macro_context(self) -> dict:
        """Return a snapshot dict for injection into the probability model.

        Returns
        -------
        dict
            Keys: ``"dxy_trend"``, ``"yield_trend"``, ``"gold_sentiment"``,
            ``"news_blackout"``, ``"active_event"``.
        """
        return {
            "dxy_trend":      self.state.dxy_trend,
            "yield_trend":    self.state.yield_trend,
            "gold_sentiment": self.state.gold_sentiment,
            "news_blackout":  self.state.news_blackout,
            "active_event":   self.state.active_event,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset state to defaults and clear the event registry."""
        self.state = MacroState()
        self._events = []

