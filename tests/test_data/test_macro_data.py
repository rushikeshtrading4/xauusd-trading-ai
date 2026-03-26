"""Tests for data/macro_data.py"""

from __future__ import annotations

import datetime

import pytest

from data.macro_data import (
    MacroDataHandler,
    MacroState,
    EventRecord,
    TREND_RISING,
    TREND_FALLING,
    TREND_NEUTRAL,
    GOLD_BULLISH,
    GOLD_BEARISH,
    GOLD_NEUTRAL,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EVENT_TIME = datetime.datetime(2026, 1, 5, 13, 30, 0, tzinfo=datetime.timezone.utc)


def _handler() -> MacroDataHandler:
    """Return a freshly initialised handler."""
    return MacroDataHandler()


# ---------------------------------------------------------------------------
# TestMacroStateDefaults
# ---------------------------------------------------------------------------


class TestMacroStateDefaults:
    def test_default_dxy_trend(self):
        h = _handler()
        assert h.state.dxy_trend == TREND_NEUTRAL

    def test_default_yield_trend(self):
        h = _handler()
        assert h.state.yield_trend == TREND_NEUTRAL

    def test_default_gold_sentiment(self):
        h = _handler()
        assert h.state.gold_sentiment == GOLD_NEUTRAL

    def test_default_news_blackout(self):
        h = _handler()
        assert h.state.news_blackout is False

    def test_default_active_event(self):
        h = _handler()
        assert h.state.active_event is None


# ---------------------------------------------------------------------------
# TestUpdateDXY
# ---------------------------------------------------------------------------


class TestUpdateDXY:
    def test_rising(self):
        h = _handler()
        h.update_dxy("RISING")
        assert h.state.dxy_trend == "RISING"

    def test_case_insensitive_falling(self):
        h = _handler()
        h.update_dxy("falling")
        assert h.state.dxy_trend == "FALLING"

    def test_neutral(self):
        h = _handler()
        h.update_dxy("NEUTRAL")
        assert h.state.dxy_trend == "NEUTRAL"

    def test_invalid_raises(self):
        h = _handler()
        with pytest.raises(ValueError):
            h.update_dxy("INVALID")

    def test_triggers_gold_sentiment_update(self):
        h = _handler()
        # yields still NEUTRAL; DXY RISING → rule 5 → BEARISH
        h.update_dxy("RISING")
        assert h.state.gold_sentiment == GOLD_BEARISH

    def test_rising_dxy_neutral_yields_gives_bearish_gold(self):
        h = _handler()
        h.update_dxy("RISING")
        assert h.state.gold_sentiment == GOLD_BEARISH


# ---------------------------------------------------------------------------
# TestUpdateYields
# ---------------------------------------------------------------------------


class TestUpdateYields:
    def test_rising(self):
        h = _handler()
        h.update_yields("RISING")
        assert h.state.yield_trend == "RISING"

    def test_case_insensitive_falling(self):
        h = _handler()
        h.update_yields("falling")
        assert h.state.yield_trend == "FALLING"

    def test_neutral(self):
        h = _handler()
        h.update_yields("NEUTRAL")
        assert h.state.yield_trend == "NEUTRAL"

    def test_invalid_raises(self):
        h = _handler()
        with pytest.raises(ValueError):
            h.update_yields("INVALID")

    def test_triggers_gold_sentiment_update(self):
        h = _handler()
        # DXY still NEUTRAL; yields RISING → rule 6 → BEARISH
        h.update_yields("RISING")
        assert h.state.gold_sentiment == GOLD_BEARISH

    def test_rising_yields_neutral_dxy_gives_bearish_gold(self):
        h = _handler()
        h.update_yields("RISING")
        assert h.state.gold_sentiment == GOLD_BEARISH


# ---------------------------------------------------------------------------
# TestGoldSentimentDerivation
# ---------------------------------------------------------------------------


class TestGoldSentimentDerivation:
    def _derive(self, dxy: str, yld: str) -> str:
        h = _handler()
        h.state.dxy_trend = dxy
        h.state.yield_trend = yld
        h.update_gold_sentiment()
        return h.state.gold_sentiment

    def test_rule1_falling_falling_bullish(self):
        assert self._derive("FALLING", "FALLING") == GOLD_BULLISH

    def test_rule2_rising_rising_bearish(self):
        assert self._derive("RISING", "RISING") == GOLD_BEARISH

    def test_rule3_falling_neutral_bullish(self):
        assert self._derive("FALLING", "NEUTRAL") == GOLD_BULLISH

    def test_rule4_neutral_falling_bullish(self):
        assert self._derive("NEUTRAL", "FALLING") == GOLD_BULLISH

    def test_rule5_rising_neutral_bearish(self):
        assert self._derive("RISING", "NEUTRAL") == GOLD_BEARISH

    def test_rule6_neutral_rising_bearish(self):
        assert self._derive("NEUTRAL", "RISING") == GOLD_BEARISH

    def test_rule7_neutral_neutral_neutral(self):
        assert self._derive("NEUTRAL", "NEUTRAL") == GOLD_NEUTRAL

    def test_conflicting_rising_falling_neutral(self):
        assert self._derive("RISING", "FALLING") == GOLD_NEUTRAL

    def test_conflicting_falling_rising_neutral(self):
        assert self._derive("FALLING", "RISING") == GOLD_NEUTRAL


# ---------------------------------------------------------------------------
# TestHighImpactWindow
# ---------------------------------------------------------------------------


class TestHighImpactWindow:
    def test_no_events_returns_false(self):
        h = _handler()
        assert h.is_high_impact_window(_EVENT_TIME) is False

    def test_exactly_at_event_time_returns_true(self):
        h = _handler()
        h.add_event("NFP", _EVENT_TIME)
        assert h.is_high_impact_window(_EVENT_TIME) is True

    def test_29_minutes_before_returns_true(self):
        h = _handler()
        h.add_event("NFP", _EVENT_TIME)
        check_time = _EVENT_TIME - datetime.timedelta(minutes=29)
        assert h.is_high_impact_window(check_time) is True

    def test_30_minutes_before_boundary_inclusive_returns_true(self):
        h = _handler()
        h.add_event("NFP", _EVENT_TIME)
        check_time = _EVENT_TIME - datetime.timedelta(minutes=30)
        assert h.is_high_impact_window(check_time) is True

    def test_31_minutes_before_returns_false(self):
        h = _handler()
        h.add_event("NFP", _EVENT_TIME)
        check_time = _EVENT_TIME - datetime.timedelta(minutes=31)
        assert h.is_high_impact_window(check_time) is False

    def test_29_minutes_after_returns_true(self):
        h = _handler()
        h.add_event("NFP", _EVENT_TIME)
        check_time = _EVENT_TIME + datetime.timedelta(minutes=29)
        assert h.is_high_impact_window(check_time) is True

    def test_31_minutes_after_returns_false(self):
        h = _handler()
        h.add_event("NFP", _EVENT_TIME)
        check_time = _EVENT_TIME + datetime.timedelta(minutes=31)
        assert h.is_high_impact_window(check_time) is False

    def test_inside_window_sets_news_blackout_true(self):
        h = _handler()
        h.add_event("CPI", _EVENT_TIME)
        h.is_high_impact_window(_EVENT_TIME)
        assert h.state.news_blackout is True

    def test_outside_window_clears_news_blackout(self):
        h = _handler()
        h.add_event("CPI", _EVENT_TIME)
        # first go inside
        h.is_high_impact_window(_EVENT_TIME)
        assert h.state.news_blackout is True
        # then go outside
        outside = _EVENT_TIME + datetime.timedelta(minutes=60)
        h.is_high_impact_window(outside)
        assert h.state.news_blackout is False

    def test_inside_window_sets_active_event_name(self):
        h = _handler()
        h.add_event("FOMC", _EVENT_TIME)
        h.is_high_impact_window(_EVENT_TIME)
        assert h.state.active_event == "FOMC"

    def test_inside_window_sets_event_time_to_scheduled_utc(self):
        h = _handler()
        h.add_event("NFP", _EVENT_TIME)
        h.is_high_impact_window(_EVENT_TIME)
        assert h.state.event_time == _EVENT_TIME

    def test_outside_window_sets_event_time_to_next_future_event(self):
        h = _handler()
        future_event_time = _EVENT_TIME + datetime.timedelta(hours=5)
        h.add_event("NFP", _EVENT_TIME)
        h.add_event("FOMC", future_event_time)
        # Check at a time between the two events (NFP window has passed)
        between = _EVENT_TIME + datetime.timedelta(hours=2)
        h.is_high_impact_window(between)
        assert h.state.event_time == future_event_time

    def test_no_events_event_time_is_none(self):
        h = _handler()
        h.is_high_impact_window(_EVENT_TIME)
        assert h.state.event_time is None

    def test_outside_window_no_future_events_event_time_is_none(self):
        h = _handler()
        h.add_event("NFP", _EVENT_TIME)
        # Check 2 hours AFTER the event — no future events remain
        after = _EVENT_TIME + datetime.timedelta(hours=2)
        h.is_high_impact_window(after)
        assert h.state.event_time is None


# ---------------------------------------------------------------------------
# TestGetMacroContext
# ---------------------------------------------------------------------------


class TestGetMacroContext:
    def test_returns_all_required_keys(self):
        h = _handler()
        ctx = h.get_macro_context()
        assert set(ctx.keys()) == {
            "dxy_trend",
            "yield_trend",
            "gold_sentiment",
            "news_blackout",
            "active_event",
        }

    def test_gold_sentiment_reflects_state(self):
        h = _handler()
        h.update_dxy("FALLING")
        h.update_yields("FALLING")
        ctx = h.get_macro_context()
        assert ctx["gold_sentiment"] == GOLD_BULLISH

    def test_news_blackout_reflects_state(self):
        h = _handler()
        h.add_event("NFP", _EVENT_TIME)
        h.is_high_impact_window(_EVENT_TIME)
        ctx = h.get_macro_context()
        assert ctx["news_blackout"] is True

    def test_dxy_trend_after_update(self):
        h = _handler()
        h.update_dxy("RISING")
        ctx = h.get_macro_context()
        assert ctx["dxy_trend"] == "RISING"

    def test_news_blackout_true_after_event_window(self):
        h = _handler()
        h.add_event("CPI", _EVENT_TIME)
        h.is_high_impact_window(_EVENT_TIME)
        ctx = h.get_macro_context()
        assert ctx["news_blackout"] is True

    def test_context_is_snapshot_not_live(self):
        h = _handler()
        ctx = h.get_macro_context()
        ctx["dxy_trend"] = "RISING"
        # mutating the returned dict must not change handler state
        assert h.state.dxy_trend == TREND_NEUTRAL


# ---------------------------------------------------------------------------
# TestAddEvent
# ---------------------------------------------------------------------------


class TestAddEvent:
    def test_add_event_stores_in_registry(self):
        h = _handler()
        h.add_event("NFP", _EVENT_TIME)
        assert len(h._events) == 1
        assert h._events[0].name == "NFP"

    def test_multiple_events_can_be_added(self):
        h = _handler()
        h.add_event("NFP", _EVENT_TIME)
        h.add_event("CPI", _EVENT_TIME + datetime.timedelta(days=1))
        assert len(h._events) == 2

    def test_only_high_impact_triggers_blackout(self):
        h = _handler()
        h.add_event("MINOR", _EVENT_TIME, impact="HIGH")
        assert h.is_high_impact_window(_EVENT_TIME) is True

    def test_low_impact_event_does_not_trigger_blackout(self):
        h = _handler()
        h.add_event("ISM", _EVENT_TIME, impact="LOW")
        assert h.is_high_impact_window(_EVENT_TIME) is False


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all_trend_state(self):
        h = _handler()
        h.update_dxy("RISING")
        h.update_yields("RISING")
        assert h.state.gold_sentiment == GOLD_BEARISH
        h.reset()
        assert h.state.dxy_trend == TREND_NEUTRAL
        assert h.state.yield_trend == TREND_NEUTRAL

    def test_reset_gold_sentiment_neutral(self):
        h = _handler()
        h.update_dxy("RISING")
        h.update_yields("RISING")
        h.reset()
        assert h.state.gold_sentiment == GOLD_NEUTRAL

    def test_reset_clears_event_registry(self):
        h = _handler()
        h.add_event("NFP", _EVENT_TIME)
        h.reset()
        # after reset, no events → window always False
        assert h.is_high_impact_window(_EVENT_TIME) is False


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_same_gold_sentiment(self):
        h1 = _handler()
        h1.update_dxy("FALLING")
        h1.update_yields("FALLING")

        h2 = _handler()
        h2.update_dxy("FALLING")
        h2.update_yields("FALLING")

        assert h1.state.gold_sentiment == h2.state.gold_sentiment

    def test_get_macro_context_idempotent(self):
        h = _handler()
        h.update_dxy("RISING")
        ctx1 = h.get_macro_context()
        ctx2 = h.get_macro_context()
        assert ctx1 == ctx2
