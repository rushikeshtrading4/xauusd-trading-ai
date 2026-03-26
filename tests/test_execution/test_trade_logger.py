"""Tests for execution/trade_logger.py"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from execution.trade_logger import (
    TradeLogger,
    TradeRecord,
    OUTCOME_WIN,
    OUTCOME_LOSS,
    OUTCOME_PENDING,
    OUTCOME_CANCELLED,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


def _signal(**overrides):
    base = {
        "pair":          "XAUUSD",
        "timeframe":     "M5",
        "bias":          "BUY",
        "entry":         2000.0,
        "stop_loss":     1990.0,
        "take_profit":   2020.0,
        "risk_reward":   2.0,
        "confidence":    80.0,
        "invalidation":  1990.0,
        "atr":           10.0,
        "mtf_bias":      "BULLISH",
        "bias_strength": "STRONG",
        "context":       "TREND",
    }
    base.update(overrides)
    return base


_FIXED_TS = "2026-01-05T13:30:00+00:00"


# ---------------------------------------------------------------------------
# 1. TestTradeRecord
# ---------------------------------------------------------------------------


class TestTradeRecord:
    def _make(self, **kw) -> TradeRecord:
        defaults = dict(
            signal_id="XAUUSD_M5_ts",
            timestamp_utc=_FIXED_TS,
            pair="XAUUSD",
            timeframe="M5",
            bias="BUY",
            entry=2000.0,
            stop_loss=1990.0,
            take_profit=2020.0,
            risk_reward=2.0,
            confidence=80.0,
            invalidation=1990.0,
            atr=10.0,
            mtf_bias="BULLISH",
            bias_strength="STRONG",
            context="TREND",
        )
        defaults.update(kw)
        return TradeRecord(**defaults)

    def test_construction_defaults(self):
        r = self._make()
        assert r.outcome    == OUTCOME_PENDING
        assert r.exit_price is None
        assert r.pnl_r      is None
        assert r.notes      == ""

    def test_is_resolved_pending(self):
        assert self._make().is_resolved() is False

    def test_is_resolved_win(self):
        assert self._make(outcome=OUTCOME_WIN).is_resolved() is True

    def test_is_resolved_loss(self):
        assert self._make(outcome=OUTCOME_LOSS).is_resolved() is True

    def test_is_resolved_cancelled(self):
        assert self._make(outcome=OUTCOME_CANCELLED).is_resolved() is True

    def test_to_dict_is_plain_dict(self):
        r = self._make()
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["pair"] == "XAUUSD"
        assert d["outcome"] == OUTCOME_PENDING

    def test_from_dict_roundtrip(self):
        r = self._make(outcome=OUTCOME_WIN, exit_price=2020.0, pnl_r=2.0)
        d = r.to_dict()
        r2 = TradeRecord.from_dict(d)
        assert r == r2


# ---------------------------------------------------------------------------
# 2. TestLogSignal
# ---------------------------------------------------------------------------


class TestLogSignal:
    def test_valid_signal_logged(self):
        logger = TradeLogger()
        rec = logger.log_signal(_signal(), timestamp_utc=_FIXED_TS)
        assert len(logger.all_records()) == 1
        assert rec.pair == "XAUUSD"
        assert rec.outcome == OUTCOME_PENDING

    def test_missing_key_raises(self):
        logger = TradeLogger()
        bad = _signal()
        del bad["entry"]
        with pytest.raises(ValueError, match="missing required key"):
            logger.log_signal(bad, timestamp_utc=_FIXED_TS)

    def test_invalid_bias_raises(self):
        logger = TradeLogger()
        with pytest.raises(ValueError, match="invalid bias"):
            logger.log_signal(_signal(bias="LONG"), timestamp_utc=_FIXED_TS)

    def test_signal_id_format(self):
        logger = TradeLogger()
        rec = logger.log_signal(_signal(), timestamp_utc=_FIXED_TS)
        assert rec.signal_id == f"XAUUSD_M5_{_FIXED_TS}"

    def test_timestamp_defaults_to_now(self):
        logger = TradeLogger()
        rec = logger.log_signal(_signal())
        # Just assert it's a non-empty ISO string with UTC offset
        assert "+" in rec.timestamp_utc or rec.timestamp_utc.endswith("Z")

    def test_custom_timestamp_used(self):
        logger = TradeLogger()
        rec = logger.log_signal(_signal(), timestamp_utc=_FIXED_TS)
        assert rec.timestamp_utc == _FIXED_TS

    def test_notes_stored(self):
        logger = TradeLogger()
        rec = logger.log_signal(_signal(), timestamp_utc=_FIXED_TS, notes="test note")
        assert rec.notes == "test note"

    def test_case_insensitive_bias_normalised(self):
        logger = TradeLogger()
        rec = logger.log_signal(_signal(bias="buy"), timestamp_utc=_FIXED_TS)
        assert rec.bias == "BUY"

    def test_sell_bias_accepted(self):
        logger = TradeLogger()
        rec = logger.log_signal(_signal(bias="SELL"), timestamp_utc=_FIXED_TS)
        assert rec.bias == "SELL"


# ---------------------------------------------------------------------------
# 3. TestResolveTrade
# ---------------------------------------------------------------------------


class TestResolveTrade:
    def _logged(self, notes: str = "", **kw) -> tuple[TradeLogger, TradeRecord]:
        logger = TradeLogger()
        rec = logger.log_signal(_signal(**kw), timestamp_utc=_FIXED_TS, notes=notes)
        return logger, rec

    def test_win_with_exit_price_sets_pnl(self):
        logger, rec = self._logged()
        updated = logger.resolve_trade(
            rec.signal_id, outcome="WIN", exit_price=2020.0
        )
        assert updated.outcome == OUTCOME_WIN
        assert updated.exit_price == 2020.0
        assert updated.pnl_r == pytest.approx(2.0)

    def test_loss_pnl_is_negative_one(self):
        logger, rec = self._logged()
        updated = logger.resolve_trade(
            rec.signal_id, outcome="LOSS", exit_price=1990.0
        )
        assert updated.outcome == OUTCOME_LOSS
        assert updated.pnl_r == pytest.approx(-1.0)

    def test_cancelled_pnl_is_none(self):
        logger, rec = self._logged()
        updated = logger.resolve_trade(rec.signal_id, outcome="CANCELLED")
        assert updated.outcome == OUTCOME_CANCELLED
        assert updated.pnl_r is None

    def test_already_resolved_raises(self):
        logger, rec = self._logged()
        logger.resolve_trade(rec.signal_id, outcome="WIN", exit_price=2020.0)
        with pytest.raises(ValueError, match="already resolved"):
            logger.resolve_trade(rec.signal_id, outcome="LOSS", exit_price=1990.0)

    def test_invalid_outcome_raises(self):
        logger, rec = self._logged()
        with pytest.raises(ValueError, match="invalid outcome"):
            logger.resolve_trade(rec.signal_id, outcome="MOOT", exit_price=2000.0)

    def test_win_without_exit_price_raises(self):
        logger, rec = self._logged()
        with pytest.raises(ValueError, match="exit_price is required"):
            logger.resolve_trade(rec.signal_id, outcome="WIN")

    def test_loss_without_exit_price_raises(self):
        logger, rec = self._logged()
        with pytest.raises(ValueError, match="exit_price is required"):
            logger.resolve_trade(rec.signal_id, outcome="LOSS")

    def test_unknown_signal_id_raises_key_error(self):
        logger = TradeLogger()
        with pytest.raises(KeyError):
            logger.resolve_trade("nonexistent", outcome="WIN", exit_price=2000.0)

    def test_notes_appended(self):
        logger, rec = self._logged(notes="initial")
        updated = logger.resolve_trade(
            rec.signal_id, outcome="CANCELLED", notes="added later"
        )
        assert "initial" in updated.notes
        assert "added later" in updated.notes

    def test_notes_empty_original_no_separator(self):
        logger, rec = self._logged()
        updated = logger.resolve_trade(
            rec.signal_id, outcome="CANCELLED", notes="only note"
        )
        assert updated.notes == "only note"


# ---------------------------------------------------------------------------
# 4. TestPnLCalculation
# ---------------------------------------------------------------------------


class TestPnLCalculation:
    def _resolve(self, bias, entry, stop_loss, exit_price, outcome="WIN"):
        logger = TradeLogger()
        rec = logger.log_signal(
            _signal(bias=bias, entry=entry, stop_loss=stop_loss),
            timestamp_utc=_FIXED_TS,
        )
        return logger.resolve_trade(
            rec.signal_id, outcome=outcome, exit_price=exit_price
        )

    def test_buy_win_pnl_r(self):
        # entry=2000, stop=1990 → risk=10; exit=2030 → pnl=(2030-2000)/10 = 3.0
        updated = self._resolve("BUY", 2000.0, 1990.0, 2030.0)
        assert updated.pnl_r == pytest.approx(3.0)

    def test_buy_loss_pnl_r(self):
        updated = self._resolve("BUY", 2000.0, 1990.0, 1990.0, outcome="LOSS")
        assert updated.pnl_r == pytest.approx(-1.0)

    def test_sell_win_pnl_r(self):
        # entry=2000, stop=2010 → risk=10; exit=1970 → pnl=(2000-1970)/10 = 3.0
        updated = self._resolve("SELL", 2000.0, 2010.0, 1970.0)
        assert updated.pnl_r == pytest.approx(3.0)

    def test_sell_loss_pnl_r(self):
        updated = self._resolve("SELL", 2000.0, 2010.0, 2010.0, outcome="LOSS")
        assert updated.pnl_r == pytest.approx(-1.0)

    def test_zero_risk_guard(self):
        # entry == stop_loss → risk = 0 → pnl_r must be None (no ZeroDivisionError)
        logger = TradeLogger()
        rec = logger.log_signal(
            _signal(entry=2000.0, stop_loss=2000.0),
            timestamp_utc=_FIXED_TS,
        )
        updated = logger.resolve_trade(
            rec.signal_id, outcome="WIN", exit_price=2020.0
        )
        assert updated.pnl_r is None


# ---------------------------------------------------------------------------
# 5. TestQueries
# ---------------------------------------------------------------------------


class TestQueries:
    def _populated_logger(self):
        logger = TradeLogger()
        r1 = logger.log_signal(_signal(), timestamp_utc="2026-01-05T10:00:00+00:00")
        r2 = logger.log_signal(_signal(), timestamp_utc="2026-01-05T11:00:00+00:00")
        logger.resolve_trade(r1.signal_id, outcome="WIN", exit_price=2020.0)
        return logger, r1, r2

    def test_all_records_returns_all(self):
        logger, r1, r2 = self._populated_logger()
        assert len(logger.all_records()) == 2

    def test_resolved_records_only_resolved(self):
        logger, _, _ = self._populated_logger()
        resolved = logger.resolved_records()
        assert len(resolved) == 1
        assert resolved[0].outcome == OUTCOME_WIN

    def test_pending_records_only_pending(self):
        logger, r1, r2 = self._populated_logger()
        pending = logger.pending_records()
        assert len(pending) == 1
        assert pending[0].outcome == OUTCOME_PENDING

    def test_get_record_found(self):
        logger, r1, _ = self._populated_logger()
        rec = logger.get_record(r1.signal_id)
        assert rec.signal_id == r1.signal_id

    def test_get_record_key_error_on_missing(self):
        logger = TradeLogger()
        with pytest.raises(KeyError):
            logger.get_record("nonexistent_id")

    def test_all_records_returns_copy(self):
        logger, _, _ = self._populated_logger()
        copy = logger.all_records()
        copy.clear()
        assert len(logger.all_records()) == 2


# ---------------------------------------------------------------------------
# 6. TestPersistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_then_load_roundtrip(self):
        logger = TradeLogger()
        rec = logger.log_signal(_signal(), timestamp_utc=_FIXED_TS)
        logger.resolve_trade(rec.signal_id, outcome="WIN", exit_price=2020.0)

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name

        try:
            logger.save(path)
            logger2 = TradeLogger()
            logger2.load(path)
            assert len(logger2.all_records()) == 1
            loaded = logger2.all_records()[0]
            assert loaded.outcome == OUTCOME_WIN
            assert loaded.pnl_r  == pytest.approx(2.0)
        finally:
            os.unlink(path)

    def test_load_invalid_json_raises(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write("not valid json {{")
            path = f.name
        try:
            with pytest.raises(Exception):
                TradeLogger().load(path)
        finally:
            os.unlink(path)

    def test_load_non_array_raises(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as f:
            json.dump({"key": "value"}, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="expected a JSON array"):
                TradeLogger().load(path)
        finally:
            os.unlink(path)

    def test_load_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            TradeLogger().load("/nonexistent/path/trades.json")

    def test_clear_empties_log(self):
        logger = TradeLogger()
        logger.log_signal(_signal(), timestamp_utc=_FIXED_TS)
        logger.clear()
        assert logger.all_records() == []

    def test_load_replaces_existing_records(self):
        logger = TradeLogger()
        logger.log_signal(_signal(), timestamp_utc="2026-01-01T00:00:00+00:00")
        rec = logger.log_signal(_signal(), timestamp_utc=_FIXED_TS)
        logger.resolve_trade(rec.signal_id, outcome="WIN", exit_price=2020.0)

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            # Save the 2-record logger to file
            logger.save(path)
            # Load into a fresh logger that already has 1 record
            logger2 = TradeLogger()
            logger2.log_signal(_signal(), timestamp_utc="2026-02-01T00:00:00+00:00")
            logger2.load(path)
            # After load, only the 2 saved records should exist
            assert len(logger2.all_records()) == 2
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 7. TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_signal_same_record_fields(self):
        logger1 = TradeLogger()
        r1 = logger1.log_signal(_signal(), timestamp_utc=_FIXED_TS)

        logger2 = TradeLogger()
        r2 = logger2.log_signal(_signal(), timestamp_utc=_FIXED_TS)

        assert r1 == r2

    def test_to_dict_from_dict_idempotent(self):
        logger = TradeLogger()
        rec = logger.log_signal(_signal(), timestamp_utc=_FIXED_TS)
        d  = rec.to_dict()
        r2 = TradeRecord.from_dict(d)
        assert r2.to_dict() == d
