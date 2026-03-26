"""
Trade logging module for XAUUSD.

Records every validated trade signal and its eventual outcome to an
in-memory log. Optionally persists to a JSON file. Pure Python, no
external dependencies. All logic is deterministic and side-effect-free
except for the intentional file I/O in save() and load().

Why in-memory first, file second
─────────────────────────────────
The in-memory log is the primary data structure. File I/O is explicit
and called by the consumer — the logger never writes to disk
automatically. This makes the logger safe to use in backtesting (where
file writes would be wasteful) and in live trading (where the consumer
decides when to persist).

TradeRecord fields
──────────────────
signal_id       str   — unique ID: "{pair}_{timeframe}_{timestamp_utc}"
timestamp_utc   str   — ISO-8601 UTC string at signal generation time
pair            str   — always "XAUUSD"
timeframe       str   — e.g. "M5"
bias            str   — "BUY" or "SELL"
entry           float — planned entry price (OB midpoint)
stop_loss       float
take_profit     float
risk_reward     float
confidence      float — 0–100
invalidation    float
atr             float
mtf_bias        str   — "BULLISH" / "BEARISH" / "NEUTRAL"
bias_strength   str   — "STRONG" / "WEAK"
context         str   — "TREND" / "CONSOLIDATION" / "REVERSAL"
outcome         str   — "WIN" / "LOSS" / "PENDING" / "CANCELLED"
exit_price      float | None — actual exit price; None until resolved
pnl_r           float | None — P&L in R-multiples; None until resolved
                               (e.g. 2.0 means won 2R, -1.0 means lost 1R)
notes           str   — free-text field; empty string by default
"""

from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTCOME_WIN       = "WIN"
OUTCOME_LOSS      = "LOSS"
OUTCOME_PENDING   = "PENDING"
OUTCOME_CANCELLED = "CANCELLED"

_VALID_OUTCOMES = frozenset({OUTCOME_WIN, OUTCOME_LOSS,
                              OUTCOME_PENDING, OUTCOME_CANCELLED})
_VALID_BIASES   = frozenset({"BUY", "SELL"})


# ---------------------------------------------------------------------------
# TradeRecord
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """Immutable-by-convention record for a single trade."""

    signal_id:     str
    timestamp_utc: str
    pair:          str
    timeframe:     str
    bias:          str
    entry:         float
    stop_loss:     float
    take_profit:   float
    risk_reward:   float
    confidence:    float
    invalidation:  float
    atr:           float
    mtf_bias:      str
    bias_strength: str
    context:       str
    outcome:       str              = OUTCOME_PENDING
    exit_price:    float | None     = None
    pnl_r:         float | None     = None
    notes:         str              = ""

    def is_resolved(self) -> bool:
        """Return True when the trade has a definitive outcome."""
        return self.outcome in (OUTCOME_WIN, OUTCOME_LOSS, OUTCOME_CANCELLED)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation (JSON-serialisable)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradeRecord":
        """Reconstruct a TradeRecord from a plain dict (e.g. loaded from JSON)."""
        return cls(**data)


# ---------------------------------------------------------------------------
# TradeLogger
# ---------------------------------------------------------------------------


class TradeLogger:
    """In-memory trade log with optional JSON persistence.

    Usage
    -----
    logger = TradeLogger()
    record = logger.log_signal(signal_dict)
    logger.resolve_trade(record.signal_id, outcome="WIN", exit_price=2055.0)
    logger.save("trades.json")
    """

    def __init__(self) -> None:
        self._records: list[TradeRecord] = []

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_signal(
        self,
        signal: dict,
        *,
        timestamp_utc: str | None = None,
        notes: str = "",
    ) -> TradeRecord:
        """Create a TradeRecord from a validated signal dict and append it.

        Parameters
        ----------
        signal : dict
            Output from ai/signal_engine or execution/signal_engine.
            Must contain: pair, timeframe, bias, entry, stop_loss,
            take_profit, risk_reward, confidence, invalidation, atr,
            mtf_bias, bias_strength, context.
        timestamp_utc : str | None
            ISO-8601 UTC string. If None, uses the current UTC time.
        notes : str
            Optional free-text annotation.

        Returns
        -------
        TradeRecord
            The newly created record (also appended to internal log).

        Raises
        ------
        ValueError
            If any required key is missing from signal, or if bias is
            not "BUY" or "SELL".
        """
        required = {
            "pair", "timeframe", "bias", "entry", "stop_loss",
            "take_profit", "risk_reward", "confidence", "invalidation",
            "atr", "mtf_bias", "bias_strength", "context",
        }
        missing = required - set(signal.keys())
        if missing:
            raise ValueError(
                f"log_signal(): missing required key(s): {sorted(missing)}"
            )

        bias = str(signal["bias"]).upper()
        if bias not in _VALID_BIASES:
            raise ValueError(
                f"log_signal(): invalid bias {signal['bias']!r}. "
                f"Expected 'BUY' or 'SELL'."
            )

        ts = timestamp_utc or _utc_now_iso()
        pair = str(signal["pair"])
        tf   = str(signal["timeframe"])
        signal_id = f"{pair}_{tf}_{ts}"

        record = TradeRecord(
            signal_id     = signal_id,
            timestamp_utc = ts,
            pair          = pair,
            timeframe     = tf,
            bias          = bias,
            entry         = float(signal["entry"]),
            stop_loss     = float(signal["stop_loss"]),
            take_profit   = float(signal["take_profit"]),
            risk_reward   = float(signal["risk_reward"]),
            confidence    = float(signal["confidence"]),
            invalidation  = float(signal["invalidation"]),
            atr           = float(signal["atr"]),
            mtf_bias      = str(signal["mtf_bias"]),
            bias_strength = str(signal["bias_strength"]),
            context       = str(signal["context"]),
            outcome       = OUTCOME_PENDING,
            exit_price    = None,
            pnl_r         = None,
            notes         = notes,
        )
        self._records.append(record)
        return record

    def resolve_trade(
        self,
        signal_id: str,
        *,
        outcome: str,
        exit_price: float | None = None,
        notes: str = "",
    ) -> TradeRecord:
        """Update an existing PENDING trade record with its outcome.

        Parameters
        ----------
        signal_id : str
            The signal_id of the record to update.
        outcome : str
            One of "WIN", "LOSS", "CANCELLED".
        exit_price : float | None
            Actual exit price. Required for WIN and LOSS; optional for CANCELLED.
        notes : str
            Appended to the existing notes field (separated by "; " if non-empty).

        Returns
        -------
        TradeRecord
            The updated record (replaced in the internal list).

        Raises
        ------
        KeyError
            If no record with signal_id exists.
        ValueError
            If outcome is invalid, trade is already resolved, or
            exit_price is required but not supplied.
        """
        outcome_upper = str(outcome).upper()
        if outcome_upper not in _VALID_OUTCOMES - {OUTCOME_PENDING}:
            raise ValueError(
                f"resolve_trade(): invalid outcome {outcome!r}. "
                f"Expected 'WIN', 'LOSS', or 'CANCELLED'."
            )

        idx, record = self._find(signal_id)
        if record.is_resolved():
            raise ValueError(
                f"resolve_trade(): trade {signal_id!r} is already resolved "
                f"with outcome {record.outcome!r}."
            )

        if outcome_upper in (OUTCOME_WIN, OUTCOME_LOSS) and exit_price is None:
            raise ValueError(
                f"resolve_trade(): exit_price is required for outcome "
                f"{outcome_upper!r}."
            )

        # Compute pnl_r
        pnl_r: float | None = None
        if exit_price is not None and outcome_upper != OUTCOME_CANCELLED:
            risk = abs(record.entry - record.stop_loss)
            if risk > 0:
                if record.bias == "BUY":
                    pnl_r = (exit_price - record.entry) / risk
                else:  # SELL
                    pnl_r = (record.entry - exit_price) / risk

        # Merge notes
        merged_notes = record.notes
        if notes:
            merged_notes = f"{merged_notes}; {notes}" if merged_notes else notes

        updated = TradeRecord(
            signal_id     = record.signal_id,
            timestamp_utc = record.timestamp_utc,
            pair          = record.pair,
            timeframe     = record.timeframe,
            bias          = record.bias,
            entry         = record.entry,
            stop_loss     = record.stop_loss,
            take_profit   = record.take_profit,
            risk_reward   = record.risk_reward,
            confidence    = record.confidence,
            invalidation  = record.invalidation,
            atr           = record.atr,
            mtf_bias      = record.mtf_bias,
            bias_strength = record.bias_strength,
            context       = record.context,
            outcome       = outcome_upper,
            exit_price    = float(exit_price) if exit_price is not None else None,
            pnl_r         = pnl_r,
            notes         = merged_notes,
        )
        self._records[idx] = updated
        return updated

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all_records(self) -> list[TradeRecord]:
        """Return a copy of all records (pending and resolved)."""
        return list(self._records)

    def resolved_records(self) -> list[TradeRecord]:
        """Return only WIN / LOSS / CANCELLED records."""
        return [r for r in self._records if r.is_resolved()]

    def pending_records(self) -> list[TradeRecord]:
        """Return only PENDING records."""
        return [r for r in self._records if r.outcome == OUTCOME_PENDING]

    def get_record(self, signal_id: str) -> TradeRecord:
        """Return the record with the given signal_id.

        Raises
        ------
        KeyError
            If no such record exists.
        """
        _, record = self._find(signal_id)
        return record

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Persist all records to a JSON file.

        Parameters
        ----------
        filepath : str
            Destination file path. Created or overwritten.
        """
        data = [r.to_dict() for r in self._records]
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def load(self, filepath: str) -> None:
        """Load records from a JSON file, replacing the current in-memory log.

        Parameters
        ----------
        filepath : str
            Source file path. Must contain a JSON array of TradeRecord dicts.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file contains malformed data.
        """
        with open(filepath, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, list):
            raise ValueError(
                f"load(): expected a JSON array, got {type(raw).__name__}"
            )
        try:
            self._records = [TradeRecord.from_dict(item) for item in raw]
        except (TypeError, KeyError) as exc:
            raise ValueError(f"load(): malformed trade record — {exc}") from exc

    def clear(self) -> None:
        """Remove all records from the in-memory log."""
        self._records = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find(self, signal_id: str) -> tuple[int, TradeRecord]:
        """Return (index, record) for the given signal_id.

        Raises
        ------
        KeyError
            If not found.
        """
        for i, r in enumerate(self._records):
            if r.signal_id == signal_id:
                return i, r
        raise KeyError(f"No trade record with signal_id={signal_id!r}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
