# XAUUSD Trading Strategy

This strategy combines institutional trading concepts with technical confirmations.

The system uses:

- market structure
- liquidity sweeps
- order blocks
- fair value gaps
- EMA trend confirmation
- RSI momentum confirmation

---

# Multi-Timeframe Model

Bias → D1 / H4 / H1
Levels → D1 / H4 / H1 / M15
Entry → M5
Precision → M1

---

# Buy Setup Conditions

A BUY signal may be generated when ALL conditions are met.

1. Higher timeframe bullish bias
2. Liquidity sweep below equal lows
3. Bullish CHOCH or BOS occurs
4. Price enters bullish order block or fair value gap
5. EMA trend alignment confirmed
6. RSI momentum bullish
7. Signal score ≥ 70

Entry:

Midpoint of order block or fair value gap.

Stop Loss:

Below liquidity sweep.

Take Profit:

Next liquidity pool or equal highs.

---

# Sell Setup Conditions

A SELL signal may be generated when ALL conditions are met.

1. Higher timeframe bearish bias
2. Liquidity sweep above equal highs
3. Bearish CHOCH or BOS occurs
4. Price enters bearish order block or fair value gap
5. EMA trend alignment confirmed
6. RSI momentum bearish
7. Signal score ≥ 70

Entry:

Midpoint of order block or fair value gap.

Stop Loss:

Above liquidity sweep.

Take Profit:

Next liquidity pool or equal lows.

---

# Trade Validation Filters

Trades should be avoided during:

- low volatility periods
- major news events (CPI, NFP, FOMC)
- conflicting macro sentiment

---

# Strategy Objective

The goal is to generate high-probability XAUUSD trade setups based on:

- liquidity engineering
- institutional order flow
- structured risk management