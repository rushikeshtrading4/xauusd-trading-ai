# Copilot Context – XAUUSD AI Trading System

This repository builds an **AI-powered trading analysis engine focused exclusively on XAUUSD (Gold vs USD).**

The system does **not perform automated trading**.
It produces **structured trade analysis and trade setups** based on quantitative and institutional trading concepts.

---

# Core Analytical Concepts

The system analyzes the market using the following components:

• Market Structure (BOS, CHOCH, Higher High / Lower Low)
• Liquidity Zones (equal highs, equal lows, liquidity sweeps)
• Institutional Order Blocks
• Fair Value Gaps (FVG)
• Technical Indicators
• Macroeconomic Drivers

Indicators used:

* EMA (20 / 50 / 200)
* RSI
* ATR
* VWAP

Macro drivers considered:

* DXY (US Dollar Index)
* US Treasury Yields
* CPI
* NFP
* FOMC policy changes

---

# Architecture Philosophy

The project follows a **modular Python architecture**.

Modules are separated by responsibility:

data → market data ingestion
indicators → indicator calculations
analysis → market structure and liquidity detection
patterns → order blocks and fair value gaps
ai → signal scoring and probability model
risk → risk management logic
execution → trade setup generation

The system processes data in the following pipeline:

Market Data
→ Indicator Calculation
→ Market Structure Analysis
→ Liquidity Detection
→ Institutional Pattern Detection
→ Signal Scoring Engine
→ Risk Manager
→ Trade Setup Output

---

# Coding Guidelines

All code generated in this repository should follow these standards:

• Python 3.11+
• Use **type hints**
• Write clear **docstrings**
• Keep functions **modular and single-purpose**
• Avoid hardcoded values unless part of strategy rules
• Prefer readable logic over overly complex abstractions

Example:

```python
def detect_liquidity_sweep(candles: pd.DataFrame) -> bool:
    """Detects if a liquidity sweep occurred in recent candles."""
```

---

# Trading Model Rules

Trade setups must follow institutional trading logic:

Required conditions:

• liquidity sweep
• market structure shift (BOS or CHOCH)
• order block or FVG interaction
• EMA trend alignment
• RSI momentum confirmation

Trades must satisfy **minimum risk-reward ratio of 1:2**.

Signals must never be generated using fabricated or missing market data.

---

# Signal Output Structure

All generated trade setups must include:

1. Market Context
2. Market Structure
3. Liquidity Zones
4. Technical Confirmation
5. Fundamental Drivers
6. Entry
7. Stop Loss
8. Take Profit
9. Risk-Reward Ratio
10. Probability Estimate
11. Invalidation Scenario
