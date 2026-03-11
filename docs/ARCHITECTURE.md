# System Architecture

The XAUUSD AI trading system is modular and designed for institutional-grade analysis.

The architecture separates market data, analysis, signal generation, and risk management.

---

## Core Modules

data/
Handles market and macroeconomic data ingestion.

analysis/
Market structure detection, swing points, liquidity zones,
order blocks and fair value gaps.

indicators/
Technical indicator calculations (EMA, RSI, ATR, VWAP).

ai/
Signal generation logic including bias models,
probability estimation and signal scoring.

risk/
Risk management including position sizing and
risk-reward calculations.

execution/
Trade setup generation and signal formatting.

config/
Centralized configuration and strategy parameters.

tests/
Unit tests validating indicators, structure detection
and signal generation.

ui/
Visualization dashboards and charting utilities.

---

# Data Flow

Market Data  
→ Indicator Calculation  
→ Market Structure Analysis  
→ Liquidity Detection  
→ Institutional Pattern Detection  
→ AI Signal Engine  
→ Risk Manager  
→ Trade Setup Output  

---

# Market Data Sources

The system ingests the following data:

- XAUUSD OHLC candles
- Multiple timeframes (M1, M5, M15, H1, H4)
- Volume data (if available)

Macro data:

- DXY (US Dollar Index)
- US10Y Treasury Yield
- CPI releases
- NFP releases
- FOMC decisions

---

# Continuous Processing Loop

The system runs continuously:

1. Fetch latest market data
2. Update indicators
3. Detect market structure
4. Detect liquidity events
5. Detect order blocks and fair value gaps
6. Score potential trade setups
7. Apply risk management rules
8. Generate trade signals