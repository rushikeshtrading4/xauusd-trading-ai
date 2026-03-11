"""
XAUUSD AI Trading System
Entry point — orchestrates the full data → analysis → signal → execution pipeline.
"""

import logging
import time
import sys

from config.settings import *

from data.market_data import MarketDataHandler
from data.macro_data import MacroDataHandler
from data.websocket_feed import WebSocketFeed

from indicators.indicator_engine import IndicatorEngine

from analysis.market_structure import MarketStructureAnalyzer
from analysis.swing_points import SwingPointDetector
from analysis.liquidity import LiquidityAnalyzer
from analysis.order_blocks import OrderBlockDetector
from analysis.fair_value_gaps import FairValueGapDetector

from ai.signal_engine import SignalEngine
from ai.bias_model import BiasModel
from ai.probability_model import ProbabilityModel

from risk.risk_manager import RiskManager
from risk.position_sizing import PositionSizer
from risk.rr_calculator import RRCalculator

from execution.trade_setup import TradeSetup
from execution.signal_formatter import SignalFormatter

from ui.dashboard import Dashboard


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/engine.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

data_logger = logging.getLogger("data_ingestion")
data_handler = logging.FileHandler("logs/data_ingestion.log")
data_logger.addHandler(data_handler)

error_logger = logging.getLogger("errors")
error_handler = logging.FileHandler("logs/errors.log")
error_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)

logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Main trading loop
# ---------------------------------------------------------------------------

def build_system():
    """Instantiate and wire together all system components."""
    market_data    = MarketDataHandler()
    macro_data     = MacroDataHandler()
    ws_feed        = WebSocketFeed()

    indicators     = IndicatorEngine()

    structure      = MarketStructureAnalyzer()
    swings         = SwingPointDetector()
    liquidity      = LiquidityAnalyzer()
    order_blocks   = OrderBlockDetector()
    fvg            = FairValueGapDetector()

    bias_model     = BiasModel()
    prob_model     = ProbabilityModel()
    signal_engine  = SignalEngine()

    risk_manager   = RiskManager()
    position_sizer = PositionSizer()
    rr_calc        = RRCalculator()

    trade_setup    = TradeSetup()
    formatter      = SignalFormatter()

    dashboard      = Dashboard()

    return {
        "market_data":    market_data,
        "macro_data":     macro_data,
        "ws_feed":        ws_feed,
        "indicators":     indicators,
        "structure":      structure,
        "swings":         swings,
        "liquidity":      liquidity,
        "order_blocks":   order_blocks,
        "fvg":            fvg,
        "bias_model":     bias_model,
        "prob_model":     prob_model,
        "signal_engine":  signal_engine,
        "risk_manager":   risk_manager,
        "position_sizer": position_sizer,
        "rr_calc":        rr_calc,
        "trade_setup":    trade_setup,
        "formatter":      formatter,
        "dashboard":      dashboard,
    }


def run_cycle(components: dict):
    """
    Execute one full analysis cycle following the data-flow pipeline:

    Market Data → Indicators → Market Structure → Liquidity →
    Order Blocks / FVG → AI Signal Engine → Risk Manager → Trade Setup Output
    """
    logger.info("─── Starting analysis cycle ───")

    # 1. Fetch latest market data
    data_logger.info("Fetching market data (XAUUSD multi-timeframe)")
    market_data = components["market_data"]
    macro_data  = components["macro_data"]

    # 2. Update indicators
    logger.info("Calculating technical indicators")
    components["indicators"]

    # 3. Detect market structure
    logger.info("Analysing market structure")
    components["structure"]
    components["swings"]

    # 4. Detect liquidity events
    logger.info("Detecting liquidity zones and sweeps")
    components["liquidity"]

    # 5. Detect order blocks and fair value gaps
    logger.info("Identifying order blocks and fair value gaps")
    components["order_blocks"]
    components["fvg"]

    # 6. Score potential trade setups
    logger.info("Running AI signal engine")
    components["bias_model"]
    components["prob_model"]
    components["signal_engine"]

    # 7. Apply risk management
    logger.info("Applying risk management rules")
    components["risk_manager"]
    components["position_sizer"]
    components["rr_calc"]

    # 8. Generate trade setup output
    logger.info("Generating trade setup")
    components["trade_setup"]
    components["formatter"]

    logger.info("─── Cycle complete ───")


def main():
    logger.info("XAUUSD AI Trading System — starting up")

    try:
        components = build_system()
        logger.info("All components initialised successfully")
    except Exception as exc:
        error_logger.error("Failed to initialise system components: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("Entering main trading loop (Ctrl+C to stop)")
    try:
        while True:
            try:
                run_cycle(components)
            except Exception as exc:
                error_logger.error("Error during analysis cycle: %s", exc, exc_info=True)

            time.sleep(60)  # Wait before next cycle

    except KeyboardInterrupt:
        logger.info("Shutdown requested — exiting cleanly")


if __name__ == "__main__":
    main()
