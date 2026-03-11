import unittest

from indicators.ema import EMA
from indicators.rsi import RSI
from indicators.atr import ATR


class TestIndicators(unittest.TestCase):
    def test_ema(self):
        ema = EMA(period=10)
        self.assertEqual(ema.period, 10)

    def test_rsi(self):
        rsi = RSI()
        self.assertEqual(rsi.period, 14)

    def test_atr(self):
        atr = ATR()
        self.assertEqual(atr.period, 14)


if __name__ == "__main__":
    unittest.main()
