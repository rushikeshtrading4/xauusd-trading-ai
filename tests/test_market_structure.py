import unittest

from analysis.market_structure import MarketStructureAnalyzer


class TestMarketStructure(unittest.TestCase):
    def test_initialization(self):
        analyzer = MarketStructureAnalyzer()
        self.assertIsNotNone(analyzer)


if __name__ == "__main__":
    unittest.main()
