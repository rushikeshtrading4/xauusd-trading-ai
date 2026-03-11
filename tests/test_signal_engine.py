import unittest

from ai.signal_engine import SignalEngine


class TestSignalEngine(unittest.TestCase):
    def test_initialization(self):
        engine = SignalEngine()
        self.assertIsNotNone(engine)


if __name__ == "__main__":
    unittest.main()
