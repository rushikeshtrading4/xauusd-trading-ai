"""Relative Strength Index (RSI) indicator."""


class RSI:
    """Relative Strength Index calculation."""
    
    def __init__(self, period: int = 14):
        self.period = period
