"""Exponential Moving Average (EMA) indicator."""


class EMA:
    """Exponential Moving Average calculation."""
    
    def __init__(self, period: int):
        self.period = period
