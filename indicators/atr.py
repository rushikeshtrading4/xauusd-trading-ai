"""Average True Range (ATR) indicator."""


class ATR:
    """Average True Range calculation."""
    
    def __init__(self, period: int = 14):
        self.period = period
