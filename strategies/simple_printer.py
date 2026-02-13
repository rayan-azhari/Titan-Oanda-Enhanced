"""
simple_printer.py
-----------------

A minimal NautilusTrader strategy that subscribes to all available instruments
and logs incoming QuoteTicks. Used for verifying connectivity and data flow.
"""

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.trading.strategy import Strategy


class SimplePrinterConfig(StrategyConfig):
    """Configuration for SimplePrinter."""
    pass


class SimplePrinter(Strategy):
    """Listens to and logs market data events."""

    def __init__(self, config: SimplePrinterConfig):
        super().__init__(config)
        self._count = 0

    def on_start(self):
        """Called when the strategy starts."""
        self.log.info("SimplePrinter started. Subscribing to all instruments...")
        
        # Subscribe to quotes for all loaded instruments
        for instrument in self.cache.instruments():
            self.subscribe_quotes(instrument.id)
            self.log.info(f"Subscribed to {instrument.id}")

    def on_quote_tick(self, tick: QuoteTick):
        """Called when a new quote tick is received."""
        self._count += 1
        # Log every 10th tick to avoid unexpected console flooding, 
        # but for verification we want to see at least the first few immediately.
        if self._count <= 5 or self._count % 10 == 0:
            self.log.info(f"[{self._count}] QUOTE {tick.instrument_id}: {tick.bid_price} / {tick.ask_price}")

    def on_stop(self):
        """Called when the strategy stops."""
        self.log.info("SimplePrinter stopped.")
