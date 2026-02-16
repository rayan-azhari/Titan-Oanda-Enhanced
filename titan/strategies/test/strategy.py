
from decimal import Decimal
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

class TestTradeConfig(StrategyConfig):
    instrument_id: str
    bar_type: str
    trade_size: int = 1000

class TestTradeStrategy(Strategy):
    def __init__(self, config: TestTradeConfig):
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_count = 0
        self.entry_submitted = False
        self.exit_submitted = False

    def on_start(self):
        self.subscribe_bars(BarType.from_str(self.config.bar_type))
        self.log.info(f"STARTED TestTradeStrategy for {self.instrument_id}")

    def on_bar(self, bar: Bar):
        self.bar_count += 1
        self.log.info(f"BAR {self.bar_count}: {bar}")
        
        # We use bar counts as a proxy for time to keep it simple
        # Assuming 5-second bars:
        # 2 minutes = 24 bars
        
        if self.bar_count >= 5 and not self.entry_submitted:
            self.log.info("⏳ Triggering ENTRY (Buy Market)...")
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(self.config.trade_size),
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)
            self.entry_submitted = True
        
        if self.bar_count >= 10 and self.entry_submitted and not self.exit_submitted:
             self.log.info("⏳ Triggering EXIT (Close Position)...")
             self.close_all_positions(self.instrument_id)
             self.exit_submitted = True
             
        if self.exit_submitted and self.bar_count >= 20:
            self.log.info("✅ Test Complete. Stopping.")
            self.stop()

    def on_order_filled(self, event):
        self.log.info(f"✨ ORDER FILLED: {event.client_order_id} {event.order_side} {event.last_qty} @ {event.last_px}")

    def on_order_rejected(self, event):
        self.log.error(f"❌ ORDER REJECTED: {event.client_order_id} - {event.reason}")
    
    def on_trade_tick(self, tick):
        self.log.info(f"TICK: {tick}")
