"""
nautilus_oanda.data
-------------------

Data client for OANDA (streaming).
"""

import asyncio
from decimal import Decimal
from typing import Optional

import oandapyV20
import oandapyV20.endpoints.pricing as pricing
from nautilus_trader.common.component import LiveDataClient
from nautilus_trader.common.component import TimeEvent
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity

from .config import OandaDataClientConfig
from .parsing import parse_datetime
from .parsing import parse_instrument_id


class OandaDataClient(LiveDataClient):
    """Streams live market data from OANDA."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        config: OandaDataClientConfig,
        msgbus,
        cache,
        clock,
    ):
        super().__init__(loop, config, msgbus, cache, clock)
        self._config = config
        self._api = oandapyV20.API(
            access_token=config.access_token,
            environment=config.environment
        )
        self._account_id = config.account_id
        self._stream_task: Optional[asyncio.Task] = None
        self._subscribed_instruments = set()

    async def connect(self):
        """Connect to OANDA stream."""
        if self._stream_task:
            return
        
        # Start the streaming loop
        self._stream_task = self._loop.create_task(self._stream_quotes())
        self._set_connected(True)

    async def disconnect(self):
        """Disconnect from OANDA stream."""
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None
        self._set_connected(False)

    async def subscribe(self, instrument_id: InstrumentId):
        """Subscribe to an instrument."""
        if instrument_id in self._subscribed_instruments:
            return
        
        self._subscribed_instruments.add(instrument_id)
        # Restart stream with new subscription list
        # OANDA requires all instruments in one request
        await self._restart_stream()

    async def unsubscribe(self, instrument_id: InstrumentId):
        """Unsubscribe from an instrument."""
        if instrument_id not in self._subscribed_instruments:
            return

        self._subscribed_instruments.discard(instrument_id)
        # Restart stream with new subscription list
        await self._restart_stream()

    async def _restart_stream(self):
        """Restart the stream with updated subscriptions."""
        if not self._connected:
            return
            
        # Cancel current stream
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        
        # Start new stream if we have subscriptions
        if self._subscribed_instruments:
            self._stream_task = self._loop.create_task(self._stream_quotes())

    async def _stream_quotes(self):
        """Fetch quotes from OANDA stream."""
        if not self._subscribed_instruments:
            return

        # Convert InstrumentIds back to OANDA format (EUR_USD)
        instruments = [
            inst.symbol.value.replace("/", "_") 
            for inst in self._subscribed_instruments
        ]
        params = {"instruments": ",".join(instruments)}
        
        # OANDA streaming is blocking, so run in executor
        # CAUTION: This means we consume the stream in a separate thread
        # and push events back to the loop.
        
        try:
            r = pricing.PricingStream(accountID=self._account_id, params=params)
            
            def stream_generator():
                return self._api.request(r)

            # Iterating the generator blocks, so we do it in an executor?
            # Actually, requests.iter_lines() is blocking.
            # We must run the entire consumption loop in the executor.
            
            await self._loop.run_in_executor(None, self._consume_stream, r)

        except asyncio.CancelledError:
            self._log.info("OANDA stream cancelled.")
        except Exception as e:
            self._log.error(f"OANDA stream error: {e}")
            # Reconnect logic would go here
            await asyncio.sleep(self._config.reconnect_delay)
            # await self._restart_stream() # Risk of recursion loop?

    def _consume_stream(self, request):
        """Blocking loop to consume OANDA stream."""
        # This runs in a thread
        try:
            for line in self._api.request(request):
                if "type" in line and line["type"] == "PRICE":
                    self._parse_quote(line)
                # Ensure we yield to check for cancellation? 
                # Thread cancellation is hard. We rely on the socket closing or specific check.
                # Oandapy might not support clean exit from iterator easily.
                
        except Exception as e:
            self._log.error(f"Stream consumption failed: {e}")

    def _parse_quote(self, data):
        """Parse JSON quote and publish QuoteTick."""
        # data example:
        # {'type': 'PRICE', 'time': '2023-10-27T...', 'bids': [{'price': '1.05', 'liquidity': 1000000}], ...}
        
        instrument_id = parse_instrument_id(data["instrument"])
        timestamp = parse_datetime(data["time"])
        
        # OANDA sends multiple levels of depth, we take the top (0)
        bid = Decimal(data["bids"][0]["price"])
        ask = Decimal(data["asks"][0]["price"])
        bid_size = int(data["bids"][0]["liquidity"])
        ask_size = int(data["asks"][0]["liquidity"])
        
        tick = QuoteTick(
            instrument_id=instrument_id,
            bid_price=Price(bid, precision=None),
            ask_price=Price(ask, precision=None),
            bid_size=Quantity(bid_size, precision=0),
            ask_size=Quantity(ask_size, precision=0),
            ts_event=timestamp.value, # Nanoseconds (uint64)
            ts_init=self._clock.timestamp_ns(),
        )
        
        # Push to message bus (thread-safe?)
        # msgbus.publish_data broken? 
        # Nautilus objects methods are usually not thread-safe if they touch shared state.
        # But handle_data just pushes to a queue usually.
        # Ideally we schedule the handle_data call on the loop.
        
        self._loop.call_soon_threadsafe(self.handle_data, tick)

    def handle_data(self, data: QuoteTick):
        """Process incoming data (run on loop)."""
        self._msgbus.publish_data(data)
