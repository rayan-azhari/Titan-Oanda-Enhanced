"""nautilus_oanda.data
-------------------

Data client for OANDA (streaming).
"""

import asyncio
from decimal import Decimal
from typing import Optional

import oandapyV20
import oandapyV20.endpoints.pricing as pricing
from nautilus_trader.live.data_client import LiveDataClient
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity

from .config import OandaDataClientConfig
from .parsing import parse_datetime, parse_instrument_id


class OandaDataClient(LiveDataClient):
    """Streams live market data from OANDA."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        client_id,
        venue,
        config: OandaDataClientConfig,
        msgbus,
        cache,
        clock,
    ):
        super().__init__(
            loop=loop,
            client_id=client_id,
            venue=venue,
            config=config,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )
        self._config = config
        self._api = oandapyV20.API(access_token=config.access_token, environment=config.environment)
        self._account_id = config.account_id
        self._stream_task: Optional[asyncio.Task] = None
        self._subscribed_instruments = set()
        self._reconnect_count: int = 0

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
        """Fetch quotes from OANDA stream with exponential backoff reconnection.

        On disconnect or error the client will retry up to
        ``reconnect_attempts`` times (from config) with exponential backoff
        delay capped at 60 seconds.  The retry counter resets whenever
        a successful connection is established (i.e. at least one quote
        arrives), so transient network blips don't exhaust the budget.
        """
        if not self._subscribed_instruments:
            return

        max_attempts = self._config.reconnect_attempts
        base_delay = self._config.reconnect_delay

        while self._connected:
            # Convert InstrumentIds back to OANDA format (EUR_USD)
            instruments = [
                inst.symbol.value.replace("/", "_") for inst in self._subscribed_instruments
            ]
            params = {"instruments": ",".join(instruments)}

            try:
                r = pricing.PricingStream(accountID=self._account_id, params=params)
                self._log.info(
                    f"OANDA stream connecting for {len(instruments)} "
                    f"instrument(s)... (attempt {self._reconnect_count + 1})"
                )

                # The OANDA v20 client's stream request is a blocking
                # generator.  Run in executor to avoid blocking the loop.
                await self._loop.run_in_executor(None, self._consume_stream, r)

                # If _consume_stream returns cleanly, the stream ended.
                # Reset counter (it was connected at some point).
                self._reconnect_count = 0

            except asyncio.CancelledError:
                self._log.info("OANDA stream cancelled.")
                return  # Clean shutdown — do not reconnect

            except Exception as e:
                self._log.error(f"OANDA stream error: {e}")

            # --- Reconnect with exponential backoff ---
            self._reconnect_count += 1
            if self._reconnect_count > max_attempts:
                self._log.error(
                    f"OANDA stream: exceeded {max_attempts} reconnect attempts. Giving up."
                )
                return

            delay = min(base_delay * (2 ** (self._reconnect_count - 1)), 60.0)
            self._log.warning(
                f"OANDA stream reconnecting in {delay:.1f}s "
                f"(attempt {self._reconnect_count}/{max_attempts})..."
            )
            await asyncio.sleep(delay)

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
        # data example: {'type': 'PRICE', 'bids': [{'price': '1.05', ...}], ...}

        instrument_id = parse_instrument_id(data["instrument"])
        timestamp = parse_datetime(data["time"])

        # Bid/Ask Mapping:
        # We use the first level of depth (index 0) which represents the best available price.
        # Liquidity is cast to standard integer units.
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
            ts_event=timestamp.value,  # Nanoseconds (uint64)
            ts_init=self._clock.timestamp_ns(),
        )

        # Thread Safety:
        # This method runs in the executor thread. We must properly schedule
        # the data handling on the main asyncio event loop using `call_soon_threadsafe`
        # to ensure thread-safety within the Nautilus core.
        self._loop.call_soon_threadsafe(self.handle_data, tick)

    def handle_data(self, data: QuoteTick):
        """Process incoming data (run on loop)."""
        self._msgbus.publish_data(data)
