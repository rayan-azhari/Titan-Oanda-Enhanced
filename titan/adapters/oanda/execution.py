"""nautilus_oanda.execution
------------------------

Execution client for OANDA (orders & positions).
"""

import asyncio
import sys
import traceback
from decimal import Decimal
from typing import Optional

import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.transactions as transactions
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.execution.reports import (
    FillReport,
    OrderStatusReport,
    PositionStatusReport,
)
from nautilus_trader.live.execution_client import LiveExecutionClient
from nautilus_trader.model.enums import (
    AccountType,
    LiquiditySide,
    OmsType,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
)
from nautilus_trader.model.events import AccountState, OrderCanceled, OrderFilled
from nautilus_trader.model.identifiers import (
    AccountId,
    ClientId,
    ClientOrderId,
    TradeId,
    Venue,
    VenueOrderId,
)
from nautilus_trader.model.objects import AccountBalance, Currency, Money, Price, Quantity
from nautilus_trader.model.orders import LimitOrder, Order

from .config import OandaExecutionClientConfig
from .parsing import parse_datetime, parse_instrument_id


class OandaExecutionClient(LiveExecutionClient):
    """Handles execution (orders) on OANDA."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        client_id: ClientId,
        venue: Venue,
        oms_type: OmsType,
        account_type: AccountType,
        base_currency: Currency | None,
        instrument_provider: InstrumentProvider,
        config: OandaExecutionClientConfig,
        msgbus,
        cache,
        clock,
    ):
        super().__init__(
            loop=loop,
            client_id=client_id,
            venue=venue,
            oms_type=oms_type,
            account_type=account_type,
            base_currency=base_currency,
            instrument_provider=instrument_provider,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            config=config,
        )
        self._config = config
        self._api = oandapyV20.API(access_token=config.access_token, environment=config.environment)
        self._account_id = config.account_id
        # Base ExecutionClient expects self.account_id to be set for reports
        # self.account_id = AccountId(f"OANDA-{config.account_id}") # Read-only
        self._stream_task: Optional[asyncio.Task] = None

    @property
    def account_id(self) -> AccountId:
        return AccountId(f"OANDA-{self._account_id}")

    async def _connect(self) -> None:
        """Connect to OANDA transaction stream and initialize account state."""
        try:
            self._log.info("Connecting to OANDA and initializing account state...")

            # 1. Fetch and send initial AccountState
            await self._update_account_state()

            # 2. Fetch and send initial Position Status (Reconciliation)
            await self._update_positions_state()

            # 3. Start streaming (Non-blocking: create background task)
            self._log.info("OANDA transaction stream starting...")
            self._stream_task = self._loop.create_task(self._stream_transactions())

        except Exception as e:
            self._log.error(f"Connection failed: {e}")
            raise

    # ... (other methods) ...

    async def _update_account_state(self):
        """Fetch account summary and emit AccountState event."""
        try:
            self._log.info("Requesting AccountSummary for AccountState...")
            r = accounts.AccountSummary(self._account_id)
            await self._loop.run_in_executor(None, lambda: self._api.request(r))

            data = r.response.get("account", {})

            # Extract account details
            currency_str = data.get("currency", "USD")
            base_currency = Currency.from_str(currency_str)
            balance_val = Decimal(data.get("balance", "0"))
            margin_used = Decimal(data.get("marginUsed", "0"))

            # OANDA's marginAvailable might slightly differ from (Balance - MarginUsed) due to PL
            # or other factors. Nautilus requires strict `total == free + locked`.
            # So we derive free from the other two authoritative values.
            # margin_available = Decimal(data.get("marginAvailable", "0"))
            free_calculated = balance_val - margin_used

            balance = AccountBalance(
                total=Money(balance_val, base_currency),
                free=Money(free_calculated, base_currency),
                locked=Money(margin_used, base_currency),
            )

            state = AccountState(
                account_id=AccountId(f"OANDA-{self._account_id}"),
                account_type=AccountType.MARGIN,
                base_currency=base_currency,
                reported=True,
                balances=[balance],
                margins=[],  # No specific margin objects for now
                info={},
                event_id=UUID4(),
                ts_event=self._clock.timestamp_ns(),
                ts_init=self._clock.timestamp_ns(),
            )

            self._send_account_state(state)
            self._log.info("AccountState sent successfully.")

        except Exception as e:
            self._log.error(f"Failed to update account state: {e}")
            sys.stderr.write(f"ERROR: Failed to update account state: {e}\n")
            traceback.print_exc()

    async def _update_positions_state(self):
        """Fetch and emit initial position state."""
        try:
            self._log.info("Reconciling initial positions...")
            # Reuse the generation logic
            reports = await self.generate_position_status_reports(None)
            for report in reports:
                self._send_position_status_report(report)
            self._log.info(f"Reconciled {len(reports)} positions.")
        except Exception as e:
            self._log.error(f"Initial position reconciliation failed: {e}")
            # Don't fail connection just because of this, but log it

    def disconnect(self):
        """Disconnect from OANDA stream."""
        if self._stream_task:
            self._stream_task.cancel()
            try:
                # We can't await here because disconnect is synchronous in base class.
                # Just cancelling is usually enough for cleanup in asyncio.
                pass
            except asyncio.CancelledError:
                pass
            self._stream_task = None
        self._set_connected(False)

    async def _stream_transactions(self):
        """Stream transaction events (fills, cancels)."""
        try:
            with open("oanda_debug.log", "a") as f:
                f.write("DEBUG: _stream_transactions STARTING\n")
            r = transactions.TransactionsStream(accountID=self._account_id)
            await self._loop.run_in_executor(None, self._consume_stream, r)
        except Exception as e:
            with open("oanda_debug.log", "a") as f:
                f.write(f"DEBUG: _stream_transactions ERROR: {e}\n")
            self._log.error(f"Transaction stream error: {e}")

    def _consume_stream(self, request):
        """Blocking loop to consume transactions."""
        try:
            for line in self._api.request(request):
                # Log everything for debug
                if "type" in line:
                    # self._log.info(f"STREAM EVENT: {line['type']} - {line.get('id', '')}")
                    pass

                if line.get("type") == "HEARTBEAT":
                    continue

                self._log.info(f"RAW STREAM DATA: {line}")  # Temporary Debug

                if line.get("type") in (
                    "ORDER_FILL",
                    "ORDER_CANCEL",
                    "ORDER_CREATE",
                    "ORDER_CLIENT_EXTENSIONS_MODIFY",
                ):
                    self._loop.call_soon_threadsafe(self._handle_event, line)
        except Exception as e:
            self._log.error(f"Transaction stream failed: {e}")

    def _handle_event(self, data: dict):
        """Process OANDA transaction event.

        Maps OANDA transaction stream events to Nautilus order events:
        - ORDER_FILL  -> OrderFilled
        - ORDER_CANCEL -> OrderCanceled
        - ORDER_CREATE -> Logged (OrderAccepted already sent on submission)

        Args:
            data: Raw OANDA transaction JSON dict.
        """
        try:
            event_type = data.get("type", "")
            transaction_id = data.get("id", "unknown")

            self._log.info(f"Processing Event: {event_type} {transaction_id}")

            # Extract the Nautilus ClientOrderId
            # 1. Try clientExtensions (common for OrderSubmit/Cancel)
            client_ext = data.get("clientExtensions", {})
            client_order_id_str = client_ext.get("id")

            # 2. Try clientOrderID (common for OrderFill)
            if not client_order_id_str:
                client_order_id_str = data.get("clientOrderID")

            # 3. For OrderCancel, sometimes it's in 'orderID' related fields? Check Logic.

            if not client_order_id_str:
                self._log.warning(
                    f"Event {event_type} {transaction_id} missing clientExtensions/clientOrderID"
                )
                return

            try:
                client_order_id = ClientOrderId(client_order_id_str)
            except Exception as e:
                self._log.warning(
                    f"Failed to create ClientOrderId from '{client_order_id_str}': {e}"
                )
                return

            venue_order_id = VenueOrderId(str(data.get("orderID", transaction_id)))

            if event_type == "ORDER_FILL":
                self._handle_fill(data, client_order_id, venue_order_id)
            elif event_type == "ORDER_CANCEL":
                self._handle_cancel(data, client_order_id, venue_order_id)
            elif event_type == "ORDER_CREATE":
                self._log.info(
                    f"Order created on OANDA: {venue_order_id} (client: {client_order_id})"
                )

        except Exception as e:
            self._log.error(f"Error handling event: {e}")

    def _handle_fill(
        self, data: dict, client_order_id: ClientOrderId, venue_order_id: VenueOrderId
    ) -> None:
        """Map an OANDA ORDER_FILL to a Nautilus OrderFilled event."""
        instrument_id = parse_instrument_id(data.get("instrument", ""))
        fill_price = Decimal(data.get("price", "0"))
        units = abs(int(data.get("units", "0")))
        pl = Decimal(data.get("pl", "0"))
        commission = Decimal(data.get("commission", "0"))
        timestamp = parse_datetime(data.get("time", ""))

        # Determine the quote currency from the instrument (e.g. EUR_USD -> USD)
        parts = data.get("instrument", "_").split("_")
        quote_ccy = parts[1] if len(parts) == 2 else "USD"

        order = self._cache.order(client_order_id)
        if order is None:
            self._log.warning(
                f"Received fill for unknown order {client_order_id}. Venue ID: {venue_order_id}"
            )
            return

        fill = OrderFilled(
            trader_id=self.trader_id,
            strategy_id=order.strategy_id,
            instrument_id=instrument_id,
            client_order_id=client_order_id,
            venue_order_id=venue_order_id,
            account_id=AccountId(f"OANDA-{self._account_id}"),
            trade_id=TradeId(str(data.get("id", "0"))),
            position_id=None,  # OANDA doesn't provide position ID on fill
            order_side=order.side,
            order_type=order.order_type,
            last_qty=Quantity(units, precision=0),
            last_px=Price.from_str(str(fill_price)),  # Use factory method
            currency=Currency.from_str(quote_ccy),
            commission=Money(commission, Currency.from_str(quote_ccy)),
            liquidity_side=LiquiditySide.TAKER,  # Assumed TAKER for market orders
            event_id=UUID4(),
            ts_event=timestamp.value,
            ts_init=self._clock.timestamp_ns(),
        )

        self._msgbus.send(endpoint="ExecEngine.process", msg=fill)
        self._log.info(
            f"ORDER_FILL: {instrument_id} {order.side.name} {units} @ {fill_price} (PnL: {pl})"
        )

    def _handle_cancel(
        self, data: dict, client_order_id: ClientOrderId, venue_order_id: VenueOrderId
    ) -> None:
        """Map an OANDA ORDER_CANCEL to a Nautilus OrderCanceled event.

        Args:
            data: Raw OANDA cancel transaction dict.
            client_order_id: Mapped Nautilus client order ID.
            venue_order_id: OANDA-side order ID.
        """
        timestamp = parse_datetime(data.get("time", ""))
        reason = data.get("reason", "UNKNOWN")

        order = self._cache.order(client_order_id)
        if order is None:
            self._log.warning(f"Received cancel for unknown order {client_order_id}.")
            return

        canceled = OrderCanceled(
            trader_id=self.trader_id,
            strategy_id=order.strategy_id,
            instrument_id=order.instrument_id,
            client_order_id=client_order_id,
            venue_order_id=venue_order_id,
            account_id=AccountId(f"OANDA-{self._account_id}"),
            ts_event=timestamp.value,
            ts_init=self._clock.timestamp_ns(),
        )

        self._msgbus.send(endpoint="ExecEngine.process", msg=canceled)
        self._log.info(f"ORDER_CANCEL: {client_order_id} reason={reason}")

    def submit_order(self, command):
        """Submit an order to OANDA."""
        # Schedule the async submission on the loop
        # command is nautilus_trader.execution.messages.SubmitOrder
        self._loop.create_task(self._submit_order_async(command.order))

    async def _submit_order_async(self, order: Order):
        # 1. Map Nautilus Order to OANDA dict
        data = self._map_order(order)

        # 2. Send request
        r = orders.OrderCreate(self._account_id, data=data)

        try:
            # The OANDA API request is blocking. We offload it to the executor
            # to avoid blocking the main asyncio event loop, which would freeze
            # the entire trading node.
            response = await self._loop.run_in_executor(None, lambda: self._api.request(r))
            self._log.info(f"Order submitted: {response}")
            # Note: NautilusTrader core usually generates the OrderSubmitted event
            # upon successful return from this method.

        except Exception as e:
            self._log.error(f"Order submission failed: {e}")
            # Generate OrderRejected event

    def _map_order(self, order: Order) -> dict:
        """Map Nautilus order to OANDA API format."""
        # OANDA expects:
        # {
        #   "order": {
        #     "units": "100",
        #     "instrument": "EUR_USD",
        #     "timeInForce": "FOK",
        #     "type": "MARKET",
        #     "positionFill": "DEFAULT"
        #   }
        # }

        units = int(order.quantity)

        if order.side == OrderSide.SELL:
            units = -units

        oanda_symbol = order.instrument_id.symbol.value.replace("/", "_")

        order_type = "MARKET"
        price = None
        if isinstance(order, LimitOrder):
            order_type = "LIMIT"
            price = f"{order.price}"

        data = {
            "order": {
                "units": str(units),
                "instrument": oanda_symbol,
                "type": order_type,
                "clientExtensions": {
                    # We map the Nautilus ClientOrderID to OANDA's clientExtensions.id
                    # This allows us to reconcile execution reports back to the specific
                    # Nautilus order instance.
                    "id": str(order.client_order_id),
                    "tag": "nautilus",
                },
            }
        }

        if price:
            data["order"]["price"] = price
            data["order"]["timeInForce"] = "GTC"  # Limit orders usually GTC

        return data

    def cancel_order(self, command) -> None:
        """Cancel an order on OANDA."""
        self._loop.create_task(self._cancel_order_async(command))

    async def _cancel_order_async(self, command) -> None:
        """Cancel an order on OANDA (Async implementation).

        Uses the OANDA OrderCancel endpoint to cancel a pending order.
        The confirmation arrives asynchronously via the transaction stream
        and is handled by _handle_cancel().

        Args:
            command: Nautilus CancelOrder command with order details.
        """
        # Resolve the venue order ID (OANDA-side) from our cache
        order = self._cache.order(command.client_order_id)
        if order is None or order.venue_order_id is None:
            self._log.error(
                f"Cannot cancel — order not found or no venue ID: {command.client_order_id}"
            )
            return

        oanda_order_id = order.venue_order_id.value
        r = orders.OrderCancel(self._account_id, oanda_order_id)

        try:
            response = await self._loop.run_in_executor(None, lambda: self._api.request(r))
            self._log.info(f"Cancel request sent for {oanda_order_id}: {response}")
        except Exception as e:
            self._log.error(f"Cancel request failed for {oanda_order_id}: {e}")
        """Generate account status reports (Reconciliation)."""
        # Kept for compatibility if called, but relying on AccountState for registration.
        # Returning empty list to avoid import errors if AccountStatusReport is removed/missing.
        return []

    async def generate_order_status_reports(self, command) -> list[OrderStatusReport]:
        """Fetch open orders and generate status reports (Reconciliation)."""
        reports = []
        try:
            # Fetch all OPEN orders
            r = orders.OrderList(self._account_id, params={"state": "PENDING"})
            data = await self._loop.run_in_executor(None, lambda: self._api.request(r))

            for o_data in data.get("orders", []):
                # 1. Parsing
                venue_order_id = VenueOrderId(str(o_data.get("id")))
                instrument_id = parse_instrument_id(o_data.get("instrument", ""))

                # Check Client ID (Nautilus ID)
                client_ext = o_data.get("clientExtensions", {})
                client_id_str = client_ext.get("id")

                if not client_id_str:
                    # Order not from Nautilus (or manual). Skip for now.
                    continue

                client_order_id = ClientOrderId(client_id_str)
                timestamp = parse_datetime(o_data.get("createTime", ""))

                # 2. Side/Qty
                units = int(o_data.get("units", "0"))
                side = OrderSide.BUY if units > 0 else OrderSide.SELL
                qty = Quantity(abs(units), precision=0)  # Total qty

                # OANDA "PENDING" orders are essentially OPEN or PARTIALLY_FILLED?
                status = OrderStatus.OPEN

                # Filled Qty: Hard to know from just the Order object in OANDA V20 without
                # transaction history. We report 0 filled, Nautilus handles fills separately.
                filled_qty = Quantity(0, precision=0)

                # Price
                price_str = o_data.get("price")
                price = Price.from_str(price_str) if price_str else None

                # Order Type Mapping
                type_str = o_data.get("type", "MARKET")
                order_type = OrderType.LIMIT if type_str == "LIMIT" else OrderType.MARKET

                # 3. Create Report
                report = OrderStatusReport(
                    instrument_id=instrument_id,
                    client_order_id=client_order_id,
                    venue_order_id=venue_order_id,
                    order_status=status,
                    order_side=side,
                    order_type=order_type,
                    time_in_force=TimeInForce.GTC
                    if o_data.get("timeInForce") == "GTC"
                    else TimeInForce.FOK,  # Defaulting
                    price=price,
                    quantity=qty,
                    filled_qty=filled_qty,  # Approximation
                    avg_px=None,
                    ts_accepted=timestamp.value,
                    ts_last=timestamp.value,
                    ts_init=self._clock.timestamp_ns(),
                )
                reports.append(report)

        except Exception as e:
            self._log.error(f"Failed to generate order status reports: {e}")
            traceback.print_exc()  # Print to stdout/stderr

        return reports

    async def generate_fill_reports(self, command) -> list[FillReport]:
        """Generate fill reports (Reconciliation). Stub: Return empty."""
        return []

    async def generate_position_status_reports(self, command) -> list[PositionStatusReport]:
        """Generate position reports by fetching open positions from OANDA (Reconciliation)."""
        reports = []
        try:
            r = positions.OpenPositions(self._account_id)
            data = await self._loop.run_in_executor(None, lambda: self._api.request(r))

            for p_data in data.get("positions", []):
                instrument_id = parse_instrument_id(p_data.get("instrument", ""))

                # OANDA V20: long.units is positive, short.units is negative
                long_units = int(p_data.get("long", {}).get("units", "0"))
                short_units = int(p_data.get("short", {}).get("units", "0"))
                net_units = long_units + short_units

                if net_units == 0:
                    continue

                # Determine position side and average price from the dominant side
                if net_units > 0:
                    position_side = PositionSide.LONG
                    avg_px_str = p_data.get("long", {}).get("averagePrice", "0")
                else:
                    position_side = PositionSide.SHORT
                    avg_px_str = p_data.get("short", {}).get("averagePrice", "0")

                report = PositionStatusReport(
                    account_id=AccountId(f"OANDA-{self._account_id}"),
                    instrument_id=instrument_id,
                    position_side=position_side,
                    quantity=Quantity(abs(net_units), precision=0),
                    report_id=UUID4(),
                    ts_last=self._clock.timestamp_ns(),
                    ts_init=self._clock.timestamp_ns(),
                    venue_position_id=None,
                    avg_px_open=Price.from_str(avg_px_str),
                )
                reports.append(report)

        except Exception as e:
            self._log.error(f"Failed to generate position status reports: {e}")
            traceback.print_exc()

        return reports
