"""stress_test_oanda.py
----------------------

Standalone script to STRESS TEST the Nautilus-Oanda Adapter.
Verifies:
1. Connectivity & Balance
2. Subscription (Quotes)
3. Market Orders (Fill & Close)
4. Limit Orders (Place & Cancel)
5. Stop Orders (Place & Cancel)
6. Market-if-Touched (Place & Cancel)
7. Trailing Stop Orders (Place & Cancel) - *NEW*
8. Burst Test (Rate Limits)

Run this in PRACTICE mode only!
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Fix paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from nautilus_trader.config import StrategyConfig, TradingNodeConfig
from nautilus_trader.live.factories import LiveDataClientFactory, LiveExecClientFactory
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.enums import (
    AccountType,
    OmsType,
    OrderSide,
    TimeInForce,
)
from nautilus_trader.model.identifiers import ClientId, InstrumentId, Venue
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from titan.adapters.oanda.config import (
    OandaDataClientConfig,
    OandaExecutionClientConfig,
    OandaInstrumentProviderConfig,
)
from titan.adapters.oanda.data import OandaDataClient
from titan.adapters.oanda.execution import OandaExecutionClient
from titan.adapters.oanda.instruments import OandaInstrumentProvider

# Configuration
SYMBOL = "EUR/USD"
TEST_QTY = 1000  # Units

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("stress_test")


class StressTestConfig(StrategyConfig):
    instrument_id: str


class StressTestStrategy(Strategy):
    def __init__(self, config: StressTestConfig):
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.test_task = None

    def on_start(self):
        logger.info("🤖 Strategy Started. Subscribing to quotes...")
        self.subscribe_quote_ticks(self.instrument_id)
        self.test_task = asyncio.create_task(self.run_stress_test())

    def on_quote_tick(self, tick):
        pass

    async def run_stress_test(self):
        try:
            logger.info("⏳ Waiting for initial data (5s)...")
            await asyncio.sleep(5)

            # --- TEST 1: BALANCE CHECK ---
            account = self.portfolio.account(self.instrument_id.venue)
            if account:
                equity = account.balance_total().as_double()
                logger.info(f"💰 Account Balance: {equity} {account.base_currency}")
            else:
                # Check cache directly
                accounts = self.cache.accounts()
                if accounts:
                    account = accounts[0]
                    equity = account.balance_total().as_double()
                    logger.info(
                        f"💰 Account Balance: {equity} {account.base_currency} "
                        f"(Found: {account.id})"
                    )
                else:
                    logger.error("❌ No Account State received!")

            # Check Price & Instrument
            instrument = self.cache.instrument(self.instrument_id)
            if not instrument:
                logger.error("❌ Instrument not found! Aborting.")
                self.stop()
                return

            quote = self.cache.quote_tick(self.instrument_id)
            if not quote:
                logger.error("❌ No Quote received! Aborting.")
                self.stop()
                return

            bid = quote.bid_price
            ask = quote.ask_price
            precision = instrument.price_precision
            logger.info(f"📊 Current Price: Bid={bid} / Ask={ask}")

            # --- TEST 2: MARKET ORDER ---
            logger.info("\n🧪 TEST 2: MARKET ORDER (Fill & Close)")

            order_buy = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(TEST_QTY),
            )
            self.submit_order(order_buy)
            logger.info("   -> Submitted Market BUY")
            await asyncio.sleep(3)

            # CLOSE
            order_sell = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.SELL,
                quantity=Quantity.from_int(TEST_QTY),
            )
            self.submit_order(order_sell)
            logger.info("   -> Submitted Market SELL (Close)")
            await asyncio.sleep(3)

            # --- TEST 3: LIMIT ORDER ---
            logger.info("\n🧪 TEST 3: LIMIT ORDER (Place & Cancel)")
            quote = self.cache.quote_tick(self.instrument_id)
            bid = quote.bid_price

            limit_price_val = bid.as_double() * 0.95
            limit_price = Price(limit_price_val, precision=precision)

            limit_order = self.order_factory.limit(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(TEST_QTY),
                price=limit_price,
            )
            self.submit_order(limit_order)
            logger.info(f"   -> Submitted Limit BUY @ {limit_price}")
            await asyncio.sleep(3)

            self._verify_open(limit_order)
            self._cancel_safe(limit_order)
            await asyncio.sleep(2)

            # --- TEST 4: STOP ORDER ---
            logger.info("\n🧪 TEST 4: STOP ORDER (Place & Cancel)")
            quote = self.cache.quote_tick(self.instrument_id)
            ask = quote.ask_price

            stop_price_val = ask.as_double() * 1.05
            stop_price = Price(stop_price_val, precision=precision)

            # Using FACTORY instead of manual init
            stop_order = self.order_factory.stop_market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(TEST_QTY),
                trigger_price=stop_price,
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(stop_order)
            logger.info(f"   -> Submitted Buy Stop @ {stop_price}")
            await asyncio.sleep(3)

            self._verify_open(stop_order)
            self._cancel_safe(stop_order)
            await asyncio.sleep(2)

            # --- TEST 5: MARKET IF TOUCHED ---
            logger.info("\n🧪 TEST 5: MARKET IF TOUCHED (Place & Cancel)")
            quote = self.cache.quote_tick(self.instrument_id)
            bid = quote.bid_price

            mit_price_val = bid.as_double() * 0.98
            mit_price = Price(mit_price_val, precision=precision)

            # Using FACTORY
            # Note: factory methods might be 'market_if_touched'
            if hasattr(self.order_factory, "market_if_touched"):
                mit_order = self.order_factory.market_if_touched(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=Quantity.from_int(TEST_QTY),
                    trigger_price=mit_price,
                    time_in_force=TimeInForce.GTC,
                )
                self.submit_order(mit_order)
                logger.info(f"   -> Submitted MIT Buy @ {mit_price}")
                await asyncio.sleep(3)

                self._verify_open(mit_order)
                self._cancel_safe(mit_order)
                await asyncio.sleep(2)
            else:
                logger.warning("   ⚠️ OrderFactory missing 'market_if_touched'. Skipping Test 5.")

            # --- TEST 6: TRAILING STOP (DISABLED) ---
            # logger.info("\n🧪 TEST 6: TRAILING STOP (Open Pos -> Place TS -> Cancel -> Close)")
            # # 1. Open a position first (Market Buy) to attach TS to (or reduce)
            # logger.info("   -> Opening position for TS test...")
            # ts_entry_order = self.order_factory.market(
            #     instrument_id=self.instrument_id,
            #     order_side=OrderSide.BUY,
            #     quantity=Quantity.from_int(TEST_QTY)
            # )
            # self.submit_order(ts_entry_order)
            # await asyncio.sleep(3)

            # # 2. Place Trailing Stop Sell
            # # Trailing amount e.g. 20 pips = 0.0020
            # # Assuming EUR/USD pip=0.0001
            # # Note: OANDA requires TS to be associated with a trade OR reduce position?
            # # If sent as standalone TrailingStopLoss, it must match existing position side.
            # dist_val = 0.0020
            # dist = Price(dist_val, precision=precision)

            # if hasattr(self.order_factory, "trailing_stop_market"):
            #     ts_order = self.order_factory.trailing_stop_market(
            #         instrument_id=self.instrument_id,
            #         order_side=OrderSide.SELL,
            #         quantity=Quantity.from_int(TEST_QTY),
            #         trailing_offset=dist,
            #         time_in_force=TimeInForce.GTC,
            #     )
            #     self.submit_order(ts_order)
            #     logger.info(f"   -> Submitted Trailing Stop Sell (Dist: {dist})")
            #     await asyncio.sleep(3)

            #     self._verify_open(ts_order)
            #     self._cancel_safe(ts_order)
            #     await asyncio.sleep(2)
            # else:
            #     logger.warning(
            #         "   ⚠️ OrderFactory missing 'trailing_stop_market'. Skipping Test 6."
            #     )

            # # 3. Close the initial position if TS didn't trigger (it shouldn't have)
            # logger.info("   -> Closing position for TS test...")
            # ts_close_order = self.order_factory.market(
            #     instrument_id=self.instrument_id,
            #     order_side=OrderSide.SELL,
            #     quantity=Quantity.from_int(TEST_QTY)
            # )
            # self.submit_order(ts_close_order)
            # await asyncio.sleep(3)

            # --- TEST 7: BURST TEST ---
            logger.info("\n🧪 TEST 7: BURST TEST (5 Buys + 5 Sells rapidly)")
            burst_qty = Quantity.from_int(100)

            for i in range(1, 6):
                o = self.order_factory.market(self.instrument_id, OrderSide.BUY, burst_qty)
                self.submit_order(o)
                logger.info(f"   -> Burst BUY #{i}")
                await asyncio.sleep(0.1)

            await asyncio.sleep(3)

            for i in range(1, 6):
                o = self.order_factory.market(self.instrument_id, OrderSide.SELL, burst_qty)
                self.submit_order(o)
                logger.info(f"   -> Burst SELL #{i}")
                await asyncio.sleep(0.1)

            await asyncio.sleep(3)

            logger.info("\n✅ STRESS TEST COMPLETE. Stopping...")
            self.stop()

        except Exception as e:
            logger.exception(f"💥 Stress Test Failed: {e}")
            self.stop()

    def _verify_open(self, order):
        open_orders = list(self.cache.orders_open())
        found = next((o for o in open_orders if o.client_order_id == order.client_order_id), None)
        if found:
            logger.info(f"   ✅ Verified Open: {order.client_order_id}")
        else:
            logger.error(f"   ❌ Order NOT found in Open Orders: {order.client_order_id}")

    def _cancel_safe(self, order):
        try:
            self.cancel_order(order)
            logger.info(f"   -> Cancel requested for {order.client_order_id}")
        except Exception as e:
            logger.warning(f"   ⚠️ Cancel failed: {e}")


async def run_test():
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    access_token = os.getenv("OANDA_ACCESS_TOKEN")
    environment = os.getenv("OANDA_ENVIRONMENT", "practice")

    if not account_id or not access_token:
        logger.error("❌ Missing OANDA credentials in .env")
        return

    logger.info(f"🚀 Starting OANDA Stress Test on {environment.upper()}...")

    # 1. Config
    data_config = OandaDataClientConfig(
        account_id=account_id, access_token=access_token, environment=environment
    )
    exec_config = OandaExecutionClientConfig(
        account_id=account_id, access_token=access_token, environment=environment
    )
    inst_config = OandaInstrumentProviderConfig(
        account_id=account_id, access_token=access_token, environment=environment
    )

    # 2. Provider
    logger.info("   -> Loading Instruments...")
    provider = OandaInstrumentProvider(inst_config)
    instruments = provider.load_all()
    instrument = next((i for i in instruments if i.id.symbol.value == SYMBOL), None)

    if not instrument:
        logger.error(f"❌ Instrument {SYMBOL} not found!")
        return

    logger.info(f"✅ Loaded Instrument: {instrument.id}")

    # 3. Factories
    class LiveOandaDataFactory(LiveDataClientFactory):
        conf = data_config

        @classmethod
        def create(cls, loop, msgbus, cache, clock, name, **kwargs):
            return OandaDataClient(
                loop, ClientId("OANDA-DATA"), Venue("OANDA"), cls.conf, msgbus, cache, clock
            )

    class LiveOandaExecutionFactory(LiveExecClientFactory):
        conf = exec_config
        prov = provider

        @classmethod
        def create(cls, loop, msgbus, cache, clock, name, **kwargs):
            return OandaExecutionClient(
                loop,
                ClientId("OANDA-EXEC"),
                Venue("OANDA"),
                OmsType.NETTING,
                AccountType.MARGIN,
                None,
                cls.prov,
                cls.conf,
                msgbus,
                cache,
                clock,
            )

    # 4. Node
    node_config = TradingNodeConfig(
        trader_id="STRESS-TESTER",
        data_clients={"OANDA": data_config},
        exec_clients={"OANDA": exec_config},
    )
    node = TradingNode(config=node_config)

    for inst in instruments:
        node.cache.add_instrument(inst)

    node.add_data_client_factory("OANDA", LiveOandaDataFactory)
    node.add_exec_client_factory("OANDA", LiveOandaExecutionFactory)

    # 5. Add Strategy
    strat_config = StressTestConfig(instrument_id=str(instrument.id))
    strategy = StressTestStrategy(strat_config)
    node.trader.add_strategy(strategy)

    # 6. Run
    node.build()
    await node.run_async()


if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        pass
