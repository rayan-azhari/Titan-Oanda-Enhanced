"""run_live_mtf.py
-----------------

Live trading runner for the Multi-Timeframe Confluence Strategy.
Connects to OANDA (Practice/Live), loads instruments, and launches the strategy.
"""

import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load Environment
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import os

from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.factories import LiveDataClientFactory, LiveExecClientFactory
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import ClientId, Venue

from execution.nautilus_oanda.config import (
    OandaDataClientConfig,
    OandaExecutionClientConfig,
    OandaInstrumentProviderConfig,
)
from execution.nautilus_oanda.data import OandaDataClient
from execution.nautilus_oanda.execution import OandaExecutionClient
from execution.nautilus_oanda.instruments import OandaInstrumentProvider

# Import our Strategy
from strategies.mtf_strategy import MTFConfluenceConfig, MTFConfluenceStrategy

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOGS_DIR = PROJECT_ROOT / ".tmp" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logging() -> logging.Logger:
    """Configure file + console logging."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"mtf_live_{date_str}.log"

    logger = logging.getLogger("titan.nautilus")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def main():
    """Run the MTF Strategy live."""
    logger = _setup_logging()

    account_id = os.getenv("OANDA_ACCOUNT_ID")
    access_token = os.getenv("OANDA_ACCESS_TOKEN")
    environment = os.getenv("OANDA_ENVIRONMENT", "practice")

    if not account_id or not access_token:
        logger.error(
            "OANDA credentials not found. Set OANDA_ACCOUNT_ID and OANDA_ACCESS_TOKEN in .env."
        )
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("  MTF CONFLUENCE LIVE — %s", environment.upper())
    # 2. Configure Adapter
    data_config = OandaDataClientConfig(
        account_id=account_id,
        access_token=access_token,
        environment=environment,
    )
    exec_config = OandaExecutionClientConfig(
        account_id=account_id,
        access_token=access_token,
        environment=environment,
    )
    inst_config = OandaInstrumentProviderConfig(
        account_id=account_id,
        access_token=access_token,
        environment=environment,
    )

    # 3. Load Instruments (Moved up to provide to factories)
    provider = OandaInstrumentProvider(inst_config)
    print("⏳ Loading instruments from OANDA...")
    instruments = provider.load_all()
    print(f"✅ Loaded {len(instruments)} instruments.")

    # 4. Configure Node with Client Configs
    node_config = TradingNodeConfig(
        trader_id="TITAN-MTF",
        data_clients={"OANDA": data_config},
        exec_clients={"OANDA": exec_config},
    )
    node = TradingNode(config=node_config)

    for inst in instruments:
        node.cache.add_instrument(inst)

    # 5. Register Clients
    # Wrapper Factories to inject config
    class LiveOandaDataFactory(LiveDataClientFactory):
        conf = data_config

        @classmethod
        def create(cls, loop, msgbus, cache, clock, name, **kwargs):
            print(f"🔧 Creating OandaDataClient ({name})...")
            return OandaDataClient(
                loop=loop,
                client_id=ClientId("OANDA-DATA"),
                venue=Venue("OANDA"),
                config=cls.conf,
                msgbus=msgbus,
                cache=cache,
                clock=clock,
            )

    class LiveOandaExecutionFactory(LiveExecClientFactory):
        conf = exec_config
        prov = provider

        @classmethod
        def create(cls, loop, msgbus, cache, clock, name, **kwargs):
            print(f"🔧 Creating OandaExecutionClient ({name})...")
            return OandaExecutionClient(
                loop=loop,
                client_id=ClientId("OANDA-EXEC"),
                venue=Venue("OANDA"),
                oms_type=OmsType.HEDGING,
                account_type=AccountType.MARGIN,
                base_currency=None,  # Or Currency.from_str("USD")
                instrument_provider=cls.prov,
                config=cls.conf,
                msgbus=msgbus,
                cache=cache,
                clock=clock,
            )

    node.add_data_client_factory("OANDA", LiveOandaDataFactory)
    node.add_exec_client_factory("OANDA", LiveOandaExecutionFactory)

    # 5. Configure Strategy
    strat_config = MTFConfluenceConfig(
        instrument_id="EUR/USD.OANDA",
        bar_types={
            "H1": "EUR/USD.OANDA-1-HOUR-MID-INTERNAL",
            "H4": "EUR/USD.OANDA-4-HOUR-MID-INTERNAL",
            "D": "EUR/USD.OANDA-1-DAY-MID-INTERNAL",
            "W": "EUR/USD.OANDA-1-WEEK-MID-INTERNAL",
        },
        risk_pct=0.01,
        leverage_cap=5.0,
        warmup_bars=1000,
    )

    strategy = MTFConfluenceStrategy(strat_config)
    node.trader.add_strategy(strategy)

    logger.info("Strategy Added. Subscriptions:")
    for tf, bt in strat_config.bar_types.items():
        logger.info(f"  {tf}: {bt}")

    # 6. Run
    print("🚀 Starting Trading Node...")

    def stop_node(*args):
        print("\n🛑 Stopping...")
        node.stop()

    signal.signal(signal.SIGINT, stop_node)
    signal.signal(signal.SIGTERM, stop_node)

    print(f"DEBUG Node: {[d for d in dir(node) if 'data' in d or 'client' in d]}")
    try:
        print(f"DEBUG Trader: {[d for d in dir(node.trader) if 'data' in d or 'client' in d]}")
    except Exception:
        pass

    try:
        node.build()
        node.run()
    except Exception as e:
        logger.exception("Fatal Runtime Error")
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
