import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load Environment
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.factories import LiveDataClientFactory, LiveExecClientFactory
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import ClientId, Venue

from titan.adapters.oanda.config import (
    OandaDataClientConfig,
    OandaExecutionClientConfig,
    OandaInstrumentProviderConfig,
)
from titan.adapters.oanda.data import OandaDataClient
from titan.adapters.oanda.execution import OandaExecutionClient
from titan.adapters.oanda.instruments import OandaInstrumentProvider

# Import Test Strategy
from titan.strategies.test.strategy import TestTradeConfig, TestTradeStrategy


def main():
    # Setup logging to file
    log_dir = PROJECT_ROOT / ".tmp" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"test_trade_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))  # Simple console output

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
    logging.getLogger("nautilus_trader").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    print(f"📄 Logging to {log_file}")

    logger = logging.getLogger("titan")

    account_id = os.getenv("OANDA_ACCOUNT_ID")
    access_token = os.getenv("OANDA_ACCESS_TOKEN")
    environment = os.getenv("OANDA_ENVIRONMENT", "practice")

    logger.info("=" * 50)
    logger.info("  LIVE TRADING VERIFICATION — %s", environment.upper())

    # 0. Auto-Download Data
    print("📥 Checking for latest data...")
    try:
        download_script = PROJECT_ROOT / "scripts" / "download_data.py"
        subprocess.check_call([sys.executable, str(download_script)])
        print("✅ Data download finished.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Data download failed with exit code {e.returncode}")
        print("❌ Data download failed. Aborting to be safe.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during data download: {e}")
        sys.exit(1)

    # 1. Configs
    data_config = OandaDataClientConfig(
        account_id=account_id, access_token=access_token, environment=environment
    )
    exec_config = OandaExecutionClientConfig(
        account_id=account_id, access_token=access_token, environment=environment
    )
    inst_config = OandaInstrumentProviderConfig(
        account_id=account_id, access_token=access_token, environment=environment
    )

    # 2. Load Instruments (Only need EUR/USD)
    provider = OandaInstrumentProvider(inst_config)
    print("⏳ Loading instruments...")
    instruments = (
        provider.load_all()
    )  # Load all to be safe, specific subscription happens in strategy
    print(f"✅ Loaded {len(instruments)} instruments.")

    target_instrument = next((i for i in instruments if i.id.value == "EUR/USD.OANDA"), None)
    if not target_instrument:
        print("❌ EUR/USD.OANDA not found!")
        return

    # 3. Configure Node
    node_config = TradingNodeConfig(
        trader_id="TITAN-TEST",
        data_clients={"OANDA": data_config},
        exec_clients={"OANDA": exec_config},
    )
    node = TradingNode(config=node_config)
    node.cache.add_instrument(target_instrument)

    # 4. Register Factories
    class LiveOandaDataFactory(LiveDataClientFactory):
        conf = data_config

        @classmethod
        def create(cls, loop, msgbus, cache, clock, name, **kwargs):
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
            return OandaExecutionClient(
                loop=loop,
                client_id=ClientId("OANDA-EXEC"),
                venue=Venue("OANDA"),
                oms_type=OmsType.NETTING,
                account_type=AccountType.MARGIN,
                base_currency=None,
                instrument_provider=cls.prov,
                config=cls.conf,
                msgbus=msgbus,
                cache=cache,
                clock=clock,
            )

    node.add_data_client_factory("OANDA", LiveOandaDataFactory)
    node.add_exec_client_factory("OANDA", LiveOandaExecutionFactory)

    # 5. Add Strategy
    # Using 5-second bars for faster testing
    strat_config = TestTradeConfig(
        instrument_id="EUR/USD.OANDA",
        bar_type="EUR/USD.OANDA-5-SECOND-MID-INTERNAL",
        trade_size=1000,
    )
    strategy = TestTradeStrategy(strat_config)
    node.trader.add_strategy(strategy)

    print("🚀 Starting Test Node...")
    print("   Expect entry after ~25s (5 bars)")
    print("   Expect exit after ~75s (15 bars)")

    try:
        node.build()
        node.run()
    except KeyboardInterrupt:
        node.stop()
    except Exception as e:
        logger.exception("Fatal Error")
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    main()
