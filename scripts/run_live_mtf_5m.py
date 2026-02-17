"""run_live_mtf_5m.py
--------------------

Live trading runner for the Multi-Timeframe Confluence Strategy (5m).
Connects to OANDA (Practice/Live), loads instruments, and launches the optimized 5m strategy.
"""

import logging
import signal
import subprocess
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

from titan.adapters.oanda.config import (
    OandaDataClientConfig,
    OandaExecutionClientConfig,
    OandaInstrumentProviderConfig,
)
from titan.adapters.oanda.data import OandaDataClient
from titan.adapters.oanda.execution import OandaExecutionClient
from titan.adapters.oanda.instruments import OandaInstrumentProvider

# Import our Strategy
from titan.strategies.mtf.strategy import MTFConfluenceConfig, MTFConfluenceStrategy

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOGS_DIR = PROJECT_ROOT / ".tmp" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logging() -> logging.Logger:
    """Configure file + console logging for Root Logger."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"mtf_5m_live_{date_str}.log"

    # Configure Root Logger to capture EVERYTHING (Strategy, Nautilus, Titan)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)
    
    # Return specific logger for the script's own messages
    return logging.getLogger("titan.nautilus")


def main():
    """Run the MTF Strategy-5m live."""
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
    logger.info("  MTF STRATEGY-5M LIVE — %s", environment.upper())

    # 1. Download Data (M5, H1, H4) for warmup
    print("📥 Checking for latest data (M5, H1, H4)...")
    try:
        download_script = PROJECT_ROOT / "scripts" / "download_data.py"
        # Download necessary TFs for EUR_USD
        # Logic: We need M5, H1, H4, D for correct warmup.
        # But download_data.py downloads ALL granularities by default if not specified? 
        # Or we loop. Let's just run it for EUR_USD and hope the default config includes them.
        # The default config in instruments.toml does include M5, H1, H4, D.
        subprocess.check_call([sys.executable, str(download_script), "--instrument", "EUR_USD"])
        print("✅ Data download finished.")
    except Exception as e:
        logger.error(f"Data download warning: {e}")
        print("⚠️ Data download encounterd an issue. Proceeding with existing data if available.")

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

    # 3. Load Instruments
    provider = OandaInstrumentProvider(inst_config)
    print("⏳ Loading instruments from OANDA...")
    instruments = provider.load_all()
    print(f"✅ Loaded {len(instruments)} instruments.")

    # 4. Configure Node
    node_config = TradingNodeConfig(
        trader_id="TITAN-MTF-5M",
        data_clients={"OANDA": data_config},
        exec_clients={"OANDA": exec_config},
    )
    node = TradingNode(config=node_config)

    for inst in instruments:
        node.cache.add_instrument(inst)

    # 5. Register Clients
    class LiveOandaDataFactory(LiveDataClientFactory):
        conf = data_config
        @classmethod
        def create(cls, loop, msgbus, cache, clock, name, **kwargs):
            return OandaDataClient(
                loop=loop, client_id=ClientId("OANDA-DATA"), venue=Venue("OANDA"),
                config=cls.conf, msgbus=msgbus, cache=cache, clock=clock,
            )

    class LiveOandaExecutionFactory(LiveExecClientFactory):
        conf = exec_config
        prov = provider
        @classmethod
        def create(cls, loop, msgbus, cache, clock, name, **kwargs):
            return OandaExecutionClient(
                loop=loop, client_id=ClientId("OANDA-EXEC"), venue=Venue("OANDA"),
                oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
                base_currency=None, instrument_provider=cls.prov, config=cls.conf,
                msgbus=msgbus, cache=cache, clock=clock,
            )

    node.add_data_client_factory("OANDA", LiveOandaDataFactory)
    node.add_exec_client_factory("OANDA", LiveOandaExecutionFactory)

    # 6. Configure Strategy (Optimized 5m)
    strat_config = MTFConfluenceConfig(
        instrument_id="EUR/USD.OANDA",
        bar_types={
            "M5": "EUR/USD.OANDA-5-MINUTE-MID-INTERNAL",
            "H1": "EUR/USD.OANDA-1-HOUR-MID-INTERNAL",
            "H4": "EUR/USD.OANDA-4-HOUR-MID-INTERNAL",
        },
        config_path="config/mtf_5m.toml",
        risk_pct=0.01,
        leverage_cap=5.0,
        warmup_bars=1000,
    )

    strategy = MTFConfluenceStrategy(strat_config)
    node.trader.add_strategy(strategy)

    logger.info("Strategy Added. M5 Optimized Mode.")

    # 7. Run
    print("🚀 Starting Trading Node (MTF-5m)...")

    def stop_node(*args):
        print("\n🛑 Stopping...")
        node.stop()

    signal.signal(signal.SIGINT, stop_node)
    signal.signal(signal.SIGTERM, stop_node)

    try:
        node.build()
        node.run()
    except Exception as e:
        logger.exception("Fatal Runtime Error")
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
