"""
run_nautilus_live.py
--------------------

Live trading entry point using NautilusTrader and the OANDA adapter.
"""

import asyncio
import signal
import sys
from decimal import Decimal
from pathlib import Path

from nautilus_trader.config import LiveDataClientConfig
from nautilus_trader.config import LiveExecutionClientConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import Venue

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load Environment
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import os
from execution.nautilus_oanda.config import OandaDataClientConfig
from execution.nautilus_oanda.config import OandaExecutionClientConfig
from execution.nautilus_oanda.config import OandaInstrumentProviderConfig
from execution.nautilus_oanda.instruments import OandaInstrumentProvider
from execution.nautilus_oanda.data import OandaDataClient
from execution.nautilus_oanda.execution import OandaExecutionClient

# Configure Logging?
# Nautilus uses structlog.


def main():
    """Run the live trading node."""
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    access_token = os.getenv("OANDA_ACCESS_TOKEN")
    environment = os.getenv("OANDA_ENVIRONMENT", "practice")

    if not account_id or not access_token:
        print("ERROR: OANDA credentials not found.")
        sys.exit(1)

    # 1. Configure the Node
    node_config = TradingNodeConfig(
        trader_id="TITAN-001",
        log_level="INFO",
    )
    node = TradingNode(config=node_config)

    # 2. Configure Adapter Components
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

    # 3. Register Clients
    # Note: We need to register factories or instances depending on Nautilus version
    # Modern Nautilus uses add_data_client_factory usually, or we can add instances if supported.
    # For custom adapters, we often register the client class and config.
    
    node.add_data_client_factory(
        "OANDA", 
        lambda loop, msgbus, cache, clock: OandaDataClient(loop, data_config, msgbus, cache, clock)
    )
    
    node.add_execution_client_factory(
        "OANDA",
        lambda loop, msgbus, cache, clock: OandaExecutionClient(loop, exec_config, msgbus, cache, clock)
    )

    # 4. Load Instruments
    provider = OandaInstrumentProvider(inst_config)
    print("⏳ Loading instruments from OANDA...")
    instruments = provider.load_all()
    print(f"✅ Loaded {len(instruments)} instruments.")
    
    for inst in instruments:
        node.add_instrument(inst)

    # 5. Build & Run
    print("🚀 Starting Trading Node...")
    
    # Register shutdown signal
    def stop_node(*args):
        print("\n🛑 Stopping...")
        node.stop()
        
    signal.signal(signal.SIGINT, stop_node)
    signal.signal(signal.SIGTERM, stop_node)

    # Run!
    node.run()

if __name__ == "__main__":
    main()
