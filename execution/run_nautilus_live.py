"""run_nautilus_live.py
--------------------

Live trading entry point using NautilusTrader and the OANDA adapter.
Authenticates, configures the trading node, registers custom OANDA components,
loads instruments, and starts the event loop.
"""

import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode

from strategies.ml_strategy import MLSignalStrategy, MLSignalStrategyConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load Environment
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import os

from execution.nautilus_oanda.config import (
    OandaDataClientConfig,
    OandaExecutionClientConfig,
    OandaInstrumentProviderConfig,
)
from execution.nautilus_oanda.data import OandaDataClient
from execution.nautilus_oanda.execution import OandaExecutionClient
from execution.nautilus_oanda.instruments import OandaInstrumentProvider
from strategies.simple_printer import SimplePrinter, SimplePrinterConfig

# ---------------------------------------------------------------------------
# Structured logging — file + console, matching run_live.py pattern
# ---------------------------------------------------------------------------
LOGS_DIR = PROJECT_ROOT / ".tmp" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logging() -> logging.Logger:
    """Configure file + console logging for the Nautilus session."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"nautilus_{date_str}.log"

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
    """Run the live trading node."""
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
    logger.info("  TITAN NAUTILUS ENGINE — %s", environment.upper())
    logger.info("=" * 50)

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
    # We register factories that default to the configuration defined above.
    # This pattern allows the trading node to instantiate clients as needed.

    node.add_data_client_factory(
        "OANDA",
        lambda loop, msgbus, cache, clock: OandaDataClient(
            loop,
            data_config,
            msgbus,
            cache,
            clock,
        ),
    )

    node.add_execution_client_factory(
        "OANDA",
        lambda loop, msgbus, cache, clock: OandaExecutionClient(
            loop,
            exec_config,
            msgbus,
            cache,
            clock,
        ),
    )

    # 4. Load Instruments
    provider = OandaInstrumentProvider(inst_config)
    print("⏳ Loading instruments from OANDA...")
    instruments = provider.load_all()
    print(f"✅ Loaded {len(instruments)} instruments.")

    for inst in instruments:
        node.add_instrument(inst)

    # 5. Load Strategy (Auto-Discover ML Model)
    print("🧠 Searching for trained ML models...")
    models_dir = PROJECT_ROOT / "models"
    model_files = sorted(
        list(models_dir.glob("*.joblib")), key=lambda f: f.stat().st_mtime, reverse=True
    )

    selected_model = None
    selected_tf = None

    # Logic:
    # 1. Prioritize models with explicit timeframe in name (e.g. _H4_)
    # 2. Fallback to latest modified file

    for m in model_files:
        if "_H4_" in m.name:
            selected_model = m
            selected_tf = "H4"
            break
        elif "_H1_" in m.name:
            selected_model = m
            selected_tf = "H1"
            break

    if not selected_model and model_files:
        selected_model = model_files[0]
        print(f"ℹ No specific timeframe found. Using latest: {selected_model.name}")

    if selected_model:
        logging.info(f"Loaded model: {selected_model.name}")

        # Infer instrument and bar type
        instrument_id = "EUR/USD"
        granularity = selected_tf if selected_tf else "H4"

        if "H1" in selected_model.name:
            granularity = "H1"
        if "M15" in selected_model.name:
            granularity = "M15"
        if "GBP_USD" in selected_model.name:
            instrument_id = "GBP/USD"

        bar_type = f"{instrument_id}-{granularity}"

        strat_config = MLSignalStrategyConfig(
            model_path=str(selected_model),
            instrument_id=instrument_id,
            bar_type=bar_type,
            risk_pct=0.02,
            warmup_bars=500,
        )

        strategy = MLSignalStrategy(strat_config)
        node.add_strategy(strategy)
        print(f"🚀 Deployed ML Strategy on {bar_type} using {selected_model.name}")

    else:
        print(
            "⚠ No ML models found in models/ directory. Running in Printer Mode (Monitoring only)."
        )
        strategy_config = SimplePrinterConfig()
        strategy = SimplePrinter(config=strategy_config)
        node.add_strategy(strategy)

    # 6. Build & Run
    print("🚀 Starting Trading Node...")

    # Register shutdown signal
    def stop_node(*args):
        print("\n🛑 Stopping...")
        node.stop()

    signal.signal(signal.SIGINT, stop_node)
    signal.signal(signal.SIGTERM, stop_node)

    # Run the node (blocking)
    node.run()


if __name__ == "__main__":
    main()
