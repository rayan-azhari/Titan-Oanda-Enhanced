"""scripts/close_all_positions.py
------------------------------

Closes all open positions on the configured OANDA account.
Usage: python scripts/close_all_positions.py
"""

import os
import sys
import logging
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

try:
    import oandapyV20
    import oandapyV20.endpoints.positions as positions
except ImportError:
    print("ERROR: oandapyV20 not installed.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("titan.cleanup")

def main():
    load_dotenv(PROJECT_ROOT / ".env")
    
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    access_token = os.getenv("OANDA_ACCESS_TOKEN")
    environment = os.getenv("OANDA_ENVIRONMENT", "practice")

    if not account_id or not access_token:
        logger.error("OANDA credentials missing in .env")
        return

    client = oandapyV20.API(access_token=access_token, environment=environment)

    # 1. Get Open Positions
    logger.info("Fetching open positions...")
    r = positions.OpenPositions(account_id)
    try:
        data = client.request(r)
    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}")
        return

    pos_list = data.get("positions", [])
    if not pos_list:
        logger.info("✅ No open positions found.")
        return

    logger.info(f"Found {len(pos_list)} open position(s). Closing...")

    # 2. Close each position
    for p in pos_list:
        instrument = p["instrument"]
        long_units = p["long"]["units"]
        short_units = p["short"]["units"]

        logger.info(f"Closing {instrument} (Long: {long_units}, Short: {short_units})...")
        
        try:
            # Prepare close payload
            close_data = {}
            if long_units != "0":
                close_data["longUnits"] = "ALL"
            if short_units != "0":
                close_data["shortUnits"] = "ALL"

            if not close_data:
                continue

            cr = positions.PositionClose(account_id, instrument, data=close_data)
            client.request(cr)
            logger.info(f"✅ Closed {instrument}")

        except Exception as e:
            logger.error(f"❌ Failed to close {instrument}: {e}")

    logger.info("Cleanup complete.")

if __name__ == "__main__":
    main()
