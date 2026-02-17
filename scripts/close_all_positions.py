import os
import sys
from pathlib import Path

import oandapyV20

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load Environment
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from titan.utils.ops import cancel_all_orders, close_all_positions


def main():
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    access_token = os.getenv("OANDA_ACCESS_TOKEN")
    environment = os.getenv("OANDA_ENVIRONMENT", "practice")

    print(f"🧹 CLEANUP: Closing all positions/orders for {account_id} ({environment})")

    client = oandapyV20.API(access_token=access_token, environment=environment)

    print("Cancelling Orders...")
    cancel_all_orders(client, account_id)

    print("Closing Positions...")
    close_all_positions(client, account_id)

    print("✅ Done.")


if __name__ == "__main__":
    main()
