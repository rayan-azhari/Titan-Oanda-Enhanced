import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
from oandapyV20 import API


def main():
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    access_token = os.getenv("OANDA_ACCESS_TOKEN")
    environment = os.getenv("OANDA_ENVIRONMENT", "practice")

    print(f"Checking OANDA Account: {account_id} ({environment})")

    api = API(access_token=access_token, environment=environment)

    # 1. Check Positions
    print("\n--- POSITIONS ---")
    try:
        r = positions.OpenPositions(account_id)
        data = api.request(r)
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error fetching positions: {e}")

    # 2. Check Orders as well
    print("\n--- ORDERS ---")
    try:
        r = orders.OrderList(account_id)
        data = api.request(r)
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error fetching orders: {e}")


if __name__ == "__main__":
    main()
