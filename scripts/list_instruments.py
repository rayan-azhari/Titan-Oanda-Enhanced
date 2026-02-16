"""list_instruments.py — List available currency pairs from OANDA.

Usage:
    uv run python scripts/list_instruments.py
"""

import os
import sys
from pathlib import Path

# Add project root to path for local execution config loading
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import oandapyV20
import oandapyV20.endpoints.accounts as accounts


def main():
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    access_token = os.getenv("OANDA_ACCESS_TOKEN")
    environment = os.getenv("OANDA_ENVIRONMENT", "practice")

    if not access_token:
        print("ERROR: OANDA_ACCESS_TOKEN not found in .env")
        sys.exit(1)

    print(f"Connecting to OANDA ({environment})...")
    client = oandapyV20.API(access_token=access_token, environment=environment)

    if not account_id:
        # Fetch the first account ID if not provided
        r = accounts.AccountList()
        try:
            resp = client.request(r)
            account_id = resp["accounts"][0]["id"]
            print(f"Using Account ID: {account_id}")
        except Exception as e:
            print(f"Error fetching account list: {e}")
            sys.exit(1)

    r = accounts.AccountInstruments(accountID=account_id)
    try:
        response = client.request(r)
        instruments = response.get("instruments", [])

        pairs = []
        for i in instruments:
            if i["type"] == "CURRENCY":
                pairs.append(i["name"])

        pairs.sort()

        print(f"\nFound {len(pairs)} currency pairs:")
        for p in pairs:
            print(f'    "{p}",')

    except Exception as e:
        print(f"Error fetching instruments: {e}")


if __name__ == "__main__":
    main()
