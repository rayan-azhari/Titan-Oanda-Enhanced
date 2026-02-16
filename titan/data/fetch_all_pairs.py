import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

import oandapyV20  # noqa: E402
import oandapyV20.endpoints.accounts as accounts  # noqa: E402


def main():
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    access_token = os.getenv("OANDA_ACCESS_TOKEN")
    environment = os.getenv("OANDA_ENVIRONMENT", "practice")

    client = oandapyV20.API(access_token=access_token, environment=environment)

    r = accounts.AccountInstruments(accountID=account_id)
    try:
        response = client.request(r)
        instruments = response.get("instruments", [])

        # Filter for currency pairs (type = CURRENCY)
        # Some might be CFDs, metals, etc. User asked for "currency pairs".
        # distinct pairs usually have type 'CURRENCY' or look like 'XXX_YYY'

        pairs = []
        for i in instruments:
            if i["type"] == "CURRENCY":
                pairs.append(i["name"])

        pairs.sort()

        print(f"Found {len(pairs)} currency pairs:")
        for p in pairs:
            print(f'    "{p}",')

    except Exception as e:
        print(f"Error fetching instruments: {e}")


if __name__ == "__main__":
    main()
