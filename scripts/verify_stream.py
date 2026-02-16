"""Quick standalone test: can we receive ticks from OANDA?"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import oandapyV20
import oandapyV20.endpoints.pricing as pricing

if __name__ == "__main__":
    api = oandapyV20.API(
        access_token=os.getenv("OANDA_ACCESS_TOKEN"),
        environment=os.getenv("OANDA_ENVIRONMENT", "practice"),
    )

    r = pricing.PricingStream(
        accountID=os.getenv("OANDA_ACCOUNT_ID"),
        params={"instruments": "EUR_USD"},
    )

    print("Connecting to OANDA stream...")
    i = 0
    for tick in api.request(r):
        i += 1
        print(f"[{i}] {json.dumps(tick)[:200]}")
        if i >= 5:
            print("\n✅ Stream is working! Received 5 ticks/heartbeats.")
            break
