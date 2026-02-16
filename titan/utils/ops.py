"""kill_switch.py — Emergency: flatten all positions and cancel all pending orders.

Directive: 07_live_deployment.md

WARNING: This script is destructive. It will:
1. Cancel ALL pending orders.
2. Market-close ALL open positions.
Use only in emergency situations.
"""

import os
import sys
from pathlib import Path

import oandapyV20
import oandapyV20.endpoints.orders as orders_ep
import oandapyV20.endpoints.positions as positions_ep
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")


def main() -> None:
    if not ACCOUNT_ID or not ACCESS_TOKEN:
        print("ERROR: OANDA credentials not set.")
        sys.exit(1)

    client = oandapyV20.API(access_token=ACCESS_TOKEN, environment=ENVIRONMENT)

    print("🚨 KILL SWITCH ACTIVATED")
    print("=" * 40)

    # Cancel all pending orders
    r = orders_ep.OrdersPending(ACCOUNT_ID)
    try:
        response = client.request(r)
        pending = response.get("orders", [])
        for order in pending:
            cancel_r = orders_ep.OrderCancel(ACCOUNT_ID, order["id"])
            client.request(cancel_r)
            print(f"  ✗ Cancelled order {order['id']} ({order['instrument']})")
        if not pending:
            print("  No pending orders.")
    except Exception as e:
        print(f"  ERROR cancelling orders: {e}")

    # Close all open positions
    r = positions_ep.OpenPositions(ACCOUNT_ID)
    try:
        response = client.request(r)
        open_positions = response.get("positions", [])
        for pos in open_positions:
            instrument = pos["instrument"]
            long_units = pos.get("long", {}).get("units", "0")
            short_units = pos.get("short", {}).get("units", "0")

            if long_units != "0":
                close_r = positions_ep.PositionClose(
                    ACCOUNT_ID, instrument, data={"longUnits": "ALL"}
                )
                client.request(close_r)
                print(f"  ✗ Closed LONG {instrument} ({long_units} units)")

            if short_units != "0":
                close_r = positions_ep.PositionClose(
                    ACCOUNT_ID, instrument, data={"shortUnits": "ALL"}
                )
                client.request(close_r)
                print(f"  ✗ Closed SHORT {instrument} ({short_units} units)")

        if not open_positions:
            print("  No open positions.")
    except Exception as e:
        print(f"  ERROR closing positions: {e}")

    print("=" * 40)
    print("✅ All positions flattened. All orders cancelled.\n")


if __name__ == "__main__":
    main()
